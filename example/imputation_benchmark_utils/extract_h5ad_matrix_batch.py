"""
Extract a gene-by-cell wide parquet from a directory of h5ad files.

This is a generalized version of the existing extraction scripts.  It can read
observed expression from X or imputed expression from a layer, and can optionally
apply per-cell normalize_total + log1p before writing.

Output columns:
  cell_id, slice_id, ccf_x, ccf_y, ccf_z, Gene1, Gene2, ...
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    with open(path) as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            gene = re.split(r"[\t,]", line, maxsplit=1)[0].strip()
            if i == 0 and gene.lower() in {"gene", "gene_name", "symbol", "name"}:
                continue
            if gene and gene not in seen:
                seen.add(gene)
                genes.append(gene)
    return genes


def sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for pattern in (r"(?:^|_)T(\d+)(?:_|$)", r"(?:^|_)slice_(\d+)(?:_|$)", r"(\d+)"):
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1)), stem
    return 10**18, stem


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        name = str(raw)
        count = seen.get(name, 0)
        out.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return out


def set_gene_names(adata, gene_name_column: str | None) -> None:
    if gene_name_column and gene_name_column in adata.var.columns:
        adata.var_names = make_unique(adata.var[gene_name_column].astype(str).tolist())


def scan_available_genes(h5ads: list[Path], gene_name_column: str | None) -> set[str]:
    available: set[str] = set()
    for path in h5ads:
        adata = sc.read_h5ad(path, backed="r")
        if gene_name_column and gene_name_column in adata.var.columns:
            names = adata.var[gene_name_column].astype(str).tolist()
        else:
            names = adata.var_names.astype(str).tolist()
        available.update(names)
        adata.file.close()
    return available


def dense_matrix(x):
    import scipy.sparse as sp

    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def get_matrix(adata, source: str, layer: str | None):
    if source == "X":
        return adata.X
    if source == "layer":
        if not layer:
            raise ValueError("--layer is required when --source layer")
        if layer not in adata.layers:
            raise KeyError(f"layer {layer!r} not found. Available layers: {list(adata.layers.keys())}")
        return adata.layers[layer]
    raise ValueError(source)


def get_coordinates(adata, ccf_key: str, coord_columns: list[str] | None, h5ad_name: str) -> np.ndarray:
    if ccf_key and ccf_key in adata.obsm:
        ccf = np.asarray(adata.obsm[ccf_key], dtype=np.float32)
    elif coord_columns:
        missing = [col for col in coord_columns if col not in adata.obs.columns]
        if missing:
            raise KeyError(f"{h5ad_name}: missing obs coordinate columns: {missing}")
        ccf = adata.obs.loc[:, coord_columns].to_numpy(dtype=np.float32)
    else:
        raise KeyError(
            f"{h5ad_name}: obsm[{ccf_key!r}] not found. Available obsm: {list(adata.obsm.keys())}. "
            "Use --coord-columns COL1 COL2 COL3 when coordinates live in obs."
        )
    if ccf.shape[1] < 3:
        raise ValueError(f"{h5ad_name}: coordinates must have at least 3 columns")
    return ccf


def normalize_log1p(matrix: np.ndarray, target_sum: float) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    matrix *= float(target_sum)
    np.log1p(matrix, out=matrix)
    return matrix


def extract_slice(
    h5ad_path: Path,
    gene_list: list[str],
    source: str,
    layer: str | None,
    normalize: str,
    target_sum: float,
    ccf_key: str,
    coord_columns: list[str] | None,
    gene_name_column: str | None,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    adata = sc.read_h5ad(h5ad_path)
    set_gene_names(adata, gene_name_column)

    ccf = get_coordinates(adata, ccf_key, coord_columns, h5ad_path.name)

    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
    present = [g for g in gene_list if g in var_to_idx]
    missing = [g for g in gene_list if g not in var_to_idx]

    data = {
        "cell_id": adata.obs_names.astype(str).to_numpy(),
        "slice_id": np.repeat(h5ad_path.stem, adata.n_obs),
        "ccf_x": ccf[:, 0],
        "ccf_y": ccf[:, 1],
        "ccf_z": ccf[:, 2],
    }

    if present:
        matrix = get_matrix(adata, source, layer)
        sub = dense_matrix(matrix[:, [var_to_idx[g] for g in present]]).astype(np.float32, copy=False)
        if normalize == "log1p_norm":
            sub = normalize_log1p(sub, target_sum=target_sum)
        for j, gene in enumerate(present):
            data[gene] = sub[:, j]
    for gene in missing:
        data[gene] = np.zeros(adata.n_obs, dtype=np.float32)

    presence = {gene: gene in var_to_idx for gene in gene_list}
    df = pd.DataFrame(data)

    del adata, data
    if present:
        del sub
    gc.collect()
    return df, presence


def merge_tmp_parquets(tmp_dir: Path, output: Path) -> None:
    import pyarrow.parquet as pq

    files = sorted(tmp_dir.glob("*.parquet"), key=sort_key)
    if not files:
        raise FileNotFoundError(f"No temporary parquet files found in {tmp_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    try:
        for path in files:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += table.num_rows
            del table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    print(f"[merge] wrote {total_rows:,} rows -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--file-glob", default="*.h5ad")
    parser.add_argument("--source", choices=["X", "layer"], default="X")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--normalize", choices=["none", "log1p_norm"], default="none")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--ccf-key", default="ccf")
    parser.add_argument(
        "--coord-columns",
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Use obs columns as ccf_x/ccf_y/ccf_z when --ccf-key is not present, e.g. --coord-columns az ay ax.",
    )
    parser.add_argument("--gene-name-column", default="gene_name")
    parser.add_argument(
        "--missing-policy",
        choices=["zero", "drop"],
        default="zero",
        help="zero: keep requested genes missing from all files as zero columns; drop: remove genes absent from all input h5ads before extraction.",
    )
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--keep-tmp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genes = read_gene_list(args.gene_list)
    h5ads = sorted(args.input_dir.glob(args.file_glob), key=sort_key)
    if not h5ads:
        raise FileNotFoundError(f"No {args.file_glob} files found in {args.input_dir}")

    tmp_dir = args.tmp_dir or args.output.with_suffix("").parent / f".{args.output.stem}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[input] {len(h5ads)} h5ad files")
    print(f"[genes] {len(genes)} requested genes")
    if args.missing_policy == "drop":
        available = scan_available_genes(h5ads, args.gene_name_column)
        before = len(genes)
        genes = [gene for gene in genes if gene in available]
        print(f"[genes] missing-policy=drop kept {len(genes)}/{before} genes present in at least one h5ad")
        if not genes:
            raise ValueError("No requested genes are present in the input h5ads")
    print(f"[matrix] source={args.source}, layer={args.layer}, normalize={args.normalize}")

    gene_presence = {g: 0 for g in genes}
    t0 = time.time()
    for i, h5ad in enumerate(h5ads, start=1):
        ts = time.time()
        df, presence = extract_slice(
            h5ad,
            genes,
            source=args.source,
            layer=args.layer,
            normalize=args.normalize,
            target_sum=args.target_sum,
            ccf_key=args.ccf_key,
            coord_columns=args.coord_columns,
            gene_name_column=args.gene_name_column,
        )
        for gene, is_present in presence.items():
            gene_presence[gene] += int(is_present)
        out = tmp_dir / f"{h5ad.stem}.parquet"
        df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
        print(f"[{i:4d}/{len(h5ads)}] {h5ad.name}: {len(df):,} cells, {time.time()-ts:.1f}s")
        del df
        gc.collect()

    merge_tmp_parquets(tmp_dir, args.output)

    coverage = pd.DataFrame(
        {"gene": list(gene_presence), "n_files_present": list(gene_presence.values()), "n_files_total": len(h5ads)}
    )
    coverage.to_csv(args.output.with_suffix(".gene_coverage.csv"), index=False)
    print(f"[coverage] {args.output.with_suffix('.gene_coverage.csv')}")
    print(f"[done] elapsed {time.time()-t0:.1f}s")

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
