"""
Compare imputation methods by gene-wise voxel expression correlation.

Each input parquet should contain ccf_x/ccf_y/ccf_z and gene columns.  The script
bins cells into voxels, averages expression per voxel, intersects voxels across
methods, and reports Pearson/Spearman correlation for each gene and method pair.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = {"cell_id", "slice_id", "ccf_x", "ccf_y", "ccf_z"}


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            gene = line.replace(",", "\t").split("\t", 1)[0].strip()
            if gene.lower() in {"gene", "gene_name", "symbol", "name"}:
                continue
            if gene and gene not in seen:
                seen.add(gene)
                genes.append(gene)
    return genes


def read_parquet(path: Path, genes: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    wanted = ["ccf_x", "ccf_y", "ccf_z"] + genes
    available = set(pq.read_schema(path).names)
    columns = [col for col in wanted if col in available]
    missing = [gene for gene in genes if gene not in available]
    if missing:
        raise KeyError(f"{path}: missing requested genes: {missing[:20]}")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def aggregate_voxels(path: Path, genes: list[str], voxel_size: float, min_cells: int) -> pd.DataFrame:
    df = read_parquet(path, genes)
    coords = np.floor(df[["ccf_x", "ccf_y", "ccf_z"]].to_numpy(dtype=np.float64) / voxel_size).astype(np.int64)
    df["voxel_id"] = (
        pd.Series(coords[:, 0].astype(str), index=df.index)
        + "_"
        + pd.Series(coords[:, 1].astype(str), index=df.index)
        + "_"
        + pd.Series(coords[:, 2].astype(str), index=df.index)
    )
    grouped = df.groupby("voxel_id", sort=False)
    counts = grouped.size().rename("n_cells")
    mean = grouped[genes].mean()
    mean = mean.loc[counts[counts >= min_cells].index]
    mean.index.name = "voxel_id"
    print(f"[voxel] {path.name}: {len(mean):,} voxels after min_cells={min_cells}")
    return mean


def corr_pair(a: np.ndarray, b: np.ndarray, method: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    if method == "spearman":
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=10.0)
    parser.add_argument("--min-cells", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.parquet) != len(args.name):
        raise ValueError("--parquet and --name must have the same length")
    genes = read_gene_list(args.gene_list)
    if not genes:
        raise ValueError(f"No genes in {args.gene_list}")

    matrices = {
        name: aggregate_voxels(path, genes, voxel_size=args.voxel_size, min_cells=args.min_cells)
        for name, path in zip(args.name, args.parquet)
    }

    rows = []
    for name_a, name_b in itertools.combinations(args.name, 2):
        common_voxels = matrices[name_a].index.intersection(matrices[name_b].index)
        a_mat = matrices[name_a].loc[common_voxels]
        b_mat = matrices[name_b].loc[common_voxels]
        print(f"[pair] {name_a} vs {name_b}: {len(common_voxels):,} shared voxels")
        for gene in genes:
            a = a_mat[gene].to_numpy(dtype=np.float64)
            b = b_mat[gene].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "method_a": name_a,
                    "method_b": name_b,
                    "gene": gene,
                    "n_voxels": len(common_voxels),
                    "pearson": corr_pair(a, b, "pearson"),
                    "spearman": corr_pair(a, b, "spearman"),
                }
            )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    summary = (
        out.groupby(["method_a", "method_b"], as_index=False)
        .agg(
            n_genes=("gene", "count"),
            median_pearson=("pearson", "median"),
            mean_pearson=("pearson", "mean"),
            median_spearman=("spearman", "median"),
            mean_spearman=("spearman", "mean"),
        )
    )
    summary_path = args.summary or args.output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"[correlation] {args.output}")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
