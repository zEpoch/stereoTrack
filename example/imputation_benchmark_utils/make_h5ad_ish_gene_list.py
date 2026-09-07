"""
Create a benchmark target gene list from h5ad files and local Allen ISH zips.

The output is:
  common genes across all h5ad slices  intersect  genes with local Allen 3D ISH zip files.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import zipfile
from pathlib import Path

import h5py
import pandas as pd


def sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for pattern in (r"(?:^|_)T(\d+)(?:_|$)", r"(?:^|_)slice_(\d+)(?:_|$)", r"(\d+)"):
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1)), stem
    return 10**18, stem


def clean_gene(raw: object) -> str | None:
    if isinstance(raw, bytes):
        raw = raw.decode()
    gene = str(raw).strip()
    if not gene or gene.lower() in {"nan", "none"}:
        return None
    return gene


def read_h5_string_array(node) -> list[object]:
    if isinstance(node, h5py.Dataset):
        return node[()].tolist()
    if isinstance(node, h5py.Group) and {"codes", "categories"}.issubset(node.keys()):
        codes = node["codes"][()]
        categories = read_h5_string_array(node["categories"])
        values: list[object] = []
        for code in codes:
            values.append(categories[int(code)] if int(code) >= 0 else "")
        return values
    raise TypeError(f"Cannot read h5ad string array from {node.name}")


def read_var_column(var_group: h5py.Group, key: str) -> list[object]:
    if key not in var_group:
        raise KeyError(key)
    return read_h5_string_array(var_group[key])


def h5ad_genes(path: Path, gene_name_column: str | None) -> list[str]:
    with h5py.File(path, "r") as handle:
        var_group = handle["var"]
        index_key = var_group.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        if gene_name_column and gene_name_column in var_group:
            raw_genes = read_var_column(var_group, gene_name_column)
        else:
            raw_genes = read_var_column(var_group, str(index_key))

    genes: list[str] = []
    seen: set[str] = set()
    for raw in raw_genes:
        gene = clean_gene(raw)
        if gene and gene not in seen:
            seen.add(gene)
            genes.append(gene)
    return genes


def common_h5ad_genes(h5ads: list[Path], gene_name_column: str | None) -> tuple[list[str], list[dict[str, object]]]:
    first_order: list[str] | None = None
    common: set[str] | None = None
    rows: list[dict[str, object]] = []

    for path in h5ads:
        genes = h5ad_genes(path, gene_name_column)
        gene_set = set(genes)
        if first_order is None:
            first_order = genes
            common = set(genes)
        else:
            common &= gene_set
        rows.append({"source": path.name, "n_genes": len(gene_set)})

    if first_order is None or common is None:
        return [], rows
    return [gene for gene in first_order if gene in common], rows


def ish_genes(ish_dir: Path) -> set[str]:
    genes: set[str] = set()
    for path in glob.glob(os.path.join(str(ish_dir), "*.zip")):
        if not zipfile.is_zipfile(path):
            continue
        stem = Path(path).stem
        if "_" in stem:
            gene = clean_gene(stem.rsplit("_", 1)[0].rstrip("*"))
            if gene:
                genes.add(gene)
    return genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--file-glob", default="*.h5ad")
    parser.add_argument("--ish-dir", type=Path, required=True)
    parser.add_argument("--gene-name-column", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5ads = sorted(args.input_dir.glob(args.file_glob), key=sort_key)
    if not h5ads:
        raise FileNotFoundError(f"No {args.file_glob} files found in {args.input_dir}")

    common_genes, slice_rows = common_h5ad_genes(h5ads, args.gene_name_column)
    common_set = set(common_genes)
    ish_set = ish_genes(args.ish_dir)
    ordered = [gene for gene in common_genes if gene in ish_set]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ordered) + ("\n" if ordered else ""))

    summary_rows = [
        {"source": "h5ad_files", "n_genes": len(common_set), "detail": f"common across {len(h5ads)} files"},
        {"source": "allen_ish_zip", "n_genes": len(ish_set), "detail": str(args.ish_dir)},
        {"source": "h5ad_common_intersect_ish", "n_genes": len(ordered), "detail": str(args.output)},
    ]
    summary_rows.extend(
        {"source": row["source"], "n_genes": row["n_genes"], "detail": "per-file h5ad genes"} for row in slice_rows
    )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.summary or args.output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.head(3).to_string(index=False))
    print(f"[genes] {len(ordered)} -> {args.output}")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
