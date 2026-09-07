"""
Create the exact gene list used for ISH correlation.

The output is the intersection of:
  1. requested gene list
  2. columns present in every method parquet
  3. Allen ISH zip files present in --ish-dir
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

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


def parquet_genes(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    return {name for name in names if name not in META_COLS}


def ish_genes(ish_dir: Path) -> set[str]:
    genes: set[str] = set()
    for path in glob.glob(os.path.join(str(ish_dir), "*.zip")):
        name = Path(path).name
        if "_" in name:
            genes.add(name.rsplit("_", 1)[0].rstrip("*"))
    return genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--ish-dir", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.parquet) != len(args.name):
        raise ValueError("--parquet and --name must have the same length")

    requested = read_gene_list(args.gene_list)
    requested_set = set(requested)
    ish_set = ish_genes(args.ish_dir)
    method_sets = {name: parquet_genes(path) for name, path in zip(args.name, args.parquet)}

    common = requested_set & ish_set
    for genes in method_sets.values():
        common &= genes
    ordered = [gene for gene in requested if gene in common]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ordered) + ("\n" if ordered else ""))

    rows = [
        {"source": "requested_gene_list", "n_genes": len(requested_set)},
        {"source": "allen_ish_zip", "n_genes": len(ish_set & requested_set)},
    ]
    for name, genes in method_sets.items():
        rows.append({"source": name, "n_genes": len(genes & requested_set)})
    rows.append({"source": "intersection_used_for_correlation", "n_genes": len(ordered)})

    summary = pd.DataFrame(rows)
    summary_path = args.summary or args.output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"[genes] {len(ordered)} -> {args.output}")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
