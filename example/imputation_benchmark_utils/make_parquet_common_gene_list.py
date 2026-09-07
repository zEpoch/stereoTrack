"""
Create a gene list from the intersection of method parquet columns.

This is used when there is no external ISH reference.  The output records the
exact genes that enter method-to-method correlation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


META_COLS = {"cell_id", "slice_id", "ccf_x", "ccf_y", "ccf_z"}


def read_optional_gene_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--gene-list", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.parquet) != len(args.name):
        raise ValueError("--parquet and --name must have the same length")

    requested = read_optional_gene_list(args.gene_list)
    method_sets = {name: parquet_genes(path) for name, path in zip(args.name, args.parquet)}

    common = None
    for genes in method_sets.values():
        common = set(genes) if common is None else common & genes
    common = common or set()

    if requested is None:
        ordered = sorted(common)
    else:
        ordered = [gene for gene in requested if gene in common]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ordered) + ("\n" if ordered else ""))

    rows = []
    if requested is not None:
        requested_set = set(requested)
        rows.append({"source": "requested_gene_list", "n_genes": len(requested_set)})
    else:
        requested_set = None

    for name, genes in method_sets.items():
        n_genes = len(genes if requested_set is None else genes & requested_set)
        rows.append({"source": name, "n_genes": n_genes})
    rows.append({"source": "intersection_used_for_method_correlation", "n_genes": len(ordered)})

    summary = pd.DataFrame(rows)
    summary_path = args.summary or args.output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"[genes] {len(ordered)} -> {args.output}")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
