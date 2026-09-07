"""Audit and build shared annotation tables from balanced inference outputs."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from common_ontology import harmonize_cell_family, harmonize_layer


DEFAULT_ROOT = Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a conservative shared cortex ontology")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_ROOT / "out/04_three_species_mae_v1_train_balanced_v1/inference_adatas",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "out/04_three_species_cross_species/ontology",
    )
    parser.add_argument("--limit-files", type=int, default=None)
    return parser.parse_args()


def obs_column(obs: pd.DataFrame, name: str, default: str = "Unknown") -> pd.Series:
    if name in obs.columns:
        return obs[name].astype("string").fillna(default).astype(str)
    return pd.Series(default, index=obs.index, dtype=str)


def aggregate_rows(parts: list[pd.DataFrame], group_columns: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=group_columns + ["n_cells"])
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.groupby(group_columns, observed=True, dropna=False)["n_cells"]
        .sum()
        .reset_index()
        .sort_values(group_columns + ["n_cells"], ascending=[True] * len(group_columns) + [False])
    )


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.h5ad"))
    if args.limit_files is not None:
        selected = []
        for species in ["macaque", "marmoset", "mouse"]:
            selected.extend([path for path in files if path.name.startswith(species + "_")][: args.limit_files])
        files = sorted(selected)
    if not files:
        raise FileNotFoundError(f"No h5ad files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    family_count_parts = []
    family_source_parts = []
    layer_count_parts = []
    layer_source_parts = []
    region_parts = []
    coverage_rows = []

    for file_number, file_path in enumerate(files, start=1):
        adata = ad.read_h5ad(file_path, backed="r")
        obs = adata.obs.copy()
        adata.file.close()

        species = obs_column(obs, "species", file_path.name.split("_", 1)[0]).str.lower()
        species_name = str(species.iloc[0])
        broad = obs_column(obs, "cell_class_broad")
        subclass = obs_column(obs, "cell_subclass")
        cell_type = obs_column(obs, "cell_type")
        layer = obs_column(obs, "layer")
        region = obs_column(obs, "region")
        region_main = obs_column(obs, "region_main")

        family_common = harmonize_cell_family(species, broad, subclass, cell_type)
        layer_common = harmonize_layer(layer)

        family_frame = pd.DataFrame({"species": species_name, "cell_family_common": family_common})
        family_count_parts.append(
            family_frame.value_counts(sort=False).rename("n_cells").reset_index()
        )

        source_frame = pd.DataFrame(
            {
                "species": species_name,
                "cell_class_broad": broad,
                "cell_subclass": subclass,
                "cell_type": cell_type,
                "cell_family_common": family_common,
            }
        )
        family_source_parts.append(
            source_frame.value_counts(sort=False).rename("n_cells").reset_index()
        )

        layer_frame = pd.DataFrame(
            {"species": species_name, "layer_source": layer, "layer_common": layer_common}
        )
        layer_source_parts.append(
            layer_frame.value_counts(sort=False).rename("n_cells").reset_index()
        )
        layer_count_parts.append(
            layer_frame[["species", "layer_common"]]
            .value_counts(sort=False)
            .rename("n_cells")
            .reset_index()
        )

        region_frame = pd.DataFrame(
            {
                "species": species_name,
                "region_main_source": region_main,
                "region_source": region,
            }
        )
        region_parts.append(
            region_frame.value_counts(sort=False).rename("n_cells").reset_index()
        )

        coverage_rows.append(
            {
                "species": species_name,
                "file": file_path.name,
                "n_cells": len(obs),
                "cell_family_unknown_fraction": float(family_common.eq("Unknown").mean()),
                "layer_common_unknown_fraction": float(layer_common.eq("Unknown").mean()),
            }
        )
        print(f"[{file_number}/{len(files)}] {file_path.name}: {len(obs)} cells")
        del obs, family_common, layer_common
        gc.collect()

    family_counts = aggregate_rows(
        family_count_parts, ["species", "cell_family_common"]
    )
    family_sources = aggregate_rows(
        family_source_parts,
        ["species", "cell_class_broad", "cell_subclass", "cell_type", "cell_family_common"],
    )
    layer_counts = aggregate_rows(layer_count_parts, ["species", "layer_common"])
    layer_sources = aggregate_rows(
        layer_source_parts, ["species", "layer_source", "layer_common"]
    )
    region_template = aggregate_rows(
        region_parts, ["species", "region_main_source", "region_source"]
    )
    region_template["region_group_common"] = ""
    region_template["region_common"] = ""
    region_template["mapping_notes"] = ""

    coverage_by_file = pd.DataFrame(coverage_rows)
    coverage_by_species = (
        coverage_by_file.assign(
            unknown_family=lambda frame: frame["n_cells"] * frame["cell_family_unknown_fraction"],
            unknown_layer=lambda frame: frame["n_cells"] * frame["layer_common_unknown_fraction"],
        )
        .groupby("species", observed=True)
        .agg(n_files=("file", "size"), n_cells=("n_cells", "sum"), unknown_family=("unknown_family", "sum"), unknown_layer=("unknown_layer", "sum"))
        .reset_index()
    )
    coverage_by_species["cell_family_unknown_fraction"] = (
        coverage_by_species["unknown_family"] / coverage_by_species["n_cells"]
    )
    coverage_by_species["layer_common_unknown_fraction"] = (
        coverage_by_species["unknown_layer"] / coverage_by_species["n_cells"]
    )
    coverage_by_species = coverage_by_species.drop(columns=["unknown_family", "unknown_layer"])

    family_counts.to_csv(args.output_dir / "cell_family_common_counts.csv", index=False)
    family_sources.to_csv(args.output_dir / "cell_family_source_mapping.csv", index=False)
    layer_counts.to_csv(args.output_dir / "layer_common_counts.csv", index=False)
    layer_sources.to_csv(args.output_dir / "layer_source_mapping.csv", index=False)
    region_template.to_csv(args.output_dir / "region_mapping_template.csv", index=False)
    coverage_by_file.to_csv(args.output_dir / "ontology_coverage_by_file.csv", index=False)
    coverage_by_species.to_csv(args.output_dir / "ontology_coverage_by_species.csv", index=False)

    print("\nCell family counts:")
    print(family_counts.to_string(index=False))
    print("\nLayer counts:")
    print(layer_counts.to_string(index=False))
    print(f"\nOntology outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
