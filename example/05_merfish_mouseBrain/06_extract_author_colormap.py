from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parent

DEFAULT_MEMBERSHIP = EXAMPLE_DIR / "cluster_to_cluster_annotation_membership.csv"
DEFAULT_H5AD = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/"
    "05_merfish_mouseBrain_mae_v1_3axes_train/rapids_analysis/"
    "merfish_mouseBrain_concat_embeddings.h5ad"
)
DEFAULT_OUTPUT_DIR = DEFAULT_H5AD.parent / "author_colormaps"

SET_ORDER = ["cluster", "supertype", "subclass", "class", "neurotransmitter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract author-provided annotation labels and colors from "
            "cluster_to_cluster_annotation_membership.csv."
        )
    )
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def strip_author_id(value) -> str:
    if pd.isna(value):
        return "NA"
    return re.sub(r"^\s*\d+\s+", "", str(value)).strip()


def clean_author_label(value) -> str:
    return "NA" if pd.isna(value) else str(value).strip()


def read_obs_categories(h5ad: Path, obs_key: str) -> list[str]:
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        if obs_key not in obs:
            raise KeyError(f"obs['{obs_key}'] does not exist in {h5ad}")
        group = obs[obs_key]
        if not isinstance(group, h5py.Group) or "categories" not in group:
            raise TypeError(f"obs['{obs_key}'] is not stored as an AnnData categorical")
        return [decode(value) for value in group["categories"][:]]


def load_membership(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "cluster_annotation_term_label",
        "cluster_annotation_term_set_label",
        "cluster_alias",
        "cluster_annotation_term_name",
        "cluster_annotation_term_set_name",
        "number_of_cells",
        "color_hex_triplet",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["annotation_set_name"] = df["cluster_annotation_term_set_name"].astype(str)
    df["annotation_set_label"] = df["cluster_annotation_term_set_label"].astype(str)
    df["annotation_term_label"] = df["cluster_annotation_term_label"].astype(str)
    df["author_label"] = df["cluster_annotation_term_name"].map(clean_author_label)
    df["author_label_clean"] = df["cluster_annotation_term_name"].map(strip_author_id)
    df["cluster_alias"] = df["cluster_alias"].astype(str)
    df["number_of_cells"] = pd.to_numeric(df["number_of_cells"], errors="coerce").fillna(0).astype(np.int64)
    df["color_hex_triplet"] = df["color_hex_triplet"].astype(str).str.strip()
    return df


def summarize_terms(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "annotation_set_name",
        "annotation_set_label",
        "annotation_term_label",
        "author_label",
        "author_label_clean",
        "color_hex_triplet",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_member_clusters=("cluster_alias", "nunique"),
            total_number_of_cells=("number_of_cells", "sum"),
        )
        .reset_index()
    )
    order = {name: idx for idx, name in enumerate(SET_ORDER)}
    summary["_set_order"] = summary["annotation_set_name"].map(order).fillna(len(order)).astype(int)
    summary = summary.sort_values(
        ["_set_order", "annotation_term_label", "author_label_clean"],
        kind="mergesort",
    ).drop(columns="_set_order")
    return summary


def one_term_per_clean_label(summary: pd.DataFrame, set_name: str) -> pd.DataFrame:
    terms = summary[summary["annotation_set_name"].eq(set_name)].copy()
    duplicated = terms["author_label_clean"].duplicated(keep=False)
    if duplicated.any():
        dupes = terms.loc[duplicated, "author_label_clean"].drop_duplicates().head(10).tolist()
        raise ValueError(f"{set_name} has non-unique cleaned labels, examples: {dupes}")
    return terms.set_index("author_label_clean", drop=False)


def write_cluster_id_colormap(df: pd.DataFrame, h5ad: Path, out_csv: Path) -> dict[str, str]:
    categories = read_obs_categories(h5ad, "cluster_id_transfer")
    cluster = df[df["annotation_set_name"].eq("cluster")].copy()
    cluster = cluster.sort_values("annotation_term_label").drop_duplicates("cluster_alias", keep="first")
    cluster_by_alias = cluster.set_index("cluster_alias", drop=False)

    rows = []
    palette = {}
    for category in categories:
        if category not in cluster_by_alias.index:
            raise ValueError(f"cluster_id_transfer category {category!r} is absent from membership CSV")
        row = cluster_by_alias.loc[category]
        palette[category] = row["color_hex_triplet"]
        rows.append(
            {
                "obs_key": "cluster_id_transfer",
                "obs_category": category,
                "cluster_alias": category,
                "annotation_set_name": row["annotation_set_name"],
                "annotation_set_label": row["annotation_set_label"],
                "annotation_term_label": row["annotation_term_label"],
                "author_label": row["author_label"],
                "author_label_clean": row["author_label_clean"],
                "number_of_cells": int(row["number_of_cells"]),
                "color_hex_triplet": row["color_hex_triplet"],
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return palette


def write_subclass_colormap(summary: pd.DataFrame, h5ad: Path, out_csv: Path) -> dict[str, str]:
    categories = read_obs_categories(h5ad, "subclass_transfer")
    subclass = one_term_per_clean_label(summary, "subclass")

    rows = []
    palette = {}
    for category in categories:
        if category not in subclass.index:
            raise ValueError(f"subclass_transfer category {category!r} is absent from membership CSV")
        row = subclass.loc[category]
        palette[category] = row["color_hex_triplet"]
        rows.append(
            {
                "obs_key": "subclass_transfer",
                "obs_category": category,
                "annotation_set_name": row["annotation_set_name"],
                "annotation_set_label": row["annotation_set_label"],
                "annotation_term_label": row["annotation_term_label"],
                "author_label": row["author_label"],
                "author_label_clean": row["author_label_clean"],
                "n_member_clusters": int(row["n_member_clusters"]),
                "total_number_of_cells": int(row["total_number_of_cells"]),
                "color_hex_triplet": row["color_hex_triplet"],
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return palette


def write_term_set_colormaps(summary: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, str]]:
    palettes = {}
    for set_name in SET_ORDER:
        subset = summary[summary["annotation_set_name"].eq(set_name)].copy()
        if subset.empty:
            continue
        out_csv = output_dir / f"author_{set_name}_colormap.csv"
        subset.to_csv(out_csv, index=False)
        palettes[set_name] = dict(zip(subset["author_label_clean"], subset["color_hex_triplet"]))
    return palettes


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_membership(args.membership)
    summary = summarize_terms(df)

    all_csv = args.output_dir / "author_label_colormap_all.csv"
    summary.to_csv(all_csv, index=False)

    term_set_palettes = write_term_set_colormaps(summary, args.output_dir)
    obs_palettes = {
        "cluster_id_transfer": write_cluster_id_colormap(
            df,
            args.h5ad,
            args.output_dir / "cluster_id_transfer_colormap.csv",
        ),
        "subclass_transfer": write_subclass_colormap(
            summary,
            args.h5ad,
            args.output_dir / "subclass_transfer_colormap.csv",
        ),
    }

    color_payload = {
        "source_membership": str(args.membership),
        "source_h5ad": str(args.h5ad),
        "obs_key_palettes": obs_palettes,
        "author_term_set_palettes": term_set_palettes,
    }
    with (args.output_dir / "author_colormap.json").open("w", encoding="utf-8") as handle:
        json.dump(color_payload, handle, ensure_ascii=False, indent=2)

    scanpy_uns_colors = {f"{key}_colors": list(palette.values()) for key, palette in obs_palettes.items()}
    with (args.output_dir / "scanpy_uns_colors.json").open("w", encoding="utf-8") as handle:
        json.dump(scanpy_uns_colors, handle, ensure_ascii=False, indent=2)

    print(f"output_dir={args.output_dir}")
    print(f"author terms={len(summary)}")
    for set_name in SET_ORDER:
        n = int(summary["annotation_set_name"].eq(set_name).sum())
        if n:
            print(f"  {set_name}: {n}")
    print(f"cluster_id_transfer colors={len(obs_palettes['cluster_id_transfer'])}")
    print(f"subclass_transfer colors={len(obs_palettes['subclass_transfer'])}")


if __name__ == "__main__":
    main()
