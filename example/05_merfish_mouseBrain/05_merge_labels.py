from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_H5AD = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/"
    "05_merfish_mouseBrain_mae_v1_train/rapids_analysis/merfish_mouseBrain_concat_embeddings.h5ad"
)

NT_SUFFIXES = [
    "Glyc-Gaba",
    "Dopa-Gaba",
    "Gly-Gaba",
    "Hist-Gaba",
    "Gaba-Glut",
    "Gaba-Chol",
    "Glut-Chol",
    "Glut-Sero",
    "Inh IMN",
    "Dopa",
    "Chol",
    "Glut",
    "Gaba",
    "NN",
    "IMN",
]

NON_NEURON_PREFIXES = [
    ("Astroependymal", "Astroependymal"),
    ("Astro", "Astro"),
    ("Oligo", "Oligo"),
    ("OPC", "OPC"),
    ("Microglia", "Microglia"),
    ("Endo", "Endo"),
    ("Peri", "Pericyte"),
    ("SMC", "SMC"),
    ("VLMC", "VLMC"),
    ("Tanycyte", "Tanycyte"),
    ("Ependymal", "Ependymal"),
    ("Hypendymal", "Hypendymal"),
    ("CHOR", "Choroid"),
    ("DC", "Dendritic"),
    ("Lymphoid", "Lymphoid"),
    ("Monocytes", "Monocyte"),
    ("BAM", "Macrophage"),
    ("OEC", "OEC"),
    ("Bergmann", "Bergmann"),
    ("ABC", "ABC"),
]

ANATOMICAL_TOKENS = set(
    "IT ET CT NP CTX ENT PPP SUB RSP ACA AON TT DP EP CLA MLI PLI "
    "D1 D2 D3 Granule Purkinje Golgi chandelier PIR TPE APr".split()
)

SPECIAL_KEEP_TWO = {
    ("Pvalb", "chandelier"),
    ("Sst", "Chodl"),
    ("Lamp5", "Lhx6"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add medium-granularity merged labels to MERFISH integrated h5ad.")
    parser.add_argument("--input", type=Path, default=DEFAULT_H5AD, help="Integrated h5ad from RAPIDS concat.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output h5ad. If omitted, labels are appended to --input in place.",
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=None,
        help="Directory for CSV mapping tables. Defaults to <h5ad parent>/merged_label_rules.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report category counts and write mapping tables.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing merged label columns.")
    return parser.parse_args()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_categorical(obs: h5py.Group, column: str, read_codes: bool = True) -> tuple[list[str], np.ndarray | None]:
    if column not in obs:
        raise KeyError(f"obs['{column}'] does not exist")
    group = obs[column]
    if not isinstance(group, h5py.Group) or "categories" not in group or "codes" not in group:
        raise TypeError(f"obs['{column}'] is expected to be an AnnData categorical column")
    categories = [decode(value) for value in group["categories"][:]]
    codes = group["codes"][:] if read_codes else None
    return categories, codes


def code_dtype(n_categories: int):
    if n_categories <= np.iinfo(np.int8).max:
        return np.int8
    if n_categories <= np.iinfo(np.int16).max:
        return np.int16
    return np.int32


def write_categorical_column(
    obs: h5py.Group,
    column: str,
    categories: list[str],
    codes: np.ndarray,
    overwrite: bool = False,
) -> None:
    if column in obs:
        if not overwrite:
            raise RuntimeError(f"obs['{column}'] already exists. Use --overwrite to replace it.")
        del obs[column]

    dtype = code_dtype(len(categories))
    group = obs.create_group(column)
    group.attrs["encoding-type"] = "categorical"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["ordered"] = False

    codes_ds = group.create_dataset("codes", data=codes.astype(dtype, copy=False), dtype=dtype, chunks=True)
    codes_ds.attrs["encoding-type"] = "array"
    codes_ds.attrs["encoding-version"] = "0.2.0"

    str_dtype = h5py.string_dtype(encoding="utf-8")
    cats_ds = group.create_dataset("categories", data=np.asarray(categories, dtype=str_dtype), dtype=str_dtype)
    cats_ds.attrs["encoding-type"] = "string-array"
    cats_ds.attrs["encoding-version"] = "0.2.0"


def update_column_order(obs: h5py.Group, columns: list[str]) -> None:
    current = [decode(value) for value in obs.attrs.get("column-order", [])]
    for column in columns:
        if column not in current:
            current.append(column)
    if "column-order" in obs.attrs:
        del obs.attrs["column-order"]
    obs.attrs.create("column-order", np.asarray(current, dtype=h5py.string_dtype("utf-8")))


def observed_counts(codes: np.ndarray, n_categories: int) -> np.ndarray:
    valid = codes[codes >= 0]
    return np.bincount(valid.astype(np.int64), minlength=n_categories)


def ccf_layerless(label: str) -> str:
    label = re.sub(r"(?i)(,\s*|/)layer\s*[0-9a-z/]+$", "", label)
    label = re.sub(
        r"(?i),\s*(molecular|granule cell|polymorph|glomerular|granular|mitral|pyramidal)\s+layer$",
        "",
        label,
    )
    label = re.sub(r"(?i),\s*(?:[1-6][ab]?|2/3)$", "", label)
    return label.strip(" ,/")


def ccf_group(label: str) -> str:
    label = ccf_layerless(label)
    label = re.sub(
        r"(?i),\s*"
        r"(anterior|posterior|dorsal|ventral|medial|lateral|central|capsular|external|"
        r"magnocellular|parvicellular|dorsolateral|dorsomedial|rostrolateral|rostral|"
        r"caudal|intermediate|apical|core|shell|ipsilateral)\s+"
        r"(part|zone|division)$",
        "",
        label,
    )
    label = re.sub(
        r"(?i),\s*"
        r"(anterior|posterior|dorsal|ventral|medial|lateral|central|external|"
        r"caudal|rostral|intermediate|apical|core|shell|ipsilateral)$",
        "",
        label,
    )
    return label.strip(" ,/")


def neurotransmitter(label: str) -> str:
    for suffix in NT_SUFFIXES:
        if label == suffix or label.endswith(" " + suffix):
            return "Gly-Gaba" if suffix == "Glyc-Gaba" else suffix
    return "Other"


def strip_neurotransmitter(label: str) -> str:
    for suffix in NT_SUFFIXES:
        if label.endswith(" " + suffix):
            return label[: -(len(suffix) + 1)].strip()
    return label


def non_neuron_group(base: str) -> str | None:
    for prefix, label in NON_NEURON_PREFIXES:
        if base == prefix or base.startswith(prefix + "-") or base.startswith(prefix + " "):
            return label
    return None


def cortical_projection_group(base: str) -> str | None:
    tokens = base.split()
    if not tokens:
        return None
    if tokens[0].startswith("L") and any(token in tokens for token in ["IT", "ET", "CT", "NP"]):
        projection = next((token for token in tokens if token in ["IT", "ET", "CT", "NP"]), "")
        target = "CTX" if "CTX" in tokens else (tokens[-1] if tokens[-1].isupper() or "-" in tokens[-1] else "")
        return " ".join(item for item in [tokens[0], projection, target] if item)
    if tokens[0] in ["IT", "NP", "CT"]:
        return " ".join(tokens[: min(2, len(tokens))])
    return None


def subclass_fine_group(label: str) -> str:
    base = strip_neurotransmitter(label).replace(" Inh", "").strip()
    non_neuron = non_neuron_group(base)
    if non_neuron is not None:
        return non_neuron

    tokens = base.split()
    if not tokens:
        return "Unknown"

    keep = [tokens[0]]
    for token in tokens[1:]:
        if (
            (keep[0], token) in SPECIAL_KEEP_TWO
            or token in ANATOMICAL_TOKENS
            or token.isupper()
            or "-" in token
            or re.fullmatch(r"L[0-9/ab]+", token)
        ):
            keep.append(token)
        else:
            break
    return " ".join(keep)


def subclass_group(label: str) -> str:
    base = strip_neurotransmitter(label).replace(" Inh", "").strip()
    non_neuron = non_neuron_group(base)
    if non_neuron is not None:
        return non_neuron

    cortical = cortical_projection_group(base)
    if cortical is not None:
        return cortical

    tokens = base.split()
    if not tokens:
        return "Unknown"

    if tokens[0] in ["Pvalb", "Sst", "Vip", "Sncg", "Lamp5"]:
        if len(tokens) > 1 and (tokens[0], tokens[1]) in SPECIAL_KEEP_TWO:
            return tokens[0] + " " + tokens[1]
        return tokens[0]

    root = tokens[0].split("-")[0]
    if root in ["STR", "ACB"] and len(tokens) > 1 and tokens[1] in ["D1", "D2", "D3"]:
        return root + " " + tokens[1]
    if root in ["CB", "CBX", "CBN"] and len(tokens) > 1 and tokens[1] in [
        "Granule",
        "PLI",
        "MLI",
        "Purkinje",
        "Golgi",
    ]:
        return root + " " + tokens[1]
    return root


def subclass_group_nt(label: str) -> str:
    group = subclass_group(label)
    nt = neurotransmitter(label)
    if nt in ["NN", "Other"]:
        return group
    return f"{group} {nt}"


def mapped_codes(
    source_categories: list[str],
    source_codes: np.ndarray,
    mapper,
) -> tuple[list[str], np.ndarray, list[str]]:
    labels_by_source = [mapper(category) for category in source_categories]
    target_categories = list(dict.fromkeys(labels_by_source))
    target_index = {label: idx for idx, label in enumerate(target_categories)}
    old_to_new = np.asarray([target_index[label] for label in labels_by_source], dtype=np.int32)

    new_codes = np.full(source_codes.shape, -1, dtype=code_dtype(len(target_categories)))
    valid = source_codes >= 0
    new_codes[valid] = old_to_new[source_codes[valid].astype(np.int64)]
    return target_categories, new_codes, labels_by_source


def context_codes(
    ccf_categories: list[str],
    ccf_codes: np.ndarray,
    major_categories: list[str],
    major_codes: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    ccf_groups = [ccf_group(category) for category in ccf_categories]
    label_to_code: dict[str, int] = {}
    pair_to_code = np.full((len(major_categories), len(ccf_categories)), -1, dtype=np.int32)

    valid = (ccf_codes >= 0) & (major_codes >= 0)
    pair_ids = np.unique(major_codes[valid].astype(np.int64) * len(ccf_categories) + ccf_codes[valid].astype(np.int64))
    for pair_id in pair_ids:
        major_idx = int(pair_id // len(ccf_categories))
        ccf_idx = int(pair_id % len(ccf_categories))
        major_label = major_categories[major_idx]
        region_label = ccf_groups[ccf_idx]
        label = region_label if major_label in ["", "nan", "n/a", "NA"] else f"{major_label}::{region_label}"
        if label not in label_to_code:
            label_to_code[label] = len(label_to_code)
        pair_to_code[major_idx, ccf_idx] = label_to_code[label]

    target_categories = [None] * len(label_to_code)
    for label, idx in label_to_code.items():
        target_categories[idx] = label

    new_codes = np.full(ccf_codes.shape, -1, dtype=code_dtype(len(target_categories)))
    new_codes[valid] = pair_to_code[major_codes[valid].astype(np.int64), ccf_codes[valid].astype(np.int64)]
    return target_categories, new_codes


def write_mapping_tables(
    mapping_dir: Path,
    ccf_categories: list[str],
    ccf_counts: np.ndarray,
    subclass_categories: list[str],
    subclass_counts: np.ndarray,
) -> None:
    mapping_dir.mkdir(parents=True, exist_ok=True)

    ccf_df = pd.DataFrame(
        {
            "ccf_region_name": ccf_categories,
            "n_cells": ccf_counts,
            "ccf_region_layerless": [ccf_layerless(category) for category in ccf_categories],
            "ccf_region_group": [ccf_group(category) for category in ccf_categories],
        }
    )
    ccf_df.to_csv(mapping_dir / "ccf_region_name_to_merged_labels.csv", index=False)

    subclass_df = pd.DataFrame(
        {
            "subclass_transfer": subclass_categories,
            "n_cells": subclass_counts,
            "subclass_neurotransmitter": [neurotransmitter(category) for category in subclass_categories],
            "subclass_fine_group": [subclass_fine_group(category) for category in subclass_categories],
            "subclass_group": [subclass_group(category) for category in subclass_categories],
            "subclass_group_nt": [subclass_group_nt(category) for category in subclass_categories],
        }
    )
    subclass_df.to_csv(mapping_dir / "subclass_transfer_to_merged_labels.csv", index=False)


def summarize(name: str, codes: np.ndarray, categories: list[str]) -> dict:
    return {
        "label": name,
        "n_categories_defined": len(categories),
        "n_categories_observed": int(np.unique(codes[codes >= 0]).size),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path
    mapping_dir = args.mapping_dir or output_path.parent / "merged_label_rules"

    if args.output is not None and output_path != input_path:
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
            output_path.unlink()
        if not args.dry_run:
            print(f"copy h5ad: {input_path} -> {output_path}")
            shutil.copy2(input_path, output_path)

    with h5py.File(output_path if not args.dry_run else input_path, "r" if args.dry_run else "r+") as handle:
        obs = handle["obs"]
        ccf_categories, ccf_codes = read_categorical(obs, "ccf_region_name")
        major_categories, major_codes = read_categorical(obs, "major_brain_region")
        subclass_categories, subclass_codes = read_categorical(obs, "subclass_transfer")

        ccf_counts = observed_counts(ccf_codes, len(ccf_categories))
        subclass_counts = observed_counts(subclass_codes, len(subclass_categories))
        write_mapping_tables(mapping_dir, ccf_categories, ccf_counts, subclass_categories, subclass_counts)

        columns: dict[str, tuple[list[str], np.ndarray]] = {}
        columns["ccf_region_layerless"] = mapped_codes(ccf_categories, ccf_codes, ccf_layerless)[:2]
        columns["ccf_region_group"] = mapped_codes(ccf_categories, ccf_codes, ccf_group)[:2]
        columns["ccf_region_context"] = context_codes(ccf_categories, ccf_codes, major_categories, major_codes)
        columns["subclass_neurotransmitter"] = mapped_codes(subclass_categories, subclass_codes, neurotransmitter)[:2]
        columns["subclass_fine_group"] = mapped_codes(subclass_categories, subclass_codes, subclass_fine_group)[:2]
        columns["subclass_group"] = mapped_codes(subclass_categories, subclass_codes, subclass_group)[:2]
        columns["subclass_group_nt"] = mapped_codes(subclass_categories, subclass_codes, subclass_group_nt)[:2]

        summary_rows = [
            summarize("ccf_region_name", ccf_codes, ccf_categories),
            summarize("subclass_transfer", subclass_codes, subclass_categories),
            summarize("major_brain_region", major_codes, major_categories),
        ]
        for column, (categories, codes) in columns.items():
            summary_rows.append(summarize(column, codes, categories))

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(mapping_dir / "merged_label_summary.csv", index=False)
        print(summary.to_string(index=False))
        print(f"\nMapping tables written to: {mapping_dir}")

        if args.dry_run:
            print("dry-run only; h5ad was not modified")
            return

        for column, (categories, codes) in columns.items():
            write_categorical_column(obs, column, categories, codes, overwrite=args.overwrite)
            print(f"wrote obs['{column}'] with {len(categories)} categories")

        update_column_order(obs, list(columns))
        print(f"\nMerged labels appended to: {output_path}")


if __name__ == "__main__":
    main()
