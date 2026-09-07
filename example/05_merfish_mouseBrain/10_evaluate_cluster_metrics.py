from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/"
    "05_merfish_mouseBrain_mae_v1_3axes_train/rapids_analysis/"
    "merfish_mouseBrain_concat_embeddings.h5ad"
)
INVALID_VALUES = {"", "na", "n/a", "nan", "none", "<na>"}


@dataclass
class EncodedColumn:
    name: str
    categories: list[str]
    codes: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MERFISH cell and niche Leiden clusters globally and within "
            "individual brain regions without loading embedding matrices."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input parent>/cluster_evaluation.",
    )
    parser.add_argument("--cell-cluster-key", default="cell_leiden")
    parser.add_argument("--cell-label-key", default="subclass_transfer")
    parser.add_argument("--niche-cluster-key", default="niche_leiden")
    parser.add_argument("--region-key", default="major_brain_region")
    parser.add_argument("--niche-fine-label-key", default="ccf_region_name")
    parser.add_argument("--donor-key", default="donor_id")
    parser.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Exact major_brain_region values. Omit to evaluate every valid region.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=100,
        help="Minimum cells required for an evaluation scope.",
    )
    parser.add_argument(
        "--min-label-cells",
        type=int,
        default=1,
        help="Drop reference labels represented by fewer cells in each scope.",
    )
    parser.add_argument(
        "--include-na-region",
        action="store_true",
        help="Include categories such as n/a as brain regions.",
    )
    parser.add_argument(
        "--skip-donor-transfer",
        action="store_true",
        help="Skip leave-one-donor-out niche-to-region label transfer.",
    )
    return parser.parse_args()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_categorical(obs: h5py.Group, key: str) -> EncodedColumn:
    if key not in obs:
        raise KeyError(f"obs['{key}'] does not exist")
    group = obs[key]
    if not isinstance(group, h5py.Group) or not {"categories", "codes"}.issubset(group.keys()):
        raise TypeError(f"obs['{key}'] must be stored as an AnnData categorical column")
    return EncodedColumn(
        name=key,
        categories=[decode(value) for value in group["categories"][:]],
        codes=group["codes"][:].astype(np.int32, copy=False),
    )


def valid_category_mask(column: EncodedColumn, include_na: bool = False) -> np.ndarray:
    if include_na:
        return np.ones(len(column.categories), dtype=bool)
    return np.asarray(
        [category.strip().lower() not in INVALID_VALUES for category in column.categories],
        dtype=bool,
    )


def valid_code_mask(column: EncodedColumn, include_na: bool = False) -> np.ndarray:
    valid_categories = valid_category_mask(column, include_na=include_na)
    valid = column.codes >= 0
    lookup = np.where(valid, column.codes, 0)
    return valid & valid_categories[lookup]


def category_code(column: EncodedColumn, category: str) -> int:
    try:
        return column.categories.index(category)
    except ValueError as error:
        available = ", ".join(column.categories)
        raise ValueError(f"Unknown {column.name} value '{category}'. Available: {available}") from error


def contingency(
    cluster: EncodedColumn,
    label: EncodedColumn,
    scope_mask: np.ndarray,
    min_label_cells: int,
) -> tuple[np.ndarray, int]:
    mask = scope_mask & valid_code_mask(cluster, include_na=False) & valid_code_mask(label, include_na=False)
    if not np.any(mask):
        return np.zeros((len(cluster.categories), len(label.categories)), dtype=np.int64), 0

    if min_label_cells > 1:
        label_counts = np.bincount(label.codes[mask], minlength=len(label.categories))
        keep_labels = label_counts >= min_label_cells
        mask &= keep_labels[np.where(label.codes >= 0, label.codes, 0)]

    n_cluster = len(cluster.categories)
    n_label = len(label.categories)
    flat = cluster.codes[mask].astype(np.int64) * n_label + label.codes[mask]
    table = np.bincount(flat, minlength=n_cluster * n_label).reshape(n_cluster, n_label)
    return table, int(mask.sum())


def comb2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return values * (values - 1.0) / 2.0


def entropy(counts: np.ndarray, total: float) -> float:
    observed = counts[counts > 0].astype(np.float64, copy=False)
    if total <= 0 or observed.size == 0:
        return 0.0
    probabilities = observed / total
    return float(-(probabilities * np.log(probabilities)).sum())


def metrics_from_table(table: np.ndarray, min_cells: int) -> dict[str, float | int | str]:
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    observed_rows = row_sums > 0
    observed_cols = col_sums > 0
    total = int(table.sum())
    result: dict[str, float | int | str] = {
        "n_cells": total,
        "n_clusters": int(observed_rows.sum()),
        "n_labels": int(observed_cols.sum()),
        "status": "ok",
    }
    metric_names = [
        "ari",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "fowlkes_mallows",
        "pair_precision",
        "pair_recall",
        "cluster_purity_weighted",
        "cluster_purity_macro",
        "label_recovery_weighted",
        "label_recovery_macro",
    ]
    if total < min_cells:
        result["status"] = "too_few_cells"
        result.update({name: np.nan for name in metric_names})
        return result
    if observed_rows.sum() < 2 or observed_cols.sum() < 2:
        result["status"] = "fewer_than_two_clusters_or_labels"
        result.update({name: np.nan for name in metric_names})
        return result

    table_obs = table[np.ix_(observed_rows, observed_cols)].astype(np.float64, copy=False)
    row = table_obs.sum(axis=1)
    col = table_obs.sum(axis=0)
    n = float(total)

    pair_same_both = float(comb2(table_obs).sum())
    pair_same_cluster = float(comb2(row).sum())
    pair_same_label = float(comb2(col).sum())
    pair_total = n * (n - 1.0) / 2.0
    expected = pair_same_cluster * pair_same_label / pair_total if pair_total > 0 else 0.0
    max_index = 0.5 * (pair_same_cluster + pair_same_label)
    ari_denom = max_index - expected
    ari = 1.0 if ari_denom == 0 else (pair_same_both - expected) / ari_denom

    row_index, col_index = np.nonzero(table_obs)
    values = table_obs[row_index, col_index]
    mi = float(
        ((values / n) * np.log((values * n) / (row[row_index] * col[col_index]))).sum()
    )
    h_cluster = entropy(row, n)
    h_label = entropy(col, n)
    homogeneity = 1.0 if h_label == 0 else mi / h_label
    completeness = 1.0 if h_cluster == 0 else mi / h_cluster
    v_denom = homogeneity + completeness

    pair_precision = pair_same_both / pair_same_cluster if pair_same_cluster > 0 else np.nan
    pair_recall = pair_same_both / pair_same_label if pair_same_label > 0 else np.nan
    fm_denom = np.sqrt(pair_same_cluster * pair_same_label)

    row_purity = table_obs.max(axis=1) / row
    col_recovery = table_obs.max(axis=0) / col
    result.update(
        {
            "ari": float(ari),
            "nmi": float(mi / ((h_label + h_cluster) / 2.0)),
            "homogeneity": float(homogeneity),
            "completeness": float(completeness),
            "v_measure": float(0.0 if v_denom == 0 else 2.0 * homogeneity * completeness / v_denom),
            "fowlkes_mallows": float(pair_same_both / fm_denom) if fm_denom > 0 else np.nan,
            "pair_precision": float(pair_precision),
            "pair_recall": float(pair_recall),
            "cluster_purity_weighted": float(table_obs.max(axis=1).sum() / n),
            "cluster_purity_macro": float(row_purity.mean()),
            "label_recovery_weighted": float(table_obs.max(axis=0).sum() / n),
            "label_recovery_macro": float(col_recovery.mean()),
        }
    )
    return result


def evaluate_pair(
    analysis: str,
    cluster: EncodedColumn,
    label: EncodedColumn,
    scope_mask: np.ndarray,
    min_cells: int,
    min_label_cells: int,
    region: str = "all",
    donor: str = "all",
) -> dict[str, object]:
    table, _ = contingency(cluster, label, scope_mask, min_label_cells=min_label_cells)
    return {
        "analysis": analysis,
        "cluster_key": cluster.name,
        "label_key": label.name,
        "region": region,
        "donor": donor,
        **metrics_from_table(table, min_cells=min_cells),
    }


def summarize_donor_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    valid = rows.loc[rows["status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame()
    id_columns = ["analysis", "cluster_key", "label_key", "region"]
    metric_columns = [
        column
        for column in valid.columns
        if column not in {*id_columns, "donor", "status"} and pd.api.types.is_numeric_dtype(valid[column])
    ]
    summary = valid.groupby(id_columns, dropna=False)[metric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    donor_counts = valid.groupby(id_columns, dropna=False)["donor"].nunique().rename("n_donors")
    return summary.merge(donor_counts.reset_index(), on=id_columns, how="left")


def classification_metrics(confusion_true_pred: np.ndarray) -> dict[str, float]:
    table = confusion_true_pred.astype(np.float64, copy=False)
    support = table.sum(axis=1)
    predicted = table.sum(axis=0)
    tp = np.diag(table)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    iou = np.divide(tp, support + predicted - tp, out=np.zeros_like(tp), where=(support + predicted - tp) > 0)
    observed = support > 0
    total = support.sum()
    return {
        "accuracy": float(tp.sum() / total) if total > 0 else np.nan,
        "balanced_accuracy": float(recall[observed].mean()) if np.any(observed) else np.nan,
        "macro_f1": float(f1[observed].mean()) if np.any(observed) else np.nan,
        "weighted_f1": float(np.average(f1[observed], weights=support[observed])) if np.any(observed) else np.nan,
        "macro_iou": float(iou[observed].mean()) if np.any(observed) else np.nan,
    }


def donor_region_transfer(
    cluster: EncodedColumn,
    region: EncodedColumn,
    donor: EncodedColumn,
    include_na_region: bool,
    min_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_code_mask(cluster) & valid_code_mask(region, include_na=include_na_region) & valid_code_mask(donor)
    n_regions = len(region.categories)
    summary_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []

    for donor_code, donor_name in enumerate(donor.categories):
        test_mask = valid & (donor.codes == donor_code)
        train_mask = valid & (donor.codes != donor_code)
        train_table, _ = contingency(cluster, region, train_mask, min_label_cells=1)
        test_table, n_test = contingency(cluster, region, test_mask, min_label_cells=1)
        if n_test < min_cells or train_table.sum() == 0:
            continue

        global_majority = int(train_table.sum(axis=0).argmax())
        cluster_support = train_table.sum(axis=1)
        cluster_to_region = train_table.argmax(axis=1)
        cluster_to_region[cluster_support == 0] = global_majority

        confusion = np.zeros((n_regions, n_regions), dtype=np.int64)
        for cluster_code in np.flatnonzero(test_table.sum(axis=1) > 0):
            predicted_region = cluster_to_region[cluster_code]
            confusion[:, predicted_region] += test_table[cluster_code, :]

        classification = classification_metrics(confusion)
        clustering = metrics_from_table(confusion.T, min_cells=min_cells)
        summary_rows.append(
            {
                "analysis": "niche_region_leave_one_donor_out",
                "held_out_donor": donor_name,
                "n_cells": n_test,
                **classification,
                "ari": clustering["ari"],
                "nmi": clustering["nmi"],
            }
        )

        support = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)
        tp = np.diag(confusion).astype(np.float64)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
        iou = np.divide(tp, support + predicted - tp, out=np.zeros_like(tp), where=(support + predicted - tp) > 0)
        for region_code in np.flatnonzero(support > 0):
            region_rows.append(
                {
                    "held_out_donor": donor_name,
                    "region": region.categories[region_code],
                    "n_true": int(support[region_code]),
                    "n_predicted": int(predicted[region_code]),
                    "precision": float(precision[region_code]),
                    "recall": float(recall[region_code]),
                    "f1": float(f1[region_code]),
                    "iou": float(iou[region_code]),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(region_rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input.parent / "cluster_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_keys = {
        args.cell_cluster_key,
        args.cell_label_key,
        args.niche_cluster_key,
        args.region_key,
        args.niche_fine_label_key,
        args.donor_key,
    }
    print(f"Reading obs columns from {args.input}")
    with h5py.File(args.input, "r") as handle:
        obs = handle["obs"]
        columns = {key: read_categorical(obs, key) for key in requested_keys}

    cell_cluster = columns[args.cell_cluster_key]
    cell_label = columns[args.cell_label_key]
    niche_cluster = columns[args.niche_cluster_key]
    region = columns[args.region_key]
    niche_fine_label = columns[args.niche_fine_label_key]
    donor = columns[args.donor_key]
    n_obs = len(region.codes)
    all_cells = np.ones(n_obs, dtype=bool)

    global_rows = [
        evaluate_pair(
            "cell_global",
            cell_cluster,
            cell_label,
            all_cells,
            args.min_cells,
            args.min_label_cells,
        ),
        evaluate_pair(
            "niche_major_region_global",
            niche_cluster,
            region,
            all_cells,
            args.min_cells,
            args.min_label_cells,
        ),
        evaluate_pair(
            "niche_fine_region_global",
            niche_cluster,
            niche_fine_label,
            all_cells,
            args.min_cells,
            args.min_label_cells,
        ),
    ]
    global_metrics = pd.DataFrame(global_rows)
    global_metrics.to_csv(output_dir / "global_clustering_metrics.csv", index=False)

    valid_regions = valid_category_mask(region, include_na=args.include_na_region)
    observed_region_counts = np.bincount(region.codes[region.codes >= 0], minlength=len(region.categories))
    if args.regions:
        selected_regions = args.regions
        for region_name in selected_regions:
            category_code(region, region_name)
    else:
        selected_regions = [
            name
            for code, name in enumerate(region.categories)
            if valid_regions[code] and observed_region_counts[code] >= args.min_cells
        ]

    regional_rows: list[dict[str, object]] = []
    donor_rows: list[dict[str, object]] = []
    for region_name in selected_regions:
        region_code = category_code(region, region_name)
        region_mask = region.codes == region_code
        regional_rows.extend(
            [
                evaluate_pair(
                    "cell_within_region",
                    cell_cluster,
                    cell_label,
                    region_mask,
                    args.min_cells,
                    args.min_label_cells,
                    region=region_name,
                ),
                evaluate_pair(
                    "niche_within_region",
                    niche_cluster,
                    niche_fine_label,
                    region_mask,
                    args.min_cells,
                    args.min_label_cells,
                    region=region_name,
                ),
            ]
        )

        for donor_code, donor_name in enumerate(donor.categories):
            donor_region_mask = region_mask & (donor.codes == donor_code)
            donor_rows.extend(
                [
                    evaluate_pair(
                        "cell_within_region",
                        cell_cluster,
                        cell_label,
                        donor_region_mask,
                        args.min_cells,
                        args.min_label_cells,
                        region=region_name,
                        donor=donor_name,
                    ),
                    evaluate_pair(
                        "niche_within_region",
                        niche_cluster,
                        niche_fine_label,
                        donor_region_mask,
                        args.min_cells,
                        args.min_label_cells,
                        region=region_name,
                        donor=donor_name,
                    ),
                ]
            )

    regional_metrics = pd.DataFrame(regional_rows)
    regional_donor_metrics = pd.DataFrame(donor_rows)
    regional_metrics.to_csv(output_dir / "regional_clustering_metrics.csv", index=False)
    regional_donor_metrics.to_csv(output_dir / "regional_donor_metrics.csv", index=False)
    summarize_donor_metrics(regional_donor_metrics).to_csv(
        output_dir / "regional_donor_summary.csv", index=False
    )

    if not args.skip_donor_transfer:
        transfer_summary, transfer_regions = donor_region_transfer(
            niche_cluster,
            region,
            donor,
            include_na_region=args.include_na_region,
            min_cells=args.min_cells,
        )
        transfer_summary.to_csv(output_dir / "niche_region_transfer_by_donor.csv", index=False)
        transfer_regions.to_csv(output_dir / "niche_region_transfer_per_region.csv", index=False)

    manifest = {
        "input": str(args.input),
        "n_obs": n_obs,
        "regions": selected_regions,
        "cell_comparison": f"{args.cell_cluster_key} vs {args.cell_label_key}",
        "global_niche_comparisons": [
            f"{args.niche_cluster_key} vs {args.region_key}",
            f"{args.niche_cluster_key} vs {args.niche_fine_label_key}",
        ],
        "within_region_niche_comparison": (
            f"{args.niche_cluster_key} vs {args.niche_fine_label_key}; comparing "
            f"against {args.region_key} after subsetting would have only one reference class"
        ),
        "min_cells": args.min_cells,
        "min_label_cells": args.min_label_cells,
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nGlobal metrics:")
    print(global_metrics[["analysis", "n_cells", "n_clusters", "n_labels", "ari", "nmi", "v_measure"]].to_string(index=False))
    print(f"\nEvaluated {len(selected_regions)} regions. Results: {output_dir}")


if __name__ == "__main__":
    main()
