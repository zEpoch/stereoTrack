from __future__ import annotations

import argparse
import gc
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


CELL_LABEL_DEFAULTS = [
    "cell_type",
    "subclass_transfer",
    "cluster_id_transfer",
    "cell_type_ontology_term_id",
]
NICHE_LABEL_DEFAULTS = [
    "major_brain_region",
    "ccf_region_name",
    "brain_section_label",
    "tissue",
    "tissue_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concat MERFISH mouse brain inference h5ad files and cluster cell/niche embeddings with rapids-singlecell."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_train/inference_adatas"),
        help="Directory containing inference h5ad files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_train/rapids_analysis"),
        help="Directory for concatenated h5ad and clustering outputs.",
    )
    parser.add_argument("--pattern", type=str, default="*.h5ad")
    parser.add_argument("--cell-key", type=str, default="cell_embedding")
    parser.add_argument("--niche-key", type=str, default="niche_embedding")
    parser.add_argument("--cell-label-cols", nargs="*", default=CELL_LABEL_DEFAULTS)
    parser.add_argument("--niche-label-cols", nargs="*", default=NICHE_LABEL_DEFAULTS)
    parser.add_argument("--n-neighbors", type=int, default=30, help="Default neighbors for both embeddings.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Default Leiden resolution for both embeddings.")
    parser.add_argument("--cell-n-neighbors", type=int, default=None, help="Neighbors for cell_embedding clustering.")
    parser.add_argument("--cell-resolution", type=float, default=None, help="Leiden resolution for cell_embedding clustering.")
    parser.add_argument("--niche-n-neighbors", type=int, default=None, help="Neighbors for niche_embedding clustering.")
    parser.add_argument("--niche-resolution", type=float, default=None, help="Leiden resolution for niche_embedding clustering.")
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--algorithm", type=str, default="ivfflat")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max-cells", type=int, default=None, help="Optional global downsample after concat.")
    parser.add_argument("--max-cells-per-file", type=int, default=None, help="Optional downsample per h5ad before concat.")
    parser.add_argument("--run-umap", action="store_true", help="Run RAPIDS UMAP for both embeddings.")
    parser.add_argument("--save-umap-plots", action="store_true", help="Save UMAP scatter PNGs. Requires --run-umap.")
    parser.add_argument("--plot-max-cells", type=int, default=200000, help="Max cells used for UMAP scatter plots.")
    parser.add_argument("--no-write-concat", action="store_true", help="Do not write the clustered concatenated h5ad.")
    parser.add_argument("--dry-run", action="store_true", help="Only concat and report shape/columns; skip RAPIDS clustering.")
    return parser.parse_args()


def list_files(input_dir: Path, pattern: str, max_files: int | None) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No h5ad files found in {input_dir} with pattern {pattern}")
    return files


def ensure_unique_obs_names(adata: ad.AnnData, file_path: Path) -> None:
    adata.obs_names = pd.Index([f"{file_path.stem}:{name}" for name in adata.obs_names.astype(str)])


def read_and_concat(args: argparse.Namespace) -> ad.AnnData:
    files = list_files(args.input_dir, args.pattern, args.max_files)
    rng = np.random.default_rng(args.random_seed)
    adatas: list[ad.AnnData] = []

    for file_idx, file_path in enumerate(files, start=1):
        adata = sc.read_h5ad(file_path)
        missing = [key for key in [args.cell_key, args.niche_key] if key not in adata.obsm]
        if missing:
            raise KeyError(f"{file_path.name}: missing obsm keys {missing}")

        if args.max_cells_per_file is not None and adata.n_obs > args.max_cells_per_file:
            selected = np.sort(rng.choice(adata.n_obs, size=args.max_cells_per_file, replace=False))
            adata = adata[selected].copy()

        ensure_unique_obs_names(adata, file_path)
        adata.obs["source_file"] = file_path.name
        adata.obs["source_stem"] = file_path.stem
        adatas.append(adata)
        print(f"[read {file_idx}/{len(files)}] {file_path.name}: {adata.n_obs} cells, {adata.n_vars} genes")

    print("Concatenating h5ad files ...")
    adata_all = ad.concat(adatas, join="outer", merge="same", uns_merge="same")
    adata_all.obs_names_make_unique()

    if args.max_cells is not None and adata_all.n_obs > args.max_cells:
        selected = np.sort(rng.choice(adata_all.n_obs, size=args.max_cells, replace=False))
        adata_all = adata_all[selected].copy()
        print(f"global downsample: {args.max_cells} cells")

    print(f"concat shape: {adata_all.n_obs} cells x {adata_all.n_vars} genes")
    print(f"obs columns: {list(adata_all.obs.columns)}")
    print(f"obsm keys: {list(adata_all.obsm.keys())}")
    print(f"layers: {list(adata_all.layers.keys())}")

    del adatas
    gc.collect()
    return adata_all


def to_label_series(obs: pd.DataFrame, column: str) -> pd.Series:
    return obs[column].astype("string").fillna("NA").astype(str)


def available_columns(obs: pd.DataFrame, requested: list[str]) -> list[str]:
    return [column for column in requested if column in obs.columns and obs[column].nunique(dropna=True) > 1]


def encode_obs_column(obs: pd.DataFrame, column: str) -> tuple[np.ndarray, list[str]]:
    series = obs[column]
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = series.cat.categories.astype(str).tolist()
        codes = series.cat.codes.to_numpy(dtype=np.int32, copy=False)
    else:
        categorical = series.astype("string").fillna("NA").astype(str).astype("category")
        categories = categorical.cat.categories.astype(str).tolist()
        codes = categorical.cat.codes.to_numpy(dtype=np.int32, copy=False)
    return codes, categories


def valid_category_codes(categories: list[str]) -> np.ndarray:
    invalid_values = {"", "NA", "nan", "None", "<NA>"}
    return np.asarray([category not in invalid_values for category in categories], dtype=bool)


def contingency_counts(obs: pd.DataFrame, cluster_key: str, label_col: str) -> pd.DataFrame:
    cluster_codes, cluster_categories = encode_obs_column(obs, cluster_key)
    label_codes, label_categories = encode_obs_column(obs, label_col)
    valid_cluster = valid_category_codes(cluster_categories)
    valid_label = valid_category_codes(label_categories)
    cluster_lookup = np.where(cluster_codes < 0, 0, cluster_codes)
    label_lookup = np.where(label_codes < 0, 0, label_codes)

    valid = (
        (cluster_codes >= 0)
        & (label_codes >= 0)
        & valid_cluster[cluster_lookup]
        & valid_label[label_lookup]
    )
    if valid.sum() == 0:
        return pd.DataFrame(columns=["cluster_code", "label_code", "count", "cluster", "label"])

    try:
        import cudf

        gdf = cudf.DataFrame(
            {
                "cluster_code": cluster_codes[valid].astype(np.int32, copy=False),
                "label_code": label_codes[valid].astype(np.int32, copy=False),
            }
        )
        counts = gdf.groupby(["cluster_code", "label_code"]).size().reset_index(name="count").to_pandas()
    except Exception as error:
        print(f"cudf contingency failed for {cluster_key} vs {label_col}; falling back to pandas. Error: {error}")
        counts = (
            pd.DataFrame(
                {
                    "cluster_code": cluster_codes[valid].astype(np.int32, copy=False),
                    "label_code": label_codes[valid].astype(np.int32, copy=False),
                }
            )
            .groupby(["cluster_code", "label_code"], observed=True)
            .size()
            .reset_index(name="count")
        )

    counts["cluster_code"] = counts["cluster_code"].astype(np.int32)
    counts["label_code"] = counts["label_code"].astype(np.int32)
    counts["count"] = counts["count"].astype(np.int64)
    counts["cluster"] = [cluster_categories[idx] for idx in counts["cluster_code"].to_numpy()]
    counts["label"] = [label_categories[idx] for idx in counts["label_code"].to_numpy()]
    return counts


def comb2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return values * (values - 1.0) / 2.0


def entropy_from_counts(counts: np.ndarray, total: float) -> float:
    counts = counts[counts > 0].astype(np.float64, copy=False)
    if total <= 0 or counts.size == 0:
        return 0.0
    probs = counts / total
    return float(-(probs * np.log(probs)).sum())


def metrics_from_contingency(
    counts_long: pd.DataFrame,
    cluster_key: str,
    label_col: str,
    embedding_name: str,
) -> dict | None:
    if counts_long.empty:
        return None

    cluster_codes = counts_long["cluster_code"].to_numpy(dtype=np.int64)
    label_codes = counts_long["label_code"].to_numpy(dtype=np.int64)
    counts = counts_long["count"].to_numpy(dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return None

    row_sums = np.bincount(cluster_codes, weights=counts)
    col_sums = np.bincount(label_codes, weights=counts)
    row_sums_obs = row_sums[row_sums > 0]
    col_sums_obs = col_sums[col_sums > 0]
    if row_sums_obs.size < 2 or col_sums_obs.size < 2:
        return None

    sum_comb_cells = float(comb2(counts).sum())
    sum_comb_rows = float(comb2(row_sums_obs).sum())
    sum_comb_cols = float(comb2(col_sums_obs).sum())
    comb_total = float(total * (total - 1.0) / 2.0)
    if comb_total == 0:
        ari = 1.0
    else:
        expected = sum_comb_rows * sum_comb_cols / comb_total
        max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
        denom = max_index - expected
        ari = 1.0 if denom == 0 else (sum_comb_cells - expected) / denom

    denom = row_sums[cluster_codes] * col_sums[label_codes]
    valid = (counts > 0) & (denom > 0)
    mi = float(((counts[valid] / total) * np.log((counts[valid] * total) / denom[valid])).sum())
    h_cluster = entropy_from_counts(row_sums_obs, total)
    h_label = entropy_from_counts(col_sums_obs, total)

    nmi_denom = (h_label + h_cluster) / 2.0
    nmi = 1.0 if nmi_denom == 0 else mi / nmi_denom
    homogeneity = 1.0 if h_label == 0 else mi / h_label
    completeness = 1.0 if h_cluster == 0 else mi / h_cluster
    v_measure = 0.0 if homogeneity + completeness == 0 else 2.0 * homogeneity * completeness / (homogeneity + completeness)

    return {
        "embedding": embedding_name,
        "cluster_key": cluster_key,
        "label_col": label_col,
        "n_cells": int(total),
        "n_clusters": int(row_sums_obs.size),
        "n_labels": int(col_sums_obs.size),
        "ari": float(ari),
        "nmi": float(nmi),
        # adjusted_mutual_info_score is intentionally skipped; it is expensive at this scale.
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
    }


def run_rapids_clustering(
    adata_all: ad.AnnData,
    rep_key: str,
    cluster_key: str,
    umap_key: str,
    n_neighbors: int,
    resolution: float,
    args: argparse.Namespace,
) -> None:
    import rapids_singlecell as rsc

    print(f"\nRAPIDS clustering for {rep_key} -> {cluster_key}")
    print(f"neighbors={n_neighbors}, resolution={resolution}, algorithm={args.algorithm}, metric={args.metric}")
    emb = np.asarray(adata_all.obsm[rep_key], dtype=np.float32)
    cluster_adata = ad.AnnData(
        X=np.zeros((adata_all.n_obs, 1), dtype=np.float32),
        obs=pd.DataFrame(index=adata_all.obs_names.copy()),
    )
    cluster_adata.obsm["X_emb"] = emb

    rsc.get.anndata_to_GPU(cluster_adata, convert_all=True)
    rsc.pp.neighbors(
        cluster_adata,
        n_neighbors=n_neighbors,
        use_rep="X_emb",
        metric=args.metric,
        algorithm=args.algorithm,
        key_added=f"{cluster_key}_neighbors",
        random_state=args.random_seed,
    )
    rsc.tl.leiden(
        cluster_adata,
        resolution=resolution,
        key_added=cluster_key,
        neighbors_key=f"{cluster_key}_neighbors",
        random_state=args.random_seed,
    )
    if args.run_umap:
        rsc.tl.umap(
            cluster_adata,
            neighbors_key=f"{cluster_key}_neighbors",
            key_added=umap_key,
            random_state=args.random_seed,
        )
    rsc.get.anndata_to_CPU(cluster_adata, convert_all=True)

    adata_all.obs[cluster_key] = cluster_adata.obs[cluster_key].astype(str).to_numpy()
    if args.run_umap and umap_key in cluster_adata.obsm:
        adata_all.obsm[umap_key] = np.asarray(cluster_adata.obsm[umap_key], dtype=np.float32)

    n_clusters = adata_all.obs[cluster_key].nunique()
    print(f"{cluster_key}: {n_clusters} clusters")

    del cluster_adata, emb
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def save_crosstab_and_summary(counts_long: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if counts_long.empty:
        return

    long_out = counts_long[["cluster", "label", "count"]].sort_values(["cluster", "label"])
    long_out.to_csv(out_dir / f"{prefix}_count_long.csv", index=False)

    counts = counts_long.pivot_table(
        index="cluster",
        columns="label",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    row_frac = counts.div(counts.sum(axis=1), axis=0)
    col_frac = counts.div(counts.sum(axis=0), axis=1)

    counts.to_csv(out_dir / f"{prefix}_count.csv")
    row_frac.to_csv(out_dir / f"{prefix}_cluster_fraction.csv")
    col_frac.to_csv(out_dir / f"{prefix}_label_fraction.csv")

    dominant = pd.DataFrame(
        {
            "cluster": counts.index.astype(str),
            "n_cells": counts.sum(axis=1).to_numpy(),
            "dominant_label": counts.idxmax(axis=1).astype(str).to_numpy(),
            "dominant_label_count": counts.max(axis=1).to_numpy(),
            "dominant_label_fraction": row_frac.max(axis=1).to_numpy(),
        }
    ).sort_values(["dominant_label_fraction", "n_cells"], ascending=False)
    dominant.to_csv(out_dir / f"{prefix}_dominant_label_by_cluster.csv", index=False)


def save_metrics_and_tables(
    adata_all: ad.AnnData,
    cluster_key: str,
    label_cols: list[str],
    embedding_name: str,
    out_dir: Path,
) -> pd.DataFrame:
    label_cols = available_columns(adata_all.obs, label_cols)
    print(f"{embedding_name} labels for comparison: {label_cols}")
    metric_rows = []
    for label_col in label_cols:
        counts_long = contingency_counts(adata_all.obs, cluster_key, label_col)
        metrics = metrics_from_contingency(
            counts_long,
            cluster_key=cluster_key,
            label_col=label_col,
            embedding_name=embedding_name,
        )
        if metrics is None:
            continue
        metric_rows.append(metrics)
        save_crosstab_and_summary(
            counts_long,
            out_dir=out_dir / embedding_name,
            prefix=f"{embedding_name}_{cluster_key}_vs_{label_col}",
        )
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics.to_csv(out_dir / embedding_name / f"{embedding_name}_metrics.csv", index=False)
    return metrics


def downsample_for_plot(n_obs: int, max_cells: int, seed: int) -> np.ndarray:
    if n_obs <= max_cells:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=max_cells, replace=False))


def save_umap_plot(
    adata_all: ad.AnnData,
    umap_key: str,
    color_col: str,
    out_png: Path,
    max_cells: int,
    seed: int,
) -> None:
    if umap_key not in adata_all.obsm or color_col not in adata_all.obs:
        return
    idx = downsample_for_plot(adata_all.n_obs, max_cells, seed)
    coords = np.asarray(adata_all.obsm[umap_key][idx], dtype=np.float32)
    labels = to_label_series(adata_all.obs.iloc[idx], color_col)
    categories = labels.value_counts().index.tolist()

    cmap = plt.colormaps.get_cmap("tab20")
    color_map = {cat: cmap(i % 20) for i, cat in enumerate(categories)}

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6), dpi=180)
    for cat in categories:
        mask = labels.eq(cat).to_numpy()
        plt.scatter(coords[mask, 0], coords[mask, 1], s=1, alpha=0.5, color=color_map[cat], rasterized=True)
    plt.title(f"{umap_key} colored by {color_col}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    if len(categories) <= 25:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[cat], markersize=5, label=cat)
            for cat in categories
        ]
        plt.legend(handles=handles, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata_all = read_and_concat(args)
    concat_path = args.output_dir / "merfish_mouseBrain_concat_embeddings.h5ad"

    if args.dry_run:
        if not args.no_write_concat:
            adata_all.write_h5ad(concat_path)
            print(f"wrote concat h5ad: {concat_path}")
        print("dry-run complete; clustering skipped")
        return

    cell_n_neighbors = args.cell_n_neighbors if args.cell_n_neighbors is not None else args.n_neighbors
    cell_resolution = args.cell_resolution if args.cell_resolution is not None else args.resolution
    niche_n_neighbors = args.niche_n_neighbors if args.niche_n_neighbors is not None else args.n_neighbors
    niche_resolution = args.niche_resolution if args.niche_resolution is not None else args.resolution

    run_rapids_clustering(
        adata_all,
        args.cell_key,
        "cell_leiden",
        "X_umap_cell",
        n_neighbors=cell_n_neighbors,
        resolution=cell_resolution,
        args=args,
    )
    run_rapids_clustering(
        adata_all,
        args.niche_key,
        "niche_leiden",
        "X_umap_niche",
        n_neighbors=niche_n_neighbors,
        resolution=niche_resolution,
        args=args,
    )

    all_metrics = []
    all_metrics.append(
        save_metrics_and_tables(
            adata_all,
            cluster_key="cell_leiden",
            label_cols=args.cell_label_cols,
            embedding_name="cell_embedding",
            out_dir=args.output_dir,
        )
    )
    all_metrics.append(
        save_metrics_and_tables(
            adata_all,
            cluster_key="niche_leiden",
            label_cols=args.niche_label_cols,
            embedding_name="niche_embedding",
            out_dir=args.output_dir,
        )
    )
    metrics = pd.concat([df for df in all_metrics if not df.empty], ignore_index=True) if any(
        not df.empty for df in all_metrics
    ) else pd.DataFrame()
    if not metrics.empty:
        metrics.to_csv(args.output_dir / "clustering_metrics_summary.csv", index=False)
        print(metrics)

    if args.save_umap_plots and not args.run_umap:
        print("Warning: --save-umap-plots requires --run-umap; skipping UMAP plots.")

    if args.run_umap and args.save_umap_plots:
        plot_specs = [
            ("X_umap_cell", "cell_leiden"),
            ("X_umap_cell", "cell_type"),
            ("X_umap_cell", "subclass_transfer"),
            ("X_umap_niche", "niche_leiden"),
            ("X_umap_niche", "major_brain_region"),
            ("X_umap_niche", "ccf_region_name"),
        ]
        for umap_key, color_col in plot_specs:
            save_umap_plot(
                adata_all,
                umap_key=umap_key,
                color_col=color_col,
                out_png=args.output_dir / "figures" / f"{umap_key}_by_{color_col}.png",
                max_cells=args.plot_max_cells,
                seed=args.random_seed,
            )

    if not args.no_write_concat:
        adata_all.write_h5ad(concat_path)
        print(f"wrote clustered concat h5ad: {concat_path}")

    print(f"analysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
