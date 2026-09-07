"""
三物种 embedding 分析脚本

功能:
1) niche embedding 自聚类，检查跨物种/分层混合情况
2) cell embedding 自聚类，识别物种特异的 cluster（unique 候选）
3) 基于 niche embedding 的跨物种脑区一致性分析（region centroid 匹配）

示例:
python 04_embedding_analysis.py \
  --input-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/inference_adatas \
  --output-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/analysis_light \
  --species-use marmoset macaque mouse
  
STEREOTRACK_ANALYSIS_THREADS=1 python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/04_three_species/04_embedding_analysis.py \
  --input-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/inference_adatas \
  --output-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/analysis_rapids_singlecell \
  --species-use marmoset macaque mouse \
  --layer-col layer \
  --region-col region \
  --celltype-col cell_type \
  --cluster-method leiden \
  --compute-backend rapids \
  --max-cells-per-file 1000 \
  --max-cells-per-species 0 \
  --max-total-cells 0

bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/04_three_species/04_run_embedding_analysis_rapids.sh
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

# HPC 节点上 OpenBLAS/OMP 线程数过高时，PCA/KMeans 可能直接 segfault。
# 必须在 numpy/sklearn 导入前设置；用户也可以在 shell 里预先覆盖这些变量。
_analysis_threads = os.environ.get("STEREOTRACK_ANALYSIS_THREADS", "1")
for _thread_env in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[_thread_env] = _analysis_threads

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


LAYER_CANDIDATES = [
    "layer",
    "layers",
    "cortex_layer",
    "lamina",
    "laminar",
]

REGION_CANDIDATES = [
    "region",
    "main_region",
    "cortex_region",
    "area_name",
    "brain_region",
]

CELLTYPE_CANDIDATES = [
    "cell_type",
    "celltype",
    "subclass",
    "class",
    "cluster",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跨物种 embedding 分析")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/inference_adatas"),
        help="推理输出目录（包含多个 h5ad）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/analysis_light"),
        help="分析结果输出目录",
    )
    parser.add_argument(
        "--species-use",
        nargs="*",
        default=None,
        help="可选：仅分析指定物种，如 marmoset macaque mouse",
    )
    parser.add_argument("--cell-key", type=str, default="cell_embedding", help="cell embedding 键名")
    parser.add_argument("--niche-key", type=str, default="niche_embedding", help="niche embedding 键名")
    parser.add_argument("--species-col", type=str, default="species", help="obs 里物种列名")
    parser.add_argument("--layer-col", type=str, default=None, help="可选：手动指定 layer 列名")
    parser.add_argument("--region-col", type=str, default=None, help="可选：手动指定 region 列名")
    parser.add_argument("--celltype-col", type=str, default=None, help="可选：手动指定 cell type 列名")
    parser.add_argument("--max-cells-per-file", type=int, default=5000, help="每个 h5ad 最多抽样多少细胞；<=0 表示不按文件限制")
    parser.add_argument("--max-cells-per-species", type=int, default=100000, help="每个物种最多抽样多少细胞；<=0 表示不限制")
    parser.add_argument("--max-total-cells", type=int, default=300000, help="总共最多抽样多少细胞；<=0 表示不限制")
    parser.add_argument(
        "--sampling-mode",
        choices=["proportional", "equal_files"],
        default="proportional",
        help="设置 species/global quota 如何分配到 slice。proportional 按 slice 细胞数分配；equal_files 每个 slice 尽量相同。",
    )
    parser.add_argument("--neighbors", type=int, default=30, help="聚类图邻居数")
    parser.add_argument("--resolution", type=float, default=1.0, help="leiden 分辨率")
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "auto", "leiden"],
        default="leiden",
        help="leiden 会做 neighbors/UMAP/Leiden；kmeans 是低依赖兜底路线。",
    )
    parser.add_argument(
        "--compute-backend",
        choices=["auto", "cpu", "rapids"],
        default="auto",
        help="leiden + rapids 会使用 rapids_singlecell；kmeans + rapids 使用 cuML PCA/KMeans。",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="KMeans cluster 数；默认按细胞数自动估计。",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--unique-threshold", type=float, default=0.80, help="判定物种 unique cluster 的占比阈值")
    parser.add_argument("--min-cluster-size", type=int, default=200, help="判定 unique cluster 的最小细胞数")
    parser.add_argument("--min-region-cells", type=int, default=100, help="region centroid 至少细胞数")
    return parser.parse_args()


def infer_species_from_stem(stem: str) -> str:
    prefix = stem.split("_", 1)[0].lower()
    if prefix in {"marmoset", "macaque", "mouse", "human"}:
        return prefix
    return "unknown"


def extract_obs_series(adata: ad.AnnData, preferred: str | None, candidates: Iterable[str], fallback: str) -> pd.Series:
    if preferred is not None and preferred in adata.obs.columns:
        return adata.obs[preferred].astype(str)
    for column in candidates:
        if column in adata.obs.columns:
            return adata.obs[column].astype(str)
    return pd.Series([fallback] * adata.n_obs, index=adata.obs_names, dtype=str)


def allocate_quota(caps: np.ndarray, quota: int, weights: np.ndarray | None = None) -> np.ndarray:
    caps = np.asarray(caps, dtype=np.int64)
    if caps.size == 0:
        return caps
    total_cap = int(caps.sum())
    if quota <= 0 or quota >= total_cap:
        return caps.copy()

    if weights is None:
        weights = caps.astype(np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(caps > 0, weights, 0.0)
    if float(weights.sum()) <= 0:
        weights = (caps > 0).astype(np.float64)

    raw = quota * weights / weights.sum()
    assigned = np.floor(raw).astype(np.int64)
    assigned = np.minimum(assigned, caps)

    remainder = int(quota - assigned.sum())
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)

    while remainder > 0:
        changed = False
        for idx in order:
            if remainder <= 0:
                break
            if assigned[idx] < caps[idx]:
                assigned[idx] += 1
                remainder -= 1
                changed = True
        if not changed:
            break
    return assigned


def collect_file_plan(args: argparse.Namespace) -> list[dict]:
    files = sorted(args.input_dir.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No h5ad found in {args.input_dir}")

    species_use = {s.lower() for s in args.species_use} if args.species_use else None
    plan = []

    for file_path in files:
        adata = ad.read_h5ad(file_path, backed="r")
        if args.cell_key not in adata.obsm or args.niche_key not in adata.obsm:
            print(f"[skip] {file_path.name}: missing {args.cell_key} or {args.niche_key}")
            adata.file.close()
            continue

        if args.species_col in adata.obs.columns:
            species = str(adata.obs[args.species_col].iloc[0]).lower()
        else:
            species = infer_species_from_stem(file_path.stem)
        n_obs = int(adata.n_obs)
        adata.file.close()

        if species_use is not None and species not in species_use:
            continue

        cap = n_obs if args.max_cells_per_file <= 0 else min(n_obs, args.max_cells_per_file)
        plan.append(
            {
                "file_path": file_path,
                "species": species,
                "n_obs": n_obs,
                "cap": int(cap),
                "keep_n": int(cap),
            }
        )

    if not plan:
        raise RuntimeError("No usable h5ad after filtering; check --species-use or embedding keys")

    if args.max_cells_per_species > 0:
        for species in sorted({row["species"] for row in plan}):
            idx = [i for i, row in enumerate(plan) if row["species"] == species]
            caps = np.array([plan[i]["cap"] for i in idx], dtype=np.int64)
            weights = (
                np.array([plan[i]["n_obs"] for i in idx], dtype=np.float64)
                if args.sampling_mode == "proportional"
                else np.ones(len(idx), dtype=np.float64)
            )
            assigned = allocate_quota(caps, args.max_cells_per_species, weights=weights)
            for i, keep_n in zip(idx, assigned):
                plan[i]["keep_n"] = int(keep_n)

    if args.max_total_cells > 0:
        current_total = sum(row["keep_n"] for row in plan)
        if current_total > args.max_total_cells:
            caps = np.array([row["keep_n"] for row in plan], dtype=np.int64)
            weights = caps.astype(np.float64)
            assigned = allocate_quota(caps, args.max_total_cells, weights=weights)
            for row, keep_n in zip(plan, assigned):
                row["keep_n"] = int(keep_n)

    summary: dict[str, dict[str, int]] = {}
    for row in plan:
        info = summary.setdefault(row["species"], {"files": 0, "cells": 0, "keep": 0})
        info["files"] += 1
        info["cells"] += int(row["n_obs"])
        info["keep"] += int(row["keep_n"])
    print("[sample-plan]")
    for species, info in sorted(summary.items()):
        print(f"  {species}: files={info['files']}, cells={info['cells']}, keep={info['keep']}")
    return plan


def read_embedding_pool(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.random_seed)
    plan = collect_file_plan(args)

    obs_parts: list[pd.DataFrame] = []
    cell_emb_parts: list[np.ndarray] = []
    niche_emb_parts: list[np.ndarray] = []

    for row in plan:
        file_path = row["file_path"]
        keep_n = int(row["keep_n"])
        if keep_n <= 0:
            print(f"[skip] {file_path.name}: sampling quota is 0")
            continue

        adata = ad.read_h5ad(file_path)
        if args.cell_key not in adata.obsm or args.niche_key not in adata.obsm:
            print(f"[skip] {file_path.name}: missing {args.cell_key} or {args.niche_key}")
            continue

        species_series = extract_obs_series(adata, args.species_col, [args.species_col], infer_species_from_stem(file_path.stem))

        if adata.n_obs > keep_n:
            selected = np.sort(rng.choice(adata.n_obs, size=keep_n, replace=False))
        else:
            selected = np.arange(adata.n_obs)

        layer_series = extract_obs_series(adata, args.layer_col, LAYER_CANDIDATES, "NA")
        region_series = extract_obs_series(adata, args.region_col, REGION_CANDIDATES, "NA")
        celltype_series = extract_obs_series(adata, args.celltype_col, CELLTYPE_CANDIDATES, "NA")

        obs_df = pd.DataFrame(
            {
                "species": species_series.iloc[selected].to_numpy(),
                "batch": file_path.stem,
                "layer_label": layer_series.iloc[selected].to_numpy(),
                "region_label": region_series.iloc[selected].to_numpy(),
                "celltype_label": celltype_series.iloc[selected].to_numpy(),
            }
        )

        obs_parts.append(obs_df)
        cell_emb_parts.append(np.asarray(adata.obsm[args.cell_key])[selected].astype(np.float32))
        niche_emb_parts.append(np.asarray(adata.obsm[args.niche_key])[selected].astype(np.float32))
        print(f"[ok] {file_path.name}: keep {len(selected)}/{adata.n_obs} cells")

    if not obs_parts:
        raise RuntimeError("No usable h5ad after filtering; check --species-use or embedding keys")

    obs_all = pd.concat(obs_parts, axis=0, ignore_index=True)
    obs_all.index = obs_all.index.astype(str)
    cell_emb_all = np.concatenate(cell_emb_parts, axis=0)
    niche_emb_all = np.concatenate(niche_emb_parts, axis=0)
    return obs_all, cell_emb_all, niche_emb_all


def blas_thread_context():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def run_kmeans_cluster(
    emb: np.ndarray,
    obs: pd.DataFrame,
    cluster_key: str,
    seed: int,
    n_clusters: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if n_clusters is None:
        n_clusters = max(8, min(80, emb.shape[0] // 3000 + 8))
    n_clusters = max(2, min(int(n_clusters), emb.shape[0]))

    with blas_thread_context():
        pca2 = PCA(n_components=2, random_state=seed).fit_transform(emb)
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(8192, emb.shape[0]),
            n_init=10,
        )
        labels = km.fit_predict(emb)

    obs_out = obs.copy()
    obs_out[cluster_key] = pd.Categorical(labels.astype(str))
    return obs_out, pca2.astype(np.float32)


def _as_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    try:
        import cupy as cp

        return cp.asnumpy(array)
    except Exception:
        pass
    if hasattr(array, "to_numpy"):
        return array.to_numpy()
    return np.asarray(array)


def run_rapids_kmeans_cluster(
    emb: np.ndarray,
    obs: pd.DataFrame,
    cluster_key: str,
    seed: int,
    n_clusters: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        import cupy as cp
        from cuml.cluster import KMeans as cuKMeans
        from cuml.decomposition import PCA as cuPCA
    except Exception as error:
        raise ImportError(f"RAPIDS/cuML is not available: {error}") from error

    if n_clusters is None:
        n_clusters = max(8, min(80, emb.shape[0] // 3000 + 8))
    n_clusters = max(2, min(int(n_clusters), emb.shape[0]))

    emb_gpu = cp.asarray(np.ascontiguousarray(emb, dtype=np.float32))
    pca_model = cuPCA(n_components=2, random_state=seed)
    pca2_gpu = pca_model.fit_transform(emb_gpu)

    try:
        km = cuKMeans(n_clusters=n_clusters, random_state=seed, max_iter=300, n_init=10)
    except TypeError:
        km = cuKMeans(n_clusters=n_clusters, random_state=seed, max_iter=300)
    labels_gpu = km.fit_predict(emb_gpu)

    pca2 = _as_numpy(pca2_gpu).astype(np.float32)
    labels = _as_numpy(labels_gpu).astype(str)
    cp.get_default_memory_pool().free_all_blocks()

    obs_out = obs.copy()
    obs_out[cluster_key] = pd.Categorical(labels)
    return obs_out, pca2


def run_embedding_cluster(
    emb: np.ndarray,
    obs: pd.DataFrame,
    cluster_key: str,
    seed: int,
    n_clusters: int | None,
    compute_backend: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if compute_backend in {"auto", "rapids"}:
        try:
            print(f"[info] {cluster_key}: using RAPIDS cuML PCA + KMeans")
            return run_rapids_kmeans_cluster(emb, obs, cluster_key, seed, n_clusters)
        except Exception as error:
            if compute_backend == "rapids":
                raise
            print(f"[warn] RAPIDS backend unavailable ({error}); fallback to CPU PCA + KMeans")

    print(f"[info] {cluster_key}: using CPU PCA + KMeans")
    return run_kmeans_cluster(emb, obs, cluster_key, seed, n_clusters)


def run_rapids_singlecell_graph_cluster(
    emb: np.ndarray,
    obs: pd.DataFrame,
    cluster_key: str,
    neighbors: int,
    resolution: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        import cupy as cp
        import rapids_singlecell as rsc
    except Exception as error:
        raise ImportError(f"rapids_singlecell is not available: {error}") from error

    obs_for_adata = obs.copy()
    obs_for_adata.index = obs_for_adata.index.astype(str)
    adata = ad.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32), obs=obs_for_adata)
    adata.obsm["X_emb"] = np.ascontiguousarray(emb, dtype=np.float32)

    print(f"[info] {cluster_key}: using rapids_singlecell neighbors + UMAP + Leiden")
    rsc.pp.neighbors(
        adata,
        use_rep="X_emb",
        n_neighbors=neighbors,
        metric="cosine",
        random_state=seed,
    )
    rsc.tl.umap(adata, random_state=seed)
    rsc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=cluster_key,
        random_state=seed,
    )

    umap = _as_numpy(adata.obsm["X_umap"]).astype(np.float32)
    obs_out = adata.obs.copy()
    cp.get_default_memory_pool().free_all_blocks()
    return obs_out, umap


def run_scanpy_graph_cluster(
    emb: np.ndarray,
    obs: pd.DataFrame,
    cluster_key: str,
    neighbors: int,
    resolution: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    import scanpy as sc

    obs_for_adata = obs.copy()
    obs_for_adata.index = obs_for_adata.index.astype(str)
    adata = ad.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32), obs=obs_for_adata)
    adata.obsm["X_emb"] = emb

    with blas_thread_context():
        sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=neighbors, metric="cosine")
        sc.tl.umap(adata, random_state=seed)
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key, random_state=seed)
    umap = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    obs_out = adata.obs.copy()
    return obs_out, umap


def run_graph_cluster(
    emb: np.ndarray,
    obs: pd.DataFrame,
    cluster_key: str,
    neighbors: int,
    resolution: float,
    seed: int,
    cluster_method: str,
    n_clusters: int | None,
    compute_backend: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if cluster_method == "kmeans":
        return run_embedding_cluster(emb, obs, cluster_key, seed, n_clusters, compute_backend)

    if compute_backend in {"auto", "rapids"}:
        try:
            return run_rapids_singlecell_graph_cluster(
                emb,
                obs,
                cluster_key=cluster_key,
                neighbors=neighbors,
                resolution=resolution,
                seed=seed,
            )
        except Exception as error:
            if compute_backend == "rapids":
                raise
            print(f"[warn] rapids_singlecell failed ({error}); fallback to CPU")

    try:
        print(f"[info] {cluster_key}: using scanpy neighbors + UMAP + Leiden")
        return run_scanpy_graph_cluster(
            emb,
            obs,
            cluster_key=cluster_key,
            neighbors=neighbors,
            resolution=resolution,
            seed=seed,
        )
    except Exception as error:
        if cluster_method == "leiden":
            raise
        print(f"[warn] leiden/umap failed ({error}), fallback to PCA + KMeans")
        return run_embedding_cluster(emb, obs, cluster_key, seed, n_clusters, compute_backend)


def save_umap_plot(umap: np.ndarray, labels: pd.Series, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labels = labels.fillna("NA").astype(str)
    categories = labels.value_counts().index.tolist()
    colors = plt.cm.get_cmap("tab20", max(20, len(categories)))
    color_map = {cat: colors(i % 20) for i, cat in enumerate(categories)}

    plt.figure(figsize=(7, 6), dpi=180)
    for cat in categories:
        mask = labels.eq(cat).to_numpy()
        plt.scatter(umap[mask, 0], umap[mask, 1], s=2, alpha=0.65, color=color_map[cat], label=cat)
    if len(categories) <= 20:
        plt.legend(markerscale=4, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_cluster_tables(obs: pd.DataFrame, cluster_key: str, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    species_tab = pd.crosstab(obs[cluster_key], obs["species"], normalize="index")
    species_cnt = pd.crosstab(obs[cluster_key], obs["species"])
    species_tab.to_csv(out_dir / f"{prefix}_cluster_by_species_ratio.csv")
    species_cnt.to_csv(out_dir / f"{prefix}_cluster_by_species_count.csv")

    if "layer_label" in obs.columns and obs["layer_label"].nunique() > 1:
        layer_tab = pd.crosstab(obs[cluster_key], obs["layer_label"], normalize="index")
        layer_cnt = pd.crosstab(obs[cluster_key], obs["layer_label"])
        layer_tab.to_csv(out_dir / f"{prefix}_cluster_by_layer_ratio.csv")
        layer_cnt.to_csv(out_dir / f"{prefix}_cluster_by_layer_count.csv")

    if "celltype_label" in obs.columns and obs["celltype_label"].nunique() > 1:
        ct_tab = pd.crosstab(obs[cluster_key], obs["celltype_label"], normalize="index")
        ct_cnt = pd.crosstab(obs[cluster_key], obs["celltype_label"])
        ct_tab.to_csv(out_dir / f"{prefix}_cluster_by_celltype_ratio.csv")
        ct_cnt.to_csv(out_dir / f"{prefix}_cluster_by_celltype_count.csv")


def find_species_unique_clusters(
    obs: pd.DataFrame,
    cluster_key: str,
    unique_threshold: float,
    min_cluster_size: int,
) -> pd.DataFrame:
    count_tab = pd.crosstab(obs[cluster_key], obs["species"])
    total = count_tab.sum(axis=1)
    frac_tab = count_tab.div(total, axis=0)

    top_species = frac_tab.idxmax(axis=1)
    top_frac = frac_tab.max(axis=1)

    unique_mask = (top_frac >= unique_threshold) & (total >= min_cluster_size)
    out = pd.DataFrame(
        {
            "cluster": count_tab.index.astype(str),
            "n_cells": total.values,
            "dominant_species": top_species.values,
            "dominant_fraction": top_frac.values,
        }
    )
    out = out.loc[unique_mask.values].sort_values(["dominant_fraction", "n_cells"], ascending=False)
    return out


def region_consistency(obs: pd.DataFrame, niche_emb: np.ndarray, min_region_cells: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_mask = obs["region_label"].astype(str).ne("NA")
    obs_valid = obs.loc[valid_mask].reset_index(drop=True)
    emb_valid = niche_emb[valid_mask.to_numpy()]

    if obs_valid.empty:
        print("[warn] no valid region_label found; skip region consistency")
        return

    grouped = obs_valid.groupby(["species", "region_label"], observed=True)
    centroid_rows = []
    for (species, region), idx in grouped.indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < min_region_cells:
            continue
        centroid = emb_valid[idx].mean(axis=0)
        row = {
            "species": species,
            "region": region,
            "n_cells": int(len(idx)),
        }
        for dim_idx, value in enumerate(centroid):
            row[f"d{dim_idx}"] = float(value)
        centroid_rows.append(row)

    centroid_df = pd.DataFrame(centroid_rows)
    if centroid_df.empty:
        print("[warn] no region has enough cells; skip region consistency")
        return

    centroid_df.to_csv(out_dir / "region_centroids.csv", index=False)

    embed_cols = [col for col in centroid_df.columns if col.startswith("d")]
    pair_rows = []
    species_list = sorted(centroid_df["species"].unique().tolist())

    for species_a in species_list:
        for species_b in species_list:
            if species_a == species_b:
                continue
            a_df = centroid_df.loc[centroid_df["species"].eq(species_a)].reset_index(drop=True)
            b_df = centroid_df.loc[centroid_df["species"].eq(species_b)].reset_index(drop=True)
            if a_df.empty or b_df.empty:
                continue
            dist = cosine_distances(a_df[embed_cols].to_numpy(), b_df[embed_cols].to_numpy())
            nn_idx = dist.argmin(axis=1)
            nn_dist = dist.min(axis=1)

            for i in range(len(a_df)):
                pair_rows.append(
                    {
                        "species_from": species_a,
                        "region_from": a_df.loc[i, "region"],
                        "n_cells_from": int(a_df.loc[i, "n_cells"]),
                        "species_to": species_b,
                        "region_to": b_df.loc[nn_idx[i], "region"],
                        "n_cells_to": int(b_df.loc[nn_idx[i], "n_cells"]),
                        "cosine_distance": float(nn_dist[i]),
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    pair_df = pair_df.sort_values("cosine_distance", ascending=True)
    pair_df.to_csv(out_dir / "region_cross_species_nn.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs, cell_emb, niche_emb = read_embedding_pool(args)
    obs.to_csv(args.output_dir / "pooled_obs_metadata.csv", index=False)
    print(f"pooled cells: {len(obs)}")
    print(f"species: {sorted(obs['species'].astype(str).unique().tolist())}")

    # 1) niche embedding clustering
    niche_obs, niche_umap = run_graph_cluster(
        niche_emb,
        obs,
        cluster_key="niche_cluster",
        neighbors=args.neighbors,
        resolution=args.resolution,
        seed=args.random_seed,
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
        compute_backend=args.compute_backend,
    )
    niche_obs.to_csv(args.output_dir / "niche_cluster_obs.csv", index=False)
    np.save(args.output_dir / "niche_umap.npy", niche_umap)
    save_cluster_tables(niche_obs, "niche_cluster", args.output_dir, "niche")
    save_umap_plot(niche_umap, niche_obs["species"], args.output_dir / "niche_umap_by_species.png", "Niche UMAP by Species")
    save_umap_plot(niche_umap, niche_obs["niche_cluster"].astype(str), args.output_dir / "niche_umap_by_cluster.png", "Niche UMAP by Cluster")
    if niche_obs["layer_label"].astype(str).nunique() > 1:
        save_umap_plot(niche_umap, niche_obs["layer_label"].astype(str), args.output_dir / "niche_umap_by_layer.png", "Niche UMAP by Layer")
    if niche_obs["region_label"].astype(str).nunique() > 1:
        save_umap_plot(niche_umap, niche_obs["region_label"].astype(str), args.output_dir / "niche_umap_by_region.png", "Niche UMAP by Region")

    # 2) cell embedding clustering + unique cluster detection
    cell_obs, cell_umap = run_graph_cluster(
        cell_emb,
        obs,
        cluster_key="cell_cluster",
        neighbors=args.neighbors,
        resolution=args.resolution,
        seed=args.random_seed,
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
        compute_backend=args.compute_backend,
    )
    cell_obs.to_csv(args.output_dir / "cell_cluster_obs.csv", index=False)
    np.save(args.output_dir / "cell_umap.npy", cell_umap)
    save_cluster_tables(cell_obs, "cell_cluster", args.output_dir, "cell")
    save_umap_plot(cell_umap, cell_obs["species"], args.output_dir / "cell_umap_by_species.png", "Cell UMAP by Species")
    save_umap_plot(cell_umap, cell_obs["cell_cluster"].astype(str), args.output_dir / "cell_umap_by_cluster.png", "Cell UMAP by Cluster")
    if cell_obs["celltype_label"].astype(str).nunique() > 1:
        save_umap_plot(cell_umap, cell_obs["celltype_label"].astype(str), args.output_dir / "cell_umap_by_celltype.png", "Cell UMAP by Cell Type")
    if cell_obs["layer_label"].astype(str).nunique() > 1:
        save_umap_plot(cell_umap, cell_obs["layer_label"].astype(str), args.output_dir / "cell_umap_by_layer.png", "Cell UMAP by Layer")

    unique_df = find_species_unique_clusters(
        cell_obs,
        cluster_key="cell_cluster",
        unique_threshold=args.unique_threshold,
        min_cluster_size=args.min_cluster_size,
    )
    unique_df.to_csv(args.output_dir / "cell_species_unique_clusters.csv", index=False)

    # 3) niche embedding region consistency across species
    region_consistency(cell_obs, niche_emb, args.min_region_cells, args.output_dir)

    print(f"analysis done: {args.output_dir}")


if __name__ == "__main__":
    main()
