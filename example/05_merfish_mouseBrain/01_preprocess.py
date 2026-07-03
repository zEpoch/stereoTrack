from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import scipy.sparse as sp
import torch
import yaml
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stereotrack import get_feature_sparse, get_spatial_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MERFISH mouse brain h5ad files for StereoTrack MAE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/05_config_merfish_mouseBrain.yaml"),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild the cache even if meta.yaml already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect files and shared genes, then exit.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def mem_usage() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024**3


def list_h5ad_files(data_path: Path) -> list[Path]:
    return sorted(path for path in data_path.glob("*.h5ad") if path.is_file())


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "yes", "y"])


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        if not name or name.lower() == "nan":
            name = "unknown"
        count = seen.get(name, 0)
        out.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return out


def prepared_gene_names(var: pd.DataFrame, cfg: dict) -> pd.Index:
    preprocess_cfg = cfg.get("preprocess", {})
    gene_name_column = preprocess_cfg.get("gene_name_column")
    if gene_name_column and gene_name_column in var.columns:
        names = var[gene_name_column].astype("string").fillna("").astype(str).tolist()
    else:
        names = var.index.astype(str).tolist()
    return pd.Index(make_unique(names))


def selected_gene_names_from_backed(adata: ad.AnnData, cfg: dict) -> pd.Index:
    names = prepared_gene_names(adata.var, cfg)
    preprocess_cfg = cfg.get("preprocess", {})
    if preprocess_cfg.get("filter_feature_is_filtered", False) and "feature_is_filtered" in adata.var.columns:
        keep = ~as_bool_series(adata.var["feature_is_filtered"]).to_numpy(dtype=bool)
        names = names[keep]
    return names


def inspect_common_genes(files: list[Path], cfg: dict) -> list[str]:
    common_genes: set[str] | None = None
    for index, file_path in enumerate(files, start=1):
        adata = ad.read_h5ad(file_path, backed="r")
        ensure_spatial_available(adata, cfg, file_path.name)
        if cfg.get("preprocess", {}).get("filter_high_quality_transfer", False) and "high_quality_transfer" not in adata.obs:
            adata.file.close()
            raise KeyError(f"{file_path.name}: filter_high_quality_transfer=true but obs['high_quality_transfer'] is missing")

        genes = set(selected_gene_names_from_backed(adata, cfg))
        common_genes = genes if common_genes is None else common_genes.intersection(genes)
        print(f"[inspect {index}/{len(files)}] {file_path.name}: cells={adata.n_obs}, genes={adata.n_vars}, selected_genes={len(genes)}")
        adata.file.close()

    return sorted(common_genes or [])


def apply_gene_names(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    adata.var["source_var_name"] = adata.var_names.astype(str)
    adata.var_names = prepared_gene_names(adata.var, cfg)
    return adata


def filter_var_features(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    preprocess_cfg = cfg.get("preprocess", {})
    if not preprocess_cfg.get("filter_feature_is_filtered", False):
        return adata
    if "feature_is_filtered" not in adata.var.columns:
        return adata
    keep = ~as_bool_series(adata.var["feature_is_filtered"]).to_numpy(dtype=bool)
    if keep.all():
        return adata
    return adata[:, keep].copy()


def filter_obs_cells(adata: ad.AnnData, cfg: dict) -> ad.AnnData:
    preprocess_cfg = cfg.get("preprocess", {})
    if preprocess_cfg.get("filter_high_quality_transfer", False):
        keep = as_bool_series(adata.obs["high_quality_transfer"]).to_numpy(dtype=bool)
        adata = adata[keep].copy()
    return adata


def ensure_spatial_available(adata: ad.AnnData, cfg: dict, file_name: str) -> None:
    preprocess_cfg = cfg.get("preprocess", {})
    spatial_key = preprocess_cfg.get("spatial_key", "ccf")
    source_key = preprocess_cfg.get("source_spatial_key", spatial_key)
    fallback_key = preprocess_cfg.get("fallback_spatial_key", "X_spatial_coords")
    coord_columns = list(preprocess_cfg.get("coord_columns", []))

    if spatial_key in adata.obsm or source_key in adata.obsm or fallback_key in adata.obsm:
        return
    if coord_columns and all(column in adata.obs.columns for column in coord_columns):
        return
    raise KeyError(
        f"{file_name}: cannot find spatial coordinates. "
        f"Expected obsm['{source_key}'], obsm['{fallback_key}'] or obs columns {coord_columns}."
    )


def ensure_spatial_key(adata: ad.AnnData, cfg: dict, file_name: str) -> ad.AnnData:
    preprocess_cfg = cfg.get("preprocess", {})
    spatial_key = preprocess_cfg.get("spatial_key", "ccf")
    source_key = preprocess_cfg.get("source_spatial_key", spatial_key)
    fallback_key = preprocess_cfg.get("fallback_spatial_key", "X_spatial_coords")
    coord_columns = list(preprocess_cfg.get("coord_columns", []))

    if spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    elif source_key in adata.obsm:
        coords = np.asarray(adata.obsm[source_key], dtype=np.float32)
    elif fallback_key in adata.obsm:
        coords = np.asarray(adata.obsm[fallback_key], dtype=np.float32)
    elif coord_columns and all(column in adata.obs.columns for column in coord_columns):
        coords = adata.obs[coord_columns].to_numpy(dtype=np.float32)
    else:
        raise KeyError(f"{file_name}: cannot build obsm['{spatial_key}']")

    adata.obsm[spatial_key] = coords
    return adata


def drop_invalid_spatial(adata: ad.AnnData, spatial_key: str, file_name: str) -> ad.AnnData:
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    keep = np.isfinite(coords).all(axis=1)
    if keep.all():
        return adata
    print(f"  {file_name}: dropping {(~keep).sum()} cells with invalid spatial coordinates")
    return adata[keep].copy()


def query_knn(tree: cKDTree, coords: np.ndarray, k: int, workers: int) -> np.ndarray:
    try:
        _, indices = tree.query(coords, k=k, workers=workers)
    except TypeError:
        _, indices = tree.query(coords, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices


def construct_knn_graph(coords: np.ndarray, n_neighbors: int = 8, chunk_size: int = 200000) -> sp.csr_matrix:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2:
        raise ValueError(f"spatial coordinates should be 2D, got shape {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("spatial coordinates contain NaN or inf")

    ranges = np.ptp(coords, axis=0)
    scale = max(float(np.max(np.abs(coords))), 1.0)
    valid_dims = np.where(ranges > 1e-8 * scale)[0]
    if len(valid_dims) == 0:
        raise ValueError("all spatial coordinate dimensions are degenerate")
    if len(valid_dims) < coords.shape[1]:
        dropped = [idx for idx in range(coords.shape[1]) if idx not in valid_dims]
        print(f"    dropping degenerate coordinate dims {dropped}; using dims {valid_dims.tolist()}")
        coords = coords[:, valid_dims]

    n_cells = coords.shape[0]
    if n_cells <= 1:
        return sp.csr_matrix((n_cells, n_cells), dtype=np.float32)

    k = min(n_neighbors + 1, n_cells)
    tree = cKDTree(coords)
    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    workers = int(os.environ.get("STEREOTRACK_KNN_WORKERS", "-1"))

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        indices = query_knn(tree, coords[start:end], k=k, workers=workers)
        rows = np.repeat(np.arange(start, end, dtype=np.int32), indices.shape[1])
        cols = indices.reshape(-1).astype(np.int32, copy=False)
        keep = cols != rows
        row_chunks.append(rows[keep])
        col_chunks.append(cols[keep])
        print(f"    kNN progress: {end}/{n_cells}", flush=True)

    rows = np.concatenate(row_chunks) if row_chunks else np.array([], dtype=np.int32)
    cols = np.concatenate(col_chunks) if col_chunks else np.array([], dtype=np.int32)
    data = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr(copy=True)
    adj.setdiag(1.0)
    adj.eliminate_zeros()
    degrees = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    degrees[degrees == 0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(degrees)
    adj = adj.multiply(inv_sqrt[:, None]).multiply(inv_sqrt[None, :]).tocsr()
    adj.data = adj.data.astype(np.float32, copy=False)
    return adj


def get_adj_sparse(adata: ad.AnnData, batch_name: str) -> sp.csr_matrix:
    if "adj_norm" in adata.obsm:
        adj_sparse = adata.obsm["adj_norm"].copy()
    elif "spatial_connectivities" in adata.obsp:
        adj_sparse = adata.obsp["spatial_connectivities"].copy()
    elif "connectivities" in adata.obsp:
        adj_sparse = adata.obsp["connectivities"].copy()
    else:
        raise KeyError(f"{batch_name}: cannot find adjacency matrix")

    if not sp.issparse(adj_sparse):
        adj_sparse = sp.csr_matrix(adj_sparse)
    return adj_sparse.tocsr()


def save_graph_npz(
    path: Path,
    feat: sp.csr_matrix,
    adj: sp.csr_matrix,
    slice_idx: int,
    coords: np.ndarray | None = None,
    compressed: bool = False,
) -> None:
    payload = {
        "feat_data": feat.data.astype(np.float16),
        "feat_indices": feat.indices.astype(np.int32),
        "feat_indptr": feat.indptr.astype(np.int32),
        "feat_shape": np.array(feat.shape),
        "adj_data": adj.data.astype(np.float16),
        "adj_indices": adj.indices.astype(np.int32),
        "adj_indptr": adj.indptr.astype(np.int32),
        "adj_shape": np.array(adj.shape),
        "slice_idx": np.array(slice_idx),
    }
    if coords is not None:
        payload["coords"] = coords.astype(np.float32)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def write_meta(meta: dict, meta_path: Path, meta_pkl_path: Path) -> None:
    with open(meta_path, "w") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False, allow_unicode=True)
    with open(meta_pkl_path, "wb") as handle:
        pickle.dump(meta, handle)


def process_slice(
    file_path: Path,
    slice_idx: int,
    common_genes: list[str],
    cache_dir: Path,
    cfg: dict,
) -> tuple[list[dict], dict]:
    preprocess_cfg = cfg.get("preprocess", {})
    training_cfg = cfg.get("training", {})
    spatial_key = preprocess_cfg.get("spatial_key", "ccf")
    patch_size = int(training_cfg.get("patch_size", 8192))
    patch_axes = int(preprocess_cfg.get("patch_axes", 2))
    n_neighbors = int(preprocess_cfg.get("n_neighbors", 8))
    knn_chunk_size = int(preprocess_cfg.get("knn_chunk_size", 200000))
    compress_npz = bool(preprocess_cfg.get("compress_npz", False))

    t0 = time.time()
    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = apply_gene_names(adata, cfg)
    adata = filter_var_features(adata, cfg)
    adata = filter_obs_cells(adata, cfg)
    adata = ensure_spatial_key(adata, cfg, file_path.name)
    if preprocess_cfg.get("drop_invalid_spatial", True):
        adata = drop_invalid_spatial(adata, spatial_key, file_path.name)
    adata = adata[:, common_genes].copy()
    adata.obs["batch"] = file_path.name
    adata.obs["slice_idx"] = slice_idx

    print(f"\n{'=' * 60}")
    print(f"slice {slice_idx + 1}: {file_path.name}, memory={mem_usage():.1f} GB")
    print(f"cells={adata.n_obs}, genes={adata.n_vars}")

    print("  [1/5] normalize/log/scale ...", end=" ", flush=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    print(f"done ({time.time() - t0:.1f}s)")

    print("  [2/5] construct kNN graph ...", flush=True)
    t1 = time.time()
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    adata.obsp["spatial_connectivities"] = construct_knn_graph(
        coords,
        n_neighbors=n_neighbors,
        chunk_size=knn_chunk_size,
    )
    print(f"  kNN graph done ({time.time() - t1:.1f}s), edges={adata.obsp['spatial_connectivities'].nnz // 2}")

    print("  [3/5] normalize adjacency ...", end=" ", flush=True)
    t1 = time.time()
    adata.obsm["adj_norm"] = normalize_adjacency(adata.obsp["spatial_connectivities"])
    print(f"done ({time.time() - t1:.1f}s)")

    print("  [4/5] get_spatial_input ...", end=" ", flush=True)
    t1 = time.time()
    adata = get_spatial_input(adata)
    print(f"done ({time.time() - t1:.1f}s)")

    print("  [5/5] save full graph and fixed patches ...", end=" ", flush=True)
    t1 = time.time()
    feat_sparse = get_feature_sparse(torch.device("cpu"), adata.obsm["spatial_input"])
    adj_sparse = get_adj_sparse(adata, file_path.name)

    graphs_dir = cache_dir / "graphs"
    patches_dir = cache_dir / "patches"
    adatas_dir = cache_dir / "adatas"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    adatas_dir.mkdir(parents=True, exist_ok=True)

    full_name = f"slice_{slice_idx}_full_graph.npz"
    full_path = graphs_dir / full_name
    save_graph_npz(full_path, feat_sparse, adj_sparse, slice_idx=slice_idx, coords=coords, compressed=compress_npz)
    slice_info = {
        "batch": file_path.name,
        "file": os.path.join("graphs", full_name),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "slice_idx": int(slice_idx),
    }

    axis_vars = coords.var(axis=0)
    n_axes = min(max(patch_axes, 1), coords.shape[1])
    top_axes = np.argsort(axis_vars)[::-1][:n_axes]

    patch_infos: list[dict] = []
    patch_id = 0
    for pass_idx, target_axis in enumerate(top_axes, start=1):
        sorted_indices = np.argsort(coords[:, target_axis])
        print(f"\n    pass {pass_idx}/{len(top_axes)} axis={target_axis}, patch_size={patch_size}")
        for start in range(0, adata.n_obs, patch_size):
            end = min(start + patch_size, adata.n_obs)
            cell_idx = np.sort(sorted_indices[start:end])
            if cell_idx.size == 0:
                continue

            feat_patch = feat_sparse[cell_idx]
            adj_patch = adj_sparse[cell_idx][:, cell_idx]
            patch_name = f"slice_{slice_idx}_patch_{patch_id}.npz"
            patch_path = patches_dir / patch_name
            save_graph_npz(patch_path, feat_patch, adj_patch, slice_idx=slice_idx, compressed=compress_npz)

            patch_infos.append(
                {
                    "batch": file_path.name,
                    "file": os.path.join("patches", patch_name),
                    "n_cells": int(cell_idx.size),
                    "n_genes": int(adata.n_vars),
                    "slice_idx": int(slice_idx),
                }
            )
            patch_id += 1

    print(f"done ({time.time() - t1:.1f}s), patches={len(patch_infos)}")

    for key in ["spatial_input", "adj_norm"]:
        if key in adata.obsm:
            del adata.obsm[key]
    if "spatial_connectivities" in adata.obsp:
        del adata.obsp["spatial_connectivities"]

    adata.X = sp.csr_matrix(adata.shape, dtype=np.float16)
    adata.write_h5ad(adatas_dir / f"slice_{slice_idx}.h5ad")

    del adata, feat_sparse, adj_sparse
    gc.collect()
    print(f"elapsed={time.time() - t0:.1f}s, memory={mem_usage():.1f} GB")
    return patch_infos, slice_info


def preprocess(cfg: dict, force: bool = False, dry_run: bool = False) -> None:
    data_path = Path(cfg["paths"]["data_path"])
    output_root = Path(cfg["paths"]["input_dir"])
    cache_dir = output_root / "cache"
    meta_path = cache_dir / "meta.yaml"
    meta_pkl_path = cache_dir / "meta.pkl"
    resume_meta = None
    resume_from = 0

    if not data_path.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")

    files = list_h5ad_files(data_path)
    if not files:
        raise FileNotFoundError(f"No h5ad files found in: {data_path}")

    if meta_path.exists() and not force:
        with open(meta_path, "r") as handle:
            existing_meta = yaml.safe_load(handle) or {}
        is_complete = bool(existing_meta.get("preprocess_complete"))
        if "preprocess_complete" not in existing_meta:
            is_complete = len(existing_meta.get("slice_info", [])) == len(files)
        if is_complete:
            print(f"Cache already exists: {cache_dir}")
            print("Use --force to rebuild")
            return
        resume_meta = existing_meta
        resume_from = len(existing_meta.get("slice_info", []))
        if resume_from > len(files):
            raise RuntimeError(f"Incomplete cache has more slices than input files: {cache_dir}. Use --force to rebuild it.")
        print(f"Incomplete cache exists: {cache_dir}")
        print(f"Resume preprocessing from slice {resume_from + 1}/{len(files)}")

    if resume_meta is not None and resume_meta.get("common_genes"):
        common_genes = list(resume_meta["common_genes"])
    else:
        common_genes = inspect_common_genes(files, cfg)
    print(f"\nfiles={len(files)}")
    print(f"common_genes={len(common_genes)}")

    if dry_run:
        print("dry-run only, no files were written")
        return

    if not common_genes:
        raise ValueError("No common genes found across h5ad files")

    if resume_meta is not None:
        meta = resume_meta
        meta["preprocess_complete"] = False
        meta.setdefault("patches", [])
        meta.setdefault("slice_info", [])
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "data_path": str(data_path),
            "common_genes": common_genes,
            "n_slices": len(files),
            "input_dim": len(common_genes),
            "patch_size": int(cfg.get("training", {}).get("patch_size", 8192)),
            "graph_method": cfg.get("preprocess", {}).get("graph_method", "knn"),
            "n_neighbors": int(cfg.get("preprocess", {}).get("n_neighbors", 8)),
            "patches": [],
            "slice_info": [],
            "preprocess_complete": False,
        }

    for slice_idx, file_path in enumerate(files[resume_from:], start=resume_from):
        patch_infos, slice_info = process_slice(file_path, slice_idx, common_genes, cache_dir, cfg)
        meta["patches"].extend(patch_infos)
        meta["slice_info"].append(slice_info)
        write_meta(meta, meta_path, meta_pkl_path)

    meta["preprocess_complete"] = True
    write_meta(meta, meta_path, meta_pkl_path)

    print("\nPreprocessing complete")
    print(f"cache_dir={cache_dir}")
    print(f"n_slices={meta['n_slices']}")
    print(f"n_patches={len(meta['patches'])}")
    print(f"input_dim={meta['input_dim']}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    preprocess(cfg, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
