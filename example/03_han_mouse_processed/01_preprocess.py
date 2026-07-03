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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stereotrack import construct_graph, get_feature_sparse, get_spatial_input, preprocess_adj_sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Han mouse brain processed h5ad files for StereoTrack MAE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_han_mouse_processed.yaml"),
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


def inspect_common_genes(files: list[Path], required_obs: list[str]) -> list[str]:
    common_genes: set[str] | None = None
    for index, file_path in enumerate(files, start=1):
        adata = ad.read_h5ad(file_path, backed="r")
        missing = [column for column in required_obs if column not in adata.obs.columns]
        if missing:
            adata.file.close()
            raise KeyError(f"{file_path.name}: missing obs columns {missing}")
        genes = set(adata.var_names.astype(str))
        common_genes = genes if common_genes is None else common_genes.intersection(genes)
        print(f"[inspect {index}/{len(files)}] {file_path.name}: cells={adata.n_obs}, genes={adata.n_vars}")
        adata.file.close()

    return sorted(common_genes or [])


def ensure_spatial_key(adata: ad.AnnData, spatial_key: str, coord_columns: list[str], file_name: str) -> ad.AnnData:
    if spatial_key in adata.obsm:
        return adata
    if all(column in adata.obs.columns for column in coord_columns):
        adata.obsm[spatial_key] = adata.obs[coord_columns].to_numpy(dtype=np.float32)
        return adata
    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 3:
        adata.obsm[spatial_key] = np.asarray(adata.obsm["spatial"][:, :3], dtype=np.float32)
        return adata
    raise KeyError(f"{file_name}: cannot build obsm['{spatial_key}'] from {coord_columns} or obsm['spatial']")


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


def save_graph_npz(path: Path, feat: sp.csr_matrix, adj: sp.csr_matrix, slice_idx: int, coords: np.ndarray | None = None) -> None:
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
    np.savez_compressed(path, **payload)


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
    coord_columns = list(preprocess_cfg.get("coord_columns", ["az", "ay", "ax"]))
    patch_size = int(training_cfg.get("patch_size", 4096))
    patch_axes = int(preprocess_cfg.get("patch_axes", 2))

    t0 = time.time()
    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if preprocess_cfg.get("filter_gene_area", True):
        min_gene_area = float(preprocess_cfg.get("min_gene_area", 0))
        if "gene_area" not in adata.obs.columns:
            raise KeyError(f"{file_path.name}: filter_gene_area=true but obs['gene_area'] is missing")
        gene_area = pd.to_numeric(adata.obs["gene_area"], errors="coerce")
        keep_mask = (gene_area > min_gene_area).fillna(False).to_numpy(dtype=bool)
        adata = adata[keep_mask].copy()

    adata = ensure_spatial_key(adata, spatial_key, coord_columns, file_path.name)
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

    print("  [2/5] construct_graph ...", end=" ", flush=True)
    t1 = time.time()
    adata = construct_graph(adata, spatial_key=spatial_key)
    print(f"done ({time.time() - t1:.1f}s)")

    print("  [3/5] preprocess_adj ...", end=" ", flush=True)
    t1 = time.time()
    adata = preprocess_adj_sparse(adata)
    print(f"done ({time.time() - t1:.1f}s)")

    print("  [4/5] get_spatial_input ...", end=" ", flush=True)
    t1 = time.time()
    adata = get_spatial_input(adata)
    print(f"done ({time.time() - t1:.1f}s)")

    print("  [5/5] save full graph and fixed patches ...", end=" ", flush=True)
    t1 = time.time()
    feat_sparse = get_feature_sparse(torch.device("cpu"), adata.obsm["spatial_input"])
    adj_sparse = get_adj_sparse(adata, file_path.name)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)

    graphs_dir = cache_dir / "graphs"
    patches_dir = cache_dir / "patches"
    adatas_dir = cache_dir / "adatas"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    adatas_dir.mkdir(parents=True, exist_ok=True)

    full_name = f"slice_{slice_idx}_full_graph.npz"
    full_path = graphs_dir / full_name
    save_graph_npz(full_path, feat_sparse, adj_sparse, slice_idx=slice_idx, coords=coords)
    slice_info = {
        "batch": file_path.name,
        "file": os.path.join("graphs", full_name),
        "n_cells": int(adata.n_obs),
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
            save_graph_npz(patch_path, feat_patch, adj_patch, slice_idx=slice_idx)

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

    preprocess_cfg = cfg.get("preprocess", {})
    coord_columns = list(preprocess_cfg.get("coord_columns", ["az", "ay", "ax"]))
    required_obs = coord_columns.copy()
    if preprocess_cfg.get("filter_gene_area", True):
        required_obs.append("gene_area")

    if resume_meta is not None and resume_meta.get("common_genes"):
        common_genes = list(resume_meta["common_genes"])
    else:
        common_genes = inspect_common_genes(files, required_obs)
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
            "patch_size": int(cfg.get("training", {}).get("patch_size", 4096)),
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
