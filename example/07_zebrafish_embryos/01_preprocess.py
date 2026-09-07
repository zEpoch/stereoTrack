from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
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
from scipy.spatial import Delaunay, cKDTree
try:
    from scipy.spatial import QhullError
except ImportError:  # scipy < 1.8
    from scipy.spatial.qhull import QhullError


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stereotrack import get_feature_sparse, get_spatial_input


DEFAULT_CONFIG = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/"
    "config/07_config_zebrafish_embryos.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess measured zebrafish embryo weMERFISH h5ad files for StereoTrack."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def mem_usage() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def list_h5ad_files(data_path: Path, cfg: dict, max_files: int | None = None) -> list[Path]:
    files = sorted(path for path in data_path.glob("*.h5ad") if path.is_file())
    include_stems = cfg.get("preprocess", {}).get("include_stems", None)
    if include_stems:
        include_order = {str(stem): idx for idx, stem in enumerate(include_stems)}
        include = set(include_order)
        files = [path for path in files if path.stem in include]
        files = sorted(files, key=lambda path: include_order[path.stem])
        missing = sorted(include.difference(path.stem for path in files))
        if missing:
            print(f"[warn] {len(missing)} configured include_stems were not found: {missing[:20]}")
    if max_files is not None:
        files = files[: max(int(max_files), 0)]
    return files


def parse_stage_and_embryo(file_path: Path) -> tuple[str, str]:
    match = re.search(r"_(A_50p|B_75p|C_6s)_(E\d+)", file_path.name)
    if not match:
        return file_path.stem, file_path.stem
    return match.group(1), match.group(2)


def spatial_key_for_file(file_path: Path, cfg: dict) -> tuple[str, str, str]:
    stage, embryo_id = parse_stage_and_embryo(file_path)
    preprocess_cfg = cfg.get("preprocess", {})
    stage_keys = preprocess_cfg.get("stage_spatial_keys", {})
    if stage in stage_keys:
        return stage, embryo_id, str(stage_keys[stage])
    return stage, embryo_id, str(preprocess_cfg.get("source_spatial_key", "spatial"))


def prepared_gene_names(adata: ad.AnnData) -> pd.Index:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw_name in adata.var_names.astype(str):
        name = raw_name if raw_name and raw_name.lower() != "nan" else "unknown"
        count = seen.get(name, 0)
        out.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return pd.Index(out)


def inspect_common_genes(files: list[Path], cfg: dict) -> list[str]:
    common_genes: set[str] | None = None
    for index, file_path in enumerate(files, start=1):
        stage, embryo_id, spatial_key = spatial_key_for_file(file_path, cfg)
        adata = ad.read_h5ad(file_path, backed="r")
        if spatial_key not in adata.obsm:
            adata.file.close()
            raise KeyError(f"{file_path.name}: obsm['{spatial_key}'] was not found")
        genes = set(prepared_gene_names(adata).tolist())
        common_genes = genes if common_genes is None else common_genes.intersection(genes)
        print(
            f"[inspect {index}/{len(files)}] {file_path.name}: "
            f"stage={stage}, embryo={embryo_id}, spatial={spatial_key}, "
            f"cells={adata.n_obs:,}, genes={adata.n_vars:,}"
        )
        adata.file.close()
    return sorted(common_genes or [])


def query_knn(tree: cKDTree, coords: np.ndarray, k: int, workers: int) -> np.ndarray:
    try:
        _, indices = tree.query(coords, k=k, workers=workers)
    except TypeError:
        _, indices = tree.query(coords, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices


def construct_knn_graph(
    coords: np.ndarray,
    n_neighbors: int,
    chunk_size: int,
) -> sp.csr_matrix:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2:
        raise ValueError(f"coordinates should be 2D, got {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain NaN or inf")

    ranges = np.ptp(coords, axis=0)
    scale = max(float(np.max(np.abs(coords))), 1.0)
    valid_dims = np.where(ranges > 1e-8 * scale)[0]
    if valid_dims.size == 0:
        raise ValueError("all coordinate dimensions are degenerate")
    if valid_dims.size < coords.shape[1]:
        dropped = [idx for idx in range(coords.shape[1]) if idx not in valid_dims]
        print(f"    dropping degenerate coordinate dims {dropped}; using dims {valid_dims.tolist()}")
        coords = coords[:, valid_dims]

    n_cells = coords.shape[0]
    if n_cells <= 1:
        return sp.csr_matrix((n_cells, n_cells), dtype=np.float32)

    k = min(int(n_neighbors) + 1, n_cells)
    tree = cKDTree(coords)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    workers = int(os.environ.get("STEREOTRACK_KNN_WORKERS", "-1"))

    for start in range(0, n_cells, int(chunk_size)):
        end = min(start + int(chunk_size), n_cells)
        indices = query_knn(tree, coords[start:end], k=k, workers=workers)
        rows = np.repeat(np.arange(start, end, dtype=np.int32), indices.shape[1])
        cols = indices.reshape(-1).astype(np.int32, copy=False)
        keep = rows != cols
        row_parts.append(rows[keep])
        col_parts.append(cols[keep])
        print(f"    kNN progress: {end:,}/{n_cells:,}", flush=True)

    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int32)
    cols = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int32)
    data = np.ones(rows.size, dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def construct_delaunay_graph(coords: np.ndarray) -> sp.csr_matrix:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"coordinates should be 2D, got {coords.shape}")
    if coords.shape[1] < 3:
        raise ValueError(f"Delaunay 3D requires at least 3 coordinate columns, got {coords.shape[1]}")
    coords = coords[:, :3]
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain NaN or inf")

    n_cells = coords.shape[0]
    if n_cells < 5:
        raise ValueError("Delaunay 3D requires at least five cells")

    ranges = np.ptp(coords, axis=0)
    scale = max(float(np.max(np.abs(coords))), 1.0)
    if np.count_nonzero(ranges > 1e-8 * scale) < 3:
        raise ValueError("Delaunay 3D requires three non-degenerate coordinate dimensions")

    try:
        simplices = Delaunay(coords, qhull_options="Qbb Qc Qz Q12 QJ").simplices
    except QhullError as error:
        raise RuntimeError("3D Delaunay construction failed; check/rescale coordinates") from error

    simplices = simplices[(simplices < n_cells).all(axis=1)]
    if simplices.size == 0:
        raise RuntimeError("3D Delaunay produced no finite tetrahedra")

    pairs = np.concatenate(
        [
            simplices[:, [0, 1]],
            simplices[:, [0, 2]],
            simplices[:, [0, 3]],
            simplices[:, [1, 2]],
            simplices[:, [1, 3]],
            simplices[:, [2, 3]],
        ],
        axis=0,
    )
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0).astype(np.int32, copy=False)

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n_cells, n_cells))
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
    out = adj.multiply(inv_sqrt[:, None]).multiply(inv_sqrt[None, :]).tocsr()
    out.data = out.data.astype(np.float32, copy=False)
    return out


def save_graph_npz(
    path: Path,
    feat: sp.csr_matrix,
    adj: sp.csr_matrix,
    slice_idx: int,
    coords: np.ndarray | None,
    compressed: bool,
) -> None:
    payload = {
        "feat_data": feat.data.astype(np.float16),
        "feat_indices": feat.indices.astype(np.int32),
        "feat_indptr": feat.indptr.astype(np.int32),
        "feat_shape": np.asarray(feat.shape, dtype=np.int64),
        "adj_data": adj.data.astype(np.float16),
        "adj_indices": adj.indices.astype(np.int32),
        "adj_indptr": adj.indptr.astype(np.int32),
        "adj_shape": np.asarray(adj.shape, dtype=np.int64),
        "slice_idx": np.asarray(slice_idx, dtype=np.int32),
    }
    if coords is not None:
        payload["coords"] = np.asarray(coords, dtype=np.float32)
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **payload)


def write_meta(meta: dict, cache_dir: Path) -> None:
    yaml_path = cache_dir / "meta.yaml"
    yaml_tmp = cache_dir / "meta.yaml.tmp"
    pkl_path = cache_dir / "meta.pkl"
    pkl_tmp = cache_dir / "meta.pkl.tmp"
    with open(yaml_tmp, "w") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False, allow_unicode=True)
    os.replace(yaml_tmp, yaml_path)
    with open(pkl_tmp, "wb") as handle:
        pickle.dump(meta, handle)
    os.replace(pkl_tmp, pkl_path)


def load_existing_meta(cache_dir: Path) -> dict | None:
    for path in [cache_dir / "meta.pkl", cache_dir / "meta.yaml"]:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            if path.suffix == ".pkl":
                with open(path, "rb") as handle:
                    return pickle.load(handle)
            with open(path, "r") as handle:
                return yaml.safe_load(handle) or None
        except Exception as error:
            print(f"[warn] failed to read existing meta {path}: {error}")
    return None


def _cache_file_ok(cache_dir: Path, rel_path: str | None) -> bool:
    if not rel_path:
        return False
    path = cache_dir / str(rel_path)
    return path.is_file() and path.stat().st_size > 0


def slice_complete(meta: dict, cache_dir: Path, slice_idx: int) -> bool:
    slice_infos = [info for info in meta.get("slice_info", []) if int(info.get("slice_idx", -1)) == slice_idx]
    if len(slice_infos) != 1:
        return False
    slice_info = slice_infos[0]
    if not _cache_file_ok(cache_dir, slice_info.get("file")):
        return False
    if not _cache_file_ok(cache_dir, slice_info.get("adata_file")):
        return False

    patches = [info for info in meta.get("patches", []) if int(info.get("slice_idx", -1)) == slice_idx]
    if not patches:
        return False
    return all(_cache_file_ok(cache_dir, info.get("file")) for info in patches)


def prune_meta_slice(meta: dict, slice_idx: int) -> None:
    meta["slice_info"] = [
        info for info in meta.get("slice_info", [])
        if int(info.get("slice_idx", -1)) != slice_idx
    ]
    meta["patches"] = [
        info for info in meta.get("patches", [])
        if int(info.get("slice_idx", -1)) != slice_idx
    ]


def sort_meta_entries(meta: dict) -> None:
    meta["slice_info"] = sorted(meta.get("slice_info", []), key=lambda info: int(info.get("slice_idx", -1)))
    meta["patches"] = sorted(
        meta.get("patches", []),
        key=lambda info: (int(info.get("slice_idx", -1)), str(info.get("file", ""))),
    )


def load_reference_common_genes(path: Path) -> list[str]:
    if path.suffix == ".pkl":
        with open(path, "rb") as handle:
            meta = pickle.load(handle)
    else:
        with open(path, "r") as handle:
            meta = yaml.safe_load(handle)

    common_genes = list(meta.get("common_genes") or meta.get("common_human_genes") or [])
    if not common_genes:
        raise ValueError(f"No common_genes found in reference meta: {path}")
    return [str(gene) for gene in common_genes]


def cache_dir_from_config(cfg: dict) -> Path:
    cache_name = str(cfg.get("preprocess", {}).get("cache_name", "cache"))
    return Path(cfg["paths"]["input_dir"]) / cache_name


def process_embryo(
    file_path: Path,
    slice_idx: int,
    common_genes: list[str],
    cache_dir: Path,
    cfg: dict,
) -> tuple[list[dict], dict]:
    preprocess_cfg = cfg.get("preprocess", {})
    training_cfg = cfg.get("training", {})
    spatial_key = str(preprocess_cfg.get("spatial_key", "ccf"))
    patch_size = int(training_cfg.get("patch_size", 4096))
    patch_axes = int(preprocess_cfg.get("patch_axes", 3))
    graph_method = str(preprocess_cfg.get("graph_method", "knn")).lower()
    n_neighbors = int(preprocess_cfg.get("n_neighbors", 8))
    knn_chunk_size = int(preprocess_cfg.get("knn_chunk_size", 100000))
    min_counts = float(preprocess_cfg.get("min_counts", 0))
    compressed = bool(preprocess_cfg.get("compress_npz", False))
    skip_normalize = bool(preprocess_cfg.get("skip_normalize_log_scale", True))
    expression_layer = preprocess_cfg.get("expression_layer", None)
    patch_mode = str(preprocess_cfg.get("patch_mode", "axis")).lower()
    stage, embryo_id, source_spatial_key = spatial_key_for_file(file_path, cfg)

    t0 = time.time()
    adata = sc.read_h5ad(file_path)
    adata.var_names = prepared_gene_names(adata)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = adata[:, common_genes].copy()
    if expression_layer:
        if expression_layer not in adata.layers:
            raise KeyError(f"{file_path.name}: layers['{expression_layer}'] was not found")
        adata.X = adata.layers[expression_layer].copy()
    source_row = np.arange(adata.n_obs, dtype=np.int64)

    coords = np.asarray(adata.obsm[source_spatial_key], dtype=np.float32)[:, :3]
    valid = np.isfinite(coords).all(axis=1)
    if not valid.all():
        print(f"  dropping {(~valid).sum():,} cells with invalid {source_spatial_key} coordinates")
        adata = adata[valid].copy()
        coords = coords[valid]
        source_row = source_row[valid]

    if min_counts > 0:
        row_sum = np.asarray(adata.X.sum(axis=1)).ravel() if sp.issparse(adata.X) else np.asarray(adata.X).sum(axis=1)
        keep = row_sum >= min_counts
        if not keep.all():
            print(f"  dropping {(~keep).sum():,} cells with total expression below {min_counts:g}")
            adata = adata[keep].copy()
            coords = coords[keep]
            source_row = source_row[keep]

    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))

    adata.obsm[spatial_key] = coords
    adata.obs["source_file"] = file_path.name
    adata.obs["source_spatial_key"] = source_spatial_key
    adata.obs["stage"] = stage
    adata.obs["embryo_id"] = embryo_id
    adata.obs["batch"] = file_path.stem
    adata.obs["slice_idx"] = slice_idx
    adata.obs["source_row"] = source_row

    print(f"\n{'=' * 70}")
    print(
        f"slice {slice_idx + 1}: {file_path.name}, stage={stage}, embryo={embryo_id}, "
        f"spatial={source_spatial_key}->{spatial_key}, expression_layer={expression_layer or 'X'}, "
        f"cells={adata.n_obs:,}, "
        f"genes={adata.n_vars:,}, memory={mem_usage():.1f} GB"
    )
    if skip_normalize:
        print("  expression: using input X directly; no normalize_total/log1p/scale")
    else:
        print("  expression: normalize_total(target_sum=1e4) + log1p + scale(zero_center=False, max_value=10)")
        sc.pp.normalize_total(adata, target_sum=float(preprocess_cfg.get("target_sum", 1e4)))
        sc.pp.log1p(adata)
        sc.pp.scale(
            adata,
            zero_center=False,
            max_value=float(preprocess_cfg.get("max_value", 10.0)),
        )
        if sp.issparse(adata.X):
            adata.X = adata.X.tocsr().astype(np.float32)
        else:
            adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))

    print(f"  [1/4] construct spatial graph ({graph_method}) ...", flush=True)
    t1 = time.time()
    if graph_method in {"delaunay_3d", "delaunay"}:
        graph = construct_delaunay_graph(coords)
    elif graph_method in {"knn", "knn_3d"}:
        graph = construct_knn_graph(coords, n_neighbors=n_neighbors, chunk_size=knn_chunk_size)
    else:
        raise ValueError("preprocess.graph_method should be 'knn' or 'delaunay_3d'")
    print(f"  graph done ({time.time() - t1:.1f}s), edges={graph.nnz // 2:,}")

    print("  [2/4] normalize adjacency ...", end=" ", flush=True)
    adj_norm = normalize_adjacency(graph)
    print("done")

    print("  [3/4] build sparse features ...", end=" ", flush=True)
    adata.obsp["spatial_connectivities"] = graph
    adata.obsm["adj_norm"] = adj_norm
    adata = get_spatial_input(adata)
    feat = get_feature_sparse(torch.device("cpu"), adata.obsm["spatial_input"]).tocsr()
    adj = adata.obsm["adj_norm"].tocsr()
    print("done")

    print("  [4/4] save full graph and fixed patches ...", flush=True)
    graphs_dir = cache_dir / "graphs"
    patches_dir = cache_dir / "patches"
    adatas_dir = cache_dir / "adatas"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    adatas_dir.mkdir(parents=True, exist_ok=True)

    full_name = f"slice_{slice_idx}_full_graph.npz"
    save_graph_npz(
        graphs_dir / full_name,
        feat,
        adj,
        slice_idx=slice_idx,
        coords=coords,
        compressed=compressed,
    )

    slice_info = {
        "batch": f"{file_path.stem}.h5ad",
        "source_file": file_path.name,
        "file": str(Path("graphs") / full_name),
        "adata_file": str(Path("adatas") / f"slice_{slice_idx}.h5ad"),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "slice_idx": int(slice_idx),
        "stage": stage,
        "embryo_id": embryo_id,
        "source_spatial_key": source_spatial_key,
        "spatial_key": spatial_key,
    }

    patch_infos: list[dict] = []
    if patch_mode == "full_graph":
        print("    patch_mode=full_graph; using one full graph per sample for training")
        patch_infos.append(
            {
                "batch": file_path.stem,
                "file": str(Path("graphs") / full_name),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "slice_idx": int(slice_idx),
                "stage": stage,
                "embryo_id": embryo_id,
                "species": stage,
            }
        )
    elif patch_mode == "axis":
        axis_vars = np.var(coords, axis=0)
        n_axes = min(max(patch_axes, 1), coords.shape[1])
        top_axes = np.argsort(axis_vars)[::-1][:n_axes]
        patch_id = 0
        for target_axis in top_axes:
            order = np.argsort(coords[:, target_axis], kind="mergesort")
            print(f"    patch pass axis={target_axis}, patch_size={patch_size}")
            for start in range(0, adata.n_obs, patch_size):
                cell_idx = np.sort(order[start : start + patch_size])
                if cell_idx.size == 0:
                    continue
                patch_name = f"slice_{slice_idx}_patch_{patch_id:04d}.npz"
                save_graph_npz(
                    patches_dir / patch_name,
                    feat[cell_idx],
                    adj[cell_idx][:, cell_idx],
                    slice_idx=slice_idx,
                    coords=None,
                    compressed=compressed,
                )
                patch_infos.append(
                    {
                        "batch": file_path.stem,
                        "file": str(Path("patches") / patch_name),
                        "n_cells": int(cell_idx.size),
                        "n_genes": int(adata.n_vars),
                        "slice_idx": int(slice_idx),
                        "stage": stage,
                        "embryo_id": embryo_id,
                        "species": stage,
                    }
                )
                patch_id += 1
    else:
        raise ValueError("preprocess.patch_mode should be 'axis' or 'full_graph'")

    for key in ["spatial_input", "adj_norm"]:
        if key in adata.obsm:
            del adata.obsm[key]
    adata.obsp.clear()
    adata.layers.clear()
    adata.X = sp.csr_matrix(adata.shape, dtype=np.float16)
    adata.write_h5ad(adatas_dir / f"slice_{slice_idx}.h5ad")

    del adata, feat, adj, adj_norm, graph
    gc.collect()
    print(f"saved {len(patch_infos)} patches; elapsed={time.time() - t0:.1f}s; memory={mem_usage():.1f} GB")
    return patch_infos, slice_info


def preprocess(cfg: dict, args: argparse.Namespace) -> None:
    data_path = Path(cfg["paths"]["data_path"])
    cache_dir = cache_dir_from_config(cfg)
    meta_path = cache_dir / "meta.yaml"
    files = list_h5ad_files(data_path, cfg, args.max_files)
    if not files:
        raise FileNotFoundError(f"No h5ad files found in {data_path}")

    existing_meta = None if args.force else load_existing_meta(cache_dir)
    if existing_meta is not None:
        if bool(existing_meta.get("preprocess_complete")) and len(existing_meta.get("slice_info", [])) == len(files):
            print(f"Complete cache already exists: {cache_dir}")
            print("Use --force to rebuild")
            return
        print(
            f"[resume] found existing cache metadata: "
            f"slices={len(existing_meta.get('slice_info', []))}/{len(files)}, "
            f"patches={len(existing_meta.get('patches', []))}"
        )

    reference_meta = cfg.get("preprocess", {}).get("reference_meta", None)
    if reference_meta:
        common_genes = load_reference_common_genes(Path(reference_meta))
        print(f"\nfiles={len(files)}, reference_common_genes={len(common_genes)}")
        for file_path in files:
            _, _, spatial_key = spatial_key_for_file(file_path, cfg)
            adata = ad.read_h5ad(file_path, backed="r")
            genes = set(prepared_gene_names(adata).tolist())
            n_obs, n_vars = adata.n_obs, adata.n_vars
            missing = [gene for gene in common_genes if gene not in genes]
            adata.file.close()
            if missing:
                raise KeyError(
                    f"{file_path.name}: missing {len(missing)} genes from reference common_genes; "
                    f"first missing genes: {missing[:20]}"
                )
            print(
                f"[reference check] {file_path.name}: spatial={spatial_key}, "
                f"cells={n_obs:,}, genes={n_vars:,}, matched={len(common_genes):,}"
            )
    else:
        if (
            existing_meta is not None
            and isinstance(existing_meta.get("common_genes"), list)
            and len(existing_meta.get("common_genes", [])) > 0
            and int(existing_meta.get("n_slices", len(files))) == len(files)
        ):
            common_genes = [str(gene) for gene in existing_meta["common_genes"]]
            print(f"\nfiles={len(files)}, reusing existing common_genes={len(common_genes)}")
        else:
            common_genes = inspect_common_genes(files, cfg)
            print(f"\nfiles={len(files)}, common_genes={len(common_genes)}")
    if not common_genes:
        raise RuntimeError("No common genes found across input h5ad files")
    if args.dry_run:
        print("dry-run only, no files were written")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = existing_meta if existing_meta is not None else {}
    meta.update(
        {
            "data_path": str(data_path),
            "common_genes": common_genes,
            "n_slices": len(files),
            "input_dim": len(common_genes),
            "patch_size": int(cfg.get("training", {}).get("patch_size", 4096)),
            "preprocess": {
                "spatial_key": str(cfg.get("preprocess", {}).get("spatial_key", "ccf")),
                "cache_name": str(cfg.get("preprocess", {}).get("cache_name", "cache")),
                "stage_spatial_keys": cfg.get("preprocess", {}).get("stage_spatial_keys", {}),
                "source_spatial_key": str(cfg.get("preprocess", {}).get("source_spatial_key", "spatial")),
                "reference_meta": cfg.get("preprocess", {}).get("reference_meta", None),
                "expression_layer": cfg.get("preprocess", {}).get("expression_layer", None),
                "skip_normalize_log_scale": bool(cfg.get("preprocess", {}).get("skip_normalize_log_scale", True)),
                "target_sum": float(cfg.get("preprocess", {}).get("target_sum", 1e4)),
                "max_value": float(cfg.get("preprocess", {}).get("max_value", 10.0)),
                "graph_method": str(cfg.get("preprocess", {}).get("graph_method", "knn")),
                "n_neighbors": int(cfg.get("preprocess", {}).get("n_neighbors", 8)),
                "patch_mode": str(cfg.get("preprocess", {}).get("patch_mode", "axis")),
            },
            "preprocess_complete": False,
        }
    )
    meta.setdefault("patches", [])
    meta.setdefault("slice_info", [])
    meta["patches"] = [
        info for info in meta["patches"]
        if 0 <= int(info.get("slice_idx", -1)) < len(files)
    ]
    meta["slice_info"] = [
        info for info in meta["slice_info"]
        if 0 <= int(info.get("slice_idx", -1)) < len(files)
    ]
    sort_meta_entries(meta)
    write_meta(meta, cache_dir)

    for slice_idx, file_path in enumerate(files):
        if not args.force and slice_complete(meta, cache_dir, slice_idx):
            print(f"[resume] skip complete slice {slice_idx + 1}/{len(files)}: {file_path.name}")
            continue
        prune_meta_slice(meta, slice_idx)
        patch_infos, slice_info = process_embryo(file_path, slice_idx, common_genes, cache_dir, cfg)
        meta["patches"].extend(patch_infos)
        meta["slice_info"].append(slice_info)
        sort_meta_entries(meta)
        write_meta(meta, cache_dir)

    sort_meta_entries(meta)
    complete = len(meta["slice_info"]) == len(files) and all(
        slice_complete(meta, cache_dir, slice_idx)
        for slice_idx in range(len(files))
    )
    meta["preprocess_complete"] = complete
    meta["n_patches"] = len(meta["patches"])
    write_meta(meta, cache_dir)

    print("\nPreprocessing complete" if complete else "\nPreprocessing incomplete")
    print(f"cache={cache_dir}")
    print(f"slices={meta['n_slices']}, patches={meta['n_patches']}, input_dim={meta['input_dim']}")


def main() -> None:
    args = parse_args()
    preprocess(load_config(args.config), args)


if __name__ == "__main__":
    main()
