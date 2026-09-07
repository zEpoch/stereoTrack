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


DEFAULT_CONFIG = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/"
    "config/06_config_mouse_embryo_e115.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the 3D E11.5 mouse embryo for StereoTrack."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-sections", type=int, default=None)
    parser.add_argument("--hvg-sample-cells", type=int, default=None)
    parser.add_argument("--hvg-n-top-genes", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def mem_usage() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def write_meta(meta: dict, cache_dir: Path) -> None:
    with open(cache_dir / "meta.yaml", "w") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False, allow_unicode=True)
    with open(cache_dir / "meta.pkl", "wb") as handle:
        pickle.dump(meta, handle)


def stratified_sample_indices(
    section_values: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> np.ndarray:
    valid = np.where(np.isfinite(section_values))[0]
    if sample_size <= 0 or sample_size >= valid.size:
        return valid

    rng = np.random.default_rng(random_seed)
    sections, inverse, counts = np.unique(
        section_values[valid], return_inverse=True, return_counts=True
    )
    raw = counts / counts.sum() * sample_size
    take = np.floor(raw).astype(int)
    take = np.maximum(take, 1)
    while take.sum() > sample_size:
        candidates = np.where(take > 1)[0]
        take[candidates[np.argmax(take[candidates] - raw[candidates])]] -= 1
    while take.sum() < sample_size:
        candidates = np.where(take < counts)[0]
        take[candidates[np.argmax(raw[candidates] - take[candidates])]] += 1

    sampled = []
    for section_idx, n_take in enumerate(take):
        pool = valid[inverse == section_idx]
        sampled.append(rng.choice(pool, size=min(int(n_take), pool.size), replace=False))
    out = np.sort(np.concatenate(sampled))
    print(f"HVG sample: {out.size} cells across {len(sections)} registered sections")
    return out


def select_highly_variable_genes(
    source: ad.AnnData,
    section_values: np.ndarray,
    preprocess_cfg: dict,
    args: argparse.Namespace,
) -> list[str]:
    n_top = int(
        args.hvg_n_top_genes
        if args.hvg_n_top_genes is not None
        else preprocess_cfg.get("hvg_n_top_genes", 4000)
    )
    if n_top <= 0 or n_top >= source.n_vars:
        return source.var_names.astype(str).tolist()

    sample_size = int(
        args.hvg_sample_cells
        if args.hvg_sample_cells is not None
        else preprocess_cfg.get("hvg_sample_cells", 200000)
    )
    seed = int(preprocess_cfg.get("random_seed", 42))
    sample_idx = stratified_sample_indices(section_values, sample_size, seed)
    sample = source[sample_idx, :].to_memory()
    sample.var_names_make_unique()
    min_cells = max(3, int(np.ceil(sample.n_obs * 1e-4)))
    sc.pp.filter_genes(sample, min_cells=min_cells)
    print(
        f"HVG detection filter: retained {sample.n_vars} genes expressed in at least "
        f"{min_cells} sampled cells"
    )
    sc.pp.normalize_total(sample, target_sum=1e4)
    sc.pp.log1p(sample)
    flavor = str(preprocess_cfg.get("hvg_flavor", "cell_ranger"))
    try:
        sc.pp.highly_variable_genes(
            sample,
            n_top_genes=min(n_top, sample.n_vars),
            flavor=flavor,
            subset=False,
        )
    except ValueError as error:
        if flavor == "seurat":
            raise
        print(f"HVG flavor={flavor} failed ({error}); retrying with flavor=seurat")
        sc.pp.highly_variable_genes(
            sample,
            n_top_genes=min(n_top, sample.n_vars),
            flavor="seurat",
            subset=False,
        )
    genes = sample.var_names[sample.var["highly_variable"].to_numpy()].astype(str).tolist()
    del sample
    gc.collect()
    if not genes:
        raise RuntimeError("HVG selection returned no genes")
    print(f"Selected {len(genes)} highly variable genes")
    return genes


def resolve_obs_filter(
    source: ad.AnnData,
    preprocess_cfg: dict,
) -> tuple[np.ndarray, dict]:
    obs_filter = preprocess_cfg.get("obs_filter") or {}
    if not bool(obs_filter.get("enabled", False)):
        return np.ones(source.n_obs, dtype=bool), {"enabled": False}

    key = str(obs_filter.get("key", ""))
    values = list(dict.fromkeys(str(value) for value in obs_filter.get("values", [])))
    if not key:
        raise ValueError("preprocess.obs_filter.key is empty")
    if key not in source.obs:
        raise KeyError(f"obs['{key}'] was not found")
    if not values:
        raise ValueError("preprocess.obs_filter.values is empty")

    labels = source.obs[key].astype("string").fillna("NA").astype(str)
    available = set(labels.unique().tolist())
    missing = [value for value in values if value not in available]
    mask = labels.isin(values).to_numpy()
    counts = labels[mask].value_counts().sort_index()
    meta = {
        "enabled": True,
        "key": key,
        "values": values,
        "missing_values": missing,
        "n_selected": int(mask.sum()),
        "counts": {str(label): int(count) for label, count in counts.items()},
    }
    print(f"obs filter: key={key}, selected={mask.sum():,}/{source.n_obs:,}")
    if missing:
        print(f"obs filter missing values ({len(missing)}): {missing}")
    if mask.sum() == 0:
        raise RuntimeError("obs filter selected zero cells")
    return mask, meta


def _query_edges(
    query_coords: np.ndarray,
    target_coords: np.ndarray,
    query_indices: np.ndarray,
    target_indices: np.ndarray,
    n_neighbors: int,
    remove_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if query_indices.size == 0 or target_indices.size == 0 or n_neighbors <= 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    extra = 1 if remove_self else 0
    k = min(n_neighbors + extra, target_indices.size)
    _, neighbors = cKDTree(target_coords).query(query_coords, k=k, workers=-1)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    rows = np.repeat(query_indices.astype(np.int32, copy=False), neighbors.shape[1])
    cols = target_indices[neighbors.reshape(-1)].astype(np.int32, copy=False)
    if remove_self:
        keep = rows != cols
        rows = rows[keep]
        cols = cols[keep]
    return rows, cols


def construct_section_aware_graph(
    coords: np.ndarray,
    section_values: np.ndarray,
    within_neighbors: int,
    cross_neighbors: int,
) -> sp.csr_matrix:
    coords = np.asarray(coords, dtype=np.float32)
    section_values = np.asarray(section_values)
    unique_sections = np.sort(np.unique(section_values))
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    for section in unique_sections:
        idx = np.where(section_values == section)[0]
        rows, cols = _query_edges(
            coords[idx, :2],
            coords[idx, :2],
            idx,
            idx,
            within_neighbors,
            remove_self=True,
        )
        row_parts.append(rows)
        col_parts.append(cols)

    for left, right in zip(unique_sections[:-1], unique_sections[1:]):
        left_idx = np.where(section_values == left)[0]
        right_idx = np.where(section_values == right)[0]
        rows, cols = _query_edges(
            coords[left_idx, :2],
            coords[right_idx, :2],
            left_idx,
            right_idx,
            cross_neighbors,
            remove_self=False,
        )
        row_parts.append(rows)
        col_parts.append(cols)
        rows, cols = _query_edges(
            coords[right_idx, :2],
            coords[left_idx, :2],
            right_idx,
            left_idx,
            cross_neighbors,
            remove_self=False,
        )
        row_parts.append(rows)
        col_parts.append(cols)

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.ones(rows.size, dtype=np.float32)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(coords.shape[0], coords.shape[0]))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    adj = adj.tocsr(copy=True)
    adj = adj + sp.eye(adj.shape[0], format="csr", dtype=np.float32)
    adj.eliminate_zeros()
    degree = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    inv_sqrt = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0,
    )
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


def process_slab(
    source: ad.AnnData,
    source_coords: np.ndarray,
    source_sections: np.ndarray,
    eligible_indices: np.ndarray,
    section_group: np.ndarray,
    gene_indices: np.ndarray,
    genes: list[str],
    slab_idx: int,
    cache_dir: Path,
    cfg: dict,
) -> tuple[list[dict], dict]:
    preprocess_cfg = cfg.get("preprocess", {})
    training_cfg = cfg.get("training", {})
    section_key = str(preprocess_cfg.get("section_key", "section_z"))
    spatial_key = str(preprocess_cfg.get("spatial_key", "ccf"))
    patch_size = int(training_cfg.get("patch_size", 4096))
    patch_axes = int(preprocess_cfg.get("patch_axes", 2))
    compressed = bool(preprocess_cfg.get("compress_npz", False))

    source_indices = eligible_indices[
        np.isin(source_sections[eligible_indices], section_group)
    ]
    t0 = time.time()
    x_rows = source.X[source_indices]
    if not sp.issparse(x_rows):
        x_rows = sp.csr_matrix(x_rows)
    x_selected = x_rows[:, gene_indices].tocsr()
    min_selected_counts = float(preprocess_cfg.get("min_selected_counts", 1))
    selected_counts = np.asarray(x_selected.sum(axis=1)).ravel()
    keep_cells = selected_counts >= min_selected_counts
    if not keep_cells.all():
        print(
            f"dropping {(~keep_cells).sum():,} cells with selected-gene counts "
            f"below {min_selected_counts:g}"
        )
        source_indices = source_indices[keep_cells]
        x_selected = x_selected[keep_cells].tocsr()
    adata = ad.AnnData(
        X=x_selected,
        obs=source.obs.iloc[source_indices].copy(),
        var=source.var.iloc[gene_indices].copy(),
    )
    adata.var_names = genes
    adata.obs_names_make_unique()
    coords = source_coords[source_indices].astype(np.float32, copy=False)
    sections = source_sections[source_indices]
    adata.obsm.clear()
    adata.obsm[spatial_key] = coords
    adata.obs[section_key] = sections
    adata.obs["source_row"] = source_indices.astype(np.int64)
    slab_name = f"slab_{slab_idx:03d}"
    adata.obs["batch"] = slab_name
    adata.obs["slice_idx"] = slab_idx

    print(f"\n{'=' * 70}")
    print(
        f"{slab_name}: cells={adata.n_obs}, sections={section_group.tolist()}, "
        f"genes={adata.n_vars}, memory={mem_usage():.1f} GB"
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    adata.X = adata.X.astype(np.float32)

    graph = construct_section_aware_graph(
        coords,
        sections,
        within_neighbors=int(preprocess_cfg.get("within_neighbors", 8)),
        cross_neighbors=int(preprocess_cfg.get("cross_neighbors", 3)),
    )
    adata.obsp["spatial_connectivities"] = graph
    adata.obsm["adj_norm"] = normalize_adjacency(graph)
    adata = get_spatial_input(adata)
    feat = get_feature_sparse(torch.device("cpu"), adata.obsm["spatial_input"]).tocsr()
    adj = adata.obsm["adj_norm"].tocsr()
    print(f"graph edges={graph.nnz // 2:,}")

    graphs_dir = cache_dir / "graphs"
    patches_dir = cache_dir / "patches"
    adatas_dir = cache_dir / "adatas"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    adatas_dir.mkdir(parents=True, exist_ok=True)

    graph_name = f"{slab_name}_full_graph.npz"
    save_graph_npz(
        graphs_dir / graph_name,
        feat,
        adj,
        slice_idx=slab_idx,
        coords=coords,
        compressed=compressed,
    )
    slice_info = {
        "batch": f"{slab_name}.h5ad",
        "file": str(Path("graphs") / graph_name),
        "adata_file": str(Path("adatas") / f"{slab_name}.h5ad"),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "slice_idx": slab_idx,
        "sections": [float(value) for value in section_group],
    }

    axis_variance = np.var(coords, axis=0)
    n_axes = min(max(patch_axes, 1), coords.shape[1])
    target_axes = np.argsort(axis_variance)[::-1][:n_axes]
    patch_infos: list[dict] = []
    patch_id = 0
    for target_axis in target_axes:
        order = np.argsort(coords[:, target_axis], kind="mergesort")
        for start in range(0, adata.n_obs, patch_size):
            cell_idx = np.sort(order[start : start + patch_size])
            if cell_idx.size == 0:
                continue
            patch_name = f"{slab_name}_patch_{patch_id:04d}.npz"
            save_graph_npz(
                patches_dir / patch_name,
                feat[cell_idx],
                adj[cell_idx][:, cell_idx],
                slice_idx=slab_idx,
                coords=None,
                compressed=compressed,
            )
            patch_infos.append(
                {
                    "batch": slab_name,
                    "file": str(Path("patches") / patch_name),
                    "n_cells": int(cell_idx.size),
                    "n_genes": int(adata.n_vars),
                    "slice_idx": slab_idx,
                }
            )
            patch_id += 1

    for key in ["spatial_input", "adj_norm"]:
        if key in adata.obsm:
            del adata.obsm[key]
    adata.obsp.clear()
    adata.X = sp.csr_matrix(adata.shape, dtype=np.float16)
    adata.write_h5ad(adatas_dir / f"{slab_name}.h5ad")

    del adata, feat, adj, graph
    gc.collect()
    print(
        f"saved {len(patch_infos)} patches; elapsed={time.time() - t0:.1f}s; "
        f"memory={mem_usage():.1f} GB"
    )
    return patch_infos, slice_info


def preprocess(cfg: dict, args: argparse.Namespace) -> None:
    data_path = Path(cfg["paths"]["data_path"])
    output_root = args.output_dir or Path(cfg["paths"]["input_dir"])
    cache_dir = output_root / "cache"
    preprocess_cfg = cfg.get("preprocess", {})
    source_spatial_key = str(preprocess_cfg.get("source_spatial_key", "z_correction"))
    section_axis = int(preprocess_cfg.get("section_axis", 2))
    slab_size = int(preprocess_cfg.get("slab_size", 5))

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    source = ad.read_h5ad(data_path, backed="r")
    if source_spatial_key not in source.obsm:
        source.file.close()
        raise KeyError(f"obsm['{source_spatial_key}'] was not found")

    coords = np.asarray(source.obsm[source_spatial_key], dtype=np.float32)[:, :3]
    valid = np.isfinite(coords).all(axis=1)
    if not valid.all():
        source.file.close()
        raise ValueError(f"{(~valid).sum()} cells have invalid {source_spatial_key} coordinates")
    obs_filter_mask, obs_filter_meta = resolve_obs_filter(source, preprocess_cfg)
    eligible_mask = valid & obs_filter_mask
    if eligible_mask.sum() == 0:
        source.file.close()
        raise RuntimeError("No cells remain after coordinate and obs filtering")
    eligible_indices = np.where(eligible_mask)[0].astype(np.int64, copy=False)
    section_values = coords[:, section_axis]
    sections = np.sort(np.unique(section_values[eligible_indices]))
    if args.max_sections is not None:
        sections = sections[: args.max_sections]
    slab_groups = [sections[start : start + slab_size] for start in range(0, len(sections), slab_size)]

    print(f"input={data_path}")
    print(f"shape={source.n_obs:,} cells x {source.n_vars:,} genes")
    print(f"eligible cells={eligible_indices.size:,}")
    print(f"spatial={source_spatial_key}, registered_sections={len(sections)}, slabs={len(slab_groups)}")
    print(
        f"eligible cells per section: "
        f"min={min(np.sum(section_values[eligible_indices] == value) for value in sections):,}, "
        f"max={max(np.sum(section_values[eligible_indices] == value) for value in sections):,}"
    )
    if args.dry_run:
        source.file.close()
        return

    meta_path = cache_dir / "meta.yaml"
    resume_meta = None
    resume_from = 0
    if meta_path.exists() and not args.force:
        with open(meta_path, "r") as handle:
            existing = yaml.safe_load(handle) or {}
        if existing.get("preprocess_complete") and len(existing.get("slice_info", [])) == len(slab_groups):
            print(f"Complete cache already exists: {cache_dir}")
            source.file.close()
            return
        resume_meta = existing
        resume_from = len(existing.get("slice_info", []))
        print(f"Resuming from slab {resume_from + 1}/{len(slab_groups)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    if resume_meta is not None and resume_meta.get("common_genes"):
        genes = list(resume_meta["common_genes"])
    else:
        genes = select_highly_variable_genes(source, section_values, preprocess_cfg, args)
    gene_lookup = {str(gene): idx for idx, gene in enumerate(source.var_names.astype(str))}
    gene_indices = np.asarray([gene_lookup[gene] for gene in genes], dtype=np.int64)

    if resume_meta is None or args.force:
        meta = {
            "data_path": str(data_path),
            "common_genes": genes,
            "n_slices": len(slab_groups),
            "input_dim": len(genes),
            "patch_size": int(cfg.get("training", {}).get("patch_size", 4096)),
            "spatial_key": str(preprocess_cfg.get("spatial_key", "ccf")),
            "source_spatial_key": source_spatial_key,
            "section_key": str(preprocess_cfg.get("section_key", "section_z")),
            "obs_filter": obs_filter_meta,
            "n_source_cells": int(source.n_obs),
            "n_eligible_cells": int(eligible_indices.size),
            "patches": [],
            "slice_info": [],
            "preprocess_complete": False,
        }
        resume_from = 0
    else:
        meta = resume_meta
        meta["preprocess_complete"] = False

    for slab_idx in range(resume_from, len(slab_groups)):
        patch_infos, slice_info = process_slab(
            source,
            coords,
            section_values,
            eligible_indices,
            slab_groups[slab_idx],
            gene_indices,
            genes,
            slab_idx,
            cache_dir,
            cfg,
        )
        meta["patches"].extend(patch_infos)
        meta["slice_info"].append(slice_info)
        write_meta(meta, cache_dir)

    meta["preprocess_complete"] = True
    write_meta(meta, cache_dir)
    source.file.close()
    print("\nPreprocessing complete")
    print(f"cache={cache_dir}")
    print(f"slabs={len(meta['slice_info'])}, patches={len(meta['patches'])}, genes={len(genes)}")


def main() -> None:
    args = parse_args()
    preprocess(load_config(args.config), args)


if __name__ == "__main__":
    main()
