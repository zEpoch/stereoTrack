import os
import yaml
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
from scipy.spatial import cKDTree
__all__ = [ 'normalize_meta', 'load_meta', 'LazyPatchDataset', 'DynamicGraphDataset']


def normalize_meta(meta):
    if not isinstance(meta, dict):
        return meta

    if "patches" not in meta and "all_patches" in meta:
        meta["patches"] = meta["all_patches"]
    if "slice_info" not in meta and "all_slice_info" in meta:
        meta["slice_info"] = meta["all_slice_info"]

    if "input_dim" not in meta:
        if isinstance(meta.get("common_human_genes"), list) and meta["common_human_genes"]:
            meta["input_dim"] = len(meta["common_human_genes"])
        elif isinstance(meta.get("patches"), list) and meta["patches"]:
            first_patch = meta["patches"][0]
            if isinstance(first_patch, dict) and "n_genes" in first_patch:
                meta["input_dim"] = int(first_patch["n_genes"])
        elif isinstance(meta.get("slice_info"), list) and meta["slice_info"]:
            first_slice = meta["slice_info"][0]
            if isinstance(first_slice, dict) and "n_genes" in first_slice:
                meta["input_dim"] = int(first_slice["n_genes"])

    if "n_patches" not in meta and isinstance(meta.get("patches"), list):
        meta["n_patches"] = len(meta["patches"])
    if "n_slices" not in meta and isinstance(meta.get("slice_info"), list):
        meta["n_slices"] = len(meta["slice_info"])

    return meta


def resolve_cache_file(cache_dir, file_name):
    direct_path = os.path.join(cache_dir, file_name)
    if os.path.exists(direct_path):
        return direct_path

    integrated_path = os.path.join(cache_dir, "integrated", file_name)
    if os.path.exists(integrated_path):
        return integrated_path

    return direct_path


def load_meta(cfg):
    cache_dir = os.path.join(cfg["paths"]["input_dir"], "cache")
    meta_yaml = os.path.join(cache_dir, "meta.yaml")
    meta_pkl = os.path.join(cache_dir, "meta.pkl")
    
    if os.path.exists(meta_yaml):
        with open(meta_yaml, "r") as f:
            return normalize_meta(yaml.safe_load(f)), cache_dir
    elif os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as f:
            return normalize_meta(pickle.load(f)), cache_dir
    else:
        raise FileNotFoundError(f"缓存不存在: {cache_dir}")

class LazyPatchDataset(Dataset):
    """
    读取预先由 K-Means 切好的、物理空间连通的固定 Patch。
    """
    def __init__(self, cache_dir, patches, species_to_id=None):
        self.cache_dir = cache_dir
        self.patches = patches
        self.species_to_id = species_to_id or {}

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        info = self.patches[idx]
        fpath = resolve_cache_file(self.cache_dir, info["file"])
        
        with np.load(fpath) as data:
            feat_sparse = sp.csr_matrix(
                (data["feat_data"].astype(np.float32), 
                 data["feat_indices"],
                 data["feat_indptr"]),
                shape=tuple(data["feat_shape"]),
            )
            adj_sparse = sp.csr_matrix(
                (data["adj_data"].astype(np.float32),
                 data["adj_indices"],
                 data["adj_indptr"]),
                shape=tuple(data["adj_shape"]),
            )
            
            slice_idx = int(data["slice_idx"]) if "slice_idx" in data else info.get("slice_idx", 0)

        # 转换为 PyTorch 需要的 Dense 张量
        feat_dense = feat_sparse.toarray()
        adj_dense = adj_sparse.toarray()

        species = info.get("species", "unknown")
        return {
            "features": torch.from_numpy(feat_dense),
            "adj": torch.from_numpy(adj_dense),
            "slice_idx": slice_idx,
            "species": species,
            "species_id": torch.tensor(self.species_to_id.get(species, -1), dtype=torch.long),
        }

class DynamicGraphDataset(Dataset):
    """
    在内存中挂载所有切片的稀疏大图。
    """
    def __init__(self, cache_dir, slice_infos, patch_size):
        self.cache_dir = cache_dir
        self.patch_size = patch_size
        self.slices = []
        self.total_cells = 0
        
        for info in slice_infos:
            fpath = resolve_cache_file(self.cache_dir, info["file"])
            data = np.load(fpath, mmap_mode="r") 
            coords = data["coords"][:]
            kdtree = cKDTree(coords)
            
            self.slices.append({
                "file_path": fpath,
                "kdtree": kdtree,
                "coords": coords,
                "n_cells": info["n_cells"]
            })
            self.total_cells += info["n_cells"]
            
        self.steps_per_epoch = self.total_cells // patch_size

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        slice_idx = np.random.randint(len(self.slices))
        s_data = self.slices[slice_idx]
        
        center_idx = np.random.randint(s_data["n_cells"])
        # 2. kdtree 查询（确保 k 不超过细胞总数）
        k_neighbors = min(self.patch_size, s_data["n_cells"])
        _, cell_idx = s_data["kdtree"].query(s_data["coords"][center_idx], k=k_neighbors)
        
        
        if isinstance(cell_idx, int):
            cell_idx = [cell_idx]
        
        # 4. 【越界兜底】：把任何大于等于 n_cells 的畸形索引强行拉回合法边界
        max_valid_idx = s_data["n_cells"] - 1
        cell_idx = np.clip(cell_idx, 0, max_valid_idx)

        with np.load(s_data["file_path"], mmap_mode="r") as data:
            feat_sparse = sp.csr_matrix(
                (data["feat_data"][:], data["feat_indices"][:], data["feat_indptr"][:]),
                shape=tuple(data["feat_shape"]),
            )
            adj_sparse = sp.csr_matrix(
                (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
                shape=tuple(data["adj_shape"]),
            )
            
            feat_sub = feat_sparse[cell_idx].toarray().astype(np.float32)
            adj_sub = adj_sparse[cell_idx][:, cell_idx].toarray().astype(np.float32)

        return {
            "features": torch.from_numpy(feat_sub),
            "adj": torch.from_numpy(adj_sub),
            "slice_idx": slice_idx,
        }
