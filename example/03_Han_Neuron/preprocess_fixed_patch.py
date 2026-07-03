import os
import gc

import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.cluster import MiniBatchKMeans
import argparse
import pickle
import time
import psutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scipy.sparse as sp

from stereotrack import (
    construct_graph,
    preprocess_adj_sparse,
    get_spatial_input,
    construct_data_multislice,
    get_feature_sparse,
)


def parse_args():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--force", action="store_true", help="强制重新预处理")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def mem_usage():
    """返回当前进程内存占用 (GB)"""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024**3


def preprocess(cfg, force=False):
    save_dir = cfg["paths"]["input_dir"]
    cache_dir = os.path.join(save_dir, "cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    meta_path = os.path.join(cache_dir, "meta.yaml")

    if os.path.exists(meta_path) and not force:
        print(f"缓存已存在: {cache_dir}")
        print("如需重新预处理，请使用 --force 参数")
        return

    # ── 读取数据 ──
    data_path = cfg["paths"]["data_path"]
    spatial_key = cfg["preprocess"]["spatial_key"]
    patch_size = cfg["training"].get("patch_size", 4096)
    
    adata_path_list = [i for i in os.listdir(data_path) if i.endswith('.h5ad')]
    # print(f"读取数据: {data_path}")
    # adata = sc.read_h5ad(data_path)
    # adata.var_names_make_unique()
    # adata.obs_names_make_unique()
    # adata.obs['z_axis'] = [str(int(i)) for i in adata.obsm[spatial_key][:,2]]
    common_genes = None
    for adata_path in adata_path_list:
        adata = sc.read_h5ad(os.path.join(data_path, adata_path))
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        common_genes = set(adata.var.index.tolist()) if common_genes is None else common_genes.intersection(set(adata.var.index.tolist()))
    common_genes = sorted(list(common_genes)) if common_genes is not None else []
    print(f"公共基因数: {len(common_genes)}")
    del adata
    gc.collect()
    # ── 按 batch 切分 ──
    batches = sorted(set(adata_path_list))
    print(f"切片数: {len(batches)}")
    print(f"切片列表: {batches}")

    # ── 逐切片预处理（边处理边保存，释放内存） ──
    
    patch_index = []
    slice_info_list = []

    adata_cache_dir = os.path.join(cache_dir, "adatas")
    Path(adata_cache_dir).mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches):
        temp = sc.read_h5ad(os.path.join(data_path, batch))
        temp.var_names_make_unique()
        temp.obs_names_make_unique()
        temp = temp[temp.obs['gene_area']>0]
        temp.obsm['ccf'] = temp.obs[['az','ay','ax']].values
        temp = temp[:, common_genes].copy()
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  切片 {i+1}/{len(batches)} (batch={batch}), 内存: {mem_usage():.1f} GB")

        n_cells = temp.shape[0]
        print(f"  细胞数: {n_cells}, 基因数: {temp.shape[1]}")

        # ── 预处理 ──
        print(f"  [1/5] normalize ...", end=" ", flush=True)
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        sc.pp.scale(temp, zero_center=False, max_value=10)
        print(f"done ({time.time()-t0:.1f}s)")

        print(f"  [2/5] construct_graph ...", end=" ", flush=True)
        t1 = time.time()
        temp = construct_graph(temp, spatial_key = spatial_key)
        print(f"done ({time.time()-t1:.1f}s)")

        print(f"  [3/5] preprocess_adj ...", end=" ", flush=True)
        t1 = time.time()
        temp = preprocess_adj_sparse(temp)
        print(f"done ({time.time()-t1:.1f}s)")

        print(f"  [4/5] get_spatial_input ...", end=" ", flush=True)
        t1 = time.time()
        temp = get_spatial_input(temp)
        print(f"done ({time.time()-t1:.1f}s)")

        # ── 提取特征和邻接矩阵 ──
        print(f"  [5/5] 提取特征 + 分 patch 保存 ...", end=" ", flush=True)
        t1 = time.time()

        feat_sparse = get_feature_sparse(torch.device("cpu"), temp.obsm["spatial_input"])

        if "adj_norm" in temp.obsm:
            adj_sparse = temp.obsm["adj_norm"].copy()
        elif "spatial_connectivities" in temp.obsp:
            adj_sparse = temp.obsp["spatial_connectivities"].copy()
        elif "connectivities" in temp.obsp:
            adj_sparse = temp.obsp["connectivities"].copy()
        else:
            raise KeyError(f"切片 {batch}: 找不到邻接矩阵")

        if not sp.issparse(adj_sparse):
            adj_sparse = sp.csr_matrix(adj_sparse)
        
        fname_full = f"slice_{i}_full_graph.npz"
        fpath_full = os.path.join(cache_dir, fname_full)
        coords_full = temp.obsm['ccf']
        
        np.savez_compressed(
            fpath_full,
            feat_data=feat_sparse.data.astype(np.float16),
            feat_indices=feat_sparse.indices.astype(np.int32),
            feat_indptr=feat_sparse.indptr.astype(np.int32),
            feat_shape=np.array(feat_sparse.shape),
            adj_data=adj_sparse.data.astype(np.float16),
            adj_indices=adj_sparse.indices.astype(np.int32),
            adj_indptr=adj_sparse.indptr.astype(np.int32),
            adj_shape=np.array(adj_sparse.shape),
            coords=coords_full.astype(np.float32)
        )

        slice_info_full = {
            "batch": batch,
            "file": fname_full,
            "n_cells": n_cells,
        }


        # ── 【核心修改】：使用 K-Means 进行正确的空间连通分块 ──
        coords = temp.obsm['ccf']
        n_cells = coords.shape[0]
        n_clusters = max(1, n_cells // patch_size + 1) # 算出需要切几块
        
        print(f"      使用 K-Means 进行空间分块 (K={n_clusters})...", end=" ", flush=True)
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=patch_size)
        labels = kmeans.fit_predict(coords)
        print("done")
        
        
        

        n_patches_this = 0
        for cluster_id in range(n_clusters):
            # 取出属于这个空间块的所有细胞索引
            cell_idx = np.where(labels == cluster_id)[0]
            
            if len(cell_idx) == 0:
                continue

            feat = feat_sparse[cell_idx]
            adj = adj_sparse[cell_idx][:, cell_idx]

            fname = f"slice_{i}_patch_{n_patches_this}.npz"
            fpath = os.path.join(cache_dir, fname)

            np.savez_compressed(
                fpath,
                feat_data=feat.data.astype(np.float16),
                feat_indices=feat.indices.astype(np.int32),
                feat_indptr=feat.indptr.astype(np.int32),
                feat_shape=np.array(feat.shape),
                adj_data=adj.data.astype(np.float16),
                adj_indices=adj.indices.astype(np.int32),
                adj_indptr=adj.indptr.astype(np.int32),
                adj_shape=np.array(adj.shape),
                slice_idx=i,
            )

            slice_info_list.append({
                "file": fname,
                "n_cells": len(cell_idx),
                "n_genes": temp.shape[1],  
                "batch": batch,
                "slice_idx": i
            })
            n_patches_this += 1

        print(f"      共切分 {n_patches_this} 个空间联通 patches。")
        del temp.obsm["spatial_input"]
        if "adj_norm" in temp.obsm:
            del temp.obsm["adj_norm"]
        if "spatial_connectivities" in temp.obsp:
            del temp.obsp["spatial_connectivities"]
        temp.write_h5ad(os.path.join(adata_cache_dir, f"slice_{i}.h5ad"))

        del temp, feat_sparse, adj_sparse
        gc.collect()

        print(f"  总耗时: {time.time()-t0:.1f}s, 内存: {mem_usage():.1f} GB")


    # ── 保存元信息 ──
    meta = {
        "common_genes": common_genes,
        "n_slices": len(batches),
        "input_dim": int(slice_info_list[0]["n_genes"]),
        "patch_size": patch_size,
        "slice_info": slice_info_list, # 存这个即可，不再需要 patches 列表
    }

    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)

    with open(os.path.join(cache_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\n{'='*60}")
    print(f"预处理完成！")
    print(f"  缓存目录: {cache_dir}")
    print(f"  切片数: {meta['n_slices']}")
    print(f"  输入维度: {meta['input_dim']}")
    for info in slice_info_list:
        print(f"    batch={info['batch']}: {info['n_cells']} cells,")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    preprocess(cfg, force=args.force)

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/preprocess_fixed_patch.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v4.yaml