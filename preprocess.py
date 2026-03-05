import os
import gc
import argparse
import pickle
import time
import psutil
from pathlib import Path

import yaml
import numpy as np
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
    save_dir = cfg["paths"]["save_dir"]
    cache_dir = os.path.join(save_dir, "cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    meta_path = os.path.join(cache_dir, "meta.yaml")

    if os.path.exists(meta_path) and not force:
        print(f"缓存已存在: {cache_dir}")
        print("如需重新预处理，请使用 --force 参数")
        return

    # ── 读取数据 ──
    data_path = cfg["paths"]["data_path"]
    batch_label = cfg["preprocess"]["batch_label"]
    spatial_key = cfg["preprocess"]["spatial_key"]

    print(f"读取数据: {data_path}")
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs['z_axis'] = [str(int(i)) for i in adata.obsm[spatial_key][:,2]]
    common_genes = sorted(adata.var.index.tolist())
    print(f"公共基因数: {len(common_genes)}")

    # ── 按 batch 切分 ──
    adata.obs[batch_label] = adata.obs[batch_label].astype(str)
    batches = sorted(set(adata.obs[batch_label]))
    print(f"切片数: {len(batches)}")
    print(f"切片列表: {batches}")

    # ── 逐切片预处理（边处理边保存，释放内存） ──
    patch_size = cfg["training"].get("patch_size", 4096)
    patch_index = []
    slice_info_list = []

    adata_cache_dir = os.path.join(cache_dir, "adatas")
    Path(adata_cache_dir).mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  切片 {i+1}/{len(batches)} (batch={batch}), 内存: {mem_usage():.1f} GB")

        temp = adata[adata.obs[batch_label] == batch].copy()
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

        # ── 保存 adata（推理用）──
        temp.write_h5ad(os.path.join(adata_cache_dir, f"slice_{i}.h5ad"))

        # ── 提取特征和邻接矩阵 ──
        print(f"  [5/5] 提取特征 + 分 patch 保存 ...", end=" ", flush=True)
        t1 = time.time()

        feat_sparse = get_feature_sparse(torch.device("cpu"), temp.obsm["spatial_input"])

        # 邻接矩阵：优先用归一化后的
        if "adj_norm" in temp.obsm:
            adj_sparse = temp.obsm["adj_norm"]
        elif "spatial_connectivities" in temp.obsp:
            adj_sparse = temp.obsp["spatial_connectivities"]
        elif "connectivities" in temp.obsp:
            adj_sparse = temp.obsp["connectivities"]
        else:
            raise KeyError(f"切片 {batch}: 找不到邻接矩阵")

        if not sp.issparse(adj_sparse):
            adj_sparse = sp.csr_matrix(adj_sparse)
        # ── 按 patch 分片保存 ──
        n_patches_this = 0
        for start in range(0, n_cells, patch_size):
            end = min(start + patch_size, n_cells)
            cell_idx = np.arange(start, end)

            feat = feat_sparse[cell_idx]
            adj = adj_sparse[cell_idx][:, cell_idx]

            # 确保是 CSR 格式
            if sp.issparse(feat):
                feat = sp.csr_matrix(feat)
            else:
                feat = sp.csr_matrix(np.asarray(feat, dtype=np.float32))

            if sp.issparse(adj):
                adj = sp.csr_matrix(adj)
            else:
                adj = sp.csr_matrix(np.asarray(adj, dtype=np.float32))

            fname = f"patch_s{i}_c{start}_{end}.npz"
            fpath = os.path.join(cache_dir, fname)

            # 稀疏保存：只存 data/indices/indptr/shape
            np.savez_compressed(
                fpath,
                # 特征矩阵（float16 省一半）
                feat_data=feat.data.astype(np.float16),
                feat_indices=feat.indices,
                feat_indptr=feat.indptr,
                feat_shape=np.array(feat.shape),
                # 邻接矩阵（float32 保精度）
                adj_data=adj.data.astype(np.float32),
                adj_indices=adj.indices,
                adj_indptr=adj.indptr,
                adj_shape=np.array(adj.shape),
            )

            patch_index.append({
                "slice_idx": i,
                "start": int(start),
                "end": int(end),
                "n_cells": int(end - start),
                "file": fname,
            })
            n_patches_this += 1

        print(f"done, {n_patches_this} patches ({time.time()-t1:.1f}s)")

        slice_info_list.append({
            "batch": batch,
            "n_cells": int(n_cells),
            "n_genes": int(temp.shape[1]),
            "n_patches": n_patches_this,
        })

        # ── 释放当前切片内存 ──
        del temp, feat_sparse, adj_sparse
        gc.collect()

        print(f"  总耗时: {time.time()-t0:.1f}s, 内存: {mem_usage():.1f} GB")

    # ── 释放原始数据 ──
    del adata
    gc.collect()

    # ── 保存元信息 ──
    meta = {
        "common_genes": common_genes,
        "n_slices": len(batches),
        "input_dim": int(slice_info_list[0]["n_genes"]),
        "patch_size": patch_size,
        "n_patches": len(patch_index),
        "slice_info": slice_info_list,
        "patches": patch_index,
    }

    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)

    with open(os.path.join(cache_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\n{'='*60}")
    print(f"预处理完成！")
    print(f"  缓存目录: {cache_dir}")
    print(f"  切片数: {meta['n_slices']}")
    print(f"  总 patch 数: {meta['n_patches']}")
    print(f"  输入维度: {meta['input_dim']}")
    for info in slice_info_list:
        print(f"    batch={info['batch']}: {info['n_cells']} cells, {info['n_patches']} patches")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    preprocess(cfg, force=args.force)