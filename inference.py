"""
独立推理脚本:
    python inference.py --config config.yaml --checkpoint out/.../checkpoints/best.pt
"""

import os
import argparse
import pickle
from pathlib import Path

import yaml
import numpy as np
import torch
import scanpy as sc

from stereotrack.mae import MAEEncoder

import scipy.sparse as sp

def parse_args():
    parser = argparse.ArgumentParser(description="推理")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_meta(cfg):
    """加载元信息（与 train_ddp.py 一致）"""
    cache_dir = os.path.join(cfg["paths"]["save_dir"], "cache")
    meta_pkl = os.path.join(cache_dir, "meta.pkl")
    meta_yaml = os.path.join(cache_dir, "meta.yaml")

    if os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as f:
            return pickle.load(f), cache_dir
    elif os.path.exists(meta_yaml):
        with open(meta_yaml, "r") as f:
            return yaml.safe_load(f), cache_dir
    else:
        raise FileNotFoundError(
            f"缓存不存在: {cache_dir}\n请先运行: python preprocess.py --config config.yaml"
        )


@torch.no_grad()
def inference():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = cfg["paths"]["save_dir"]
    adata_dir = os.path.join(save_dir, "adatas")
    
    adata_save_dir = os.path.join(save_dir, "inference_adatas")
    Path(adata_dir).mkdir(parents=True, exist_ok=True)
    Path(adata_save_dir).mkdir(parents=True, exist_ok=True)
    # ── 加载元信息 ──
    meta, cache_dir = load_meta(cfg)
    mcfg = cfg["model"]
    input_dim = meta["input_dim"]
    n_slices = meta["n_slices"]

    print(f"缓存目录: {cache_dir}")
    print(f"切片数: {n_slices}, 输入维度: {input_dim}")
    # ── 加载模型 ──
    model = MAEEncoder(
        input_dim=input_dim,
        hidden_dim=mcfg["hidden_dim"],
        latent_dim=mcfg["latent_dim"],
        dropout_rate=mcfg["dropout_rate"],
        gene_mask_ratio=mcfg["gene_mask_ratio"],
        cell_mask_ratio=mcfg["cell_mask_ratio"],
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"模型加载自: {args.checkpoint} (epoch {ckpt['epoch']}, loss {ckpt['best_loss']:.4f})")

    # ── 按切片分组 patch ──
    patches_by_slice = {}
    for p in meta["patches"]:
        s_idx = p["slice_idx"]
        if s_idx not in patches_by_slice:
            patches_by_slice[s_idx] = []
        patches_by_slice[s_idx].append(p)

    # 每个切片内按 start 排序，确保拼接顺序正确
    for s_idx in patches_by_slice:
        patches_by_slice[s_idx].sort(key=lambda x: x["start"])

    # ── 逐切片推理 ──
    for s_idx in range(n_slices):
        slice_info = meta["slice_info"][s_idx]
        patches = patches_by_slice.get(s_idx, [])
        n_cells = slice_info["n_cells"]

        print(f"\n切片 {s_idx} (z={slice_info['batch']}): {n_cells} cells, {len(patches)} patches")

        z_cell_list = []
        z_niche_list = []
        x_recon_list = []

        for p_idx, patch in enumerate(patches):
            # ── 从磁盘加载稀疏 patch ──
            fpath = os.path.join(cache_dir, patch["file"])
            data = np.load(fpath)

            feat = sp.csr_matrix(
                (data["feat_data"].astype(np.float32),
                 data["feat_indices"],
                 data["feat_indptr"]),
                shape=tuple(data["feat_shape"]),
            ).toarray()

            adj = sp.csr_matrix(
                (data["adj_data"],
                 data["adj_indices"],
                 data["adj_indptr"]),
                shape=tuple(data["adj_shape"]),
            ).toarray()

            feat = torch.from_numpy(feat).to(device)
            adj = torch.from_numpy(adj).to(device)

            # ── 前向推理 ──
            with torch.amp.autocast("cuda"):
                z_cell, z_niche = model.encode(feat, adj)
                x_recon = model.decoder(z_niche)

            z_cell_list.append(z_cell.cpu().numpy())
            z_niche_list.append(z_niche.cpu().numpy())
            x_recon_list.append(x_recon.cpu().numpy())

            del feat, adj, z_cell, z_niche, x_recon
            torch.cuda.empty_cache()

            if (p_idx + 1) % 10 == 0:
                print(f"  patch {p_idx+1}/{len(patches)} done")

        # ── 拼接结果 ──
        z_cell_np = np.concatenate(z_cell_list, axis=0)
        z_niche_np = np.concatenate(z_niche_list, axis=0)
        x_recon_np = np.concatenate(x_recon_list, axis=0)

        assert z_cell_np.shape[0] == n_cells, \
            f"切片 {s_idx} 细胞数不匹配: 拼接={z_cell_np.shape[0]}, 预期={n_cells}"

        # ── 加载原始 adata 拿 obs/var/spatial 信息 ──
        adata_cache = os.path.join(cache_dir, "adatas", f"slice_{s_idx}.h5ad")
        if os.path.exists(adata_cache):
            adata_orig = sc.read_h5ad(adata_cache)
        else:
            print(f"  警告: 未找到 {adata_cache}，创建空 AnnData")
            adata_orig = None

        # ── 构建输出 AnnData ──
        adata_out = sc.AnnData(X=x_recon_np)

        if adata_orig is not None:
            adata_out.obs = adata_orig.obs.copy()
            adata_out.var = adata_orig.var.copy()
            # 复制 spatial 等 obsm
            for key in adata_orig.obsm:
                if key != "cell_embedding" and key != "niche_embedding":
                    adata_out.obsm[key] = adata_orig.obsm[key]

        adata_out.obsm["cell_embedding"] = z_cell_np
        adata_out.obsm["niche_embedding"] = z_niche_np

        # ── 保存 ──
        out_path = os.path.join(adata_save_dir, f"slice_{s_idx}.h5ad")
        adata_out.write_h5ad(out_path)
        print(f"  → 已保存: {out_path} ({n_cells} cells)")

        del z_cell_list, z_niche_list, x_recon_list, adata_out
        if adata_orig is not None:
            del adata_orig

    print("\n推理全部完成！")
    print(f"输出目录: {adata_dir}")


if __name__ == "__main__":
    inference()