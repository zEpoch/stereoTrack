"""
独立推理脚本:
    python inference.py --config config.yaml --checkpoint out/.../checkpoints/best.pt
"""
import gc
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    cache_dir = os.path.join(cfg["paths"]["input_dir"], "cache")
    meta_yaml = os.path.join(cache_dir, "meta.yaml")
    meta_pkl = os.path.join(cache_dir, "meta.pkl")

    
    if os.path.exists(meta_yaml):
        with open(meta_yaml, "r") as f:
            return yaml.safe_load(f), cache_dir
    elif os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as f:
            return pickle.load(f), cache_dir
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
    adata_save_dir = os.path.join(save_dir, "inference_adatas")
    Path(adata_save_dir).mkdir(parents=True, exist_ok=True)

    meta, cache_dir = load_meta(cfg)
    mcfg = cfg["model"]
    input_dim = meta["input_dim"]
    n_slices = meta["n_slices"]
    batch_size = args.batch_size

    print(f"缓存目录: {cache_dir}")
    print(f"切片数: {n_slices}, 输入维度: {input_dim}")

    model = MAEEncoder(
        input_dim=input_dim,
        hidden_dim=mcfg["hidden_dim"],
        latent_dim=mcfg["latent_dim"],
        dropout_rate=mcfg["dropout_rate"],
        gene_mask_ratio=mcfg["gene_mask_ratio"],
        cell_mask_ratio=mcfg["cell_mask_ratio"],
        n_encoder_layers=mcfg.get("n_encoder_layers", 2),
        n_decoder_layers=mcfg.get("n_decoder_layers", 1),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # === 兼容 PyTorch Lightning 和传统 PyTorch 保存的检查点格式 ===
    if "state_dict" in ckpt:
        raw_dict = ckpt["state_dict"]  # Lightning 格式
        epoch = ckpt.get("epoch", "未知")
    elif "model_state" in ckpt:
        raw_dict = ckpt["model_state"] # 传统格式
        epoch = ckpt.get("epoch", "未知")
    else:
        raw_dict = ckpt # 纯 state_dict 的情况
        epoch = "未知"

    # 去除可能存在的 Lightning 'model.' 或 DDP 'module.' 前缀
    state_dict = {}
    for k, v in raw_dict.items():
        new_k = k.replace("module.", "").replace("model.", "")
        state_dict[new_k] = v

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    best_loss = ckpt.get("best_loss", "未知")
    if isinstance(best_loss, float):
        print(f"模型加载自: {args.checkpoint} (epoch {epoch}, loss {best_loss:.4f})")
    else:
        print(f"模型加载自: {args.checkpoint} (epoch {epoch}, loss {best_loss})")

    # ── 逐切片推理（使用 Core-Halo 无缝子图策略） ──
    for s_idx in range(n_slices):
        # if s_idx != 61: continue
        slice_info = meta["slice_info"][s_idx]
        n_cells = slice_info["n_cells"]

        print(f"\n切片 {s_idx} (z={slice_info['batch']}): 全图 {n_cells} cells")

        # 1. 加载全图数据 (由于前面已是稀疏保存，因此内存占用很小)
        fpath = os.path.join(cache_dir, slice_info["file"])
        # data = np.load(fpath)
        with np.load(fpath) as data:
            feat_sparse = sp.csr_matrix(
                    (data["feat_data"].astype(np.float32), 
                        data["feat_indices"].astype(np.int64), 
                        data["feat_indptr"].astype(np.int64)),
                    shape=tuple(data["feat_shape"]),
            )
            adj_sparse = sp.csr_matrix(
                (data["adj_data"].astype(np.float32), 
                 data["adj_indices"].astype(np.int64), 
                 data["adj_indptr"].astype(np.int64)),
                shape=tuple(data["adj_shape"]),
            )

        z_cell_list, z_niche_list, x_recon_list = [], [], []

        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            
            core_idx = np.arange(start, end)
            
            neighbors = adj_sparse[core_idx].nonzero()[1]
            
            sub_nodes = np.unique(np.concatenate([core_idx, neighbors]))
            
            core_mask = np.isin(sub_nodes, core_idx)

            feat_sub = feat_sparse[sub_nodes].toarray()
            adj_sub = adj_sparse[sub_nodes][:, sub_nodes].toarray()

            feat_ts = torch.from_numpy(feat_sub).to(device)
            adj_ts = torch.from_numpy(adj_sub).to(device)

            with torch.amp.autocast("cuda"):
                z_cell, z_niche = model.encode(feat_ts, adj_ts)
                x_recon = model.decoder(z_niche)

            z_cell_list.append(z_cell[core_mask].cpu().numpy())
            z_niche_list.append(z_niche[core_mask].cpu().numpy())
            x_recon_list.append(x_recon[core_mask].cpu().numpy())

            del feat_ts, adj_ts, z_cell, z_niche, x_recon
            torch.cuda.empty_cache()

            print(f"  预测进度: {end}/{n_cells} cells")

        z_cell_np = np.concatenate(z_cell_list, axis=0)
        z_niche_np = np.concatenate(z_niche_list, axis=0)
        x_recon_np = np.concatenate(x_recon_list, axis=0)

        # 4. 写入原图 AnnData
        real_slice_name = slice_info["batch"]  # 这里面的值是 "total_gene_T239_mouse_f001..." 等
        
        # 【修改】：预处理脚本里写入 adata 的名称是 "slice_{i}.h5ad"，并非原数据长名称
        # 我们需要在 metadata 里面把它的 slice_idx 取出来，而不是用 batch 名
        orig_h5ad_path = os.path.join(cache_dir, "adatas", f"slice_{s_idx}.h5ad")
        
        adata_orig = sc.read_h5ad(orig_h5ad_path, backed='r')

        adata_out = sc.AnnData(X=x_recon_np)

        if adata_orig is not None:
            adata_out.obs = adata_orig.obs.copy()
            adata_out.var = adata_orig.var.copy()
            for key in adata_orig.obsm:
                if key not in ["cell_embedding", "niche_embedding"]:
                    adata_out.obsm[key] = adata_orig.obsm[key]

        adata_out.obsm["cell_embedding"] = z_cell_np
        adata_out.obsm["niche_embedding"] = z_niche_np

        out_path = os.path.join(adata_save_dir, real_slice_name)
        if not out_path.endswith('.h5ad'):
            out_path += '.h5ad'
            
        adata_out.write_h5ad(out_path)
        print(f"  → 已保存: {out_path}")

        del z_cell_list, z_niche_list, x_recon_list, adata_out, adata_orig
        gc.collect()

    print("\n推理全部完成！")

if __name__ == "__main__":
    inference()

# python inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/config_hanMouse3D_v2.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae/checkpoints/best.pt
# CUDA_VISIBLE_DEVICES=1 python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v3.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v3/checkpoints/best-epoch=20-train_loss=1.3307.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v7.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v7/checkpoints/best-epoch=48-train_loss_epoch=1.3202.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v8.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v8/checkpoints/best-epoch=48-train_loss_epoch=1.3233.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v9.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v9/checkpoints/best-epoch=48-train_loss_epoch=1.3207.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v10.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v10/checkpoints/best-epoch=48-train_loss_epoch=1.3211.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v11.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v11/checkpoints/best-epoch=48-train_loss_epoch=1.3209.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v12.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12/checkpoints/best-epoch=48-train_loss_epoch=1.3208.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v12.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12/checkpoints/best-epoch=48-train_loss_epoch=1.3208.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae/checkpoints/best.pt


# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v12_2.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12_2/checkpoints/last.ckpt

# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_Han_Neuron/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_hanMouse3D_v12_3.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12_3/checkpoints/best-epoch=23-train_loss_epoch=1.3211.ckpt