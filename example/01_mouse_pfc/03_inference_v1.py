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

def _strip_state_dict(raw):
    """去除 Lightning/DDP 前缀，并兼容旧版 decoder 命名。"""
    state = {}
    for k, v in raw.items():
        kk = k
        while kk.startswith("module.") or kk.startswith("model."):
            if kk.startswith("module."):
                kk = kk[len("module."):]
            elif kk.startswith("model."):
                kk = kk[len("model."):]

        # 兼容旧 checkpoint: decoder/binary_decoder -> niche_decoder/niche_binary_decoder
        if kk.startswith("decoder."):
            kk = "niche_decoder." + kk[len("decoder."):]
        elif kk.startswith("binary_decoder."):
            kk = "niche_binary_decoder." + kk[len("binary_decoder."):]

        state[kk] = v
    return state

def _extract_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        raw_dict, epoch = ckpt["state_dict"], ckpt.get("epoch", "未知")
        best_loss = ckpt.get("best_loss", "未知")
    elif "model_state" in ckpt:
        raw_dict, epoch = ckpt["model_state"], ckpt.get("epoch", "未知")
        best_loss = ckpt.get("best_loss", "未知")
    else:
        raw_dict, epoch, best_loss = ckpt, "未知", ckpt.get("best_loss", "未知") if isinstance(ckpt, dict) else "未知"
    return _strip_state_dict(raw_dict), epoch, best_loss

def _has_dmt_weight(state_dict):
    return any(("dmt_" in k or "dmt_cell" in k or "dmt_niche" in k) for k in state_dict.keys())

def load_meta(cfg):
    """加载元信息（与 train_ddp.py 一致）"""
    cache_dir = os.path.join(cfg["paths"]["input_dir"], "cache")
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
    adata_save_dir = os.path.join(save_dir, "inference_adatas")
    Path(adata_save_dir).mkdir(parents=True, exist_ok=True)

    meta, cache_dir = load_meta(cfg)
    mcfg = cfg["model"]
    tcfg = cfg.get("training", {})
    input_dim = meta["input_dim"]
    n_slices = meta["n_slices"]
    batch_size = args.batch_size  # 推理时的批次大小

    print(f"缓存目录: {cache_dir}")
    print(f"切片数: {n_slices}, 输入维度: {input_dim}")
    
    state_dict, epoch, best_loss = _extract_checkpoint(args.checkpoint)

    model = MAEEncoder(
        input_dim=input_dim,
        hidden_dim=mcfg["hidden_dim"],
        latent_dim=mcfg["latent_dim"],
        dropout_rate=mcfg["dropout_rate"],
        gene_mask_ratio=mcfg["gene_mask_ratio"],
        cell_mask_ratio=mcfg["cell_mask_ratio"],
        n_encoder_layers=mcfg.get("n_encoder_layers", 2),
        n_decoder_layers=mcfg.get("n_decoder_layers", 1),
        niche_self_weight=float(mcfg.get("niche_self_weight", 0.5)),
        lambda_cell_recon=float(tcfg.get("lambda_cell_recon", 1.0)),
        lambda_niche_recon=float(tcfg.get("lambda_niche_recon", 1.0)),
        lambda_cell_mask=float(tcfg.get("lambda_cell_mask", 1.0)),

    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("严格加载 checkpoint 失败，尝试 strict=False 兼容加载。")
        print(f"原始错误: {e}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"缺失参数: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"多余参数: {incompatible.unexpected_keys}")

    model.to(device)
    model.eval()
    if isinstance(best_loss, float):
        print(f"模型加载自: {args.checkpoint} (epoch {epoch}, loss {best_loss:.4f})")
    else:
        print(f"模型加载自: {args.checkpoint} (epoch {epoch}, loss {best_loss})")
    adatas = []
    # ── 逐切片推理（使用 Core-Halo 无缝子图策略） ──
    for s_idx in range(n_slices):
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

        z_cell_list, z_niche_list = [], []
        x_niche_recon_list, x_cell_recon_list = [], []

        # 2. 按顺序分块推理，带着各自的真实邻近边界（Halo）一起放入模型
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

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                z_cell, z_niche = model.encode(feat_ts, adj_ts)
                x_niche_recon = model.niche_decoder(z_niche)
                x_cell_recon = model.cell_decoder(z_cell)
            # 【魔法在这里】：算完之后，只剔出 Core 核心细胞的结果（抛弃辅助边缘细胞）！
            # 这样拼贴起来，没有任何截断缝隙！
            z_cell_list.append(z_cell[core_mask].cpu().numpy())
            z_niche_list.append(z_niche[core_mask].cpu().numpy())
            x_niche_recon_list.append(x_niche_recon[core_mask].cpu().numpy())
            x_cell_recon_list.append(x_cell_recon[core_mask].cpu().numpy())
            


            del feat_ts, adj_ts, z_cell, z_niche, x_niche_recon, x_cell_recon

            torch.cuda.empty_cache()

            print(f"  预测进度: {end}/{n_cells} cells")

        z_cell_np = np.concatenate(z_cell_list, axis=0)
        z_niche_np = np.concatenate(z_niche_list, axis=0)
        x_niche_recon_np = np.concatenate(x_niche_recon_list, axis=0)
        x_cell_recon_np = np.concatenate(x_cell_recon_list, axis=0)


        # 4. 写入原图 AnnData
        adata_cache = os.path.join(cache_dir, "adatas", f"slice_{s_idx}.h5ad")
        adata_orig = sc.read_h5ad(adata_cache) if os.path.exists(adata_cache) else None
        
        # 默认将 niche 分支重建结果放在 X 中；cell 分支重建结果额外保存在 layers。
        adata_out = sc.AnnData(adata_orig.X.astype(np.float32))
        if adata_orig is not None:
            adata_out.raw = adata_orig.copy()
        
        if adata_orig is not None:
            adata_out.obs = adata_orig.obs.copy()
            adata_out.var = adata_orig.var.copy()
            for key in adata_orig.obsm:
                if key not in ["cell_embedding", "niche_embedding", 'adj_norm', 'spatial_input']:
                    adata_out.obsm[key] = adata_orig.obsm[key]

        adata_out.obsm["cell_embedding"] = z_cell_np
        adata_out.obsm["niche_embedding"] = z_niche_np
        adata_out.layers["niche_recon"] = x_niche_recon_np.astype(np.float32)
        adata_out.layers["cell_recon"] = x_cell_recon_np.astype(np.float32)

        out_path = os.path.join(adata_save_dir, f"slice_{s_idx}.h5ad")
        adata_out.write_h5ad(out_path)
        
        print(f"  → 已保存: {out_path}")
        adatas.append(adata_out)
        del z_cell_list, z_niche_list, x_niche_recon_list, x_cell_recon_list, adata_out, adata_orig
        gc.collect()

    adata = sc.concat(adatas, axis=0)
    adata_out_path = os.path.join(save_dir, "inference_result.h5ad")
    adata.write_h5ad(adata_out_path)
    # 保存 cell emebdding 和 niche embedding 为单独的 npy 文件，方便后续分析使用
    np.save(os.path.join(save_dir, "cell_embedding.npy"), adata.obsm["cell_embedding"])
    np.save(os.path.join(save_dir, "niche_embedding.npy"), adata.obsm["niche_embedding"])
    print(f"\n所有切片已合并并保存: {adata_out_path}")

if __name__ == "__main__":
    inference()



# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v12_2.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12_2/checkpoints/last.ckpt
'''
python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/03_inference_v1.py \
    --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v1.yaml \
    --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_merfish_pfc_v1/checkpoints/last-v4.ckpt
'''

'''
python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/03_inference_v1.py \
    --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v4.yaml \
    --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_merfish_pfc_v4/checkpoints/last-v2.ckpt 
'''
