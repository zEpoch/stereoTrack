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
import csv
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
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--genes", nargs="*", default=None, help="Optional gene subset to write.")
    parser.add_argument("--gene-file", type=str, default=None, help="Optional text file with one gene per line.")
    parser.add_argument("--all-genes", action="store_true", help="Write all macaque common genes.")
    parser.add_argument("--embedding-only", action="store_true", help="Only write obsm['niche_embedding']; skip decoder and expression output.")
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--embedding-output-format", choices=["h5ad", "npy"], default="h5ad")
    parser.add_argument("--compression", choices=["none", "lzf", "gzip"], default="lzf")
    parser.add_argument("--skip-existing", action="store_true", help="Skip non-empty slice output files.")
    parser.add_argument("--start-slice", type=int, default=0)
    parser.add_argument("--end-slice", type=int, default=None, help="Exclusive end slice index.")
    return parser.parse_args()


def append_manifest(manifest_path, row):
    fieldnames = [
        "slice_idx",
        "batch",
        "n_cells",
        "n_dim",
        "dtype",
        "embedding_path",
        "obs_names_path",
    ]
    write_header = not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0
    with open(manifest_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def output_prefix_from_slice_info(slice_info):
    batch = str(slice_info.get("batch", ""))
    stem = Path(batch).stem
    return stem or f"slice_{slice_info.get('slice_idx', 'unknown')}"


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
# ...existing code...


def read_gene_file(path):
    genes = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                genes.extend(item.strip() for item in line.split(",") if item.strip())
    return genes


def requested_genes(args, meta):
    common_genes = list(meta.get("common_genes") or meta.get("common_human_genes") or [])
    if args.all_genes:
        return common_genes
    genes = []
    if args.gene_file:
        genes.extend(read_gene_file(args.gene_file))
    if args.genes:
        genes.extend(args.genes)
    if genes:
        return list(dict.fromkeys(genes))
    return [
        "SLC17A7", "GPR83", "CCBE1", "CUX2", "GPC5", "PDZD2", "CUX1", "MYLK", "PDCH1",
        "RORB", "IL1RAPL2", "ETV1", "TLE4", "SEMA3E", "GAD1", "GAD2", "ADARB2", "LAMP5",
        "FBXL7", "KIT", "EYA4", "CALB2", "RELN", "VIP", "SOX6", "TRPS1", "ADAMTSL1",
        "PVALB", "POSTN", "SST", "CALB1", "SLC1A2", "SLC1A3", "PTPRZ1", "PDGFRA",
        "COL9A1", "PLP1", "ITGAM", "RGS5", "COL1A2",
    ]

@torch.no_grad()
def inference():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = cfg["paths"]["save_dir"]
    adata_dir = os.path.join(save_dir, "adatas")
    adata_save_dir = args.output_dir or os.path.join(save_dir, "inference_adatas")
    Path(adata_dir).mkdir(parents=True, exist_ok=True)
    Path(adata_save_dir).mkdir(parents=True, exist_ok=True)

    meta, cache_dir = load_meta(cfg)
    mcfg = cfg["model"]
    input_dim = meta["input_dim"]
    n_slices = meta["n_slices"]
    batch_size = args.batch_size  # 推理时的批次大小

    print(f"缓存目录: {cache_dir}")
    print(f"切片数: {n_slices}, 输入维度: {input_dim}")
    output_genes = requested_genes(args, meta)
    if args.embedding_only:
        print(
            "输出模式: embedding-only, "
            f"format={args.embedding_output_format}, niche_embedding dtype={args.embedding_dtype}"
        )
    else:
        print(f"输出基因数: {len(output_genes)}")

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
    
    # 兼容 DDP 保存的 "module." 前缀
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
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"[warn] strict checkpoint load failed, retry with strict=False: {exc}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[warn] missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warn] unexpected keys: {incompatible.unexpected_keys}")
    model.to(device)
    model.eval()
    best_loss = ckpt.get("best_loss", "未知")
    if isinstance(best_loss, float):
        print(f"模型加载自: {args.checkpoint} (epoch {epoch}, loss {best_loss:.4f})")
    else:
        print(f"模型加载自: {args.checkpoint} (epoch {epoch}, loss {best_loss})")
    
    # ── 逐切片推理（使用 Core-Halo 无缝子图策略） ──
    end_slice = n_slices if args.end_slice is None else min(args.end_slice, n_slices)
    for s_idx in range(args.start_slice, end_slice):
        slice_info = meta["slice_info"][s_idx]
        n_cells = slice_info["n_cells"]
        output_prefix = output_prefix_from_slice_info(slice_info)
        if args.embedding_only and args.embedding_output_format == "npy":
            out_path = os.path.join(adata_save_dir, f"{output_prefix}.niche_embedding.npy")
        else:
            out_path = os.path.join(adata_save_dir, f"{output_prefix}.h5ad")
        if args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"\n[skip] 切片 {s_idx}: {out_path} exists")
            continue

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

        # 2. 按顺序分块推理，带着各自的真实邻近边界（Halo）一起放入模型
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            
            # 【重点】：我们真正需要预测结果的“核心细胞”索引
            core_idx = np.arange(start, end)
            
            # 【重点】：找出它们所有的连通邻居
            neighbors = adj_sparse[core_idx].nonzero()[1]
            
            # 将核心细胞 + 边缘邻居组合成局部的、保持完美原貌的子图（Subgraph）
            sub_nodes = np.unique(np.concatenate([core_idx, neighbors]))
            
            # 生成遮罩：子图中哪些是刚才选中的 Core？
            core_mask = np.isin(sub_nodes, core_idx)

            # 抽取真正的子图特征与连边
            feat_sub = feat_sparse[sub_nodes].toarray()
            adj_sub = adj_sparse[sub_nodes][:, sub_nodes].toarray()

            feat_ts = torch.from_numpy(feat_sub).to(device)
            adj_ts = torch.from_numpy(adj_sub).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                z_cell, z_niche = model.encode(feat_ts, adj_ts)
                x_recon = None if args.embedding_only else model.niche_decoder(z_niche)

            # 【魔法在这里】：算完之后，只剔出 Core 核心细胞的结果（抛弃辅助边缘细胞）！
            # 这样拼贴起来，没有任何截断缝隙！
            if not args.embedding_only:
                z_cell_list.append(z_cell[core_mask].cpu().numpy())
            z_niche_list.append(z_niche[core_mask].cpu().numpy())
            if x_recon is not None:
                x_recon_list.append(x_recon[core_mask].cpu().numpy())

            del feat_ts, adj_ts, z_cell, z_niche, x_recon
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"  预测进度: {end}/{n_cells} cells")

        # 3. 无缝拼接结果（保证长度必定等于 n_cells）
        z_niche_np = np.concatenate(z_niche_list, axis=0)
        if args.embedding_dtype == "float16":
            z_niche_np = z_niche_np.astype(np.float16, copy=False)
        else:
            z_niche_np = z_niche_np.astype(np.float32, copy=False)
        if not args.embedding_only:
            z_cell_np = np.concatenate(z_cell_list, axis=0)
            x_recon_np = np.concatenate(x_recon_list, axis=0)

        adata_cache = os.path.join(cache_dir, "adatas", f"slice_{s_idx}.h5ad")
        if args.embedding_only and args.embedding_output_format == "npy":
            np.save(out_path, z_niche_np)
            obs_names_path = os.path.join(adata_save_dir, f"{output_prefix}.obs_names.txt")
            if os.path.exists(adata_cache):
                adata_orig = sc.read_h5ad(adata_cache)
                with open(obs_names_path, "w") as handle:
                    handle.write("\n".join(map(str, adata_orig.obs_names)) + "\n")
                del adata_orig
            else:
                obs_names_path = ""
            append_manifest(
                os.path.join(adata_save_dir, "niche_embedding_manifest.csv"),
                {
                    "slice_idx": s_idx,
                    "batch": slice_info["batch"],
                    "n_cells": int(z_niche_np.shape[0]),
                    "n_dim": int(z_niche_np.shape[1]),
                    "dtype": str(z_niche_np.dtype),
                    "embedding_path": out_path,
                    "obs_names_path": obs_names_path,
                },
            )
            print(f"  → 已保存 embedding: {out_path}, shape={z_niche_np.shape}, dtype={z_niche_np.dtype}")
            del z_niche_list, x_recon_list, z_niche_np, feat_sparse, adj_sparse
            gc.collect()
            continue

        # 4. 写入原图 AnnData
        adata_orig = sc.read_h5ad(adata_cache) if os.path.exists(adata_cache) else None
        if args.embedding_only:
            adata_out = sc.AnnData(X=sp.csr_matrix((n_cells, 0), dtype=np.float32))
            adata_out.var = adata_out.var.iloc[:0].copy()
        else:
            adata_out = sc.AnnData(X=x_recon_np)

        if adata_orig is not None:
            adata_out.obs = adata_orig.obs.copy()
            if not args.embedding_only:
                adata_out.var = adata_orig.var.copy()
            for key in adata_orig.obsm:
                if key not in ["cell_embedding", "niche_embedding", 'adj_norm', 'spatial', 'spatial_input']:
                    adata_out.obsm[key] = adata_orig.obsm[key]

        if not args.embedding_only:
            adata_out.obsm["cell_embedding"] = z_cell_np
        adata_out.obsm["niche_embedding"] = z_niche_np

        if adata_orig is not None and not args.embedding_only:
            # 取交集以免个别基因在当前切片或数据集中不存在而报错
            valid_genes = [g for g in output_genes if g in adata_out.var_names]
            adata_out = adata_out[:, valid_genes].copy()
            print(f"  输出 {len(valid_genes)} 个基因")
        # =========================================================================

        if not args.embedding_only:
            with open(os.path.join(adata_save_dir, "genes_used_for_stereotrack_output.txt"), "w") as handle:
                handle.write("\n".join(map(str, adata_out.var_names)) + "\n")

        if args.compression == "none":
            adata_out.write_h5ad(out_path)
        else:
            adata_out.write_h5ad(out_path, compression=args.compression)
        print(f"  → 已保存: {out_path}")

        del z_niche_list, x_recon_list, adata_out, adata_orig
        if not args.embedding_only:
            del z_cell_list, z_cell_np, x_recon_np
        del z_niche_np
        gc.collect()

    print("\n推理全部完成！")

if __name__ == "__main__":
    inference()

# CUDA_VISIBLE_DEVICES=1 python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v2.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v2/checkpoints/last.ckpt
# nohup python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v2.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v2/checkpoints/last.ckpt > inference.log 2>&1 &
# nohup python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v4.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v4/checkpoints/best-epoch=43-train_loss_epoch=1.1247.ckpt > inference.log 2>&1 &
# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v4.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v4/checkpoints/best-epoch=43-train_loss_epoch=1.1247.ckpt


# python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/inference.py --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v12_2.yaml --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12_2/checkpoints/last.ckpt
