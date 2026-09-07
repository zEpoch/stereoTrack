"""
三物种独立推理脚本：
    python 03_inference_v1.py \
        --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/04_config_three_species.yaml \
        --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/checkpoints/last.ckpt

特点：
    - 不合并多个 h5ad
    - 每个 slice 单独输出一个 h5ad
    - 兼容 integrated/ 下的 cache 文件
    - 默认只保存 cell/niche embedding，避免 reconstruction layers 占满磁盘
"""
import gc
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pickle
from contextlib import nullcontext

import anndata as ad
import numpy as np
import scanpy as sc
import torch
import yaml

import scipy.sparse as sp

from stereotrack.dataset import normalize_meta, resolve_cache_file
from stereotrack.mae import MAEEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="三物种推理")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/04_config_three_species.yaml",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="启用 CUDA autocast。默认关闭，避免 sparse adjacency 推理时 dtype 不匹配。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="可选：推理 h5ad 输出目录；默认写到 cfg['paths']['save_dir']/inference_adatas",
    )
    parser.add_argument(
        "--write-recon-layers",
        action="store_true",
        help="写出 niche_recon/cell_recon layers。文件会非常大，通常不建议开启。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果输出 h5ad 已存在，则跳过该 slice，便于崩溃后续跑。",
    )
    return parser.parse_args()


def _strip_state_dict(raw):
    """去除 Lightning/DDP 前缀，并兼容旧版 decoder 命名。"""
    state = {}
    for key, value in raw.items():
        cleaned_key = key
        while cleaned_key.startswith("module.") or cleaned_key.startswith("model."):
            if cleaned_key.startswith("module."):
                cleaned_key = cleaned_key[len("module."):]
            elif cleaned_key.startswith("model."):
                cleaned_key = cleaned_key[len("model."):]

        if cleaned_key.startswith("decoder."):
            cleaned_key = "niche_decoder." + cleaned_key[len("decoder."):]
        elif cleaned_key.startswith("binary_decoder."):
            cleaned_key = "niche_binary_decoder." + cleaned_key[len("binary_decoder."):]

        # Prototype banks belong to the Lightning training objective, not MAEEncoder inference.
        if cleaned_key.startswith("alignment_"):
            continue

        state[cleaned_key] = value
    return state


def _extract_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if "state_dict" in ckpt:
        raw_dict = ckpt["state_dict"]
        epoch = ckpt.get("epoch", "未知")
        best_loss = ckpt.get("best_loss", "未知")
    elif "model_state" in ckpt:
        raw_dict = ckpt["model_state"]
        epoch = ckpt.get("epoch", "未知")
        best_loss = ckpt.get("best_loss", "未知")
    else:
        raw_dict = ckpt
        epoch = "未知"
        best_loss = ckpt.get("best_loss", "未知") if isinstance(ckpt, dict) else "未知"
    return _strip_state_dict(raw_dict), epoch, best_loss


def load_meta(cfg):
    cache_dir = os.path.join(cfg["paths"]["input_dir"], "cache")
    meta_pkl = os.path.join(cache_dir, "meta.pkl")
    meta_yaml = os.path.join(cache_dir, "meta.yaml")

    if os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as handle:
            return normalize_meta(pickle.load(handle)), cache_dir
    if os.path.exists(meta_yaml):
        with open(meta_yaml, "r") as handle:
            return normalize_meta(yaml.safe_load(handle)), cache_dir
    raise FileNotFoundError(f"缓存不存在: {cache_dir}\n请先运行: python 01_preprocess.py --config config.yaml")


def find_cached_adata(cache_dir, species, slice_idx, batch_name=None, species_slice_idx=None):
    batch_stem = Path(batch_name).stem if batch_name is not None else None
    candidates = [
        os.path.join(cache_dir, "adatas", f"{species}_{batch_stem}.h5ad") if batch_stem else None,
        os.path.join(cache_dir, "integrated", "adatas", f"{species}_{batch_stem}.h5ad") if batch_stem else None,
        os.path.join(cache_dir, "adatas", f"{species}_slice_{species_slice_idx}.h5ad") if species_slice_idx is not None else None,
        os.path.join(cache_dir, "integrated", "adatas", f"{species}_slice_{species_slice_idx}.h5ad") if species_slice_idx is not None else None,
        os.path.join(cache_dir, "adatas", f"{species}_slice_{slice_idx}.h5ad"),
        os.path.join(cache_dir, "integrated", "adatas", f"{species}_slice_{slice_idx}.h5ad"),
    ]
    for path in candidates:
        if path is not None and os.path.exists(path):
            return path
    return None


def make_light_adata(adata_orig, n_cells, input_dim, write_recon_layers):
    n_vars = input_dim if write_recon_layers else 0
    x_empty = sp.csr_matrix((n_cells, n_vars), dtype=np.float32)

    if adata_orig is None:
        return ad.AnnData(x_empty)

    obs = adata_orig.obs.copy()
    if write_recon_layers and adata_orig.n_vars == input_dim:
        adata_out = ad.AnnData(x_empty, obs=obs, var=adata_orig.var.copy())
    elif not write_recon_layers:
        adata_out = ad.AnnData(x_empty, obs=obs, var=adata_orig.var.iloc[:0].copy())
    else:
        adata_out = ad.AnnData(x_empty, obs=obs)

    for key in adata_orig.obsm:
        if key not in {"cell_embedding", "niche_embedding", "adj_norm", "spatial_input"}:
            adata_out.obsm[key] = adata_orig.obsm[key]
    return adata_out


def autocast_for_device(device, enabled=False):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def scipy_to_torch_sparse(matrix, device):
    matrix = matrix.tocoo()
    indices_np = np.vstack([matrix.row, matrix.col]).astype(np.int64)
    indices = torch.from_numpy(indices_np).to(device)
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False)).to(device)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=device).coalesce()


@torch.no_grad()
def inference():
    args = parse_args()
    with open(args.config, "r") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = cfg["paths"]["save_dir"]
    adata_save_dir = args.output_dir or os.path.join(save_dir, "inference_adatas")
    Path(adata_save_dir).mkdir(parents=True, exist_ok=True)

    meta, cache_dir = load_meta(cfg)
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})
    input_dim = meta["input_dim"]
    n_slices = meta["n_slices"]
    batch_size = args.batch_size

    print(f"缓存目录: {cache_dir}")
    print(f"切片数: {n_slices}, 输入维度: {input_dim}")
    print(f"输出模式: {'embedding + reconstruction layers' if args.write_recon_layers else 'embedding only'}")

    state_dict, epoch, best_loss = _extract_checkpoint(args.checkpoint)

    model = MAEEncoder(
        input_dim=input_dim,
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        dropout_rate=model_cfg["dropout_rate"],
        gene_mask_ratio=model_cfg["gene_mask_ratio"],
        cell_mask_ratio=model_cfg["cell_mask_ratio"],
        n_encoder_layers=model_cfg.get("n_encoder_layers", 2),
        n_decoder_layers=model_cfg.get("n_decoder_layers", 1),
        niche_self_weight=float(model_cfg.get("niche_self_weight", 0.5)),
        lambda_cell_recon=float(train_cfg.get("lambda_cell_recon", 1.0)),
        lambda_niche_recon=float(train_cfg.get("lambda_niche_recon", 1.0)),
        lambda_cell_mask=float(train_cfg.get("lambda_cell_mask", 1.0)),
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        print("严格加载 checkpoint 失败，尝试 strict=False 兼容加载。")
        print(f"原始错误: {error}")
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

    species_slice_counts = {}
    for slice_idx in range(n_slices):
        slice_info = meta["slice_info"][slice_idx]
        n_cells = int(slice_info["n_cells"])
        species = slice_info.get("species", "unknown")
        batch_name = slice_info.get("batch", f"slice_{slice_idx}")
        species_slice_idx = species_slice_counts.get(species, 0)
        species_slice_counts[species] = species_slice_idx + 1
        output_name = f"{species}_{Path(batch_name).stem}.h5ad"
        out_path = os.path.join(adata_save_dir, output_name)

        if args.skip_existing and os.path.exists(out_path):
            print(f"\n切片 {slice_idx} ({species}, {batch_name}): 输出已存在，跳过 {out_path}")
            continue

        print(f"\n切片 {slice_idx} ({species}, {batch_name}): 全图 {n_cells} cells")

        fpath = resolve_cache_file(cache_dir, slice_info["file"])
        with np.load(fpath) as data:
            feat_sparse = sp.csr_matrix(
                (
                    data["feat_data"].astype(np.float32),
                    data["feat_indices"].astype(np.int64),
                    data["feat_indptr"].astype(np.int64),
                ),
                shape=tuple(data["feat_shape"]),
            )
            adj_sparse = sp.csr_matrix(
                (
                    data["adj_data"].astype(np.float32),
                    data["adj_indices"].astype(np.int64),
                    data["adj_indptr"].astype(np.int64),
                ),
                shape=tuple(data["adj_shape"]),
            )

        z_cell_list, z_niche_list = [], []
        x_niche_recon_list, x_cell_recon_list = [], []

        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            core_idx = np.arange(start, end)
            neighbors = adj_sparse[core_idx].nonzero()[1]
            sub_nodes = np.unique(np.concatenate([core_idx, neighbors]))
            core_mask = np.isin(sub_nodes, core_idx)

            feat_sub = feat_sparse[sub_nodes].toarray()
            adj_sub = adj_sparse[sub_nodes][:, sub_nodes].tocsr()
            feat_ts = torch.from_numpy(feat_sub).to(device)
            adj_ts = scipy_to_torch_sparse(adj_sub, device)

            with autocast_for_device(device, enabled=args.use_amp):
                z_cell, z_niche = model.encode(feat_ts, adj_ts)
                if args.write_recon_layers:
                    x_niche_recon = model.niche_decoder(z_niche)
                    x_cell_recon = model.cell_decoder(z_cell)

            z_cell_list.append(z_cell[core_mask].cpu().numpy())
            z_niche_list.append(z_niche[core_mask].cpu().numpy())
            if args.write_recon_layers:
                x_niche_recon_list.append(x_niche_recon[core_mask].cpu().numpy())
                x_cell_recon_list.append(x_cell_recon[core_mask].cpu().numpy())

            del feat_ts, adj_ts, z_cell, z_niche
            if args.write_recon_layers:
                del x_niche_recon, x_cell_recon
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"  预测进度: {end}/{n_cells} cells (subgraph={len(sub_nodes)}, nnz={adj_sub.nnz})")

        z_cell_np = np.concatenate(z_cell_list, axis=0).astype(np.float32)
        z_niche_np = np.concatenate(z_niche_list, axis=0).astype(np.float32)

        adata_cache = find_cached_adata(cache_dir, species, slice_idx, batch_name, species_slice_idx)
        adata_orig = sc.read_h5ad(adata_cache) if adata_cache is not None else None
        if adata_orig is None:
            print("  [warn] 未找到原始 slice h5ad，只保存 embedding 和基础 obs")

        adata_out = make_light_adata(adata_orig, n_cells, input_dim, args.write_recon_layers)

        adata_out.obsm["cell_embedding"] = z_cell_np
        adata_out.obsm["niche_embedding"] = z_niche_np
        if args.write_recon_layers:
            adata_out.layers["niche_recon"] = np.concatenate(x_niche_recon_list, axis=0).astype(np.float32)
            adata_out.layers["cell_recon"] = np.concatenate(x_cell_recon_list, axis=0).astype(np.float32)

        if "species" not in adata_out.obs:
            adata_out.obs["species"] = species
        if "batch" not in adata_out.obs:
            adata_out.obs["batch"] = batch_name
        adata_out.obs["slice_idx"] = slice_idx
        adata_out.uns["stereotrack_inference"] = {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": str(epoch),
            "model_variant": Path(save_dir).name,
            "embedding_only": str(not args.write_recon_layers),
        }

        adata_out.write_h5ad(out_path)
        print(f"  → 已保存: {out_path}")

        del z_cell_list, z_niche_list, x_niche_recon_list, x_cell_recon_list, adata_out, adata_orig
        del feat_sparse, adj_sparse, z_cell_np, z_niche_np
        gc.collect()

    print(f"\n所有切片已逐个保存到: {adata_save_dir}")


if __name__ == "__main__":
    inference()
