"""
纯训练脚本（内存优化版），按需加载 patch:
  先运行: python preprocess.py --config config.yaml
  再训练:
    单机单卡: torchrun --nproc_per_node=1 train_ddp.py --config config.yaml
    单机多卡: torchrun --nproc_per_node=4 train_ddp.py --config config.yaml
    多机多卡: torchrun --nnodes=2 --nproc_per_node=4 --rdzv_endpoint=MASTER:29500 train_ddp.py --config config.yaml
"""

import os
import gc
import argparse
import pickle
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stereotrack.mae import MAEEncoder

import scipy.sparse as sp

# ═══════════════════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="DDP 训练")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.batch_size is not None:
        cfg["training"]["patch_size"] = args.batch_size
    if args.n_epochs is not None:
        cfg["training"]["n_epochs"] = args.n_epochs
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# 2. 分布式工具
# ═══════════════════════════════════════════════════════════════════════════
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 加载元信息（很小，几 KB）
# ═══════════════════════════════════════════════════════════════════════════
def load_meta(cfg):
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
            f"缓存不存在: {cache_dir}\n"
            f"请先运行: python preprocess.py --config config.yaml"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Dataset：懒加载，每次 __getitem__ 从磁盘读一个 npz
# ═══════════════════════════════════════════════════════════════════════════
'''
class LazyPatchDataset(Dataset):
    """
    不把数据全部载入内存，每次 __getitem__ 从磁盘读取一个 patch 的 npz。
    单个 patch (4096 cells × ~30000 genes) 约 500MB dense，
    但 npz 压缩后通常只有几十 MB，读取速度可接受。
    """

    def __init__(self, cache_dir, patches):
        """
        Args:
            cache_dir: 缓存目录路径
            patches: meta["patches"] 列表
        """
        self.cache_dir = cache_dir
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        info = self.patches[idx]
        fpath = os.path.join(self.cache_dir, info["file"])
        data = np.load(fpath)
        return {
            "features": torch.from_numpy(data["features"]),
            "adj": torch.from_numpy(data["adj"]),
            "slice_idx": info["slice_idx"],
        }
'''
class LazyPatchDataset(Dataset):
    """
    懒加载稀疏 patch，每次 __getitem__ 从磁盘读取并转为 dense tensor。
    内存中只存一个 patch 的 dense 数据。
    """

    def __init__(self, cache_dir, patches):
        self.cache_dir = cache_dir
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        info = self.patches[idx]
        fpath = os.path.join(self.cache_dir, info["file"])
        data = np.load(fpath)

        # 重建稀疏矩阵 → dense tensor
        feat = sp.csr_matrix(
            (data["feat_data"].astype(np.float32),  # float16 → float32
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

        return {
            "features": torch.from_numpy(feat),
            "adj": torch.from_numpy(adj),
            "slice_idx": info["slice_idx"],
        }


def collate_fn(batch):
    """batch_size=1，直接返回单个 patch"""
    assert len(batch) == 1
    return batch[0]


# ═══════════════════════════════════════════════════════════════════════════
# 5. 训练
# ═══════════════════════════════════════════════════════════════════════════
def train(cfg, args):
    rank, world_size, local_rank, device = setup_distributed()
    log(f"世界大小: {world_size}, 当前 rank: {rank}, 设备: {device}")

    save_dir = cfg["paths"]["save_dir"]
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── 加载元信息（几 KB，所有 rank 都可以读） ──
    meta, cache_dir = load_meta(cfg)
    log(f"缓存目录: {cache_dir}")
    log(f"  切片数: {meta['n_slices']}, patch 数: {meta['n_patches']}, 输入维度: {meta['input_dim']}")

    # ── 超参数 ──
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    n_epochs = tcfg["n_epochs"]
    lr = tcfg["learning_rate"]
    patience = tcfg.get("patience", 20)
    grad_accum = tcfg.get("grad_accum_steps", 1)
    use_amp = tcfg.get("use_amp", True)
    input_dim = meta["input_dim"]

    # ── 懒加载数据集（不占内存！） ──
    dataset = LazyPatchDataset(cache_dir, meta["patches"])
    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    log(f"数据集: {len(dataset)} 个 patch（懒加载模式）")

    # ── 模型 ──
    model = MAEEncoder(
        input_dim=input_dim,
        hidden_dim=mcfg["hidden_dim"],
        latent_dim=mcfg["latent_dim"],
        dropout_rate=mcfg["dropout_rate"],
        gene_mask_ratio=mcfg["gene_mask_ratio"],
        cell_mask_ratio=mcfg["cell_mask_ratio"],
    ).to(device)

    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    raw_model = model.module if hasattr(model, "module") else model

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── 恢复训练 ──
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        log(f"恢复训练: epoch {start_epoch}, best_loss={best_loss:.4f}")

    # ── TensorBoard ──
    tb_writer = None
    if is_main():
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(save_dir, "logs"))

    # ── 训练循环 ──
    log(f"开始训练: {n_epochs} epochs, grad_accum={grad_accum}, AMP={use_amp}")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        total_gene = 0.0
        total_cell = 0.0
        n_batches = 0

        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            feat = batch["features"].to(device, non_blocking=True)
            adj = batch["adj"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                z_cell, z_niche, loss_dict = model(feat, adj)
                loss = loss_dict["loss_total"] / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss_dict["loss_total"].item()
            total_gene += loss_dict.get("loss_gene_mask", loss_dict.get("loss_gene", torch.tensor(0))).item() if isinstance(loss_dict.get("loss_gene_mask", 0), torch.Tensor) else loss_dict.get("loss_gene_mask", 0)
            total_cell += loss_dict.get("loss_cell_mask", loss_dict.get("loss_cell", torch.tensor(0))).item() if isinstance(loss_dict.get("loss_cell_mask", 0), torch.Tensor) else loss_dict.get("loss_cell_mask", 0)
            n_batches += 1

            del feat, adj, z_cell, z_niche, loss

        # ── 汇总 ──
        avg_loss = total_loss / max(n_batches, 1)
        avg_gene = total_gene / max(n_batches, 1)
        avg_cell = total_cell / max(n_batches, 1)

        if dist.is_initialized():
            stats = torch.tensor([avg_loss, avg_gene, avg_cell, float(n_batches)], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_n = stats[3].item()
            avg_loss = stats[0].item() / max(total_n, 1) * n_batches  # 加权平均
            avg_gene = stats[1].item() / max(total_n, 1) * n_batches
            avg_cell = stats[2].item() / max(total_n, 1) * n_batches
            # 简化：直接取均值
            avg_loss = stats[0].item() / world_size
            avg_gene = stats[1].item() / world_size
            avg_cell = stats[2].item() / world_size

        scheduler.step(avg_loss)

        log(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | "
            f"Gene: {avg_gene:.4f} | Cell: {avg_cell:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if tb_writer:
            tb_writer.add_scalar("Loss/total", avg_loss, epoch)
            tb_writer.add_scalar("Loss/gene", avg_gene, epoch)
            tb_writer.add_scalar("Loss/cell", avg_cell, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # ── Checkpoint ──
        if is_main():
            ckpt = {
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
                log(f"  ✓ 最佳模型已保存 (loss={best_loss:.4f})")
            else:
                patience_counter += 1

        # ── Early stopping ──
        if dist.is_initialized():
            stop = torch.tensor([1 if patience_counter >= patience else 0],
                                dtype=torch.int64, device=device)
            dist.broadcast(stop, src=0)
            if stop.item():
                log(f"Early stopping at epoch {epoch+1}")
                break
        elif patience_counter >= patience:
            log(f"Early stopping at epoch {epoch+1}")
            break

        gc.collect()
        torch.cuda.empty_cache()

    if tb_writer:
        tb_writer.close()
    cleanup_distributed()
    log("训练完成！")


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    train(cfg, args)