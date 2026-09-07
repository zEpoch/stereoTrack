"""
基于 PyTorch Lightning 重构的训练脚本
运行方式:
    单机单卡: python train_lightning.py --config config.yaml
    单机多卡: python train_lightning.py --config config.yaml从而自动调用全部可用GPU (无需强依赖torchrun)
"""

import os
import argparse
import pickle
import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import scipy.sparse as sp
from scipy.spatial import cKDTree

# python tracks/stereoTrack/train_pl.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy


import torch.nn as nn
import torch.nn.functional as F
from stereotrack.mae import MAEEncoder
from stereotrack.dataset import load_meta, LazyPatchDataset, DynamicGraphDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning 训练")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r") as f:
        return yaml.safe_load(f)



def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


class SpeciesBalancedSampler(Sampler):
    """Sample the same number of cached patches from every species each epoch."""

    def __init__(self, patches, samples_per_species=None, seed=0):
        species_indices = {}
        for index, patch in enumerate(patches):
            species = str(patch.get("species", "unknown"))
            species_indices.setdefault(species, []).append(index)

        if len(species_indices) < 2:
            raise ValueError("Species-balanced sampling requires at least two species")

        self.species_indices = {
            species: torch.tensor(indices, dtype=torch.long)
            for species, indices in sorted(species_indices.items())
        }
        if samples_per_species is None:
            samples_per_species = (len(patches) + len(self.species_indices) - 1) // len(self.species_indices)
        self.samples_per_species = int(samples_per_species)
        if self.samples_per_species <= 0:
            raise ValueError("samples_per_species must be positive")

        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.samples_per_species * len(self.species_indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _sample_species(self, indices, generator):
        n_indices = int(indices.numel())
        full_repeats, remainder = divmod(self.samples_per_species, n_indices)
        sampled = []

        # Shuffled full passes ensure that small species are covered before repeating patches.
        for _ in range(full_repeats):
            sampled.append(indices[torch.randperm(n_indices, generator=generator)])
        if remainder:
            order = torch.randperm(n_indices, generator=generator)[:remainder]
            sampled.append(indices[order])
        return torch.cat(sampled)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = [
            self._sample_species(indices, generator)
            for indices in self.species_indices.values()
        ]
        sampled = torch.cat(sampled)
        sampled = sampled[torch.randperm(sampled.numel(), generator=generator)]
        return iter(sampled.tolist())



class MAELightning(pl.LightningModule):
    def __init__(self, cfg, input_dim, species_names=None):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg
        mcfg = cfg["model"]
        tcfg = cfg["training"]
        species_names = list(species_names or [])
        self.species_names = species_names
        self.species_to_id = {name: idx for idx, name in enumerate(species_names)}
        
        self.model = MAEEncoder(
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

        self.lr = float(tcfg["learning_rate"])
        self.n_epochs = tcfg["n_epochs"]
        self.lambda_bin = float(tcfg.get("lambda_binary", 0.0))   # <-- 新增
        align_cfg = cfg.get("alignment", tcfg.get("alignment", {})) or {}
        self.alignment_enabled = bool(align_cfg.get("enabled", False))
        self.lambda_alignment = float(align_cfg.get("lambda_alignment", 0.0))
        self.alignment_target = str(align_cfg.get("target", "both"))
        self.alignment_metric = str(align_cfg.get("metric", "cosine"))
        self.alignment_warmup_epochs = int(align_cfg.get("warmup_epochs", 5))
        self.alignment_ramp_epochs = int(align_cfg.get("ramp_epochs", 5))
        self.alignment_ema_momentum = float(align_cfg.get("ema_momentum", 0.95))
        self.alignment_normalize = bool(align_cfg.get("normalize", True))
        self.alignment_sync_prototypes = bool(align_cfg.get("sync_prototypes", True))

        if self.alignment_target.lower() not in {"cell", "niche", "both"}:
            raise ValueError("alignment.target must be one of: cell, niche, both")
        if self.alignment_metric.lower() not in {"cosine", "mse"}:
            raise ValueError("alignment.metric must be one of: cosine, mse")

        n_species = max(1, len(species_names))
        latent_dim = int(mcfg["latent_dim"])
        self.register_buffer("alignment_cell_bank", torch.zeros(n_species, latent_dim))
        self.register_buffer("alignment_niche_bank", torch.zeros(n_species, latent_dim))
        self.register_buffer("alignment_bank_ready", torch.zeros(n_species, dtype=torch.bool))
    
    def forward(self, features, adj):
        return self.model(features, adj)

    def _alignment_weight(self):
        if not self.alignment_enabled or self.lambda_alignment <= 0:
            return 0.0
        if len(self.species_names) < 2:
            return 0.0
        if self.current_epoch < self.alignment_warmup_epochs:
            return 0.0
        if self.alignment_ramp_epochs <= 0:
            return self.lambda_alignment
        ramp_pos = self.current_epoch - self.alignment_warmup_epochs + 1
        ramp = min(1.0, max(0.0, ramp_pos / float(self.alignment_ramp_epochs)))
        return self.lambda_alignment * ramp

    def _species_id_from_batch(self, batch):
        species_id = batch.get("species_id", None)
        if isinstance(species_id, torch.Tensor):
            species_id = int(species_id.detach().cpu().item())
        elif species_id is not None:
            species_id = int(species_id)
        else:
            species = batch.get("species", "unknown")
            species_id = self.species_to_id.get(str(species), -1)
        if species_id < 0 or species_id >= len(self.species_names):
            return None
        return species_id

    def _prepare_alignment_vector(self, vector):
        if self.alignment_normalize:
            return F.normalize(vector, dim=0)
        return vector

    def _prototype_loss_one(self, mean_z, bank, species_id):
        ready = self.alignment_bank_ready.clone()
        ready[species_id] = False
        if int(ready.sum().item()) < 1:
            return mean_z.new_tensor(0.0)

        target = bank[ready].mean(dim=0).detach()
        mean_z = self._prepare_alignment_vector(mean_z)
        target = self._prepare_alignment_vector(target)

        if self.alignment_metric == "mse":
            return F.mse_loss(mean_z, target)
        return 1.0 - F.cosine_similarity(mean_z.unsqueeze(0), target.unsqueeze(0)).mean()

    def _sync_alignment_banks(self, z_cell, z_niche, species_id):
        """Update identical, cell-count-weighted prototype banks on every DDP rank."""
        with torch.no_grad():
            species = torch.tensor([species_id], dtype=torch.long, device=z_cell.device)
            count = torch.tensor([z_cell.shape[0]], dtype=z_cell.dtype, device=z_cell.device)
            sums = torch.stack((z_cell.detach().sum(dim=0), z_niche.detach().sum(dim=0)))

            gathered_species = [species]
            gathered_counts = [count]
            gathered_sums = [sums]
            if self.alignment_sync_prototypes and dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                gathered_species = [torch.empty_like(species) for _ in range(world_size)]
                gathered_counts = [torch.empty_like(count) for _ in range(world_size)]
                gathered_sums = [torch.empty_like(sums) for _ in range(world_size)]
                dist.all_gather(gathered_species, species)
                dist.all_gather(gathered_counts, count)
                dist.all_gather(gathered_sums, sums)

            grouped = {}
            for rank_species, rank_count, rank_sums in zip(
                gathered_species, gathered_counts, gathered_sums
            ):
                rank_species_id = int(rank_species.item())
                if rank_species_id not in grouped:
                    grouped[rank_species_id] = [rank_sums.clone(), rank_count.clone()]
                else:
                    grouped[rank_species_id][0].add_(rank_sums)
                    grouped[rank_species_id][1].add_(rank_count)

            for rank_species_id, (species_sums, species_count) in grouped.items():
                cell_mean = species_sums[0] / species_count.clamp_min(1.0)
                niche_mean = species_sums[1] / species_count.clamp_min(1.0)
                if bool(self.alignment_bank_ready[rank_species_id]):
                    momentum = self.alignment_ema_momentum
                    self.alignment_cell_bank[rank_species_id].mul_(momentum).add_(
                        cell_mean, alpha=1.0 - momentum
                    )
                    self.alignment_niche_bank[rank_species_id].mul_(momentum).add_(
                        niche_mean, alpha=1.0 - momentum
                    )
                else:
                    self.alignment_cell_bank[rank_species_id].copy_(cell_mean)
                    self.alignment_niche_bank[rank_species_id].copy_(niche_mean)
                    self.alignment_bank_ready[rank_species_id] = True

    def _alignment_loss(self, z_cell, z_niche, batch):
        if not self.alignment_enabled or len(self.species_names) < 2:
            return z_cell.new_tensor(0.0)

        species_id = self._species_id_from_batch(batch)
        if species_id is None:
            return z_cell.new_tensor(0.0)

        self._sync_alignment_banks(z_cell, z_niche, species_id)

        losses = []
        target = self.alignment_target.lower()
        if target in {"cell", "both"}:
            mean_cell = z_cell.mean(dim=0)
            losses.append(self._prototype_loss_one(mean_cell, self.alignment_cell_bank, species_id))
        if target in {"niche", "both"}:
            mean_niche = z_niche.mean(dim=0)
            losses.append(self._prototype_loss_one(mean_niche, self.alignment_niche_bank, species_id))

        if not losses:
            return z_cell.new_tensor(0.0)
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        feat = batch["features"]
        adj = batch["adj"]

        z_cell, z_niche, loss_dict = self(feat, adj)
        
        binary_loss = (
            loss_dict.get('loss_niche_recon_bin', 0.0) +
            loss_dict.get('loss_cell_recon_bin', 0.0) +
            loss_dict.get('loss_masked_cell_bin', 0.0)
        )
        alignment_loss = self._alignment_loss(z_cell, z_niche, batch)
        alignment_weight = self._alignment_weight()
        loss = loss_dict["loss_total"] + self.lambda_bin * binary_loss + alignment_weight * alignment_loss
        
        curr_batch_size = feat.shape[0]

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=curr_batch_size)
        
        niche_recon_loss = loss_dict.get("loss_niche_recon", 0.0)
        cell_recon_loss = loss_dict.get("loss_cell_recon", 0.0)
        masked_cell_recon_loss = loss_dict.get("loss_masked_cell_recon", 0.0)
        self.log("train_niche_recon_loss", niche_recon_loss, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=curr_batch_size)
        self.log("train_cell_recon_loss", cell_recon_loss, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=curr_batch_size)
        self.log("train_masked_cell_recon_loss", masked_cell_recon_loss, sync_dist=True, on_step=True, on_epoch=True,
                 batch_size=curr_batch_size)

        if self.lambda_bin > 0:
            self.log("train_binary_loss", binary_loss,
                     sync_dist=True, on_step=True, on_epoch=True, batch_size=curr_batch_size)
            self.log("train_niche_recon_bin", loss_dict.get("loss_niche_recon_bin", 0.0),
                     sync_dist=True, on_step=True, on_epoch=True, batch_size=curr_batch_size)
            self.log("train_cell_recon_bin", loss_dict.get("loss_cell_recon_bin", 0.0),
                     sync_dist=True, on_step=True, on_epoch=True, batch_size=curr_batch_size)
            self.log("train_masked_cell_bin", loss_dict.get("loss_masked_cell_bin", 0.0),
                     sync_dist=True, on_step=True, on_epoch=True, batch_size=curr_batch_size)

        if self.alignment_enabled:
            self.log("train_alignment_loss", alignment_loss,
                     sync_dist=True, on_step=True, on_epoch=True, batch_size=curr_batch_size)
            self.log("train_alignment_weight", torch.tensor(alignment_weight, device=self.device),
                     sync_dist=True, on_step=True, on_epoch=True, batch_size=curr_batch_size)

        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.logged_metrics.get("train_loss_epoch")
        if avg_loss is not None:
            print(f"\n👉 Epoch [{self.current_epoch + 1}/{self.n_epochs}] 结束 | 平均总 Loss: {avg_loss:.4f}")

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.3, 
            patience=50
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
                "interval": "epoch",
            },
        }
        



def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    cfg = load_config(args)
    
    meta, cache_dir = load_meta(cfg)
    input_dim = meta["input_dim"]
    
    tcfg = cfg["training"]
    species_names = sorted({patch.get("species", "unknown") for patch in meta["patches"]})
    species_to_id = {name: idx for idx, name in enumerate(species_names)}
    dataset = LazyPatchDataset(cache_dir, meta["patches"], species_to_id=species_to_id)

    balanced_sampling = bool(tcfg.get("species_balanced_sampling", False))
    sampler = None
    if balanced_sampling:
        sampler = SpeciesBalancedSampler(
            meta["patches"],
            samples_per_species=tcfg.get("samples_per_species"),
            seed=tcfg.get("sampling_seed", 0),
        )
        patch_counts = {
            species: int(indices.numel())
            for species, indices in sampler.species_indices.items()
        }
        print(
            "[sampling] species-balanced patches enabled: "
            f"available={patch_counts}, samples_per_species={sampler.samples_per_species}, "
            f"samples_per_epoch={len(sampler)}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=not balanced_sampling,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = MAELightning(cfg, input_dim, species_names=species_names)

    save_dir = cfg["paths"]["save_dir"]
    logging_cfg = cfg.get("logging", {})
    loggers = []

    if logging_cfg.get("use_tensorboard", False):
        tb_dir = logging_cfg.get("tensorboard_save_dir", save_dir)
        loggers.append(TensorBoardLogger(save_dir=tb_dir, name="tensorboard_logs"))

    if logging_cfg.get("use_wandb", True):
        wandb_dir = logging_cfg.get("wandb_save_dir", save_dir)
        os.makedirs(wandb_dir, exist_ok=True)
        loggers.append(WandbLogger(
            project=logging_cfg.get("wandb_project", "StereoTrack"),
            name=logging_cfg.get("wandb_name", "mae_run"),
            save_dir=wandb_dir,
            config=cfg
        ))

    final_logger = loggers if len(loggers) > 0 else False



    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="best-{epoch:02d}-{train_loss_epoch:.4f}",
        save_top_k=1,
        monitor="train_loss_epoch", 
        mode="min",
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="train_loss_epoch", 
        patience=cfg["training"]['patience'],
        mode="min"
    )

    callbacks = [ckpt_callback, early_stop_callback]
    if final_logger:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    strategy = DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=tcfg["n_epochs"],
        accelerator="gpu",
        devices="auto",
        # strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        strategy=strategy,
        precision="16-mixed" if tcfg.get("use_amp", True) else "32-true",
        accumulate_grad_batches=tcfg.get("grad_accum_steps", 1), 
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=final_logger,
        log_every_n_steps=10
    )

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
