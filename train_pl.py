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
from torch.utils.data import Dataset, DataLoader
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



class MAELightning(pl.LightningModule):
    def __init__(self, cfg, input_dim):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg
        mcfg = cfg["model"]
        tcfg = cfg["training"]
        
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
    
    def forward(self, features, adj):
        return self.model(features, adj)

    def training_step(self, batch, batch_idx):
        feat = batch["features"]
        adj = batch["adj"]

        z_cell, z_niche, loss_dict = self(feat, adj)
        
        binary_loss = (
            loss_dict.get('loss_niche_recon_bin', 0.0) +
            loss_dict.get('loss_cell_recon_bin', 0.0) +
            loss_dict.get('loss_masked_cell_bin', 0.0)
        )
        loss = loss_dict["loss_total"] + self.lambda_bin * binary_loss
        
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
    
    patch_size = cfg["training"].get("patch_size", 4096)
    dataset = LazyPatchDataset(cache_dir, meta["patches"])

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = MAELightning(cfg, input_dim)

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

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    tcfg = cfg["training"]
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
        callbacks=[ckpt_callback, early_stop_callback, lr_monitor],
        logger=final_logger,
        log_every_n_steps=10
    )

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()