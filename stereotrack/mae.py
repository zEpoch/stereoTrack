import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAEEncoder(nn.Module):
    """
    空间感知的 Masked Autoencoder 编码器
    
    两种遮盖策略：
    1. gene_mask:  随机遮盖基因   → 学习转录组内部结构
    2. cell_mask:  随机遮盖整个细胞 → 学习空间微环境依赖
    """
    def __init__(
        self, 
        input_dim: int,          # 基因数量 G
        hidden_dim: int = 512,   # 隐层维度
        latent_dim: int = 64,    # 嵌入维度
        dropout_rate: float = 0.1,
        gene_mask_ratio: float = 0.5,   # 遮盖50%基因
        cell_mask_ratio: float = 0.2,   # 遮盖20%细胞
    ):
        super().__init__()
        
        self.gene_mask_ratio = gene_mask_ratio
        self.cell_mask_ratio = cell_mask_ratio
        self.latent_dim = latent_dim
        
        # ── 编码器 ──────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # ── 解码器（只在训练时用）────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()  # 基因表达非负
        )
        
    # ── 遮盖工具 ────────────────────────────────────────
    
    def mask_genes(self, x: torch.Tensor):
        """
        随机遮盖部分基因
        x: [N, G]
        return: x_masked [N, G], mask [N, G] bool
        """
        mask = torch.rand_like(x) < self.gene_mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask
    
    def mask_cells(self, x: torch.Tensor):
        """
        随机遮盖整个细胞
        x: [N, G]
        return: x_masked [N, G], mask [N] bool
        """
        mask = torch.rand(x.shape[0], device=x.device) \
                    < self.cell_mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask
    
    # ── 前向传播 ─────────────────────────────────────────
    
    def encode(self, x: torch.Tensor, adj: torch.Tensor):
        """
        编码 + 空间聚合
        
        x:   [N, G]  基因表达矩阵
        adj: [N, N]  空间邻接矩阵（稀疏）
        
        return:
            z_cell:  [N, d]  细胞嵌入  → 用于细胞类型聚类
            z_niche: [N, d]  生态位嵌入 → 用于niche识别/轨迹
        """
        z_cell = self.encoder(x)                  # [N, d]
        z_niche = torch.mm(adj.T, z_cell)         # [N, d] ← 核心创新
        return z_cell, z_niche
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        训练前向传播
        同时计算基因遮盖loss和细胞遮盖loss
        
        return:
            z_cell:     [N, d]
            z_niche:    [N, d]
            loss_dict:  各项loss的字典
        """
        loss_dict = {}
        
        # ── Task 1: 基因级遮盖 ──────────────────────────
        x_gene_masked, gene_mask = self.mask_genes(x)
        z_cell_1, z_niche_1 = self.encode(x_gene_masked, adj)
        x_recon_1 = self.decoder(z_niche_1)
        
        # 只在未遮盖的基因上计算loss（重建可见部分）
        loss_gene = F.mse_loss(
            x_recon_1[~gene_mask], 
            x[~gene_mask]
        )
        loss_dict['loss_gene_mask'] = loss_gene
        
        # ── Task 2: 细胞级遮盖 ──────────────────────────
        x_cell_masked, cell_mask = self.mask_cells(x)
        z_cell_2, z_niche_2 = self.encode(x_cell_masked, adj)
        x_recon_2 = self.decoder(z_niche_2)
        
        # 只在被遮盖的细胞上计算loss
        # 被遮盖细胞的z_niche完全来自邻居 → 空间依赖学习
        if cell_mask.sum() > 0:
            loss_cell = F.mse_loss(
                x_recon_2[cell_mask], 
                x[cell_mask]
            )
            loss_dict['loss_cell_mask'] = loss_cell
        else:
            loss_dict['loss_cell_mask'] = torch.tensor(
                0.0, device=x.device
            )
        
        # ── 合并loss ────────────────────────────────────
        loss_dict['loss_total'] = (
            loss_dict['loss_gene_mask'] + 
            loss_dict['loss_cell_mask']
        )
        
        # 返回基因遮盖版本的嵌入（更稳定）
        return z_cell_1, z_niche_1, loss_dict
