import torch
import torch.nn as nn
import torch.nn.functional as F



class MAEEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,          # 基因数量 G
        hidden_dim: int = 512,   # 隐层维度
        latent_dim: int = 64,    # 嵌入维度
        dropout_rate: float = 0.1,
        gene_mask_ratio: float = 0.5,   # 遮盖50%基因
        cell_mask_ratio: float = 0.2,   # 遮盖20%细胞
        n_encoder_layers: int = 2,  # 编码器隐藏层数
        n_decoder_layers: int = 1,  # 解码器隐藏层数
        niche_self_weight: float = 0.5,  # z_niche 中保留自身 cell identity 的比例
        lambda_cell_recon: float = 1.0,  # z_cell 直接重建表达的权重
        lambda_niche_recon: float = 1.0,  # z_niche 重建遮盖基因的权重
        lambda_cell_mask: float = 1.0,  # 遮盖细胞后由邻域恢复表达的权重

    ):
        super().__init__()
        
        self.gene_mask_ratio = gene_mask_ratio
        self.cell_mask_ratio = cell_mask_ratio
        self.latent_dim = latent_dim
        self.niche_self_weight = niche_self_weight
        self.lambda_cell_recon = lambda_cell_recon
        self.lambda_niche_recon = lambda_niche_recon
        self.lambda_cell_mask = lambda_cell_mask
        
         # ── 动态构建编码器 ──────────────────────────────────────
        enc_layers = []
        current_dim = input_dim
        for _ in range(n_encoder_layers):
            enc_layers.append(nn.Linear(current_dim, hidden_dim))
            enc_layers.append(nn.BatchNorm1d(hidden_dim))
            enc_layers.append(nn.LeakyReLU(0.2))
            enc_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        enc_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        


        

        # ── 解码器 ──────────────────────────────────────
        # niche_decoder/niche_binary_decoder: niche 分支，用于空间邻域表达重建
        # cell_decoder/cell_binary_decoder: cell 分支，直接约束 z_cell 学到细胞身份信息
        self.niche_decoder = self._build_decoder(latent_dim, hidden_dim, input_dim, n_decoder_layers, output_relu=True)
        self.niche_binary_decoder = self._build_decoder(latent_dim, hidden_dim, input_dim, n_decoder_layers, output_relu=False)
        self.cell_decoder = self._build_decoder(latent_dim, hidden_dim, input_dim, n_decoder_layers, output_relu=True)
        self.cell_binary_decoder = self._build_decoder(latent_dim, hidden_dim, input_dim, n_decoder_layers, output_relu=False)
        
    @staticmethod
    def _build_decoder(latent_dim, hidden_dim, input_dim, n_decoder_layers, output_relu: bool):
        layers = []
        current_dim = latent_dim
        for _ in range(n_decoder_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, input_dim))
        if output_relu:
            layers.append(nn.ReLU())  # 基因表达非负
        return nn.Sequential(*layers)
        
    @staticmethod
    def _safe_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return F.huber_loss(pred[mask], target[mask], delta=1.0)
    
    @staticmethod
    def _safe_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits[mask], target[mask])
    
    @staticmethod
    def _spatial_context(x: torch.Tensor, adj: torch.Tensor):
        """
        计算每个细胞的邻域表达上下文。

        注意：这里假设 adj 已经在数据预处理阶段完成归一化，例如 D^-1 A 或 D^-1/2 A D^-1/2。
        如果 adj 包含 self-loop，那么 context 会包含自身表达；如果希望 pure-neighbor niche，
        应在构图/预处理阶段去掉 self-loop 或降低 self-loop 权重。
        """
        if adj.is_sparse:
            return torch.sparse.mm(adj.transpose(0, 1), x)
        return torch.mm(adj.T, x)
        
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
        mask = torch.rand(x.shape[0], device=x.device) < self.cell_mask_ratio
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
        z_neighbor = self._spatial_context(z_cell, adj)  # [N, d] 空间邻域聚合
        z_niche = self.niche_self_weight * z_cell + (1.0 - self.niche_self_weight) * z_neighbor
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
        
        # ── 二值 mask 目标 ─────────────────────────────
        mask_target = (x > 0).float()   # shape: [N, G]
        context_target = self._spatial_context(x, adj)  # 邻域/local context 表达目标 [N, G]
        context_mask_target = (context_target > 0).float()
        
        # ── Task 1: 基因级遮盖 ──────────────────────────
        x_gene_masked, gene_mask = self.mask_genes(x)
        z_cell_1, z_niche_1 = self.encode(x_gene_masked, adj)
        
        # niche 分支：从 z_niche 重建邻域/local context，而不是当前细胞自身表达
        x_recon_1 = self.niche_decoder(z_niche_1)
        mask_logits_1 = self.niche_binary_decoder(z_niche_1)
        loss_niche_recon = self._safe_huber_loss(x_recon_1, context_target, gene_mask)
        loss_dict['loss_niche_recon'] = loss_niche_recon
        loss_dict['loss_niche_recon_bin'] = self._safe_bce_with_logits(mask_logits_1, context_mask_target, gene_mask)
        
        # cell 分支：直接约束 z_cell 重建被遮盖基因，提升细胞类型/identity 表征
        x_recon_cell_1 = self.cell_decoder(z_cell_1)
        cell_logits_1 = self.cell_binary_decoder(z_cell_1)
        loss_cell_recon = self._safe_huber_loss(x_recon_cell_1, x, gene_mask)
        loss_dict['loss_cell_recon'] = loss_cell_recon
        loss_dict['loss_cell_recon_bin'] = self._safe_bce_with_logits(cell_logits_1, mask_target, gene_mask)
        
        
        # ── Task 2: 细胞级遮盖 ──────────────────────────
        x_cell_masked, cell_mask = self.mask_cells(x)
        z_cell_2, z_niche_2 = self.encode(x_cell_masked, adj)
        x_recon_2 = self.niche_decoder(z_niche_2)
        mask_logits_2 = self.niche_binary_decoder(z_niche_2)
        
        # 只在被遮盖的细胞上计算 loss。
        # 目标不是恢复该细胞自身表达 x，而是恢复它的邻域/local context 表达 context_target，
        # 从而让 z_niche 更偏向 microenvironment / niche，而不是 cell type。
        if cell_mask.sum() > 0:
            loss_masked_cell_recon = F.huber_loss(
                x_recon_2[cell_mask],
                context_target[cell_mask],
                delta=1.0,
            )
            loss_dict['loss_masked_cell_recon'] = loss_masked_cell_recon
            
            loss_masked_cell_bin = F.binary_cross_entropy_with_logits(
                mask_logits_2[cell_mask],
                context_mask_target[cell_mask]
            )
            loss_dict['loss_masked_cell_bin'] = loss_masked_cell_bin
            
        else:
            loss_dict['loss_masked_cell_recon'] = torch.tensor(0.0, device=x.device)
            loss_dict['loss_masked_cell_bin'] = torch.tensor(0.0, device=x.device)
        
        # ── 合并loss ────────────────────────────────────
        loss_dict['loss_total'] = (
            self.lambda_niche_recon * loss_dict['loss_niche_recon'] +
            self.lambda_cell_recon * loss_dict['loss_cell_recon'] +
            self.lambda_cell_mask * loss_dict['loss_masked_cell_recon']
        )

        
        return z_cell_1, z_niche_1, loss_dict

    def get_dmt_embeddings(self, z_cell, z_niche):
        """
        直接返回：
        cell_high, cell_vis, niche_high, niche_vis, cell_experts, niche_experts
        """
        if not getattr(self, "use_dmt", False) or getattr(self, "dmt_cell", None) is None or getattr(self, "dmt_niche", None) is None:
            return None
        h_cell_high, h_cell_vis, exp_cell = self.dmt_cell(z_cell)
        h_niche_high, h_niche_vis, exp_niche = self.dmt_niche(z_niche)
        return h_cell_high, h_cell_vis, h_niche_high, h_niche_vis, exp_cell, exp_niche