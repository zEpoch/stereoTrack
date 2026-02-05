# StereoTrack 模型架构与流程详解

## 一、数据预处理流程

### 步骤 1: 基础表达数据预处理
```python
preprocess_adata([adata], 1)
```
**做了什么:**
- 归一化基因表达矩阵 (normalization)
- 对数转换 (log transformation)
- 可能的特征选择 (feature selection)
- 标准化基因名称为大写

**输入:** 原始 AnnData 对象，包含 `adata.X` (细胞 × 基因表达矩阵)
**输出:** 预处理后的表达矩阵存储在 `adata.X`

---

### 步骤 2: 构建空间图结构
```python
construct_graph([adata], 1, graph_types=["knn3d"], data_types=["ST"])
```
**做了什么:**
- 基于细胞的空间坐标 (`adata.obsm['spatial']`) 构建 KNN 图
- 使用 3D K近邻算法连接空间上相近的细胞
- 生成邻接矩阵，存储细胞间的空间邻近关系

**输入:** `adata.obsm['spatial']` - 细胞空间坐标 (N × 3)
**输出:** `adata.obsm['adj']` - 稀疏邻接矩阵 (N × N)
```
示例:
细胞 A 的 3 个最近邻是 B, C, D
则 adj[A, B] = adj[A, C] = adj[A, D] = 1
```

---

### 步骤 3: 归一化邻接矩阵
```python
preprocess_adj_sparse([adata], 1, data_types=["ST"])
```
**做了什么:**
- 对邻接矩阵进行归一化处理
- 通常使用对称归一化: D^(-1/2) * A * D^(-1/2)
- D 是度矩阵，A 是邻接矩阵

**输入:** `adata.obsm['adj']` - 原始邻接矩阵
**输出:** `adata.obsm['adj_normalized']` - 归一化邻接矩阵
```
作用: 使得每个细胞的邻居贡献权重和为1
```

---

### 步骤 4: 生成空间输入特征
```python
get_spatial_input([adata], 1, mode="norm")
```
**做了什么:**
- 结合表达数据和空间信息生成模型输入特征
- 可能包括: 原始表达 + 邻居表达的聚合
- 归一化处理确保特征在合理范围

**输入:** `adata.X` (表达) + `adata.obsm['adj_normalized']` (空间关系)
**输出:** `adata.obsm['spatial_input']` - 空间增强的特征矩阵 (N × G)

---

## 二、数据封装与模型输入

### Dataset 封装
```python
class StereoTrackGraphDataset:
    def __init__(self, g, adata):
        self.g = g              # DGL 图对象
        self.adata = adata      # AnnData 对象
        self.n_nodes = g.number_of_nodes()  # 细胞数量
```
**作用:**
- 封装图结构和数据
- 提供索引访问接口
- `__getitem__` 返回细胞索引

---

### DataLoader 封装
```python
class StereoTrackDataLoader:
    def __iter__(self):
        for indices in self.dataloader:
            blocks = {
                "single": indices,           # 当前批次细胞索引
                "spatial": (indices, all_indices)  # 空间邻居采样
            }
            yield blocks
```
**作用:**
1. **批次采样**: 每次采样 `batch_size` 个细胞
2. **邻居采样**: 为每个细胞采样其空间邻居
3. **构建子图**: 构建当前批次的子图用于训练

**数据流:**
```
原始数据 (N 个细胞)
    ↓
批次采样 (batch_size 个细胞)
    ↓
提取特征和邻接关系
    ↓
输入模型
```

---

## 三、模型输入详解

### 训练时的模型输入
```python
# 在 train.py 中
for blocks in dataloader:
    row_indices = list(blocks["single"])          # [64] 批次细胞索引
    col_indices = list(blocks["spatial"][1])      # [N] 所有细胞索引
    
    # 输入 1: 批次特征矩阵
    batch_features = features[row_indices]         # shape: [64, 2000]
    # 从 spatial_input 中提取当前批次的特征
    
    # 输入 2: 批次邻接矩阵
    adj_block = adj[row_indices, :][:, col_indices]  # shape: [64, N]
    # 提取当前批次细胞与所有细胞的邻接关系
```

**两个关键输入:**
1. **batch_features**: 当前批次细胞的表达特征 (64 × 2000)
2. **adj_block**: 当前批次细胞的空间邻接关系 (64 × N)

---

## 四、模型架构详解

### 整体架构
```
输入 (x, adj)
    ↓
┌─────────────────────┐
│   Encoder           │
│  - Linear + BN + Act│ → [64, 512] hidden
│  - Dropout          │
│  - Linear + BN + Act│ → [64, 512] hidden
│  - Dropout          │
│  - Mean branch      │ → [64, 64] z_mean
│  - LogVar branch    │ → [64, 64] z_log_var
└─────────────────────┘
    ↓
z_distribution = Normal(z_mean, z_log_var)  # VAE 分布
z_spatial = adj^T @ z_mean                  # 空间聚合 [N, 64]
    ↓
┌─────────────────────┐
│   Decoder           │
│  h = adj @ z_spatial│ → 反向空间传播 [64, 64]
│  x_recon = h @ W_gene│ → 重构表达 [64, 2000]
│  ReLU activation    │
└─────────────────────┘
    ↓
输出: x_recon, z_distribution, z_spatial, z_mean
```

### Encoder 详细结构
```python
Input: x [batch_size, n_genes]
    ↓
Dropout(0.2)
    ↓
Linear(n_genes → 512)
    ↓
BatchNorm1d(512)
    ↓
LeakyReLU(0.2)
    ↓
Dropout(0.2)
    ↓
Linear(512 → 512)
    ↓
BatchNorm1d(512)
    ↓
LeakyReLU(0.2)
    ↓
         ├─→ Linear(512 → 64) → z_mean
         └─→ Linear(512 → 64) → Softplus → z_log_var
    ↓
z_distribution = Normal(z_mean, z_log_var)
z_spatial = adj^T @ z_mean  # [N, 64] 空间邻居的embedding聚合
```

**关键点:**
- **z_mean**: 细胞的潜在表示 (embedding)
- **z_spatial**: 通过邻接矩阵聚合邻居的embedding
- **VAE结构**: 使用重参数化技巧采样

---

### Decoder 详细结构
```python
Input: z_spatial [N, latent_dim], adj [batch_size, N]
    ↓
h = adj @ z_spatial  # [batch_size, latent_dim]
# 聚合当前批次每个细胞的邻居embedding
    ↓
x_recon = h @ gene_embedding[:, var_index]  # [batch_size, n_genes]
# 通过基因embedding矩阵重构表达
    ↓
ReLU(x_recon)  # 确保非负
    ↓
Output: x_recon [batch_size, n_genes]
```

**关键概念:**
- **gene_embedding**: 学习的基因表示矩阵 [latent_dim, n_all_genes]
- **空间解码**: 先聚合空间信息，再投影到基因空间

---

## 五、损失函数计算

### Loss 组成
```python
def compute_loss(model, x, adj, x_recon, z_distribution, mask):
    # 1. 重构损失 (Reconstruction Loss)
    recon_loss = MSE(x_recon[mask], x[mask])
    # 只在训练集/验证集掩码的细胞上计算
    # 衡量重构的表达值与真实表达值的差异
    
    # 2. KL散度损失 (KL Divergence)
    kl_loss = KL(z_distribution || N(0,1))
    # 衡量学到的潜在分布与标准正态分布的差异
    # 正则化潜在空间，使其更平滑
    
    # 3. 总损失
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss
```

### 详细解释

**1. 重构损失 (Reconstruction Loss)**
```python
recon_loss = F.mse_loss(x_recon[mask], x[mask])
```
- **目的**: 确保重构的基因表达接近真实表达
- **公式**: MSE = (1/N) * Σ(x_recon - x)²
- **意义**: 越小说明重构质量越高
- **为什么用 mask**: 
  - 训练时只在训练集上计算
  - 验证时只在验证集上计算
  - 防止过拟合

**2. KL散度损失 (KL Divergence)**
```python
kl_loss = D.kl_divergence(
    z_distribution,      # N(z_mean, z_log_var)
    D.Normal(0.0, 1.0)   # N(0, 1)
).sum(dim=1).mean() / x.shape[1]
```
- **目的**: 正则化潜在空间
- **公式**: KL(q||p) = Σ[q(z) * log(q(z)/p(z))]
- **意义**: 
  - 使学到的分布接近标准正态分布
  - 防止 embedding 空间坍塌
  - 确保 embedding 具有良好的结构

**为什么除以 `x.shape[1]`?**
- 归一化 KL 损失
- 平衡重构损失和 KL 损失的尺度
- x.shape[1] 是基因数量

---

## 六、训练流程

```python
for epoch in range(n_epochs):
    # 训练阶段
    model.train()
    for blocks in dataloader:
        # 1. 提取批次数据
        batch_features = features[row_indices]  # [64, 2000]
        adj_block = adj[row_indices, :]        # [64, N]
        
        # 2. 前向传播
        x_recon, z_dist, z_spatial, z_mean = model(batch_features, adj_block)
        
        # 3. 计算损失
        loss, recon_loss, kl_loss = compute_loss(
            model, batch_features, adj_block, x_recon, z_dist, train_mask
        )
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        # 在验证集上评估
        val_loss = ...
    
    # 5. 学习率调整
    scheduler.step(val_loss)
    
    # 6. 早停检查
    if val_loss < best_val_loss:
        save_model()
    else:
        patience_counter += 1
```

---

## 七、推理流程

```python
def inference_stereotrack(adata, model_path, ...):
    # 1. 加载训练好的模型
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 2. 逐批次推理
    embeddings = []
    reconstructions = []
    
    with torch.no_grad():
        for blocks in dataloader:
            # 前向传播
            x_recon, z_dist, z_spatial, z_mean = model(features, adj)
            
            # 收集结果
            embeddings.append(z_mean)        # 细胞 embedding
            reconstructions.append(x_recon)   # 基因表达 imputation
    
    # 3. 拼接所有批次
    embeddings = np.vstack(embeddings)          # [N, 64]
    reconstructions = np.vstack(reconstructions) # [N, 2000]
    
    # 4. 保存结果
    adata.obsm['stereotrack_embedding'] = embeddings
    adata.layers['stereotrack_imputation'] = reconstructions
```

---

## 八、数据流总结

```
原始 h5ad 数据
    ↓ preprocess_adata
标准化表达矩阵
    ↓ construct_graph  
空间邻接图
    ↓ preprocess_adj_sparse
归一化邻接矩阵
    ↓ get_spatial_input
空间增强特征
    ↓ construct_data
DGL 图 + 稀疏矩阵
    ↓ Dataset + DataLoader
批次采样
    ↓ 提取特征和邻接
batch_features [64, 2000]
adj_block [64, N]
    ↓ Model Forward
Encoder → z_mean [64, 64]
         z_spatial [N, 64]
    ↓ Decoder
x_recon [64, 2000]
    ↓ Compute Loss
recon_loss + kl_loss
    ↓ Backward
更新参数
    ↓ Inference
embedding [N, 64]
imputation [N, 2000]
```

---

## 九、关键维度说明

| 变量 | 维度 | 说明 |
|------|------|------|
| adata.X | [N, G] | N 个细胞，G 个基因 |
| spatial_input | [N, G] | 空间增强的特征 |
| adj_normalized | [N, N] | 归一化邻接矩阵 |
| batch_features | [B, G] | B=batch_size |
| adj_block | [B, N] | 批次与全局的邻接 |
| z_mean | [B, L] | L=latent_dim (64) |
| z_spatial | [N, L] | 所有细胞的空间embedding |
| x_recon | [B, G] | 重构的表达 |

---

## 十、关键创新点

1. **空间感知的 VAE**: 结合空间邻接信息学习 embedding
2. **双向空间传播**: 
   - Encoder: z_spatial = adj^T @ z_mean (聚合邻居)
   - Decoder: h = adj @ z_spatial (反向传播)
3. **基因 Embedding**: 学习通用的基因表示矩阵
4. **图神经网络思想**: 虽然不是显式的 GNN，但通过邻接矩阵实现了类似的消息传递
