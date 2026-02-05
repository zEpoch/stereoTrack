# StereoTrack Usage Guide

## 快速开始

### 1. 完整流程示例

运行完整的demo示例：

```bash
python example_demo.py
```

### 2. 分步骤使用

#### 步骤 1: 数据预处理

```python
import scanpy as sc
from fusemap.preprocess import (
    preprocess_adata,
    construct_graph,
    preprocess_adj_sparse,
    get_spatial_input
)

# 加载数据
adata = sc.read_h5ad("your_data.h5ad")

# 预处理
adata.var.index = [g.upper() for g in adata.var.index]
preprocess_adata([adata], 1)
construct_graph([adata], 1, ["knn3d"], ["ST"])
preprocess_adj_sparse([adata], 1, ["ST"])
get_spatial_input([adata], 1, "norm")
```

#### 步骤 2: 训练模型

```python
from stereotrack.single_time import train_stereotrack, seed_all

# 设置随机种子
seed_all(42)

# 训练
model, adata_trained = train_stereotrack(
    adata=adata,
    save_dir="./output",
    hidden_dim=512,
    latent_dim=64,
    dropout_rate=0.2,
    n_epochs=50,
    batch_size=64,
    learning_rate=0.001,
    patience=5
)
```

#### 步骤 3: 推理获取结果

```python
from stereotrack.single_time import inference_stereotrack

# 推理
adata_result = inference_stereotrack(
    adata=adata_trained,
    model_path="./output/trained_model/stereotrack_best.pt",
    save_dir="./output",
    hidden_dim=512,
    latent_dim=64,
    dropout_rate=0.2,
    batch_size=64
)

# 访问结果
embeddings = adata_result.obsm['stereotrack_embedding']
imputation = adata_result.layers['stereotrack_imputation']
```

## 输出文件说明

- `stereotrack_best.pt`: 训练好的模型权重
- `stereotrack_result.h5ad`: 训练后的AnnData对象（包含embedding）
- `stereotrack_inference_result.h5ad`: 推理后的AnnData对象（包含embedding和imputation）
- `stereotrack_embeddings.npy`: 细胞embedding矩阵
- `stereotrack_imputation.npy`: 基因表达imputation矩阵

## 参数说明

- `hidden_dim`: 隐藏层维度 (默认: 512)
- `latent_dim`: 潜在空间维度 (默认: 64)
- `dropout_rate`: Dropout比率 (默认: 0.2)
- `n_epochs`: 训练轮数 (默认: 50)
- `batch_size`: 批次大小 (默认: 64)
- `learning_rate`: 学习率 (默认: 0.001)
- `patience`: 早停耐心值 (默认: 5)
