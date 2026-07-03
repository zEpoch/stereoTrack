import os
import gc
import torch
import warnings
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from pathlib import Path
import scipy.sparse as sp

from stereotrack import (
    construct_graph,
    preprocess_adj_sparse,
    get_spatial_input,
    construct_data_multislice,
    MultiSliceGraphDataset,
    MultiSliceDataLoader,
    get_feature_sparse,
)
from stereotrack.mae import MAEEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)
import dgl.dataloading as dgl_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter

# ── 路径设置 ─────────────────────────────────────────────────────────────
save_dir = '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out'
log_dir = f"{save_dir}/03_han_mouse_mae/logs"
adata_dir = f"{save_dir}/03_han_mouse_mae/adatas"
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(save_dir).mkdir(parents=True, exist_ok=True)
Path(adata_dir).mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"可用 GPU 数量: {n_gpus}")

# 为每个切片分配 GPU（轮询）
def get_device_for_slice(slice_idx):
    if n_gpus > 1:
        return torch.device(f"cuda:{slice_idx % n_gpus}")
    return device

# ── 数据加载（优化：只做一次 concat 取公共基因，然后立即释放） ─────────────
data_root = '/home/share/huadjyin/home/zhoutao3/tracks/example_data/03_han_mouse_brain'
path_list = [i for i in os.listdir(data_root) if i.endswith('.h5ad')]

# 第一遍：加载并过滤，获取公共基因集
adatas_raw = []
for z in path_list:
    temp = sc.read_h5ad(f'{data_root}/{z}')
    temp = temp[temp.obs['gene_area'] > 0].copy()
    if len(temp) < 300:
        del temp; continue
    temp.obsm['ccf'] = temp.obs[['az', 'ay', 'ax']].values * 10
    temp.obs['track_names'] = z
    temp.var_names_make_unique()
    temp.obs_names_make_unique()
    adatas_raw.append(temp)

# 取公共基因（只取基因名交集，不做 concat）
common_genes = set(adatas_raw[0].var_names)
for a in adatas_raw[1:]:
    common_genes &= set(a.var_names)
common_genes = sorted(common_genes)
print(f"公共基因数: {len(common_genes)}")

# 第二遍：逐切片预处理（避免 concat 产生的内存峰值）
adatas = []
for temp in adatas_raw:
    temp = temp[:, common_genes].copy()
    sc.pp.normalize_total(temp, target_sum=1e4)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    temp = construct_graph(temp)
    temp = preprocess_adj_sparse(temp)
    temp = get_spatial_input(temp)
    temp.var.index = [g.upper() for g in temp.var.index]
    adatas.append(temp)

# 释放原始数据
del adatas_raw
gc.collect()

# ── 超参数 ───────────────────────────────────────────────────────────────
batch_size = 4096
hidden_dim = 512
latent_dim = 64
n_epochs = 50
dropout_rate = 0.1
learning_rate = 0.001
patience = 20
gene_mask_ratio = 0.5
cell_mask_ratio = 0.2

# ── 构建图数据与 DataLoader ──────────────────────────────────────────────
n_slices = len(adatas)
adj_all, g_all = construct_data_multislice(adatas)
dataset_all = [MultiSliceGraphDataset(g, adata) for g, adata in zip(g_all, adatas)]
sampler = dgl_loader.MultiLayerFullNeighborSampler(1)
dataloader = MultiSliceDataLoader(
    dataset_all=dataset_all,
    sampler=sampler,
    batch_size=batch_size,
    shuffle=True,
    n_slices=n_slices,
    drop_last=False,
)

# 特征保持在 CPU 稀疏格式，按需转移到 GPU（节省显存）
features_all = [get_feature_sparse(torch.device("cpu"), adata.obsm["spatial_input"]) for adata in adatas]

# ── 获取输入维度 ─────────────────────────────────────────────────────────
input_dim = adatas[0].shape[1]

# ── 初始化 MAE 模型（多 GPU：每张卡一个副本，共享参数）─────────────────────
# 由于邻接矩阵不可沿 batch 拆分，DataParallel 不适用于图网络
# 改用：主模型在 GPU:0，按切片轮询将数据发送到不同 GPU
if n_gpus > 1:
    print(f"使用 {n_gpus} 个 GPU，按切片轮询分配")
    # 在每个 GPU 上创建模型副本
    models = []
    for gpu_id in range(n_gpus):
        dev = torch.device(f"cuda:{gpu_id}")
        m = MAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            gene_mask_ratio=gene_mask_ratio,
            cell_mask_ratio=cell_mask_ratio,
        )
        m.to(dev)
        models.append(m)
    # 主模型用于保存/加载和优化器
    raw_model = models[0]
    # 同步所有模型参数
    def sync_params():
        state = models[0].state_dict()
        for m in models[1:]:
            m.load_state_dict(state)
    sync_params()
else:
    raw_model = MAEEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        gene_mask_ratio=gene_mask_ratio,
        cell_mask_ratio=gene_mask_ratio,
    )
    raw_model.to(device)
    models = [raw_model]

optimizer = torch.optim.AdamW(
    raw_model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

best_loss = float('inf')
patience_counter = 0


def sparse_slice_to_dense_tensor(mat, row_idx, col_idx=None, dev=device):
    """从稀疏矩阵中取子矩阵并转为 GPU tensor，避免整体 todense
    
    - 若 col_idx=None，只按行切片（用于 features：取细胞子集，保留所有列）
    - 若 col_idx 有值，按行和列同时切片（用于 adj：取子图）
    """
    sub = mat[row_idx]
    if col_idx is not None:
        sub = sub[:, col_idx]
    if sp.issparse(sub):
        sub = sub.toarray()
    elif isinstance(sub, np.matrix):
        sub = np.asarray(sub)
    return torch.FloatTensor(sub).to(dev)


# ── 训练循环 ─────────────────────────────────────────────────────────────
train_batch = 0
for epoch in range(n_epochs):
    for m in models:
        m.train()
    total_loss = 0
    total_gene_loss = 0
    total_cell_loss = 0
    num_batches = 0

    # 多 GPU 时，收集各卡上的梯度用于累积
    # 策略：每个 batch 根据 slice_idx 选择对应 GPU 上的模型前向，
    # 然后将梯度搬回主模型更新，再同步参数
    for blocks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        slice_idx = blocks["slice_idx"]
        indices = blocks["single"]
        features = features_all[slice_idx]
        adj = adj_all[slice_idx]

        if isinstance(indices, torch.Tensor):
            indices_np = indices.cpu().numpy()
        else:
            indices_np = indices

        # 选择该切片对应的 GPU 和模型
        if n_gpus > 1:
            gpu_id = slice_idx % n_gpus
            dev = torch.device(f"cuda:{gpu_id}")
            cur_model = models[gpu_id]
        else:
            dev = device
            cur_model = models[0]

        # 按需从稀疏矩阵取子集并转 GPU（不预加载全量到 GPU）
        features_batch = sparse_slice_to_dense_tensor(features, indices_np, dev=dev)          # 只切行：(batch, n_genes)
        adj_batch = sparse_slice_to_dense_tensor(adj, indices_np, col_idx=indices_np, dev=dev) # 行列都切：(batch, batch)

        # MAE 前向传播
        z_cell, z_niche, loss_dict = cur_model(features_batch, adj_batch)
        loss = loss_dict['loss_total']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n_gpus > 1:
            sync_params()

        # 及时释放大 tensor
        del features_batch, adj_batch, z_cell, z_niche

        total_loss += loss.item()
        total_gene_loss += loss_dict['loss_gene_mask'].item()
        total_cell_loss += loss_dict['loss_cell_mask'].item()
        num_batches += 1
        train_batch += 1

        writer.add_scalar('Loss/Total', loss.item(), train_batch)
        writer.add_scalar('Loss/GeneMask', loss_dict['loss_gene_mask'].item(), train_batch)
        writer.add_scalar('Loss/CellMask', loss_dict['loss_cell_mask'].item(), train_batch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], train_batch)

    avg_loss = total_loss / num_batches
    avg_gene_loss = total_gene_loss / num_batches
    avg_cell_loss = total_cell_loss / num_batches

    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | "
          f"Gene: {avg_gene_loss:.4f} | Cell: {avg_cell_loss:.4f}")

    writer.add_scalar('Loss/avg_total', avg_loss, epoch)
    writer.add_scalar('Loss/avg_gene_mask', avg_gene_loss, epoch)
    writer.add_scalar('Loss/avg_cell_mask', avg_cell_loss, epoch)

    scheduler.step(avg_loss)
    torch.save(raw_model.state_dict(), f"{save_dir}/03_han_mouse_mae/mae_{epoch}.pt")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # 每轮结束后回收
    gc.collect()
    torch.cuda.empty_cache()

writer.close()

# ── 推理：多 GPU 并行提取嵌入 ────────────────────────────────────────────
import concurrent.futures

raw_model.load_state_dict(torch.load(f"{save_dir}/03_han_mouse_mae/mae_{epoch}.pt"))
raw_model.eval()
infer_batch_size = 4096


def infer_one_slice(slice_idx):
    """在指定 GPU 上对一个切片做推理"""
    adata = adatas[slice_idx]
    z = adata.obs['track_names'].iloc[0]
    adj = adj_all[slice_idx]
    features = features_all[slice_idx]
    dev = torch.device(f"cuda:{slice_idx % n_gpus}")

    # 每个线程创建自己的模型副本
    model_copy = MAEEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        dropout_rate=dropout_rate, gene_mask_ratio=gene_mask_ratio,
        cell_mask_ratio=cell_mask_ratio,
    )
    model_copy.load_state_dict(raw_model.state_dict())
    model_copy.to(dev)
    model_copy.eval()

    n_cells = adata.shape[0]
    z_cell_list, z_niche_list, x_recon_list = [], [], []

    with torch.no_grad():
        for start in range(0, n_cells, infer_batch_size):
            end = min(start + infer_batch_size, n_cells)
            idx = np.arange(start, end)
            features_batch = sparse_slice_to_dense_tensor(features, idx, dev=dev)
            adj_batch = sparse_slice_to_dense_tensor(adj, idx, col_idx=idx, dev=dev)

            z_cell, z_niche = model_copy.encode(features_batch, adj_batch)
            x_recon = model_copy.decoder(z_niche)

            z_cell_list.append(z_cell.cpu().numpy())
            z_niche_list.append(z_niche.cpu().numpy())
            x_recon_list.append(x_recon.cpu().numpy())

            del features_batch, adj_batch, z_cell, z_niche, x_recon

    z_cell_np = np.concatenate(z_cell_list)
    z_niche_np = np.concatenate(z_niche_list)
    x_recon_np = np.concatenate(x_recon_list)

    adata2 = sc.AnnData(x_recon_np)
    adata2.obs = adata.obs.copy()
    adata2.var = adata.var.copy()
    adata2.obsm = dict(adata.obsm)
    adata2.obsm['cell_embedding'] = z_cell_np
    adata2.obsm['niche_embedding'] = z_niche_np
    adata2.write_h5ad(f"{adata_dir}/{z}")
    print(f"切片 {z} 推理完成 (GPU:{slice_idx % n_gpus})")


# 按 GPU 分组，同一 GPU 上的切片串行，不同 GPU 并行
with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
    futures = [executor.submit(infer_one_slice, i) for i in range(len(adatas))]
    concurrent.futures.wait(futures)

print("MAE 训练与推理完成！")
