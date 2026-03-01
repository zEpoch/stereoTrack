import torch
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
import scipy.sparse as sp

from stereotrack import (
    construct_graph,
    preprocess_adj_sparse,
    get_spatial_input,
    compute_loss,
    construct_data_multislice,
    MultiSliceGraphDataset,
    MultiSliceDataLoader,
    StereoTrackModel,
    get_feature_sparse
)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import dgl.dataloading as dgl_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

save_dir = '/data/work/stereotrack/out'


Path(save_dir).mkdir(parents=True, exist_ok=True)
# Path(f"{save_dir}/trained_model").mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


adata = sc.read_h5ad('/data/work/dataset/02_spateo_data/mouse_E9.5_embryo.h5ad')
adata.obs['z'] = [str(i).split('.')[0] for i in adata.obsm['3d_align_spatial'][:,2]]
adatas = []
for z in set(adata.obs['z']):
    temp = adata[adata.obs['z'] == z].copy()
    sc.pp.normalize_total(temp, target_sum=1e4)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    temp = construct_graph(temp)
    temp = preprocess_adj_sparse(temp)
    temp = get_spatial_input(temp)
    adatas.append(temp)
    
for i, adata in enumerate(adatas):
    adata.var.index = [g.upper() for g in adata.var.index]

all_genes = set()
for adata in adatas:
    all_genes.update(adata.var.index)
all_genes = sorted(list(all_genes))

batch_size = 2048
hidden_dim=512
latent_dim=64
n_epochs=10
dropout_rate= 0.2
learning_rate = 0.001
patience = 20

n_slices = len(adatas)
all_genes = sorted(list(adatas[0].var.index))
adj_all, g_all = construct_data_multislice(adatas)
dataset_all = [MultiSliceGraphDataset(g, adata) for g, adata in zip(g_all, adatas)]
sampler = dgl_loader.MultiLayerFullNeighborSampler(1)
dataloader = MultiSliceDataLoader(
    dataset_all=dataset_all,
    sampler=sampler,
    batch_size=batch_size,
    shuffle=True,
    n_slices=n_slices,
    drop_last=False
)

features_all = [get_feature_sparse(device, adata.obsm["spatial_input"]) for adata in adatas]


model = StereoTrackModel(
    len(all_genes),
    hidden_dim,
    latent_dim,
    dropout_rate,
    all_genes,
    all_genes,
    use_spatial_decoder=False
)
model.to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

best_loss = float('inf')
patience_counter = 0

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0

    for blocks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        slice_idx = blocks["slice_idx"]
        indices = blocks["single"]
        features = features_all[slice_idx]
        adj = adj_all[slice_idx]
        if isinstance(indices, torch.Tensor):
            indices_np = indices.cpu().numpy()
        else:
            indices_np = indices
        features_batch = torch.FloatTensor(
            features[indices_np].toarray() if sp.issparse(features) else features[indices_np]
        ).to(device)

        adj_batch = torch.FloatTensor(
            adj[indices_np][:, indices_np].todense() if sp.issparse(adj) else adj[indices_np][:, indices_np]
        ).to(device)

        x_recon, z_dist, z_spatial, z_mean = model(features_batch, adj_batch)

        loss, recon_loss, kl_loss = compute_loss(
            model, features_batch, adj_batch, x_recon, z_dist, z_spatial, z_mean
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches

    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"{save_dir}/nouse_spatial_decoder/stereotrack_best.pt")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break

model.load_state_dict(torch.load(f"{save_dir}/nouse_spatial_decoder/stereotrack_best.pt"))
model.eval()

for slice_idx, adata in enumerate(adatas):
    z = adata.obs['z'][0]
    adj = adj_all[slice_idx]
    features = features_all[slice_idx]

    with torch.no_grad():
        adj_tensor = torch.FloatTensor(adj.todense() if sp.issparse(adj) else adj).to(device)
        features_tensor = torch.FloatTensor(
            features.toarray() if sp.issparse(features) else features
        ).to(device)
        x_recon, _, z_spatial, z_mean = model(features_tensor, adj_tensor)
        z_mean = z_mean.cpu().numpy()
        z_spatial = z_spatial.cpu().numpy()
        x_recon = x_recon.cpu().numpy()

    adata2=sc.AnnData(x_recon)
    adata2.obs=adata.obs
    adata2.var=adata.var
    adata2.obsm=adata.obsm
    adata2.obsm['cell_embedding'] = z_mean
    adata2.obsm['niche_embedding'] = z_spatial
    adata2.write_h5ad(f"{save_dir}/nouse_spatial_decoder/adatas/{z}.h5ad")



'''
test data
https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/mouse_E11.5_embryo.h5ad
https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/mouse_E9.5_embryo.h5ad
'''