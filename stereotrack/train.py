import torch
import torch.nn.functional as F
import torch.distributions as D
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import dgl.dataloading as dgl_loader
from stereotrack.process import (
    construct_graph, 
    preprocess_adj_sparse, 
    get_spatial_input,
)
from stereotrack.model import StereoTrackModel
from stereotrack.dataset import (
    MultiSliceGraphDataset,
    MultiSliceDataLoader,
    construct_data_multislice
)


def compute_loss(model, x, adj, x_recon, z_distribution, z_spatial, z_mean, beta = 1.0):
    recon_loss = F.mse_loss(x_recon, x)
    kl_loss = D.kl_divergence(z_distribution, D.Normal(0.0, 1.0)).sum(dim=1).mean() / x.shape[1]
    lambda_recon = 1.0
    lambda_kl = beta
    total_loss = lambda_recon * recon_loss + lambda_kl * kl_loss
    return total_loss, recon_loss, kl_loss

def train_stereotrack_multislice(adatas, 
                                 save_dir, 
                                 hidden_dim=512, 
                                 latent_dim=64, 
                                 dropout_rate=0.2, 
                                 n_epochs=50, 
                                 learning_rate=0.001, 
                                 batch_size=512,
                                 patience=5):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Path(f"{save_dir}/trained_model").mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Number of slices: {len(adatas)}")
    
    logging.info("Preprocessing all slices...")
    for i, adata in enumerate(adatas):
        adata.var.index = [g.upper() for g in adata.var.index]
        logging.info(f"Slice {i} before: {adata.n_obs} cells, {adata.n_vars} genes")
    
    processed_adatas = []
    for i, adata in enumerate(adatas):
        adata = preprocess(adata)
        adata = construct_graph(adata)
        adata = preprocess_adj_sparse(adata)
        adata = get_spatial_input(adata)
        processed_adatas.append(adata)
    
    adatas = processed_adatas
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
    
    logging.info(f"Dataloader created with batch_size={batch_size}")
    
    features_all = [get_feature_sparse(device, adata.obsm["spatial_input"]) for adata in adatas]
    
    model = StereoTrackModel(
        len(all_genes),
        hidden_dim,
        latent_dim,
        dropout_rate,
        all_genes,
        all_genes
    )
    model.to(device)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
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
        
        # Iterate through batches (each block is one slice's batch)
        for blocks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            # Get slice index and data
            slice_idx = blocks["slice_idx"]
            indices = blocks["single"]
            
            # Get features and adj for this slice
            features = features_all[slice_idx]
            adj = adj_all[slice_idx]
            
            # Convert to dense and move to device
            features_batch = torch.FloatTensor(
                features[indices].toarray() if sp.issparse(features) else features[indices]
            ).to(device)
            
            adj_batch = torch.FloatTensor(
                adj[indices][:, indices].todense() if sp.issparse(adj) else adj[indices][:, indices]
            ).to(device)
            
            # Forward pass
            x_recon, z_dist, z_spatial, z_mean = model(features_batch, adj_batch)
            
            # Compute loss
            loss, recon_loss, kl_loss = compute_loss(
                model, features_batch, adj_batch, x_recon, z_dist, z_spatial, z_mean
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
        
        # Average losses over all batches
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        
        logging.info(
            f"Epoch {epoch+1}/{n_epochs} - "
            f"Loss: {avg_loss:.4f}, "
            f"Recon: {avg_recon:.4f}, "
            f"KL: {avg_kl:.4f}"
        )
        
        # Scheduler step
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/trained_model/stereotrack_best.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break
    
    # Extract embeddings for each slice
    model.load_state_dict(torch.load(f"{save_dir}/trained_model/stereotrack_best.pt"))
    model.eval()
    
    logging.info("Extracting embeddings for each slice...")
    for slice_idx, adata in enumerate(adatas):
        adj = adj_all[slice_idx]
        features = features_all[slice_idx]
        
        with torch.no_grad():
            adj_tensor = torch.FloatTensor(adj.todense() if sp.issparse(adj) else adj).to(device)
            features_tensor = torch.FloatTensor(
                features.toarray() if sp.issparse(features) else features
            ).to(device)
            _, _, _, z_mean = model(features_tensor, adj_tensor)
            embeddings = z_mean.cpu().numpy()
        
        # Save to adata
        adata.obsm['stereotrack_embedding'] = embeddings
        adata.write_h5ad(f"{save_dir}/stereotrack_slice_{slice_idx}_result.h5ad")
        logging.info(f"Slice {slice_idx}: embedding shape {embeddings.shape}")
    
    logging.info("Training completed for all slices!")
    
    return model, adatas
