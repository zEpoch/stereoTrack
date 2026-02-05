import torch
import torch.nn.functional as F
import torch.distributions as D
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from pathlib import Path
import dgl.dataloading as dgl_dataload
from stereotrack.single_time.dataset import (
    StereoTrackGraphDataset,
    StereoTrackDataLoader,
    construct_data,
    get_feature_sparse,
    construct_mask,
)
from stereotrack.single_time.process import (
    preprocess_adata_with_multislice, 
    construct_graph, 
    preprocess_adj_sparse, 
    get_spatial_input,
    get_feature_sparse
)
from stereotrack.single_time.model import StereoTrackModel

def compute_loss(model, x, adj, x_recon, z_distribution, mask):
    """
    Compute reconstruction and KL divergence loss
    
    Parameters
    ----------
    model : StereoTrackModel
        The model
    x : torch.Tensor
        Input features
    adj : torch.Tensor
        Adjacency matrix
    x_recon : torch.Tensor
        Reconstructed features
    z_distribution : torch.distributions.Normal
        Latent distribution
    mask : torch.Tensor
        Training/validation mask
        
    Returns
    -------
    loss : torch.Tensor
        Total loss
    recon_loss : torch.Tensor
        Reconstruction loss
    kl_loss : torch.Tensor
        KL divergence loss
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon[mask], x[mask])
    
    # KL divergence
    kl_loss = D.kl_divergence(z_distribution, D.Normal(0.0, 1.0)).sum(dim=1).mean() / x.shape[1]
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_stereotrack(adata, save_dir, hidden_dim=512, latent_dim=64, 
                     dropout_rate=0.2, n_epochs=50, batch_size=64, 
                     learning_rate=0.001, patience=5):
    """
    Train StereoTrack model
    
    Parameters
    ----------
    adata : AnnData
        Input spatial data (already preprocessed with construct_graph)
    save_dir : str
        Directory to save outputs
    hidden_dim : int
        Hidden dimension
    latent_dim : int
        Latent dimension
    dropout_rate : float
        Dropout rate
    n_epochs : int
        Number of epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    patience : int
        Early stopping patience
        
    Examples
    --------
    >>> train_stereotrack(adata, './output')
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{save_dir}/trained_model").mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess if needed
    if "adj_normalized" not in adata.obsm:
        logging.info("Preprocessing data...")
        adata.var.index = [g.upper() for g in adata.var.index]
        preprocess_adata_with_multislice([adata])
        construct_graph([adata])
        preprocess_adj_sparse([adata])
        get_spatial_input([adata])
    
    # Construct graph and data
    adj, g = construct_data(adata)
    features = get_feature_sparse(device, adata.obsm["spatial_input"])
    
    # Create dataset and dataloader
    dataset = StereoTrackGraphDataset(g, adata)
    train_mask, val_mask = construct_mask(dataset, g)
    
    dataloader = StereoTrackDataLoader(
        dataset,
        dgl_dataload.MultiLayerFullNeighborSampler(1),
        batch_size,
        shuffle=True,
        drop_last=False
    )
    
    dataloader_test = StereoTrackDataLoader(
        dataset,
        dgl_dataload.MultiLayerFullNeighborSampler(1),
        batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Initialize model
    var_names = list(adata.var.index)
    all_genes = sorted(var_names)
    
    model = StereoTrackModel(
        adata.n_vars,
        hidden_dim,
        latent_dim,
        dropout_rate,
        var_names,
        all_genes
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for blocks in dataloader:
            row_indices = list(blocks["single"])
            col_indices = list(blocks["spatial"][1])
            
            # Get batch data
            batch_features = torch.FloatTensor(features[row_indices].toarray()).to(device)
            adj_block = torch.FloatTensor(
                adj[row_indices, :].tocsc()[:, col_indices].todense()
            ).to(device)
            
            batch_train_mask = train_mask[row_indices]
            
            # Forward pass
            x_recon, z_dist, z_spatial, z_mean = model(batch_features, adj_block)
            
            # Compute loss
            loss, recon_loss, kl_loss = compute_loss(
                model, batch_features, adj_block, x_recon, z_dist, batch_train_mask
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for blocks in dataloader_test:
                row_indices = list(blocks["single"])
                col_indices = list(blocks["spatial"][1])
                
                batch_features = torch.FloatTensor(features[row_indices].toarray()).to(device)
                adj_block = torch.FloatTensor(
                    adj[row_indices, :].tocsc()[:, col_indices].todense()
                ).to(device)
                
                batch_val_mask = val_mask[row_indices]
                
                x_recon, z_dist, z_spatial, z_mean = model(batch_features, adj_block)
                loss, _, _ = compute_loss(
                    model, batch_features, adj_block, x_recon, z_dist, batch_val_mask
                )
                
                val_loss += loss.item()
        
        # Logging
        avg_train_loss = train_loss / len(dataloader)
        avg_val_loss = val_loss / len(dataloader_test)
        
        logging.info(
            f"Epoch {epoch+1}/{n_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/trained_model/stereotrack_best.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break
    
    # Extract embeddings
    model.load_state_dict(torch.load(f"{save_dir}/trained_model/stereotrack_best.pt"))
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for blocks in dataloader_test:
            row_indices = list(blocks["single"])
            col_indices = list(blocks["spatial"][1])
            
            batch_features = torch.FloatTensor(features[row_indices].toarray()).to(device)
            adj_block = torch.FloatTensor(
                adj[row_indices, :].tocsc()[:, col_indices].todense()
            ).to(device)
            
            _, _, _, z_mean = model(batch_features, adj_block)
            embeddings.append(z_mean.cpu().numpy())
    
    import numpy as np
    embeddings = np.vstack(embeddings)
    
    # Save embeddings to adata
    adata.obsm['stereotrack_embedding'] = embeddings
    adata.write_h5ad(f"{save_dir}/stereotrack_result.h5ad")
    
    logging.info("Training completed!")
    
    return model, adata
