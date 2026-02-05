import torch
import numpy as np
import logging
from pathlib import Path
import dgl.dataloading as dgl_dataload
from stereotrack.single_time.dataset import (
    StereoTrackGraphDataset,
    StereoTrackDataLoader,
    construct_data,
    get_feature_sparse
)
from stereotrack.single_time.model import StereoTrackModel
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import scipy
import anndata as ad
from scipy.spatial import Delaunay

def contains_only_integers(arr):
    return np.all(arr % 1 == 0)

def preprocess(adata):
    ### filter genes
    if isinstance(adata.X, np.ndarray):
        if contains_only_integers(adata.X):
            adata = adata[:, np.sum(adata.X, axis=0) > 1]
            adata = adata[:, np.max(adata.X, axis=0) > 1]
            ### filter cells
            adata = adata[np.sum(adata.X, axis=1) > 5]
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata)  # , target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, zero_center=False, max_value=10)
    if sp.issparse(adata.X):
        if contains_only_integers(adata.X.toarray()):
            adata = adata[:, np.sum(adata.X.toarray(), axis=0) > 1]
            adata = adata[:, np.max(adata.X.toarray(), axis=0) > 1]
            ### filter cells
            adata = adata[np.sum(adata.X, axis=1) > 5]

            adata.layers["counts"] = adata.X.copy()

            sc.pp.normalize_total(adata)  # , target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, zero_center=False, max_value=10)
    adata.var.index = [i.upper() for i in adata.var.index]
    return adata

def preprocess_adata_with_multislice(adatas):
    adata_all = []
    for adata in adatas:
        adata.var.index = [i.upper() for i in adata.var.index]

        ### keep unique genes
        _, indices = np.unique(adata.var.index, return_index=True)
        adata =adata[:, indices]
        adata = preprocess(adata)
        
        adata_all.append(adata)
    adata = ad.concat(adata_all)
    return adata

def construct_graph(adata,
                    spatial_key='spatial'
                    ):
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates not found in adata.obsm['{spatial_key}']")
    data = adata.obsm[spatial_key]
    tri = Delaunay(data)
    indptr, indices = tri.vertex_neighbor_vertices
    adjacency_matrix = csr_matrix(
        (np.ones_like(indices, dtype=np.float64), indices, indptr),
        shape=(data.shape[0], data.shape[0]),
    )
    adata.obsm["adj"] = adjacency_matrix
    return adata

def preprocess_adj_sparse(adata):
    if  "adj_normalized" not in adata.obsm:
        raise ValueError("Adjacency matrix not found in adata.obsm['adj']")
    adj = sp.coo_matrix(adata.obsm["adj"])
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_.dot(degree_mat_inv_sqrt)
        .transpose()
        .dot(degree_mat_inv_sqrt)
        .tocoo()
    )
    adata.obsm[
        "adj_normalized"
    ] = adj_normalized  # sparse_mx_to_torch_sparse_tensor(adj_normalized)
    adata.obsm["adj_normalized"] = adata.obsm["adj_normalized"].tocsr()
    return adata

def get_spatial_input(adata):
    if isinstance(adata.X, np.ndarray):
        adata.obsm["spatial_input"]= csr_matrix(adata.X)
    else:
        adata.obsm["spatial_input"] = adata.X
    return adata

def inference_stereotrack(adata, 
                          model_path, 
                          save_dir, 
                          hidden_dim=512, 
                          latent_dim=64, 
                          dropout_rate=0.2, 
                          batch_size=64):
    """
    Inference with trained StereoTrack model to get embeddings and imputation
    
    Parameters
    ----------
    adata : AnnData
        Input spatial data (already preprocessed with construct_graph)
    model_path : str
        Path to trained model file
    save_dir : str
        Directory to save outputs
    hidden_dim : int
        Hidden dimension (must match training)
    latent_dim : int
        Latent dimension (must match training)
    dropout_rate : float
        Dropout rate (must match training)
    batch_size : int
        Batch size for inference
        
    Returns
    -------
    adata : AnnData
        AnnData with embeddings and imputed values
        
    Examples
    --------
    >>> adata = inference_stereotrack(adata, './output/trained_model/stereotrack_best.pt', './output')
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Construct graph and data
    adj, g = construct_data(adata)
    features = get_feature_sparse(adata.obsm["spatial_input"])
    
    # Create dataset and dataloader
    dataset = StereoTrackGraphDataset(g, adata)
    
    dataloader = StereoTrackDataLoader(
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
    
    # Load trained weights
    logging.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Extract embeddings and reconstructions
    embeddings = []
    reconstructions = []
    
    logging.info("Extracting embeddings and reconstructions...")
    with torch.no_grad():
        for blocks in dataloader:
            row_indices = list(blocks["single"])
            col_indices = list(blocks["spatial"][1])
            
            batch_features = torch.FloatTensor(features[row_indices].toarray()).to(device)
            adj_block = torch.FloatTensor(
                adj[row_indices, :].tocsc()[:, col_indices].todense()
            ).to(device)
            
            x_recon, z_dist, z_spatial, z_mean = model(batch_features, adj_block)
            
            embeddings.append(z_mean.cpu().numpy())
            reconstructions.append(x_recon.cpu().numpy())
    
    # Concatenate results
    embeddings = np.vstack(embeddings)
    reconstructions = np.vstack(reconstructions)
    
    # Save to adata
    adata.obsm['stereotrack_embedding'] = embeddings
    adata.layers['stereotrack_imputation'] = reconstructions
    
    # Save results
    output_path = f"{save_dir}/stereotrack_inference_result.h5ad"
    adata.write_h5ad(output_path)
    logging.info(f"Results saved to {output_path}")
    
    # Save embeddings separately as numpy array
    np.save(f"{save_dir}/stereotrack_embeddings.npy", embeddings)
    np.save(f"{save_dir}/stereotrack_imputation.npy", reconstructions)
    
    logging.info("Inference completed!")
    logging.info(f"Embedding shape: {embeddings.shape}")
    logging.info(f"Imputation shape: {reconstructions.shape}")
    
    return adata
