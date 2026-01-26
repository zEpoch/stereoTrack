"""
Graph Neural Network models for expression matrix embedding and imputation.

This module provides a unified GAT-based autoencoder for both embedding and imputation tasks.
All computations are optimized for GPU with batch training support.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from typing import Optional, Union, Tuple
import warnings
from anndata import AnnData
from .utils import build_spatial_graph, validate_spatial_coords

try:
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("torch_geometric not available. GNN models will not work. "
                  "Install with: pip install torch-geometric")


def get_default_device(device: str = "auto") -> str:
    """
    Get default device for computation.
    
    Parameters
    ----------
    device
        Device string: 'auto', 'cpu', or 'cuda'.
        (Default: 'auto')
    
    Returns
    -------
    str
        Device string ('cpu' or 'cuda').
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class SpatialGATAutoEncoder(nn.Module):
    """
    Unified GAT-based autoencoder for spatial transcriptomics data.
    
    This model performs both:
    - Embedding: Uses the latent representation (middle layer) as cell embeddings
    - Imputation: Uses the reconstructed expression matrix (output) for imputation
    
    All layers use GAT (Graph Attention Network) for better spatial information integration.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list = [256, 128, 64],
        latent_dim: int = 32,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        gat_heads: int = 4
    ):
        """
        Parameters
        ----------
        n_features
            Number of input features (genes).
        hidden_dims
            List of hidden layer dimensions for encoder.
            (Default: [256, 128, 64])
        latent_dim
            Dimension of latent representation (embedding dimension).
            (Default: 32)
        dropout
            Dropout rate.
            (Default: 0.1)
        use_batch_norm
            Whether to use batch normalization.
            (Default: True)
        gat_heads
            Number of attention heads for GAT layers.
            (Default: 4)
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GNN models. "
                            "Install with: pip install torch-geometric")
        
        super(SpatialGATAutoEncoder, self).__init__()
        
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.gat_heads = gat_heads
        
        # Encoder: GAT layers
        self.encoder_convs = nn.ModuleList()
        encoder_dims = [n_features] + hidden_dims + [latent_dim]
        
        for i in range(len(encoder_dims) - 1):
            in_dim = encoder_dims[i]
            out_dim = encoder_dims[i + 1]
            
            # For intermediate layers, use multi-head GAT
            # For last layer (latent), use single head
            if i < len(encoder_dims) - 2:
                # Multi-head GAT for hidden layers
                out_dim_per_head = out_dim // gat_heads
                self.encoder_convs.append(
                    GATConv(in_dim, out_dim_per_head, heads=gat_heads, dropout=dropout, concat=True)
                )
            else:
                # Single-head GAT for latent layer
                self.encoder_convs.append(
                    GATConv(in_dim, out_dim, heads=1, dropout=dropout, concat=False)
                )
        
        # Decoder: GAT layers (reverse structure)
        self.decoder_convs = nn.ModuleList()
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [n_features]
        
        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i]
            out_dim = decoder_dims[i + 1]
            
            if i < len(decoder_dims) - 2:
                # Multi-head GAT for hidden layers
                out_dim_per_head = out_dim // gat_heads
                self.decoder_convs.append(
                    GATConv(in_dim, out_dim_per_head, heads=gat_heads, dropout=dropout, concat=True)
                )
            else:
                # Single-head GAT for output layer
                self.decoder_convs.append(
                    GATConv(in_dim, out_dim, heads=1, dropout=dropout, concat=False)
                )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        if use_batch_norm:
            self.encoder_batch_norms = nn.ModuleList([
                nn.BatchNorm1d(dim) for dim in hidden_dims + [latent_dim]
            ])
            self.decoder_batch_norms = nn.ModuleList([
                nn.BatchNorm1d(dim) for dim in hidden_dims[::-1]
            ])
    
    def encode(self, x, edge_index):
        """
        Encode expression matrix to latent representation (embedding).
        
        Parameters
        ----------
        x
            Node features (n_cells, n_features).
        edge_index
            Graph edge indices (2, n_edges).
        
        Returns
        -------
        torch.Tensor
            Cell embeddings (n_cells, latent_dim).
        """
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x, edge_index)
            if i < len(self.encoder_convs) - 1:
                if self.use_batch_norm:
                    x = self.encoder_batch_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        return x
    
    def decode(self, z, edge_index):
        """
        Decode latent representation back to expression matrix.
        
        Parameters
        ----------
        z
            Latent representation (n_cells, latent_dim).
        edge_index
            Graph edge indices (2, n_edges).
        
        Returns
        -------
        torch.Tensor
            Reconstructed expression matrix (n_cells, n_features).
        """
        for i, conv in enumerate(self.decoder_convs):
            z = conv(z, edge_index)
            if i < len(self.decoder_convs) - 1:
                if self.use_batch_norm and i < len(self.decoder_batch_norms):
                    z = self.decoder_batch_norms[i](z)
                z = self.activation(z)
                z = self.dropout(z)
        return z
    
    def forward(self, x, edge_index):
        """
        Forward pass: encode -> decode (for imputation).
        
        Parameters
        ----------
        x
            Expression matrix (n_cells, n_features).
        edge_index
            Graph edge indices (2, n_edges).
        
        Returns
        -------
        torch.Tensor
            Reconstructed expression matrix (n_cells, n_features).
        """
        z = self.encode(x, edge_index)
        x_recon = self.decode(z, edge_index)
        return x_recon


def prepare_graph_data(
    expression_matrix: np.ndarray,
    adjacency_matrix: Union[np.ndarray, csr_matrix],
    device: str = "auto"
) -> Data:
    """
    Prepare PyTorch Geometric Data object from expression matrix and adjacency.
    All data is moved to the specified device (GPU by default if available).
    
    Parameters
    ----------
    expression_matrix
        Gene expression matrix (n_cells, n_genes).
    adjacency_matrix
        Spatial adjacency matrix (n_cells, n_cells).
    device
        Device to use ('auto', 'cpu', or 'cuda').
        (Default: 'auto')
    
    Returns
    -------
    torch_geometric.data.Data
        PyG Data object on the specified device.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required.")
    
    device = get_default_device(device)
    
    # Convert to dense if sparse
    if isinstance(adjacency_matrix, csr_matrix):
        adjacency_matrix = adjacency_matrix.toarray()
    
    # Convert to edge_index format (on CPU first, then move to device)
    edge_index = torch.from_numpy(np.array(np.where(adjacency_matrix > 0))).long()
    
    # Convert expression to tensor
    x = torch.FloatTensor(expression_matrix)
    
    data = Data(x=x, edge_index=edge_index)
    data = data.to(device)
    
    return data


def train_gnn_imputation(
    expression_matrix: np.ndarray,
    adjacency_matrix: Union[np.ndarray, csr_matrix],
    model: Optional[SpatialGATAutoEncoder] = None,
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: Optional[int] = None,
    train_mask: Optional[np.ndarray] = None,
    val_mask: Optional[np.ndarray] = None,
    device: str = "auto",
    verbose: bool = True
) -> Tuple[SpatialGATAutoEncoder, dict]:
    """
    Train GAT autoencoder model for expression matrix imputation (MSE reconstruction).
    Supports batch training for memory efficiency.
    
    Parameters
    ----------
    expression_matrix
        Gene expression matrix (n_cells, n_genes).
    adjacency_matrix
        Spatial adjacency matrix (n_cells, n_cells).
    model
        Pre-initialized model. If None, creates a new model.
        (Default: None)
    n_epochs
        Number of training epochs.
        (Default: 100)
    lr
        Learning rate.
        (Default: 0.001)
    batch_size
        Batch size for training. If None, uses all cells (full batch).
        For large datasets, use batch_size to reduce memory usage.
        (Default: None)
    train_mask
        Boolean mask for training cells. If None, uses all cells.
        (Default: None)
    val_mask
        Boolean mask for validation cells. If None, uses 20% of cells.
        (Default: None)
    device
        Device to use ('auto', 'cpu', or 'cuda').
        (Default: 'auto')
    verbose
        Whether to print training progress.
        (Default: True)
    
    Returns
    -------
    tuple
        (trained_model, training_history)
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required.")
    
    device = get_default_device(device)
    
    # Prepare data (moved to device)
    data = prepare_graph_data(expression_matrix, adjacency_matrix, device=device)
    
    # Create model if not provided
    if model is None:
        model = SpatialGATAutoEncoder(
            n_features=expression_matrix.shape[1],
            hidden_dims=[256, 128, 64],
            latent_dim=32,
            dropout=0.1,
            use_batch_norm=True
        )
    model = model.to(device)
    
    # Setup masks
    n_cells = expression_matrix.shape[0]
    if train_mask is None:
        train_mask = np.ones(n_cells, dtype=bool)
        # Use 80% for training, 20% for validation
        val_indices = np.random.choice(n_cells, int(0.2 * n_cells), replace=False)
        train_mask[val_indices] = False
    
    if val_mask is None:
        val_mask = ~train_mask
    
    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    # Determine batch size
    if batch_size is None:
        batch_size = n_cells  # Full batch
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (full graph, but loss computed on batches)
        x_recon = model(data.x, data.edge_index)
        
        # Batch training: compute loss on batches
        train_loss_batches = []
        train_indices = torch.where(train_mask)[0]
        
        if len(train_indices) > 0:
            # Shuffle training indices
            shuffled_indices = train_indices[torch.randperm(len(train_indices), device=device)]
            
            for start in range(0, len(shuffled_indices), batch_size):
                batch_idx = shuffled_indices[start:start + batch_size]
                if len(batch_idx) > 0:
                    batch_loss = criterion(x_recon[batch_idx], data.x[batch_idx])
                    train_loss_batches.append(batch_loss)
            
            if len(train_loss_batches) > 0:
                train_loss = sum(train_loss_batches) / len(train_loss_batches)
            else:
                train_loss = torch.tensor(0.0, device=device)
        else:
            train_loss = torch.tensor(0.0, device=device)
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation (full batch for validation)
        model.eval()
        with torch.no_grad():
            x_recon_val = model(data.x, data.edge_index)
            val_loss = criterion(x_recon_val[val_mask], data.x[val_mask])
        
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}")
    
    return model, history


def get_gnn_embedding(
    expression_matrix: np.ndarray,
    adjacency_matrix: Union[np.ndarray, csr_matrix],
    model: Optional[SpatialGATAutoEncoder] = None,
    latent_dim: int = 32,
    device: str = "auto"
) -> np.ndarray:
    """
    Get GNN-based cell embeddings from the latent representation.
    
    This function uses the encode() method of the GAT autoencoder to get embeddings.
    If no model is provided, it creates an untrained model (embeddings may not be meaningful).
    
    Parameters
    ----------
    expression_matrix
        Gene expression matrix (n_cells, n_genes).
    adjacency_matrix
        Spatial adjacency matrix (n_cells, n_cells).
    model
        Pre-trained GAT autoencoder model. If None, creates an untrained model.
        (Default: None)
    latent_dim
        Dimension of output embedding (latent dimension).
        (Default: 32)
    device
        Device to use ('auto', 'cpu', or 'cuda').
        (Default: 'auto')
    
    Returns
    -------
    np.ndarray
        Cell embeddings (n_cells, latent_dim).
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required.")
    
    device = get_default_device(device)
    
    # Prepare data (moved to device)
    data = prepare_graph_data(expression_matrix, adjacency_matrix, device=device)
    
    # Create model if not provided
    if model is None:
        model = SpatialGATAutoEncoder(
            n_features=expression_matrix.shape[1],
            hidden_dims=[256, 128, 64],
            latent_dim=latent_dim,
            dropout=0.1,
            use_batch_norm=True
        )
        warnings.warn("Using untrained model. Consider training first for better embeddings.")
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Get embedding from latent representation
        embeddings = model.encode(data.x, data.edge_index)
    
    return embeddings.cpu().numpy()


def get_imputed_expression(
    expression_matrix: np.ndarray,
    adjacency_matrix: Union[np.ndarray, csr_matrix],
    model: SpatialGATAutoEncoder,
    device: str = "auto"
) -> np.ndarray:
    """
    Get imputed expression matrix from trained GAT autoencoder.
    
    Parameters
    ----------
    expression_matrix
        Gene expression matrix (n_cells, n_genes).
    adjacency_matrix
        Spatial adjacency matrix (n_cells, n_cells).
    model
        Trained GAT autoencoder model.
    device
        Device to use ('auto', 'cpu', or 'cuda').
        (Default: 'auto')
    
    Returns
    -------
    np.ndarray
        Imputed expression matrix (n_cells, n_genes).
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required.")
    
    device = get_default_device(device)
    
    # Prepare data (moved to device)
    data = prepare_graph_data(expression_matrix, adjacency_matrix, device=device)
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Get imputed expression (reconstructed output)
        imputed = model(data.x, data.edge_index)
    
    return imputed.cpu().numpy()



def run_3d_gnn_imputation(
    adata: AnnData,
    spatial_key: str = "X_spatial",
    layer: str = "X",
    gnn_key: str = "X_gnn",
    imputed_layer: str = "X_gnn_imputed",
    graph_key: str = "spatial_graph",
    n_epochs: int = 100,
    batch_size: Optional[int] = None,
    latent_dim: int = 64,
    device: str = "auto",
    verbose: bool = True,
) -> AnnData:
    """
    High-level pipeline to run 3D GNN-based embedding and imputation.

    This function:
    1. Validates 3D spatial coordinates in ``adata.obsm[spatial_key]``
    2. Builds a Delaunay-based spatial graph (3D)
    3. Trains a GAT autoencoder (MSE reconstruction)
    4. Writes:
       - GNN embedding to ``adata.obsm[gnn_key]``
       - Imputed expression to ``adata.layers[imputed_layer]``
       - Spatial graph to ``adata.obsp[graph_key]``

    Parameters
    ----------
    adata
        Input AnnData object containing 3D spatial transcriptomics data.
    spatial_key
        Key in ``adata.obsm`` containing spatial coordinates (n_cells, 3).
        (Default: "X_spatial")
    layer
        Expression layer to use:
        - "X": uses ``adata.X``
        - otherwise: uses ``adata.layers[layer]``
        (Default: "X")
    gnn_key
        Key in ``adata.obsm`` to store GNN embeddings.
        (Default: "X_gnn")
    imputed_layer
        Key in ``adata.layers`` to store imputed expression matrix.
        (Default: "X_gnn_imputed")
    graph_key
        Key in ``adata.obsp`` to store Delaunay-based spatial graph (adjacency matrix).
        (Default: "spatial_graph")
    n_epochs
        Number of training epochs for the GAT autoencoder.
        (Default: 100)
    batch_size
        Batch size for training. If None, uses full batch.
        (Default: None)
    latent_dim
        Dimension of latent representation (embedding dimension).
        (Default: 32)
    device
        Device to use ("auto", "cpu", or "cuda").
        "auto" uses GPU if available.
        (Default: "auto")
    verbose
        Whether to print training progress.
        (Default: True)

    Returns
    -------
    AnnData
        The same AnnData object with updated ``obsm``, ``layers`` and ``obsp``.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError(
            "torch_geometric is required for GNN models. "
            "Install with: pip install torch-geometric"
        )

    # 1. Validate and get 3D spatial coordinates
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial coordinates not found in adata.obsm['{spatial_key}']."
        )
    coords = adata.obsm[spatial_key]
    dim = validate_spatial_coords(coords)
    if dim != 3:
        raise ValueError(
            f"run_3d_gnn_imputation expects 3D coordinates, got {dim}D in adata.obsm['{spatial_key}']."
        )

    # 2. Build 3D Delaunay-based spatial graph
    if verbose:
        print(f"Building 3D Delaunay spatial graph from '{spatial_key}'...")
    spatial_adj = build_spatial_graph(coords, method="delaunay")
    adata.obsp[graph_key] = spatial_adj

    # 3. Prepare expression matrix
    if layer == "X":
        expr = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        expr = adata.layers[layer]
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.array(expr)

    # 4. Train GAT autoencoder
    if verbose:
        print("Training SpatialGATAutoEncoder for 3D GNN imputation...")
    model, history = train_gnn_imputation(
        expression_matrix=expr,
        adjacency_matrix=spatial_adj,
        model=None,
        n_epochs=n_epochs,
        lr=0.001,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

    # 5. Get embedding and imputed expression
    if verbose:
        print("Computing GNN embeddings and imputed expression...")
    embedding = get_gnn_embedding(
        expression_matrix=expr,
        adjacency_matrix=spatial_adj,
        model=model,
        latent_dim=latent_dim,
        device=device,
    )
    imputed = get_imputed_expression(
        expression_matrix=expr,
        adjacency_matrix=spatial_adj,
        model=model,
        device=device,
    )

    # 6. Write results back to AnnData
    adata.obsm[gnn_key] = embedding
    adata.layers[imputed_layer] = imputed

    if verbose:
        print(
            f"GNN embedding stored in adata.obsm['{gnn_key}'], "
            f"imputed expression stored in adata.layers['{imputed_layer}'], "
            f"spatial graph stored in adata.obsp['{graph_key}']."
        )

    return adata
