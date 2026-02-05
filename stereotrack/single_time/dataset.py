import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp


class StereoTrackGraphDataset(Dataset):
    """
    Dataset for single 3D spatial data similar to FuseMap's CustomGraphDataset
    
    Parameters
    ----------
    g : dgl.DGLGraph
        The graph structure
    adata : AnnData
        The anndata object containing spatial data
    
    Examples
    --------
    >>> dataset = StereoTrackGraphDataset(g, adata)
    """
    def __init__(self, g, adata):
        self.g = g
        self.n_nodes = g.number_of_nodes()
        self.adata = adata

    def __len__(self):
        return self.n_nodes

    def __getitem__(self, idx):
        return idx


class StereoTrackDataLoader:
    """
    DataLoader for StereoTrack that samples graph neighborhoods
    
    Parameters
    ----------
    dataset : StereoTrackGraphDataset
        The dataset to load
    sampler : dgl.dataloading.Sampler
        The sampling strategy
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle data
    drop_last : bool
        Whether to drop last incomplete batch
        
    Examples
    --------
    >>> dataloader = StereoTrackDataLoader(dataset, sampler, 64, True, False)
    """
    def __init__(self, dataset, sampler, batch_size, shuffle, drop_last):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def __iter__(self):
        for indices in self.dataloader:
            blocks = {
                "single": indices,
                "spatial": self.sampler.sample_blocks(self.dataset.g, indices),
            }
            yield blocks

    def __len__(self):
        return len(self.dataloader)


def construct_data(adata, model=None):
    """
    Construct graph and adjacency matrix for StereoTrack
    
    Parameters
    ----------
    adata : AnnData
        The anndata object with preprocessed spatial data
    model : nn.Module, optional
        Model with learnable adjacency (for scRNA-seq style data)
        
    Returns
    -------
    adj : scipy.sparse.csr_matrix or torch.Tensor
        Adjacency matrix
    g : dgl.DGLGraph
        DGL graph
        
    Examples
    --------
    >>> adj, g = construct_data(adata)
    """
    adj_coo = adata.obsm["adj_normalized"].tocoo()
    adj = adata.obsm["adj_normalized"]
    g = dgl.graph((adj_coo.row, adj_coo.col))
    
    return adj, g


def get_feature_sparse(device, feature):
    """
    Prepare features for StereoTrack
    
    Parameters
    ----------
    device : torch.device
        Device to use
    feature : scipy.sparse matrix
        Feature matrix
        
    Returns
    -------
    feature : scipy.sparse matrix
        Feature matrix (kept sparse for efficiency)
        
    Examples
    --------
    >>> features = get_feature_sparse(device, adata.obsm["spatial_input"])
    """
    return feature.copy()


def construct_mask(dataset, g, train_pct=0.85):
    """
    Construct training and validation masks
    
    Parameters
    ----------
    dataset : StereoTrackGraphDataset
        The dataset
    g : dgl.DGLGraph
        The graph
    train_pct : float
        Percentage of data for training
        
    Returns
    -------
    train_mask : torch.Tensor
        Training mask
    val_mask : torch.Tensor
        Validation mask
        
    Examples
    --------
    >>> train_mask, val_mask = construct_stereotrack_mask(dataset, g)
    """
    num_train = int(len(dataset) * train_pct)
    nodes_order = np.random.permutation(g.number_of_nodes())
    train_id = nodes_order[:num_train]
    
    train_mask = torch.zeros(len(dataset), dtype=torch.bool)
    train_mask[train_id] = True
    val_mask = ~train_mask
    
    return train_mask, val_mask
