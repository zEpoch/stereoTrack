import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import itertools


class StereoTrackGraphDataset(Dataset):
    def __init__(self, g, adata):
        self.g = g
        self.n_nodes = g.number_of_nodes()
        self.adata = adata

    def __len__(self):
        return self.n_nodes

    def __getitem__(self, idx):
        return idx


class StereoTrackDataLoader:
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
            if isinstance(indices, torch.Tensor):
                indices = indices.flatten()
            blocks = {
                "single": indices,
                "spatial": self.sampler.sample_blocks(self.dataset.g, indices),
            }
            yield blocks

    def __len__(self):
        return len(self.dataloader)


class MultiSliceGraphDataset(Dataset):
    """
    Dataset for multiple slices (similar to FuseMap's CustomGraphDataset)
    
    Parameters
    ----------
    g : dgl.DGLGraph
        The graph structure for this slice
    adata : AnnData
        The anndata object containing spatial data
    
    Examples
    --------
    >>> dataset = MultiSliceGraphDataset(g, adata)
    """
    def __init__(self, g, adata):
        self.g = g
        self.n_nodes = g.number_of_nodes()
        self.adata = adata

    def __len__(self):
        return self.n_nodes

    def __getitem__(self, idx):
        return idx


class MultiSliceDataLoader:

    def __init__(self, dataset_all, sampler, batch_size, shuffle, n_slices, drop_last):
        self.dataset_all = dataset_all
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_slices = n_slices

        self.dataloader = []
        for i in range(n_slices):
            self.dataloader.append(
                DataLoader(
                    self.dataset_all[i],
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
            )

    def __iter__(self):
        dataloader_iters = [iter(dl) for dl in self.dataloader]
        active_loaders = list(range(self.n_slices))
        
        while active_loaders:
            for slice_idx in list(active_loaders):
                try:
                    indices = next(dataloader_iters[slice_idx])
                    if isinstance(indices, torch.Tensor):
                        indices = indices.flatten().cpu().numpy()
                    elif not isinstance(indices, np.ndarray):
                        indices = np.array(indices)
                    
                    n_nodes = self.dataset_all[slice_idx].n_nodes
                    if indices.max() >= n_nodes:
                        raise ValueError(
                            f"Index {indices.max()} out of range for slice {slice_idx} "
                            f"with {n_nodes} nodes"
                        )
                    
                    blocks = {
                        "slice_idx": slice_idx,
                        "single": indices,
                        "spatial": self.sampler.sample_blocks(
                            self.dataset_all[slice_idx].g, torch.from_numpy(indices)
                        ),
                    }
                    yield blocks
                except StopIteration:
                    active_loaders.remove(slice_idx)

    def __len__(self):
        return sum([len(dl) for dl in self.dataloader])


def construct_data(adata, model=None):

    adj_coo = adata.obsm["adj_normalized"].tocoo()
    adj = adata.obsm["adj_normalized"]
    g = dgl.graph((adj_coo.row, adj_coo.col))
    
    return adj, g


def get_feature_sparse(device, feature):

    return feature.copy()

def construct_data_multislice(adatas):

    adj_all = []
    g_all = []
    
    for adata in adatas:
        adj_coo = adata.obsm["adj_normalized"].tocoo()
        adj_all.append(adata.obsm["adj_normalized"])
        g_all.append(dgl.graph((adj_coo.row, adj_coo.col)))
    
    return adj_all, g_all
