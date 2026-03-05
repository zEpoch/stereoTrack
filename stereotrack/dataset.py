import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import itertools

# ═══════════════════════════════════════════════════════════════════════════
# 3. 数据集：将所有切片的 patch 展平为一个大 Dataset
# ═══════════════════════════════════════════════════════════════════════════
class SpatialPatchDataset(Dataset):
    """
    将多切片数据展平：每个样本 = (slice_idx, cell_indices_in_batch)
    这里采用 "按 patch 采样" 策略：
      - 预先将每个切片的细胞按固定 patch_size 分成若干 patch
      - 每个 patch 作为一个样本，DataLoader 的 batch_size=1（每次取一个 patch）
        或者 batch_size=N（取 N 个 patch 拼接）
    
    这样 DistributedSampler 可以自动在多卡间均匀分配 patch。
    """

    def __init__(self, adatas, adj_all, features_all, patch_size=4096):
        super().__init__()
        self.adatas = adatas
        self.adj_all = adj_all
        self.features_all = features_all
        self.patch_size = patch_size

        # 构建 (slice_idx, start, end) 索引
        self.patches = []
        for s_idx, adata in enumerate(adatas):
            n_cells = adata.shape[0]
            for start in range(0, n_cells, patch_size):
                end = min(start + patch_size, n_cells)
                self.patches.append((s_idx, start, end))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        s_idx, start, end = self.patches[idx]
        cell_idx = np.arange(start, end)

        # 从稀疏矩阵中取子集 → dense numpy
        feat_sub = self.features_all[s_idx][cell_idx]
        adj_sub = self.adj_all[s_idx][cell_idx][:, cell_idx]

        if sp.issparse(feat_sub):
            feat_sub = feat_sub.toarray()
        if sp.issparse(adj_sub):
            adj_sub = adj_sub.toarray()

        feat_sub = np.asarray(feat_sub, dtype=np.float32)
        adj_sub = np.asarray(adj_sub, dtype=np.float32)

        return {
            "features": torch.from_numpy(feat_sub),
            "adj": torch.from_numpy(adj_sub),
            "slice_idx": s_idx,
            "cell_start": start,
            "cell_end": end,
        }


def patch_collate_fn(batch):
    """
    每次只取 1 个 patch（因为不同 patch 大小可能不同，且 adj 是方阵）。
    如果需要多 patch 拼接，需要 block-diagonal 拼接 adj。
    这里简单起见 batch_size=1。
    """
    assert len(batch) == 1, "当前实现每次只处理 1 个 patch，请设置 DataLoader batch_size=1"
    item = batch[0]
    return {
        "features": item["features"],
        "adj": item["adj"],
        "slice_idx": item["slice_idx"],
        "cell_start": item["cell_start"],
        "cell_end": item["cell_end"],
    }

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
