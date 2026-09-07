import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay, cKDTree
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError
import warnings
def contains_only_integers(arr):
    return np.all(arr % 1 == 0)


def _drop_degenerate_dims(coords, tol=1e-8):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"spatial coordinates should be 2D, got shape {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("spatial coordinates contain NaN or inf")

    ranges = np.ptp(coords, axis=0)
    scale = max(float(np.max(np.abs(coords))), 1.0)
    valid_dims = np.where(ranges > tol * scale)[0]
    if len(valid_dims) == 0:
        raise ValueError("all spatial coordinate dimensions are degenerate")
    if len(valid_dims) < coords.shape[1]:
        degenerate = [i for i in range(coords.shape[1]) if i not in valid_dims]
        warnings.warn(
            f"construct_graph: 检测到退化维度 {degenerate}，"
            f"已自动丢弃，仅使用维度 {valid_dims.tolist()} 构建空间图。"
        )
    return coords[:, valid_dims]


def _adj_from_simplices(simplices, n_cells):
    simplices = simplices[np.all(simplices < n_cells, axis=1)]
    rows = []
    cols = []
    for i in range(simplices.shape[1]):
        for j in range(i + 1, simplices.shape[1]):
            rows.append(simplices[:, i])
            cols.append(simplices[:, j])
            rows.append(simplices[:, j])
            cols.append(simplices[:, i])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _knn_adj(coords, n_neighbors=6):
    n_cells = coords.shape[0]
    if n_cells <= 1:
        return sp.csr_matrix((n_cells, n_cells), dtype=np.float32)

    k = min(n_neighbors + 1, n_cells)
    _, indices = cKDTree(coords).query(coords, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]

    neighbors = indices[:, 1:]
    rows = np.repeat(np.arange(n_cells), neighbors.shape[1])
    cols = neighbors.reshape(-1)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def construct_graph(adata, spatial_key="spatial"):
    """
    基于 Delaunay 三角剖分构建空间邻接图（稀疏版，大幅降低内存）。
    """
    coords = _drop_degenerate_dims(adata.obsm[spatial_key])

    n_cells = coords.shape[0]
    print(f"    Delaunay: {n_cells} 点, 维度 {coords.shape[1]}", end=" ", flush=True)

    if coords.shape[1] < 2:
        warnings.warn("construct_graph: 有效空间维度小于 2，改用 kNN 构图。")
        adj = _knn_adj(coords)
    else:
        try:
            tri = Delaunay(coords, qhull_options="QJ Qbb Qc Qz Q12")
            adj = _adj_from_simplices(tri.simplices, n_cells)
        except QhullError as error:
            warnings.warn(f"construct_graph: Delaunay 构图失败，改用 kNN 构图。原始错误: {error}")
            adj = _knn_adj(coords)

    print(f"→ 边数: {adj.nnz // 2}", flush=True)

    adata.obsp["spatial_connectivities"] = adj
    return adata

def preprocess_adj_sparse(adata):
    """对稀疏邻接矩阵做归一化: D^{-1/2} A D^{-1/2}"""
    if "spatial_connectivities" in adata.obsp:
        adj = adata.obsp["spatial_connectivities"].copy()
    elif "connectivities" in adata.obsp:
        adj = adata.obsp["connectivities"].copy()
    else:
        raise KeyError("找不到邻接矩阵，请先运行 construct_graph")

    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    # 加自环
    adj = adj + sp.eye(adj.shape[0], format="csr")

    # 归一化 D^{-1/2} A D^{-1/2}
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # 避免除零
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

    adata.obsm["adj_norm"] = adj_norm
    return adata

def get_spatial_input(adata):
    if isinstance(adata.X, np.ndarray):
        adata.obsm["spatial_input"]= csr_matrix(adata.X)
    else:
        adata.obsm["spatial_input"] = adata.X
    return adata


def get_feature_sparse(device, spatial_input):
    """将空间输入转换为稀疏特征矩阵（CSR 格式）。"""
    if sp.issparse(spatial_input):
        feat_sparse = spatial_input.tocsr()
    else:
        feat_sparse = csr_matrix(spatial_input)
    return feat_sparse
