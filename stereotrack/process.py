import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
import warnings
def contains_only_integers(arr):
    return np.all(arr % 1 == 0)

def construct_graph(adata, spatial_key="spatial"):
    """
    基于 Delaunay 三角剖分构建空间邻接图（稀疏版，大幅降低内存）。
    """
    coords = np.array(adata.obsm[spatial_key])

    # 检测退化维度（方差为 0），自动丢弃
    variances = np.var(coords, axis=0)
    valid_dims = np.where(variances > 1e-10)[0]
    if len(valid_dims) < coords.shape[1]:
        degenerate = [i for i in range(coords.shape[1]) if i not in valid_dims]
        warnings.warn(
            f"construct_graph: 检测到退化维度 {degenerate}（方差为0），"
            f"已自动丢弃，仅使用维度 {valid_dims.tolist()} 构建空间图。"
        )
        coords = coords[:, valid_dims]

    n_cells = coords.shape[0]
    print(f"    Delaunay: {n_cells} 点, 维度 {coords.shape[1]}", end=" ", flush=True)

    # Delaunay 三角剖分
    tri = Delaunay(coords)

    # 直接从三角形提取边 → 稀疏矩阵（不经过 dense！）
    # 每个单纯形的顶点两两相连
    simplices = tri.simplices  # (n_simplices, 3) for 2D
    rows = []
    cols = []
    for i in range(simplices.shape[1]):
        for j in range(i + 1, simplices.shape[1]):
            rows.append(simplices[:, i])
            cols.append(simplices[:, j])
            # 对称
            rows.append(simplices[:, j])
            cols.append(simplices[:, i])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)

    # 构建稀疏邻接矩阵，自动去重
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    # 去重：值 > 0 的都设为 1
    adj.data[:] = 1.0
    # 去除自环
    adj.setdiag(0)
    adj.eliminate_zeros()

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
