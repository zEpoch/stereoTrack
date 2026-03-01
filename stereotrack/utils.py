import ot
import torch
import random
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import norm
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from numpy.random import RandomState
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Optional, Tuple, Union
from sklearn.metrics.pairwise import euclidean_distances

def seed_all(seed=42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def setup_logging(log_file=None):

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def nearest_neighbors(coord, coords, n_neighbors=5):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(coords)
    _, neighs = neigh.kneighbors(np.atleast_2d(coord))
    return neighs

def kmeans_centers(coords: np.ndarray, n_clusters: int = 2) -> np.ndarray:

    cell_coordinates = coords
    kmeans = KMeans(n_clusters, random_state=8)
    kmeans.fit(cell_coordinates)
    cluster_centers = kmeans.cluster_centers_
    print("kmeans cluster centers:")
    list(map(print, cluster_centers))
    return cluster_centers

def set_start_cells(
    adata: AnnData,
    select_way: Literal["coordinates", "cell_type"],
    cell_type: Optional[str] = None,
    start_point: Optional[Tuple[int, int]] = None,
    spatial_key: str = "spatial",
    split: bool = False,
    n_clusters: int = 2,
    n_neigh: int = 5,
) -> list:
    if select_way == "coordinates":
        if start_point is None:
            raise ValueError(
                "`start_point` must be specified in the 'coordinates' mode."
            )

        start_cells = nearest_neighbors(start_point, adata.obsm[spatial_key], n_neigh)[
            0
        ]

        if cell_type is not None:
            type_cells = np.where(adata.obs["cluster"] == cell_type)[0]
            start_cells = set(start_cells).intersection(set(type_cells))

    elif select_way == "cell_type":
        if cell_type is None:
            raise ValueError("in 'cell_type' mode, `cell_type` cannot be None.")

        start_cells = np.where(adata.obs["cluster"] == cell_type)[0]

        if split:
            mask = adata.obs["cluster"] == cell_type
            cell_coords = adata.obsm[spatial_key][mask]
            cluster_centers = kmeans_centers(cell_coords, n_clusters=n_clusters)

            select_cluster_coords = adata.obsm[spatial_key].copy()
            select_cluster_coords[np.logical_not(mask)] = 1e10
            start_cells = nearest_neighbors(
                cluster_centers, select_cluster_coords, n_neigh
            ).flatten()
    else:
        raise ValueError("`select_way` must choose from 'coordinates' or 'cell_type'.")

    return list(start_cells)
    
def get_ot_matrix(
    adata: AnnData,
    data_type: str = 'spatial',
    alpha1: float = 0.5,
    alpha2: float = 0.5,
    use_rep: str = 'cell_embedding',
    spatial_key: str = 'X_spatial',
    lambd: float = 1e-1,
    numItermax: int = 1000,
    use_gpu: bool = False,
) -> np.ndarray:

    if use_rep not in adata.obsm:
        raise ValueError(f"use_rep '{use_rep}' is not in adata.obsm")
    
    embedding = np.ascontiguousarray(adata.obsm[use_rep], dtype=np.float64)

    def _compute_dist_matrix(X: np.ndarray) -> np.ndarray:
        """利用广播和向量化快速计算归一化距离矩阵"""
        # 使用 ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b 展开，避免重复计算
        sq_norm = np.einsum('ij,ij->i', X, X)  # 比 (X**2).sum(axis=1) 更快
        D = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (X @ X.T)
        np.clip(D, 0, None, out=D)             # 消除浮点误差导致的负值
        total = D.sum()
        if total > 0:
            D /= total                          # in-place 归一化，避免拷贝
        return D

    def getM(alpha1: float, alpha2: float) -> np.ndarray:
        if data_type == "spatial":
            newcoor = np.ascontiguousarray(adata.obsm[spatial_key], dtype=np.float64)
            m1 = _compute_dist_matrix(newcoor)
            m2 = _compute_dist_matrix(embedding)
            M = alpha2 * m1
            M += alpha1 * m2                   # in-place，减少内存分配
            max_val = M.max()
            if max_val > 0:
                M /= max_val
        elif data_type == "single-cell":
            M = _compute_dist_matrix(embedding)
            max_val = M.max()
            if max_val > 0:
                M /= max_val
        else:
            raise ValueError(
                "Please give the right data type, choose from 'spatial' or 'single-cell'."
            )
        return M

    M = getM(alpha1, alpha2)

    np.fill_diagonal(M, M.max() * 1000)     

    n = adata.n_obs
    a = np.full(n, 1.0 / n)
    b = a.copy()

    if use_gpu:
        try:
            import torch
            a_t = torch.from_numpy(a)
            b_t = torch.from_numpy(b)
            M_t = torch.from_numpy(M)
            Gs = ot.sinkhorn(a_t, b_t, M_t, lambd,
                             numItermax=numItermax).numpy()
        except Exception:
            print("GPU 加速失败，回退到 CPU")
            Gs = np.array(ot.sinkhorn(a, b, M, lambd, numItermax=numItermax))
    else:
        # log-domain sinkhorn 数值更稳定且通常更快
        Gs = np.array(
            ot.sinkhorn_log(a, b, M, lambd, numItermax=numItermax)
        )

    return Gs


def get_ptime(adata: AnnData, start_cells: list):
    select_trans = adata.obsp["trans"][start_cells]
    
    if hasattr(select_trans, "toarray"):
        cell_tran = np.asarray(select_trans.sum(axis=0)).ravel()
    else:
        cell_tran = np.sum(select_trans, axis=0)
    
    adata.obs["tran"] = cell_tran

    n = adata.n_obs
    sort_idx = np.argsort(cell_tran)[::-1]     # 降序排列的原始索引
    ptime = np.empty(n, dtype=np.float32)
    ptime[sort_idx] = np.arange(n, dtype=np.float32) / (n - 1)  # 一次性赋值

    return ptime

def get_neigh_trans(
    adata: AnnData, spatial_key: str, n_neigh_pos: int = 10, n_neigh_gene: int = 0
):
    if n_neigh_pos == 0 and n_neigh_gene == 0:
        raise ValueError(
            "the number of position neighbors and gene neighbors cannot be zero at the same time."
        )

    N = adata.n_obs

    # ── 1. 空间邻居：向量化构建二阶邻居 ──────────────────────────────────────
    if n_neigh_pos:
        nn = NearestNeighbors(n_neighbors=n_neigh_pos + 1, n_jobs=-1)
        nn.fit(adata.obsm[spatial_key])
        _, neigh_pos = nn.kneighbors(adata.obsm[spatial_key])
        neigh_pos = neigh_pos[:, 1:]                    # 去除自身，shape (N, K)

        K = neigh_pos.shape[1]
        rows_1 = np.repeat(np.arange(N), K)
        cols_1 = neigh_pos.ravel()

        adj1 = csr_matrix(
            (np.ones(len(rows_1), dtype=np.bool_), (rows_1, cols_1)),
            shape=(N, N),
        )
        adj2 = adj1 @ adj1
        adj2 = adj2 + adj1
        adj2.setdiag(0)
        adj2.eliminate_zeros()
        adj2 = adj2.tocsr()
        neigh_pos_indices = adj2.indices
        neigh_pos_indptr  = adj2.indptr

    # ── 2. 基因邻居 ───────────────────────────────────────────────────────────
    if n_neigh_gene:
        if "X_pca" not in adata.obsm:
            print("X_pca is not in adata.obsm, automatically do PCA first.")
            sc.tl.pca(adata)
        sc.pp.neighbors(
            adata, use_rep="X_pca", key_added="X_pca", n_neighbors=n_neigh_gene
        )
        neigh_gene = adata.obsp["X_pca_distances"].tocsr()
        neigh_gene.eliminate_zeros()

    # ── 3. 将 trans 统一转为 numpy 二维数组（兼容 ndarray 和稀疏矩阵）─────────
    raw_trans = adata.obsp["trans"]
    if hasattr(raw_trans, "toarray"):
        trans = raw_trans.toarray()                     # 稀疏矩阵 → 密集 ndarray
    else:
        trans = np.asarray(raw_trans)                   # 确保是 ndarray

    # ── 4. 合并邻居索引，向量化构建 CSR ───────────────────────────────────────
    indptr  = np.zeros(N + 1, dtype=np.int64)
    indices_list = []
    csr_data_list = []

    for i in range(N):
        if n_neigh_pos == 0:
            n_all = neigh_gene[i].indices
        elif n_neigh_gene == 0:
            n_all = neigh_pos_indices[neigh_pos_indptr[i]:neigh_pos_indptr[i+1]]
        else:
            g_idx = neigh_gene[i].indices
            p_idx = neigh_pos_indices[neigh_pos_indptr[i]:neigh_pos_indptr[i+1]]
            n_all = np.union1d(p_idx, g_idx)

        vals = trans[i, n_all]                          # 直接索引，返回 1D ndarray

        row_sum = vals.sum()                            # 标量，if 判断不再报错
        if row_sum > 0:
            vals = vals / row_sum

        indptr[i + 1] = indptr[i] + len(n_all)
        indices_list.append(n_all)
        csr_data_list.append(vals)

    all_indices = np.concatenate(indices_list)
    all_data    = np.concatenate(csr_data_list)

    trans_neigh_csr = csr_matrix(
        (all_data, all_indices, indptr), shape=(N, N)
    )
    return trans_neigh_csr

def get_velocity_grid(
    adata,
    P: np.ndarray,
    V: np.ndarray,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
) -> tuple:

    grids = []
    for dim in range(P.shape[1]):
        m, M = np.min(P[:, dim]), np.max(P[:, dim])
        span = np.abs(M - m)
        m -= 0.01 * span                        # 修复原始 bug：M 被修改前先算 span
        M += 0.01 * span
        gr = np.linspace(m, M, int(grid_num * density))
        grids.append(gr)

    meshes = np.meshgrid(*grids)
    P_grid = np.vstack([i.flat for i in meshes]).T

    n_neighbors = max(1, int(P.shape[0] / grid_num))
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(P)
    dists, neighs = nn.kneighbors(P_grid)

    scale = np.mean([grid[1] - grid[0] for grid in grids]) * smooth

    # 用手动高斯公式替换 norm.pdf，避免 scipy 函数调用开销
    weight = np.exp(-0.5 * (dists / scale) ** 2)   # 省略常数系数，不影响归一化结果
    p_mass = weight.sum(1)

    V_grid = (V[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]

    P_grid = np.stack(grids)
    ns = P_grid.shape[1]
    V_grid = V_grid.T.reshape(2, ns, ns)

    mass = np.sqrt((V_grid * V_grid).sum(0))        # 等价于 np.linalg.norm，但更快
    min_mass = np.clip(1e-5, None, np.percentile(mass, 99) * 0.01)
    V_grid[0][mass < min_mass] = np.nan

    adata.uns["P_grid"] = P_grid
    adata.uns["V_grid"] = V_grid

    return P_grid, V_grid
    
def get_velocity(
    adata: AnnData,
    spatial_key: str,
    n_neigh_pos: int = 10,
    n_neigh_gene: int = 0,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
) -> tuple:

    adata.obsp["trans_neigh_csr"] = get_neigh_trans(
        adata, spatial_key, n_neigh_pos, n_neigh_gene
    )

    trans_csr = adata.obsp["trans_neigh_csr"]       # 本地引用，避免重复访问 adata
    position = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    ptime = adata.obs["ptime"].values               # 转为 NumPy array，避免 .iloc 开销

    # ── 核心向量化：用 COO 格式一次性处理所有 (cell, neigh) 对 ──────────────
    cx = trans_csr.tocoo()
    rows = cx.row                                   # 每条边的 cell 索引
    cols = cx.col                                   # 每条边的 neigh 索引
    data = cx.data.copy()                           # 转移概率

    # ptime(neigh) < ptime(cell) 时反向，即乘 -1
    mask = ptime[cols] < ptime[rows]
    data[mask] *= -1.0

    # 位移向量
    diff = position[cols] - position[rows]          # shape (E, 2)
    dist = np.linalg.norm(diff, axis=1, keepdims=True)  # shape (E, 1)
    dist = np.maximum(dist, 1e-12)                  # 防除零
    unit_diff = diff / dist                         # 单位方向向量

    weighted = data[:, None] * unit_diff            # shape (E, 2)

    # 每个 cell 的邻居数（用于归一化）
    n_neigh_per_cell = np.diff(trans_csr.indptr)    # shape (N,)
    n_neigh_per_cell = np.maximum(n_neigh_per_cell, 1)

    # 按 cell 累加：用 np.add.at 或 bincount 替代 Python 循环
    V = np.zeros(position.shape, dtype=np.float64)
    np.add.at(V, rows, weighted)
    V /= n_neigh_per_cell[:, None]                  # 广播归一化

    adata.obsm["velocity_" + spatial_key] = V
    print(f"The velocity of cells store in 'velocity_{spatial_key}'.")

    P_grid, V_grid = get_velocity_grid(
        adata,
        P=position,
        V=V,
        grid_num=grid_num,
        smooth=smooth,
        density=density,
    )
    return P_grid, V_grid



def run_track(
    adata,
    select_way: Literal["coordinates", "cell_type"] = "coordinates",
    # ── set_start_cells 相关 ──────────────────────────────────────────────
    start_point: Optional[Tuple[int, int]] = None,  # select_way='coordinates' 时必填
    cell_type: Optional[str] = None,                # select_way='cell_type' 时必填
    spatial_key: str = 'X_spatial',
    n_neigh: int = 5,
    split: bool = False,                            # cell_type 模式下是否 kmeans 分割
    n_clusters: int = 2,                            # split=True 时的聚类数
    # ── get_ot_matrix 相关 ───────────────────────────────────────────────
    data_type: str = 'spatial',
    alpha1: float = 0.6,
    alpha2: float = 0.4,
    use_rep: str = 'cell_embedding',
    lambd: float = 1e-1,
    numItermax: int = 1000,
    use_gpu: bool = True,
    # ── get_velocity 相关 ────────────────────────────────────────────────
    n_neigh_pos: int = 50,
    n_neigh_gene: int = 0,
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
    # ── 分批相关 ─────────────────────────────────────────────────────────
    batch_size: int = 10000,
    random_state: int = 42,
):
    """
    整合 stereotrack 主流程：start_cells -> OT矩阵 -> ptime -> velocity
    当细胞数 > batch_size 时，将数据随机分批处理后合并结果。
    最终只在 adata 中保留:
        - adata.obs['ptime']
        - adata.uns['E_grid']
        - adata.uns['V_grid']

    Parameters
    ----------
    adata        : AnnData
    select_way   : 'coordinates' 或 'cell_type'
    start_point  : 起始点坐标，select_way='coordinates' 时必填，如 [-1050, 3850]
    cell_type    : 细胞类型名，select_way='cell_type' 时必填；
                   select_way='coordinates' 时可选（用于进一步过滤邻居）
    spatial_key  : obsm 中空间坐标的 key
    n_neigh      : set_start_cells 的邻居数
    split        : cell_type 模式下是否用 KMeans 分割起始细胞
    n_clusters   : split=True 时的聚类数
    data_type    : OT 矩阵类型，'spatial' 或 'single-cell'
    alpha1       : OT 矩阵基因距离权重
    alpha2       : OT 矩阵空间距离权重
    use_rep      : obsm 中 embedding 的 key
    lambd        : OT sinkhorn 正则化系数
    numItermax   : OT sinkhorn 最大迭代次数
    use_gpu      : 是否使用 GPU 加速 OT 计算
    n_neigh_pos  : get_velocity 空间邻居数
    n_neigh_gene : get_velocity 基因邻居数，0 表示不使用
    grid_num     : velocity grid 网格数
    smooth       : velocity grid 平滑系数
    density      : velocity grid 密度
    batch_size   : 大数据分批阈值
    random_state : 随机种子
    """
    # ── 参数合法性前置检查 ────────────────────────────────────────────────
    if select_way == "coordinates" and start_point is None:
        raise ValueError("select_way='coordinates' 时，start_point 不能为 None。")
    if select_way == "cell_type" and cell_type is None:
        raise ValueError("select_way='cell_type' 时，cell_type 不能为 None。")

    n_cells = adata.n_obs

    # ------------------------------------------------------------------ #
    # 内部：在子集 adata_sub 上跑完整流程
    # ------------------------------------------------------------------ #
    def _run_on_subset(adata_sub):
        start_cells = set_start_cells(
            adata_sub,
            select_way=select_way,
            cell_type=cell_type,
            start_point=start_point,
            spatial_key=spatial_key,
            split=split,
            n_clusters=n_clusters,
            n_neigh=n_neigh,
        )
        adata_sub.obsp["trans"] = get_ot_matrix(
            adata_sub,
            data_type=data_type,
            alpha1=alpha1,
            alpha2=alpha2,
            use_rep=use_rep,
            spatial_key=spatial_key,
            lambd=lambd,
            numItermax=numItermax,
            use_gpu=use_gpu,
        )
        ptime = get_ptime(adata_sub, start_cells)

        # get_velocity 内部依赖 adata.obs["ptime"]，必须提前写入
        adata_sub.obs["ptime"] = np.array(ptime)

        E_grid, V_grid = get_velocity(
            adata_sub,
            spatial_key=spatial_key,
            n_neigh_pos=n_neigh_pos,
            n_neigh_gene=n_neigh_gene,
            grid_num=grid_num,
            smooth=smooth,
            density=density,
        )
        return ptime, E_grid, V_grid

    # ------------------------------------------------------------------ #
    # 内部：按细胞数加权平均合并多个 batch 的 grid
    # ------------------------------------------------------------------ #
    def _merge_grids(grid_list, weight_list):
        total_weight = sum(weight_list)
        weights = np.array(weight_list, dtype=np.float64) / total_weight
        E_merged = sum(w * E for w, (E, V) in zip(weights, grid_list))
        V_merged = sum(w * V for w, (E, V) in zip(weights, grid_list))
        return E_merged, V_merged

    # ------------------------------------------------------------------ #
    # 小数据：直接跑
    # ------------------------------------------------------------------ #
    if n_cells <= batch_size:
        print(f"细胞数 {n_cells} <= {batch_size}，直接处理...")
        adata_tmp = adata.copy()
        ptime, E_grid, V_grid = _run_on_subset(adata_tmp)

    # ------------------------------------------------------------------ #
    # 大数据：随机分批，分批计算 ptime 和 grid，最后合并
    # ------------------------------------------------------------------ #
    else:
        print(f"细胞数 {n_cells} > {batch_size}，随机分批（每批 {batch_size}）处理...")
        np.random.seed(random_state)
        all_indices = np.random.permutation(n_cells)
        batches = [
            all_indices[i: i + batch_size]
            for i in range(0, n_cells, batch_size)
        ]
        print(f"共 {len(batches)} 个 batch。")

        ptime_full = np.zeros(n_cells, dtype=np.float64)
        grid_list   = []
        weight_list = []

        for i, idx in enumerate(batches):
            print(f"  处理 batch {i+1}/{len(batches)}，细胞数 {len(idx)}...")
            adata_sub = adata[idx].copy()
            ptime_sub, E_grid_sub, V_grid_sub = _run_on_subset(adata_sub)
            ptime_full[idx] = np.array(ptime_sub)
            grid_list.append((E_grid_sub, V_grid_sub))
            weight_list.append(len(idx))

        print("合并各 batch 的 velocity grid...")
        E_grid, V_grid = _merge_grids(grid_list, weight_list)
        ptime = ptime_full

    # ------------------------------------------------------------------ #
    # 写入目标字段，清理中间数据
    # ------------------------------------------------------------------ #
    adata.obs['ptime']  = np.array(ptime)
    adata.uns['E_grid'] = E_grid
    adata.uns['V_grid'] = V_grid

    for key in ['trans', 'trans_neigh_csr']:
        if key in adata.obsp:
            del adata.obsp[key]
    vel_key = "velocity_" + spatial_key
    if vel_key in adata.obsm:
        del adata.obsm[vel_key]

    print("完成！adata.obs['ptime']、adata.uns['E_grid']、adata.uns['V_grid'] 已就绪。")
    return adata



def get_velocity_3d(
    adata,
    spatial_key: str = 'X_spatial_3d',
    n_neigh_pos: int = 50,
    n_neigh_gene: int = 0,
    grid_num: int = 20,       # 3D 网格点数，注意 grid_num^3 会很大，建议<=20
    smooth: float = 0.5,
    density: float = 1.0,
):
    """
    三维空间转录组的 velocity 计算，输出3D网格上的速度场。
    
    Returns
    -------
    E_grid : np.ndarray, shape (3, M)  — 网格点坐标 (x, y, z)
    V_grid : np.ndarray, shape (3, M)  — 网格点速度 (vx, vy, vz)
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    position = np.asarray(adata.obsm[spatial_key], dtype=np.float64)  # (N, 3)
    ptime    = adata.obs["ptime"].values
    trans    = adata.obsp["trans"]

    assert position.shape[1] == 3, f"期望3D坐标，但 {spatial_key} 的维度为 {position.shape[1]}"

    # ── 1. 空间邻居 ──────────────────────────────────────────────────────
    nn_pos = NearestNeighbors(n_neighbors=n_neigh_pos + 1).fit(position)
    dist_pos, idx_pos = nn_pos.kneighbors(position)
    dist_pos, idx_pos = dist_pos[:, 1:], idx_pos[:, 1:]   # 去掉自身

    # ── 2. 构建 trans_neigh（只保留空间邻居内的转移概率）────────────────
    n = adata.n_obs
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_rank, j in enumerate(idx_pos[i]):
            v = trans[i, j]
            if v > 0:
                rows.append(i)
                cols.append(j)
                vals.append(v)
    trans_neigh = csr_matrix((vals, (rows, cols)), shape=(n, n))
    adata.obsp["trans_neigh_csr"] = trans_neigh

    # ── 3. 计算每个细胞的3D速度向量 ──────────────────────────────────────
    cx = trans_neigh.tocoo()
    i_idx, j_idx, t_vals = cx.row, cx.col, cx.data

    delta_pos   = position[j_idx] - position[i_idx]   # (E, 3)
    delta_ptime = ptime[j_idx] - ptime[i_idx]          # (E,)

    # 只保留 ptime 增大的方向（前向转移）
    forward_mask = delta_ptime > 0
    weights = t_vals * forward_mask

    # 加权求和速度
    V_cell = np.zeros((n, 3), dtype=np.float64)
    np.add.at(V_cell[:, 0], i_idx, weights * delta_pos[:, 0])
    np.add.at(V_cell[:, 1], i_idx, weights * delta_pos[:, 1])
    np.add.at(V_cell[:, 2], i_idx, weights * delta_pos[:, 2])

    # ── 4. 构建3D网格 ─────────────────────────────────────────────────────
    x_min, x_max = position[:, 0].min(), position[:, 0].max()
    y_min, y_max = position[:, 1].min(), position[:, 1].max()
    z_min, z_max = position[:, 2].min(), position[:, 2].max()

    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    margin_z = (z_max - z_min) * 0.05

    gx = np.linspace(x_min - margin_x, x_max + margin_x, grid_num)
    gy = np.linspace(y_min - margin_y, y_max + margin_y, grid_num)
    gz = np.linspace(z_min - margin_z, z_max + margin_z, grid_num)

    # meshgrid -> (grid_num^3, 3) 的网格点
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    grid_points = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)  # (M, 3)

    # ── 5. 将细胞速度插值到网格点（KNN 加权平均）────────────────────────
    nn_grid = NearestNeighbors(n_neighbors=min(n_neigh_pos, n)).fit(position)
    grid_dist, grid_idx = nn_grid.kneighbors(grid_points)  # (M, k)

    # 高斯核权重
    sigma = smooth * (x_max - x_min) / grid_num
    gauss_w = np.exp(- grid_dist ** 2 / (2 * sigma ** 2))  # (M, k)
    gauss_w /= gauss_w.sum(axis=1, keepdims=True) + 1e-10

    V_grid_xyz = np.einsum('mk,kd->md', gauss_w, V_cell[grid_idx.ravel()].reshape(
        grid_points.shape[0], -1, 3
    ).mean(axis=1))   # 简化写法，下面用显式循环更清晰

    # 显式加权插值
    V_grid_xyz = np.zeros((len(grid_points), 3), dtype=np.float64)
    for d in range(3):
        V_neigh = V_cell[grid_idx, d]          # (M, k)
        V_grid_xyz[:, d] = (gauss_w * V_neigh).sum(axis=1)

    # ── 6. 密度过滤：去掉远离细胞的空网格点 ─────────────────────────────
    min_dist_to_cell = grid_dist[:, 0]
    threshold = np.percentile(min_dist_to_cell, density * 100)
    keep_mask = min_dist_to_cell <= threshold

    E_grid = grid_points[keep_mask].T    # (3, M_kept)
    V_grid = V_grid_xyz[keep_mask].T     # (3, M_kept)

    return E_grid, V_grid  # (E,)




def run_stereotrack_3d(
    adata,
    select_way: Literal["coordinates", "cell_type"] = "coordinates",
    start_point: Optional[Tuple[int, int, int]] = None,  # 3D: [x, y, z]
    cell_type: Optional[str] = None,
    spatial_key: str = 'X_spatial_3d',
    n_neigh: int = 5,
    split: bool = False,
    n_clusters: int = 2,
    data_type: str = 'spatial',
    alpha1: float = 0.6,
    alpha2: float = 0.4,
    use_rep: str = 'cell_embedding',
    lambd: float = 1e-1,
    numItermax: int = 1000,
    use_gpu: bool = True,
    n_neigh_pos: int = 50,
    n_neigh_gene: int = 0,
    grid_num: int = 20,       # 3D 建议 <=20，否则网格点数 grid_num^3 过大
    smooth: float = 0.5,
    density: float = 1.0,
    batch_size: int = 10000,
    random_state: int = 42,
):
    """
    三维空间转录组版本的 stereotrack 主流程。
    最终只在 adata 中保留:
        - adata.obs['ptime']
        - adata.uns['E_grid']   shape (3, M)
        - adata.uns['V_grid']   shape (3, M)
    """
    if select_way == "coordinates" and start_point is None:
        raise ValueError("select_way='coordinates' 时，start_point 不能为 None。")
    if select_way == "cell_type" and cell_type is None:
        raise ValueError("select_way='cell_type' 时，cell_type 不能为 None。")
    if start_point is not None and len(start_point) != 3:
        raise ValueError(f"3D 模式下 start_point 应为长度3的坐标，当前为 {start_point}。")

    n_cells = adata.n_obs

    def _run_on_subset(adata_sub):
        start_cells = set_start_cells(
            adata_sub,
            select_way=select_way,
            cell_type=cell_type,
            start_point=start_point,
            spatial_key=spatial_key,
            split=split,
            n_clusters=n_clusters,
            n_neigh=n_neigh,
        )
        adata_sub.obsp["trans"] = get_ot_matrix(
            adata_sub,
            data_type=data_type,
            alpha1=alpha1,
            alpha2=alpha2,
            use_rep=use_rep,
            spatial_key=spatial_key,
            lambd=lambd,
            numItermax=numItermax,
            use_gpu=use_gpu,
        )
        ptime = get_ptime(adata_sub, start_cells)
        adata_sub.obs["ptime"] = np.array(ptime)

        # 使用3D版本的 get_velocity
        E_grid, V_grid = get_velocity_3d(
            adata_sub,
            spatial_key=spatial_key,
            n_neigh_pos=n_neigh_pos,
            n_neigh_gene=n_neigh_gene,
            grid_num=grid_num,
            smooth=smooth,
            density=density,
        )
        return ptime, E_grid, V_grid

    def _merge_grids(grid_list, weight_list):
        total_weight = sum(weight_list)
        weights = np.array(weight_list, dtype=np.float64) / total_weight
        E_merged = sum(w * E for w, (E, V) in zip(weights, grid_list))
        V_merged = sum(w * V for w, (E, V) in zip(weights, grid_list))
        return E_merged, V_merged

    if n_cells <= batch_size:
        print(f"细胞数 {n_cells} <= {batch_size}，直接处理...")
        ptime, E_grid, V_grid = _run_on_subset(adata.copy())
    else:
        print(f"细胞数 {n_cells} > {batch_size}，随机分批（每批 {batch_size}）处理...")
        np.random.seed(random_state)
        all_indices = np.random.permutation(n_cells)
        batches = [all_indices[i: i + batch_size] for i in range(0, n_cells, batch_size)]
        print(f"共 {len(batches)} 个 batch。")

        ptime_full  = np.zeros(n_cells, dtype=np.float64)
        grid_list, weight_list = [], []

        for i, idx in enumerate(batches):
            print(f"  处理 batch {i+1}/{len(batches)}，细胞数 {len(idx)}...")
            adata_sub = adata[idx].copy()
            ptime_sub, E_grid_sub, V_grid_sub = _run_on_subset(adata_sub)
            ptime_full[idx] = np.array(ptime_sub)
            grid_list.append((E_grid_sub, V_grid_sub))
            weight_list.append(len(idx))

        print("合并各 batch 的 velocity grid...")
        E_grid, V_grid = _merge_grids(grid_list, weight_list)
        ptime = ptime_full

    adata.obs['ptime']  = np.array(ptime)
    adata.uns['E_grid'] = E_grid   # shape (3, M)
    adata.uns['V_grid'] = V_grid   # shape (3, M)

    for key in ['trans', 'trans_neigh_csr']:
        if key in adata.obsp:
            del adata.obsp[key]

    print("完成！adata.obs['ptime']、adata.uns['E_grid']、adata.uns['V_grid'] 已就绪。")
    return adata