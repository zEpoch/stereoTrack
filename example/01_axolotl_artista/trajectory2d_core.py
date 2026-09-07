from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import Delaunay, QhullError
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors


def _obsm(adata: AnnData, key_or_basis: str) -> tuple[str, np.ndarray]:
    if key_or_basis in adata.obsm:
        return key_or_basis, np.asarray(adata.obsm[key_or_basis])
    x_key = "X_" + key_or_basis
    if x_key in adata.obsm:
        return x_key, np.asarray(adata.obsm[x_key])
    raise KeyError(f"Neither obsm['{key_or_basis}'] nor obsm['{x_key}'] was found")


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"{name} should be a 2D array, got shape={values.shape}")
    return values


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    center = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - center) / scale


def _normalize_distance(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    total = float(np.nansum(dist))
    if total > 0:
        dist = dist / total
    max_val = float(np.nanmax(dist))
    if max_val > 0:
        dist = dist / max_val
    return dist


def _pairwise_sqeuclidean(values: np.ndarray, standardize: bool = True) -> np.ndarray:
    values = _as_2d(values, "values")
    x = _standardize(values) if standardize else np.asarray(values, dtype=np.float64)
    sq = np.einsum("ij,ij->i", x, x)
    dist = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(dist, 0.0, out=dist)
    return _normalize_distance(dist)


def _pairwise_cosine(values: np.ndarray, standardize: bool = True) -> np.ndarray:
    values = _as_2d(values, "values")
    x = _standardize(values) if standardize else np.asarray(values, dtype=np.float64)
    denom = np.linalg.norm(x, axis=1)
    denom = np.maximum(denom[:, None] * denom[None, :], 1e-12)
    sim = (x @ x.T) / denom
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    return _normalize_distance(dist)


def _select_dims(values: np.ndarray, dims: Iterable[int] | None) -> np.ndarray:
    values = _as_2d(values, "values")
    if dims is None:
        return values
    return values[:, list(dims)]


def get_ot_matrix(
    adata: AnnData,
    embedding_key: str = "niche_embedding",
    spatial_key: str = "X_spatial",
    alpha_embedding: float = 0.5,
    alpha_spatial: float = 0.5,
    embedding_dims: Iterable[int] | None = None,
    spatial_dims: Iterable[int] | None = (0, 1),
    embedding_metric: Literal["sqeuclidean", "cosine"] = "sqeuclidean",
    spatial_metric: Literal["sqeuclidean", "cosine"] = "sqeuclidean",
    standardize: bool = True,
    lambd: float = 1e-1,
    numItermax: int = 1000,
    diagonal_penalty: float = 1000.0,
    store_key: str = "trans",
    dtype: str | np.dtype = "float32",
) -> np.ndarray:
    """Compute OT transition probabilities and store them in ``adata.obsp[store_key]``."""
    try:
        import ot
    except ImportError as exc:
        raise ImportError("POT is required: pip install POT") from exc

    if alpha_embedding < 0 or alpha_spatial < 0:
        raise ValueError("alpha_embedding and alpha_spatial should be non-negative")
    alpha_sum = float(alpha_embedding + alpha_spatial)
    if alpha_sum <= 0:
        raise ValueError("At least one alpha should be positive")
    alpha_embedding = float(alpha_embedding) / alpha_sum
    alpha_spatial = float(alpha_spatial) / alpha_sum

    _, embedding = _obsm(adata, embedding_key)
    _, spatial = _obsm(adata, spatial_key)
    embedding = _select_dims(embedding, embedding_dims)
    spatial = _select_dims(spatial, spatial_dims)

    if embedding.shape[0] != adata.n_obs or spatial.shape[0] != adata.n_obs:
        raise ValueError("embedding and spatial arrays should have one row per cell")
    if not np.isfinite(embedding).all():
        raise ValueError(f"obsm['{embedding_key}'] contains NaN or inf")
    if not np.isfinite(spatial).all():
        raise ValueError(f"obsm['{spatial_key}'] contains NaN or inf")

    dist_fn = {
        "sqeuclidean": _pairwise_sqeuclidean,
        "cosine": _pairwise_cosine,
    }
    if embedding_metric not in dist_fn or spatial_metric not in dist_fn:
        raise ValueError("metrics should be 'sqeuclidean' or 'cosine'")

    M = np.zeros((adata.n_obs, adata.n_obs), dtype=np.float64)
    if alpha_spatial > 0:
        M += alpha_spatial * dist_fn[spatial_metric](spatial, standardize=standardize)
    if alpha_embedding > 0:
        M += alpha_embedding * dist_fn[embedding_metric](embedding, standardize=standardize)

    max_val = float(np.nanmax(M))
    if max_val > 0:
        M /= max_val
    np.fill_diagonal(M, max(float(np.nanmax(M)), 1.0) * float(diagonal_penalty))

    n = adata.n_obs
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = a.copy()
    try:
        trans = np.asarray(ot.sinkhorn_log(a, b, M, reg=lambd, numItermax=numItermax))
    except Exception:
        trans = np.asarray(ot.sinkhorn(a, b, M, reg=lambd, numItermax=numItermax))

    trans = trans.astype(dtype, copy=False)
    adata.obsp[store_key] = trans
    return trans


def get_ptime(
    adata: AnnData,
    obs_key: str,
    cell_type: str | Iterable[str],
    trans_key: str = "trans",
    store_key: str = "ptime",
    tran_store_key: str = "tran",
) -> np.ndarray:
    """Infer pseudotime from start cell types and ``adata.obsp[trans_key]``."""
    if trans_key not in adata.obsp:
        raise KeyError(f"adata.obsp['{trans_key}'] was not found. Run get_ot_matrix first.")
    if obs_key not in adata.obs:
        raise KeyError(f"adata.obs['{obs_key}'] was not found")

    if isinstance(cell_type, str):
        target = {cell_type}
    else:
        target = {str(x) for x in cell_type}

    labels = adata.obs[obs_key].astype(str)
    start_mask = labels.isin(target).to_numpy()
    start_cells = np.where(start_mask)[0]
    if start_cells.size == 0:
        raise ValueError(f"No start cells found in obs['{obs_key}'] for {sorted(target)}")

    select_trans = adata.obsp[trans_key][start_cells]
    if sp.issparse(select_trans):
        cell_tran = np.asarray(select_trans.sum(axis=0)).ravel()
    else:
        cell_tran = np.asarray(select_trans).sum(axis=0).ravel()

    n = adata.n_obs
    sort_idx = np.argsort(cell_tran)[::-1]
    ptime = np.empty(n, dtype=np.float32)
    if n <= 1:
        ptime[:] = 0.0
    else:
        ptime[sort_idx] = np.arange(n, dtype=np.float32) / float(n - 1)

    adata.obs[tran_store_key] = cell_tran.astype(np.float32, copy=False)
    adata.obs[store_key] = ptime
    return ptime


def _get_neigh_trans(
    adata: AnnData,
    basis_key: str,
    n_neigh_pos: int,
    trans_key: str,
) -> sp.csr_matrix:
    if n_neigh_pos <= 0:
        raise ValueError("n_neigh_pos should be positive")

    _, position = _obsm(adata, basis_key)
    position = _as_2d(position, basis_key)[:, :2]
    n = adata.n_obs
    k = min(int(n_neigh_pos) + 1, n)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(position)
    _, neigh_pos = nn.kneighbors(position)
    neigh_pos = neigh_pos[:, 1:]

    rows_1 = np.repeat(np.arange(n, dtype=np.int64), neigh_pos.shape[1])
    cols_1 = neigh_pos.ravel().astype(np.int64, copy=False)
    adj1 = sp.csr_matrix(
        (np.ones(rows_1.size, dtype=bool), (rows_1, cols_1)),
        shape=(n, n),
    )
    adj2 = (adj1 @ adj1) + adj1
    adj2 = adj2.tolil()
    adj2.setdiag(0)
    adj2 = adj2.tocsr()
    adj2.eliminate_zeros()

    trans = adata.obsp[trans_key]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices_parts = []
    data_parts = []
    for i in range(n):
        n_all = adj2.indices[adj2.indptr[i] : adj2.indptr[i + 1]]
        if n_all.size == 0:
            vals = np.empty(0, dtype=np.float32)
        elif sp.issparse(trans):
            vals = np.asarray(trans.getrow(i)[:, n_all].toarray()).ravel()
        else:
            vals = np.asarray(trans[i, n_all]).ravel()
        row_sum = float(vals.sum())
        if row_sum > 0:
            vals = vals / row_sum
        indptr[i + 1] = indptr[i] + n_all.size
        indices_parts.append(n_all)
        data_parts.append(vals.astype(np.float32, copy=False))

    indices = np.concatenate(indices_parts) if indices_parts else np.empty(0, dtype=np.int64)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def _local_support_radius(
    position: np.ndarray,
    grid_step: float,
    distance_quantile: float,
    distance_scale: float,
) -> float:
    support_k = min(2, position.shape[0])
    if support_k > 1:
        support_nn = NearestNeighbors(n_neighbors=support_k, n_jobs=-1)
        support_nn.fit(position)
        cell_nn_dist, _ = support_nn.kneighbors(position)
        local_step = cell_nn_dist[:, 1]
        local_step = local_step[np.isfinite(local_step) & (local_step > 0)]
        if local_step.size:
            radius = float(np.quantile(local_step, distance_quantile)) * float(distance_scale)
        else:
            radius = grid_step * float(distance_scale)
    else:
        radius = grid_step * float(distance_scale)
    return max(float(radius), grid_step * 1.5)


def _delaunay_support_mask(
    position: np.ndarray,
    grid_points: np.ndarray,
    max_edge: float,
) -> tuple[np.ndarray, int]:
    pos_unique = np.unique(position, axis=0)
    if pos_unique.shape[0] < 3:
        return np.ones(grid_points.shape[0], dtype=bool), 0

    try:
        tri = Delaunay(pos_unique)
    except QhullError:
        return np.ones(grid_points.shape[0], dtype=bool), 0

    simplices = tri.simplices
    pts = pos_unique[simplices]
    edge01 = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    edge12 = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    edge20 = np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1)
    simplex_ok = np.maximum.reduce([edge01, edge12, edge20]) <= float(max_edge)

    simplex_id = tri.find_simplex(grid_points)
    keep = np.zeros(grid_points.shape[0], dtype=bool)
    inside = simplex_id >= 0
    keep[inside] = simplex_ok[simplex_id[inside]]
    return keep, int(np.count_nonzero(simplex_ok))


def _velocity_grid(
    adata: AnnData,
    position: np.ndarray,
    velocity: np.ndarray,
    grid_num: int,
    smooth: float,
    density: float,
    mask_empty_grid: bool,
    max_grid_distance: float | None,
    grid_distance_quantile: float,
    grid_distance_scale: float,
    min_mass_quantile: float,
    support_method: Literal["both", "delaunay", "distance", "none"],
    delaunay_edge_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    grids = []
    for dim in range(position.shape[1]):
        lo, hi = float(np.min(position[:, dim])), float(np.max(position[:, dim]))
        span = abs(hi - lo)
        lo -= 0.01 * span
        hi += 0.01 * span
        grids.append(np.linspace(lo, hi, int(grid_num * density)))

    meshes = np.meshgrid(*grids)
    grid_points = np.vstack([x.flat for x in meshes]).T
    n_neighbors = max(1, min(position.shape[0], int(position.shape[0] / grid_num)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(position)
    dists, neighs = nn.kneighbors(grid_points)

    if len(grids[0]) > 1:
        scale = np.mean([grid[1] - grid[0] for grid in grids]) * smooth
        grid_step = float(np.mean([grid[1] - grid[0] for grid in grids]))
    else:
        scale = 1.0
        grid_step = 1.0
    scale = max(float(scale), 1e-12)
    weight = norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(axis=1)

    V_grid = (velocity[neighs] * weight[:, :, None]).sum(axis=1)
    V_grid /= np.maximum(1.0, p_mass)[:, None]

    empty_mask = np.zeros(grid_points.shape[0], dtype=bool)
    support_radius = None
    support_triangles = None
    if mask_empty_grid and support_method != "none":
        if max_grid_distance is None:
            support_radius = _local_support_radius(
                position,
                grid_step=grid_step,
                distance_quantile=grid_distance_quantile,
                distance_scale=grid_distance_scale,
            )
        else:
            support_radius = float(max_grid_distance)

        if support_method in {"distance", "both"}:
            empty_mask |= dists[:, 0] > support_radius
        if support_method in {"delaunay", "both"}:
            keep, support_triangles = _delaunay_support_mask(
                position,
                grid_points,
                max_edge=support_radius * float(delaunay_edge_scale),
            )
            empty_mask |= ~keep

    E_grid = np.stack(grids)
    ns = E_grid.shape[1]
    V_grid_flat = V_grid

    mass = np.linalg.norm(V_grid_flat, axis=1)
    finite_mass = mass[np.isfinite(mass)]
    if finite_mass.size:
        min_mass = max(1e-12, float(np.quantile(finite_mass, min_mass_quantile)))
        low_mass_mask = mass < min_mass
    else:
        low_mass_mask = np.ones_like(mass, dtype=bool)
    cutoff = empty_mask | low_mass_mask
    V_grid_flat[cutoff] = np.nan
    V_grid = V_grid_flat.T.reshape(2, ns, ns)

    adata.uns["E_grid"] = E_grid
    adata.uns["P_grid"] = E_grid
    adata.uns["V_grid"] = V_grid
    adata.uns["velocity_grid_mask"] = cutoff.reshape(ns, ns)
    adata.uns["velocity_grid_support_radius"] = support_radius
    adata.uns["velocity_grid_support_method"] = support_method
    adata.uns["velocity_grid_support_triangles"] = support_triangles
    return E_grid, V_grid


def get_velocity(
    adata: AnnData,
    basis: str = "spatial",
    n_neigh_pos: int = 10,
    trans_key: str = "trans",
    ptime_key: str = "ptime",
    grid_num: int = 50,
    smooth: float = 0.5,
    density: float = 1.0,
    mask_empty_grid: bool = True,
    max_grid_distance: float | None = None,
    grid_distance_quantile: float = 0.99,
    grid_distance_scale: float = 2.0,
    min_mass_quantile: float = 0.01,
    support_method: Literal["both", "delaunay", "distance", "none"] = "both",
    delaunay_edge_scale: float = 1.5,
    velocity_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D cell velocity and return ``E_grid, V_grid`` for streamplot.

    ``support_method='both'`` keeps the spaTrack-style velocity calculation but
    masks grid points that are outside local Delaunay support or too far from
    observed cells. This avoids streamplot arrows in tissue cavities.
    """
    if trans_key not in adata.obsp:
        raise KeyError(f"adata.obsp['{trans_key}'] was not found")
    if ptime_key not in adata.obs:
        raise KeyError(f"adata.obs['{ptime_key}'] was not found. Run get_ptime first.")

    basis_key, position = _obsm(adata, basis)
    position = _as_2d(position, basis_key)[:, :2].astype(np.float64, copy=False)
    ptime = np.asarray(adata.obs[ptime_key], dtype=np.float64)

    trans_neigh = _get_neigh_trans(adata, basis_key, n_neigh_pos, trans_key)
    adata.obsp["trans_neigh_csr"] = trans_neigh

    coo = trans_neigh.tocoo()
    rows = coo.row
    cols = coo.col
    prob = coo.data.astype(np.float64, copy=True)
    prob[ptime[cols] < ptime[rows]] *= -1.0

    diff = position[cols] - position[rows]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    unit = diff / np.maximum(dist, 1e-12)
    weighted = prob[:, None] * unit

    velocity = np.zeros(position.shape, dtype=np.float64)
    np.add.at(velocity, rows, weighted)
    n_per_cell = np.maximum(np.diff(trans_neigh.indptr), 1)
    velocity /= n_per_cell[:, None]

    if velocity_key is None:
        velocity_key = "velocity_" + basis_key
    adata.obsm[velocity_key] = velocity.astype(np.float32, copy=False)

    return _velocity_grid(
        adata,
        position=position,
        velocity=velocity,
        grid_num=grid_num,
        smooth=smooth,
        density=density,
        mask_empty_grid=mask_empty_grid,
        max_grid_distance=max_grid_distance,
        grid_distance_quantile=grid_distance_quantile,
        grid_distance_scale=grid_distance_scale,
        min_mass_quantile=min_mass_quantile,
        support_method=support_method,
        delaunay_edge_scale=delaunay_edge_scale,
    )
