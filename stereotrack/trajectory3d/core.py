from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree
try:
    from scipy.spatial import QhullError
except ImportError:  # scipy < 1.8
    from scipy.spatial.qhull import QhullError
from sklearn.neighbors import NearestNeighbors


@dataclass
class Trajectory3DResult:
    """Small container for the sampled 3D trajectory backbone."""

    backbone_indices: np.ndarray
    backbone_ptime: np.ndarray
    backbone_velocity: np.ndarray
    grid_points: Optional[np.ndarray]
    grid_velocity: Optional[np.ndarray]
    topology: sparse.csr_matrix
    transport_shape: Tuple[int, int]


def _as_2d_array(x: np.ndarray, key: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{key} should be a 2D array, got shape {arr.shape}")
    return arr


def _check_3d_coords(coords: np.ndarray, key: str) -> np.ndarray:
    coords = _as_2d_array(coords, key).astype(np.float32, copy=False)
    if coords.shape[1] < 3:
        raise ValueError(f"{key} should contain at least 3 coordinate columns")
    return coords[:, :3]


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr).all(axis=1)
    return mask


def _standardize_train_apply(
    train: np.ndarray,
    query: Optional[np.ndarray] = None,
    enabled: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    train = np.asarray(train, dtype=np.float32)
    if not enabled:
        return train, None if query is None else np.asarray(query, dtype=np.float32)

    center = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale[scale < 1e-12] = 1.0
    train_out = (train - center) / scale
    if query is None:
        return train_out.astype(np.float32, copy=False), None
    query_out = (np.asarray(query, dtype=np.float32) - center) / scale
    return train_out.astype(np.float32, copy=False), query_out.astype(np.float32, copy=False)


def _pairwise_squared_distance(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    sq = np.einsum("ij,ij->i", x, x)
    dist = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(dist, 0.0, out=dist)
    return dist.astype(np.float32, copy=False)


def _normalize_cost(cost: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    positive = cost[cost > 0]
    if positive.size == 0:
        return cost
    if quantile is None or quantile >= 1.0:
        denom = float(positive.max())
    else:
        denom = float(np.quantile(positive, quantile))
    if not np.isfinite(denom) or denom <= 0:
        denom = float(positive.max()) or 1.0
    cost /= denom
    np.clip(cost, 0.0, 1.0, out=cost)
    return cost


def _axis_groups_from_gaps(
    coords: np.ndarray,
    axis: int = 2,
    gap_multiplier: float = 8.0,
    gap_threshold: Optional[float] = None,
) -> np.ndarray:
    values = np.asarray(coords[:, axis], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    diffs = np.diff(sorted_values)
    positive = diffs[diffs > 0]

    groups_sorted = np.zeros(values.shape[0], dtype=np.int32)
    if positive.size > 0:
        if gap_threshold is None:
            robust_step = float(np.median(positive))
            threshold = robust_step * float(gap_multiplier)
        else:
            threshold = float(gap_threshold)
        if threshold > 0:
            breaks = np.where(diffs > threshold)[0] + 1
            if breaks.size:
                groups_sorted = np.zeros(values.shape[0], dtype=np.int32)
                groups_sorted[breaks] = 1
                groups_sorted = np.cumsum(groups_sorted)

    groups = np.empty_like(groups_sorted)
    groups[order] = groups_sorted
    return groups


def _obs_groups(adata, key: str, axis_values: np.ndarray) -> np.ndarray:
    if key not in adata.obs.columns:
        raise KeyError(f"obs['{key}'] was not found")
    raw = np.asarray(adata.obs[key].astype(str))
    labels = np.unique(raw)
    medians = []
    for label in labels:
        medians.append(np.nanmedian(axis_values[raw == label]))
    ordered = [label for _, label in sorted(zip(medians, labels), key=lambda item: item[0])]
    label_to_code = {label: i for i, label in enumerate(ordered)}
    return np.asarray([label_to_code[label] for label in raw], dtype=np.int32)


def _axis_bin_labels(coords: np.ndarray, axis: int, n_bins: int) -> np.ndarray:
    values = np.asarray(coords[:, axis], dtype=np.float64)
    if n_bins <= 1 or np.nanmin(values) == np.nanmax(values):
        return np.zeros(values.shape[0], dtype=np.int32)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int32)
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int32)


def sample_backbone_indices(
    valid_indices: Sequence[int],
    sample_size: Optional[int] = 8192,
    random_state: int = 42,
    include_indices: Optional[Sequence[int]] = None,
    strata: Optional[Sequence[object]] = None,
    max_include: int = 512,
) -> np.ndarray:
    """Sample a reproducible 3D backbone while preserving requested start cells."""

    valid_indices = np.unique(np.asarray(valid_indices, dtype=np.int64))
    if sample_size is None or sample_size <= 0 or sample_size >= valid_indices.size:
        return valid_indices

    rng = np.random.RandomState(random_state)

    if include_indices is None:
        include = np.empty(0, dtype=np.int64)
    else:
        include = np.intersect1d(valid_indices, np.asarray(include_indices, dtype=np.int64))
        if include.size > max_include:
            include = np.sort(rng.choice(include, size=max_include, replace=False))

    if include.size >= sample_size:
        return np.sort(rng.choice(include, size=sample_size, replace=False))

    remaining_mask = ~np.isin(valid_indices, include)
    remaining = valid_indices[remaining_mask]
    n_needed = sample_size - include.size
    if remaining.size <= n_needed:
        return np.sort(np.concatenate([include, remaining]))

    if strata is None:
        sampled = rng.choice(remaining, size=n_needed, replace=False)
        return np.sort(np.concatenate([include, sampled]))

    strata = np.asarray(strata)
    if strata.shape[0] != valid_indices.shape[0]:
        raise ValueError("strata should have the same length as valid_indices")
    strata_remaining = strata[remaining_mask]
    labels, counts = np.unique(strata_remaining, return_counts=True)
    raw = counts.astype(float) / counts.sum() * n_needed
    take = np.floor(raw).astype(int)
    if n_needed >= labels.size:
        take = np.maximum(take, 1)
    while take.sum() > n_needed:
        idx = np.argmax(take)
        take[idx] -= 1
    while take.sum() < n_needed:
        fractional = raw - np.floor(raw)
        idx = int(np.argmax(fractional - (take >= counts) * 2.0))
        if take[idx] < counts[idx]:
            take[idx] += 1
        else:
            break

    sampled_parts = []
    for label, n_take in zip(labels, take):
        if n_take <= 0:
            continue
        pool = remaining[strata_remaining == label]
        sampled_parts.append(rng.choice(pool, size=min(n_take, pool.size), replace=False))
    sampled = np.concatenate(sampled_parts) if sampled_parts else np.empty(0, dtype=np.int64)

    if sampled.size < n_needed:
        unused = np.setdiff1d(remaining, sampled, assume_unique=False)
        extra = rng.choice(unused, size=n_needed - sampled.size, replace=False)
        sampled = np.concatenate([sampled, extra])
    return np.sort(np.concatenate([include, sampled]))


def build_gap_aware_topology(
    coords: np.ndarray,
    groups: Optional[np.ndarray] = None,
    axis: int = 2,
    within_neighbors: int = 12,
    cross_neighbors: int = 4,
    coord_scale: Optional[Sequence[float]] = None,
    max_within_distance: Optional[float] = None,
    max_cross_lateral_distance: Optional[float] = None,
    max_cross_axis_gap: Optional[float] = None,
    symmetric: bool = True,
) -> sparse.csr_matrix:
    """Build a topology graph that avoids unconstrained jumps across large 3D gaps."""

    coords = _check_3d_coords(coords, "coords")
    n_cells = coords.shape[0]
    if groups is None:
        groups = _axis_groups_from_gaps(coords, axis=axis)
    groups = np.asarray(groups)
    if groups.shape[0] != n_cells:
        raise ValueError("groups should have one value per coordinate row")

    if coord_scale is None:
        scaled = coords.astype(np.float32, copy=True)
    else:
        scale = np.asarray(coord_scale, dtype=np.float32)
        if scale.shape[0] != 3:
            raise ValueError("coord_scale should contain three values")
        scaled = coords.astype(np.float32, copy=True) * scale[None, :]

    rows = []
    cols = []

    unique_groups = np.unique(groups)
    for group in unique_groups:
        idx = np.where(groups == group)[0]
        if idx.size <= 1 or within_neighbors <= 0:
            continue
        k = min(within_neighbors + 1, idx.size)
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(scaled[idx])
        dist, neigh = nn.kneighbors(scaled[idx])
        for local_i in range(idx.size):
            src = idx[local_i]
            for d, local_j in zip(dist[local_i, 1:], neigh[local_i, 1:]):
                if max_within_distance is not None and d > max_within_distance:
                    continue
                rows.append(src)
                cols.append(idx[local_j])

    if cross_neighbors > 0 and unique_groups.size > 1:
        group_order = sorted(
            unique_groups.tolist(),
            key=lambda g: float(np.nanmedian(coords[groups == g, axis])),
        )
        lateral_dims = [dim for dim in range(3) if dim != axis]
        for left, right in zip(group_order[:-1], group_order[1:]):
            left_idx = np.where(groups == left)[0]
            right_idx = np.where(groups == right)[0]
            if left_idx.size == 0 or right_idx.size == 0:
                continue
            axis_gap = abs(
                float(np.nanmedian(coords[right_idx, axis]))
                - float(np.nanmedian(coords[left_idx, axis]))
            )
            if max_cross_axis_gap is not None and axis_gap > max_cross_axis_gap:
                continue
            k_lr = min(cross_neighbors, right_idx.size)
            nn_r = NearestNeighbors(n_neighbors=k_lr, algorithm="auto").fit(scaled[right_idx][:, lateral_dims])
            dist_lr, neigh_lr = nn_r.kneighbors(scaled[left_idx][:, lateral_dims])
            for local_i in range(left_idx.size):
                for d, local_j in zip(dist_lr[local_i], neigh_lr[local_i]):
                    if max_cross_lateral_distance is not None and d > max_cross_lateral_distance:
                        continue
                    rows.append(left_idx[local_i])
                    cols.append(right_idx[local_j])

            k_rl = min(cross_neighbors, left_idx.size)
            nn_l = NearestNeighbors(n_neighbors=k_rl, algorithm="auto").fit(scaled[left_idx][:, lateral_dims])
            dist_rl, neigh_rl = nn_l.kneighbors(scaled[right_idx][:, lateral_dims])
            for local_i in range(right_idx.size):
                for d, local_j in zip(dist_rl[local_i], neigh_rl[local_i]):
                    if max_cross_lateral_distance is not None and d > max_cross_lateral_distance:
                        continue
                    rows.append(right_idx[local_i])
                    cols.append(left_idx[local_j])

    if not rows:
        raise RuntimeError("No topology edges were created; relax neighbor or distance parameters")

    data = np.ones(len(rows), dtype=np.float32)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    graph = graph.tocoo()
    keep = graph.row != graph.col
    graph = sparse.csr_matrix((graph.data[keep], (graph.row[keep], graph.col[keep])), shape=(n_cells, n_cells))
    if symmetric:
        graph = graph.maximum(graph.T).tocsr()
        graph = graph.tocoo()
        keep = graph.row != graph.col
        graph = sparse.csr_matrix((graph.data[keep], (graph.row[keep], graph.col[keep])), shape=(n_cells, n_cells))
    return graph


def build_geometry_topology_3d(
    coords: np.ndarray,
    mode: str = "delaunay_3d",
    n_neighbors: int = 16,
    local_scale_neighbors: int = 8,
    edge_length_factor: Optional[float] = 1.5,
    max_edge_length: Optional[float] = None,
    coord_scale: Optional[Sequence[float]] = None,
) -> sparse.csr_matrix:
    """Build a genuine 3D geometry graph and prune nonlocal spatial bridges."""

    coords = _check_3d_coords(coords, "coords")
    n_cells = coords.shape[0]
    if n_cells < 2:
        raise ValueError("At least two cells are required to build a 3D topology")
    if mode not in {"delaunay_3d", "pure_delaunay_3d", "mutual_knn_3d"}:
        raise ValueError("mode should be 'delaunay_3d', 'pure_delaunay_3d', or 'mutual_knn_3d'")

    if coord_scale is None:
        scaled = coords.astype(np.float64, copy=True)
    else:
        scale = np.asarray(coord_scale, dtype=np.float64)
        if scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError("coord_scale should contain three positive values")
        scaled = coords.astype(np.float64, copy=True) * scale[None, :]

    needs_local_scale = mode != "pure_delaunay_3d" and edge_length_factor is not None and edge_length_factor > 0
    if needs_local_scale:
        local_k = min(max(int(local_scale_neighbors), 1) + 1, n_cells)
        local_nn = NearestNeighbors(n_neighbors=local_k, algorithm="auto").fit(scaled)
        local_dist, _ = local_nn.kneighbors(scaled)
        local_scale = local_dist[:, -1]
        positive_scale = local_scale[local_scale > 0]
        fallback_scale = float(np.median(positive_scale)) if positive_scale.size else 1.0
        local_scale[local_scale <= 0] = fallback_scale
    else:
        local_scale = None

    if mode in {"delaunay_3d", "pure_delaunay_3d"}:
        if n_cells < 5:
            if mode == "pure_delaunay_3d":
                raise RuntimeError("Pure 3D Delaunay requires at least five cells")
            mode = "mutual_knn_3d"
        else:
            try:
                tetrahedra = Delaunay(
                    scaled,
                    qhull_options="Qbb Qc Qz Q12 QJ",
                ).simplices
            except QhullError as error:
                raise RuntimeError(
                    "3D Delaunay construction failed; try mutual_knn_3d or rescale coordinates"
                ) from error
            # Qz may introduce a point-at-infinity index in degenerate inputs.
            tetrahedra = tetrahedra[(tetrahedra < n_cells).all(axis=1)]
            if tetrahedra.size == 0:
                raise RuntimeError("3D Delaunay produced no finite tetrahedra")
            pairs = np.concatenate(
                [
                    tetrahedra[:, [0, 1]],
                    tetrahedra[:, [0, 2]],
                    tetrahedra[:, [0, 3]],
                    tetrahedra[:, [1, 2]],
                    tetrahedra[:, [1, 3]],
                    tetrahedra[:, [2, 3]],
                ],
                axis=0,
            )
            pairs.sort(axis=1)
            pairs = np.unique(pairs, axis=0)

    if mode == "mutual_knn_3d":
        k = min(max(int(n_neighbors), 1) + 1, n_cells)
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(scaled)
        _, neigh = nn.kneighbors(scaled)
        rows = np.repeat(np.arange(n_cells, dtype=np.int64), k - 1)
        cols = neigh[:, 1:].reshape(-1).astype(np.int64, copy=False)
        directed = sparse.csr_matrix(
            (np.ones(rows.size, dtype=bool), (rows, cols)),
            shape=(n_cells, n_cells),
        )
        mutual = directed.minimum(directed.T).tocoo()
        keep_upper = mutual.row < mutual.col
        pairs = np.column_stack(
            [mutual.row[keep_upper], mutual.col[keep_upper]]
        ).astype(np.int64, copy=False)

    if pairs.size == 0:
        raise RuntimeError("No candidate edges were created for the 3D topology")

    edge_length = np.linalg.norm(
        scaled[pairs[:, 0]] - scaled[pairs[:, 1]],
        axis=1,
    )
    keep = np.isfinite(edge_length) & (edge_length > 0)
    if mode != "pure_delaunay_3d" and edge_length_factor is not None and edge_length_factor > 0:
        local_limit = float(edge_length_factor) * np.sqrt(
            local_scale[pairs[:, 0]] * local_scale[pairs[:, 1]]
        )
        keep &= edge_length <= local_limit
    if mode != "pure_delaunay_3d" and max_edge_length is not None:
        keep &= edge_length <= float(max_edge_length)
    pairs = pairs[keep]
    if pairs.size == 0:
        raise RuntimeError(
            "All 3D topology edges were removed; increase edge_length_factor or max_edge_length"
        )

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_cells),
    )
    graph.eliminate_zeros()
    return graph


def build_streaming_mutual_knn_topology_3d(
    coords: np.ndarray,
    n_neighbors: int = 16,
    query_chunk_size: int = 8192,
    max_edge_length: Optional[float] = None,
    coord_scale: Optional[Sequence[float]] = None,
    workers: int = -1,
) -> sparse.csr_matrix:
    """Query every cell once in chunks and assemble one global mutual 3D-KNN graph."""

    coords = _check_3d_coords(coords, "coords")
    n_cells = coords.shape[0]
    if n_cells < 2:
        raise ValueError("At least two cells are required to build a 3D topology")
    if coord_scale is None:
        scaled = coords.astype(np.float32, copy=False)
    else:
        scale = np.asarray(coord_scale, dtype=np.float32)
        if scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError("coord_scale should contain three positive values")
        scaled = coords.astype(np.float32, copy=True) * scale[None, :]

    k = min(max(int(n_neighbors), 1) + 1, n_cells)
    n_edges_per_row = k - 1
    neighbor_indices = np.empty((n_cells, n_edges_per_row), dtype=np.int32)
    edge_valid = np.ones((n_cells, n_edges_per_row), dtype=np.uint8)
    tree = cKDTree(scaled)
    query_chunk_size = max(int(query_chunk_size), 1)
    for start in range(0, n_cells, query_chunk_size):
        stop = min(start + query_chunk_size, n_cells)
        try:
            distance, neighbor = tree.query(scaled[start:stop], k=k, workers=workers)
        except TypeError:  # scipy < 1.6
            distance, neighbor = tree.query(scaled[start:stop], k=k)
        neighbor_indices[start:stop] = neighbor[:, 1:].astype(np.int32, copy=False)
        if max_edge_length is not None:
            edge_valid[start:stop] = (
                distance[:, 1:] <= float(max_edge_length)
            ).astype(np.uint8)

    indptr = np.arange(
        0,
        n_cells * n_edges_per_row + 1,
        n_edges_per_row,
        dtype=np.int64,
    )
    directed = sparse.csr_matrix(
        (edge_valid.ravel(), neighbor_indices.ravel(), indptr),
        shape=(n_cells, n_cells),
    )
    directed.eliminate_zeros()
    directed.sum_duplicates()
    directed.sort_indices()
    mutual = directed.minimum(directed.T).astype(np.float32).tocsr()
    mutual.eliminate_zeros()
    if mutual.nnz == 0:
        raise RuntimeError(
            "The global mutual 3D-KNN graph contains no edges; increase topology_neighbors"
        )
    return mutual


def _edge_costs(
    embedding: np.ndarray,
    coords: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    alpha_embedding: float,
    alpha_spatial: float,
    standardize_embedding: bool,
    standardize_spatial: bool,
    spatial_cost_mode: str = "euclidean",
) -> np.ndarray:
    if spatial_cost_mode not in {"euclidean", "binary"}:
        raise ValueError("spatial_cost_mode should be 'euclidean' or 'binary'")
    emb, _ = _standardize_train_apply(embedding, enabled=standardize_embedding)
    emb_cost = np.sum((emb[rows] - emb[cols]) ** 2, axis=1)
    emb_cost = emb_cost / (np.quantile(emb_cost[emb_cost > 0], 0.99) if np.any(emb_cost > 0) else 1.0)
    if spatial_cost_mode == "binary":
        spa_cost = np.ones(rows.shape[0], dtype=np.float32)
    else:
        xyz, _ = _standardize_train_apply(coords, enabled=standardize_spatial)
        spa_cost = np.sum((xyz[rows] - xyz[cols]) ** 2, axis=1)
        spa_cost = spa_cost / (np.quantile(spa_cost[spa_cost > 0], 0.99) if np.any(spa_cost > 0) else 1.0)
    return alpha_embedding * np.clip(emb_cost, 0, 1) + alpha_spatial * np.clip(spa_cost, 0, 1)


def build_edge_distance_graph(
    embedding: np.ndarray,
    coords: np.ndarray,
    topology: sparse.csr_matrix,
    alpha_embedding: float = 0.7,
    alpha_spatial: float = 0.3,
    min_edge_distance: float = 1e-4,
    standardize_embedding: bool = True,
    standardize_spatial: bool = False,
    edge_chunk_size: int = 100000,
    quantile_sample_size: int = 500000,
    spatial_cost_mode: str = "euclidean",
) -> sparse.csr_matrix:
    """Assign symmetric embedding-plus-spatial distances to topology edges."""

    if spatial_cost_mode not in {"euclidean", "binary"}:
        raise ValueError("spatial_cost_mode should be 'euclidean' or 'binary'")
    upper = sparse.triu(topology, k=1).tocoo()
    if upper.nnz == 0:
        raise RuntimeError("The topology graph contains no undirected edges")

    embedding = np.asarray(embedding, dtype=np.float32)
    coords = _check_3d_coords(coords, "coords")
    if standardize_embedding:
        emb_scale = np.nanstd(embedding, axis=0).astype(np.float32)
        emb_scale[emb_scale < 1e-12] = 1.0
    else:
        emb_scale = np.ones(embedding.shape[1], dtype=np.float32)
    if standardize_spatial:
        spatial_scale = np.nanstd(coords, axis=0).astype(np.float32)
        spatial_scale[spatial_scale < 1e-12] = 1.0
    else:
        spatial_scale = np.ones(3, dtype=np.float32)

    edge_chunk_size = max(int(edge_chunk_size), 1)

    def raw_cost(edge_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emb_out = np.empty(edge_positions.size, dtype=np.float32)
        spa_out = np.empty(edge_positions.size, dtype=np.float32)
        for start in range(0, edge_positions.size, edge_chunk_size):
            stop = min(start + edge_chunk_size, edge_positions.size)
            pos = edge_positions[start:stop]
            rows = upper.row[pos]
            cols = upper.col[pos]
            emb_diff = (embedding[rows] - embedding[cols]) / emb_scale
            emb_out[start:stop] = np.einsum("ij,ij->i", emb_diff, emb_diff)
            if spatial_cost_mode == "binary":
                spa_out[start:stop] = 1.0
            else:
                xyz_diff = (coords[rows] - coords[cols]) / spatial_scale
                spa_out[start:stop] = np.einsum("ij,ij->i", xyz_diff, xyz_diff)
        return emb_out, spa_out

    if upper.nnz <= quantile_sample_size:
        sample_positions = np.arange(upper.nnz, dtype=np.int64)
    else:
        rng = np.random.default_rng(42)
        sample_positions = np.sort(
            rng.choice(upper.nnz, size=int(quantile_sample_size), replace=False)
        )
    emb_sample, spa_sample = raw_cost(sample_positions)
    positive_emb = emb_sample[emb_sample > 0]
    positive_spa = spa_sample[spa_sample > 0]
    emb_denom = float(np.quantile(positive_emb, 0.99)) if positive_emb.size else 1.0
    spa_denom = 1.0 if spatial_cost_mode == "binary" else (
        float(np.quantile(positive_spa, 0.99)) if positive_spa.size else 1.0
    )

    distance = np.empty(upper.nnz, dtype=np.float32)
    for start in range(0, upper.nnz, edge_chunk_size):
        stop = min(start + edge_chunk_size, upper.nnz)
        emb_cost, spa_cost = raw_cost(np.arange(start, stop, dtype=np.int64))
        distance[start:stop] = (
            alpha_embedding * np.clip(emb_cost / emb_denom, 0.0, 1.0)
            + alpha_spatial * np.clip(spa_cost / spa_denom, 0.0, 1.0)
        )
    np.maximum(distance, float(min_edge_distance), out=distance)
    upper_distance = sparse.csr_matrix(
        (distance, (upper.row, upper.col)),
        shape=topology.shape,
    )
    return (upper_distance + upper_distance.T).tocsr()


def graph_softmax_from_edge_distance(
    edge_distance_graph: sparse.csr_matrix,
    lambd: float = 1e-1,
) -> sparse.csr_matrix:
    """Convert a sparse edge-distance graph into row-normalized transition weights."""

    transition = edge_distance_graph.copy().astype(np.float32)
    transition.data = np.exp(
        -transition.data / max(float(lambd), 1e-8)
    ).astype(np.float32)
    row_sum = np.asarray(transition.sum(axis=1)).ravel()
    row_scale = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(row_sum),
        where=row_sum > 0,
    )
    return sparse.diags(row_scale).dot(transition).tocsr()


def compute_transport_matrix(
    embedding: np.ndarray,
    coords: np.ndarray,
    topology: Optional[sparse.csr_matrix] = None,
    alpha_embedding: float = 0.7,
    alpha_spatial: float = 0.3,
    lambd: float = 1e-1,
    num_iter_max: int = 1000,
    mode: str = "sinkhorn",
    cost_quantile: float = 0.99,
    topology_penalty: float = 0.25,
    non_edge_cost: Optional[float] = None,
    standardize_embedding: bool = True,
    standardize_spatial: bool = False,
    spatial_cost_mode: str = "euclidean",
):
    """Compute sampled-cell transport from embedding and 3D spatial costs."""

    embedding = _as_2d_array(embedding, "embedding").astype(np.float32, copy=False)
    coords = _check_3d_coords(coords, "coords")
    if embedding.shape[0] != coords.shape[0]:
        raise ValueError("embedding and coords should contain the same number of rows")

    if mode not in {"sinkhorn", "graph_softmax"}:
        raise ValueError("mode should be 'sinkhorn' or 'graph_softmax'")
    if spatial_cost_mode not in {"euclidean", "binary"}:
        raise ValueError("spatial_cost_mode should be 'euclidean' or 'binary'")

    if mode == "graph_softmax":
        if topology is None:
            raise ValueError("graph_softmax mode requires a topology graph")
        coo = topology.tocoo()
        costs = _edge_costs(
            embedding,
            coords,
            coo.row,
            coo.col,
            alpha_embedding,
            alpha_spatial,
            standardize_embedding,
            standardize_spatial,
            spatial_cost_mode=spatial_cost_mode,
        )
        weights = np.exp(-costs / max(float(lambd), 1e-8)).astype(np.float32)
        trans = sparse.csr_matrix((weights, (coo.row, coo.col)), shape=topology.shape)
        row_sum = np.asarray(trans.sum(axis=1)).ravel()
        row_scale = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
        return sparse.diags(row_scale).dot(trans).tocsr()

    emb, _ = _standardize_train_apply(embedding, enabled=standardize_embedding)
    xyz, _ = _standardize_train_apply(coords, enabled=standardize_spatial)

    cost = alpha_embedding * _normalize_cost(_pairwise_squared_distance(emb), cost_quantile)
    if alpha_spatial:
        if spatial_cost_mode == "binary":
            if topology is None:
                raise ValueError("spatial_cost_mode='binary' requires a topology graph")
            spatial_cost = topology.astype(bool).toarray().astype(np.float32)
            cost += alpha_spatial * spatial_cost
        else:
            cost += alpha_spatial * _normalize_cost(_pairwise_squared_distance(xyz), cost_quantile)

    if topology is not None and (topology_penalty or non_edge_cost is not None):
        mask = topology.astype(bool).toarray()
        np.fill_diagonal(mask, True)
        if non_edge_cost is not None:
            cost[~mask] = float(non_edge_cost)
        else:
            cost[~mask] += float(topology_penalty)

    max_cost = float(np.nanmax(cost)) if np.isfinite(cost).any() else 1.0
    np.fill_diagonal(cost, max_cost * 1000.0)

    try:
        import ot
    except ImportError as error:
        raise ImportError("POT is required for sinkhorn mode. Install package 'POT' or use mode='graph_softmax'.") from error

    n = cost.shape[0]
    marginal = np.full(n, 1.0 / n, dtype=np.float64)
    transport = ot.sinkhorn_log(
        marginal,
        marginal,
        cost.astype(np.float64, copy=False),
        float(lambd),
        numItermax=int(num_iter_max),
    )
    return np.asarray(transport, dtype=np.float32)


def compute_pseudotime(
    transport,
    start_indices: Sequence[int],
    topology: Optional[sparse.csr_matrix] = None,
    mode: str = "transport",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pseudotime on the sampled backbone."""

    start_indices = np.asarray(start_indices, dtype=np.int64)
    if start_indices.size == 0:
        raise ValueError("No start cells are present in the sampled backbone")

    n = transport.shape[0]
    if mode not in {"transport", "graph_distance", "weighted_graph_distance"}:
        raise ValueError(
            "mode should be 'transport', 'graph_distance', or "
            "'weighted_graph_distance'"
        )

    if mode == "transport":
        selected = transport[start_indices]
        if sparse.issparse(selected):
            score = np.asarray(selected.sum(axis=0)).ravel()
        else:
            score = np.asarray(selected.sum(axis=0)).ravel()
        if np.any(score > 0):
            order = np.argsort(score)[::-1]
            ptime = np.empty(n, dtype=np.float32)
            ptime[order] = np.arange(n, dtype=np.float32) / max(n - 1, 1)
            return ptime, score.astype(np.float32)
        if topology is None:
            raise RuntimeError("Transport scores are all zero and no topology was provided")

    if topology is None:
        raise ValueError(f"{mode} pseudotime requires a topology graph")

    distance_graph = topology
    if mode == "weighted_graph_distance":
        coo = topology.tocoo()
        rows = coo.row.astype(np.int64, copy=False)
        cols = coo.col.astype(np.int64, copy=False)
        if sparse.issparse(transport):
            transition = np.asarray(transport[rows, cols]).ravel()
        else:
            transition = np.asarray(transport[rows, cols]).ravel()
        transition = np.clip(transition.astype(np.float64, copy=False), 1e-12, 1.0)
        edge_distance = np.maximum(-np.log(transition), 1e-6)
        distance_graph = sparse.csr_matrix(
            (edge_distance, (rows, cols)), shape=topology.shape
        )

    dist = dijkstra(
        distance_graph,
        directed=False,
        indices=start_indices,
        min_only=True,
    )
    finite = np.isfinite(dist)
    ptime = np.full(n, np.nan, dtype=np.float32)
    if finite.any():
        d = dist[finite]
        denom = float(d.max() - d.min())
        ptime[finite] = 0.0 if denom <= 0 else ((d - d.min()) / denom).astype(np.float32)
    score = np.zeros(n, dtype=np.float32)
    score[finite] = 1.0 / (1.0 + dist[finite]).astype(np.float32)
    return ptime, score


def compute_velocity_on_topology(
    coords: np.ndarray,
    ptime: np.ndarray,
    transport,
    topology: sparse.csr_matrix,
    forward_only: bool = True,
    normalize_displacement: bool = True,
    average_by_weight: bool = True,
) -> np.ndarray:
    """Compute cell-level 3D velocity vectors along topology edges."""

    coords = _check_3d_coords(coords, "coords").astype(np.float32, copy=False)
    ptime = np.asarray(ptime, dtype=np.float32)
    coo = topology.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)

    if sparse.issparse(transport):
        vals = np.asarray(transport[rows, cols]).ravel().astype(np.float32)
    else:
        vals = np.asarray(transport[rows, cols], dtype=np.float32)

    row_sum = np.bincount(rows, weights=vals, minlength=coords.shape[0]).astype(np.float32)
    fallback = vals <= 0
    if fallback.any():
        vals[fallback] = 1.0
        row_sum = np.bincount(rows, weights=vals, minlength=coords.shape[0]).astype(np.float32)
    vals = vals / np.maximum(row_sum[rows], 1e-12)

    delta_t = ptime[cols] - ptime[rows]
    valid = np.isfinite(delta_t)
    if forward_only:
        valid &= delta_t > 0
        signed_vals = vals
    else:
        signed_vals = vals * np.where(delta_t >= 0, 1.0, -1.0).astype(np.float32)

    rows = rows[valid]
    cols = cols[valid]
    signed_vals = signed_vals[valid]
    disp = coords[cols] - coords[rows]
    if normalize_displacement:
        norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp = disp / np.maximum(norm, 1e-12)

    weighted = disp * signed_vals[:, None]
    velocity = np.zeros_like(coords, dtype=np.float32)
    np.add.at(velocity, rows, weighted)

    if average_by_weight:
        denom = np.bincount(rows, weights=np.abs(signed_vals), minlength=coords.shape[0]).astype(np.float32)
        velocity = velocity / np.maximum(denom[:, None], 1e-12)
    return velocity


def map_backbone_to_full(
    coords_full: np.ndarray,
    coords_backbone: np.ndarray,
    ptime_backbone: np.ndarray,
    velocity_backbone: np.ndarray,
    valid_full_mask: Optional[np.ndarray] = None,
    n_neighbors: int = 8,
    chunk_size: int = 200000,
    backbone_components: Optional[np.ndarray] = None,
    component_has_root: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map sampled-backbone pseudotime and velocity back to all cells by spatial interpolation."""

    coords_full = _check_3d_coords(coords_full, "coords_full")
    coords_backbone = _check_3d_coords(coords_backbone, "coords_backbone")
    if valid_full_mask is None:
        valid_full_mask = np.ones(coords_full.shape[0], dtype=bool)

    n_neighbors = min(int(n_neighbors), coords_backbone.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(coords_backbone)
    ptime_full = np.full(coords_full.shape[0], np.nan, dtype=np.float32)
    velocity_full = np.full(coords_full.shape, np.nan, dtype=np.float32)

    if backbone_components is not None:
        backbone_components = np.asarray(backbone_components, dtype=np.int64)
        if backbone_components.shape != (coords_backbone.shape[0],):
            raise ValueError("backbone_components should have one value per backbone cell")
        if component_has_root is None:
            raise ValueError("component_has_root is required with backbone_components")
        component_has_root = np.asarray(component_has_root, dtype=bool)

    valid_indices = np.where(valid_full_mask)[0]
    for start in range(0, valid_indices.size, chunk_size):
        idx = valid_indices[start : start + chunk_size]
        dist, neigh = nn.kneighbors(coords_full[idx])
        scale = np.nanmedian(dist[:, -1]) if dist.size else 1.0
        scale = max(float(scale), 1e-6)
        weight = np.exp(-0.5 * (dist / scale) ** 2).astype(np.float32)
        if backbone_components is not None:
            query_component = backbone_components[neigh[:, 0]]
            same_component = backbone_components[neigh] == query_component[:, None]
            rooted = component_has_root[query_component]
            weight *= same_component & rooted[:, None]
        weight_sum = weight.sum(axis=1, keepdims=True)
        weight = weight / np.maximum(weight_sum, 1e-12)
        ptime_neighbors = np.where(weight > 0, ptime_backbone[neigh], 0.0)
        velocity_neighbors = np.where(
            weight[:, :, None] > 0,
            velocity_backbone[neigh],
            0.0,
        )
        ptime_full[idx] = np.sum(weight * ptime_neighbors, axis=1)
        velocity_full[idx] = np.sum(weight[:, :, None] * velocity_neighbors, axis=1)
        unassigned = weight_sum[:, 0] <= 0
        if unassigned.any():
            ptime_full[idx[unassigned]] = np.nan
            velocity_full[idx[unassigned]] = np.nan
    return ptime_full, velocity_full


def interpolate_to_grid(
    coords: np.ndarray,
    velocity: np.ndarray,
    grid_shape: Optional[Sequence[int]] = (30, 30, 18),
    n_neighbors: int = 24,
    density_quantile: float = 0.85,
    max_grid_distance: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Interpolate sampled velocity to a filtered 3D grid for vector visualization."""

    if grid_shape is None:
        return None, None
    coords = _check_3d_coords(coords, "coords")
    velocity = _check_3d_coords(velocity, "velocity")
    if len(grid_shape) == 1:
        grid_shape = (int(grid_shape[0]),) * 3
    if len(grid_shape) != 3:
        raise ValueError("grid_shape should be an int-like sequence of length 3")

    mins = np.nanpercentile(coords, 0.5, axis=0)
    maxs = np.nanpercentile(coords, 99.5, axis=0)
    grids = [np.linspace(mins[d], maxs[d], int(grid_shape[d])) for d in range(3)]
    mesh = np.meshgrid(*grids, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)

    n_neighbors = min(int(n_neighbors), coords.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(coords)
    dist, neigh = nn.kneighbors(grid_points)
    nearest = dist[:, 0]
    if max_grid_distance is not None:
        keep = nearest <= float(max_grid_distance)
    else:
        keep = nearest <= np.quantile(nearest, float(density_quantile))
    grid_points = grid_points[keep]
    dist = dist[keep]
    neigh = neigh[keep]

    scale = np.nanmedian(dist[:, -1]) if dist.size else 1.0
    scale = max(float(scale), 1e-6)
    weight = np.exp(-0.5 * (dist / scale) ** 2).astype(np.float32)
    weight = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    grid_velocity = np.sum(weight[:, :, None] * velocity[neigh], axis=1).astype(np.float32)
    return grid_points, grid_velocity


def _select_start_indices(
    adata,
    coords: np.ndarray,
    valid_mask: np.ndarray,
    start_indices: Optional[Sequence[int]] = None,
    start_mask: Optional[Sequence[bool]] = None,
    start_obs: Optional[str] = None,
    start_values: Optional[Iterable[object]] = None,
    start_point: Optional[Sequence[float]] = None,
    n_start_neighbors: int = 128,
) -> np.ndarray:
    starts = []
    if start_indices is not None:
        starts.append(np.asarray(start_indices, dtype=np.int64))
    if start_mask is not None:
        mask = np.asarray(start_mask, dtype=bool)
        if mask.shape[0] != coords.shape[0]:
            raise ValueError("start_mask should have one value per cell")
        starts.append(np.where(mask)[0])
    if start_obs is not None:
        if start_values is None:
            raise ValueError("start_values should be supplied with start_obs")
        if start_obs not in adata.obs.columns:
            raise KeyError(f"obs['{start_obs}'] was not found")
        values = {str(v) for v in start_values}
        starts.append(np.where(adata.obs[start_obs].astype(str).isin(values))[0])
    if start_point is not None:
        valid_indices = np.where(valid_mask)[0]
        point = np.asarray(start_point, dtype=np.float32).reshape(1, -1)
        if point.shape[1] != 3:
            raise ValueError("start_point should contain exactly three coordinates")
        nn = NearestNeighbors(n_neighbors=min(n_start_neighbors, valid_indices.size)).fit(coords[valid_indices])
        _, neigh = nn.kneighbors(point)
        starts.append(valid_indices[neigh[0]])

    if not starts:
        raise ValueError("Provide start_obs/start_values, start_point, start_mask, or start_indices")
    out = np.unique(np.concatenate(starts).astype(np.int64))
    out = out[valid_mask[out]]
    if out.size == 0:
        raise ValueError("No valid start cells were selected")
    return out


def run_embedding_trajectory_3d(
    adata,
    spatial_key: str = "ccf",
    rep_key: str = "cell_embedding",
    start_obs: Optional[str] = None,
    start_values: Optional[Iterable[object]] = None,
    start_point: Optional[Sequence[float]] = None,
    start_indices: Optional[Sequence[int]] = None,
    start_mask: Optional[Sequence[bool]] = None,
    slice_key: Optional[str] = None,
    sample_size: Optional[int] = 8192,
    random_state: int = 42,
    axis: int = 2,
    sample_axis_bins: int = 24,
    axis_gap_multiplier: float = 8.0,
    axis_gap_threshold: Optional[float] = None,
    within_neighbors: int = 12,
    cross_neighbors: int = 4,
    coord_scale: Optional[Sequence[float]] = None,
    max_within_distance: Optional[float] = None,
    max_cross_lateral_distance: Optional[float] = None,
    max_cross_axis_gap: Optional[float] = None,
    topology_mode: str = "section_knn",
    topology_neighbors: int = 16,
    topology_query_chunk_size: int = 8192,
    local_scale_neighbors: int = 8,
    edge_length_factor: Optional[float] = 1.5,
    max_edge_length: Optional[float] = None,
    alpha_embedding: float = 0.7,
    alpha_spatial: float = 0.3,
    lambd: float = 1e-1,
    num_iter_max: int = 1000,
    transport_mode: str = "sinkhorn",
    topology_penalty: float = 0.25,
    non_edge_cost: Optional[float] = None,
    spatial_cost_mode: str = "euclidean",
    ptime_mode: str = "transport",
    grid_shape: Optional[Sequence[int]] = (30, 30, 18),
    grid_density_quantile: float = 0.85,
    map_full: bool = True,
    map_neighbors: int = 8,
    ptime_key: str = "ptime_3d",
    velocity_key: str = "velocity_3d",
    uns_key: str = "trajectory_3d",
    store_transport: bool = False,
) -> Trajectory3DResult:
    """Run an embedding-based, gap-aware 3D trajectory on a sampled backbone."""

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] was not found")
    if rep_key not in adata.obsm:
        raise KeyError(f"obsm['{rep_key}'] was not found")

    coords_all = _check_3d_coords(adata.obsm[spatial_key], spatial_key)
    emb_all = _as_2d_array(adata.obsm[rep_key], rep_key).astype(np.float32, copy=False)
    if coords_all.shape[0] != emb_all.shape[0]:
        raise ValueError(f"obsm['{spatial_key}'] and obsm['{rep_key}'] have different cell counts")

    valid_mask = _finite_mask(coords_all, emb_all)
    valid_indices = np.where(valid_mask)[0]
    start_full = _select_start_indices(
        adata,
        coords_all,
        valid_mask,
        start_indices=start_indices,
        start_mask=start_mask,
        start_obs=start_obs,
        start_values=start_values,
        start_point=start_point,
    )

    if slice_key is not None:
        full_groups = _obs_groups(adata, slice_key, coords_all[:, axis])
        strata = full_groups[valid_indices]
    else:
        strata = _axis_bin_labels(coords_all[valid_indices], axis=axis, n_bins=sample_axis_bins)

    if topology_mode == "streaming_mutual_knn_3d":
        backbone_indices = valid_indices
    else:
        backbone_indices = sample_backbone_indices(
            valid_indices,
            sample_size=sample_size,
            random_state=random_state,
            include_indices=start_full,
            strata=strata,
        )
    coords = coords_all[backbone_indices]
    emb = emb_all[backbone_indices]

    if slice_key is not None:
        groups = full_groups[backbone_indices]
    else:
        groups = _axis_groups_from_gaps(
            coords,
            axis=axis,
            gap_multiplier=axis_gap_multiplier,
            gap_threshold=axis_gap_threshold,
        )

    if topology_mode == "section_knn":
        topology = build_gap_aware_topology(
            coords,
            groups=groups,
            axis=axis,
            within_neighbors=within_neighbors,
            cross_neighbors=cross_neighbors,
            coord_scale=coord_scale,
            max_within_distance=max_within_distance,
            max_cross_lateral_distance=max_cross_lateral_distance,
            max_cross_axis_gap=max_cross_axis_gap,
        )
    elif topology_mode in {"delaunay_3d", "pure_delaunay_3d", "mutual_knn_3d"}:
        topology = build_geometry_topology_3d(
            coords,
            mode=topology_mode,
            n_neighbors=topology_neighbors,
            local_scale_neighbors=local_scale_neighbors,
            edge_length_factor=edge_length_factor,
            max_edge_length=max_edge_length,
            coord_scale=coord_scale,
        )
    elif topology_mode == "streaming_mutual_knn_3d":
        topology = build_streaming_mutual_knn_topology_3d(
            coords,
            n_neighbors=topology_neighbors,
            query_chunk_size=topology_query_chunk_size,
            max_edge_length=max_edge_length,
            coord_scale=coord_scale,
        )
    else:
        raise ValueError(
            "topology_mode should be 'section_knn', 'delaunay_3d', "
            "'pure_delaunay_3d', 'mutual_knn_3d', or 'streaming_mutual_knn_3d'"
        )

    edge_distance_graph = build_edge_distance_graph(
        emb,
        coords,
        topology,
        alpha_embedding=alpha_embedding,
        alpha_spatial=alpha_spatial,
        spatial_cost_mode=spatial_cost_mode,
    )
    if transport_mode == "graph_softmax":
        transport = graph_softmax_from_edge_distance(
            edge_distance_graph,
            lambd=lambd,
        )
    else:
        transport = compute_transport_matrix(
            emb,
            coords,
            topology=topology,
            alpha_embedding=alpha_embedding,
            alpha_spatial=alpha_spatial,
            lambd=lambd,
            num_iter_max=num_iter_max,
            mode=transport_mode,
            topology_penalty=topology_penalty,
            non_edge_cost=non_edge_cost,
            spatial_cost_mode=spatial_cost_mode,
        )

    start_positions = np.searchsorted(backbone_indices, start_full)
    present = (
        (start_positions < backbone_indices.size)
        & (backbone_indices[np.minimum(start_positions, backbone_indices.size - 1)] == start_full)
    )
    start_local = start_positions[present].astype(np.int64, copy=False)
    n_components, component_labels = sparse.csgraph.connected_components(
        topology,
        directed=False,
    )
    root_components = np.unique(component_labels[start_local])
    component_has_root = np.isin(np.arange(n_components), root_components)
    print(
        "3D topology: "
        f"mode={topology_mode}, nodes={topology.shape[0]:,}, "
        f"edges={topology.nnz // 2:,}, components={n_components:,}, "
        f"rooted_components={int(component_has_root.sum()):,}"
    )

    if ptime_mode == "edge_cost_graph_distance":
        ptime, tran_score = compute_pseudotime(
            transport,
            start_local,
            topology=edge_distance_graph,
            mode="graph_distance",
        )
    else:
        ptime, tran_score = compute_pseudotime(
            transport,
            start_local,
            topology=topology,
            mode=ptime_mode,
        )
    velocity = compute_velocity_on_topology(coords, ptime, transport, topology)
    rooted_backbone = np.isfinite(ptime)
    velocity[~rooted_backbone] = np.nan
    grid_points, grid_velocity = interpolate_to_grid(
        coords[rooted_backbone],
        velocity[rooted_backbone],
        grid_shape=grid_shape,
        density_quantile=grid_density_quantile,
    )

    covers_all_valid = backbone_indices.size == valid_indices.size and np.array_equal(
        backbone_indices,
        valid_indices,
    )
    if covers_all_valid:
        ptime_full = np.full(coords_all.shape[0], np.nan, dtype=np.float32)
        velocity_full = np.full(coords_all.shape, np.nan, dtype=np.float32)
        ptime_full[backbone_indices] = ptime
        velocity_full[backbone_indices] = velocity
        ptime_full[start_full] = 0.0
        adata.obs[ptime_key] = ptime_full
        adata.obsm[velocity_key] = velocity_full
    elif map_full:
        ptime_full, velocity_full = map_backbone_to_full(
            coords_all,
            coords,
            ptime,
            velocity,
            valid_full_mask=valid_mask,
            n_neighbors=map_neighbors,
            backbone_components=component_labels,
            component_has_root=component_has_root,
        )
        # Preserve exact backbone values after interpolation and anchor every
        # user-selected root cell at pseudotime zero.
        ptime_full[backbone_indices] = ptime
        velocity_full[backbone_indices] = velocity
        ptime_full[start_full] = 0.0
        adata.obs[ptime_key] = ptime_full
        adata.obsm[velocity_key] = velocity_full
    else:
        ptime_full = np.full(coords_all.shape[0], np.nan, dtype=np.float32)
        velocity_full = np.full(coords_all.shape, np.nan, dtype=np.float32)
        ptime_full[backbone_indices] = ptime
        velocity_full[backbone_indices] = velocity
        adata.obs[ptime_key] = ptime_full
        adata.obsm[velocity_key] = velocity_full

    adata.uns[uns_key] = {
        "spatial_key": spatial_key,
        "rep_key": rep_key,
        "ptime_key": ptime_key,
        "velocity_key": velocity_key,
        "backbone_indices": backbone_indices.astype(np.int64),
        "backbone_coords": coords.astype(np.float32),
        "backbone_ptime": ptime.astype(np.float32),
        "backbone_velocity": velocity.astype(np.float32),
        "backbone_tran_score": tran_score.astype(np.float32),
        "topology_indices": topology.indices.astype(np.int64),
        "topology_indptr": topology.indptr.astype(np.int64),
        "topology_shape": np.asarray(topology.shape, dtype=np.int64),
        "backbone_component": component_labels.astype(np.int32),
        "component_has_root": component_has_root.astype(bool),
        "edge_distance_data": edge_distance_graph.data.astype(np.float32),
        "edge_distance_indices": edge_distance_graph.indices.astype(np.int64),
        "edge_distance_indptr": edge_distance_graph.indptr.astype(np.int64),
        "grid_points": None if grid_points is None else grid_points.astype(np.float32),
        "grid_velocity": None if grid_velocity is None else grid_velocity.astype(np.float32),
        "params": {
            "sample_size": sample_size,
            "random_state": random_state,
            "axis": axis,
            "slice_key": slice_key,
            "within_neighbors": within_neighbors,
            "cross_neighbors": cross_neighbors,
            "topology_mode": topology_mode,
            "topology_neighbors": topology_neighbors,
            "topology_query_chunk_size": topology_query_chunk_size,
            "local_scale_neighbors": local_scale_neighbors,
            "edge_length_factor": edge_length_factor,
            "max_edge_length": max_edge_length,
            "alpha_embedding": alpha_embedding,
            "alpha_spatial": alpha_spatial,
            "spatial_cost_mode": spatial_cost_mode,
            "transport_mode": transport_mode,
            "ptime_mode": ptime_mode,
        },
    }
    if store_transport:
        adata.uns[uns_key]["transport"] = transport

    return Trajectory3DResult(
        backbone_indices=backbone_indices,
        backbone_ptime=ptime,
        backbone_velocity=velocity,
        grid_points=grid_points,
        grid_velocity=grid_velocity,
        topology=topology,
        transport_shape=transport.shape,
    )
