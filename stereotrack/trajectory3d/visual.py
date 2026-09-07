from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _hex_colors_from_values(values: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    import matplotlib.cm as cm

    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    normed = np.zeros(values.shape[0], dtype=np.float32)
    if finite.any():
        lo = float(np.nanmin(values[finite]))
        hi = float(np.nanmax(values[finite]))
        denom = max(hi - lo, 1e-12)
        normed[finite] = (values[finite] - lo) / denom
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(normed)
    return (
        (np.clip(rgba[:, 0] * 255, 0, 255).astype(np.uint32) << 16)
        | (np.clip(rgba[:, 1] * 255, 0, 255).astype(np.uint32) << 8)
        | np.clip(rgba[:, 2] * 255, 0, 255).astype(np.uint32)
    )


def _subsample(n: int, max_items: Optional[int], random_state: int) -> np.ndarray:
    if max_items is None or max_items <= 0 or n <= max_items:
        return np.arange(n)
    rng = np.random.RandomState(random_state)
    return np.sort(rng.choice(np.arange(n), size=max_items, replace=False))


def _trajectory_uns(adata, uns_key: str) -> dict:
    if uns_key not in adata.uns:
        raise KeyError(f"adata.uns['{uns_key}'] was not found")
    return adata.uns[uns_key]


def plot_trajectory_k3d(
    adata,
    spatial_key: str = "ccf",
    ptime_key: str = "ptime_3d",
    uns_key: str = "trajectory_3d",
    max_points: int = 60000,
    max_vectors: int = 3000,
    point_size: float = 2.5,
    point_opacity: float = 0.35,
    vector_scale: float = 80.0,
    vector_length_fraction: Optional[float] = 0.025,
    vector_color: int = 0xD73027,
    background_color: int = 0xFFFFFF,
    cmap_name: str = "viridis",
    save_html: Optional[Sequence[str]] = None,
    random_state: int = 42,
):
    """Create a k3d 3D pseudotime scatter plus vector field plot."""

    try:
        import k3d
    except ImportError as error:
        raise ImportError("k3d is not installed in this environment. Install k3d to use plot_trajectory_k3d.") from error

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] was not found")
    if ptime_key not in adata.obs.columns:
        raise KeyError(f"obs['{ptime_key}'] was not found")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :3]
    ptime = np.asarray(adata.obs[ptime_key], dtype=np.float32)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(ptime)
    idx = np.where(valid)[0]
    idx = idx[_subsample(idx.size, max_points, random_state)]
    positions = coords[idx].astype(np.float32)
    colors = _hex_colors_from_values(ptime[idx], cmap_name=cmap_name)

    plot = k3d.plot(name="stereoTrack 3D trajectory", background_color=background_color, camera_auto_fit=True)
    plot += k3d.points(
        positions,
        colors=colors,
        point_size=point_size,
        shader="flat",
        opacity=point_opacity,
        name="cells_ptime",
    )

    info = _trajectory_uns(adata, uns_key)
    grid_points = info.get("grid_points", None)
    grid_velocity = info.get("grid_velocity", None)
    if grid_points is None or grid_velocity is None:
        grid_points = info.get("backbone_coords", None)
        grid_velocity = info.get("backbone_velocity", None)

    if grid_points is not None and grid_velocity is not None:
        grid_points = np.asarray(grid_points, dtype=np.float32)
        grid_velocity = np.asarray(grid_velocity, dtype=np.float32)
        velocity_norm = np.linalg.norm(grid_velocity, axis=1)
        valid_v = (
            np.isfinite(grid_points).all(axis=1)
            & np.isfinite(grid_velocity).all(axis=1)
            & np.isfinite(velocity_norm)
            & (velocity_norm > 1e-8)
        )
        vector_idx = np.where(valid_v)[0]
        vector_idx = vector_idx[_subsample(vector_idx.size, max_vectors, random_state)]
        origins = grid_points[vector_idx].astype(np.float32)
        vectors = grid_velocity[vector_idx].astype(np.float32)
        if vector_length_fraction is not None and vector_length_fraction > 0:
            extent = float(np.nanmax(np.ptp(coords[valid], axis=0)))
            arrow_length = max(extent * float(vector_length_fraction), 1e-6)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-8) * arrow_length
        else:
            vectors *= float(vector_scale)
        plot += k3d.vectors(
            origins,
            vectors,
            color=vector_color,
            head_size=2.5,
            line_width=0.8,
            name="trajectory_vectors",
        )

    plot.grid_visible = False
    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(plot.get_snapshot())
    return plot


def plot_trajectory_plotly_cones(
    adata,
    spatial_key: str = "ccf",
    ptime_key: str = "ptime_3d",
    uns_key: str = "trajectory_3d",
    max_points: int = 80000,
    max_vectors: int = 5000,
    vector_scale: float = 1.0,
    save_html: Optional[Sequence[str]] = None,
    random_state: int = 42,
):
    """Fallback interactive 3D cone plot using plotly."""

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError("plotly is not installed in this environment.") from error

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :3]
    ptime = np.asarray(adata.obs[ptime_key], dtype=np.float32)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(ptime)
    idx = np.where(valid)[0]
    idx = idx[_subsample(idx.size, max_points, random_state)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords[idx, 0],
            y=coords[idx, 1],
            z=coords[idx, 2],
            mode="markers",
            marker=dict(size=1.5, color=ptime[idx], colorscale="Viridis", opacity=0.65),
            name="cells_ptime",
        )
    )

    info = _trajectory_uns(adata, uns_key)
    grid_points = info.get("grid_points", None)
    grid_velocity = info.get("grid_velocity", None)
    if grid_points is None or grid_velocity is None:
        grid_points = info.get("backbone_coords", None)
        grid_velocity = info.get("backbone_velocity", None)

    if grid_points is not None and grid_velocity is not None:
        grid_points = np.asarray(grid_points, dtype=np.float32)
        grid_velocity = np.asarray(grid_velocity, dtype=np.float32)
        valid_v = np.isfinite(grid_points).all(axis=1) & np.isfinite(grid_velocity).all(axis=1)
        vector_idx = np.where(valid_v)[0]
        vector_idx = vector_idx[_subsample(vector_idx.size, max_vectors, random_state)]
        vec = grid_velocity[vector_idx] * float(vector_scale)
        fig.add_trace(
            go.Cone(
                x=grid_points[vector_idx, 0],
                y=grid_points[vector_idx, 1],
                z=grid_points[vector_idx, 2],
                u=vec[:, 0],
                v=vec[:, 1],
                w=vec[:, 2],
                sizemode="absolute",
                sizeref=1.0,
                anchor="tail",
                colorscale="Greys",
                showscale=False,
                name="trajectory_vectors",
            )
        )

    fig.update_layout(
        template="simple_white",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, b=0, t=20),
    )
    if save_html is not None:
        fig.write_html(str(save_html), include_plotlyjs="cdn")
    return fig


def plot_trajectory_plotly_streamtube(
    adata,
    spatial_key: str = "ccf",
    ptime_key: str = "ptime_3d",
    uns_key: str = "trajectory_3d",
    max_points: int = 80000,
    max_tubes: int = 12000,
    vector_scale: float = 1.0,
    root_key: str = "origin_root",
    save_html: Optional[Sequence[str]] = None,
    random_state: int = 42,
):
    """Fallback interactive 3D streamtube plot using Plotly."""

    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError("plotly is not installed in this environment.") from error

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :3]
    ptime = np.asarray(adata.obs[ptime_key], dtype=np.float32)
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(ptime)
    idx = np.where(valid)[0]
    idx = idx[_subsample(idx.size, max_points, random_state)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords[idx, 0],
            y=coords[idx, 1],
            z=coords[idx, 2],
            mode="markers",
            marker=dict(size=1.2, color=ptime[idx], colorscale="Viridis", opacity=0.45),
            name="cells_ptime",
        )
    )
    if root_key in adata.obs.columns:
        roots = np.asarray(adata.obs[root_key], dtype=bool) & valid
        root_idx = np.where(roots)[0]
        if root_idx.size:
            fig.add_trace(
                go.Scatter3d(
                    x=coords[root_idx, 0],
                    y=coords[root_idx, 1],
                    z=coords[root_idx, 2],
                    mode="markers",
                    marker=dict(
                        size=3.5,
                        color="#D73027",
                        opacity=0.95,
                        line=dict(color="#FFFFFF", width=0.5),
                    ),
                    name="entropy_roots",
                )
            )

    info = _trajectory_uns(adata, uns_key)
    grid_points = info.get("grid_points", None)
    grid_velocity = info.get("grid_velocity", None)
    if grid_points is None or grid_velocity is None:
        grid_points = info.get("backbone_coords", None)
        grid_velocity = info.get("backbone_velocity", None)

    if grid_points is not None and grid_velocity is not None:
        grid_points = np.asarray(grid_points, dtype=np.float32)
        grid_velocity = np.asarray(grid_velocity, dtype=np.float32)
        valid_v = np.isfinite(grid_points).all(axis=1) & np.isfinite(grid_velocity).all(axis=1)
        vector_idx = np.where(valid_v)[0]
        vector_idx = vector_idx[_subsample(vector_idx.size, max_tubes, random_state)]
        vec = grid_velocity[vector_idx] * float(vector_scale)
        fig.add_trace(
            go.Streamtube(
                x=grid_points[vector_idx, 0],
                y=grid_points[vector_idx, 1],
                z=grid_points[vector_idx, 2],
                u=vec[:, 0],
                v=vec[:, 1],
                w=vec[:, 2],
                colorscale="Viridis",
                showscale=False,
                maxdisplayed=max_tubes,
                sizeref=0.5,
                name="trajectory_streamtube",
            )
        )

    fig.update_layout(
        template="simple_white",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, b=0, t=20),
    )
    if save_html is not None:
        fig.write_html(str(save_html), include_plotlyjs="cdn")
    return fig


def plot_trajectory_streamplot_projections(
    adata,
    spatial_key: str = "ccf",
    ptime_key: str = "ptime_3d",
    uns_key: str = "trajectory_3d",
    grid_size: int = 70,
    n_neighbors: int = 24,
    density_quantile: float = 0.85,
    max_points: int = 80000,
    root_key: str = "origin_root",
    save_path: Optional[Sequence[str]] = None,
    random_state: int = 42,
):
    """Draw XY, XZ, and YZ projections as publication-friendly streamplots."""

    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors

    coords_all = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :3]
    ptime_all = np.asarray(adata.obs[ptime_key], dtype=np.float32)
    valid_cells = np.isfinite(coords_all).all(axis=1) & np.isfinite(ptime_all)
    cell_idx = np.where(valid_cells)[0]
    cell_idx = cell_idx[_subsample(cell_idx.size, max_points, random_state)]
    root_idx = np.empty(0, dtype=np.int64)
    if root_key in adata.obs.columns:
        root_mask = np.asarray(adata.obs[root_key], dtype=bool) & valid_cells
        root_idx = np.where(root_mask)[0]

    info = _trajectory_uns(adata, uns_key)
    coords = np.asarray(info["backbone_coords"], dtype=np.float32)
    velocity = np.asarray(info["backbone_velocity"], dtype=np.float32)
    ptime = np.asarray(info["backbone_ptime"], dtype=np.float32)
    valid = (
        np.isfinite(coords).all(axis=1)
        & np.isfinite(velocity).all(axis=1)
        & np.isfinite(ptime)
    )
    coords = coords[valid]
    velocity = velocity[valid]
    if coords.shape[0] < 2:
        raise ValueError("At least two finite trajectory points are required")

    projections = [((0, 1), "X", "Y"), ((0, 2), "X", "Z"), ((1, 2), "Y", "Z")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
    for ax, (dims, x_label, y_label) in zip(axes, projections):
        projected = coords[:, dims]
        lower = np.nanpercentile(projected, 0.5, axis=0)
        upper = np.nanpercentile(projected, 99.5, axis=0)
        gx = np.linspace(lower[0], upper[0], int(grid_size))
        gy = np.linspace(lower[1], upper[1], int(grid_size))
        mesh_x, mesh_y = np.meshgrid(gx, gy)
        query = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

        k = min(int(n_neighbors), projected.shape[0])
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(projected)
        distance, neighbor = nn.kneighbors(query)
        scale = max(float(np.nanmedian(distance[:, -1])), 1e-6)
        weight = np.exp(-0.5 * (distance / scale) ** 2).astype(np.float32)
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
        neighbor_velocity = np.take(velocity[neighbor], dims, axis=2)
        vector = np.sum(weight[:, :, None] * neighbor_velocity, axis=1)
        keep = distance[:, 0] <= np.quantile(distance[:, 0], density_quantile)
        u = vector[:, 0].reshape(mesh_x.shape)
        v = vector[:, 1].reshape(mesh_y.shape)
        mask = ~keep.reshape(mesh_x.shape) | ~np.isfinite(u) | ~np.isfinite(v)
        u = np.ma.array(u, mask=mask)
        v = np.ma.array(v, mask=mask)
        speed = np.ma.array(np.hypot(u, v), mask=mask)

        ax.scatter(
            coords_all[cell_idx, dims[0]],
            coords_all[cell_idx, dims[1]],
            c=ptime_all[cell_idx],
            s=1.0,
            cmap="viridis",
            alpha=0.16,
            linewidths=0,
            rasterized=True,
        )
        ax.streamplot(
            gx,
            gy,
            u,
            v,
            color=speed,
            cmap="magma",
            density=1.35,
            linewidth=0.8,
            arrowsize=1.25,
            minlength=0.08,
        )
        if root_idx.size:
            ax.scatter(
                coords_all[root_idx, dims[0]],
                coords_all[root_idx, dims[1]],
                s=18,
                c="#D73027",
                edgecolors="white",
                linewidths=0.35,
                alpha=0.9,
                zorder=5,
                label="entropy roots",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if dims == (0, 1):
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")
        ax.set_title(f"{x_label}{y_label} projection")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    if root_idx.size:
        axes[0].legend(frameon=False, loc="best", markerscale=1.4)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240, bbox_inches="tight")
    return fig


def _farthest_point_subset(points: np.ndarray, max_items: int) -> np.ndarray:
    """Select spatially distributed points with deterministic farthest sampling."""

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= max_items:
        return np.arange(points.shape[0], dtype=np.int64)
    center = np.nanmedian(points, axis=0)
    first = int(np.argmin(np.linalg.norm(points - center, axis=1)))
    selected = np.empty(max_items, dtype=np.int64)
    selected[0] = first
    min_distance = np.linalg.norm(points - points[first], axis=1)
    for position in range(1, max_items):
        next_idx = int(np.argmax(min_distance))
        selected[position] = next_idx
        distance = np.linalg.norm(points - points[next_idx], axis=1)
        np.minimum(min_distance, distance, out=min_distance)
    return selected


def _weighted_field_value(
    tree,
    values: np.ndarray,
    point: np.ndarray,
    n_neighbors: int,
    support_radius: float,
) -> Optional[np.ndarray]:
    distance, neighbor = tree.query(point, k=n_neighbors)
    distance = np.atleast_1d(distance).astype(np.float32, copy=False)
    neighbor = np.atleast_1d(neighbor).astype(np.int64, copy=False)
    if not np.isfinite(distance[0]) or distance[0] > support_radius:
        return None
    scale = max(float(np.median(distance)), support_radius * 0.08, 1e-6)
    weight = np.exp(-0.5 * (distance / scale) ** 2).astype(np.float32)
    weight_sum = float(weight.sum())
    if weight_sum <= 0:
        return None
    return np.sum(values[neighbor] * weight[:, None], axis=0) / weight_sum


def _weighted_scalar_value(
    tree,
    values: np.ndarray,
    point: np.ndarray,
    n_neighbors: int,
) -> float:
    distance, neighbor = tree.query(point, k=n_neighbors)
    distance = np.atleast_1d(distance).astype(np.float32, copy=False)
    neighbor = np.atleast_1d(neighbor).astype(np.int64, copy=False)
    scale = max(float(np.median(distance)), 1e-6)
    weight = np.exp(-0.5 * (distance / scale) ** 2).astype(np.float32)
    return float(np.sum(values[neighbor] * weight) / max(float(weight.sum()), 1e-12))


def plot_trajectory_streamlines_3d(
    adata,
    spatial_key: str = "ccf",
    ptime_key: str = "ptime_3d",
    velocity_key: str = "velocity_3d",
    uns_key: str = "trajectory_3d",
    root_key: str = "origin_root",
    max_points: int = 60000,
    max_streamlines: int = 72,
    max_steps: int = 180,
    step_size: float = 0.009,
    field_neighbors: int = 12,
    min_streamline_points: int = 8,
    arrow_positions: Sequence[float] = (0.52, 0.82),
    arrow_length_fraction: float = 0.032,
    view_angles: Sequence[Sequence[float]] = ((20.0, -65.0), (70.0, -90.0)),
    save_path: Optional[Sequence[str]] = None,
    save_pdf: Optional[Sequence[str]] = None,
    save_data: Optional[Sequence[str]] = None,
    random_state: int = 42,
    title: Optional[str] = None,
):
    """Integrate and draw directed streamlines in the complete 3D coordinate system."""

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from scipy.spatial import cKDTree

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] was not found")
    if ptime_key not in adata.obs.columns:
        raise KeyError(f"obs['{ptime_key}'] was not found")
    if velocity_key not in adata.obsm:
        raise KeyError(f"obsm['{velocity_key}'] was not found")
    if root_key not in adata.obs.columns:
        raise KeyError(f"obs['{root_key}'] was not found")

    coords_all = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[:, :3]
    velocity_all = np.asarray(adata.obsm[velocity_key], dtype=np.float32)[:, :3]
    ptime_all = np.asarray(adata.obs[ptime_key], dtype=np.float32)
    roots_all = np.asarray(adata.obs[root_key], dtype=bool)
    valid_cells = (
        np.isfinite(coords_all).all(axis=1)
        & np.isfinite(velocity_all).all(axis=1)
        & np.isfinite(ptime_all)
    )
    root_idx = np.where(roots_all & valid_cells)[0]
    if root_idx.size == 0:
        raise ValueError("No finite root cells are available for 3D streamline integration")

    info = _trajectory_uns(adata, uns_key)
    field_coords = info.get("grid_points", None)
    field_velocity = info.get("grid_velocity", None)
    if field_coords is None or field_velocity is None:
        field_coords = np.asarray(info["backbone_coords"], dtype=np.float32)
        field_velocity = np.asarray(info["backbone_velocity"], dtype=np.float32)
    else:
        field_coords = np.asarray(field_coords, dtype=np.float32)
        field_velocity = np.asarray(field_velocity, dtype=np.float32)

    field_valid = (
        np.isfinite(field_coords).all(axis=1)
        & np.isfinite(field_velocity).all(axis=1)
        & (np.linalg.norm(field_velocity, axis=1) > 1e-8)
    )
    field_coords = field_coords[field_valid]
    field_velocity = field_velocity[field_valid]
    if field_coords.shape[0] < 8:
        raise ValueError("Too few finite non-zero vectors are available for 3D streamlines")

    finite_coords = coords_all[valid_cells]
    lower = np.nanpercentile(finite_coords, 0.25, axis=0).astype(np.float32)
    upper = np.nanpercentile(finite_coords, 99.75, axis=0).astype(np.float32)
    extent = np.maximum(upper - lower, 1e-6)

    normalized_field = (field_coords - lower) / extent
    normalized_velocity = field_velocity / extent
    in_bounds = (
        np.isfinite(normalized_field).all(axis=1)
        & (normalized_field >= -0.05).all(axis=1)
        & (normalized_field <= 1.05).all(axis=1)
    )
    normalized_field = normalized_field[in_bounds]
    normalized_velocity = normalized_velocity[in_bounds]
    field_tree = cKDTree(normalized_field)
    field_neighbors = min(max(int(field_neighbors), 1), normalized_field.shape[0])

    support_sample = normalized_field[
        _subsample(normalized_field.shape[0], 50000, random_state)
    ]
    support_k = min(2, normalized_field.shape[0])
    support_distance, _ = field_tree.query(support_sample, k=support_k)
    support_distance = np.atleast_2d(support_distance)
    nearest_other = support_distance[:, -1]
    support_radius = max(float(np.quantile(nearest_other, 0.99)) * 3.0, step_size * 3.0)

    ptime_coords = (coords_all[valid_cells] - lower) / extent
    ptime_values = ptime_all[valid_cells]
    ptime_tree = cKDTree(ptime_coords)
    ptime_neighbors = min(8, ptime_coords.shape[0])

    normalized_roots = (coords_all[root_idx] - lower) / extent
    root_support_distance, _ = field_tree.query(normalized_roots, k=1)
    supported_roots = (
        (normalized_roots >= -0.02).all(axis=1)
        & (normalized_roots <= 1.02).all(axis=1)
        & np.isfinite(root_support_distance)
        & (root_support_distance <= support_radius)
    )
    seed_pool = normalized_roots[supported_roots]
    if seed_pool.shape[0] == 0:
        raise RuntimeError("No root cell lies inside the supported 3D velocity field")
    seed_local = _farthest_point_subset(seed_pool, max(int(max_streamlines), 1))
    seed_points = seed_pool[seed_local]

    streamlines = []
    streamline_ptime = []
    for seed in seed_points:
        point = seed.astype(np.float32, copy=True)
        points = [point.copy()]
        times = [0.0]
        for _ in range(max(int(max_steps), 1)):
            vector = _weighted_field_value(
                field_tree,
                normalized_velocity,
                point,
                field_neighbors,
                support_radius,
            )
            if vector is None:
                break
            speed = float(np.linalg.norm(vector))
            if not np.isfinite(speed) or speed <= 1e-8:
                break
            direction = vector / speed
            midpoint = point + 0.5 * float(step_size) * direction
            midpoint_vector = _weighted_field_value(
                field_tree,
                normalized_velocity,
                midpoint,
                field_neighbors,
                support_radius,
            )
            if midpoint_vector is not None:
                midpoint_speed = float(np.linalg.norm(midpoint_vector))
                if np.isfinite(midpoint_speed) and midpoint_speed > 1e-8:
                    direction = midpoint_vector / midpoint_speed
            next_point = point + float(step_size) * direction
            if (next_point < -0.02).any() or (next_point > 1.02).any():
                break
            next_time = _weighted_scalar_value(
                ptime_tree,
                ptime_values,
                next_point,
                ptime_neighbors,
            )
            if next_time + 0.035 < times[-1]:
                break
            next_time = max(next_time, times[-1])
            if len(points) > 12:
                recent = np.asarray(points[-12:-4])
                if np.min(np.linalg.norm(recent - next_point, axis=1)) < step_size * 0.45:
                    break
            points.append(next_point.copy())
            times.append(next_time)
            point = next_point
            if next_time >= 0.995:
                break
        if len(points) >= max(int(min_streamline_points), 2):
            line = lower + np.asarray(points, dtype=np.float32) * extent
            streamlines.append(line)
            streamline_ptime.append(np.asarray(times, dtype=np.float32))

    if not streamlines:
        raise RuntimeError(
            "No 3D streamlines passed the minimum length; increase field support or reduce step_size"
        )

    if save_data is not None:
        data_path = Path(save_data)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with data_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["streamline_id", "point_order", "x", "y", "z", "ptime_3d"])
            for line_id, (line, times) in enumerate(zip(streamlines, streamline_ptime)):
                for point_order, (point, ptime) in enumerate(zip(line, times)):
                    writer.writerow([line_id, point_order, *point.tolist(), float(ptime)])

    display_idx = np.where(valid_cells)[0]
    display_idx = display_idx[_subsample(display_idx.size, max_points, random_state)]
    ptime_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    n_views = len(view_angles)
    if n_views == 0:
        raise ValueError("view_angles should contain at least one (elevation, azimuth) pair")
    fig = plt.figure(figsize=(7.6 * n_views, 7.0))
    axes = [fig.add_subplot(1, n_views, index + 1, projection="3d") for index in range(n_views)]
    max_extent = float(np.max(extent))
    arrow_length = max(max_extent * float(arrow_length_fraction), 1e-6)

    for ax, angle in zip(axes, view_angles):
        scatter = ax.scatter(
            coords_all[display_idx, 0],
            coords_all[display_idx, 1],
            coords_all[display_idx, 2],
            c=ptime_all[display_idx],
            s=0.7,
            cmap="viridis",
            norm=ptime_norm,
            alpha=0.13,
            linewidths=0,
            rasterized=True,
        )
        for line, times in zip(streamlines, streamline_ptime):
            segments = np.stack([line[:-1], line[1:]], axis=1)
            collection = Line3DCollection(
                segments,
                cmap="viridis",
                norm=ptime_norm,
                linewidth=1.45,
                alpha=0.96,
            )
            collection.set_array(0.5 * (times[:-1] + times[1:]))
            ax.add_collection3d(collection)
            for fraction in arrow_positions:
                arrow_idx = min(
                    max(int(round(float(fraction) * (line.shape[0] - 2))), 0),
                    line.shape[0] - 2,
                )
                delta = line[arrow_idx + 1] - line[arrow_idx]
                if np.linalg.norm(delta) <= 1e-8:
                    continue
                ax.quiver(
                    *line[arrow_idx],
                    *delta,
                    length=arrow_length,
                    normalize=True,
                    arrow_length_ratio=0.55,
                    color="#151515",
                    linewidth=1.15,
                )
        ax.scatter(
            coords_all[root_idx, 0],
            coords_all[root_idx, 1],
            coords_all[root_idx, 2],
            s=13,
            c="#D73027",
            edgecolors="white",
            linewidths=0.3,
            alpha=0.9,
            depthshade=False,
            label="entropy roots",
        )
        ax.view_init(elev=float(angle[0]), azim=float(angle[1]))
        ax.set_box_aspect(extent)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.legend(frameon=False, loc="upper right")

    if title:
        fig.suptitle(title, fontsize=14)
    colorbar_ax = fig.add_axes([0.915, 0.22, 0.014, 0.56])
    fig.colorbar(scatter, cax=colorbar_ax, label="3D pseudotime")
    fig.subplots_adjust(left=0.02, right=0.89, bottom=0.02, top=0.92, wspace=0.02)
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=260, bbox_inches="tight")
    if save_pdf is not None:
        path = Path(save_pdf)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig, streamlines
