# 3D Embedding Trajectory

This module runs a gap-aware 3D trajectory from stereoTrack embeddings.

The intended Fig4 workflow is:

1. sample a reproducible 3D backbone, default `8192` cells;
2. build a topology graph that respects large gaps between sections;
3. compute transport on the sampled backbone from `cell_embedding` or `niche_embedding`;
4. infer pseudotime and 3D velocity on the topology graph;
5. map pseudotime and velocity back to all cells;
6. visualize the result with optional `k3d` vectors or Plotly cones.

Minimal Python usage:

```python
from stereotrack.trajectory3d import run_embedding_trajectory_3d

run_embedding_trajectory_3d(
    adata,
    spatial_key="ccf",
    rep_key="cell_embedding",
    start_obs="cell_type",
    start_values=["Radial glia"],
    slice_key="slice_idx",
    sample_size=8192,
    within_neighbors=12,
    cross_neighbors=4,
)
```

For very large data or environments without POT, use the sparse graph mode:

```python
run_embedding_trajectory_3d(
    adata,
    spatial_key="ccf",
    rep_key="cell_embedding",
    start_obs="cell_type",
    start_values=["Radial glia"],
    transport_mode="graph_softmax",
    ptime_mode="graph_distance",
)
```

Outputs:

- `adata.obs["ptime_3d"]`
- `adata.obsm["velocity_3d"]`
- `adata.uns["trajectory_3d"]`, including sampled backbone coordinates, velocity, topology CSR arrays, and grid vectors

Optional visualization:

```python
from stereotrack.trajectory3d import (
    plot_trajectory_k3d,
    plot_trajectory_plotly_cones,
    plot_trajectory_plotly_streamtube,
)

plot_trajectory_k3d(adata, save_html="trajectory3d_k3d.html")
plot_trajectory_plotly_cones(adata, save_html="trajectory3d_plotly.html")
plot_trajectory_plotly_streamtube(adata, save_html="trajectory3d_streamtube.html")
```

`k3d` is optional and is not required for the trajectory calculation.
