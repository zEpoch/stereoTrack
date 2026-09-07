from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import scanpy as sc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stereotrack.trajectory3d import (
    plot_trajectory_k3d,
    plot_trajectory_plotly_cones,
    plot_trajectory_plotly_streamtube,
    run_embedding_trajectory_3d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gap-aware 3D trajectory from stereoTrack embeddings.")
    parser.add_argument("--input", type=Path, required=True, help="Input h5ad with 3D coordinates and embeddings.")
    parser.add_argument("--output", type=Path, required=True, help="Output h5ad with ptime_3d and velocity_3d.")
    parser.add_argument("--spatial-key", type=str, default="ccf")
    parser.add_argument("--rep-key", type=str, default="cell_embedding")
    parser.add_argument("--start-obs", type=str, default=None, help="obs column used to select starting cells.")
    parser.add_argument("--start-values", nargs="*", default=None, help="Values in --start-obs used as starting cells.")
    parser.add_argument("--start-point", nargs=3, type=float, default=None, help="3D coordinate used to select starting cells.")
    parser.add_argument("--slice-key", type=str, default=None, help="Optional obs column defining sections/slices.")
    parser.add_argument("--sample-size", type=int, default=8192)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--within-neighbors", type=int, default=12)
    parser.add_argument("--cross-neighbors", type=int, default=4)
    parser.add_argument("--alpha-embedding", type=float, default=0.7)
    parser.add_argument("--alpha-spatial", type=float, default=0.3)
    parser.add_argument("--transport-mode", choices=["sinkhorn", "graph_softmax"], default="sinkhorn")
    parser.add_argument("--ptime-mode", choices=["transport", "graph_distance"], default="transport")
    parser.add_argument("--grid-shape", nargs=3, type=int, default=(30, 30, 18))
    parser.add_argument("--k3d-html", type=Path, default=None)
    parser.add_argument("--plotly-html", type=Path, default=None)
    parser.add_argument("--streamtube-html", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = sc.read_h5ad(args.input)
    run_embedding_trajectory_3d(
        adata,
        spatial_key=args.spatial_key,
        rep_key=args.rep_key,
        start_obs=args.start_obs,
        start_values=args.start_values,
        start_point=args.start_point,
        slice_key=args.slice_key,
        sample_size=args.sample_size,
        random_state=args.random_state,
        within_neighbors=args.within_neighbors,
        cross_neighbors=args.cross_neighbors,
        alpha_embedding=args.alpha_embedding,
        alpha_spatial=args.alpha_spatial,
        transport_mode=args.transport_mode,
        ptime_mode=args.ptime_mode,
        grid_shape=args.grid_shape,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)

    if args.k3d_html is not None:
        plot_trajectory_k3d(adata, spatial_key=args.spatial_key, save_html=args.k3d_html)
    if args.plotly_html is not None:
        plot_trajectory_plotly_cones(adata, spatial_key=args.spatial_key, save_html=args.plotly_html)
    if args.streamtube_html is not None:
        plot_trajectory_plotly_streamtube(adata, spatial_key=args.spatial_key, save_html=args.streamtube_html)


if __name__ == "__main__":
    main()
