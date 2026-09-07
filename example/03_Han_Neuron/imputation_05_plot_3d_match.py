#!/usr/bin/env python
"""
Plot Han imputation results and Allen ISH in 3D.

For each requested gene this writes two HTML files:
  1. {gene}.cell.html   -- single-cell point clouds for Han_lognorm, StereoTrack, SpaLP
  2. {gene}.volume.html -- Allen-ISH-grid voxel clouds for Han_lognorm, StereoTrack, SpaLP, ISH

Cell-level plotting does not sample cells. It plots all finite cells with
expression greater than --cell-min-value.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import k3d
import matplotlib.colors as mcolors
import numpy as np
import trimesh

UTILS = Path(__file__).resolve().parents[1] / "imputation_benchmark_utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from h5ad_voxel_ish_correlation import (  # noqa: E402
    MatrixReader,
    get_matrix_node,
    ish_gene_map,
    load_ish_volume,
    read_coordinates,
    read_ish_meta,
    read_var_names,
    sort_key,
    voxel_flat_indices,
)


ROOT = Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack")
OUT = ROOT / "out/03_han_mouse_processed_mae_v1_train/imputation_benchmark"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    input_dir: Path
    source: str
    layer: str | None
    normalize: str


def make_cmap():
    hex_colors = ["#0000FF", "#0CFDF1", "#FFFF00", "#FF0000", "#D60000"]
    return mcolors.LinearSegmentedColormap.from_list("custom_cmap", hex_colors)


def normalize_values(values: np.ndarray) -> np.ndarray:
    """Match the plotting normalization used in 03_han_mouse/05_denoise_plot.py."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    expr_min = float(np.nanmin(finite))
    expr_max = float(np.nanmax(finite))
    if expr_max <= expr_min:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - expr_min) / (expr_max - expr_min), 0.0, 1.0).astype(np.float32)


def colors_from_values(values: np.ndarray, cmap) -> np.ndarray:
    norm = normalize_values(values)
    rgba = cmap(norm)
    return (
        (np.clip(rgba[:, 0] * 255, 0, 255).astype(np.uint32) << 16)
        | (np.clip(rgba[:, 1] * 255, 0, 255).astype(np.uint32) << 8)
        | np.clip(rgba[:, 2] * 255, 0, 255).astype(np.uint32)
    )


def scale_zero_center_false(values: np.ndarray, max_value: float, std: float | None = None) -> np.ndarray:
    """Approximate sc.pp.scale(..., zero_center=False, max_value=max_value) for one gene."""
    values = values.astype(np.float32, copy=False)
    if std is None:
        finite = values[np.isfinite(values)]
        if finite.size <= 1:
            return np.zeros_like(values, dtype=np.float32)
        std = float(np.std(finite, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(values, dtype=np.float32)
    scaled = values / std
    if max_value is not None:
        scaled = np.minimum(scaled, float(max_value))
    return scaled.astype(np.float32, copy=False)


def read_gene_values(
    handle: h5py.File,
    gene: str,
    spec: MethodSpec,
    gene_name_column: str | None,
    target_sum: float,
    total_chunk_size: int,
    scale_max_value: float,
    scale_std: float | None = None,
) -> tuple[np.ndarray | None, bool]:
    var_names = read_var_names(handle, gene_name_column)
    var_to_idx = {name: idx for idx, name in enumerate(var_names)}
    if gene not in var_to_idx:
        return None, False

    matrix = MatrixReader(get_matrix_node(handle, spec.source, spec.layer))
    values = matrix.read_cols([var_to_idx[gene]]).reshape(-1).astype(np.float32, copy=False)

    if spec.normalize == "log1p_scale":
        totals = matrix.sum_cols_in_chunks(list(range(int(matrix.shape[1]))), total_chunk_size)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
        values *= float(target_sum)
        np.log1p(values, out=values)
        values = scale_zero_center_false(values, scale_max_value, std=scale_std)

    return values, True


def read_gene_log1p_unscaled(
    handle: h5py.File,
    gene: str,
    spec: MethodSpec,
    gene_name_column: str | None,
    target_sum: float,
    total_chunk_size: int,
) -> tuple[np.ndarray | None, bool]:
    var_names = read_var_names(handle, gene_name_column)
    var_to_idx = {name: idx for idx, name in enumerate(var_names)}
    if gene not in var_to_idx:
        return None, False

    matrix = MatrixReader(get_matrix_node(handle, spec.source, spec.layer))
    values = matrix.read_cols([var_to_idx[gene]]).reshape(-1).astype(np.float32, copy=False)
    totals = matrix.sum_cols_in_chunks(list(range(int(matrix.shape[1]))), total_chunk_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    values *= float(target_sum)
    np.log1p(values, out=values)
    return values, True


def compute_global_log1p_std(
    gene: str,
    spec: MethodSpec,
    file_glob: str,
    gene_name_column: str | None,
    target_sum: float,
    total_chunk_size: int,
) -> float | None:
    """Compute the one-gene std after normalize_total + log1p across all files."""
    if spec.normalize != "log1p_scale":
        return None

    n = 0
    mean = 0.0
    m2 = 0.0
    missing = 0
    for path in list_h5ads(spec.input_dir, file_glob):
        with h5py.File(path, "r") as handle:
            values, present = read_gene_log1p_unscaled(handle, gene, spec, gene_name_column, target_sum, total_chunk_size)
            if not present or values is None:
                missing += 1
                continue
            finite = values[np.isfinite(values)].astype(np.float64, copy=False)
            n_b = int(finite.size)
            if n_b:
                mean_b = float(finite.mean())
                m2_b = float(((finite - mean_b) ** 2).sum())
                if n == 0:
                    n = n_b
                    mean = mean_b
                    m2 = m2_b
                else:
                    delta = mean_b - mean
                    n_new = n + n_b
                    mean = mean + delta * n_b / n_new
                    m2 = m2 + m2_b + delta * delta * n * n_b / n_new
                    n = n_new
        gc.collect()

    if n <= 1:
        print(f"[scale] {spec.name} {gene}: not enough finite values for global scale; missing_files={missing}")
        return None
    std = float(np.sqrt(m2 / (n - 1)))
    print(f"[scale] {spec.name} {gene}: global log1p std={std:.6g}, n={n:,}, missing_files={missing}")
    return std


def list_h5ads(input_dir: Path, file_glob: str) -> list[Path]:
    files = sorted(input_dir.glob(file_glob), key=sort_key)
    if not files:
        raise FileNotFoundError(f"No {file_glob} files found in {input_dir}")
    return files


def collect_cell_points(
    gene: str,
    spec: MethodSpec,
    file_glob: str,
    coord_columns: list[str],
    ccf_scale: float,
    gene_name_column: str | None,
    target_sum: float,
    total_chunk_size: int,
    scale_max_value: float,
    scale_std: float | None,
    cell_min_value: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positions: list[np.ndarray] = []
    values: list[np.ndarray] = []
    stats = {"files": 0, "cells": 0, "finite": 0, "plotted": 0, "gene_missing_files": 0}

    for path in list_h5ads(spec.input_dir, file_glob):
        with h5py.File(path, "r") as handle:
            coords = read_coordinates(handle, "", coord_columns, path.name).astype(np.float32) * float(ccf_scale)
            vals, present = read_gene_values(
                handle,
                gene,
                spec,
                gene_name_column,
                target_sum,
                total_chunk_size,
                scale_max_value,
                scale_std,
            )
            stats["files"] += 1
            stats["cells"] += int(coords.shape[0])
            if not present or vals is None:
                stats["gene_missing_files"] += 1
                continue

            finite = np.isfinite(coords).all(axis=1) & np.isfinite(vals)
            mask = finite & (vals > cell_min_value)
            stats["finite"] += int(finite.sum())
            stats["plotted"] += int(mask.sum())
            if mask.any():
                positions.append(coords[mask, :3].astype(np.float32, copy=False))
                values.append(vals[mask].astype(np.float32, copy=False))

        gc.collect()

    if not positions:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), stats
    return np.concatenate(positions, axis=0), np.concatenate(values, axis=0), stats


def aggregate_gene_to_voxels(
    gene: str,
    spec: MethodSpec,
    file_glob: str,
    coord_columns: list[str],
    ccf_scale: float,
    gene_name_column: str | None,
    target_sum: float,
    total_chunk_size: int,
    scale_max_value: float,
    scale_std: float | None,
    meta: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dims = meta["dims"]
    n_flat = dims[0] * dims[1] * dims[2]
    sum_all = np.zeros(n_flat, dtype=np.float32)
    cnt_all = np.zeros(n_flat, dtype=np.int32)
    stats = {"files": 0, "cells": 0, "finite": 0, "in_bounds": 0, "gene_missing_files": 0}

    for path in list_h5ads(spec.input_dir, file_glob):
        with h5py.File(path, "r") as handle:
            coords = read_coordinates(handle, "", coord_columns, path.name)
            vals, present = read_gene_values(
                handle,
                gene,
                spec,
                gene_name_column,
                target_sum,
                total_chunk_size,
                scale_max_value,
                scale_std,
            )
            stats["files"] += 1
            stats["cells"] += int(coords.shape[0])
            if not present or vals is None:
                stats["gene_missing_files"] += 1
                continue

            flat_idx, in_bounds, finite = voxel_flat_indices(coords, meta, ccf_scale)
            stats["finite"] += int(finite.sum())
            stats["in_bounds"] += int(in_bounds.sum())
            if flat_idx.size == 0:
                continue
            np.add.at(sum_all, flat_idx, vals[in_bounds])
            cnt_all += np.bincount(flat_idx, minlength=n_flat).astype(np.int32)

        gc.collect()

    unique = np.flatnonzero(cnt_all > 0).astype(np.int64)
    if unique.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), stats

    values = sum_all[unique] / cnt_all[unique]
    coords = voxel_centers(unique, meta)
    return coords, values.astype(np.float32, copy=False), stats


def voxel_centers(flat_indices: np.ndarray, meta: dict) -> np.ndarray:
    dims = meta["dims"]
    spacing = meta["spacing"]
    offset = meta["offset"]
    vz = flat_indices // (dims[1] * dims[0])
    vy = (flat_indices % (dims[1] * dims[0])) // dims[0]
    vx = flat_indices % dims[0]
    coords = np.column_stack(
        [
            vx * spacing[0] + offset[0],
            vy * spacing[1] + offset[1],
            vz * spacing[2] + offset[2],
        ]
    )
    return coords.astype(np.float32)


def load_ish_points(gene: str, ish_dir: Path, min_value: float, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    mapping = ish_gene_map(ish_dir)
    if gene not in mapping:
        raise FileNotFoundError(f"No ISH zip found for gene {gene}")
    vol = load_ish_volume(mapping[gene]).astype(np.float32, copy=False)
    mask = np.isfinite(vol) & (vol > min_value)
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    vz, vy, vx = np.nonzero(mask)
    flat = vz.astype(np.int64) * (meta["dims"][1] * meta["dims"][0]) + vy.astype(np.int64) * meta["dims"][0] + vx.astype(np.int64)
    return voxel_centers(flat, meta), vol[mask].astype(np.float32, copy=False)


def scale_ish_meta(meta: dict, scale: float) -> dict:
    """Scale ISH physical coordinates while keeping voxel dimensions unchanged."""
    return {
        **meta,
        "spacing": tuple(float(x) * float(scale) for x in meta["spacing"]),
        "offset": tuple(float(x) * float(scale) for x in meta["offset"]),
    }


def add_mesh(plot, mesh_path: Path, opacity: float = 0.12) -> None:
    if not mesh_path or not mesh_path.exists():
        return
    mesh = trimesh.load(str(mesh_path))
    plot += k3d.mesh(
        vertices=mesh.vertices.astype(np.float32),
        indices=mesh.faces.astype(np.uint32),
        color=0xB0B0B0,
        opacity=opacity,
        wireframe=False,
        name="mouseMesh",
    )


def add_points(plot, name: str, positions: np.ndarray, values: np.ndarray, cmap, point_size: float) -> None:
    if positions.size == 0:
        print(f"[plot] {name}: no points")
        return
    colors = colors_from_values(values, cmap)
    # opacities = normalize_values(values)
    plot += k3d.points(
        positions.astype(np.float32, copy=False),
        colors=colors,
        # opacities=opacities,
        # opacities = np.log(values.astype(np.float32, copy=False)),
        opacities=normalize_values(values),
        point_size=point_size,
        shader="3d",
        name=name,
    )
    print(f"[plot] {name}: {positions.shape[0]:,} points")


def write_cell_html(gene: str, methods: list[MethodSpec], args, cmap, camera) -> list[dict[str, object]]:
    plot = k3d.plot(name=f"{gene} cell-level expression", background_color=0xFFFFFF, camera_auto_fit=False)
    if args.with_mesh:
        add_mesh(plot, args.mesh_path, opacity=args.mesh_opacity)

    rows = []
    for spec in methods:
        t0 = time.time()
        scale_std = compute_global_log1p_std(
            gene,
            spec,
            args.file_glob,
            args.gene_name_column if spec.name != "Han_lognorm" else None,
            args.target_sum,
            args.total_chunk_size,
        )
        positions, values, stats = collect_cell_points(
            gene,
            spec,
            args.file_glob,
            args.coord_columns,
            args.ccf_scale,
            args.gene_name_column if spec.name != "Han_lognorm" else None,
            args.target_sum,
            args.total_chunk_size,
            args.scale_max_value,
            scale_std,
            args.cell_min_value,
        )
        add_points(plot, spec.name, positions, values, cmap, args.cell_point_size)
        stats.update({"gene": gene, "method": spec.name, "mode": "cell", "seconds": round(time.time() - t0, 2)})
        rows.append(stats)
        del positions, values
        gc.collect()

    plot.camera = camera
    plot.grid_visible = False
    out = args.output_dir / f"{gene}.cell.html"
    out.write_text(plot.get_snapshot(), encoding="utf-8")
    print(f"[write] {out}")
    return rows


def write_volume_html(gene: str, methods: list[MethodSpec], args, cmap, camera, meta: dict) -> list[dict[str, object]]:
    plot = k3d.plot(name=f"{gene} voxel expression vs ISH", background_color=0xFFFFFF, camera_auto_fit=False)
    if args.with_mesh:
        add_mesh(plot, args.mesh_path, opacity=args.mesh_opacity)
        
    rows = []

    for spec in methods:
        t0 = time.time()
        scale_std = compute_global_log1p_std(
            gene,
            spec,
            args.file_glob,
            args.gene_name_column if spec.name != "Han_lognorm" else None,
            args.target_sum,
            args.total_chunk_size,
        )
        coords, values, stats = aggregate_gene_to_voxels(
            gene,
            spec,
            args.file_glob,
            args.coord_columns,
            args.ccf_scale,
            args.gene_name_column if spec.name != "Han_lognorm" else None,
            args.target_sum,
            args.total_chunk_size,
            args.scale_max_value,
            scale_std,
            meta,
        )
        mask = np.isfinite(values) & (values > args.volume_min_value)
        coords = coords[mask]
        values = values[mask]
        add_points(plot, spec.name, coords, values, cmap, args.voxel_point_size)
        stats.update({"gene": gene, "method": spec.name, "mode": "volume", "plotted": int(mask.sum()), "seconds": round(time.time() - t0, 2)})
        rows.append(stats)
        del coords, values
        gc.collect()

    coords, values = load_ish_points(gene, args.ish_dir, args.ish_min_value, meta)
    add_points(plot, "Allen_ISH", coords, values, cmap, args.voxel_point_size)
    rows.append({"gene": gene, "method": "Allen_ISH", "mode": "volume", "plotted": int(coords.shape[0])})

    plot.camera = camera
    plot.grid_visible = False
    out = args.output_dir / f"{gene}.volume.html"
    out.write_text(plot.get_snapshot(), encoding="utf-8")
    print(f"[write] {out}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT / "han_3d_match_html")
    parser.add_argument("--han-raw", type=Path, default=Path("/home/share/huadjyin/home/zhoutao3/tracks/example_data/03_han_mouse_brain/processed"))
    parser.add_argument("--stereotrack", type=Path, default=OUT / "stereotrack_gene_inference_adatas")
    parser.add_argument("--spalp", type=Path, default=Path("/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/03_han_mouse_brain_mae/inference_adatas"))
    parser.add_argument("--ish-dir", type=Path, default=ROOT / "Imputation/00.ISH_Data/allen_3d_ish_data")
    parser.add_argument("--mesh-path", type=Path, default=ROOT / "out/03_han_mouse_brain_mae/average_template_10_sur.stl")
    parser.add_argument("--file-glob", default="*.h5ad")
    parser.add_argument("--coord-columns", nargs=3, default=["az", "ay", "ax"])
    parser.add_argument("--ccf-scale", type=float, default=100.0)
    parser.add_argument("--ish-scale", type=float, default=10.0)
    parser.add_argument("--gene-name-column", default=None)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--total-chunk-size", type=int, default=2048)
    parser.add_argument("--cell-min-value", type=float, default=0.0)
    parser.add_argument("--volume-min-value", type=float, default=0.0)
    parser.add_argument("--ish-min-value", type=float, default=0.0)
    parser.add_argument("--cell-point-size", type=float, default=200.0)
    parser.add_argument("--voxel-point-size", type=float, default=1000.0)
    parser.add_argument("--with-mesh", action="store_true")
    parser.add_argument("--mesh-opacity", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        MethodSpec("Han_lognorm", args.han_raw, "X", None, "log1p_scale"),
        MethodSpec("StereoTrack", args.stereotrack, "layer", "niche_recon", "none"),
        MethodSpec("SpaLP", args.spalp, "X", None, "none"),
    ]

    ish_map = ish_gene_map(args.ish_dir)
    missing = [gene for gene in args.genes if gene not in ish_map]
    if missing:
        raise FileNotFoundError(f"Genes without local ISH zip: {missing}")
    raw_meta = read_ish_meta(ish_map[args.genes[0]])
    meta = scale_ish_meta(raw_meta, args.ish_scale)
    print(f"[ISH grid raw] dims={raw_meta['dims']}, spacing={raw_meta['spacing']}, offset={raw_meta['offset']}")
    print(f"[ISH grid plot] dims={meta['dims']}, spacing={meta['spacing']}, offset={meta['offset']}, ish_scale={args.ish_scale}")
    print(f"[coords] columns={args.coord_columns}, ccf_scale={args.ccf_scale}")

    camera = [
        11485.284267462972,
        -14993.204389281993,
        127606.14254467492,
        69173.64700316277,
        57841.691822745124,
        43527.33431330532,
        0.21578089746207177,
        -0.8402988312227976,
        -0.4973293461440416,
    ]
    cmap = make_cmap()
    summary_rows: list[dict[str, object]] = []
    for gene in args.genes:
        print(f"\n{'=' * 80}\n[gene] {gene}\n{'=' * 80}")
        summary_rows.extend(write_cell_html(gene, methods, args, cmap, camera))
        summary_rows.extend(write_volume_html(gene, methods, args, cmap, camera, meta))

    if summary_rows:
        import pandas as pd

        summary = pd.DataFrame(summary_rows)
        summary_path = args.output_dir / "plot_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
