from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import pickle
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


@contextmanager
def atomic_file(path: Path, mode: str):
    """Publish a complete file only after its contents have reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_pickle(path: Path, value) -> None:
    with atomic_file(path, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_aggregate_progress(args: argparse.Namespace, n_genes: int):
    path = args.aggregate_checkpoint
    if path.exists():
        with path.open("rb") as handle:
            completed, state, n_cells, n_finite = pickle.load(handle)
        print(f"    [resume] {path.name}: {completed} files already aggregated", flush=True)
        return completed, state, n_cells, n_finite
    return 0, empty_state(n_genes), 0, 0


def save_aggregate_progress(args, completed, total, state, n_cells, n_finite):
    if completed % args.checkpoint_every == 0 or completed == total:
        save_pickle(args.aggregate_checkpoint, (completed, state, n_cells, n_finite))
        print(f"    [checkpoint] {completed}/{total} files saved", flush=True)


def run_manifest(args, meta, genes):
    """Reject stale checkpoints when settings or source files have changed."""
    paths = [args.model_dir / "cache" / "meta.pkl"]
    for info in meta["slice_info"]:
        paths.append(args.model_dir / "cache" / info["file"])
        stem = Path(info["batch"]).stem
        if "StereoTrack" in args.methods:
            paths.append(args.stereotrack_dir / f"{stem}.expression.{args.stereotrack_dtype}.npy")
        if "SpaLP" in args.methods:
            paths.append(args.spalp_dir / f"{stem}.h5ad")
    inputs = []
    for path in paths:
        stat = path.stat()
        inputs.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    return {
        "version": 1,
        "settings": {key: getattr(args, key) for key in (
            "methods", "stereotrack_dtype", "voxel_size", "voxel_method",
            "positive_filter", "chunk_size", "ccf_key", "ccf_scale",
        )},
        "genes": list(map(str, genes)),
        "inputs": inputs,
    }


@dataclass
class VoxelAggregate:
    keys: np.ndarray
    index: dict[tuple[int, int, int], int]
    mean_all: np.ndarray
    mean_expr: np.ndarray
    frac_expr: np.ndarray
    n_cells_total: int
    n_finite_total: int


def sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for pattern in (r"(?:^|_)T(\d+)(?:_|\.|$)", r"(?:^|_)slice_(\d+)(?:_|\.|$)", r"(\d+)"):
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1)), stem
    return 10**18, stem


def clean_value(raw: object) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode()
    return str(raw)


def read_h5_array(node):
    if isinstance(node, h5py.Dataset):
        return node[()]
    if isinstance(node, h5py.Group) and {"codes", "categories"}.issubset(node.keys()):
        codes = node["codes"][()]
        categories = read_h5_array(node["categories"])
        return np.asarray([categories[int(code)] if int(code) >= 0 else "" for code in codes])
    raise TypeError(f"Cannot read h5ad array from {node.name}")


def read_var_names(handle: h5py.File) -> list[str]:
    var_group = handle["var"]
    index_key = clean_value(var_group.attrs.get("_index", "_index"))
    return [clean_value(item) for item in read_h5_array(var_group[index_key]).tolist()]


def read_obsm_3d(handle: h5py.File, key: str) -> np.ndarray:
    if "obsm" not in handle or key not in handle["obsm"]:
        raise KeyError(f"Missing obsm[{key!r}] in {handle.filename}")
    coords = np.asarray(read_h5_array(handle["obsm"][key]), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"{handle.filename}: obsm[{key!r}] must have at least 3 columns")
    return coords[:, :3]


def h5_sparse_matrix(group: h5py.Group):
    shape_raw = group["shape"][()] if "shape" in group else group.attrs["shape"]
    shape = tuple(int(x) for x in shape_raw)
    data = group["data"][()]
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    encoding = clean_value(group.attrs.get("encoding-type", "csr_matrix"))
    if encoding == "csc_matrix":
        return sp.csc_matrix((data, indices, indptr), shape=shape)
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def h5_read_cols(handle: h5py.File, gene_indices: list[int]) -> np.ndarray:
    node = handle["X"]
    indices = np.asarray(gene_indices, dtype=np.int64)
    if indices.size == 0:
        return np.zeros((int(node.shape[0]), 0), dtype=np.float32)
    if isinstance(node, h5py.Group):
        return h5_sparse_matrix(node)[:, indices].toarray().astype(np.float32, copy=False)

    order = np.argsort(indices)
    sorted_idx = indices[order]
    sub = np.asarray(node[:, sorted_idx], dtype=np.float32)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return sub[:, inv]


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("rb") as handle:
        return pickle.load(handle)


def voxelize(coords: np.ndarray, scale: float, voxel_size: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(coords).all(axis=1)
    scaled = coords[:, :3].astype(np.float64, copy=False) * float(scale)
    if method == "round":
        voxels = np.rint(scaled / voxel_size).astype(np.int64)
    elif method == "floor":
        voxels = np.floor(scaled / voxel_size).astype(np.int64)
    else:
        raise ValueError(method)
    return voxels[finite], finite


def append_rows(
    voxel_index: dict[tuple[int, int, int], int],
    voxel_keys: list[tuple[int, int, int]],
    sum_all: np.ndarray,
    sum_expr: np.ndarray,
    cnt_expr: np.ndarray,
    cnt_all: np.ndarray,
    local_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.empty(local_keys.shape[0], dtype=np.int64)
    n_new = 0
    for i, raw_key in enumerate(local_keys):
        key = (int(raw_key[0]), int(raw_key[1]), int(raw_key[2]))
        idx = voxel_index.get(key)
        if idx is None:
            idx = len(voxel_keys)
            voxel_index[key] = idx
            voxel_keys.append(key)
            n_new += 1
        rows[i] = idx

    if n_new:
        n_genes = sum_all.shape[1]
        sum_all = np.vstack([sum_all, np.zeros((n_new, n_genes), dtype=np.float32)])
        sum_expr = np.vstack([sum_expr, np.zeros((n_new, n_genes), dtype=np.float32)])
        cnt_expr = np.vstack([cnt_expr, np.zeros((n_new, n_genes), dtype=np.float32)])
        cnt_all = np.concatenate([cnt_all, np.zeros(n_new, dtype=np.int64)])
    return rows, sum_all, sum_expr, cnt_expr, cnt_all


def add_values_to_voxels(
    values: np.ndarray,
    coords: np.ndarray,
    aggregate_state: tuple[dict[tuple[int, int, int], int], list[tuple[int, int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, int, int], int], list[tuple[int, int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    voxel_index, voxel_keys, sum_all, sum_expr, cnt_expr, cnt_all = aggregate_state
    voxels, finite = voxelize(coords, args.ccf_scale, args.voxel_size, args.voxel_method)
    if voxels.size == 0:
        return voxel_index, voxel_keys, sum_all, sum_expr, cnt_expr, cnt_all, int(coords.shape[0]), int(finite.sum())

    vals = values[finite]
    local_keys, inverse = np.unique(voxels, axis=0, return_inverse=True)
    rows, sum_all, sum_expr, cnt_expr, cnt_all = append_rows(
        voxel_index, voxel_keys, sum_all, sum_expr, cnt_expr, cnt_all, local_keys
    )
    n_local = local_keys.shape[0]
    n_genes = values.shape[1]
    local_sum_all = np.zeros((n_local, n_genes), dtype=np.float32)
    local_sum_expr = np.zeros((n_local, n_genes), dtype=np.float32)
    local_cnt_expr = np.zeros((n_local, n_genes), dtype=np.float32)
    local_cnt_all = np.bincount(inverse, minlength=n_local).astype(np.int64)
    positive = vals > 0
    np.add.at(local_sum_all, inverse, vals)
    np.add.at(local_sum_expr, inverse, vals * positive)
    np.add.at(local_cnt_expr, inverse, positive.astype(np.float32))
    sum_all[rows] += local_sum_all
    sum_expr[rows] += local_sum_expr
    cnt_expr[rows] += local_cnt_expr
    cnt_all[rows] += local_cnt_all
    return voxel_index, voxel_keys, sum_all, sum_expr, cnt_expr, cnt_all, int(coords.shape[0]), int(finite.sum())


def finish_aggregate(
    voxel_index: dict[tuple[int, int, int], int],
    voxel_keys: list[tuple[int, int, int]],
    sum_all: np.ndarray,
    sum_expr: np.ndarray,
    cnt_expr: np.ndarray,
    cnt_all: np.ndarray,
    n_cells_total: int,
    n_finite_total: int,
) -> VoxelAggregate:
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_all = sum_all / cnt_all[:, None]
        mean_expr = np.divide(sum_expr, cnt_expr, out=np.zeros_like(sum_expr), where=cnt_expr > 0)
        frac_expr = cnt_expr / cnt_all[:, None]
    return VoxelAggregate(
        keys=np.asarray(voxel_keys, dtype=np.int64),
        index=voxel_index,
        mean_all=mean_all.astype(np.float32, copy=False),
        mean_expr=mean_expr.astype(np.float32, copy=False),
        frac_expr=frac_expr.astype(np.float32, copy=False),
        n_cells_total=n_cells_total,
        n_finite_total=n_finite_total,
    )


def empty_state(n_genes: int):
    return (
        {},
        [],
        np.zeros((0, n_genes), dtype=np.float32),
        np.zeros((0, n_genes), dtype=np.float32),
        np.zeros((0, n_genes), dtype=np.float32),
        np.zeros(0, dtype=np.int64),
    )


def aggregate_reference(meta: dict, cache_dir: Path, gene_indices: list[int], args: argparse.Namespace) -> VoxelAggregate:
    completed, state, n_cells_total, n_finite_total = load_aggregate_progress(args, len(gene_indices))
    for i, info in enumerate(meta["slice_info"], start=1):
        if i <= completed:
            continue
        path = cache_dir / info["file"]
        with np.load(path) as data:
            feat = sp.csr_matrix(
                (
                    data["feat_data"].astype(np.float32),
                    data["feat_indices"].astype(np.int64),
                    data["feat_indptr"].astype(np.int64),
                ),
                shape=tuple(data["feat_shape"]),
            )
            values = feat[:, gene_indices].toarray().astype(np.float32, copy=False)
            coords = np.asarray(data["coords"], dtype=np.float32)
        *state, n_cells, n_finite = add_values_to_voxels(values, coords, state, args)
        n_cells_total += n_cells
        n_finite_total += n_finite
        save_aggregate_progress(args, i, len(meta["slice_info"]), state, n_cells_total, n_finite_total)
        if i % args.progress_every == 0 or i == len(meta["slice_info"]):
            print(f"    reference: file {i}/{len(meta['slice_info'])}, voxels={len(state[1]):,}")
        del feat, values, coords
        gc.collect()
    return finish_aggregate(*state, n_cells_total, n_finite_total)


def aggregate_stereotrack(meta: dict, cache_dir: Path, infer_dir: Path, gene_indices: list[int], args: argparse.Namespace) -> VoxelAggregate:
    completed, state, n_cells_total, n_finite_total = load_aggregate_progress(args, len(gene_indices))
    suffix = f".expression.{args.stereotrack_dtype}.npy"
    for i, info in enumerate(meta["slice_info"], start=1):
        if i <= completed:
            continue
        stem = Path(info["batch"]).stem
        x_path = infer_dir / f"{stem}{suffix}"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing StereoTrack inference file: {x_path}")
        with np.load(cache_dir / info["file"]) as data:
            coords = np.asarray(data["coords"], dtype=np.float32)
        matrix = np.load(x_path, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(f"{x_path}: expected a 2D expression matrix, got shape={matrix.shape}")
        expected_genes = int(meta.get("input_dim", len(meta.get("common_genes", []))))
        if matrix.shape[1] != expected_genes:
            raise ValueError(
                f"{x_path}: expected {expected_genes} expression genes, got {matrix.shape[1]}. "
                "This looks like an embedding npy; use imputation_benchmark/stereotrack_expression_npy, "
                "not niche_embedding_by_sample."
            )
        values = np.asarray(matrix[:, gene_indices], dtype=np.float32)
        if values.shape[0] != coords.shape[0]:
            raise ValueError(f"{x_path}: n_cells={values.shape[0]} does not match coords={coords.shape[0]}")
        *state, n_cells, n_finite = add_values_to_voxels(values, coords, state, args)
        n_cells_total += n_cells
        n_finite_total += n_finite
        save_aggregate_progress(args, i, len(meta["slice_info"]), state, n_cells_total, n_finite_total)
        if i % args.progress_every == 0 or i == len(meta["slice_info"]):
            print(f"    StereoTrack: file {i}/{len(meta['slice_info'])}, voxels={len(state[1]):,}")
        del matrix, values, coords
        gc.collect()
    return finish_aggregate(*state, n_cells_total, n_finite_total)


def aggregate_spalp(meta: dict, spalp_dir: Path, genes: list[str], chunk_genes: list[str], args: argparse.Namespace) -> VoxelAggregate:
    completed, state, n_cells_total, n_finite_total = load_aggregate_progress(args, len(chunk_genes))
    for i, info in enumerate(meta["slice_info"], start=1):
        if i <= completed:
            continue
        stem = Path(info["batch"]).stem
        h5ad_path = spalp_dir / f"{stem}.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Missing SpaLP inference file: {h5ad_path}")
        with h5py.File(h5ad_path, "r") as handle:
            var_names = read_var_names(handle)
            var_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
            present_idx = [var_to_idx[gene] for gene in chunk_genes]
            values = h5_read_cols(handle, present_idx)
            coords = read_obsm_3d(handle, args.ccf_key)
        *state, n_cells, n_finite = add_values_to_voxels(values, coords, state, args)
        n_cells_total += n_cells
        n_finite_total += n_finite
        save_aggregate_progress(args, i, len(meta["slice_info"]), state, n_cells_total, n_finite_total)
        if i % args.progress_every == 0 or i == len(meta["slice_info"]):
            print(f"    SpaLP: file {i}/{len(meta['slice_info'])}, voxels={len(state[1]):,}")
        del values, coords
        gc.collect()
    return finish_aggregate(*state, n_cells_total, n_finite_total)


def shared_indices(a: VoxelAggregate, b: VoxelAggregate) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(set(a.index).intersection(b.index))
    idx_a = np.asarray([a.index[key] for key in ordered], dtype=np.int64)
    idx_b = np.asarray([b.index[key] for key in ordered], dtype=np.int64)
    return idx_a, idx_b


def pearson_pair(a: np.ndarray, b: np.ndarray, positive_filter: str) -> tuple[float, int]:
    valid = np.isfinite(a) & np.isfinite(b)
    if positive_filter == "both":
        valid &= (a > 0) & (b > 0)
    elif positive_filter == "either":
        valid &= (a > 0) | (b > 0)
    elif positive_filter != "none":
        raise ValueError(positive_filter)

    a = a[valid]
    b = b[valid]
    n_valid = int(a.size)
    if n_valid < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan, n_valid
    return float(np.corrcoef(a, b)[0, 1]), n_valid


def compare_to_reference(genes: list[str], reference: VoxelAggregate, method_name: str, method: VoxelAggregate, args: argparse.Namespace):
    idx_ref, idx_method = shared_indices(reference, method)
    print(f"  [pair] Macaque_lognormscale vs {method_name}: shared_voxels={idx_ref.size:,}")
    rows = []
    for j, gene in enumerate(genes):
        row = {
            "reference": "Macaque_lognormscale",
            "method": method_name,
            "gene": gene,
            "n_voxels_shared": int(idx_ref.size),
        }
        for agg_name in ["mean_all", "mean_expr", "frac_expr"]:
            ref_values = getattr(reference, agg_name)[idx_ref, j]
            method_values = getattr(method, agg_name)[idx_method, j]
            pearson, n_valid = pearson_pair(ref_values, method_values, args.positive_filter)
            row[f"pearson_{agg_name}"] = pearson
            row[f"n_valid_{agg_name}"] = n_valid
        rows.append(row)
    return rows


def write_summary(df: pd.DataFrame, output: Path, summary: Path) -> None:
    rows = []
    for agg_name in ["mean_all", "mean_expr", "frac_expr"]:
        col = f"pearson_{agg_name}"
        grouped = df.dropna(subset=[col]).groupby("method", as_index=False)
        for _, item in grouped.agg(n_genes=("gene", "count"), median=(col, "median"), mean=(col, "mean")).iterrows():
            rows.append(
                {
                    "method": item["method"],
                    "metric": col,
                    "n_genes": int(item["n_genes"]),
                    "median": float(item["median"]),
                    "mean": float(item["mean"]),
                }
            )
    summary_df = pd.DataFrame(rows, columns=["method", "metric", "n_genes", "median", "mean"])
    with atomic_file(summary, "w") as handle:
        summary_df.to_csv(handle, index=False)
    print(summary_df.to_string(index=False))
    print(f"[output] {output}")
    print(f"[summary] {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Macaque v1 method-to-lognormscale voxel correlation.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stereotrack-dir", type=Path, required=True)
    parser.add_argument("--spalp-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["StereoTrack", "SpaLP"], default=["StereoTrack", "SpaLP"])
    parser.add_argument("--stereotrack-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--voxel-size", type=float, default=200.0)
    parser.add_argument("--voxel-method", choices=["floor", "round"], default="floor")
    parser.add_argument("--positive-filter", choices=["none", "either", "both"], default="none")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--ccf-key", default="ccf")
    parser.add_argument("--ccf-scale", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save aggregation state every N input files (default: 20); resume is automatic.")
    args = parser.parse_args()
    for key in ("chunk_size", "progress_every", "checkpoint_every", "voxel_size"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    args.methods = list(dict.fromkeys(args.methods))
    if args.output.resolve() == args.summary.resolve():
        parser.error("--output and --summary must be different paths")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(str(args.output) + ".checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another benchmark is using {checkpoint_dir}; wait for it to stop.")
        run_benchmark(args, checkpoint_dir)


def publish_results(all_rows, args):
    df = pd.DataFrame(all_rows)
    with atomic_file(args.output, "w") as handle:
        df.to_csv(handle, index=False, float_format="%.6f")
    write_summary(df, args.output, args.summary)
    print(f"[saved] {len(df):,} method/gene results", flush=True)


def run_benchmark(args, checkpoint_dir: Path) -> None:
    cache_dir = args.model_dir / "cache"
    meta = load_meta(cache_dir / "meta.pkl")
    genes = list(meta["common_genes"])
    if args.max_genes > 0:
        genes = genes[: args.max_genes]
    if not genes:
        raise ValueError("No genes to benchmark")
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = run_manifest(args, meta, genes)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit("Inputs or benchmark settings differ from the saved run. "
                             "Restore the original settings to resume, or use a new OUTPUT_PREFIX / --output.")
    else:
        if args.output.exists() or args.summary.exists():
            raise SystemExit("Existing CSV has no checkpoint metadata and cannot be safely resumed. "
                             "Use a new OUTPUT_PREFIX / --output to preserve it and start a new run.")
        with atomic_file(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "macaque_v1_common_genes_lognormscale.txt").write_text("\n".join(map(str, genes)) + "\n")

    print(f"[genes] {len(genes):,}")
    print(f"[voxel] size={args.voxel_size}, method={args.voxel_method}, positive_filter={args.positive_filter}")
    print(f"[methods] {args.methods}")
    print(f"[checkpoint] {checkpoint_dir}; aggregation saved every {args.checkpoint_every} files")

    all_rows = []
    completed_results = set()
    for start in range(0, len(genes), args.chunk_size):
        for method in args.methods:
            result_path = checkpoint_dir / f"chunk_{start:06d}_{method}.results.pkl"
            if result_path.exists():
                with result_path.open("rb") as handle:
                    all_rows.extend(pickle.load(handle))
                completed_results.add((start, method))
    if all_rows:
        print(f"[resume] restored {len(all_rows):,} method/gene results")
        publish_results(all_rows, args)

    for start in range(0, len(genes), args.chunk_size):
        pending = [method for method in args.methods if (start, method) not in completed_results]
        if not pending:
            print(f"[skip] chunk {start + 1}-{min(start + args.chunk_size, len(genes))}: all methods saved")
            # Also finish cleanup if interrupted after publishing the last method.
            for path in checkpoint_dir.glob(f"chunk_{start:06d}_*.aggregate.pkl"):
                path.unlink()
            continue
        chunk_genes = genes[start : start + args.chunk_size]
        chunk_indices = list(range(start, start + len(chunk_genes)))
        print(f"[chunk] {start + 1}-{start + len(chunk_genes)}")

        print("  [aggregate] Macaque_lognormscale from cache npz")
        args.aggregate_checkpoint = checkpoint_dir / f"chunk_{start:06d}_reference.aggregate.pkl"
        reference = aggregate_reference(meta, cache_dir, chunk_indices, args)

        for method in pending:
            args.aggregate_checkpoint = checkpoint_dir / f"chunk_{start:06d}_{method}.aggregate.pkl"
            print(f"  [aggregate] {method}")
            if method == "StereoTrack":
                aggregate = aggregate_stereotrack(meta, cache_dir, args.stereotrack_dir, chunk_indices, args)
            else:
                aggregate = aggregate_spalp(meta, args.spalp_dir, genes, chunk_genes, args)
            rows = compare_to_reference(chunk_genes, reference, method, aggregate, args)
            save_pickle(checkpoint_dir / f"chunk_{start:06d}_{method}.results.pkl", rows)
            all_rows.extend(rows)
            publish_results(all_rows, args)
            args.aggregate_checkpoint.unlink(missing_ok=True)
            del aggregate
            gc.collect()

        del reference
        for path in checkpoint_dir.glob(f"chunk_{start:06d}_*.aggregate.pkl"):
            path.unlink()
        gc.collect()

    print(f"[done] {len(all_rows):,}/{len(genes) * len(args.methods):,} method/gene results saved")


if __name__ == "__main__":
    main()
