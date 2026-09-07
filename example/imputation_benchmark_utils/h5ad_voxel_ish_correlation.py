"""
Compute ISH correlation directly from h5ad files after voxel aggregation.

This avoids writing large cell-by-gene parquet files. Expression is read in
gene chunks, accumulated onto the Allen ISH voxel grid, and correlated with the
matching ISH volume.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import re
import time
import warnings
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)


def sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for pattern in (r"(?:^|_)T(\d+)(?:_|$)", r"(?:^|_)slice_(\d+)(?:_|$)", r"(\d+)"):
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1)), stem
    return 10**18, stem


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            gene = re.split(r"[\t,]", line, maxsplit=1)[0].strip()
            if gene.lower() in {"gene", "gene_name", "symbol", "name"}:
                continue
            if gene and gene not in seen:
                seen.add(gene)
                genes.append(gene)
    return genes


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


def read_var_names(handle: h5py.File, gene_name_column: str | None) -> list[str]:
    var_group = handle["var"]
    index_key = var_group.attrs.get("_index", "_index")
    index_key = clean_value(index_key)
    key = gene_name_column if gene_name_column and gene_name_column in var_group else index_key
    return [clean_value(item) for item in read_h5_array(var_group[key]).tolist()]


def read_coordinates(
    handle: h5py.File,
    ccf_key: str,
    coord_columns: list[str] | None,
    label: str,
) -> np.ndarray:
    if ccf_key and "obsm" in handle and ccf_key in handle["obsm"]:
        coords = np.asarray(read_h5_array(handle["obsm"][ccf_key]), dtype=np.float32)
    elif coord_columns:
        missing = [col for col in coord_columns if col not in handle["obs"]]
        if missing:
            raise KeyError(f"{label}: missing obs coordinate columns: {missing}")
        coords = np.column_stack([np.asarray(read_h5_array(handle["obs"][col]), dtype=np.float32) for col in coord_columns])
    else:
        raise KeyError(f"{label}: no coordinates found. Use --coord-columns or --ccf-key.")
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"{label}: coordinates must be an n_cells x 3 matrix")
    return coords[:, :3]


def matrix_from_group(group: h5py.Group):
    encoding = clean_value(group.attrs.get("encoding-type", ""))
    raw_shape = group["shape"][()] if "shape" in group else group.attrs["shape"]
    shape = tuple(int(x) for x in raw_shape)
    data = group["data"][()]
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    if encoding == "csc_matrix":
        return sparse.csc_matrix((data, indices, indptr), shape=shape)
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


class MatrixReader:
    def __init__(self, node):
        self.node = node
        self.matrix = None
        if isinstance(node, h5py.Dataset):
            self.kind = "dense"
            self.shape = node.shape
        else:
            self.kind = "sparse"
            self.matrix = matrix_from_group(node)
            self.shape = self.matrix.shape

    def read_cols(self, indices: list[int] | np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return np.zeros((int(self.shape[0]), 0), dtype=np.float32)
        if self.kind == "sparse":
            return self.matrix[:, indices].toarray().astype(np.float32, copy=False)

        order = np.argsort(indices)
        sorted_idx = indices[order]
        sub = np.asarray(self.node[:, sorted_idx], dtype=np.float32)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return sub[:, inv]

    def sum_cols_in_chunks(self, indices: list[int], chunk_size: int) -> np.ndarray:
        totals = np.zeros(int(self.shape[0]), dtype=np.float32)
        for start in range(0, len(indices), chunk_size):
            sub = self.read_cols(indices[start : start + chunk_size])
            totals += sub.sum(axis=1, dtype=np.float32)
            del sub
        return totals


def get_matrix_node(handle: h5py.File, source: str, layer: str | None):
    if source == "X":
        return handle["X"]
    if source == "layer":
        if not layer:
            raise ValueError("--layer is required when --source layer")
        if "layers" not in handle or layer not in handle["layers"]:
            available = list(handle["layers"].keys()) if "layers" in handle else []
            raise KeyError(f"layer {layer!r} not found. Available layers: {available}")
        return handle["layers"][layer]
    raise ValueError(source)


def read_ish_meta(zip_path: str, channel: str = "energy"):
    with zipfile.ZipFile(zip_path) as zf:
        mhd_text = zf.read(f"{channel}.mhd").decode("utf-8")
        meta = {}
        for line in mhd_text.strip().splitlines():
            key, _, val = line.partition("=")
            meta[key.strip()] = val.strip()
    return {
        "dims": tuple(map(int, meta["DimSize"].split())),
        "spacing": tuple(map(float, meta["ElementSpacing"].split())),
        "offset": tuple(map(float, meta["Offset"].split())),
        "dtype": meta["ElementType"],
    }


def load_ish_volume(zip_path: str, channel: str = "energy") -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        mhd_text = zf.read(f"{channel}.mhd").decode("utf-8")
        meta = {}
        for line in mhd_text.strip().splitlines():
            key, _, val = line.partition("=")
            meta[key.strip()] = val.strip()
        dims = list(map(int, meta["DimSize"].split()))
        dtype_map = {"MET_FLOAT": np.float32, "MET_UCHAR": np.uint8}
        raw = zf.read(meta["ElementDataFile"])
        return np.frombuffer(raw, dtype=dtype_map[meta["ElementType"]]).reshape(dims[2], dims[1], dims[0])


def ish_gene_map(ish_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in glob.glob(os.path.join(str(ish_dir), "*.zip")):
        stem = Path(path).stem
        if "_" not in stem:
            continue
        gene = stem.rsplit("_", 1)[0].rstrip("*")
        mapping.setdefault(gene, path)
    return mapping


def filter_genes_with_valid_ish(genes: list[str], ish_map: dict[str, str]) -> tuple[list[str], list[dict[str, object]]]:
    valid: list[str] = []
    bad_rows: list[dict[str, object]] = []
    for gene in genes:
        path = ish_map.get(gene)
        if path is None:
            bad_rows.append({"gene": gene, "zip_path": "", "reason": "missing_ish_zip", "size_bytes": 0})
            continue
        try:
            is_valid = zipfile.is_zipfile(path)
        except OSError as error:
            bad_rows.append({"gene": gene, "zip_path": path, "reason": f"os_error: {error}", "size_bytes": 0})
            continue
        if not is_valid:
            size = os.path.getsize(path) if os.path.exists(path) else 0
            bad_rows.append({"gene": gene, "zip_path": path, "reason": "bad_zip_file", "size_bytes": size})
            continue
        valid.append(gene)
    return valid, bad_rows


def voxel_flat_indices(coords: np.ndarray, meta: dict, ccf_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = meta["dims"]
    spacing = meta["spacing"]
    offset = meta["offset"]
    finite = np.isfinite(coords).all(axis=1)

    x_um = coords[:, 0] * ccf_scale
    y_um = coords[:, 1] * ccf_scale
    z_um = coords[:, 2] * ccf_scale

    vx = np.zeros(coords.shape[0], dtype=np.int32)
    vy = np.zeros(coords.shape[0], dtype=np.int32)
    vz = np.zeros(coords.shape[0], dtype=np.int32)
    vx[finite] = np.round((x_um[finite] - offset[0]) / spacing[0]).astype(np.int32)
    vy[finite] = np.round((y_um[finite] - offset[1]) / spacing[1]).astype(np.int32)
    vz[finite] = np.round((z_um[finite] - offset[2]) / spacing[2]).astype(np.int32)
    in_bounds = finite & (
        (vx >= 0) & (vx < dims[0]) &
        (vy >= 0) & (vy < dims[1]) &
        (vz >= 0) & (vz < dims[2])
    )
    flat = vz.astype(np.int64) * (dims[1] * dims[0]) + vy.astype(np.int64) * dims[0] + vx.astype(np.int64)
    return flat[in_bounds], in_bounds, finite


def compute_correlations(
    genes: list[str],
    sum_all: np.ndarray,
    sum_expr: np.ndarray,
    cnt_expr: np.ndarray,
    cnt_all: np.ndarray,
    unique_voxels: np.ndarray,
    meta: dict,
    ish_map: dict[str, str],
) -> list[dict[str, object]]:
    dims = meta["dims"]
    uv_vz = unique_voxels // (dims[1] * dims[0])
    uv_vy = (unique_voxels % (dims[1] * dims[0])) // dims[0]
    uv_vx = unique_voxels % dims[0]
    cnt = cnt_all[unique_voxels].astype(np.float32)

    results: list[dict[str, object]] = []
    for j, gene in enumerate(genes):
        ish_zip = ish_map.get(gene)
        if ish_zip is None:
            continue

        mean_all = sum_all[unique_voxels, j] / cnt
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_expr = np.where(cnt_expr[unique_voxels, j] > 0, sum_expr[unique_voxels, j] / cnt_expr[unique_voxels, j], 0.0)
        frac_expr = cnt_expr[unique_voxels, j] / cnt

        try:
            vol = load_ish_volume(ish_zip)
        except (zipfile.BadZipFile, KeyError, ValueError, OSError) as error:
            print(f"[warn] skip bad ISH: gene={gene}, zip={ish_zip}, error={error}")
            continue
        ish_vals = vol[uv_vz, uv_vy, uv_vx].astype(np.float64)
        del vol

        result: dict[str, object] = {"gene": gene, "n_voxels_covered": int(len(unique_voxels))}
        for agg_name, st_vals in {
            "mean_all": mean_all,
            "mean_expr": mean_expr,
            "frac_expr": frac_expr,
        }.items():
            if agg_name == "frac_expr":
                valid = (ish_vals > 0) & np.isfinite(st_vals)
            else:
                valid = (ish_vals > 0) & (st_vals > 0) & np.isfinite(st_vals)
            n_valid = int(valid.sum())
            result[f"n_valid_{agg_name}"] = n_valid
            if n_valid < 10:
                result[f"pearson_{agg_name}"] = np.nan
                result[f"pearson_p_{agg_name}"] = np.nan
                result[f"spearman_{agg_name}"] = np.nan
                result[f"spearman_p_{agg_name}"] = np.nan
                continue
            x = st_vals[valid]
            y = ish_vals[valid]
            r, p = stats.pearsonr(x, y)
            rho, p_s = stats.spearmanr(x, y)
            result[f"pearson_{agg_name}"] = r
            result[f"pearson_p_{agg_name}"] = p
            result[f"spearman_{agg_name}"] = rho
            result[f"spearman_p_{agg_name}"] = p_s
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--file-glob", default="*.h5ad")
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--ish-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["X", "layer"], default="X")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--normalize", choices=["none", "log1p_norm"], default="none")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--ccf-key", default="ccf")
    parser.add_argument("--coord-columns", nargs=3, default=None)
    parser.add_argument("--gene-name-column", default=None)
    parser.add_argument("--ccf-scale", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bad-ish-log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5ads = sorted(args.input_dir.glob(args.file_glob), key=sort_key)
    if not h5ads:
        raise FileNotFoundError(f"No {args.file_glob} files found in {args.input_dir}")
    requested_genes = read_gene_list(args.gene_list)
    ish_map = ish_gene_map(args.ish_dir)
    genes, bad_ish_rows = filter_genes_with_valid_ish(requested_genes, ish_map)
    bad_ish_log = args.bad_ish_log or args.output.with_suffix(".bad_ish.csv")
    if bad_ish_rows:
        pd.DataFrame(bad_ish_rows).to_csv(bad_ish_log, index=False)
        print(f"[warn] excluded {len(bad_ish_rows)} genes with missing/bad ISH -> {bad_ish_log}")
    if not genes:
        raise ValueError("No requested genes have a readable ISH zip")
    ref_ish = ish_map[genes[0]]
    meta = read_ish_meta(ref_ish)
    dims = meta["dims"]
    n_flat = dims[0] * dims[1] * dims[2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[input] {len(h5ads)} h5ad files")
    print(f"[genes] {len(requested_genes)} requested, {len(genes)} readable, {len(ish_map)} local ISH genes")
    print(f"[grid] dims={dims}, n_flat={n_flat}, chunk_size={args.chunk_size}")
    print(f"[matrix] source={args.source}, layer={args.layer}, normalize={args.normalize}")

    all_results: list[dict[str, object]] = []
    n_cells_total = 0
    n_finite_total = 0
    n_in_bounds_total = 0
    totals_cache: dict[Path, np.ndarray] = {}
    t0 = time.time()
    for chunk_start in range(0, len(genes), args.chunk_size):
        chunk_genes = genes[chunk_start : chunk_start + args.chunk_size]
        n_chunk = len(chunk_genes)
        sum_all = np.zeros((n_flat, n_chunk), dtype=np.float32)
        sum_expr = np.zeros((n_flat, n_chunk), dtype=np.float32)
        cnt_expr = np.zeros((n_flat, n_chunk), dtype=np.float32)
        cnt_all = np.zeros(n_flat, dtype=np.int32)

        for file_i, h5ad_path in enumerate(h5ads, start=1):
            with h5py.File(h5ad_path, "r") as handle:
                var_names = read_var_names(handle, args.gene_name_column)
                var_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
                matrix = MatrixReader(get_matrix_node(handle, args.source, args.layer))
                coords = read_coordinates(handle, args.ccf_key, args.coord_columns, h5ad_path.name)
                flat_idx, in_bounds, finite = voxel_flat_indices(coords, meta, args.ccf_scale)
                if chunk_start == 0:
                    n_cells_total += int(coords.shape[0])
                    n_finite_total += int(finite.sum())
                    n_in_bounds_total += int(in_bounds.sum())
                if flat_idx.size == 0:
                    continue

                present_all = [var_to_idx[gene] for gene in genes if gene in var_to_idx]
                totals = None
                if args.normalize == "log1p_norm":
                    totals = totals_cache.get(h5ad_path)
                    if totals is None:
                        totals = matrix.sum_cols_in_chunks(present_all, args.chunk_size)
                        totals_cache[h5ad_path] = totals

                values = np.zeros((coords.shape[0], n_chunk), dtype=np.float32)
                present_pos = [j for j, gene in enumerate(chunk_genes) if gene in var_to_idx]
                if present_pos:
                    present_idx = [var_to_idx[chunk_genes[j]] for j in present_pos]
                    values[:, present_pos] = matrix.read_cols(present_idx)
                if args.normalize == "log1p_norm":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values = np.divide(values, totals[:, None], out=np.zeros_like(values), where=totals[:, None] > 0)
                    values *= float(args.target_sum)
                    np.log1p(values, out=values)

                vals = values[in_bounds]
                np.add.at(sum_all, flat_idx, vals)
                np.add.at(cnt_expr, flat_idx, (vals > 0).astype(np.float32))
                np.add.at(sum_expr, flat_idx, vals * (vals > 0))
                cnt_all += np.bincount(flat_idx, minlength=n_flat).astype(np.int32)

                del matrix, coords, values, vals, totals
                gc.collect()

            if file_i % args.progress_every == 0 or file_i == len(h5ads):
                print(f"  chunk {chunk_start + 1}-{chunk_start + n_chunk}: file {file_i}/{len(h5ads)}")

        unique_voxels = np.flatnonzero(cnt_all > 0).astype(np.int64)
        if chunk_start == 0 and n_cells_total:
            print(
                f"[coords] cells={n_cells_total:,}, finite={n_finite_total:,} "
                f"({n_finite_total / n_cells_total * 100:.2f}%), "
                f"in_bounds={n_in_bounds_total:,} ({n_in_bounds_total / n_cells_total * 100:.2f}%)"
            )
        chunk_results = compute_correlations(chunk_genes, sum_all, sum_expr, cnt_expr, cnt_all, unique_voxels, meta, ish_map)
        all_results.extend(chunk_results)
        elapsed = time.time() - t0
        print(f"[chunk] {chunk_start + 1}-{chunk_start + n_chunk}: {len(chunk_results)} genes, elapsed={elapsed:.1f}s")

        del sum_all, sum_expr, cnt_expr, cnt_all
        gc.collect()

    df = pd.DataFrame(all_results)
    col_order = ["gene", "n_voxels_covered"]
    for agg_name in ["mean_all", "mean_expr", "frac_expr"]:
        col_order.extend([
            f"pearson_{agg_name}", f"spearman_{agg_name}",
            f"n_valid_{agg_name}",
            f"pearson_p_{agg_name}", f"spearman_p_{agg_name}",
        ])
    df = df[[col for col in col_order if col in df.columns]]
    df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"[done] {len(df)} genes -> {args.output}")


if __name__ == "__main__":
    main()
