"""
Compare h5ad-based imputation methods after direct voxel aggregation.

This avoids writing large cell-by-gene parquet files. Each method is read in
gene chunks, aggregated to spatial voxels, and compared gene-wise on shared
voxels with Pearson/Spearman correlation.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from h5ad_voxel_ish_correlation import (  # noqa: E402
    MatrixReader,
    get_matrix_node,
    read_coordinates,
    read_gene_list,
    read_var_names,
    sort_key,
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    input_dir: Path
    file_glob: str
    source: str
    layer: str | None
    normalize: str
    ccf_key: str
    coord_columns: list[str] | None
    gene_name_column: str | None
    ccf_scale: float


@dataclass
class VoxelAggregate:
    keys: np.ndarray
    index: dict[tuple[int, int, int], int]
    mean_all: np.ndarray
    mean_expr: np.ndarray
    frac_expr: np.ndarray
    n_cells_total: int
    n_finite_total: int
    n_voxel_cells_total: int


def none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value)
    if value in {"", "-", "None", "none", "NULL", "null"}:
        return None
    return value


def parse_coord_columns(value: str | None) -> list[str] | None:
    value = none_if_blank(value)
    if value is None:
        return None
    cols = [part.strip() for part in value.split(",") if part.strip()]
    if len(cols) != 3:
        raise ValueError(f"--coord-columns entries must be '-' or 'x,y,z', got {value!r}")
    return cols


def validate_parallel_args(args: argparse.Namespace) -> list[MethodSpec]:
    fields = [
        args.name,
        args.input_dir,
        args.file_glob,
        args.source,
        args.layer,
        args.normalize,
        args.ccf_key,
        args.coord_columns,
        args.gene_name_column,
        args.ccf_scale,
    ]
    lengths = {len(field) for field in fields}
    if len(lengths) != 1:
        raise ValueError(
            "Method options must be repeated the same number of times: "
            "--name/--input-dir/--file-glob/--source/--layer/--normalize/"
            "--ccf-key/--coord-columns/--gene-name-column/--ccf-scale"
        )
    specs: list[MethodSpec] = []
    for i in range(len(args.name)):
        specs.append(
            MethodSpec(
                name=args.name[i],
                input_dir=args.input_dir[i],
                file_glob=args.file_glob[i],
                source=args.source[i],
                layer=none_if_blank(args.layer[i]),
                normalize=args.normalize[i],
                ccf_key=none_if_blank(args.ccf_key[i]) or "",
                coord_columns=parse_coord_columns(args.coord_columns[i]),
                gene_name_column=none_if_blank(args.gene_name_column[i]),
                ccf_scale=float(args.ccf_scale[i]),
            )
        )
    return specs


def voxelize(coords: np.ndarray, ccf_scale: float, voxel_size: float, voxel_method: str) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(coords).all(axis=1)
    scaled = coords[:, :3].astype(np.float64, copy=False) * float(ccf_scale)
    if voxel_method == "round":
        voxels = np.rint(scaled / float(voxel_size)).astype(np.int64)
    elif voxel_method == "floor":
        voxels = np.floor(scaled / float(voxel_size)).astype(np.int64)
    else:
        raise ValueError(voxel_method)
    return voxels[finite], finite


def corr_pair(a: np.ndarray, b: np.ndarray, method: str, positive_filter: str) -> tuple[float, int]:
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
    if method == "spearman":
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(a, b)[0, 1]), n_valid


def read_gene_indices(handle: h5py.File, spec: MethodSpec) -> tuple[list[str], dict[str, int]]:
    var_names = read_var_names(handle, spec.gene_name_column)
    return var_names, {gene: idx for idx, gene in enumerate(var_names)}


def normalization_indices(var_names: list[str], var_to_idx: dict[str, int], genes: list[str], scope: str) -> list[int]:
    if scope == "all":
        return list(range(len(var_names)))
    if scope == "requested":
        return [var_to_idx[gene] for gene in genes if gene in var_to_idx]
    raise ValueError(scope)


def append_global_rows(
    voxel_index: dict[tuple[int, int, int], int],
    voxel_keys: list[tuple[int, int, int]],
    sum_all: np.ndarray,
    sum_expr: np.ndarray,
    cnt_expr: np.ndarray,
    cnt_all: np.ndarray,
    local_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.empty(local_keys.shape[0], dtype=np.int64)
    new_rows: list[tuple[int, int, int]] = []
    for i, raw_key in enumerate(local_keys):
        key = (int(raw_key[0]), int(raw_key[1]), int(raw_key[2]))
        idx = voxel_index.get(key)
        if idx is None:
            idx = len(voxel_keys)
            voxel_index[key] = idx
            voxel_keys.append(key)
            new_rows.append(key)
        rows[i] = idx

    if new_rows:
        n_new = len(new_rows)
        n_genes = sum_all.shape[1]
        sum_all = np.vstack([sum_all, np.zeros((n_new, n_genes), dtype=np.float32)])
        sum_expr = np.vstack([sum_expr, np.zeros((n_new, n_genes), dtype=np.float32)])
        cnt_expr = np.vstack([cnt_expr, np.zeros((n_new, n_genes), dtype=np.float32)])
        cnt_all = np.concatenate([cnt_all, np.zeros(n_new, dtype=np.int64)])
    return rows, sum_all, sum_expr, cnt_expr, cnt_all


def aggregate_method(
    spec: MethodSpec,
    genes: list[str],
    norm_genes: list[str],
    args: argparse.Namespace,
    totals_cache: dict[Path, np.ndarray],
) -> VoxelAggregate:
    h5ads = sorted(spec.input_dir.glob(spec.file_glob), key=sort_key)
    if not h5ads:
        raise FileNotFoundError(f"{spec.name}: no {spec.file_glob} files found in {spec.input_dir}")

    voxel_index: dict[tuple[int, int, int], int] = {}
    voxel_keys: list[tuple[int, int, int]] = []
    n_genes = len(genes)
    sum_all = np.zeros((0, n_genes), dtype=np.float32)
    sum_expr = np.zeros((0, n_genes), dtype=np.float32)
    cnt_expr = np.zeros((0, n_genes), dtype=np.float32)
    cnt_all = np.zeros(0, dtype=np.int64)

    n_cells_total = 0
    n_finite_total = 0
    n_voxel_cells_total = 0
    for file_i, h5ad_path in enumerate(h5ads, start=1):
        with h5py.File(h5ad_path, "r") as handle:
            var_names, var_to_idx = read_gene_indices(handle, spec)
            matrix = MatrixReader(get_matrix_node(handle, spec.source, spec.layer))
            coords = read_coordinates(handle, spec.ccf_key, spec.coord_columns, h5ad_path.name)
            voxels, finite = voxelize(coords, spec.ccf_scale, args.voxel_size, args.voxel_method)
            n_cells_total += int(coords.shape[0])
            n_finite_total += int(finite.sum())
            n_voxel_cells_total += int(voxels.shape[0])
            if voxels.size == 0:
                continue

            present_pos = [j for j, gene in enumerate(genes) if gene in var_to_idx]
            values = np.zeros((coords.shape[0], n_genes), dtype=np.float32)
            if present_pos:
                present_idx = [var_to_idx[genes[j]] for j in present_pos]
                values[:, present_pos] = matrix.read_cols(present_idx)
            if spec.normalize == "log1p_norm":
                totals = totals_cache.get(h5ad_path)
                if totals is None:
                    total_idx = normalization_indices(var_names, var_to_idx, norm_genes, args.normalize_scope)
                    totals = matrix.sum_cols_in_chunks(total_idx, args.norm_chunk_size)
                    totals_cache[h5ad_path] = totals
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.divide(values, totals[:, None], out=np.zeros_like(values), where=totals[:, None] > 0)
                values *= float(args.target_sum)
                np.log1p(values, out=values)

            vals = values[finite]
            local_keys, inverse = np.unique(voxels, axis=0, return_inverse=True)
            rows, sum_all, sum_expr, cnt_expr, cnt_all = append_global_rows(
                voxel_index, voxel_keys, sum_all, sum_expr, cnt_expr, cnt_all, local_keys
            )
            local_sum_all = np.zeros((local_keys.shape[0], n_genes), dtype=np.float32)
            local_sum_expr = np.zeros((local_keys.shape[0], n_genes), dtype=np.float32)
            local_cnt_expr = np.zeros((local_keys.shape[0], n_genes), dtype=np.float32)
            local_cnt_all = np.bincount(inverse, minlength=local_keys.shape[0]).astype(np.int64)
            positive = vals > 0
            np.add.at(local_sum_all, inverse, vals)
            np.add.at(local_sum_expr, inverse, vals * positive)
            np.add.at(local_cnt_expr, inverse, positive.astype(np.float32))
            sum_all[rows] += local_sum_all
            sum_expr[rows] += local_sum_expr
            cnt_expr[rows] += local_cnt_expr
            cnt_all[rows] += local_cnt_all

            del matrix, coords, values, vals, local_sum_all, local_sum_expr, local_cnt_expr
            gc.collect()

        if file_i % args.progress_every == 0 or file_i == len(h5ads):
            print(f"    {spec.name}: file {file_i}/{len(h5ads)}, voxels={len(voxel_keys):,}")

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_all = sum_all / cnt_all[:, None]
        mean_expr = np.divide(sum_expr, cnt_expr, out=np.zeros_like(sum_expr), where=cnt_expr > 0)
        frac_expr = cnt_expr / cnt_all[:, None]
    keys = np.asarray(voxel_keys, dtype=np.int64)
    return VoxelAggregate(
        keys=keys,
        index=voxel_index,
        mean_all=mean_all.astype(np.float32, copy=False),
        mean_expr=mean_expr.astype(np.float32, copy=False),
        frac_expr=frac_expr.astype(np.float32, copy=False),
        n_cells_total=n_cells_total,
        n_finite_total=n_finite_total,
        n_voxel_cells_total=n_voxel_cells_total,
    )


def shared_indices(a: VoxelAggregate, b: VoxelAggregate) -> tuple[np.ndarray, np.ndarray]:
    common = set(a.index).intersection(b.index)
    ordered = sorted(common)
    idx_a = np.asarray([a.index[key] for key in ordered], dtype=np.int64)
    idx_b = np.asarray([b.index[key] for key in ordered], dtype=np.int64)
    return idx_a, idx_b


def compare_chunk(
    genes: list[str],
    aggregates: dict[str, VoxelAggregate],
    method_names: list[str],
    positive_filter: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name_a, name_b in itertools.combinations(method_names, 2):
        agg_a = aggregates[name_a]
        agg_b = aggregates[name_b]
        idx_a, idx_b = shared_indices(agg_a, agg_b)
        print(f"  [pair] {name_a} vs {name_b}: shared_voxels={idx_a.size:,}")
        for j, gene in enumerate(genes):
            row: dict[str, object] = {
                "method_a": name_a,
                "method_b": name_b,
                "gene": gene,
                "n_voxels_shared": int(idx_a.size),
            }
            for agg_name in ["mean_all", "mean_expr", "frac_expr"]:
                mat_a = getattr(agg_a, agg_name)
                mat_b = getattr(agg_b, agg_name)
                a = mat_a[idx_a, j]
                b = mat_b[idx_b, j]
                pearson, n_valid = corr_pair(a, b, "pearson", positive_filter)
                spearman, _ = corr_pair(a, b, "spearman", positive_filter)
                row[f"pearson_{agg_name}"] = pearson
                row[f"spearman_{agg_name}"] = spearman
                row[f"n_valid_{agg_name}"] = n_valid
            rows.append(row)
    return rows


def write_summary(df: pd.DataFrame, output: Path, summary: Path | None) -> None:
    long_rows: list[dict[str, object]] = []
    for agg_name in ["mean_all", "mean_expr", "frac_expr"]:
        for corr_name in ["pearson", "spearman"]:
            col = f"{corr_name}_{agg_name}"
            grouped = df.dropna(subset=[col]).groupby(["method_a", "method_b"], as_index=False)
            for _, row in grouped.agg(n_genes=("gene", "count"), median=(col, "median"), mean=(col, "mean")).iterrows():
                long_rows.append(
                    {
                        "method_a": row["method_a"],
                        "method_b": row["method_b"],
                        "metric": f"{corr_name}_{agg_name}",
                        "n_genes": int(row["n_genes"]),
                        "median": float(row["median"]),
                        "mean": float(row["mean"]),
                    }
                )
    summary_df = pd.DataFrame(long_rows)
    summary_path = summary or output.with_suffix(".summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"[summary] {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--file-glob", action="append", required=True)
    parser.add_argument("--source", choices=["X", "layer"], action="append", required=True)
    parser.add_argument("--layer", action="append", required=True)
    parser.add_argument("--normalize", choices=["none", "log1p_norm"], action="append", required=True)
    parser.add_argument("--ccf-key", action="append", required=True)
    parser.add_argument("--coord-columns", action="append", required=True)
    parser.add_argument("--gene-name-column", action="append", required=True)
    parser.add_argument("--ccf-scale", type=float, action="append", required=True)
    parser.add_argument("--voxel-size", type=float, default=200.0)
    parser.add_argument("--voxel-method", choices=["floor", "round"], default="floor")
    parser.add_argument("--positive-filter", choices=["none", "either", "both"], default="none")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--norm-chunk-size", type=int, default=1024)
    parser.add_argument("--normalize-scope", choices=["all", "requested"], default="all")
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = validate_parallel_args(args)
    method_names = [spec.name for spec in specs]
    genes = read_gene_list(args.gene_list)
    if not genes:
        raise ValueError(f"No genes in {args.gene_list}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[genes] {len(genes):,} from {args.gene_list}")
    print(f"[voxel] size={args.voxel_size}, method={args.voxel_method}, positive_filter={args.positive_filter}")
    print(f"[normalize] scope={args.normalize_scope}, target_sum={args.target_sum}")

    all_rows: list[dict[str, object]] = []
    t0 = time.time()
    totals_caches: dict[str, dict[Path, np.ndarray]] = {spec.name: {} for spec in specs}
    for chunk_start in range(0, len(genes), args.chunk_size):
        chunk_genes = genes[chunk_start : chunk_start + args.chunk_size]
        print(f"[chunk] {chunk_start + 1}-{chunk_start + len(chunk_genes)}")
        aggregates: dict[str, VoxelAggregate] = {}
        for spec in specs:
            print(
                f"  [method] {spec.name}: source={spec.source}, layer={spec.layer}, "
                f"normalize={spec.normalize}, input={spec.input_dir}"
            )
            agg = aggregate_method(spec, chunk_genes, genes, args, totals_caches[spec.name])
            aggregates[spec.name] = agg
            finite_pct = agg.n_finite_total / max(agg.n_cells_total, 1) * 100
            print(
                f"  [coords] {spec.name}: cells={agg.n_cells_total:,}, finite={agg.n_finite_total:,} "
                f"({finite_pct:.2f}%), voxel_cells={agg.n_voxel_cells_total:,}, voxels={agg.keys.shape[0]:,}"
            )
        all_rows.extend(compare_chunk(chunk_genes, aggregates, method_names, args.positive_filter))
        elapsed = time.time() - t0
        print(f"[chunk done] {chunk_start + 1}-{chunk_start + len(chunk_genes)}, elapsed={elapsed:.1f}s")
        del aggregates
        gc.collect()

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"[done] {len(df):,} rows -> {args.output}")
    write_summary(df, args.output, args.summary)


if __name__ == "__main__":
    main()
