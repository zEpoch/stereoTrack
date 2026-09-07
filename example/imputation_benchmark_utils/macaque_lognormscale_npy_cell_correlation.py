from __future__ import annotations

import argparse
import gc
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from macaque_lognormscale_npy_method_correlation import h5_read_cols, read_var_names  # noqa: E402


class PearsonAccumulator:
    def __init__(self, n_genes: int) -> None:
        self.n = np.zeros(n_genes, dtype=np.int64)
        self.sum_x = np.zeros(n_genes, dtype=np.float64)
        self.sum_y = np.zeros(n_genes, dtype=np.float64)
        self.sum_x2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_y2 = np.zeros(n_genes, dtype=np.float64)
        self.sum_xy = np.zeros(n_genes, dtype=np.float64)

    def add(self, ref: np.ndarray, pred: np.ndarray, mode: str) -> None:
        valid = np.isfinite(ref) & np.isfinite(pred)
        if mode == "ref_expr":
            valid &= ref > 0
        elif mode == "either_expr":
            valid &= (ref > 0) | (pred > 0)
        elif mode == "both_expr":
            valid &= (ref > 0) & (pred > 0)
        elif mode != "all":
            raise ValueError(mode)

        self.n += valid.sum(axis=0, dtype=np.int64)
        ref_valid = np.where(valid, ref, 0.0)
        pred_valid = np.where(valid, pred, 0.0)
        self.sum_x += ref_valid.sum(axis=0, dtype=np.float64)
        self.sum_y += pred_valid.sum(axis=0, dtype=np.float64)
        self.sum_x2 += np.square(ref_valid, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.sum_y2 += np.square(pred_valid, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.sum_xy += (ref_valid.astype(np.float64, copy=False) * pred_valid.astype(np.float64, copy=False)).sum(
            axis=0, dtype=np.float64
        )

    def pearson(self) -> np.ndarray:
        n = self.n.astype(np.float64)
        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        denom_x = n * self.sum_x2 - self.sum_x * self.sum_x
        denom_y = n * self.sum_y2 - self.sum_y * self.sum_y
        denominator = np.sqrt(denom_x * denom_y)
        out = np.full(self.n.shape, np.nan, dtype=np.float64)
        valid = (self.n >= 3) & np.isfinite(denominator) & (denominator > 0)
        out[valid] = numerator[valid] / denominator[valid]
        return out


def load_meta(model_dir: Path) -> dict:
    meta_path = model_dir / "cache" / "meta.pkl"
    with meta_path.open("rb") as handle:
        return pickle.load(handle)


def load_reference_chunk(cache_npz: Path, gene_indices: list[int]) -> np.ndarray:
    with np.load(cache_npz) as data:
        feat = sp.csr_matrix(
            (
                data["feat_data"].astype(np.float32),
                data["feat_indices"].astype(np.int64),
                data["feat_indptr"].astype(np.int64),
            ),
            shape=tuple(data["feat_shape"]),
        )
    values = feat[:, gene_indices].toarray().astype(np.float32, copy=False)
    return values


def load_stereotrack_chunk(infer_dir: Path, stem: str, gene_indices: list[int], dtype_name: str) -> np.ndarray:
    path = infer_dir / f"{stem}.expression.{dtype_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing StereoTrack inference npy: {path}")
    matrix = np.load(path, mmap_mode="r")
    return np.asarray(matrix[:, gene_indices], dtype=np.float32)


def load_spalp_chunk(infer_dir: Path, stem: str, genes: list[str]) -> np.ndarray:
    path = infer_dir / f"{stem}.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"Missing SpaLP inference h5ad: {path}")
    with h5py.File(path, "r") as handle:
        var_names = read_var_names(handle)
        var_to_idx = {gene: idx for idx, gene in enumerate(var_names)}
        missing = [gene for gene in genes if gene not in var_to_idx]
        if missing:
            raise KeyError(f"{path}: missing {len(missing)} genes, first missing={missing[:5]}")
        indices = [var_to_idx[gene] for gene in genes]
        values = h5_read_cols(handle, indices)
    return values.astype(np.float32, copy=False)


def add_slice(
    meta_info: dict,
    cache_dir: Path,
    methods: list[str],
    accumulators: dict[str, dict[str, PearsonAccumulator]],
    chunk_genes: list[str],
    gene_indices: list[int],
    args: argparse.Namespace,
) -> None:
    stem = Path(meta_info["batch"]).stem
    ref = load_reference_chunk(cache_dir / meta_info["file"], gene_indices)

    if "StereoTrack" in methods:
        pred = load_stereotrack_chunk(args.stereotrack_dir, stem, gene_indices, args.stereotrack_dtype)
        if pred.shape != ref.shape:
            raise ValueError(f"StereoTrack {stem}: shape {pred.shape} does not match reference {ref.shape}")
        for mode, acc in accumulators["StereoTrack"].items():
            acc.add(ref, pred, mode)
        del pred

    if "SpaLP" in methods:
        pred = load_spalp_chunk(args.spalp_dir, stem, chunk_genes)
        if pred.shape != ref.shape:
            raise ValueError(f"SpaLP {stem}: shape {pred.shape} does not match reference {ref.shape}")
        for mode, acc in accumulators["SpaLP"].items():
            acc.add(ref, pred, mode)
        del pred

    del ref


def rows_from_accumulators(chunk_genes: list[str], accumulators: dict[str, dict[str, PearsonAccumulator]]) -> list[dict]:
    rows = []
    for method, mode_to_acc in accumulators.items():
        mode_values = {mode: acc.pearson() for mode, acc in mode_to_acc.items()}
        for j, gene in enumerate(chunk_genes):
            row = {"reference": "Macaque_lognormscale", "method": method, "gene": gene}
            for mode, acc in mode_to_acc.items():
                row[f"pearson_{mode}"] = mode_values[mode][j]
                row[f"n_{mode}"] = int(acc.n[j])
            rows.append(row)
    return rows


def write_summary(df: pd.DataFrame, summary_path: Path) -> None:
    rows = []
    for col in [c for c in df.columns if c.startswith("pearson_")]:
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
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell-level macaque method-to-lognormscale per-gene Pearson benchmark.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stereotrack-dir", type=Path, required=True)
    parser.add_argument("--spalp-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["StereoTrack", "SpaLP"], default=["StereoTrack", "SpaLP"])
    parser.add_argument("--stereotrack-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.model_dir / "cache"
    meta = load_meta(args.model_dir)
    genes = list(meta["common_genes"])
    if args.max_genes > 0:
        genes = genes[: args.max_genes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "macaque_v1_common_genes_lognormscale.txt").write_text("\n".join(map(str, genes)) + "\n")

    modes = ["all", "ref_expr", "either_expr", "both_expr"]
    all_rows = []
    print(f"[genes] {len(genes):,}")
    print(f"[methods] {args.methods}")
    print(f"[reference] {cache_dir}/slice_*_full_graph.npz feat matrix")

    for start in range(0, len(genes), args.chunk_size):
        chunk_genes = genes[start : start + args.chunk_size]
        gene_indices = list(range(start, start + len(chunk_genes)))
        accumulators = {
            method: {mode: PearsonAccumulator(len(chunk_genes)) for mode in modes}
            for method in args.methods
        }
        print(f"[chunk] {start + 1}-{start + len(chunk_genes)}")

        for i, info in enumerate(meta["slice_info"], start=1):
            add_slice(info, cache_dir, args.methods, accumulators, chunk_genes, gene_indices, args)
            if i % args.progress_every == 0 or i == len(meta["slice_info"]):
                print(f"  file {i}/{len(meta['slice_info'])}")
            gc.collect()

        all_rows.extend(rows_from_accumulators(chunk_genes, accumulators))
        del accumulators
        gc.collect()

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False, float_format="%.6f")
    write_summary(df, args.summary)
    print(f"[output] {args.output}")
    print(f"[summary] {args.summary}")


if __name__ == "__main__":
    main()
