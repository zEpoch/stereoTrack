"""
Attach standardized obs annotations to existing inference h5ad files.

This repairs inference outputs produced before 03_inference_v1.py matched
species-local cache h5ad names correctly. It does not rerun the model and keeps
the existing cell_embedding/niche_embedding arrays.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import scanpy as sc
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stereotrack.dataset import normalize_meta


DEFAULT_CONFIG = Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/04_config_three_species.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补齐推理 h5ad 中的标准化 obs 注释")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1_train/inference_adatas"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认覆盖 input-dir 内原文件；如果指定，则写到新目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印匹配情况，不写文件。",
    )
    parser.add_argument(
        "--species-use",
        nargs="*",
        default=None,
        help="可选：只处理指定物种，如 macaque mouse。",
    )
    return parser.parse_args()


def load_meta(config_path: Path) -> tuple[dict, Path]:
    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)
    cache_dir = Path(cfg["paths"]["input_dir"]) / "cache"
    meta_pkl = cache_dir / "meta.pkl"
    meta_yaml = cache_dir / "meta.yaml"

    if meta_pkl.exists():
        with meta_pkl.open("rb") as handle:
            return normalize_meta(pickle.load(handle)), cache_dir
    if meta_yaml.exists():
        with meta_yaml.open("r") as handle:
            return normalize_meta(yaml.safe_load(handle)), cache_dir
    raise FileNotFoundError(f"Cannot find meta.pkl/meta.yaml in {cache_dir}")


def build_cache_map(meta: dict, cache_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    species_meta = meta.get("species", {})

    if species_meta:
        for species, one_species_meta in species_meta.items():
            for local_idx, slice_info in enumerate(one_species_meta.get("slice_info", [])):
                batch_stem = Path(slice_info["batch"]).stem
                inference_name = f"{species}_{batch_stem}.h5ad"
                cache_path = cache_dir / "integrated" / "adatas" / f"{species}_slice_{local_idx}.h5ad"
                mapping[inference_name] = cache_path
        return mapping

    species_counts: dict[str, int] = {}
    for slice_info in meta["slice_info"]:
        species = slice_info.get("species", "unknown")
        local_idx = species_counts.get(species, 0)
        species_counts[species] = local_idx + 1
        batch_stem = Path(slice_info["batch"]).stem
        inference_name = f"{species}_{batch_stem}.h5ad"
        cache_path = cache_dir / "integrated" / "adatas" / f"{species}_slice_{local_idx}.h5ad"
        mapping[inference_name] = cache_path
    return mapping


def merge_obs(inference_path: Path, cache_path: Path, output_path: Path, dry_run: bool) -> str:
    if not cache_path.exists():
        return f"[missing-cache] {inference_path.name}: {cache_path}"

    inference_backed = sc.read_h5ad(inference_path, backed="r")
    cache_backed = sc.read_h5ad(cache_path, backed="r")
    n_inference = inference_backed.n_obs
    n_cache = cache_backed.n_obs
    old_columns = set(inference_backed.obs.columns)
    cache_columns = set(cache_backed.obs.columns)
    inference_backed.file.close()
    cache_backed.file.close()

    if n_inference != n_cache:
        return f"[n_obs-mismatch] {inference_path.name}: inference={n_inference}, cache={n_cache}"

    added_columns = sorted(cache_columns.difference(old_columns))
    if dry_run:
        return f"[dry-run] {inference_path.name}: add {len(added_columns)} obs columns from {cache_path.name}"

    if not added_columns:
        return f"[skip] {inference_path.name}: obs already complete"

    adata = sc.read_h5ad(inference_path)
    cache_adata = sc.read_h5ad(cache_path, backed="r")
    cache_obs = cache_adata.obs.copy()
    cache_var0 = cache_adata.var.iloc[:0].copy()
    cache_adata.file.close()

    old_obs = adata.obs.copy()
    for column in old_obs.columns:
        if column not in cache_obs.columns:
            cache_obs[column] = old_obs[column].to_numpy()
    for column in ["species", "batch", "slice_idx"]:
        if column in old_obs.columns:
            cache_obs[column] = old_obs[column].to_numpy()

    adata.obs = cache_obs
    if adata.n_vars == 0:
        adata.var = cache_var0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    adata.write_h5ad(tmp_path)
    os.replace(tmp_path, output_path)
    return f"[ok] {inference_path.name}: add {len(added_columns)} obs columns"


def main() -> None:
    args = parse_args()
    meta, cache_dir = load_meta(args.config)
    cache_map = build_cache_map(meta, cache_dir)
    output_dir = args.output_dir or args.input_dir

    files = sorted(args.input_dir.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No inference h5ad files found in {args.input_dir}")

    species_use = {species.lower() for species in args.species_use} if args.species_use else None
    for inference_path in files:
        species = inference_path.name.split("_", 1)[0].lower()
        if species_use is not None and species not in species_use:
            continue
        cache_path = cache_map.get(inference_path.name)
        if cache_path is None:
            print(f"[no-map] {inference_path.name}")
            continue
        output_path = output_dir / inference_path.name
        print(merge_obs(inference_path, cache_path, output_path, args.dry_run))


if __name__ == "__main__":
    main()
