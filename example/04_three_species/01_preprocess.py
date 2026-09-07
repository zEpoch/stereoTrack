from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import scanpy as sc
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stereotrack import construct_graph, get_feature_sparse, get_spatial_input, preprocess_adj_sparse


SPECIES_CONFIG = {
	"marmoset": {
		"data_path": Path("/home/share/huadjyin/home/zhoutao3/tracks/example_data/06_marmoset_brain"),
		"ortholog_col": "marmosetGene",
	},
	"macaque": {
		"data_path": Path("/home/share/huadjyin/home/zhoutao3/tracks/example_data/07_macaque_brain/cortex"),
		"ortholog_col": "macaqueGene",
	},
	"mouse": {
		"data_path": Path("/home/share/huadjyin/home/zhoutao3/tracks/example_data/13_mouse_cortex/processed"),
		"ortholog_col": "mouseGene",
	},
}

HUMAN_COL = "humanGene"
COUNT_COLS = ["humanOrthNum", "macaqueOrthNum", "marmosetOrthNum", "mouseOrthNum"]
DEFAULT_ANNOTATION_DIR = Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1/annotations")
STANDARD_ANNOTATION_COLUMNS = [
	"source_obs_name",
	"region",
	"region_detail",
	"region_main",
	"region_raw",
	"layer",
	"layer_raw",
	"cell_class",
	"cell_class_broad",
	"cell_subclass",
	"cell_type",
	"cell_type_raw",
	"annotation_source",
]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Preprocess marmoset, macaque and mouse cortex data with one-to-one ortholog genes.")
	parser.add_argument(
		"--ortholog-file",
		type=Path,
		default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/04_three_species/mart_export.humanMacaqeMarmosetMouse.oneToOneOrth.ensembl91.20220428.txt"),
		help="One-to-one ortholog table across human, macaque, marmoset and mouse.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1"),
		help="Root directory that will contain the cache/ subdir expected by train_pl.py.",
	)
	parser.add_argument(
		"--annotation-dir",
		type=Path,
		default=DEFAULT_ANNOTATION_DIR,
		help="Directory produced by 00_prepare_annotations.py.",
	)
	parser.add_argument("--patch-size", type=int, default=4096, help="Patch size used when splitting each slice.")
	parser.add_argument("--force", action="store_true", help="Rebuild the cache even if meta.yaml already exists.")
	parser.add_argument("--dry-run", action="store_true", help="Only report shared genes and exit.")
	return parser.parse_args()


def mem_usage() -> float:
	proc = psutil.Process(os.getpid())
	return proc.memory_info().rss / 1024**3


def load_ortholog_table(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path, sep="\t")
	df.columns = [column.strip() for column in df.columns]
	required_columns = {HUMAN_COL, "macaqueGene", "marmosetGene", "mouseGene"}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(f"Missing columns in ortholog table: {sorted(missing_columns)}")

	for column in [HUMAN_COL, "macaqueGene", "marmosetGene", "mouseGene"]:
		df[column] = df[column].astype(str).str.strip()

	for column in COUNT_COLS:
		if column in df.columns:
			df[column] = pd.to_numeric(df[column], errors="coerce")

	if all(column in df.columns for column in COUNT_COLS):
		one_to_one_mask = (df[COUNT_COLS] == 1).all(axis=1)
		df = df.loc[one_to_one_mask].copy()

	df = df.dropna(subset=[HUMAN_COL, "macaqueGene", "marmosetGene", "mouseGene"])
	df = df[df[HUMAN_COL].ne("")].copy()
	return df.reset_index(drop=True)


def list_h5ad_files(data_path: Path) -> list[Path]:
	return sorted(path for path in data_path.glob("*.h5ad") if path.is_file())


def gene_lookup(var_names: pd.Index) -> dict[str, str]:
	lookup: dict[str, str] = {}
	for name in var_names.astype(str):
		lookup.setdefault(name.upper(), name)
	return lookup


def species_file_human_genes(adata: ad.AnnData, ortholog_df: pd.DataFrame, species_col: str) -> set[str]:
	var_upper = {name.upper() for name in adata.var_names.astype(str)}
	ortholog_upper = ortholog_df[species_col].astype(str).str.upper()
	matched = ortholog_df.loc[ortholog_upper.isin(var_upper), HUMAN_COL].astype(str)
	return set(matched.tolist())


def compute_shared_human_genes(ortholog_df: pd.DataFrame, species_files: dict[str, list[Path]]) -> tuple[list[str], dict[str, set[str]]]:
	species_common: dict[str, set[str]] = {}
	for species_name, files in species_files.items():
		species_col = SPECIES_CONFIG[species_name]["ortholog_col"]
		common_genes: set[str] | None = None
		for file_path in files:
			adata = ad.read_h5ad(file_path)
			if species_name == "mouse" and "ccf" not in adata.obsm:
				if all(column in adata.obs.columns for column in ["az", "ay", "ax"]):
					adata.obsm["ccf"] = adata.obs[["az", "ay", "ax"]].values
			file_genes = species_file_human_genes(adata, ortholog_df, species_col)
			common_genes = file_genes if common_genes is None else common_genes.intersection(file_genes)
			print(f"{species_name}: {file_path.name} -> {len(file_genes)} ortholog genes")
		species_common[species_name] = common_genes or set()
		print(f"{species_name}: {len(species_common[species_name])} genes shared across files")

	shared = species_common["marmoset"].intersection(species_common["macaque"], species_common["mouse"])
	ordered_shared = [gene for gene in ortholog_df[HUMAN_COL].astype(str).tolist() if gene in shared]
	ordered_shared = list(dict.fromkeys(ordered_shared))
	return ordered_shared, species_common


def ensure_ccf(adata: ad.AnnData) -> ad.AnnData:
	if "ccf" in adata.obsm:
		return adata
	if all(column in adata.obs.columns for column in ["az", "ay", "ax"]):
		adata.obsm["ccf"] = adata.obs[["az", "ay", "ax"]].values
		return adata
	if all(column in adata.obs.columns for column in ["rz", "ry", "rx"]):
		adata.obsm["ccf"] = adata.obs[["rz", "ry", "rx"]].values
		return adata
	if all(column in adata.obs.columns for column in ["x", "y", "z"]):
		adata.obsm["ccf"] = adata.obs[["x", "y", "z"]].values
		return adata
	raise KeyError("Cannot construct ccf coordinates from obs columns")


def attach_standard_annotations(
	adata: ad.AnnData,
	species_name: str,
	file_path: Path,
	annotation_dir: Path,
) -> ad.AnnData:
	annotation_path = annotation_dir / species_name / f"{file_path.stem}.obs_annotations.csv.gz"
	if not annotation_path.exists():
		print(f"  [warn] annotation sidecar not found: {annotation_path}")
		for column in STANDARD_ANNOTATION_COLUMNS:
			if column not in adata.obs.columns:
				adata.obs[column] = "Unknown"
		adata.obs["source_obs_name"] = adata.obs_names.astype(str)
		return adata

	ann = pd.read_csv(annotation_path, dtype=str)
	if "source_obs_name" not in ann.columns:
		raise KeyError(f"{annotation_path}: missing source_obs_name")
	ann = ann.drop_duplicates("source_obs_name").set_index("source_obs_name")
	ann = ann.reindex(adata.obs_names.astype(str))

	for column in STANDARD_ANNOTATION_COLUMNS:
		if column == "source_obs_name":
			adata.obs[column] = adata.obs_names.astype(str)
			continue
		if column in ann.columns:
			adata.obs[column] = ann[column].fillna("Unknown").astype(str).to_numpy()
		else:
			adata.obs[column] = "Unknown"

	matched = adata.obs["annotation_source"].ne("Unknown").sum()
	print(f"  annotations: {matched}/{adata.n_obs} cells from {annotation_path.name}")
	return adata


def select_common_genes(
	adata: ad.AnnData,
	ortholog_df: pd.DataFrame,
	species_col: str,
	common_human_genes: list[str],
) -> ad.AnnData:
	var_lookup = gene_lookup(adata.var_names)
	human_to_species = dict(zip(ortholog_df[HUMAN_COL].astype(str), ortholog_df[species_col].astype(str)))

	selected_source_genes: list[str] = []
	selected_human_genes: list[str] = []
	for human_gene in common_human_genes:
		species_gene = human_to_species.get(human_gene)
		if species_gene is None:
			continue
		source_gene = var_lookup.get(str(species_gene).upper())
		if source_gene is None:
			continue
		selected_source_genes.append(source_gene)
		selected_human_genes.append(human_gene)

	if not selected_source_genes:
		raise ValueError("No common ortholog genes found in the current sample")

	adata = adata[:, selected_source_genes].copy()
	adata.var["source_gene"] = selected_source_genes
	adata.var["human_gene"] = selected_human_genes
	adata.var_names = pd.Index(selected_human_genes)
	adata.var_names_make_unique()
	return adata


def process_species(
	species_name: str,
	data_files: list[Path],
	ortholog_df: pd.DataFrame,
	common_human_genes: list[str],
	cache_root: Path,
	patch_size: int,
	annotation_dir: Path,
	) -> dict:
	species_col = SPECIES_CONFIG[species_name]["ortholog_col"]
	species_cache_dir = cache_root / "integrated"
	adata_cache_dir = species_cache_dir / "adatas"
	species_cache_dir.mkdir(parents=True, exist_ok=True)
	adata_cache_dir.mkdir(parents=True, exist_ok=True)

	patch_infos = []
	slice_infos = []
	for index, file_path in enumerate(data_files):
		adata = ad.read_h5ad(file_path)
		adata.var_names_make_unique()
		adata.obs_names_make_unique()
		adata = attach_standard_annotations(adata, species_name, file_path, annotation_dir)
		adata = ensure_ccf(adata)
		adata = select_common_genes(adata, ortholog_df, species_col, common_human_genes)
		adata.obs["species"] = species_name
		adata.obs["sample_id"] = file_path.stem

		t0 = time.time()
		print(f"\n{'=' * 60}")
		print(f"{species_name} slice {index + 1}/{len(data_files)}: {file_path.name}, memory={mem_usage():.1f} GB")
		print(f"cells={adata.n_obs}, genes={adata.n_vars}")

		sc.pp.normalize_total(adata, target_sum=1e4)
		sc.pp.log1p(adata)
		sc.pp.scale(adata, zero_center=False, max_value=10)

		adata = construct_graph(adata, spatial_key="ccf")
		adata = preprocess_adj_sparse(adata)
		adata = get_spatial_input(adata)

		feat_sparse = get_feature_sparse(torch.device("cpu"), adata.obsm["spatial_input"])
		if "adj_norm" in adata.obsm:
			adj_sparse = adata.obsm["adj_norm"].copy()
		elif "spatial_connectivities" in adata.obsp:
			adj_sparse = adata.obsp["spatial_connectivities"].copy()
		elif "connectivities" in adata.obsp:
			adj_sparse = adata.obsp["connectivities"].copy()
		else:
			raise KeyError(f"{file_path.name}: cannot find adjacency matrix")

		if not sp.issparse(adj_sparse):
			adj_sparse = sp.csr_matrix(adj_sparse)

		coords_full = adata.obsm["ccf"]
		full_graph_name = f"{species_name}_{file_path.stem}_full_graph.npz"
		full_graph_path = species_cache_dir / full_graph_name
		full_graph_relpath = os.path.join("integrated", full_graph_name)
		np.savez_compressed(
			full_graph_path,
			feat_data=feat_sparse.data.astype(np.float16),
			feat_indices=feat_sparse.indices.astype(np.int32),
			feat_indptr=feat_sparse.indptr.astype(np.int32),
			feat_shape=np.array(feat_sparse.shape),
			adj_data=adj_sparse.data.astype(np.float16),
			adj_indices=adj_sparse.indices.astype(np.int32),
			adj_indptr=adj_sparse.indptr.astype(np.int32),
			adj_shape=np.array(adj_sparse.shape),
			coords=coords_full.astype(np.float32),
		)

		slice_infos.append({"species": species_name, "batch": file_path.name, "file": full_graph_relpath, "n_cells": int(adata.n_obs)})

		coords = coords_full
		axis_vars = coords.var(axis=0)
		top_two_axes = np.argsort(axis_vars)[::-1][: min(2, coords.shape[1])]
		n_patches_this = 0
		for pass_idx, target_axis in enumerate(top_two_axes):
			axis_name = ["X", "Y", "Z"][target_axis] if target_axis < 3 else f"Dim_{target_axis}"
			print(f"  pass {pass_idx + 1}/{len(top_two_axes)} axis={axis_name}")
			sorted_indices = np.argsort(coords[:, target_axis])
			for start in range(0, adata.n_obs, patch_size):
				end = min(start + patch_size, adata.n_obs)
				cell_idx = np.sort(sorted_indices[start:end])
				if len(cell_idx) == 0:
					continue
				feat = feat_sparse[cell_idx]
				adj = adj_sparse[cell_idx][:, cell_idx]
				patch_name = f"{species_name}_{file_path.stem}_patch_{n_patches_this}.npz"
				patch_path = species_cache_dir / patch_name
				patch_relpath = os.path.join("integrated", patch_name)
				np.savez_compressed(
					patch_path,
					feat_data=feat.data.astype(np.float16),
					feat_indices=feat.indices.astype(np.int32),
					feat_indptr=feat.indptr.astype(np.int32),
					feat_shape=np.array(feat.shape),
					adj_data=adj.data.astype(np.float16),
					adj_indices=adj.indices.astype(np.int32),
					adj_indptr=adj.indptr.astype(np.int32),
					adj_shape=np.array(adj.shape),
					slice_idx=index,
				)
				patch_infos.append({
					"species": species_name,
					"batch": file_path.name,
					"file": patch_relpath,
					"n_cells": int(len(cell_idx)),
					"n_genes": int(adata.n_vars),
					"slice_idx": index,
				})
				n_patches_this += 1

		print(f"  patches={n_patches_this}")
		if "spatial_input" in adata.obsm:
			del adata.obsm["spatial_input"]
		if "adj_norm" in adata.obsm:
			del adata.obsm["adj_norm"]
		if "spatial_connectivities" in adata.obsp:
			del adata.obsp["spatial_connectivities"]
		adata.write_h5ad(adata_cache_dir / f"{species_name}_slice_{index}.h5ad")

		del adata, feat_sparse, adj_sparse
		gc.collect()
		print(f"  elapsed={time.time() - t0:.1f}s, memory={mem_usage():.1f} GB")

	return {
		"data_path": str(SPECIES_CONFIG[species_name]["data_path"]),
		"n_slices": len(data_files),
		"patches": patch_infos,
		"slice_info": slice_infos,
		"input_dim": len(common_human_genes),
	}


def main() -> None:
	args = parse_args()
	ortholog_file = args.ortholog_file
	output_root = args.output_dir
	cache_root = output_root / "cache"
	meta_path = cache_root / "meta.yaml"
	meta_pkl_path = cache_root / "meta.pkl"

	if not ortholog_file.exists():
		raise FileNotFoundError(f"Ortholog file does not exist: {ortholog_file}")

	ortholog_df = load_ortholog_table(ortholog_file)
	species_files = {species_name: list_h5ad_files(cfg["data_path"]) for species_name, cfg in SPECIES_CONFIG.items()}
	for species_name, files in species_files.items():
		if not files:
			raise FileNotFoundError(f"No h5ad files found for {species_name}: {SPECIES_CONFIG[species_name]['data_path']}")

	common_human_genes, species_common = compute_shared_human_genes(ortholog_df, species_files)
	print(f"shared human genes across three species: {len(common_human_genes)}")

	if args.dry_run:
		print("dry-run only, no files were written")
		return

	if meta_path.exists() and not args.force:
		print(f"Cache already exists: {cache_root}")
		print("Use --force to rebuild")
		return

	cache_root.mkdir(parents=True, exist_ok=True)
	meta = {
		"ortholog_file": str(ortholog_file),
		"patch_size": args.patch_size,
		"common_human_genes": common_human_genes,
		"input_dim": len(common_human_genes),
		"patches": [],
		"slice_info": [],
		"species_common_genes": {name: sorted(list(genes)) for name, genes in species_common.items()},
		"all_patches": [],
		"all_slice_info": [],
		"species": {},
	}

	for species_name, files in species_files.items():
		species_meta = process_species(
			species_name,
			files,
			ortholog_df,
			common_human_genes,
			cache_root,
			args.patch_size,
			args.annotation_dir,
		)
		meta["species"][species_name] = species_meta
		meta["all_patches"].extend(species_meta["patches"])
		meta["all_slice_info"].extend(species_meta["slice_info"])
		meta["patches"].extend(species_meta["patches"])
		meta["slice_info"].extend(species_meta["slice_info"])

	with open(meta_path, "w") as handle:
		yaml.safe_dump(meta, handle, sort_keys=False, allow_unicode=True)

	with open(meta_pkl_path, "wb") as handle:
		pickle.dump(meta, handle)

	print("\nPreprocessing complete")
	print(f"cache_root={cache_root}")
	print(f"shared_genes={len(common_human_genes)}")
	for species_name in SPECIES_CONFIG:
		print(f"{species_name}: {meta['species'][species_name]['n_slices']} slices")


if __name__ == "__main__":
	main()
