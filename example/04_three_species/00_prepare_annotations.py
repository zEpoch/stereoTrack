from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)


SPECIES_CONFIG = {
	"marmoset": {
		"data_path": Path("/home/share/huadjyin/home/zhoutao3/tracks/example_data/06_marmoset_brain"),
	},
	"macaque": {
		"data_path": Path("/home/share/huadjyin/home/zhoutao3/tracks/example_data/07_macaque_brain/cortex"),
	},
	"mouse": {
		"data_path": Path("/home/share/huadjyin/home/zhoutao3/tracks/example_data/13_mouse_cortex/processed"),
	},
}

DEFAULT_MARMOSET_ANNOTATION = Path(
	"/home/share/huadjyin/home/zhoutao3/tracks/example_data/06_marmoset_brain/Cellbin_annotation_marmoset_all_addsegment.csv"
)
DEFAULT_OUTPUT_DIR = Path(
	"/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/04_three_species_mae_v1/annotations"
)

STANDARD_COLUMNS = [
	"source_obs_name",
	"species",
	"sample_id",
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

UNKNOWN = "Unknown"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Prepare standardized obs annotation sidecars for three-species cortex integration. "
			"The sidecars are consumed by 01_preprocess.py."
		)
	)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--marmoset-data-dir", type=Path, default=SPECIES_CONFIG["marmoset"]["data_path"])
	parser.add_argument("--macaque-data-dir", type=Path, default=SPECIES_CONFIG["macaque"]["data_path"])
	parser.add_argument("--mouse-data-dir", type=Path, default=SPECIES_CONFIG["mouse"]["data_path"])
	parser.add_argument("--marmoset-annotation", type=Path, default=DEFAULT_MARMOSET_ANNOTATION)
	parser.add_argument("--species", nargs="*", default=["marmoset", "macaque", "mouse"], choices=["marmoset", "macaque", "mouse"])
	parser.add_argument("--chunk-size", type=int, default=500000, help="Rows per chunk when scanning the marmoset CSV.")
	parser.add_argument("--limit-files", type=int, default=None, help="Optional smoke-test limit per species.")
	parser.add_argument("--force", action="store_true", help="Overwrite existing sidecar files.")
	parser.add_argument("--dry-run", action="store_true", help="Inspect inputs and report intended outputs without writing.")
	return parser.parse_args()


def list_h5ad_files(data_path: Path, limit: int | None = None) -> list[Path]:
	files = sorted(path for path in data_path.glob("*.h5ad") if path.is_file())
	if limit is not None:
		files = files[:limit]
	return files


def as_clean_series(values, index: pd.Index | None = None) -> pd.Series:
	if isinstance(values, pd.Series):
		series = values.astype("string").fillna(UNKNOWN).astype(str)
	else:
		series = pd.Series(values, index=index, dtype="string").fillna(UNKNOWN).astype(str)
	series = series.replace({"": UNKNOWN, "nan": UNKNOWN, "None": UNKNOWN, "<NA>": UNKNOWN})
	return series


def standardize_layer(values) -> pd.Series:
	series = as_clean_series(values)
	out = []
	for raw in series:
		label = str(raw).strip()
		if not label or label in {UNKNOWN, "NA"}:
			out.append(UNKNOWN)
			continue
		label = label.replace("Layer", "").replace("layer", "").strip()
		if label.lower().startswith("l"):
			out.append("L" + label[1:])
		else:
			out.append("L" + label)
	return pd.Series(out, index=series.index, dtype=str)


def parse_macaque_layer(region_values) -> pd.Series:
	series = as_clean_series(region_values)
	out = []
	for value in series:
		match = re.search(r"-l([0-9/]+)$", value, flags=re.IGNORECASE)
		out.append(f"L{match.group(1)}" if match else UNKNOWN)
	return pd.Series(out, index=series.index, dtype=str)


def harmonize_cell_class_broad(
	cell_class,
	cell_subclass=None,
	cell_type=None,
) -> pd.Series:
	cell_class = as_clean_series(cell_class)
	cell_subclass = as_clean_series(cell_subclass if cell_subclass is not None else [UNKNOWN] * len(cell_class), index=cell_class.index)
	cell_type = as_clean_series(cell_type if cell_type is not None else [UNKNOWN] * len(cell_class), index=cell_class.index)
	out = []
	for cls, sub, ctype in zip(cell_class, cell_subclass, cell_type):
		text = f"{cls} {sub} {ctype}".lower()
		if cls.upper() == "GLU" or cls.lower() in {"glut", "glutamatergic"} or "excitatory" in text or "_glu_" in text:
			out.append("Glutamatergic")
		elif cls.upper() == "GABA" or cls.lower() in {"gaba", "gabaergic"} or "inhibitory" in text or "_gaba_" in text:
			out.append("GABAergic")
		elif cls in {UNKNOWN, "NA"}:
			out.append(UNKNOWN)
		else:
			out.append("Non-neuronal")
	return pd.Series(out, index=cell_class.index, dtype=str)


def source_obs_names(adata: ad.AnnData) -> pd.Index:
	return pd.Index(adata.obs_names.astype(str), name="source_obs_name")


def sidecar_path(output_dir: Path, species: str, file_path: Path) -> Path:
	return output_dir / species / f"{file_path.stem}.obs_annotations.csv.gz"


def write_sidecar(df: pd.DataFrame, output_path: Path, force: bool, dry_run: bool) -> None:
	missing = [column for column in STANDARD_COLUMNS if column not in df.columns]
	if missing:
		raise ValueError(f"Missing standardized columns: {missing}")
	df = df.loc[:, STANDARD_COLUMNS].copy()
	if output_path.exists() and not force:
		print(f"[skip] {output_path} exists; use --force to overwrite")
		return
	if dry_run:
		print(f"[dry-run] would write {output_path} rows={len(df)}")
		return
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	print(f"[write] {output_path} rows={len(df)}")


def load_backed_obs(file_path: Path) -> tuple[pd.DataFrame, pd.Index]:
	adata = ad.read_h5ad(file_path, backed="r")
	obs = adata.obs.copy()
	names = source_obs_names(adata)
	adata.file.close()
	return obs, names


def base_annotation_frame(species: str, sample_id: str, obs_names: pd.Index) -> pd.DataFrame:
	return pd.DataFrame(
		{
			"source_obs_name": obs_names.astype(str),
			"species": species,
			"sample_id": sample_id,
		}
	)


def marmoset_cache_path(output_dir: Path, slice_id: str) -> Path:
	return output_dir / "_marmoset_csv_cache" / f"{slice_id}.csv.gz"


def collect_marmoset_wanted_ids(files: list[Path]) -> dict[str, set[str]]:
	wanted: dict[str, set[str]] = {}
	for file_path in files:
		_, obs_names = load_backed_obs(file_path)
		wanted[file_path.stem] = set(obs_names.astype(str))
		print(f"[marmoset ids] {file_path.name}: {len(wanted[file_path.stem])} cells")
	return wanted


def scan_marmoset_csv(
	annotation_csv: Path,
	output_dir: Path,
	wanted_by_slice: dict[str, set[str]],
	chunk_size: int,
	force: bool,
	dry_run: bool,
) -> None:
	if not annotation_csv.exists():
		raise FileNotFoundError(f"Marmoset annotation CSV not found: {annotation_csv}")

	cache_dir = output_dir / "_marmoset_csv_cache"
	if not dry_run:
		cache_dir.mkdir(parents=True, exist_ok=True)
		if force:
			for slice_id in wanted_by_slice:
				cache_file = marmoset_cache_path(output_dir, slice_id)
				if cache_file.exists():
					cache_file.unlink()

	usecols = ["slice", "cell_label", "region", "layer", "cluster1", "cluster2", "celltype", "celltype_raw"]
	total_matched = 0
	for chunk_idx, chunk in enumerate(pd.read_csv(annotation_csv, usecols=usecols, chunksize=chunk_size), start=1):
		chunk = chunk.loc[chunk["slice"].astype(str).isin(wanted_by_slice)].copy()
		if chunk.empty:
			continue
		cell_numeric = pd.to_numeric(chunk["cell_label"], errors="coerce")
		chunk["source_obs_name"] = cell_numeric.astype("Int64").astype(str)
		chunk = chunk.loc[chunk["source_obs_name"].ne("<NA>")]
		for slice_id, sub in chunk.groupby("slice", observed=True):
			slice_id = str(slice_id)
			wanted = wanted_by_slice.get(slice_id)
			if not wanted:
				continue
			sub = sub.loc[sub["source_obs_name"].isin(wanted)].copy()
			if sub.empty:
				continue
			sub = sub.drop_duplicates("source_obs_name")
			total_matched += len(sub)
			if dry_run:
				continue
			cache_file = marmoset_cache_path(output_dir, slice_id)
			write_header = not cache_file.exists()
			sub.to_csv(cache_file, mode="a", header=write_header, index=False)
		if chunk_idx % 10 == 0:
			print(f"[marmoset csv] scanned chunks={chunk_idx}, matched_rows={total_matched}")
	print(f"[marmoset csv] total matched rows={total_matched}")


def prepare_marmoset(files: list[Path], annotation_csv: Path, output_dir: Path, chunk_size: int, force: bool, dry_run: bool) -> None:
	wanted_by_slice = collect_marmoset_wanted_ids(files)
	scan_marmoset_csv(annotation_csv, output_dir, wanted_by_slice, chunk_size, force=force, dry_run=dry_run)
	if dry_run:
		return

	for file_path in files:
		obs, obs_names = load_backed_obs(file_path)
		df = base_annotation_frame("marmoset", file_path.stem, obs_names)
		df["region_raw"] = as_clean_series(obs.get("region", UNKNOWN), index=obs.index).to_numpy()
		df["region"] = df["region_raw"]
		df["region_detail"] = df["region_raw"]
		df["region_main"] = df["region_raw"]
		df["layer_raw"] = as_clean_series(obs.get("layer", UNKNOWN), index=obs.index).to_numpy()
		df["layer"] = standardize_layer(df["layer_raw"]).to_numpy()

		cache_file = marmoset_cache_path(output_dir, file_path.stem)
		if cache_file.exists():
			ann = pd.read_csv(cache_file, dtype=str).drop_duplicates("source_obs_name").set_index("source_obs_name")
			ann = ann.reindex(df["source_obs_name"].astype(str))
			has_ann = ann["celltype"].notna().to_numpy()
			df.loc[has_ann, "region_raw"] = as_clean_series(ann.loc[has_ann, "region"]).to_numpy()
			df.loc[has_ann, "region"] = df.loc[has_ann, "region_raw"]
			df.loc[has_ann, "region_detail"] = df.loc[has_ann, "region_raw"]
			df.loc[has_ann, "region_main"] = df.loc[has_ann, "region_raw"]
			df.loc[has_ann, "layer_raw"] = as_clean_series(ann.loc[has_ann, "layer"]).to_numpy()
			df.loc[has_ann, "layer"] = standardize_layer(df.loc[has_ann, "layer_raw"]).to_numpy()
			df["cell_class"] = as_clean_series(ann["cluster1"]).to_numpy()
			df["cell_subclass"] = as_clean_series(ann["cluster2"]).to_numpy()
			df["cell_type"] = as_clean_series(ann["celltype"]).to_numpy()
			df["cell_type_raw"] = as_clean_series(ann["celltype_raw"]).to_numpy()
			df["annotation_source"] = np.where(has_ann, "marmoset_csv", "h5ad_region_layer_only")
		else:
			df["cell_class"] = UNKNOWN
			df["cell_subclass"] = UNKNOWN
			df["cell_type"] = UNKNOWN
			df["cell_type_raw"] = UNKNOWN
			df["annotation_source"] = "h5ad_region_layer_only"
		df["cell_class_broad"] = harmonize_cell_class_broad(df["cell_class"], df["cell_subclass"], df["cell_type"]).to_numpy()
		write_sidecar(df, sidecar_path(output_dir, "marmoset", file_path), force=force, dry_run=dry_run)


def prepare_macaque(files: list[Path], output_dir: Path, force: bool, dry_run: bool) -> None:
	for file_path in files:
		obs, obs_names = load_backed_obs(file_path)
		df = base_annotation_frame("macaque", file_path.stem, obs_names)
		region_raw = as_clean_series(obs.get("region", UNKNOWN), index=obs.index)
		main_region = as_clean_series(obs.get("main_region", UNKNOWN), index=obs.index)
		cell_class = as_clean_series(obs.get("Class", UNKNOWN), index=obs.index)
		cell_subclass = as_clean_series(obs.get("SubClass", UNKNOWN), index=obs.index)
		cell_type = as_clean_series(obs.get("celltype", UNKNOWN), index=obs.index)
		df["region_raw"] = region_raw.to_numpy()
		df["region"] = main_region.to_numpy()
		df["region_detail"] = region_raw.to_numpy()
		df["region_main"] = main_region.to_numpy()
		df["layer_raw"] = parse_macaque_layer(region_raw).to_numpy()
		df["layer"] = standardize_layer(df["layer_raw"]).to_numpy()
		df["cell_class"] = cell_class.to_numpy()
		df["cell_class_broad"] = harmonize_cell_class_broad(cell_class, cell_subclass, cell_type).to_numpy()
		df["cell_subclass"] = cell_subclass.to_numpy()
		df["cell_type"] = cell_type.to_numpy()
		df["cell_type_raw"] = cell_type.to_numpy()
		df["annotation_source"] = "macaque_h5ad_obs"
		write_sidecar(df, sidecar_path(output_dir, "macaque", file_path), force=force, dry_run=dry_run)


def prepare_mouse(files: list[Path], output_dir: Path, force: bool, dry_run: bool) -> None:
	for file_path in files:
		obs, obs_names = load_backed_obs(file_path)
		df = base_annotation_frame("mouse", file_path.stem, obs_names)
		region_raw = as_clean_series(obs.get("region", UNKNOWN), index=obs.index)
		cortex_area = as_clean_series(obs.get("cortex_area", UNKNOWN), index=obs.index)
		cortex_region = as_clean_series(obs.get("cortex_region", UNKNOWN), index=obs.index)
		cortex_main = as_clean_series(obs.get("cortex_main_region", obs.get("main_region", UNKNOWN)), index=obs.index)
		layer_raw = as_clean_series(obs.get("cortex_layer", UNKNOWN), index=obs.index)
		cell_class = as_clean_series(obs.get("cell_class", UNKNOWN), index=obs.index)
		cell_subclass = as_clean_series(obs.get("cell_subclass", UNKNOWN), index=obs.index)
		cell_type = as_clean_series(obs.get("cell_cluster", UNKNOWN), index=obs.index)
		df["region_raw"] = region_raw.to_numpy()
		df["region"] = cortex_area.to_numpy()
		df["region_detail"] = cortex_region.to_numpy()
		df["region_main"] = cortex_main.to_numpy()
		df["layer_raw"] = layer_raw.to_numpy()
		df["layer"] = standardize_layer(layer_raw).to_numpy()
		df["cell_class"] = cell_class.to_numpy()
		df["cell_class_broad"] = harmonize_cell_class_broad(cell_class, cell_subclass, cell_type).to_numpy()
		df["cell_subclass"] = cell_subclass.to_numpy()
		df["cell_type"] = cell_type.to_numpy()
		df["cell_type_raw"] = cell_type.to_numpy()
		df["annotation_source"] = "mouse_h5ad_obs"
		write_sidecar(df, sidecar_path(output_dir, "mouse", file_path), force=force, dry_run=dry_run)


def write_summary(output_dir: Path, dry_run: bool) -> None:
	if dry_run:
		return
	rows = []
	for species_dir in sorted(path for path in output_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
		for file_path in sorted(species_dir.glob("*.obs_annotations.csv.gz")):
			df = pd.read_csv(file_path, usecols=["species", "sample_id", "region", "layer", "cell_class_broad", "cell_subclass", "cell_type"])
			rows.append(
				{
					"species": df["species"].iloc[0],
					"sample_id": df["sample_id"].iloc[0],
					"n_cells": len(df),
					"n_region": df["region"].nunique(dropna=True),
					"n_layer": df["layer"].nunique(dropna=True),
					"n_cell_class_broad": df["cell_class_broad"].nunique(dropna=True),
					"n_cell_subclass": df["cell_subclass"].nunique(dropna=True),
					"n_cell_type": df["cell_type"].nunique(dropna=True),
					"unknown_cell_type_fraction": float(df["cell_type"].eq(UNKNOWN).mean()),
				}
			)
	if rows:
		summary = pd.DataFrame(rows)
		summary.to_csv(output_dir / "annotation_summary.csv", index=False)
		print(f"[write] {output_dir / 'annotation_summary.csv'} rows={len(summary)}")


def main() -> None:
	args = parse_args()
	data_dirs = {
		"marmoset": args.marmoset_data_dir,
		"macaque": args.macaque_data_dir,
		"mouse": args.mouse_data_dir,
	}

	files_by_species = {
		species: list_h5ad_files(data_dirs[species], args.limit_files)
		for species in args.species
	}
	for species, files in files_by_species.items():
		if not files:
			raise FileNotFoundError(f"No h5ad files found for {species}: {data_dirs[species]}")
		print(f"[input] {species}: {len(files)} h5ad files from {data_dirs[species]}")

	if "marmoset" in files_by_species:
		prepare_marmoset(
			files_by_species["marmoset"],
			args.marmoset_annotation,
			args.output_dir,
			args.chunk_size,
			force=args.force,
			dry_run=args.dry_run,
		)
	if "macaque" in files_by_species:
		prepare_macaque(files_by_species["macaque"], args.output_dir, force=args.force, dry_run=args.dry_run)
	if "mouse" in files_by_species:
		prepare_mouse(files_by_species["mouse"], args.output_dir, force=args.force, dry_run=args.dry_run)

	write_summary(args.output_dir, dry_run=args.dry_run)
	print(f"Done. Annotation sidecars are in: {args.output_dir}")


if __name__ == "__main__":
	main()
