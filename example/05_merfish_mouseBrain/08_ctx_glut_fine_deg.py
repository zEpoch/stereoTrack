from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


DEFAULT_WORK_DIR = Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack")
DEFAULT_TRAIN_DIR = DEFAULT_WORK_DIR / "out/05_merfish_mouseBrain_mae_v1_3axes_train"
DEFAULT_RAW_DIR = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/example_data/14_merfish_mouseBrain/raw_data"
)
DEFAULT_CONCAT = DEFAULT_TRAIN_DIR / "rapids_analysis/merfish_mouseBrain_concat_embeddings.h5ad"
DEFAULT_MAP = DEFAULT_TRAIN_DIR / "figures_out/spatial_cell/leiden_author_map.csv"
DEFAULT_OUTPUT = DEFAULT_TRAIN_DIR / "fine_celltype_deg_ctx_glut"
DEFAULT_TARGETS = [
    "L2/3 IT CTX Glut",
    "L4/5 IT CTX Glut",
    "L6 CT CTX Glut",
]
DEFAULT_SECTIONS = [
    "C57BL6J-1.080",
    "C57BL6J-3.007",
    "C57BL6J-3.015",
]
ANNOTATION_COLUMNS = [
    "concat_row",
    "raw_obs_name",
    "cell_leiden",
    "niche_leiden",
    "subclass_transfer",
    "cluster_id_transfer",
    "cell_type",
    "cell_type_ontology_term_id",
    "major_brain_region",
    "ccf_region_name",
    "brain_section_label",
    "high_quality_transfer",
    "source_file",
    "source_stem",
    "batch",
    "slice_idx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MERFISH CTX glutamatergic author subclasses, compute markers "
            "for StereoTrack cell_leiden fine groups, and save section expression h5ad."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--concat-h5ad", type=Path, default=DEFAULT_CONCAT)
    parser.add_argument("--leiden-author-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--sections", nargs="*", default=DEFAULT_SECTIONS)
    parser.add_argument("--author-key", default="subclass_transfer")
    parser.add_argument("--leiden-key", default="cell_leiden")
    parser.add_argument(
        "--selection-mode",
        choices=["author_and_leiden", "leiden_only"],
        default="author_and_leiden",
        help=(
            "author_and_leiden tests fine Leiden groups inside the author subclass. "
            "leiden_only uses all cells from Leiden groups mapped to the author subclass."
        ),
    )
    parser.add_argument("--min-cells-per-group", type=int, default=20)
    parser.add_argument("--method", default="wilcoxon", choices=["wilcoxon", "t-test", "logreg"])
    parser.add_argument("--deg-backend", default="auto", choices=["auto", "rapids", "scanpy"])
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--max-cells-per-target", type=int, default=None)
    parser.add_argument("--max-section-cells", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--save-target-h5ad", action="store_true")
    parser.add_argument("--keep-filtered-genes", action="store_true")
    parser.add_argument("--no-section-obsm", action="store_true")
    parser.add_argument(
        "--section-obsm-keys",
        nargs="*",
        default=[
            "cell_embedding",
            "niche_embedding",
            "X_umap_cell",
            "X_umap_niche",
            "X_umap",
            "X_CCF",
            "X_spatial_coords",
            "ccf",
        ],
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR / "inference_adatas",
        help="Directory used to look for optional imputation layers from per-slice inference h5ad files.",
    )
    parser.add_argument("--imputation-layers", nargs="*", default=["niche_recon", "cell_recon"])
    parser.add_argument("--no-attach-imputation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return value or "label"


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        if not name or name.lower() == "nan":
            name = "unknown"
        count = seen.get(name, 0)
        out.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return out


def raw_obs_name_from_concat(obs_names: pd.Index, source_stem: pd.Series) -> pd.Series:
    obs_names = obs_names.astype(str)
    stems = source_stem.astype(str).to_numpy()
    raw_names = []
    for obs_name, stem in zip(obs_names, stems):
        prefix = f"{stem}:"
        if obs_name.startswith(prefix):
            raw_names.append(obs_name[len(prefix) :])
        elif ":" in obs_name:
            raw_names.append(obs_name.split(":", 1)[1])
        else:
            raw_names.append(obs_name)
    return pd.Series(raw_names, index=obs_names, dtype="string")


def load_concat_obs(path: Path) -> pd.DataFrame:
    adata = sc.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    obs["concat_row"] = np.arange(obs.shape[0], dtype=np.int64)
    if "source_stem" not in obs.columns:
        if "source_file" not in obs.columns:
            adata.file.close()
            raise KeyError("concat h5ad must contain obs['source_stem'] or obs['source_file']")
        obs["source_stem"] = obs["source_file"].astype(str).str.replace(r"\.h5ad$", "", regex=True)
    obs["raw_obs_name"] = raw_obs_name_from_concat(obs.index, obs["source_stem"]).to_numpy()
    adata.file.close()
    return obs


def load_leiden_mapping(path: Path, targets: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    mapping = pd.read_csv(path)
    required = {"cell_leiden", "matched_author_class", "matched_fraction"}
    missing = required - set(mapping.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    mapping["cell_leiden"] = mapping["cell_leiden"].astype(str)
    mapping["matched_author_class"] = mapping["matched_author_class"].astype(str)
    out = {}
    for target in targets:
        sub = mapping[mapping["matched_author_class"].eq(target)].copy()
        sub = sub.sort_values(["matched_fraction", "cell_leiden"], ascending=[False, True])
        out[target] = sub["cell_leiden"].astype(str).tolist()
    return mapping, out


def raw_var_mask_and_names(raw: ad.AnnData, keep_filtered_genes: bool) -> tuple[np.ndarray, list[str]]:
    if not keep_filtered_genes and "feature_is_filtered" in raw.var.columns:
        filtered = raw.var["feature_is_filtered"]
        if pd.api.types.is_bool_dtype(filtered):
            is_filtered = filtered.fillna(False).astype(bool).to_numpy()
        else:
            is_filtered = (
                filtered.astype("string")
                .fillna("false")
                .str.lower()
                .isin(["true", "1", "yes", "y"])
                .to_numpy()
            )
        keep = ~is_filtered
    else:
        keep = np.ones(raw.n_vars, dtype=bool)

    if "gene_name" in raw.var.columns:
        names = raw.var.loc[keep, "gene_name"].astype(str).tolist()
    else:
        names = raw.var_names[keep].astype(str).tolist()
    return keep, make_unique(names)


def attach_annotations(
    adata: ad.AnnData,
    meta: pd.DataFrame,
    source_stem: str,
) -> ad.AnnData:
    meta = meta.copy()
    raw_names = meta["raw_obs_name"].astype(str).tolist()
    adata.obs_names = pd.Index([f"{source_stem}:{name}" for name in raw_names])
    for column in ANNOTATION_COLUMNS:
        if column in meta.columns:
            adata.obs[column] = meta[column].to_numpy()
    return adata


def read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        shape = (0,) + tuple(dataset.shape[1:])
        return np.empty(shape, dtype=dataset.dtype)
    order = np.argsort(rows, kind="mergesort")
    sorted_rows = rows[order]
    if np.unique(sorted_rows).size != sorted_rows.size:
        values = np.asarray([dataset[int(row)] for row in sorted_rows])
    else:
        values = np.asarray(dataset[sorted_rows])
    reverse = np.empty_like(order)
    reverse[order] = np.arange(order.size)
    return values[reverse]


def attach_concat_obsm(
    adata: ad.AnnData,
    concat_h5ad: Path,
    obsm_keys: list[str],
) -> list[str]:
    if "concat_row" not in adata.obs.columns:
        raise KeyError("adata.obs['concat_row'] is required to attach concat obsm arrays")
    rows = adata.obs["concat_row"].to_numpy(dtype=np.int64)
    attached: list[str] = []
    with h5py.File(concat_h5ad, "r") as handle:
        obsm = handle.get("obsm")
        if obsm is None:
            return attached
        for key in obsm_keys:
            if key not in obsm:
                continue
            obj = obsm[key]
            if not hasattr(obj, "shape"):
                continue
            adata.obsm[key] = read_h5_rows(obj, rows).astype(np.float32, copy=False)
            attached.append(key)
    return attached


def attach_imputation_layers(
    adata: ad.AnnData,
    inference_dir: Path,
    layer_names: list[str],
) -> list[str]:
    if not inference_dir.is_dir() or not layer_names:
        return []

    attached: list[str] = []
    layer_blocks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in layer_names}
    layer_var_names: dict[str, list[str]] = {}
    row_blocks: list[np.ndarray] = []

    for source_stem, obs_idx in adata.obs.groupby("source_stem", sort=False, observed=True).groups.items():
        obs_pos = adata.obs.index.get_indexer(obs_idx)
        raw_names = adata.obs.iloc[obs_pos]["raw_obs_name"].astype(str).tolist()
        inference_path = inference_dir / f"{source_stem}.h5ad"
        if not inference_path.is_file():
            continue

        with h5py.File(inference_path, "r") as handle:
            if "layers" not in handle or not list(handle["layers"].keys()):
                continue
            inf = sc.read_h5ad(inference_path, backed="r")
            row_pos = inf.obs_names.get_indexer(raw_names)
            var_names = inf.var_names.astype(str).tolist()
            inf.file.close()
            if (row_pos < 0).any():
                missing = np.asarray(raw_names, dtype=object)[row_pos < 0][:10].tolist()
                raise KeyError(f"{inference_path.name}: missing cells for imputation layers: {missing}")
            row_blocks.append(obs_pos)
            for layer_name in layer_names:
                if layer_name not in handle["layers"]:
                    continue
                values = read_h5_rows(handle["layers"][layer_name], row_pos).astype(np.float32, copy=False)
                layer_blocks[layer_name].append((obs_pos, values))
                layer_var_names[layer_name] = var_names

    if not row_blocks:
        return attached

    n_obs = adata.n_obs
    for layer_name, blocks in layer_blocks.items():
        if not blocks:
            continue
        var_names = layer_var_names.get(layer_name, [])
        n_vars = blocks[0][1].shape[1]
        values = np.zeros((n_obs, n_vars), dtype=np.float32)
        filled = np.zeros(n_obs, dtype=bool)
        for obs_pos, block in blocks:
            values[obs_pos] = block
            filled[obs_pos] = True
        if not filled.all():
            print(f"  imputation layer {layer_name}: only filled {filled.sum():,}/{n_obs:,} cells; skipping")
            continue
        if n_vars == adata.n_vars and list(adata.var_names.astype(str)) == var_names:
            adata.layers[layer_name] = values
        else:
            key = f"imputation_{layer_name}"
            adata.obsm[key] = values
            adata.uns[f"{key}_var_names"] = var_names
        attached.append(layer_name)
    return attached


def attach_model_outputs(
    adata: ad.AnnData,
    concat_h5ad: Path,
    obsm_keys: list[str],
    inference_dir: Path,
    imputation_layers: list[str],
    attach_obsm: bool,
    attach_imputation: bool,
) -> dict[str, list[str]]:
    summary = {"obsm": [], "imputation_layers": []}
    if attach_obsm:
        summary["obsm"] = attach_concat_obsm(adata, concat_h5ad, obsm_keys)
    if attach_imputation:
        summary["imputation_layers"] = attach_imputation_layers(adata, inference_dir, imputation_layers)
    return summary


def read_expression_subset(
    raw_dir: Path,
    meta: pd.DataFrame,
    keep_filtered_genes: bool,
) -> ad.AnnData:
    if meta.empty:
        raise ValueError("No cells selected")

    adatas: list[ad.AnnData] = []
    for source_stem, sub_meta in meta.groupby("source_stem", sort=False, observed=True):
        raw_path = raw_dir / f"{source_stem}.h5ad"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        raw = sc.read_h5ad(raw_path, backed="r")
        keep_var, gene_names = raw_var_mask_and_names(raw, keep_filtered_genes)
        raw_names = sub_meta["raw_obs_name"].astype(str).tolist()
        row_pos = raw.obs_names.get_indexer(raw_names)
        if (row_pos < 0).any():
            missing = np.asarray(raw_names, dtype=object)[row_pos < 0][:10].tolist()
            raw.file.close()
            raise KeyError(f"{raw_path.name}: missing raw obs names: {missing}")

        order = np.argsort(row_pos)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(order.size)
        tmp = raw[row_pos[order], keep_var].to_memory()
        tmp = tmp[reverse, :].copy()
        raw.file.close()

        tmp.var["source_var_name"] = tmp.var_names.astype(str)
        tmp.var_names = gene_names
        tmp.var_names_make_unique()
        tmp = attach_annotations(tmp, sub_meta, str(source_stem))
        adatas.append(tmp)
        print(f"  loaded {source_stem}: {tmp.n_obs:,} cells x {tmp.n_vars:,} genes")

    if len(adatas) == 1:
        out = adatas[0]
    else:
        out = ad.concat(adatas, join="inner", merge="same", uns_merge="same")
        out.obs_names_make_unique()
    if sp.issparse(out.X):
        out.X = out.X.tocsr()
    return out


def prepare_for_deg_scanpy(adata: ad.AnnData) -> ad.AnnData:
    work = adata.copy()
    if not sp.issparse(work.X):
        work.X = sp.csr_matrix(work.X)
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    return work


def rank_genes_groups_rapids(
    adata: ad.AnnData,
    groupby: str,
    method: str,
) -> pd.DataFrame:
    import rapids_singlecell as rsc

    if method == "logreg":
        raise ValueError("RAPIDS rank_genes_groups does not use method='logreg' in this wrapper")
    work = adata.copy()
    if not sp.issparse(work.X):
        work.X = sp.csr_matrix(work.X)
    rsc.get.anndata_to_GPU(work)
    rsc.pp.normalize_total(work, target_sum=1e4)
    rsc.pp.log1p(work)
    rsc.tl.rank_genes_groups(work, groupby=groupby, method=method, use_raw=False)
    rsc.get.anndata_to_CPU(work)
    return sc.get.rank_genes_groups_df(work, group=None)


def rank_genes_groups_scanpy(
    adata: ad.AnnData,
    groupby: str,
    method: str,
) -> pd.DataFrame:
    work = prepare_for_deg_scanpy(adata)
    sc.tl.rank_genes_groups(work, groupby=groupby, method=method, use_raw=False)
    return sc.get.rank_genes_groups_df(work, group=None)


def rank_genes_groups_table(
    adata: ad.AnnData,
    groupby: str,
    method: str,
    backend: str,
) -> tuple[pd.DataFrame, str]:
    if backend in {"auto", "rapids"}:
        try:
            deg = rank_genes_groups_rapids(adata, groupby=groupby, method=method)
            return deg, "rapids"
        except Exception as error:
            if backend == "rapids":
                raise
            print(f"  RAPIDS DEG failed, falling back to Scanpy CPU. Error: {error}")
    deg = rank_genes_groups_scanpy(adata, groupby=groupby, method=method)
    return deg, "scanpy"


def write_rank_genes_tables(
    adata: ad.AnnData,
    groupby: str,
    target: str,
    output_dir: Path,
    method: str,
    top_n: int,
    backend: str,
) -> None:
    label = sanitize_label(target)
    deg, used_backend = rank_genes_groups_table(adata, groupby=groupby, method=method, backend=backend)
    deg.insert(0, "target_author_class", target)
    deg.insert(1, "deg_backend", used_backend)
    deg.to_csv(output_dir / f"{label}.cell_leiden_deg.all.csv", index=False)

    top = (
        deg.sort_values(["group", "pvals_adj", "scores"], ascending=[True, True, False])
        .groupby("group", group_keys=False, observed=True)
        .head(top_n)
        .reset_index(drop=True)
    )
    top.to_csv(output_dir / f"{label}.cell_leiden_deg.top{top_n}.csv", index=False)


def cell_counts_table(meta: pd.DataFrame, targets: dict[str, list[str]], author_key: str, leiden_key: str) -> pd.DataFrame:
    rows = []
    for target, clusters in targets.items():
        for leiden in clusters:
            in_leiden = meta[leiden_key].astype(str).eq(str(leiden))
            in_author = meta[author_key].astype(str).eq(target)
            rows.append(
                {
                    "target_author_class": target,
                    "cell_leiden": str(leiden),
                    "n_leiden_total": int(in_leiden.sum()),
                    "n_author_and_leiden": int((in_author & in_leiden).sum()),
                    "fraction_author_within_leiden": float((in_author & in_leiden).sum() / max(in_leiden.sum(), 1)),
                }
            )
    return pd.DataFrame(rows)


def select_target_meta(
    obs: pd.DataFrame,
    target: str,
    clusters: list[str],
    author_key: str,
    leiden_key: str,
    selection_mode: str,
) -> pd.DataFrame:
    leiden = obs[leiden_key].astype(str)
    mask = leiden.isin([str(cluster) for cluster in clusters])
    if selection_mode == "author_and_leiden":
        mask &= obs[author_key].astype(str).eq(target)
    return obs.loc[mask].copy()


def maybe_downsample(meta: pd.DataFrame, max_cells: int | None, seed: int) -> pd.DataFrame:
    if max_cells is None or meta.shape[0] <= max_cells:
        return meta
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(meta.shape[0], size=max_cells, replace=False))
    return meta.iloc[selected].copy()


def save_section_expression_h5ads(
    obs: pd.DataFrame,
    raw_dir: Path,
    concat_h5ad: Path,
    inference_dir: Path,
    output_dir: Path,
    sections: list[str],
    keep_filtered_genes: bool,
    obsm_keys: list[str],
    imputation_layers: list[str],
    attach_obsm: bool,
    attach_imputation: bool,
    max_section_cells: int | None,
    random_seed: int,
) -> pd.DataFrame:
    section_dir = output_dir / "section_expression_h5ad"
    section_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for section in sections:
        section_meta = obs[obs["brain_section_label"].astype(str).eq(section)].copy()
        if section_meta.empty:
            rows.append({"brain_section_label": section, "n_cells": 0, "output_h5ad": ""})
            print(f"[section] {section}: no cells")
            continue
        section_meta = maybe_downsample(section_meta, max_section_cells, random_seed)
        print(f"[section] {section}: loading {section_meta.shape[0]:,} cells")
        adata = read_expression_subset(raw_dir, section_meta, keep_filtered_genes)
        attached = attach_model_outputs(
            adata,
            concat_h5ad=concat_h5ad,
            obsm_keys=obsm_keys,
            inference_dir=inference_dir,
            imputation_layers=imputation_layers,
            attach_obsm=attach_obsm,
            attach_imputation=attach_imputation,
        )
        adata.write_h5ad(section_dir / f"{section}.expression_with_leiden.h5ad", compression="gzip")
        rows.append(
            {
                "brain_section_label": section,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "attached_obsm": ";".join(attached["obsm"]),
                "attached_imputation_layers": ";".join(attached["imputation_layers"]),
                "output_h5ad": str(section_dir / f"{section}.expression_with_leiden.h5ad"),
            }
        )
        del adata
        gc.collect()
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "section_expression_h5ad.summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"concat={args.concat_h5ad}")
    obs = load_concat_obs(args.concat_h5ad)
    for required in [args.author_key, args.leiden_key, "source_stem", "raw_obs_name", "brain_section_label"]:
        if required not in obs.columns:
            raise KeyError(f"concat obs is missing {required!r}")
    obs[args.leiden_key] = obs[args.leiden_key].astype(str)
    obs[args.author_key] = obs[args.author_key].astype(str)
    print(f"concat obs loaded: {obs.shape[0]:,} cells")

    mapping, target_clusters = load_leiden_mapping(args.leiden_author_map, args.targets)
    mapping[mapping["matched_author_class"].isin(args.targets)].to_csv(
        args.output_dir / "target_cell_leiden_author_mapping.csv",
        index=False,
    )
    counts = cell_counts_table(obs, target_clusters, args.author_key, args.leiden_key)
    counts.to_csv(args.output_dir / "target_cell_leiden_counts.csv", index=False)
    print(counts.to_string(index=False))

    if args.dry_run:
        print("dry-run only, no expression or DEG files were written")
        return

    save_section_expression_h5ads(
        obs=obs,
        raw_dir=args.raw_dir,
        concat_h5ad=args.concat_h5ad,
        inference_dir=args.inference_dir,
        output_dir=args.output_dir,
        sections=args.sections,
        keep_filtered_genes=args.keep_filtered_genes,
        obsm_keys=args.section_obsm_keys,
        imputation_layers=args.imputation_layers,
        attach_obsm=not args.no_section_obsm,
        attach_imputation=not args.no_attach_imputation,
        max_section_cells=args.max_section_cells,
        random_seed=args.random_seed,
    )

    deg_dir = args.output_dir / "deg_tables"
    h5ad_dir = args.output_dir / "target_expression_h5ad"
    deg_dir.mkdir(parents=True, exist_ok=True)
    if args.save_target_h5ad:
        h5ad_dir.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        clusters = target_clusters[target]
        if not clusters:
            print(f"[target] {target}: no mapped cell_leiden clusters")
            continue
        target_meta = select_target_meta(
            obs,
            target=target,
            clusters=clusters,
            author_key=args.author_key,
            leiden_key=args.leiden_key,
            selection_mode=args.selection_mode,
        )
        target_meta = maybe_downsample(target_meta, args.max_cells_per_target, args.random_seed)
        print(f"[target] {target}: {target_meta.shape[0]:,} cells, clusters={clusters}")
        if target_meta.empty:
            continue

        adata = read_expression_subset(args.raw_dir, target_meta, args.keep_filtered_genes)
        group_counts = adata.obs[args.leiden_key].astype(str).value_counts()
        keep_groups = group_counts[group_counts >= args.min_cells_per_group].index.astype(str).tolist()
        adata = adata[adata.obs[args.leiden_key].astype(str).isin(keep_groups)].copy()
        adata.obs[args.leiden_key] = adata.obs[args.leiden_key].astype(str).astype("category")
        print(f"  DEG cells={adata.n_obs:,}, groups kept={keep_groups}")

        if args.save_target_h5ad:
            attach_model_outputs(
                adata,
                concat_h5ad=args.concat_h5ad,
                obsm_keys=args.section_obsm_keys,
                inference_dir=args.inference_dir,
                imputation_layers=args.imputation_layers,
                attach_obsm=not args.no_section_obsm,
                attach_imputation=not args.no_attach_imputation,
            )
            adata.write_h5ad(
                h5ad_dir / f"{sanitize_label(target)}.expression_with_leiden.h5ad",
                compression="gzip",
            )
        if adata.obs[args.leiden_key].nunique() < 2:
            print(f"  skip DEG for {target}: fewer than two Leiden groups after filtering")
        else:
            write_rank_genes_tables(
                adata,
                groupby=args.leiden_key,
                target=target,
                output_dir=deg_dir,
                method=args.method,
                top_n=args.top_n,
                backend=args.deg_backend,
            )
        del adata
        gc.collect()

    print(f"\nDone: {args.output_dir}")


if __name__ == "__main__":
    main()
