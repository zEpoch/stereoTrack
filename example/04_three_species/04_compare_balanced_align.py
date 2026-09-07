"""Paired evaluation of the balanced baseline and cell-aligned model embeddings."""

from __future__ import annotations

import argparse
import gc
from itertools import combinations
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_ROOT = Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack")
EMBEDDING_KEYS = {
    "cell": "cell_embedding",
    "niche": "niche_embedding",
}
LABEL_KEYS = {
    "cell": ["cell_class_broad", "cell_class", "cell_subclass", "cell_type"],
    "niche": ["layer", "region_main", "region"],
}
INVALID_LABELS = {"", "na", "nan", "none", "unknown", "unassigned"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare balanced and aligned StereoTrack embeddings")
    parser.add_argument(
        "--balanced-dir",
        type=Path,
        default=DEFAULT_ROOT / "out/04_three_species_mae_v1_train_balanced_v1/inference_adatas",
    )
    parser.add_argument(
        "--align-dir",
        type=Path,
        default=DEFAULT_ROOT / "out/04_three_species_mae_v1_train_align_v1/inference_adatas",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "out/04_three_species_balanced_vs_align",
    )
    parser.add_argument("--max-cells-per-file", type=int, default=2000)
    parser.add_argument("--max-cells-per-species", type=int, default=30000)
    parser.add_argument("--min-cells-per-group", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--all-cells-streaming",
        action="store_true",
        help="Use every cell with exact streaming statistics instead of concatenating sampled embeddings.",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    return parser.parse_args()


def allocate_quota(caps: np.ndarray, quota: int, weights: np.ndarray) -> np.ndarray:
    caps = np.asarray(caps, dtype=np.int64)
    if quota <= 0 or quota >= int(caps.sum()):
        return caps.copy()

    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(caps > 0, weights, 0.0)
    raw = quota * weights / np.maximum(weights.sum(), 1e-12)
    assigned = np.minimum(np.floor(raw).astype(np.int64), caps)
    remainder = int(quota - assigned.sum())
    order = np.argsort(-(raw - np.floor(raw)))

    while remainder > 0:
        changed = False
        for index in order:
            if assigned[index] < caps[index]:
                assigned[index] += 1
                remainder -= 1
                changed = True
                if remainder == 0:
                    break
        if not changed:
            break
    return assigned


def infer_species(file_name: str) -> str:
    return file_name.split("_", 1)[0].lower()


def collect_plan(args: argparse.Namespace) -> list[dict]:
    balanced_files = {path.name: path for path in args.balanced_dir.glob("*.h5ad")}
    align_files = {path.name: path for path in args.align_dir.glob("*.h5ad")}
    common_names = sorted(set(balanced_files).intersection(align_files))
    if not common_names:
        raise FileNotFoundError("No paired h5ad files were found")
    if set(balanced_files) != set(align_files):
        missing_align = sorted(set(balanced_files).difference(align_files))
        missing_balanced = sorted(set(align_files).difference(balanced_files))
        raise RuntimeError(
            f"Inference file sets differ: missing_align={missing_align[:5]}, "
            f"missing_balanced={missing_balanced[:5]}"
        )

    plan = []
    for name in common_names:
        with h5py.File(balanced_files[name], "r") as handle:
            missing = [key for key in EMBEDDING_KEYS.values() if f"obsm/{key}" not in handle]
            if missing:
                raise KeyError(f"{name}: missing embeddings {missing}")
            n_obs = int(handle[f"obsm/{EMBEDDING_KEYS['cell']}"].shape[0])
        species = infer_species(name)
        cap = n_obs if args.max_cells_per_file <= 0 else min(n_obs, args.max_cells_per_file)
        plan.append(
            {
                "name": name,
                "balanced_path": balanced_files[name],
                "align_path": align_files[name],
                "species": species,
                "n_obs": n_obs,
                "cap": int(cap),
                "keep_n": int(cap),
            }
        )

    for species in sorted({row["species"] for row in plan}):
        indices = [index for index, row in enumerate(plan) if row["species"] == species]
        caps = np.array([plan[index]["cap"] for index in indices], dtype=np.int64)
        weights = np.array([plan[index]["n_obs"] for index in indices], dtype=np.float64)
        assigned = allocate_quota(caps, args.max_cells_per_species, weights)
        for index, keep_n in zip(indices, assigned):
            plan[index]["keep_n"] = int(keep_n)

    return plan


def clean_label(values: pd.Series) -> np.ndarray:
    return values.astype("string").fillna("Unknown").astype(str).to_numpy()


def read_paired_sample(args: argparse.Namespace, plan: list[dict]):
    rng = np.random.default_rng(args.random_seed)
    metadata_parts = []
    embedding_parts = {
        model: {representation: [] for representation in EMBEDDING_KEYS}
        for model in ["balanced", "align"]
    }

    for file_number, row in enumerate(plan, start=1):
        keep_n = row["keep_n"]
        if keep_n <= 0:
            continue

        balanced = ad.read_h5ad(row["balanced_path"])
        aligned = ad.read_h5ad(row["align_path"])
        if balanced.n_obs != aligned.n_obs:
            raise ValueError(f"{row['name']}: cell counts differ between models")
        if not np.array_equal(balanced.obs_names.to_numpy(), aligned.obs_names.to_numpy()):
            raise ValueError(f"{row['name']}: obs_names differ between models")

        if keep_n < balanced.n_obs:
            selected = np.sort(rng.choice(balanced.n_obs, size=keep_n, replace=False))
        else:
            selected = np.arange(balanced.n_obs)

        metadata = pd.DataFrame(
            {
                "file": row["name"],
                "cell_index": selected,
                "obs_name": balanced.obs_names.to_numpy()[selected],
                "species": row["species"],
            }
        )
        for label_key in sorted({key for keys in LABEL_KEYS.values() for key in keys}):
            if label_key in balanced.obs.columns:
                metadata[label_key] = clean_label(balanced.obs[label_key].iloc[selected])
            else:
                metadata[label_key] = "Unknown"
        metadata_parts.append(metadata)

        for representation, embedding_key in EMBEDDING_KEYS.items():
            embedding_parts["balanced"][representation].append(
                np.asarray(balanced.obsm[embedding_key])[selected].astype(np.float32)
            )
            embedding_parts["align"][representation].append(
                np.asarray(aligned.obsm[embedding_key])[selected].astype(np.float32)
            )

        print(
            f"[{file_number}/{len(plan)}] {row['name']}: "
            f"keep={len(selected)}/{balanced.n_obs}, species={row['species']}"
        )
        del balanced, aligned
        gc.collect()

    metadata = pd.concat(metadata_parts, ignore_index=True)
    embeddings = {
        model: {
            representation: np.concatenate(parts, axis=0)
            for representation, parts in model_parts.items()
        }
        for model, model_parts in embedding_parts.items()
    }
    return metadata, embeddings


def unit_normalize(embedding: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / np.maximum(norms, 1e-12)


def one_health_row(embedding: np.ndarray, normalization: str) -> dict:
    x = np.asarray(embedding, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    norms = np.linalg.norm(x, axis=1)
    if normalization == "unit":
        x = unit_normalize(x)

    centered = x - x.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=1)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    variance_sum = float(eigenvalues.sum())
    probabilities = eigenvalues / max(variance_sum, 1e-12)
    positive = probabilities > 0
    effective_rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    participation_ratio = float(variance_sum**2 / max(float(np.square(eigenvalues).sum()), 1e-12))
    descending = np.sort(probabilities)[::-1]
    near_zero_threshold = max(float(np.median(std)) * 0.01, 1e-8)

    return {
        "n_cells": int(x.shape[0]),
        "n_dimensions": int(x.shape[1]),
        "normalization": normalization,
        "mean_embedding_norm": float(norms.mean()),
        "median_dimension_std": float(np.median(std)),
        "min_dimension_std": float(std.min()),
        "near_zero_dimensions": int((std < near_zero_threshold).sum()),
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / x.shape[1],
        "participation_ratio": participation_ratio,
        "top1_variance_ratio": float(descending[:1].sum()),
        "top5_variance_ratio": float(descending[:5].sum()),
    }


def embedding_health(metadata: pd.DataFrame, embeddings: dict) -> pd.DataFrame:
    rows = []
    scopes = {"pooled": np.ones(len(metadata), dtype=bool)}
    scopes.update(
        {
            f"species:{species}": metadata["species"].eq(species).to_numpy()
            for species in sorted(metadata["species"].unique())
        }
    )
    for model, model_embeddings in embeddings.items():
        for representation, embedding in model_embeddings.items():
            for scope, mask in scopes.items():
                for normalization in ["raw", "unit"]:
                    row = one_health_row(embedding[mask], normalization)
                    row.update({"model": model, "representation": representation, "scope": scope})
                    rows.append(row)
    return pd.DataFrame(rows)


def valid_label_mask(values: pd.Series) -> np.ndarray:
    lowered = values.fillna("Unknown").astype(str).str.strip().str.lower()
    return ~lowered.isin(INVALID_LABELS).to_numpy()


def build_group_centroids(
    metadata: pd.DataFrame,
    embedding: np.ndarray,
    label_key: str,
    min_cells: int,
):
    valid = valid_label_mask(metadata[label_key])
    unit_embedding = unit_normalize(np.asarray(embedding, dtype=np.float64))
    groups = {}
    compactness_rows = []

    valid_indices = np.flatnonzero(valid)
    grouped = metadata.iloc[valid_indices].groupby(["species", label_key], observed=True).indices
    for (species, label), relative_indices in grouped.items():
        indices = valid_indices[np.asarray(relative_indices, dtype=np.int64)]
        if len(indices) < min_cells:
            continue
        cells = unit_embedding[indices]
        centroid = cells.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        compactness = float(np.mean(1.0 - cells @ centroid))
        groups[(str(species), str(label))] = {
            "centroid": centroid,
            "n_cells": int(len(indices)),
            "compactness": compactness,
        }
        compactness_rows.append(
            {
                "species": str(species),
                "label": str(label),
                "n_cells": int(len(indices)),
                "mean_cosine_distance_to_centroid": compactness,
            }
        )
    return groups, compactness_rows


def cross_species_label_metrics(
    metadata: pd.DataFrame,
    embeddings: dict,
    min_cells: int,
):
    detail_rows = []
    compactness_rows = []

    for model, model_embeddings in embeddings.items():
        for representation, embedding in model_embeddings.items():
            for label_key in LABEL_KEYS[representation]:
                groups, current_compactness = build_group_centroids(
                    metadata, embedding, label_key, min_cells
                )
                for row in current_compactness:
                    row.update(
                        {"model": model, "representation": representation, "label_key": label_key}
                    )
                    compactness_rows.append(row)

                species_list = sorted({species for species, _ in groups})
                for species_from in species_list:
                    source_labels = sorted(label for species, label in groups if species == species_from)
                    for species_to in species_list:
                        if species_from == species_to:
                            continue
                        target_labels = sorted(label for species, label in groups if species == species_to)
                        if not target_labels:
                            continue
                        target_centroids = np.stack(
                            [groups[(species_to, label)]["centroid"] for label in target_labels]
                        )
                        target_lookup = {label: index for index, label in enumerate(target_labels)}

                        for label in source_labels:
                            if label not in target_lookup:
                                continue
                            source = groups[(species_from, label)]
                            distances = 1.0 - target_centroids @ source["centroid"]
                            nearest_index = int(np.argmin(distances))
                            same_index = target_lookup[label]
                            wrong_mask = np.ones(len(target_labels), dtype=bool)
                            wrong_mask[same_index] = False
                            nearest_wrong = float(distances[wrong_mask].min()) if wrong_mask.any() else np.nan
                            same_distance = float(distances[same_index])
                            detail_rows.append(
                                {
                                    "model": model,
                                    "representation": representation,
                                    "label_key": label_key,
                                    "species_from": species_from,
                                    "species_to": species_to,
                                    "label": label,
                                    "n_cells_from": source["n_cells"],
                                    "n_cells_to": groups[(species_to, label)]["n_cells"],
                                    "nearest_label": target_labels[nearest_index],
                                    "nearest_label_match": target_labels[nearest_index] == label,
                                    "same_label_cosine_distance": same_distance,
                                    "nearest_wrong_cosine_distance": nearest_wrong,
                                    "same_vs_wrong_margin": nearest_wrong - same_distance,
                                }
                            )

    detail = pd.DataFrame(detail_rows)
    compactness = pd.DataFrame(compactness_rows)
    return detail, compactness


def summarize_label_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["model", "representation", "label_key"]
    for keys, group in detail.groupby(group_columns, observed=True):
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_cross_species_queries": int(len(group)),
                "nearest_label_match_accuracy": float(group["nearest_label_match"].mean()),
                "mean_same_label_cosine_distance": float(group["same_label_cosine_distance"].mean()),
                "median_same_label_cosine_distance": float(group["same_label_cosine_distance"].median()),
                "mean_same_vs_wrong_margin": float(group["same_vs_wrong_margin"].mean()),
                "positive_margin_fraction": float((group["same_vs_wrong_margin"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_compactness(compactness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["model", "representation", "label_key"]
    for keys, group in compactness.groupby(group_columns, observed=True):
        weights = group["n_cells"].to_numpy(dtype=np.float64)
        values = group["mean_cosine_distance_to_centroid"].to_numpy(dtype=np.float64)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_species_label_groups": int(len(group)),
                "weighted_mean_within_group_cosine_distance": float(np.average(values, weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def paired_model_comparison(
    table: pd.DataFrame,
    index_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    output = None
    for value in value_columns:
        pivot = table.pivot_table(index=index_columns, columns="model", values=value, aggfunc="first")
        pivot = pivot.rename(columns=lambda model: f"{value}_{model}")
        if {f"{value}_balanced", f"{value}_align"}.issubset(pivot.columns):
            pivot[f"{value}_align_minus_balanced"] = (
                pivot[f"{value}_align"] - pivot[f"{value}_balanced"]
            )
        output = pivot if output is None else output.join(pivot, how="outer")
    return output.reset_index()


def global_species_distances(metadata: pd.DataFrame, embeddings: dict) -> pd.DataFrame:
    rows = []
    species_list = sorted(metadata["species"].unique())
    for model, model_embeddings in embeddings.items():
        for representation, embedding in model_embeddings.items():
            unit_embedding = unit_normalize(np.asarray(embedding, dtype=np.float64))
            centroids = {}
            for species in species_list:
                centroid = unit_embedding[metadata["species"].eq(species).to_numpy()].mean(axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
                centroids[species] = centroid
            for species_a, species_b in combinations(species_list, 2):
                rows.append(
                    {
                        "model": model,
                        "representation": representation,
                        "species_a": species_a,
                        "species_b": species_b,
                        "cosine_distance": float(1.0 - centroids[species_a] @ centroids[species_b]),
                    }
                )
    return pd.DataFrame(rows)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    x_cov = x.T @ x
    y_cov = y.T @ y
    numerator = float(np.square(cross).sum())
    denominator = float(np.sqrt(np.square(x_cov).sum() * np.square(y_cov).sum()))
    return numerator / max(denominator, 1e-12)


def paired_cka(metadata: pd.DataFrame, embeddings: dict) -> pd.DataFrame:
    rows = []
    scopes = {"pooled": np.ones(len(metadata), dtype=bool)}
    scopes.update(
        {
            f"species:{species}": metadata["species"].eq(species).to_numpy()
            for species in sorted(metadata["species"].unique())
        }
    )
    for representation in EMBEDDING_KEYS:
        for scope, mask in scopes.items():
            rows.append(
                {
                    "representation": representation,
                    "scope": scope,
                    "n_cells": int(mask.sum()),
                    "linear_cka_balanced_vs_align": linear_cka(
                        embeddings["balanced"][representation][mask],
                        embeddings["align"][representation][mask],
                    ),
                }
            )
    return pd.DataFrame(rows)


class MomentAccumulator:
    def __init__(self):
        self.n = 0
        self.sum = None
        self.cross = None
        self.norm_sum = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[0] == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values found in an embedding")
        if self.sum is None:
            self.sum = np.zeros(values.shape[1], dtype=np.float64)
            self.cross = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
        self.n += int(values.shape[0])
        self.sum += values.sum(axis=0)
        self.cross += values.T @ values
        self.norm_sum += float(np.linalg.norm(values, axis=1).sum())

    def health_row(self, normalization: str) -> dict:
        if self.n < 2:
            raise ValueError("At least two cells are required for embedding health statistics")
        centered_cross = self.cross - np.outer(self.sum, self.sum) / self.n
        covariance = centered_cross / (self.n - 1)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        variance_sum = float(eigenvalues.sum())
        probabilities = eigenvalues / max(variance_sum, 1e-12)
        positive = probabilities > 0
        effective_rank = float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        participation_ratio = float(
            variance_sum**2 / max(float(np.square(eigenvalues).sum()), 1e-12)
        )
        descending = np.sort(probabilities)[::-1]
        near_zero_threshold = max(float(np.median(std)) * 0.01, 1e-8)
        return {
            "n_cells": self.n,
            "n_dimensions": int(len(self.sum)),
            "normalization": normalization,
            "mean_embedding_norm": self.norm_sum / self.n,
            "median_dimension_std": float(np.median(std)),
            "min_dimension_std": float(std.min()),
            "near_zero_dimensions": int((std < near_zero_threshold).sum()),
            "effective_rank": effective_rank,
            "effective_rank_ratio": effective_rank / len(self.sum),
            "participation_ratio": participation_ratio,
            "top1_variance_ratio": float(descending[:1].sum()),
            "top5_variance_ratio": float(descending[:5].sum()),
        }


class CKAAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_x = None
        self.sum_y = None
        self.xtx = None
        self.yty = None
        self.xty = None

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("Paired embeddings have different shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite values found in paired embeddings")
        if self.sum_x is None:
            dimensions = x.shape[1]
            self.sum_x = np.zeros(dimensions, dtype=np.float64)
            self.sum_y = np.zeros(dimensions, dtype=np.float64)
            self.xtx = np.zeros((dimensions, dimensions), dtype=np.float64)
            self.yty = np.zeros((dimensions, dimensions), dtype=np.float64)
            self.xty = np.zeros((dimensions, dimensions), dtype=np.float64)
        self.n += int(x.shape[0])
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.xtx += x.T @ x
        self.yty += y.T @ y
        self.xty += x.T @ y

    def value(self) -> float:
        centered_xtx = self.xtx - np.outer(self.sum_x, self.sum_x) / self.n
        centered_yty = self.yty - np.outer(self.sum_y, self.sum_y) / self.n
        centered_xty = self.xty - np.outer(self.sum_x, self.sum_y) / self.n
        numerator = float(np.square(centered_xty).sum())
        denominator = float(
            np.sqrt(np.square(centered_xtx).sum() * np.square(centered_yty).sum())
        )
        return numerator / max(denominator, 1e-12)


def update_label_groups(
    group_sums: dict,
    model: str,
    representation: str,
    unit_embedding: np.ndarray,
    label_values: dict[str, np.ndarray],
    species: str,
) -> None:
    for label_key in LABEL_KEYS[representation]:
        values = label_values[label_key]
        lowered = np.char.lower(np.char.strip(values.astype(str)))
        valid = ~np.isin(lowered, list(INVALID_LABELS))
        if not valid.any():
            continue
        codes, labels = pd.factorize(values[valid], sort=False)
        counts = np.bincount(codes, minlength=len(labels)).astype(np.int64)
        indicator = sp.csr_matrix(
            (np.ones(len(codes), dtype=np.float64), (codes, np.arange(len(codes)))),
            shape=(len(labels), len(codes)),
        )
        sums = np.asarray(indicator @ unit_embedding[valid])
        for index, label in enumerate(labels):
            key = (model, representation, label_key, species, str(label))
            if key not in group_sums:
                group_sums[key] = [0, np.zeros(unit_embedding.shape[1], dtype=np.float64)]
            group_sums[key][0] += int(counts[index])
            group_sums[key][1] += sums[index]


def finalize_streaming_labels(group_sums: dict, min_cells: int):
    grouped_models = {}
    compactness_rows = []
    for (model, representation, label_key, species, label), (n_cells, vector_sum) in group_sums.items():
        if n_cells < min_cells:
            continue
        vector_norm = float(np.linalg.norm(vector_sum))
        centroid = vector_sum / max(vector_norm, 1e-12)
        compactness = 1.0 - vector_norm / n_cells
        grouped_models.setdefault((model, representation, label_key), {})[(species, label)] = {
            "centroid": centroid,
            "n_cells": n_cells,
        }
        compactness_rows.append(
            {
                "model": model,
                "representation": representation,
                "label_key": label_key,
                "species": species,
                "label": label,
                "n_cells": n_cells,
                "mean_cosine_distance_to_centroid": compactness,
            }
        )

    detail_rows = []
    for (model, representation, label_key), groups in grouped_models.items():
        species_list = sorted({species for species, _ in groups})
        for species_from in species_list:
            source_labels = sorted(label for species, label in groups if species == species_from)
            for species_to in species_list:
                if species_from == species_to:
                    continue
                target_labels = sorted(label for species, label in groups if species == species_to)
                if not target_labels:
                    continue
                target_centroids = np.stack(
                    [groups[(species_to, label)]["centroid"] for label in target_labels]
                )
                target_lookup = {label: index for index, label in enumerate(target_labels)}
                for label in source_labels:
                    if label not in target_lookup:
                        continue
                    source = groups[(species_from, label)]
                    distances = 1.0 - target_centroids @ source["centroid"]
                    nearest_index = int(np.argmin(distances))
                    same_index = target_lookup[label]
                    wrong_mask = np.ones(len(target_labels), dtype=bool)
                    wrong_mask[same_index] = False
                    nearest_wrong = float(distances[wrong_mask].min()) if wrong_mask.any() else np.nan
                    same_distance = float(distances[same_index])
                    detail_rows.append(
                        {
                            "model": model,
                            "representation": representation,
                            "label_key": label_key,
                            "species_from": species_from,
                            "species_to": species_to,
                            "label": label,
                            "n_cells_from": source["n_cells"],
                            "n_cells_to": groups[(species_to, label)]["n_cells"],
                            "nearest_label": target_labels[nearest_index],
                            "nearest_label_match": target_labels[nearest_index] == label,
                            "same_label_cosine_distance": same_distance,
                            "nearest_wrong_cosine_distance": nearest_wrong,
                            "same_vs_wrong_margin": nearest_wrong - same_distance,
                        }
                    )
    return pd.DataFrame(detail_rows), pd.DataFrame(compactness_rows)


def save_comparison_tables(
    output_dir: Path,
    health: pd.DataFrame,
    detail: pd.DataFrame,
    compactness: pd.DataFrame,
    species_distances: pd.DataFrame,
    cka: pd.DataFrame,
) -> None:
    health.to_csv(output_dir / "embedding_health.csv", index=False)
    paired_model_comparison(
        health,
        ["representation", "scope", "normalization"],
        [
            "effective_rank",
            "effective_rank_ratio",
            "participation_ratio",
            "top1_variance_ratio",
            "top5_variance_ratio",
            "median_dimension_std",
            "near_zero_dimensions",
        ],
    ).to_csv(output_dir / "embedding_health_comparison.csv", index=False)

    detail.to_csv(output_dir / "cross_species_label_matching_detail.csv", index=False)
    label_summary = summarize_label_metrics(detail)
    label_summary.to_csv(output_dir / "cross_species_label_matching_summary.csv", index=False)
    paired_model_comparison(
        label_summary,
        ["representation", "label_key"],
        [
            "nearest_label_match_accuracy",
            "mean_same_label_cosine_distance",
            "mean_same_vs_wrong_margin",
            "positive_margin_fraction",
        ],
    ).to_csv(output_dir / "cross_species_label_matching_comparison.csv", index=False)

    compactness.to_csv(output_dir / "label_compactness_detail.csv", index=False)
    compactness_summary = summarize_compactness(compactness)
    compactness_summary.to_csv(output_dir / "label_compactness_summary.csv", index=False)
    paired_model_comparison(
        compactness_summary,
        ["representation", "label_key"],
        ["weighted_mean_within_group_cosine_distance"],
    ).to_csv(output_dir / "label_compactness_comparison.csv", index=False)

    species_distances.to_csv(output_dir / "global_species_centroid_distances.csv", index=False)
    paired_model_comparison(
        species_distances,
        ["representation", "species_a", "species_b"],
        ["cosine_distance"],
    ).to_csv(output_dir / "global_species_centroid_distances_comparison.csv", index=False)
    cka.to_csv(output_dir / "paired_linear_cka.csv", index=False)


def run_all_cells_streaming(args: argparse.Namespace, plan: list[dict]) -> None:
    health_accumulators = {}
    cka_accumulators = {}
    label_group_sums = {}
    count_rows = []
    all_label_keys = sorted({key for keys in LABEL_KEYS.values() for key in keys})

    for file_number, row in enumerate(plan, start=1):
        obs_adata = ad.read_h5ad(row["balanced_path"], backed="r")
        label_values = {
            label_key: (
                obs_adata.obs[label_key].astype("string").fillna("Unknown").astype(str).to_numpy()
                if label_key in obs_adata.obs.columns
                else np.full(obs_adata.n_obs, "Unknown", dtype=object)
            )
            for label_key in all_label_keys
        }
        obs_adata.file.close()

        with h5py.File(row["balanced_path"], "r") as balanced_handle, h5py.File(
            row["align_path"], "r"
        ) as align_handle:
            n_cells = int(balanced_handle[f"obsm/{EMBEDDING_KEYS['cell']}"].shape[0])
            if int(align_handle[f"obsm/{EMBEDDING_KEYS['cell']}"].shape[0]) != n_cells:
                raise ValueError(f"{row['name']}: cell counts differ between models")

            for start in range(0, n_cells, args.chunk_size):
                end = min(start + args.chunk_size, n_cells)
                chunk_labels = {key: values[start:end] for key, values in label_values.items()}
                chunk_embeddings = {"balanced": {}, "align": {}}
                for model, handle in [("balanced", balanced_handle), ("align", align_handle)]:
                    for representation, embedding_key in EMBEDDING_KEYS.items():
                        values = np.asarray(handle[f"obsm/{embedding_key}"][start:end], dtype=np.float32)
                        if not np.isfinite(values).all():
                            raise ValueError(f"{row['name']}: non-finite {model} {embedding_key}")
                        chunk_embeddings[model][representation] = values
                        unit_values = unit_normalize(values.astype(np.float64))
                        for scope in ["pooled", f"species:{row['species']}"]:
                            for normalization, current_values in [("raw", values), ("unit", unit_values)]:
                                key = (model, representation, scope, normalization)
                                health_accumulators.setdefault(key, MomentAccumulator()).update(current_values)
                        update_label_groups(
                            label_group_sums,
                            model,
                            representation,
                            unit_values,
                            chunk_labels,
                            row["species"],
                        )

                for representation in EMBEDDING_KEYS:
                    for scope in ["pooled", f"species:{row['species']}"]:
                        key = (representation, scope)
                        cka_accumulators.setdefault(key, CKAAccumulator()).update(
                            chunk_embeddings["balanced"][representation],
                            chunk_embeddings["align"][representation],
                        )

        count_rows.append({"file": row["name"], "species": row["species"], "n_cells": n_cells})
        print(f"[{file_number}/{len(plan)}] {row['name']}: streamed {n_cells} cells")
        del label_values
        gc.collect()

    pd.DataFrame(count_rows).to_csv(args.output_dir / "all_cell_counts.csv", index=False)

    health_rows = []
    for (model, representation, scope, normalization), accumulator in health_accumulators.items():
        row = accumulator.health_row(normalization)
        row.update({"model": model, "representation": representation, "scope": scope})
        health_rows.append(row)
    health = pd.DataFrame(health_rows)

    detail, compactness = finalize_streaming_labels(
        label_group_sums, args.min_cells_per_group
    )

    species_distance_rows = []
    species_list = sorted({row["species"] for row in plan})
    for model in ["balanced", "align"]:
        for representation in EMBEDDING_KEYS:
            centroids = {}
            for species in species_list:
                accumulator = health_accumulators[(model, representation, f"species:{species}", "unit")]
                centroid = accumulator.sum / accumulator.n
                centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
                centroids[species] = centroid
            for species_a, species_b in combinations(species_list, 2):
                species_distance_rows.append(
                    {
                        "model": model,
                        "representation": representation,
                        "species_a": species_a,
                        "species_b": species_b,
                        "cosine_distance": float(1.0 - centroids[species_a] @ centroids[species_b]),
                    }
                )
    species_distances = pd.DataFrame(species_distance_rows)

    cka = pd.DataFrame(
        [
            {
                "representation": representation,
                "scope": scope,
                "n_cells": accumulator.n,
                "linear_cka_balanced_vs_align": accumulator.value(),
            }
            for (representation, scope), accumulator in cka_accumulators.items()
        ]
    )
    save_comparison_tables(args.output_dir, health, detail, compactness, species_distances, cka)
    print(f"All-cell streaming comparison complete: {args.output_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plan = collect_plan(args)
    if args.all_cells_streaming:
        for row in plan:
            row["keep_n"] = row["n_obs"]
    plan_df = pd.DataFrame(plan).drop(columns=["balanced_path", "align_path"])
    plan_df.to_csv(args.output_dir / "sampling_plan.csv", index=False)
    print(
        "Sampling plan:",
        plan_df.groupby("species", observed=True)[["n_obs", "keep_n"]].sum().to_dict("index"),
    )

    if args.all_cells_streaming:
        run_all_cells_streaming(args, plan)
        return

    metadata, embeddings = read_paired_sample(args, plan)
    metadata.to_csv(args.output_dir / "sampled_obs_metadata.csv.gz", index=False, compression="gzip")

    health = embedding_health(metadata, embeddings)
    detail, compactness = cross_species_label_metrics(
        metadata, embeddings, args.min_cells_per_group
    )
    species_distances = global_species_distances(metadata, embeddings)
    cka = paired_cka(metadata, embeddings)
    save_comparison_tables(args.output_dir, health, detail, compactness, species_distances, cka)
    print(f"Comparison complete: {args.output_dir}")


if __name__ == "__main__":
    main()
