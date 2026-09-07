#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_INPUT_DIR = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/"
    "01_axolotl_artista_downstream_mae_v1_train/inference_full_adatas"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/"
    "01_axolotl_artista_downstream_mae_v1_train/dpi_label_transfer_sankey"
)

CELLTYPE_COLORMAP = {
    "cckIN": "#ff8400",
    "CMPN": "#ffe2df",
    "CP": "#1a00ab",
    "dpEX": "#00e9ff",
    "IMN": "#8a83c3",
    "MCG": "#808000",
    "mpEX": "#ff2940",
    "mpIN": "#f4f4d8",
    "MSN": "#957c85",
    "nptxEX": "#ae0059",
    "npyIN": "#ff88ca",
    "ntng1IN": "#84bd8b",
    "obNBL": "#004fb5",
    "Oligo": "#fac7c1",
    "reaEGC": "#ca0000",
    "ribEGC": "#af1127",
    "rIPC1": "#f5531e",
    "rIPC2": "#559cbe",
    "rIPC4": "#ffb6ff",
    "scgnIN": "#85b0ff",
    "sfrpEGC": "#f338f3",
    "sstIN": "#00ffa3",
    "tlNBL": "#9ea8d2",
    "Unknown": "#7da59c",
    "VLMC": "#faff00",
    "wntEGC": "#50c508",
    "WSN": "#00898d",
}


@dataclass
class DpiData:
    dpi: int
    files: list[Path]
    embedding: np.ndarray
    labels: np.ndarray
    obs_names: np.ndarray
    samples: np.ndarray


def parse_dpi_from_name(name: str, pattern: str = r"(?P<dpi>\d+)DPI") -> int:
    match = re.search(pattern, name)
    if match is None:
        raise ValueError(f"Cannot parse DPI from {name!r} with pattern {pattern!r}")
    if "dpi" in match.groupdict():
        return int(match.group("dpi"))
    return int(match.group(1))


def discover_h5ads(input_dir: Path, pattern: str = "*.h5ad") -> list[Path]:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No h5ad files found in {input_dir} with pattern={pattern!r}")
    return paths


def group_files_by_dpi(
    paths: Iterable[Path],
    dpi_pattern: str = r"(?P<dpi>\d+)DPI",
) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for path in paths:
        dpi = parse_dpi_from_name(path.stem, dpi_pattern)
        grouped.setdefault(dpi, []).append(path)
    return dict(sorted(grouped.items()))


def _subsample_indices(n: int, max_cells: int | None, seed: int) -> np.ndarray:
    if max_cells is None or n <= max_cells:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_cells, replace=False))


def load_dpi_data(
    files: list[Path],
    dpi: int,
    embedding_key: str = "niche_embedding",
    label_key: str = "Annotation",
    max_cells_per_file: int | None = None,
    seed: int = 0,
) -> DpiData:
    embeddings = []
    labels = []
    obs_names = []
    samples = []
    for file_idx, path in enumerate(files):
        adata = sc.read_h5ad(path)
        if embedding_key not in adata.obsm:
            raise KeyError(f"{path} does not contain adata.obsm[{embedding_key!r}]")
        if label_key not in adata.obs:
            raise KeyError(f"{path} does not contain adata.obs[{label_key!r}]")

        idx = _subsample_indices(adata.n_obs, max_cells_per_file, seed + file_idx)
        emb = np.asarray(adata.obsm[embedding_key])[idx]
        lab = adata.obs[label_key].astype(str).to_numpy()[idx]
        obs = adata.obs_names.to_numpy()[idx]

        valid = np.isfinite(emb).all(axis=1) & pd.notna(lab) & (lab != "nan")
        embeddings.append(emb[valid])
        labels.append(lab[valid])
        obs_names.append(obs[valid])
        samples.append(np.full(np.count_nonzero(valid), path.stem, dtype=object))

    if not embeddings:
        raise ValueError(f"No valid cells for DPI={dpi}")

    return DpiData(
        dpi=dpi,
        files=files,
        embedding=np.vstack(embeddings).astype(np.float32, copy=False),
        labels=np.concatenate(labels).astype(str),
        obs_names=np.concatenate(obs_names).astype(str),
        samples=np.concatenate(samples).astype(str),
    )


def load_all_dpi_data(
    input_dir: Path = DEFAULT_INPUT_DIR,
    file_pattern: str = "*.h5ad",
    dpi_pattern: str = r"(?P<dpi>\d+)DPI",
    embedding_key: str = "niche_embedding",
    label_key: str = "Annotation",
    dpi_order: list[int] | None = None,
    max_cells_per_file: int | None = None,
    seed: int = 0,
) -> tuple[list[DpiData], pd.DataFrame]:
    grouped = group_files_by_dpi(discover_h5ads(input_dir, file_pattern), dpi_pattern)
    if dpi_order is None:
        dpi_order = sorted(grouped)
    missing = [dpi for dpi in dpi_order if dpi not in grouped]
    if missing:
        raise ValueError(f"These DPI values were requested but no files were found: {missing}")

    dpi_data = [
        load_dpi_data(
            grouped[dpi],
            dpi=dpi,
            embedding_key=embedding_key,
            label_key=label_key,
            max_cells_per_file=max_cells_per_file,
            seed=seed + dpi,
        )
        for dpi in dpi_order
    ]
    manifest = pd.DataFrame(
        [
            {
                "dpi": data.dpi,
                "n_files": len(data.files),
                "n_cells": data.embedding.shape[0],
                "n_labels": len(np.unique(data.labels)),
                "files": ";".join(path.name for path in data.files),
            }
            for data in dpi_data
        ]
    )
    return dpi_data, manifest


def make_classifier(
    classifier: str = "knn",
    n_neighbors: int = 30,
    weights: str = "distance",
    random_state: int = 0,
    xgboost_device: str = "cpu",
):
    if classifier == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1),
        )
    if classifier == "logistic":
        from sklearn.linear_model import LogisticRegression

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    if classifier == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Use --classifier knn or install xgboost.") from exc

        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=xgboost_device,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError("classifier should be one of: knn, logistic, xgboost")


def run_pair_label_transfer(
    source: DpiData,
    target: DpiData,
    classifier: str = "knn",
    n_neighbors: int = 30,
    weights: str = "distance",
    random_state: int = 0,
    xgboost_device: str = "cpu",
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(source.labels)
    model = make_classifier(
        classifier=classifier,
        n_neighbors=min(n_neighbors, max(1, source.embedding.shape[0])),
        weights=weights,
        random_state=random_state,
        xgboost_device=xgboost_device,
    )

    if classifier == "xgboost":
        model.fit(source.embedding, y_train)
        pred_code = model.predict(target.embedding).astype(int)
        proba = model.predict_proba(target.embedding)
    else:
        model.fit(source.embedding, source.labels)
        pred = model.predict(target.embedding).astype(str)
        pred_code = encoder.transform(pred)
        proba = model.predict_proba(target.embedding) if hasattr(model, "predict_proba") else None

    pred_label = encoder.inverse_transform(pred_code)
    confidence = np.full(target.embedding.shape[0], np.nan, dtype=np.float32)
    if proba is not None:
        confidence = np.max(proba, axis=1).astype(np.float32, copy=False)

    pred = pd.DataFrame(
        {
            "source_dpi": source.dpi,
            "target_dpi": target.dpi,
            "target_sample": target.samples,
            "target_obs_name": target.obs_names,
            "pred_label": pred_label.astype(str),
            "true_label": target.labels.astype(str),
            "confidence": confidence,
        }
    )
    train_labels = set(source.labels.astype(str))
    comparable = pred["true_label"].isin(train_labels)
    comparable_pred = pred.loc[comparable]
    stats = {
        "source_dpi": source.dpi,
        "target_dpi": target.dpi,
        "n_train": int(source.embedding.shape[0]),
        "n_test": int(target.embedding.shape[0]),
        "n_train_labels": int(len(np.unique(source.labels))),
        "n_target_labels": int(len(np.unique(target.labels))),
        "n_comparable_test": int(comparable.sum()),
        "classifier": classifier,
        "xgboost_device": xgboost_device if classifier == "xgboost" else "",
        "accuracy_common_labels": np.nan,
        "balanced_accuracy_common_labels": np.nan,
    }
    if comparable_pred.shape[0] > 0:
        stats["accuracy_common_labels"] = float(
            accuracy_score(comparable_pred["true_label"], comparable_pred["pred_label"])
        )
        stats["balanced_accuracy_common_labels"] = float(
            balanced_accuracy_score(comparable_pred["true_label"], comparable_pred["pred_label"])
        )
    return pred, stats


def run_all_label_transfer(
    dpi_data: list[DpiData],
    classifier: str = "knn",
    n_neighbors: int = 30,
    weights: str = "distance",
    random_state: int = 0,
    xgboost_device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_tables = []
    stats_rows = []
    for i in range(len(dpi_data) - 1):
        pred, stats = run_pair_label_transfer(
            dpi_data[i],
            dpi_data[i + 1],
            classifier=classifier,
            n_neighbors=n_neighbors,
            weights=weights,
            random_state=random_state + i,
            xgboost_device=xgboost_device,
        )
        pred_tables.append(pred)
        stats_rows.append(stats)
    return pd.concat(pred_tables, ignore_index=True), pd.DataFrame(stats_rows)


def build_sankey_links(
    predictions: pd.DataFrame,
    filter_prop: float = 0.05,
    min_count: int = 50,
) -> pd.DataFrame:
    links = []
    for (source_dpi, target_dpi), data in predictions.groupby(["source_dpi", "target_dpi"], sort=True):
        counts = (
            data.groupby(["pred_label", "true_label"], observed=True)
            .size()
            .reset_index(name="value")
        )
        pred_totals = data["pred_label"].value_counts()
        true_totals = data["true_label"].value_counts()
        for row in counts.itertuples(index=False):
            source_filter = filter_prop * pred_totals.loc[row.pred_label]
            target_filter = filter_prop * true_totals.loc[row.true_label]
            if row.value >= min_count and (row.value >= source_filter or row.value >= target_filter):
                links.append(
                    {
                        "source_dpi": int(source_dpi),
                        "target_dpi": int(target_dpi),
                        "source_label": str(row.pred_label),
                        "target_label": str(row.true_label),
                        "value": int(row.value),
                    }
                )
    return pd.DataFrame(links)


def _hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_sankey(
    links: pd.DataFrame,
    output_html: Path | None = None,
    title: str = "DPI label transfer Sankey",
    width: int = 1200,
    height: int = 800,
    fixed_columns: bool = True,
) -> go.Figure:
    if links.empty:
        raise ValueError("No Sankey links to plot. Lower --filter-prop or --min-count.")

    from plotly.colors import qualitative

    labels = sorted(set(links["source_label"]).union(set(links["target_label"])))
    palette = qualitative.Alphabet + qualitative.Dark24 + qualitative.Light24 + qualitative.Set3
    label_color = {
        lab: CELLTYPE_COLORMAP.get(lab, palette[i % len(palette)])
        for i, lab in enumerate(labels)
    }

    nodes = []
    dpi_values = sorted(set(links["source_dpi"]).union(set(links["target_dpi"])))
    dpi_to_x = {
        dpi: i / max(len(dpi_values) - 1, 1)
        for i, dpi in enumerate(dpi_values)
    }
    stage_node_counts = {}
    for dpi in dpi_values:
        stage_labels = sorted(
            set(links.loc[links["source_dpi"] == dpi, "source_label"])
            .union(set(links.loc[links["target_dpi"] == dpi, "target_label"]))
        )
        stage_node_counts[dpi] = len(stage_labels)
        for lab in stage_labels:
            nodes.append((int(dpi), str(lab)))
    node_index = {node: i for i, node in enumerate(nodes)}

    source = [node_index[(int(row.source_dpi), str(row.source_label))] for row in links.itertuples()]
    target = [node_index[(int(row.target_dpi), str(row.target_label))] for row in links.itertuples()]
    value = links["value"].astype(int).tolist()
    node_labels = [lab for _, lab in nodes]
    node_colors = [label_color[lab] for _, lab in nodes]
    node_kwargs = {}
    arrangement = "snap"
    if fixed_columns:
        node_x = [dpi_to_x[dpi] for dpi, _ in nodes]
        stage_seen = {dpi: 0 for dpi in dpi_values}
        node_y = []
        for dpi, _ in nodes:
            n_stage = max(stage_node_counts[dpi], 1)
            node_y.append((stage_seen[dpi] + 0.5) / n_stage)
            stage_seen[dpi] += 1
        node_kwargs = {"x": node_x, "y": node_y}
        arrangement = "fixed"
    link_colors = [_hex_to_rgba(label_color[str(row.source_label)], 0.35) for row in links.itertuples()]
    dpi_annotations = [
        dict(
            x=dpi_to_x[dpi],
            y=1.08,
            xref="paper",
            yref="paper",
            text=f"{dpi}DPI",
            showarrow=False,
            font=dict(size=14, color="black"),
        )
        for dpi in dpi_values
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement=arrangement,
                node=dict(
                    pad=12,
                    thickness=14,
                    line=dict(color="rgba(0,0,0,0.25)", width=0.4),
                    label=node_labels,
                    color=node_colors,
                    **node_kwargs,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colors,
                    customdata=links[
                        ["source_dpi", "target_dpi", "source_label", "target_label", "value"]
                    ].astype(str).to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}DPI %{customdata[2]} -> "
                        "%{customdata[1]}DPI %{customdata[3]}<br>"
                        "cells=%{customdata[4]}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(
        title_text=title,
        width=width,
        height=height,
        font_size=11,
        annotations=dpi_annotations,
        margin=dict(l=20, r=20, t=90, b=20),
    )
    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html))
    return fig


def run_pipeline(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    file_pattern: str = "*.h5ad",
    dpi_pattern: str = r"(?P<dpi>\d+)DPI",
    embedding_key: str = "niche_embedding",
    label_key: str = "Annotation",
    dpi_order: list[int] | None = None,
    classifier: str = "knn",
    n_neighbors: int = 30,
    weights: str = "distance",
    xgboost_device: str = "cpu",
    max_cells_per_file: int | None = None,
    filter_prop: float = 0.05,
    min_count: int = 10,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, go.Figure]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi_data, manifest = load_all_dpi_data(
        input_dir=input_dir,
        file_pattern=file_pattern,
        dpi_pattern=dpi_pattern,
        embedding_key=embedding_key,
        label_key=label_key,
        dpi_order=dpi_order,
        max_cells_per_file=max_cells_per_file,
        seed=random_state,
    )
    predictions, stats = run_all_label_transfer(
        dpi_data,
        classifier=classifier,
        n_neighbors=n_neighbors,
        weights=weights,
        random_state=random_state,
        xgboost_device=xgboost_device,
    )
    links = build_sankey_links(predictions, filter_prop=filter_prop, min_count=min_count)
    fig = plot_sankey(
        links,
        output_html=output_dir / "dpi_label_transfer_sankey.html",
        title=f"",
    )

    manifest.to_csv(output_dir / "dpi_sample_manifest.csv", index=False)
    predictions.to_csv(output_dir / "dpi_pair_predictions.csv", index=False)
    stats.to_csv(output_dir / "dpi_pair_metrics.csv", index=False)
    links.to_csv(output_dir / "dpi_sankey_links.csv", index=False)
    return predictions, stats, links, fig


def parse_dpi_order(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(x) for x in re.split(r"[,\\s]+", value.strip()) if x]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer cell-type labels from each DPI to the next DPI using niche embeddings and plot a Sankey diagram."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-pattern", default="*.h5ad")
    parser.add_argument("--dpi-pattern", default=r"(?P<dpi>\d+)DPI")
    parser.add_argument("--dpi-order", default=None, help="Comma/space separated order, e.g. '2,5,10,15,20,30,60'.")
    parser.add_argument("--embedding-key", default="niche_embedding")
    parser.add_argument("--label-key", default="Annotation")
    parser.add_argument("--classifier", choices=["knn", "logistic", "xgboost"], default="knn")
    parser.add_argument("--xgboost-device", default="cpu", help="Use 'cuda' on a GPU node, or 'cpu'.")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--weights", choices=["uniform", "distance"], default="distance")
    parser.add_argument("--max-cells-per-file", type=int, default=None)
    parser.add_argument("--filter-prop", type=float, default=0.05)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    predictions, stats, links, _ = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        dpi_pattern=args.dpi_pattern,
        embedding_key=args.embedding_key,
        label_key=args.label_key,
        dpi_order=parse_dpi_order(args.dpi_order),
        classifier=args.classifier,
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        xgboost_device=args.xgboost_device,
        max_cells_per_file=args.max_cells_per_file,
        filter_prop=args.filter_prop,
        min_count=args.min_count,
        random_state=args.random_state,
    )
    print(f"[predictions] {predictions.shape[0]:,} rows")
    print(f"[metrics] {args.output_dir / 'dpi_pair_metrics.csv'}")
    print(f"[links] {links.shape[0]:,} links")
    print(f"[sankey] {args.output_dir / 'dpi_label_transfer_sankey.html'}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
