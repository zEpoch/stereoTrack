from __future__ import annotations

import argparse
import csv
import gc
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereotrack.mae import MAEEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write StereoTrack reconstructed expression as per-slice .npy files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--start-slice", type=int, default=0)
    parser.add_argument("--end-slice", type=int, default=None)
    return parser.parse_args()


def load_meta(cfg: dict) -> tuple[dict, Path]:
    cache_dir = Path(cfg["paths"]["input_dir"]) / "cache"
    meta_pkl = cache_dir / "meta.pkl"
    meta_yaml = cache_dir / "meta.yaml"
    if meta_pkl.exists():
        with meta_pkl.open("rb") as handle:
            return pickle.load(handle), cache_dir
    if meta_yaml.exists():
        with meta_yaml.open() as handle:
            return yaml.safe_load(handle), cache_dir
    raise FileNotFoundError(f"Missing cache metadata in {cache_dir}")


def clean_state_dict(ckpt: object) -> tuple[dict, object, object]:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        epoch = ckpt.get("epoch", "unknown")
        loss = ckpt.get("best_loss", ckpt.get("train_loss", "unknown"))
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        raw = ckpt["model_state"]
        epoch = ckpt.get("epoch", "unknown")
        loss = ckpt.get("best_loss", "unknown")
    else:
        raw = ckpt
        epoch = "unknown"
        loss = "unknown"

    state = {}
    for key, value in raw.items():
        state[key.replace("module.", "").replace("model.", "")] = value
    return state, epoch, loss


def stem_from_slice_info(slice_info: dict) -> str:
    return Path(str(slice_info.get("batch", ""))).stem or f"slice_{slice_info.get('slice_idx', 'unknown')}"


def append_manifest(path: Path, row: dict) -> None:
    fields = [
        "slice_idx",
        "batch",
        "n_cells",
        "n_genes",
        "dtype",
        "expression_path",
        "cache_npz",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    meta, cache_dir = load_meta(cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    model = MAEEncoder(
        input_dim=int(meta["input_dim"]),
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        dropout_rate=model_cfg["dropout_rate"],
        gene_mask_ratio=model_cfg["gene_mask_ratio"],
        cell_mask_ratio=model_cfg["cell_mask_ratio"],
        n_encoder_layers=model_cfg.get("n_encoder_layers", 2),
        n_decoder_layers=model_cfg.get("n_decoder_layers", 1),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state, epoch, loss = clean_state_dict(ckpt)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        print(f"[warn] strict checkpoint load failed, retry strict=False: {exc}")
        incompatible = model.load_state_dict(state, strict=False)
        print(f"[warn] missing={incompatible.missing_keys}")
        print(f"[warn] unexpected={incompatible.unexpected_keys}")
    model.to(device)
    model.eval()

    common_genes = list(meta["common_genes"])
    (output_dir / "genes.txt").write_text("\n".join(map(str, common_genes)) + "\n")

    print(f"[model] {args.checkpoint}")
    print(f"[checkpoint] epoch={epoch}, loss={loss}")
    print(f"[cache] {cache_dir}")
    print(f"[output] {output_dir}")
    print(f"[shape] slices={meta['n_slices']}, genes={meta['input_dim']}, dtype={args.output_dtype}, device={device}")

    end_slice = int(meta["n_slices"]) if args.end_slice is None else min(args.end_slice, int(meta["n_slices"]))
    manifest = output_dir / "manifest.csv"
    dtype = np.float16 if args.output_dtype == "float16" else np.float32

    for slice_idx in range(args.start_slice, end_slice):
        slice_info = meta["slice_info"][slice_idx]
        stem = stem_from_slice_info(slice_info)
        n_cells = int(slice_info["n_cells"])
        n_genes = int(meta["input_dim"])
        out_path = output_dir / f"{stem}.expression.{args.output_dtype}.npy"
        tmp_path = output_dir / f".{stem}.expression.{args.output_dtype}.tmp.npy"
        cache_npz = cache_dir / slice_info["file"]

        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            print(f"[skip] {stem}: {out_path}")
            continue

        print(f"[slice] {slice_idx}/{end_slice - 1} {stem}: cells={n_cells:,}")
        with np.load(cache_npz) as data:
            feat_sparse = sp.csr_matrix(
                (
                    data["feat_data"].astype(np.float32),
                    data["feat_indices"].astype(np.int64),
                    data["feat_indptr"].astype(np.int64),
                ),
                shape=tuple(data["feat_shape"]),
            )
            adj_sparse = sp.csr_matrix(
                (
                    data["adj_data"].astype(np.float32),
                    data["adj_indices"].astype(np.int64),
                    data["adj_indptr"].astype(np.int64),
                ),
                shape=tuple(data["adj_shape"]),
            )

        if tmp_path.exists():
            tmp_path.unlink()
        out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype, shape=(n_cells, n_genes))

        for start in range(0, n_cells, args.batch_size):
            stop = min(start + args.batch_size, n_cells)
            core_idx = np.arange(start, stop)
            neighbors = adj_sparse[core_idx].nonzero()[1]
            sub_nodes = np.unique(np.concatenate([core_idx, neighbors]))
            core_mask = np.isin(sub_nodes, core_idx)

            feat_sub = feat_sparse[sub_nodes].toarray()
            adj_sub = adj_sparse[sub_nodes][:, sub_nodes].toarray()
            feat_ts = torch.from_numpy(feat_sub).to(device)
            adj_ts = torch.from_numpy(adj_sub).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.use_amp)):
                _, z_niche = model.encode(feat_ts, adj_ts)
                x_recon = model.niche_decoder(z_niche)

            block = x_recon[core_mask].detach().cpu().numpy().astype(dtype, copy=False)
            out[start:stop, :] = block

            del feat_sub, adj_sub, feat_ts, adj_ts, z_niche, x_recon, block
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if stop == n_cells or stop % (args.batch_size * 10) == 0:
                print(f"  [progress] {stem}: {stop:,}/{n_cells:,}")

        out.flush()
        del out, feat_sparse, adj_sparse
        tmp_path.replace(out_path)
        append_manifest(
            manifest,
            {
                "slice_idx": slice_idx,
                "batch": slice_info["batch"],
                "n_cells": n_cells,
                "n_genes": n_genes,
                "dtype": args.output_dtype,
                "expression_path": str(out_path),
                "cache_npz": str(cache_npz),
            },
        )
        print(f"[written] {out_path}")
        gc.collect()

    print("[done] StereoTrack expression npy inference finished.")


if __name__ == "__main__":
    main()
