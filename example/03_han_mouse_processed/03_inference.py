from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stereotrack.dataset import normalize_meta, resolve_cache_file
from stereotrack.mae import MAEEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for Han mouse processed MAE checkpoints.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/03_config_han_mouse_processed.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--genes", nargs="*", default=None, help="Optional gene subset to write into X.")
    parser.add_argument("--gene-file", type=Path, default=None, help="Optional text file with one gene per line.")
    parser.add_argument("--write-niche-layer", action="store_true", help="Deprecated; layers['niche_recon'] is always written.")
    parser.add_argument("--write-cell-layer", action="store_true", help="Also write cell decoder reconstruction to layers['cell_recon'].")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def load_meta(cfg: dict) -> tuple[dict, str]:
    cache_dir = os.path.join(cfg["paths"]["input_dir"], "cache")
    meta_pkl = os.path.join(cache_dir, "meta.pkl")
    meta_yaml = os.path.join(cache_dir, "meta.yaml")

    if os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as handle:
            return normalize_meta(pickle.load(handle)), cache_dir
    if os.path.exists(meta_yaml):
        with open(meta_yaml, "r") as handle:
            return normalize_meta(yaml.safe_load(handle)), cache_dir
    raise FileNotFoundError(f"Cache not found: {cache_dir}. Run 01_preprocess.py first.")


def requested_genes(args: argparse.Namespace) -> list[str] | None:
    genes: list[str] = []
    if args.gene_file is not None:
        with open(args.gene_file, "r") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    genes.extend(item.strip() for item in line.split(",") if item.strip())
    if args.genes:
        genes.extend(args.genes)
    if not genes:
        return None
    return list(dict.fromkeys(genes))


def select_gene_indices(meta: dict, genes: list[str] | None) -> tuple[np.ndarray | slice, list[str]]:
    common_genes = list(meta.get("common_genes") or meta.get("common_human_genes") or [])
    if not common_genes:
        common_genes = [f"gene_{idx}" for idx in range(int(meta["input_dim"]))]

    if genes is None:
        return slice(None), common_genes

    gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    missing = [gene for gene in genes if gene not in gene_to_idx]
    if missing:
        print(f"Warning: {len(missing)} requested genes are not in common_genes: {missing[:10]}")
    present = [gene for gene in genes if gene in gene_to_idx]
    if not present:
        raise ValueError("None of the requested genes are present in common_genes")
    return np.asarray([gene_to_idx[gene] for gene in present], dtype=np.int64), present


def strip_state_dict(raw: dict) -> dict:
    state = {}
    for key, value in raw.items():
        cleaned_key = key
        while cleaned_key.startswith("module.") or cleaned_key.startswith("model."):
            if cleaned_key.startswith("module."):
                cleaned_key = cleaned_key[len("module.") :]
            elif cleaned_key.startswith("model."):
                cleaned_key = cleaned_key[len("model.") :]

        if cleaned_key.startswith("decoder."):
            cleaned_key = "niche_decoder." + cleaned_key[len("decoder.") :]
        elif cleaned_key.startswith("binary_decoder."):
            cleaned_key = "niche_binary_decoder." + cleaned_key[len("binary_decoder.") :]

        state[cleaned_key] = value
    return state


def extract_checkpoint(path: Path) -> tuple[dict, object, object]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        raw_dict = ckpt["state_dict"]
        epoch = ckpt.get("epoch", "unknown")
        best_loss = ckpt.get("best_loss", "unknown")
    elif "model_state" in ckpt:
        raw_dict = ckpt["model_state"]
        epoch = ckpt.get("epoch", "unknown")
        best_loss = ckpt.get("best_loss", "unknown")
    else:
        raw_dict = ckpt
        epoch = "unknown"
        best_loss = ckpt.get("best_loss", "unknown") if isinstance(ckpt, dict) else "unknown"
    return strip_state_dict(raw_dict), epoch, best_loss


def sparse_from_npz(data: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            data[f"{prefix}_data"].astype(np.float32),
            data[f"{prefix}_indices"].astype(np.int64),
            data[f"{prefix}_indptr"].astype(np.int64),
        ),
        shape=tuple(data[f"{prefix}_shape"]),
    )


def build_output_adata(
    cache_dir: str,
    slice_idx: int,
    selected_genes: list[str],
    x_input: np.ndarray,
    x_niche: np.ndarray,
    x_cell: np.ndarray | None,
    z_cell: np.ndarray,
    z_niche: np.ndarray,
    slice_info: dict
) -> ad.AnnData:
    adata_cache = os.path.join(cache_dir, "adatas", f"slice_{slice_idx}.h5ad")
    if os.path.exists(adata_cache):
        adata_cached = sc.read_h5ad(adata_cache)
        if set(selected_genes).issubset(set(adata_cached.var_names)):
            var = adata_cached.var.loc[selected_genes].copy()
        else:
            var = None

        adata_out = ad.AnnData(
            X=x_input.astype(np.float32),
            obs=adata_cached.obs.copy(),
            var=var,
        )
        if var is None:
            adata_out.var_names = selected_genes

        for key in adata_cached.obsm.keys():
            adata_out.obsm[key] = np.asarray(adata_cached.obsm[key]).copy()
    else:
        adata_out = ad.AnnData(x_input.astype(np.float32))
        adata_out.var_names = selected_genes

    adata_out.obsm["cell_embedding"] = z_cell.astype(np.float32)
    adata_out.obsm["niche_embedding"] = z_niche.astype(np.float32)
    adata_out.layers["niche_recon"] = x_niche.astype(np.float32)
    if x_cell is not None:
        adata_out.layers["cell_recon"] = x_cell.astype(np.float32)
    adata_out.obs["batch"] = slice_info.get("batch", f"slice_{slice_idx}")
    adata_out.obs["slice_idx"] = slice_idx
    return adata_out


@torch.no_grad()
def inference() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    meta, cache_dir = load_meta(cfg)

    train_cfg = cfg.get("training", {})
    model_cfg = cfg["model"]
    batch_size = args.batch_size or int(train_cfg.get("infer_batch_size", 4096))
    selected_idx, selected_genes = select_gene_indices(meta, requested_genes(args))

    save_dir = args.output_dir or Path(cfg["paths"]["save_dir"]) / "inference_adatas"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    selected_idx_t = None if isinstance(selected_idx, slice) else torch.as_tensor(selected_idx, dtype=torch.long, device=device)
    input_dim = int(meta["input_dim"])
    n_slices = int(meta["n_slices"])
    print(f"cache_dir={cache_dir}")
    print(f"n_slices={n_slices}, input_dim={input_dim}, output_genes={len(selected_genes)}")

    model = MAEEncoder(
        input_dim=input_dim,
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
        dropout_rate=model_cfg["dropout_rate"],
        gene_mask_ratio=model_cfg["gene_mask_ratio"],
        cell_mask_ratio=model_cfg["cell_mask_ratio"],
        n_encoder_layers=model_cfg.get("n_encoder_layers", 2),
        n_decoder_layers=model_cfg.get("n_decoder_layers", 1),
        niche_self_weight=float(model_cfg.get("niche_self_weight", 0.5)),
        lambda_cell_recon=float(train_cfg.get("lambda_cell_recon", 1.0)),
        lambda_niche_recon=float(train_cfg.get("lambda_niche_recon", 1.0)),
        lambda_cell_mask=float(train_cfg.get("lambda_cell_mask", 1.0)),
    )

    state_dict, epoch, best_loss = extract_checkpoint(args.checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        print("Strict checkpoint loading failed; retrying with strict=False.")
        print(f"Original error: {error}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")

    model.to(device)
    model.eval()
    print(f"loaded checkpoint={args.checkpoint} epoch={epoch} best_loss={best_loss}")

    for slice_idx, slice_info in enumerate(meta["slice_info"]):
        n_cells = int(slice_info["n_cells"])
        batch_name = slice_info.get("batch", f"slice_{slice_idx}.h5ad")
        print(f"\nSlice {slice_idx + 1}/{n_slices}: {batch_name}, cells={n_cells}")

        fpath = resolve_cache_file(cache_dir, slice_info["file"])
        with np.load(fpath) as data:
            feat_sparse = sparse_from_npz(data, "feat")
            adj_sparse = sparse_from_npz(data, "adj")

        z_cell_list: list[np.ndarray] = []
        z_niche_list: list[np.ndarray] = []
        x_input_list: list[np.ndarray] = []
        x_niche_list: list[np.ndarray] = []
        x_cell_list: list[np.ndarray] = []

        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            core_idx = np.arange(start, end)
            neighbors = adj_sparse[core_idx].nonzero()[1]
            sub_nodes = np.unique(np.concatenate([core_idx, neighbors]))
            core_mask = np.isin(sub_nodes, core_idx)

            feat_ts = torch.from_numpy(feat_sparse[sub_nodes].toarray()).to(device)
            adj_ts = torch.from_numpy(adj_sparse[sub_nodes][:, sub_nodes].toarray()).to(device)
            core_mask_ts = torch.from_numpy(core_mask).to(device=device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                z_cell, z_niche = model.encode(feat_ts, adj_ts)
                x_niche = model.niche_decoder(z_niche)
                x_cell = model.cell_decoder(z_cell) if args.write_cell_layer else None

            z_cell_core = z_cell[core_mask_ts]
            z_niche_core = z_niche[core_mask_ts]
            x_input_core = feat_ts[core_mask_ts]
            x_niche_core = x_niche[core_mask_ts]
            if selected_idx_t is not None:
                x_input_core = x_input_core.index_select(1, selected_idx_t)
                x_niche_core = x_niche_core.index_select(1, selected_idx_t)

            z_cell_list.append(z_cell_core.cpu().numpy())
            z_niche_list.append(z_niche_core.cpu().numpy())
            x_input_list.append(x_input_core.cpu().numpy())
            x_niche_list.append(x_niche_core.cpu().numpy())
            if x_cell is not None:
                x_cell_core = x_cell[core_mask_ts]
                if selected_idx_t is not None:
                    x_cell_core = x_cell_core.index_select(1, selected_idx_t)
                x_cell_list.append(x_cell_core.cpu().numpy())
            else:
                x_cell_core = None

            del feat_ts, adj_ts, core_mask_ts, z_cell, z_niche, x_niche, x_cell
            del z_cell_core, z_niche_core, x_input_core, x_niche_core, x_cell_core
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  progress: {end}/{n_cells}")

        z_cell_np = np.concatenate(z_cell_list, axis=0)
        z_niche_np = np.concatenate(z_niche_list, axis=0)
        x_input_np = np.concatenate(x_input_list, axis=0)
        x_niche_np = np.concatenate(x_niche_list, axis=0)
        x_cell_np = np.concatenate(x_cell_list, axis=0) if x_cell_list else None

        adata_out = build_output_adata(
            cache_dir=cache_dir,
            slice_idx=slice_idx,
            selected_genes=selected_genes,
            x_input=x_input_np,
            x_niche=x_niche_np,
            x_cell=x_cell_np,
            z_cell=z_cell_np,
            z_niche=z_niche_np,
            slice_info=slice_info
        )

        output_name = batch_name if str(batch_name).endswith(".h5ad") else f"{batch_name}.h5ad"
        out_path = save_dir / output_name
        adata_out.write_h5ad(out_path)
        print(f"  saved: {out_path}")

        del feat_sparse, adj_sparse, z_cell_list, z_niche_list, x_input_list, x_niche_list, x_cell_list, adata_out
        gc.collect()

    print(f"\nInference complete: {save_dir}")


if __name__ == "__main__":
    inference()
