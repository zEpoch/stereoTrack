"""
空间转录组多切片整合（DMT embedding 版本）
  - 读取 inference_adatas/ 下所有 slice_*.h5ad
  - 合并后加载外部 DMT embedding 到 adata.obsm
  - 基于 cell_dmt_highdim / niche_dmt_highdim 直接做邻域图 + Leiden 聚类
  - 不进行 UMAP
"""

import os
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import yaml

# ── 尝试使用 rapids_singlecell（GPU 加速），失败则退回 scanpy ──
try:
    import rapids_singlecell as rsc
    from rapids_singlecell import preprocessing as rsc_pp
    RSC_AVAILABLE = True
    print("[INFO] rapids_singlecell 可用，使用 GPU 加速")
except ImportError:
    RSC_AVAILABLE = False
    print("[INFO] rapids_singlecell 不可用，使用 scanpy (CPU)")


def parse_args():
    parser = argparse.ArgumentParser(description="整合与聚类（DMT embedding）")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--n_neighbors", type=int, default=30, help="构建邻域图时的邻居数")
    parser.add_argument("--resolution", type=str, default="0.8", help="兼容参数：当未单独设置 cell/niche_resolution 时使用该值")
    parser.add_argument("--cell_resolution", type=str, default=None, help="cell_dmt_highdim 的 Leiden 分辨率，支持逗号分隔")
    parser.add_argument("--niche_resolution", type=str, default=None, help="niche_dmt_highdim 的 Leiden 分辨率，支持逗号分隔")

    parser.add_argument("--cell_dmt_highdim_path", type=str,
                        default="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_merfish_pfc_v3/dmt_out_cell/X_dmt_highdim.npy")
    parser.add_argument("--cell_dmt_path", type=str,
                        default="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_merfish_pfc_v3/dmt_out_cell/X_dmt.npy")
    parser.add_argument("--niche_dmt_highdim_path", type=str,
                        default="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_merfish_pfc_v3/dmt_out_niche/X_dmt_highdim.npy")
    parser.add_argument("--niche_dmt_path", type=str,
                        default="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_merfish_pfc_v3/dmt_out_niche/X_dmt.npy")
    return parser.parse_args()


def _parse_resolution_list(raw, fallback=None):
    if raw is None or str(raw).strip() == "":
        raw = fallback
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip() != ""]
    if len(vals) == 0:
        raise ValueError(f"非法分辨率参数: {raw}")
    return vals


def compute_neighbors_and_cluster(adata, use_rep, resolutions, n_neighbors=30, use_gpu=True):
    cluster_keys = []

    if use_gpu and RSC_AVAILABLE:
        rsc_pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
        for res in resolutions:
            key = f"leiden_{use_rep}_r{res}"
            rsc.tl.leiden(adata, resolution=res, key_added=key)
            cluster_keys.append(key)
            print(f"  Leiden (GPU) r={res} → adata.obs['{key}']")
    else:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
        for res in resolutions:
            key = f"leiden_{use_rep}_r{res}"
            sc.tl.leiden(adata, resolution=res, key_added=key)
            cluster_keys.append(key)
            print(f"  Leiden (CPU) r={res} → adata.obs['{key}']")

    return cluster_keys


def load_and_attach_dmt_embeddings(adata, args):
    x_cell_high = np.load(args.cell_dmt_highdim_path)
    x_cell = np.load(args.cell_dmt_path)
    x_niche_high = np.load(args.niche_dmt_highdim_path)
    x_niche = np.load(args.niche_dmt_path)

    n_obs = adata.n_obs
    for name, arr in [
        ("cell_dmt_highdim", x_cell_high),
        ("cell_dmt", x_cell),
        ("niche_dmt_highdim", x_niche_high),
        ("niche_dmt", x_niche),
    ]:
        if arr.shape[0] != n_obs:
            raise ValueError(f"{name} 行数({arr.shape[0]}) 与 adata.n_obs({n_obs}) 不一致")
        adata.obsm[name] = arr
        print(f"[INFO] 已加载 {name}: {arr.shape}")


def main():
    args = parse_args()
    cell_resolutions = _parse_resolution_list(args.cell_resolution, fallback=args.resolution)
    niche_resolutions = _parse_resolution_list(args.niche_resolution, fallback=args.resolution)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    save_dir = cfg["paths"]["save_dir"]
    adata_in_dir = os.path.join(save_dir, "inference_adatas")
    adata_out_dir = os.path.join(save_dir, "integrated_adatas")
    Path(adata_out_dir).mkdir(parents=True, exist_ok=True)

    # 1) 加载切片
    slice_files = sorted(Path(adata_in_dir).glob("slice_*.h5ad"))
    if not slice_files:
        raise FileNotFoundError(f"在 {adata_in_dir} 下未找到 slice_*.h5ad")

    adata_list = []
    raw_list = []
    for fpath in slice_files:
        ad = sc.read_h5ad(fpath)
        raw_ad = ad.raw.to_adata() if ad.raw is not None else ad.copy()
        raw_list.append(raw_ad)
        ad.obs["slice"] = fpath.stem
        adata_list.append(ad)
        print(f"加载 {fpath}，细胞数: {ad.n_obs}")

    # 2) 合并
    adata_all = sc.concat(adata_list, join="inner", label="batch", index_unique="-")
    raw_merged = sc.concat(raw_list, join="inner", label="batch")
    adata_all.raw = raw_merged.copy()
    print(f"\n合并后 AnnData: {adata_all.n_obs} cells, {adata_all.n_vars} genes")

    # 3) 附加 DMT embedding
    load_and_attach_dmt_embeddings(adata_all, args)

    # 4) 基于 highdim embedding 聚类（不做 UMAP）
    emb_res_map = {
        "cell_dmt_highdim": cell_resolutions,
        "niche_dmt_highdim": niche_resolutions,
    }

    for emb_key, res_list in emb_res_map.items():
        print(f"\n{'='*60}")
        print(f"处理 embedding: {emb_key}")
        print(f"形状: {adata_all.obsm[emb_key].shape}")
        cluster_keys = compute_neighbors_and_cluster(
            adata_all,
            use_rep=emb_key,
            resolutions=res_list,
            n_neighbors=args.n_neighbors,
            use_gpu=RSC_AVAILABLE,
        )
        print(f"  {emb_key} 聚类完成，obs keys: {cluster_keys}")

    # 5) 保存
    out_path = os.path.join(adata_out_dir, "integrated_dmt.h5ad")
    adata_all.write_h5ad(out_path)
    print(f"\n最终 AnnData 已保存: {out_path}")
    print(f"obs 列: {list(adata_all.obs.columns)}")
    print(f"obsm keys: {list(adata_all.obsm.keys())}")
    print("\n全部完成！")


if __name__ == "__main__":
    main()
