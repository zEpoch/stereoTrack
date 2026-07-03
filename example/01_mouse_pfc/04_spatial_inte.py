"""
空间转录组多切片整合与聚类
  - 读取 inference_adatas/ 下所有 slice_*.h5ad
  - 对 cell_embedding 和 niche_embedding 进行 Leiden 聚类
  - 结果保存在 obsm 和 obs 中
"""

import os
import sys
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description="整合与聚类")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--n_neighbors", type=int, default=30,
                        help="构建邻域图时的邻居数")
    parser.add_argument("--resolution", type=str, default="0.8",
                        help="兼容参数：当未单独设置 cell/niche_resolution 时使用该值")
    parser.add_argument("--cell_resolution", type=str, default=None,
                        help="cell_embedding 的 Leiden 分辨率，支持逗号分隔，如 0.4,0.8,1.2")
    parser.add_argument("--niche_resolution", type=str, default=None,
                        help="niche_embedding 的 Leiden 分辨率，支持逗号分隔，如 1.0,1.5")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()



def _parse_resolution_list(raw, fallback=None):
    """将 '0.4,0.8' 或单值字符串解析为 [0.4, 0.8]；fallback 为空时回退。"""
    if raw is None or str(raw).strip() == "":
        raw = fallback
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip() != ""]
    if len(vals) == 0:
        raise ValueError(f"非法分辨率参数: {raw}")
    return vals



def compute_neighbors_and_cluster(adata, 
                                  use_rep, 
                                  resolutions, 
                                  n_neighbors=30, 
                                  use_gpu=True):
    """
    基于 use_rep 构建邻居图，用多个分辨率做 Leiden 聚类
    """
    cluster_keys = []
    if use_gpu and RSC_AVAILABLE:
        # GPU 邻居图
        rsc_pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
        rsc.tl.umap(adata, key_added=f"{use_rep}_umap")
        for res in resolutions:
            key = f"leiden_{use_rep}_r{res}"
            rsc.tl.leiden(adata, resolution=res, key_added=key)
            cluster_keys.append(key)
            print(f"  Leiden (GPU) r={res} → adata.obs['{key}']")
    else:
        # CPU
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
        sc.tl.umap(adata, key_added=f"{use_rep}_umap")  # 可选，构建 UMAP 以便可视化
        for res in resolutions:
            key = f"leiden_{use_rep}_r{res}"
            sc.tl.leiden(adata, resolution=res, key_added=key)
            cluster_keys.append(key)
            print(f"  Leiden (CPU) r={res} → adata.obs['{key}']")
    return cluster_keys


def main():
    args = parse_args()
    # 解析分辨率参数（支持逗号分隔）
    cell_resolutions = _parse_resolution_list(args.cell_resolution, fallback=args.resolution)
    niche_resolutions = _parse_resolution_list(args.niche_resolution, fallback=args.resolution)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    save_dir = cfg["paths"]["save_dir"]
    adata_in_dir = os.path.join(save_dir, "inference_adatas")
    adata_out_dir = os.path.join(save_dir, "integrated_adatas")
    Path(adata_out_dir).mkdir(parents=True, exist_ok=True)


    # ── 1. 加载所有切片 h5ad ──
    slice_files = sorted(Path(adata_in_dir).glob("slice_*.h5ad"))
    if not slice_files:
        raise FileNotFoundError(f"在 {adata_in_dir} 下未找到 slice_*.h5ad，请先运行推理脚本")
    adata_list = []
    raw_list = []
    for fpath in slice_files:
        ad = sc.read_h5ad(fpath)
        raw_ad = ad.raw.to_adata()
        raw_list.append(raw_ad)
        slice_id = fpath.stem  # 如 "slice_0"
        ad.obs["slice"] = slice_id
        adata_list.append(ad)
        print(f"加载 {fpath}，细胞数: {ad.n_obs}")


    # ── 2. 拼接所有切片 ──
    adata_all = sc.concat(adata_list, join='inner', label='batch', index_unique="-")
    raw_merged = sc.concat(raw_list, join='inner', label='batch')
    adata_all.raw = raw_merged.copy()  # 将合并后的 raw 赋值回 adata_all
    print(f"\n合并后 AnnData: {adata_all.n_obs} cells, {adata_all.n_vars} genes")
    print(f"现有 obsm keys: {list(adata_all.obsm.keys())}")


    # ── 3. 对每种 embedding 分别整合 + 聚类 ──
    embedding_resolution_map = {
        "cell_embedding": cell_resolutions,
        "niche_embedding": niche_resolutions,
    }
    embedding_keys = ["cell_embedding", "niche_embedding"]

    for emb_key in embedding_keys:
        if emb_key not in adata_all.obsm:
            print(f"\n[警告] obsm 中找不到 '{emb_key}'，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"处理 embedding: {emb_key}")
        print(f"形状: {adata_all.obsm[emb_key].shape}")

        # 3b. 构建邻居图 + 多分辨率聚类
        cluster_keys = compute_neighbors_and_cluster(
            adata_all,
            use_rep=emb_key,
            resolutions=embedding_resolution_map.get(emb_key, cell_resolutions),
            n_neighbors=args.n_neighbors,
            use_gpu=RSC_AVAILABLE,
        )

        print(f"  {emb_key} 聚类完成，obs keys: {cluster_keys}")

    # ── 4. 保存结果 ──
    out_path = os.path.join(adata_out_dir, "integrated.h5ad")
    adata_all.write_h5ad(out_path)
    print(f"\n最终 AnnData 已保存: {out_path}")
    print(f"obs 列: {list(adata_all.obs.columns)}")
    print(f"obsm keys: {list(adata_all.obsm.keys())}")

    # ── 5. 可选：为每个切片单独保存 ──
    # for slice_id in adata_all.obs["slice"].unique():
    #     sub = adata_all[adata_all.obs["slice"] == slice_id].copy()
    #     sub_path = os.path.join(adata_out_dir, f"{slice_id}_integrated.h5ad")
    #     sub.write_h5ad(sub_path)
    #     print(f"  → {sub_path}")

    print("\n全部完成！")


if __name__ == "__main__":
    main()
    


'''
CUDA_VISIBLE_DEVICES=3 python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/04_spatial_inte.py \
  --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v4.yaml  \
  --cell_resolution 1.5 \
  --niche_resolution 1 \
  --n_neighbors 30 \
  --device cuda:0
'''

'''
python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/04_spatial_inte.py \
  --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v3.yaml  \
  --cell_resolution 0.2 \
  --niche_resolution 1 \
  --n_neighbors 30 \
  --device cuda:0
'''
