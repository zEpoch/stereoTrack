"""
绘图脚本:
  - 绘制整体空间分布图（leiden_cell_embedding_r0.8 + leiden_niche_embedding_r0.8，
    并可额外通过 --spatial_color 指定其他 obs 列着色）
  - 绘制 cell_embedding 和 niche_embedding 的 UMAP 图（含 batch 整合效果）
示例:
  python 05_plot.py --config config.yaml --spatial_color CellType \
      --umap_cell_color leiden_cell_embedding_r0.8 \
      --umap_niche_color leiden_niche_embedding_r0.8
"""

import os

import gc
import yaml
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import matplotlib.colors as mcolors
from matplotlib.font_manager import fontManager, FontProperties
import os
fontManager.addfont('/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/Arial.ttf')
font = FontProperties(fname='/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name


tick_font = FontProperties(fname='/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/Arial-ItalicMT.otf', style = 'italic')

from matplotlib_scalebar.scalebar import ScaleBar


def parse_args():
    parser = argparse.ArgumentParser(description="空间与 UMAP 绘图")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--adata", type=str, default=None,
                        help="h5ad 路径，默认 integrated_adatas/integrated.h5ad")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认为 save_dir/plots")
    parser.add_argument("--spatial_color", type=str, default=None,
                        help="额外绘制指定 obs 列的空间分布图（如 CellType）")
    parser.add_argument("--umap_cell_color", type=str,
                        default="leiden_cell_embedding_r0.8",
                        help="cell embedding UMAP 聚类着色")
    parser.add_argument("--umap_niche_color", type=str,
                        default="leiden_niche_embedding_r0.8",
                        help="niche embedding UMAP 聚类着色")
    parser.add_argument("--batch_col", type=str, default=None,
                        help="批次列名（默认尝试 'batch' 或从 config 中读取）")
    parser.add_argument("--scale_factor", type=float, default=0.1,
                        help="坐标单位 -> 毫米的转换因子（例如 2mm/199pixel）")
    parser.add_argument("--figsize", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def plot_spatial(adata: sc.AnnData, 
                 color_col: str, 
                 output_dir: str, 
                 scale_factor: float = 0.1, 
                 figsize: tuple = (8, 8),
                 dpi: int = 150, 
                 colormap: dict = None):
    """使用 sc.pl.embedding 绘制空间分布图，并添加 scale bar"""
    if "spatial" not in adata.obsm:
        print("[警告] 没有 'spatial' 数据，跳过空间图")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0)   # figure 背景透明
    ax.set_facecolor('none') # axes 背景透明
    
    sc.pl.embedding(
        adata,
        basis="spatial",                # 使用空间坐标
        color=color_col,
        ax=ax,
        show=False,
        legend_loc="right margin",      # 图例放在右侧
        title=f"Spatial ({color_col})",
        s = 2,
        frameon=False,                  # 不显示坐标轴框
        palette = colormap
    )
    ax.set_aspect("equal")

    # 添加比例尺
    scalebar = ScaleBar(
        scale_factor, "µm", fixed_value=500,
        location="lower left", frameon=False
    )
    ax.add_artist(scalebar)

    safe_color = color_col.replace("/", "_")
    save_path = os.path.join(output_dir, f"spatial_{safe_color}.pdf")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True, facecolor='none')
    plt.close(fig)
    print(f"空间图已保存: {save_path}")


def plot_umap(adata, umap_key, color_col, output_dir, figsize, dpi):
    """绘制 UMAP 图（颜色来自 obs 列）"""
    # 优先使用指定 key，否则回退到 'umap'
    if umap_key not in adata.obsm:
        if "umap" in adata.obsm:
            umap_key = "umap"
        else:
            print(f"[警告] 找不到 umap 数据 '{umap_key}'，跳过")
            return

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0)   # figure 背景透明
    ax.set_facecolor('none') # axes 背景透明
    
    sc.pl.embedding(
        adata, basis=umap_key, color=color_col,
        ax=ax, show=False, legend_loc="right margin",
        title=f"UMAP ({umap_key})  –  {color_col}"
    )
    ax.set_aspect("equal")
    safe_name = umap_key.replace("/", "_")
    safe_color = color_col.replace("/", "_")
    save_path = os.path.join(output_dir, f"umap_{safe_name}_{safe_color}.pdf")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True, facecolor='none')
    plt.close(fig)
    print(f"UMAP 图已保存: {save_path}")


def create_custom_cmap_from_hex(name, hex_colors, positions=None):
    rgb_colors = [mcolors.hex2color(c) for c in hex_colors]
    if positions is None:
        positions = np.linspace(0, 1, len(hex_colors))
    else:
        positions = np.array(positions) / 100.0
    cmap = mcolors.LinearSegmentedColormap.from_list(name, 
                                                       list(zip(positions, rgb_colors)))
    return cmap

hex_colors = ['#b4dae5', '#fcfdff','#fc0703', '#8c000a']
positions = [0, 33, 66, 100] 

my_cmap = create_custom_cmap_from_hex('my_custom_cmap', hex_colors, positions)
use_genes = [
    'Grin2a', 'Grin2b', 'Slc32a1', 'Gad1', 'Pou3f1', 'Syt6', 'Fezf2',
    'Tshz2', 'Rorb', 'Nptx2', 'Cux2', 'Otof', 'Sncg', 'Lamp5', 'Cnr1',
    'Pvalb', 'Reln', 'Lhx6', 'Gjb6', 'Mog', 'Serpinf1', 'Cldn5', 'Ctss', 'Pdgfra'
]

colormap = {
    'L5 ET 1': '#375998', 'L5 ET 2': '#c04424', 'L6 CT 1': '#18bd50',
    'L6 CT 2': '#f70086', 'L6 CT 3': '#f700bd', 'L5/6 NP': '#f7e43c', 'L6 CT 4': '#f5e1a2', 
    'L4/5 IT 2': '#f69f9e', 'L5 IT 3': '#b00e9f', 'L5 IT 1': '#278357', 'L6 IT 2': '#7822b2', 
    'L2/3 IT 2': '#91ad29', 'L2/3 IT 1': '#1effcd', 'L2/3 IT 3': '#2ed8ff', 'L2/3 IT 4': '#dc9cfa', 'L4/5 IT 1': '#a900fa', 
    'L5 IT 2': '#84660c', 'L6 IT 1': '#bd73a5', 
    'Lamp5 3': '#7e2d20', 'Lamp5 1': '#aaf302', 'Lamp5 2': '#bbccff', 'Sncg 1': '#bd0032', 'Vip 1': '#f1a623', 'Vip 2': '#b2446c', 
    'Pvalb 5': '#f28301', 'Pvalb 6': '#a2c9ee', 'Pvalb 1': '#3b00fa', 'Pvalb 2': '#222222', 'Pvalb 3': '#f1c201', 'Pvalb 4': '#865691', 'Sst 1': '#838381', 
    'Sst 5': '#f89279', 'Sst 6': '#614e97', 'Sst 2': '#008756', 'Sst 4': '#1a66a2', 
    'Sncg 2': '#c1b080', 'Sst 3': '#e28dac', 
    'Astro 3': '#f5212f', 'Astro 1': '#5b5156', 'Astro 2': '#e1e1e1', 
    'Oligo 1': '#2a7f91', 'Oligo 2': '#d75ff7', 
    'VLMC': '#dbd200', 
    'Endo 5': '#af0168', 'Endo 3': '#3182fd', 'Endo 4': '#fdae16', 'Endo 1': '#fd02f9', 'Endo 2': '#0fff33', 
    'Microglia 1': '#b4eeb4', 'Microglia 2': '#7ed6d0',
    'OPC 1': '#683b79', 'OPC 2': '#67afff',
}
        


def dotplot(adata, 
    target_genes, 
    groupby,
    categories_order,
    tick_font,
    font_name,
    my_cmap,
    save_path,
    use_raw):
    plot = sc.pl.dotplot(
        adata, 
        var_names=use_genes, 
        groupby=groupby, 
        show=False, 
        swap_axes=True, 
        categories_order=categories_order,
        standard_scale='var',
        cmap=my_cmap,
        edgecolor='none',
        use_raw=use_raw
    )
    ax = plot['mainplot_ax']
    for text in ax.get_yticklabels():
        text.set_fontproperties(tick_font)
        text.set_fontsize(13) 
    for text in ax.get_xticklabels():
        text.set_fontsize(13) 
        text.set_ha('right')
        text.set_rotation(45)
        text.set_fontfamily(font_name)
    plt.savefig(
        save_path,
        transparent=True, 
        facecolor='none',
        edgecolor='none',
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
        
    
def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    save_dir = cfg["paths"]["save_dir"]
    output_dir = args.output_dir or os.path.join(save_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    adata_path = args.adata or os.path.join(
        save_dir, "integrated_adatas", "integrated.h5ad"
    )
    print(f"读取 AnnData: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    
    save_path = os.path.join(output_dir, "dotplot_leiden.pdf")
    dotplot(
        adata = adata, 
        target_genes = use_genes, 
        groupby = args.umap_cell_color,
        categories_order = None,
        tick_font = tick_font,
        font_name = font_name,
        my_cmap = my_cmap,
        save_path = save_path,
        use_raw = True
    )
    categories_order = [
        'L5 ET 1',  'L5 ET 2', 'L6 CT 1','L6 CT 2', 'L6 CT 3', 'L5/6 NP', 'L6 CT 4', 
        'L4/5 IT 2', 'L5 IT 3', 'L5 IT 1', 'L6 IT 2', 
        'L2/3 IT 2', 'L2/3 IT 1', 'L2/3 IT 3', 'L2/3 IT 4', 'L4/5 IT 1', 
        'L5 IT 2', 'L6 IT 1', 
        'Lamp5 3', 'Lamp5 1', 'Lamp5 2', 'Sncg 1', 'Vip 1', 'Vip 2', 
        'Pvalb 5', 'Pvalb 6', 'Pvalb 1', 'Pvalb 2', 'Pvalb 3', 'Pvalb 4', 'Sst 1', 
        'Sst 5', 'Sst 6', 'Sst 2', 'Sst 4', 
        'Sncg 2', 'Sst 3', 
        'Astro 3', 'Astro 1', 'Astro 2', 
        'Oligo 1', 'Oligo 2', 
        'VLMC', 
        'Endo 5', 'Endo 3', 'Endo 4', 'Endo 1', 'Endo 2', 
        'Microglia 1', 'Microglia 2',
        'OPC 1', 'OPC 2',
    ]
    
    save_path = os.path.join(output_dir, "dotplot_subcluster_raw.pdf")
    dotplot(
        adata = adata, 
        target_genes = use_genes, 
        groupby = 'subcluster',
        categories_order = categories_order,
        tick_font = tick_font,
        font_name = font_name,
        my_cmap = my_cmap,
        save_path = save_path,
        use_raw = True
    )
    
    # save_path = os.path.join(output_dir, "dotplot_subcluster_reconstructed.pdf")
    # dotplot(
    #     adata = adata, 
    #     target_genes = use_genes, 
    #     groupby = 'subcluster',
    #     categories_order = categories_order,
    #     tick_font = tick_font,
    #     font_name = font_name,
    #     my_cmap = my_cmap,
    #     save_path = save_path,
    #     use_raw = False
    # )
    
    
    
    # 确定 batch 列
    batch_col = args.batch_col
    if batch_col is None:
        # 尝试从 config 获取
        batch_col = cfg.get("batch_col", None)
    if batch_col is None:
        # 自动检测常见批次列
        for cand in ["batch", "sample", "library_id"]:
            if cand in adata.obs.columns:
                batch_col = cand
                break
    if batch_col is None or batch_col not in adata.obs.columns:
        print("[警告] 未找到批次列，跳过 batch 整合效果图。可用 --batch_col 指定")
        batch_col = None

    # ── 空间图 ──
    # 始终绘制两个 leiden 聚类空间图[
    for cluster_col in [args.umap_niche_color, args.umap_cell_color]:
        if cluster_col in adata.obs.columns:
            plot_spatial(adata = adata, 
                         color_col = cluster_col, 
                         output_dir = output_dir,
                         scale_factor = args.scale_factor, 
                         figsize = tuple(args.figsize), 
                         dpi = args.dpi, )
                        #  colormap = colormap)
        else:
            print(f"[警告] 列 '{cluster_col}' 不存在，跳过空间图")

    plot_umap(adata, "uamp", 'subcluster',
              output_dir, tuple(args.figsize), args.dpi)

    plot_umap(adata, "uamp", batch_col,
              output_dir, tuple(args.figsize), args.dpi)
    # ── cell embedding UMAP ──
    # 聚类着色
    plot_umap(adata, "cell_embedding_umap", args.umap_cell_color,
              output_dir, tuple(args.figsize), args.dpi)
    # batch 整合效果
    if batch_col:
        plot_umap(adata, "cell_embedding_umap", batch_col,
                  output_dir, tuple(args.figsize), args.dpi)

    # ── niche embedding UMAP ──
    # 聚类着色
    plot_umap(adata, "niche_embedding_umap", args.umap_niche_color,
              output_dir, tuple(args.figsize), args.dpi)
    # batch 整合效果
    if batch_col:
        plot_umap(adata, "niche_embedding_umap", batch_col,
                  output_dir, tuple(args.figsize), args.dpi)

    print(f"全部完成，结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
    
'''
python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/05_plot.py \
    --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v1.yaml \
    --spatial_color CellType \
    --umap_cell_color leiden_cell_embedding_r2.0 \
    --umap_niche_color leiden_niche_embedding_r1.0 \
    --batch_col slice

'''
'''
python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/05_plot.py \
    --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v3.yaml \
    --spatial_color subcluster \
    --umap_cell_color leiden_cell_embedding_r3.0 \
    --umap_niche_color leiden_niche_embedding_r1.0 \
    --batch_col slice
'''