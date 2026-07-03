
import os
import ot
import gc
import k3d
import torch
import trimesh
import warnings
from tqdm import *
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import scipy.sparse as sp
import matplotlib.cm as cm
from anndata import AnnData
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from numpy.random import RandomState
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import jaccard_score
from scipy.stats import fisher_exact, norm
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Optional, Tuple, Union
from matplotlib.colors import ListedColormap, rgb2hex
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.font_manager import fontManager, FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def gene_3d_plot(h5ad_input, 
                 gene, 
                 output_path, 
                 cmap,
                 camera,
                 basis="ccf",
                ):
    """
    根据给定的 h5ad 和基因列表生成 k3d 3D 散点图并保存为 HTML。
    :param h5ad_input: str (h5ad 文件路径) 或 AnnData 对象
    :param gene_list: list, 需要绘制的基因名称列表
    :param output_html: str, 保存的 HTML 文件名
    :param basis: str, obsm 中用于 3D 坐标的键 (如 'X_umap', 且需包含 3 列)
    """
    if isinstance(h5ad_input, str):
        adata = sc.read_h5ad(h5ad_input)
    else:
        adata = h5ad_input

    if basis not in adata.obsm.keys():
        raise ValueError(f"在 adata.obsm 中找不到 {basis}")
        
    coords = adata.obsm[basis] * 10
    if coords.shape[1] < 3:
        raise ValueError(f"坐标 {basis} 的维度小于 3，无法进行 3D 绘图")
    positions = coords[:, :3].astype(np.float32)

    mesh = trimesh.load('/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae/average_template_10_sur.stl')
    
    plot = k3d.plot(name="Gene Expression 3D Plot",background_color=0xffffff,camera_auto_fit=False)
    
    stl_mesh = k3d.mesh(
        vertices=mesh.vertices.astype(np.float32),
        indices  =mesh.faces.astype(np.uint32),
        color    =0xB0B0B0,
        opacity  =0.20,
        wireframe=False,
        name='mouseMesh'
    )
    plot += stl_mesh

    
    if gene not in adata.var_names:
        print(f"警告：基因 {gene} 不在数据集中，已跳过。")
        return
        
    expr = adata[:, gene].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = expr.flatten()

    mask = expr > 0

    if not np.any(mask):
        print(f"{gene}: 无表达细胞，跳过")
        return

    expr = expr[mask]
    positions = coords[mask] 

    expr_min, expr_max = expr.min(), expr.max()
    if expr_max > expr_min:
        expr_norm = (expr - expr_min) / (expr_max - expr_min)
    else:
        expr_norm = np.zeros_like(expr)

    cmap = cmap
    colors = cmap(expr_norm)

    colors_hex = (
        (np.clip(colors[:, 0] * 255, 0, 255).astype(np.uint32) << 16) |
        (np.clip(colors[:, 1] * 255, 0, 255).astype(np.uint32) << 8) |
        (np.clip(colors[:, 2] * 255, 0, 255).astype(np.uint32))
    )

    point_cloud = k3d.points(
        positions, 
        colors=colors_hex, 
        point_size=150.0, 
        # opacities = np.log(expr),
        opacities = expr,
        shader='3d', 
        name=gene
    )

    plot += point_cloud
    plot.camera = camera
    plot.grid_visible = False
    with open(os.path.join(output_path,gene+'.html'), 'w', encoding='utf-8') as f:
        f.write(plot.get_snapshot())
    return plot
import matplotlib.colors as mcolors # 新增导入
hex_colors = ['#0000FF','#0CFDF1', '#FFFF00', '#FF0000', '#D60000'] 
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', hex_colors)



adata_path = '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12_3/inference_adatas'
adata_list = [i for i in os.listdir(adata_path) if i.endswith('.h5ad')]
genes = ['Tcf7l2', 'Foxp2','Hes5','Slc1a3','Grik5','Lamp5',  'Cbln4', 'Rorb', 'Mbp', 'Trpc6', 'Scn4b', 'Sox11', 'Tcf4', 'Sox4', 'Grik2']
adatas = []
for adata_li in adata_list:
    adata = sc.read_h5ad(os.path.join(adata_path, adata_li))
    adata = adata[:,genes].copy()
    del adata.obsm['cell_embedding']
    del adata.obsm['niche_embedding']
    del adata.obsm['spatial']
    adatas.append(adata)
    gc.collect()


adata = ad.concat(adatas)
adata.obsm['ccf'] = adata.obsm['ccf'] * 10
genes = ['Foxp2','Tcf7l2', 'Hes5','Slc1a3','Grik5','Lamp5',  'Cbln4', 'Rorb', 'Mbp', 'Trpc6', 'Scn4b', 'Sox11', 'Tcf4', 'Sox4', 'Grik2']
camera = [11485.284267462972,
 -14993.204389281993,
 127606.14254467492,
 69173.64700316277,
 57841.691822745124,
 43527.33431330532,
 0.21578089746207177,
 -0.8402988312227976,
 -0.4973293461440416]
output_path="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v12_3/gene_3d_plot"
for gene in genes:
    if os.path.exists(os.path.join(output_path, gene+'.html')):
        print(f"{gene}.html 已存在，跳过绘制")
        continue
    plot = gene_3d_plot(adata, 
             gene, 
             output_path = output_path,
             basis="ccf",
             cmap = cmap,
             camera=camera)
    # break