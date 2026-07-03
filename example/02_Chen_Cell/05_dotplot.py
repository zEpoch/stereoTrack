import os

import gc
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

target_genes = [
            "SLC17A7", "GPR83", "CCBE1", "CUX2", "GPC5", "PDZD2", "CUX1", "MYLK", 
            "RORB", "IL1RAPL2", "ETV1", "TLE4", "SEMA3E", "GAD1", "GAD2", "ADARB2", "LAMP5", 
            "FBXL7", "KIT", "EYA4", "CALB2", "RELN", "VIP", "SOX6", "TRPS1", "ADAMTSL1", 
            "PVALB", "POSTN", "SST", "CALB1", "SLC1A2", "SLC1A3", "PTPRZ1", "PDGFRA", 
            "COL9A1", "PLP1", "ITGAM", "RGS5", "COL1A2"
        ]
categories_order = [
    'L2', 'L2/3', 'L2/3/4', 'L3/4', 'L3/4/5', 'L4',  'L4/5', 'L4/5/6', 'L5/6', 'L6', 'LAMP5',
    'RELN', 'VIP_RELN',  'VIP', 'PV_CHC', 'PVALB', 'SST', 'ASC', 'OPC', 'OLG', 'MG',  'EC', 'VLMC',
]

def create_custom_cmap_from_hex(name, hex_colors, positions=None):
    rgb_colors = [mcolors.hex2color(c) for c in hex_colors]
    if positions is None:
        positions = np.linspace(0, 1, len(hex_colors))
    else:
        positions = np.array(positions) / 100.0
    cmap = mcolors.LinearSegmentedColormap.from_list(name, 
                                                       list(zip(positions, rgb_colors)))
    return cmap



hex_colors = ['#FFFFFF', '#FDEEE6', '#F9D4C4', '#EC7455', '#C4201F']
positions = [0, 25, 50, 75, 100] 

my_cmap = create_custom_cmap_from_hex('my_custom_cmap', hex_colors, positions)


adata_path = '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v4/inference_adatas'
adata_list = [i for i in os.listdir(adata_path) if i.endswith('.h5ad')]
adatas = []
for adata_file in adata_list:
    adata = sc.read_h5ad(os.path.join(adata_path, adata_file))
    del adata.obsm['cell_embedding']
    del adata.obsm['niche_embedding']
    adatas.append(adata)

adata_concat = sc.concat(adatas, axis=0)
del adatas
gc.collect()


plot = sc.pl.dotplot(
    adata_concat, 
    var_names=target_genes, 
    groupby='SubClass', 
    show=False, 
    swap_axes=True, 
    categories_order=categories_order,
    standard_scale='var',
    cmap=my_cmap,
    edgecolor='none' 
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
    '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/figs/inference_dotplot.pdf',
    transparent=True, 
    facecolor='none',
    edgecolor='none',
    bbox_inches='tight',
    dpi=300
)
plt.close()


adata_path = '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v4/cache/adatas'
adata_list = [i for i in os.listdir(adata_path) if i.endswith('.h5ad')]
adatas = []
for adata_file in adata_list:
    adata = sc.read_h5ad(os.path.join(adata_path, adata_file))
    adata = adata[:,target_genes].copy()
    del adata.obsm
    del adata.obsp
    adatas.append(adata)

adata_concat = sc.concat(adatas, axis=0)
del adatas
gc.collect()
plot = sc.pl.dotplot(
    adata_concat, 
    var_names=target_genes, 
    groupby='SubClass', 
    show=False, 
    swap_axes=True, 
    categories_order=categories_order,
    standard_scale='var',
    cmap=my_cmap,
    edgecolor='none' 
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


plt.tight_layout()
plt.savefig(
    '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/figs/raw_dotplot.pdf',
    transparent=True, 
    facecolor='none',
    edgecolor='none',
    bbox_inches='tight',
    dpi=300
)
plt.close()



adata = sc.read_h5ad('/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v4/inference_adatas/slice_57.h5ad')
plot = sc.pl.embedding(
    adata,
    basis = 'ccf',
    color = 'SLC17A7',
    cmap = my_cmap,
    show = False
)
plot.set_aspect('equal')
plt.savefig(
    '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/figs/inference_SLC17A7.png',
    bbox_inches ='tight',
    dpi = 450)
plt.close()

adata = sc.read_h5ad('/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/02_macaque_brain_mae_v4/cache/adatas/slice_57.h5ad')
plot = sc.pl.embedding(
    adata,
    basis = 'ccf',
    color = 'SLC17A7',
    cmap = my_cmap,
    show = False
)
plot.set_aspect('equal')
plt.savefig(
    '/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/02_Chen_Cell/figs/raw_SLC17A7.png',
    bbox_inches ='tight',
    dpi = 450)
plt.close()


