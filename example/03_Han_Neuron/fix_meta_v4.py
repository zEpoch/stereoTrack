import os
import glob
import numpy as np
import yaml
import pickle

# 【请修改为你的实际 cache 目录路径！】
cache_dir = "/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_brain_mae_v4/cache"

def main():
    print(f"扫描目录: {cache_dir}")
    patch_files = glob.glob(os.path.join(cache_dir, "slice_*_patch_*.npz"))
    full_files = glob.glob(os.path.join(cache_dir, "slice_*_full_graph.npz"))
    
    slice_info_list = []
    slice_info_list_full = []
    input_dim = 0
    
    print(f"找到 {len(patch_files)} 个 Patch 文件, {len(full_files)} 个全图文件。正在读取信息...")
    
    # 1. 组装 Patch 列表 (供 train_pl.py)
    for f in patch_files:
        fname = os.path.basename(f)
        # 用 mmap 极速只读个头部
        data = np.load(f, mmap_mode="r")
        n_cells, n_genes = data["feat_shape"]
        slice_idx = int(data.get("slice_idx", fname.split("_")[1]))
        input_dim = int(n_genes)
        
        slice_info_list.append({
            "file": fname,
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "batch": f"slice_{slice_idx}", 
            "slice_idx": slice_idx
        })
        data.close()

    # 2. 组装 Full Graph 列表 (供 inference.py)
    for f in full_files:
        fname = os.path.basename(f)
        data = np.load(f, mmap_mode="r")
        n_cells = int(data["feat_shape"][0])
        slice_idx = int(fname.split("_")[1])
        
        slice_info_list_full.append({
            "batch": f"slice_{slice_idx}",
            "file": fname,
            "n_cells": n_cells,
        })
        data.close()

    # 根据 slice_idx 排序一下，保持整洁
    slice_info_list.sort(key=lambda x: (x["slice_idx"], x["file"]))
    slice_info_list_full.sort(key=lambda x: x["file"])

    # 3. 写出 Meta
    meta = {
        "common_genes": [], # 只有推理想知道具体基因名才用得到，大部分情况为空没关系
        "n_slices": len(full_files),
        "input_dim": input_dim,
        "patch_size": 16384, # 仅作记录 
        "patches": slice_info_list,
        "slice_info": slice_info_list_full
    }

    yaml_path = os.path.join(cache_dir, "meta.yaml")
    pkl_path = os.path.join(cache_dir, "meta.pkl")

    with open(yaml_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)

    with open(pkl_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"大功告成！已成功瞬间生成 metadata")

if __name__ == "__main__":
    main()