import yaml

# 加载 yaml 格式的 meta 信息
with open('/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/07_macaque_brain_mae/cache/meta.yaml', 'r') as f:
    meta = yaml.safe_load(f)

# 动态计算总细胞数
total_cells = sum(info["n_cells"] for info in meta["slice_info"])
print(f"总细胞数: {total_cells}")
# 总细胞数: 30975704