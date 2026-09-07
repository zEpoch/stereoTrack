# 02_Chen_Cell cortex_v2

This is the v2 macaque cortex pipeline for StereoTrack.

Differences from the older `02_Chen_Cell` workflow:

- Input h5ad files are read from `tracks/example_data/07_macaque_brain/cortex_v2`.
- Spatial graph construction uses `adata.obsm["spatial"]`.
- Expression values are used directly from `adata.X`; no `normalize_total`, `log1p`, or `scale` is applied.
- Outputs are written to separate `02_macaque_brain_cortex_v2_mae_v1` directories.

Local run:

```bash
cd /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack
bash example/02_Chen_Cell_cortex_v2/01_run_preprocess.sh
bash example/02_Chen_Cell_cortex_v2/02_run_train.sh
bash example/02_Chen_Cell_cortex_v2/03_run_inference_embeddings.sh
```

Use `03_run_inference_full.sh` only when you need decoded expression matrices; it can write very large h5ad files.

