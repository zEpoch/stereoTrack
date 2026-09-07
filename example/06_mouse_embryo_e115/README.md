# E11.5 mouse embryo neural subset

This workflow trains StereoTrack on a neural-related subset of the registered
E11.5 mouse embryo.

The input is
`/home/share/huadjyin/home/zhoutao3/tracks/example_data/05_mouse_embyro_e115/mouse_E11.5_embryo.h5ad`.
Coordinates are read from `obsm['z_correction']`, stored as `obsm['ccf']` in
the preprocessing cache, and split by the registered z axis into 3D slabs.

The current preprocessing keeps cells whose `obs['mapped_celltype']` is in the
neural cell-type whitelist in
`/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/06_config_mouse_embryo_e115.yaml`.
It does not run HVG selection: `hvg_n_top_genes: 0` means all input genes are
used, followed by per-slab `normalize_total`, `log1p`, and sparse scaling.

## Run

```bash
bash example/06_mouse_embryo_e115/01_submit_preprocess.sh
bash example/06_mouse_embryo_e115/02_submit_train.sh
bash example/06_mouse_embryo_e115/03_submit_inference_last.sh
```

If you want training to wait for a preprocessing job, pass the preprocessing
job id:

```bash
bash example/06_mouse_embryo_e115/02_submit_train.sh 342954
```

Trajectory analysis is intentionally not included in this folder right now.
It should be redesigned separately after inspecting the new neural-subset
embeddings.

