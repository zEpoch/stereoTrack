# Zebrafish embryo measured weMERFISH

This example trains StereoTrack on the measured zebrafish embryo h5ad files in:

`/home/share/huadjyin/home/zhoutao3/tracks/example_data/15_zebrafish_embryos/measured`

Each embryo file is one training batch. The six measured files share 495 genes.
The current preprocessing uses the input `X` directly and intentionally skips
`normalize_total`, `log1p`, and `scale`.

## Spatial Coordinates

- `A_50p` and `B_75p` files use `obsm['global_sphere']`.
- `C_6s` files use `obsm['spatial']`.
- The selected coordinates are copied to `obsm['ccf']` in the cache and
  inference outputs.

## Run

```bash
bash example/07_zebrafish_embryos/01_submit_preprocess.sh
bash example/07_zebrafish_embryos/02_submit_train.sh
bash example/07_zebrafish_embryos/03_submit_inference_last.sh
```

To make training wait for preprocessing:

```bash
bash example/07_zebrafish_embryos/02_submit_train.sh <preprocess_job_id>
```

The config is:

`/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/07_config_zebrafish_embryos.yaml`

