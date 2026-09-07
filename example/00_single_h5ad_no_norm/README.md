# Single h5ad no-normalization local pipeline

This template preprocesses one already-transformed h5ad file, trains StereoTrack for 50 epochs, then writes full inference outputs locally.

Required environment variable:

```bash
export DATA_PATH=/path/to/your_file.h5ad
```

Useful optional variables:

```bash
export RUN_NAME=my_single_dataset_no_norm
export SOURCE_SPATIAL_KEY=spatial
export GRAPH_METHOD=knn
export N_NEIGHBORS=12
export PATCH_MODE=axis
export PATCH_AXES=3
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda:0
```

Run locally:

```bash
bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/00_single_h5ad_no_norm/01_run_preprocess.sh
bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/00_single_h5ad_no_norm/02_run_train.sh
bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/00_single_h5ad_no_norm/03_run_inference.sh
```

Outputs:

- Preprocess cache: `tracks/stereoTrack/out/${RUN_NAME}/cache`
- Checkpoints: `tracks/stereoTrack/out/${RUN_NAME}_train/checkpoints`
- Inference h5ad: `tracks/stereoTrack/out/${RUN_NAME}_train/inference_full_adatas`

The generated config is written to `tracks/stereoTrack/config/${RUN_NAME}.yaml`.
