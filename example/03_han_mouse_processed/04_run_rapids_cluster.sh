#!/bin/bash
# Run RAPIDS clustering for Han mouse processed inference outputs.
#
# Usage:
#   bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_han_mouse_processed/04_run_rapids_cluster.sh

set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/rapids_sc.sh

python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/03_han_mouse_processed/04_rapids_cluster_embeddings.py \
  --input-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/inference_adatas \
  --output-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/rapids_analysis \
  --cell-label-cols cell_subclass cell_class cell_cluster \
  --niche-label-cols layer region main_region region_description \
  --cell-n-neighbors 30 \
  --cell-resolution 1 \
  --niche-n-neighbors 30 \
  --niche-resolution 1 \
  --algorithm ivfflat \
  --resolution 1.0