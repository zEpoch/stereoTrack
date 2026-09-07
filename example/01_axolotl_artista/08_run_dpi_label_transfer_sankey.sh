#!/usr/bin/env bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -u "${SCRIPT_DIR}/08_dpi_label_transfer_sankey.py" \
  --input-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_axolotl_artista_downstream_mae_v1_train/inference_full_adatas \
  --output-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_axolotl_artista_downstream_mae_v1_train/dpi_label_transfer_sankey \
  --embedding-key niche_embedding \
  --label-key Annotation \
  --dpi-order "2,5,10,15,20" \
  --classifier knn \
  --n-neighbors 30 \
  --weights distance \
  --filter-prop 0.08 \
  --min-count 50
