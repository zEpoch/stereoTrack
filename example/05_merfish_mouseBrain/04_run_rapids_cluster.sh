#!/bin/bash
# Run RAPIDS clustering for MERFISH mouse brain inference outputs.
#
# Usage:
#   bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/05_merfish_mouseBrain/04_run_rapids_cluster.sh
#
# To choose a specific GPU on an interactive node:
#   CUDA_VISIBLE_DEVICES=3 bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/05_merfish_mouseBrain/04_run_rapids_cluster.sh

set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/rapids_sc.sh

CUDA_VISIBLE_DEVICES=3 python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/05_merfish_mouseBrain/04_rapids_cluster_embeddings.py \
  --input-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/inference_adatas \
  --output-dir /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/rapids_analysis \
  --cell-label-cols cell_type subclass_transfer cluster_id_transfer cell_type_ontology_term_id \
  --niche-label-cols major_brain_region ccf_region_name brain_section_label tissue tissue_type \
  --cell-n-neighbors 30 \
  --cell-resolution 15 \
  --niche-n-neighbors 30 \
  --niche-resolution 3 \
  --algorithm ivfflat \
  --resolution 1.0 \
  --run-umap
