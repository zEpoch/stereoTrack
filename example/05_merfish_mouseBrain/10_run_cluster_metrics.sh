#!/bin/bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack}"
INPUT="${INPUT:-${WORK_DIR}/out/05_merfish_mouseBrain_mae_v1_3axes_train/rapids_analysis/merfish_mouseBrain_concat_embeddings.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/out/05_merfish_mouseBrain_mae_v1_3axes_train/cluster_evaluation}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

python3 -u "${WORK_DIR}/example/05_merfish_mouseBrain/10_evaluate_cluster_metrics.py" \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
