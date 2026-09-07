#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/02_config_Chen_Cell_cortex_v2.yaml"
TRAIN_DIR="${WORK_DIR}/out/02_macaque_brain_cortex_v2_mae_v1_train"
OUTPUT_DIR="${TRAIN_DIR}/niche_embedding_by_sample"
DEVICE="${STEREOTRACK_DEVICE:-cuda:0}"
BATCH_SIZE="${STEREOTRACK_BATCH_SIZE:-4096}"

CHECKPOINT_FILE="${1:-${TRAIN_DIR}/checkpoints/last.ckpt}"
if [[ $# -gt 0 ]]; then
  shift
fi

python -u "${WORK_DIR}/example/02_Chen_Cell/inference.py" \
  --config "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --embedding-only \
  --embedding-dtype float16 \
  --embedding-output-format npy \
  --compression lzf \
  --skip-existing \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  "$@"

