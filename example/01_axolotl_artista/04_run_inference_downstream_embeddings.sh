#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/01_config_axolotl_artista_downstream.yaml"
TRAIN_DIR="${WORK_DIR}/out/01_axolotl_artista_downstream_mae_v1_train"
DEFAULT_CHECKPOINT="${TRAIN_DIR}/checkpoints/last.ckpt"
OUTPUT_DIR="${TRAIN_DIR}/inference_embeddings"
DEVICE="${STEREOTRACK_DEVICE:-cuda:0}"
BATCH_SIZE="${STEREOTRACK_BATCH_SIZE:-1024}"

CHECKPOINT_FILE="${1:-${DEFAULT_CHECKPOINT}}"
if [[ $# -gt 0 ]]; then
  shift
fi

python -u "${WORK_DIR}/example/01_axolotl_artista/03_inference.py" \
  --config "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --embeddings-only \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  "$@"
