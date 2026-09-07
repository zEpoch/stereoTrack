#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh
source /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/00_single_h5ad_no_norm/00_config.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

prepare_config

CHECKPOINT_FILE="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [ -z "${CHECKPOINT_FILE}" ]; then
  CHECKPOINT_FILE="$(find "${SAVE_DIR}/checkpoints" -maxdepth 1 -type f -name 'best*.ckpt' | sort | tail -n 1)"
fi
if [ -z "${CHECKPOINT_FILE}" ] || [ ! -s "${CHECKPOINT_FILE}" ]; then
  CHECKPOINT_FILE="${SAVE_DIR}/checkpoints/last.ckpt"
fi
if [ ! -s "${CHECKPOINT_FILE}" ]; then
  echo "[error] checkpoint not found in ${SAVE_DIR}/checkpoints" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${SAVE_DIR}/inference_full_adatas}"
mkdir -p "${OUTPUT_DIR}"

cd "${WORK_DIR}"
python -u example/06_mouse_embryo_e115/03_inference.py \
  --config "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${INFER_BATCH_SIZE}" \
  --all-genes \
  --write-cell-layer \
  "$@"
