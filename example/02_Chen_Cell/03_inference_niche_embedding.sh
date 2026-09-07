#!/bin/bash
set -euo pipefail

source "/home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh"

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG="${ROOT}/config/02_config_Chen_Cell_v5.yaml"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
OUTPUT_DIR="${MODEL_DIR}/gsmap_niche_embedding_by_sample"
DEVICE="${STEREOTRACK_DEVICE:-cuda:0}"
BATCH_SIZE="${STEREOTRACK_BATCH_SIZE:-4096}"

CHECKPOINT="${1:-}"
if [ -z "${CHECKPOINT}" ]; then
  CHECKPOINT="$(find "${MODEL_DIR}/checkpoints" -maxdepth 1 -type f -name 'best*.ckpt' | sort | tail -n 1)"
fi
if [ -z "${CHECKPOINT}" ] || [ ! -s "${CHECKPOINT}" ]; then
  echo "Checkpoint not found. Pass it explicitly: bash $0 /path/to/best.ckpt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python "${ROOT}/example/02_Chen_Cell/inference.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --embedding-only \
  --embedding-dtype float16 \
  --embedding-output-format npy \
  --compression lzf \
  --skip-existing \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --start-slice 28 \
  --end-slice 30 
