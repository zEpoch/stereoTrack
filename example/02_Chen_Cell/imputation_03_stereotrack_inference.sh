#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG="${ROOT}/config/02_config_Chen_Cell_v5.yaml"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
OUTPUT_DIR="${MODEL_DIR}/imputation_benchmark/stereotrack_all_gene_inference_adatas"

source "/home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CHECKPOINT="${1:-}"
if [ -z "${CHECKPOINT}" ]; then
  CHECKPOINT="$(find "${MODEL_DIR}/checkpoints" -maxdepth 1 -type f -name 'best*.ckpt' | sort | tail -n 1)"
fi
if [ -z "${CHECKPOINT}" ] || [ ! -s "${CHECKPOINT}" ]; then
  echo "Checkpoint not found. Pass it explicitly: bash $0 /path/to/best.ckpt" >&2
  exit 1
fi

python "${ROOT}/example/02_Chen_Cell/inference.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --all-genes \
  --batch_size 4096 \
  --device cuda:0
