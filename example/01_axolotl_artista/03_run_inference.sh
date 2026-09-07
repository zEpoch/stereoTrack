#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/01_config_axolotl_artista.yaml"
DEFAULT_CHECKPOINT="${WORK_DIR}/out/01_axolotl_artista_mae_v1_train/checkpoints/last.ckpt"
OUTPUT_DIR="${WORK_DIR}/out/01_axolotl_artista_mae_v1_train/inference_full_adatas"
CHECKPOINT_FILE="${1:-${DEFAULT_CHECKPOINT}}"
if [[ $# -gt 0 ]]; then
  shift
fi

python -u "${WORK_DIR}/example/01_axolotl_artista/03_inference.py" \
  --config "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
