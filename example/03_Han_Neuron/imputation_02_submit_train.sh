#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG="${ROOT}/config/03_config_han_mouse_processed.yaml"
LOG_DIR="${ROOT}/out/03_han_mouse_processed_mae_v1_train/job_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

dsub \
  -n "03_han_mouse_processed_mae_v1" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=128;gpu=4;mem=180000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/train_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/train_${TIMESTAMP}.err" \
  bash "${ROOT}/train_pl.sh" "${CONFIG}"

echo "Submitted Han training; logs: ${LOG_DIR}"
