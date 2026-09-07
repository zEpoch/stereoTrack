#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG="${ROOT}/config/02_config_Chen_Cell_v5.yaml"
LOG_DIR="${ROOT}/out/02_macaque_brain_mae_v5/job_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

dsub \
  -n "02_macaque_brain_mae_v5" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=120;gpu=4;mem=180000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/train_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/train_${TIMESTAMP}.err" \
  bash "${ROOT}/train_pl.sh" "${CONFIG}"

echo "Submitted macaque training; logs: ${LOG_DIR}"
