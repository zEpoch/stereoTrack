#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${WORK_DIR}/out/01_axolotl_artista_downstream_mae_v1/job_logs"
RUNNER="${WORK_DIR}/example/01_axolotl_artista/04_run_preprocess_downstream.sh"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
  -n "01_axolotl_down_preprocess" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=16;mem=120000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/downstream_preprocess_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/downstream_preprocess_${TIMESTAMP}.err" \
  bash "${RUNNER}" "$@"

echo "Downstream preprocessing submitted. Logs: ${LOG_DIR}"
