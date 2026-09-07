#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${WORK_DIR}/out/06_mouse_embryo_e115_mae_v1/job_logs"
RUNNER="${WORK_DIR}/example/06_mouse_embryo_e115/01_run_preprocess.sh"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
  -n "06_embryo_e115_preprocess" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=64;mem=200000" \
  -T "172800" \
  -N 1 \
  -oo "${LOG_DIR}/preprocess_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/preprocess_${TIMESTAMP}.err" \
  bash "${RUNNER}"

echo "Preprocessing submitted. Logs: ${LOG_DIR}"
