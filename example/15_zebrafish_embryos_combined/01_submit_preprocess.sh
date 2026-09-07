#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${WORK_DIR}/out/15_zebrafish_embryos_combined_mae_v1/job_logs"
RUNNER="${WORK_DIR}/example/15_zebrafish_embryos_combined/01_run_preprocess.sh"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
  -n "15_zfish_combined_preprocess" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=32;mem=120000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/preprocess_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/preprocess_${TIMESTAMP}.err" \
  bash "${RUNNER}" "$@"

echo "Preprocessing submitted. Logs: ${LOG_DIR}"
