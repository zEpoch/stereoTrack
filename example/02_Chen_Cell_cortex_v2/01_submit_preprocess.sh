#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${WORK_DIR}/out/02_macaque_brain_cortex_v2_mae_v1/job_logs"
RUNNER="${WORK_DIR}/example/02_Chen_Cell_cortex_v2/01_run_preprocess.sh"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
  -n "02_macaque_cortex_v2_preprocess" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=32;mem=160000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/preprocess_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/preprocess_${TIMESTAMP}.err" \
  bash "${RUNNER}" "$@"

echo "Preprocessing submitted. Logs: ${LOG_DIR}"

