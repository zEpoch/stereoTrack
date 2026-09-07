#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
OUT_DIR="${WORK_DIR}/out/01_axolotl_artista_mae_v1"
LOG_DIR="${OUT_DIR}/job_logs"
RUNNER="${WORK_DIR}/example/01_axolotl_artista/01_run_preprocess.sh"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
  -n "01_axolotl_preprocess" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=32;mem=160000" \
  -T "172800" \
  -N 1 \
  -oo "${LOG_DIR}/preprocess_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/preprocess_${TIMESTAMP}.err" \
  bash "${RUNNER}" "$@"

echo "Preprocessing submitted. Logs: ${LOG_DIR}"
