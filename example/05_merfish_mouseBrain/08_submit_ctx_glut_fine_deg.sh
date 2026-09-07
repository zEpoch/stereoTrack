#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
OUT_DIR="${WORK_DIR}/out/05_merfish_mouseBrain_mae_v1_3axes_train/fine_celltype_deg_ctx_glut"
LOG_DIR="${OUT_DIR}/job_logs"
RUNNER="${WORK_DIR}/example/05_merfish_mouseBrain/08_run_ctx_glut_fine_deg.sh"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
  -n "05_ctx_glut_fine_deg" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=32;gpu=1;mem=180000" \
  -T "172800" \
  -N 1 \
  -oo "${LOG_DIR}/ctx_glut_deg_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/ctx_glut_deg_${TIMESTAMP}.err" \
  bash "${RUNNER}" "$@"

echo "CTX glut fine DEG submitted. Logs: ${LOG_DIR}"
