#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
LOG_DIR="${MODEL_DIR}/job_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

dsub \
  -n "stereotrack_macaque_expr_npy" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=32;gpu=1;mem=260000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/stereotrack_expr_npy_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/stereotrack_expr_npy_${TIMESTAMP}.err" \
  bash "${ROOT}/example/02_Chen_Cell/imputation_03_run_stereotrack_npy_inference.sh"

echo "Submitted StereoTrack macaque expression npy inference."
echo "Logs: ${LOG_DIR}/stereotrack_expr_npy_${TIMESTAMP}.{out,err}"
