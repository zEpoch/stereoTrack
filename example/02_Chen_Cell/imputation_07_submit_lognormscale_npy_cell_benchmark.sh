#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
LOG_DIR="${MODEL_DIR}/job_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

dsub \
  -n "macaque_cell_lognorm_bench" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=32;mem=200000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/macaque_cell_lognorm_benchmark_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/macaque_cell_lognorm_benchmark_${TIMESTAMP}.err" \
  bash "${ROOT}/example/02_Chen_Cell/imputation_07_lognormscale_npy_cell_benchmark.sh"

echo "Submitted macaque cell-level lognormscale method benchmark."
echo "Logs: ${LOG_DIR}/macaque_cell_lognorm_benchmark_${TIMESTAMP}.{out,err}"
