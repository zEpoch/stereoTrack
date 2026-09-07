#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
SCRIPT="${ROOT}/example/02_Chen_Cell/imputation_05_lognormscale_method_benchmark.sh"
LOG_DIR="${ROOT}/out/02_macaque_brain_mae_v5/job_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

dsub \
  -n "macaque_v1_lognormscale_bench" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=64;mem=240000" \
  -T "172800" \
  -N 1 \
  -oo "${LOG_DIR}/macaque_v1_lognormscale_bench_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/macaque_v1_lognormscale_bench_${TIMESTAMP}.err" \
  bash "${SCRIPT}"

echo "Submitted macaque v1 lognormscale method benchmark; logs: ${LOG_DIR}"
