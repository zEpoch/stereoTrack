#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
OUTPUT_DIR="${WORK_DIR}/out/04_three_species_balanced_vs_align"
LOG_DIR="${OUTPUT_DIR}/job_logs"
RUNNER="${WORK_DIR}/example/04_three_species/04_run_compare_balanced_align.sh"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "04_three_species_compare" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=16;mem=120000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/compare_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/compare_${TIMESTAMP}.err" \
    bash "${RUNNER}" --all-cells-streaming

echo "All-cell balanced-vs-alignment comparison submitted; logs: ${LOG_DIR}"
