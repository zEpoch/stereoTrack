#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
OUTPUT_DIR="${WORK_DIR}/out/04_three_species_cross_species/ontology"
LOG_DIR="${OUTPUT_DIR}/job_logs"
RUNNER="${WORK_DIR}/example/04_three_species/06_run_common_ontology.sh"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "04_three_species_ontology" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=16;mem=120000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/ontology_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/ontology_${TIMESTAMP}.err" \
    bash "${RUNNER}"

echo "Common ontology audit submitted; logs: ${LOG_DIR}"
