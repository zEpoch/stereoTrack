#!/bin/bash
# dsub submit script for RAPIDS embedding clustering.

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${WORK_DIR}/out/05_merfish_mouseBrain_mae_v1_3axes_train/rapids_analysis/job_logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "05_merfish_mouseBrain_rapids_cluster" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=32;gpu=1;mem=200000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/rapids_cluster_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/rapids_cluster_${TIMESTAMP}.err" \
    bash "${WORK_DIR}/example/05_merfish_mouseBrain/04_run_rapids_cluster.sh"

echo "RAPIDS clustering job submitted. Logs: ${LOG_DIR}"
