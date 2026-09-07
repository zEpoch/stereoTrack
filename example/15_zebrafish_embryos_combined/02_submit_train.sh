#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/15_config_zebrafish_embryos_combined.yaml"
LOG_DIR="${WORK_DIR}/out/15_zebrafish_embryos_combined_mae_v1_train/job_logs"
DEPENDENCY_JOB_ID="${1:-${DEPENDENCY_JOB_ID:-}}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DSUB_ARGS=(
  -n "15_zfish_combined_mae_v1"
  -A "root.project.P23Z10200N0876"
  -R "cpu=32;gpu=1;mem=120000"
  -T "86400"
  -N 1
  -oo "${LOG_DIR}/train_${TIMESTAMP}.out"
  -eo "${LOG_DIR}/train_${TIMESTAMP}.err"
)

if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  DSUB_ARGS+=(-D "${DEPENDENCY_JOB_ID}=SUCCEEDED")
fi

dsub "${DSUB_ARGS[@]}" bash "${WORK_DIR}/train_pl.sh" "${CONFIG_FILE}"

echo "Training submitted. Logs: ${LOG_DIR}"
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  echo "Dependency: ${DEPENDENCY_JOB_ID}=SUCCEEDED"
fi
