#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
RUNNER="${WORK_DIR}/example/02_Chen_Cell_cortex_v2/02_run_train.sh"
LOG_DIR="${WORK_DIR}/out/02_macaque_brain_cortex_v2_mae_v1_train/job_logs"
DEPENDENCY_JOB_ID="${1:-${DEPENDENCY_JOB_ID:-}}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DSUB_ARGS=(
  -n "02_macaque_cortex_v2_train"
  -A "root.project.P23Z10200N0876"
  -R "cpu=128;gpu=4;mem=180000"
  -T "172800"
  -N 1
  -oo "${LOG_DIR}/train_${TIMESTAMP}.out"
  -eo "${LOG_DIR}/train_${TIMESTAMP}.err"
)

if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  DSUB_ARGS+=(-D "${DEPENDENCY_JOB_ID}=SUCCEEDED")
fi

dsub "${DSUB_ARGS[@]}" bash "${RUNNER}"

echo "Training submitted. Logs: ${LOG_DIR}"
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  echo "Dependency: ${DEPENDENCY_JOB_ID}=SUCCEEDED"
fi
