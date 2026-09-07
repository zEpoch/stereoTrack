#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
TRAIN_DIR="${WORK_DIR}/out/02_macaque_brain_cortex_v2_mae_v1_train"
LOG_DIR="${TRAIN_DIR}/inference_logs"
RUNNER="${WORK_DIR}/example/02_Chen_Cell_cortex_v2/03_run_inference_full.sh"
DEPENDENCY_JOB_ID="${1:-${DEPENDENCY_JOB_ID:-}}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DSUB_ARGS=(
  -n "02_macaque_cortex_v2_full"
  -A "root.project.P23Z10200N0876"
  -R "cpu=32;gpu=1;mem=160000"
  -T "172800"
  -N 1
  -oo "${LOG_DIR}/full_${TIMESTAMP}.out"
  -eo "${LOG_DIR}/full_${TIMESTAMP}.err"
)

if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  DSUB_ARGS+=(-D "${DEPENDENCY_JOB_ID}=SUCCEEDED")
fi

dsub "${DSUB_ARGS[@]}" bash "${RUNNER}" "${TRAIN_DIR}/checkpoints/last.ckpt"

echo "Full inference submitted. Logs: ${LOG_DIR}"
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  echo "Dependency: ${DEPENDENCY_JOB_ID}=SUCCEEDED"
fi

