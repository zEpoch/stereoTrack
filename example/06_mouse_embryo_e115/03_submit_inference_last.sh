#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
TRAIN_DIR="${WORK_DIR}/out/06_mouse_embryo_e115_mae_v1_train"
CHECKPOINT_FILE="${TRAIN_DIR}/checkpoints/last.ckpt"
LOG_DIR="${TRAIN_DIR}/inference_logs"
RUNNER="${WORK_DIR}/example/06_mouse_embryo_e115/03_run_inference.sh"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_FILE}" >&2
  exit 1
fi

dsub \
  -n "06_embryo_e115_inference" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=16;gpu=1;mem=180000" \
  -T "172800" \
  -N 1 \
  -oo "${LOG_DIR}/inference_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/inference_${TIMESTAMP}.err" \
  bash "${RUNNER}" "${CHECKPOINT_FILE}"

echo "Inference submitted. Logs: ${LOG_DIR}"
