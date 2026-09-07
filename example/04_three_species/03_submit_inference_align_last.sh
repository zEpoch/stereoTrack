#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
MODEL_DIR="${WORK_DIR}/out/04_three_species_mae_v1_train_align_v1"
CONFIG_FILE="${WORK_DIR}/config/04_config_three_species_align.yaml"
CHECKPOINT_FILE="${MODEL_DIR}/checkpoints/last.ckpt"
OUTPUT_DIR="${MODEL_DIR}/inference_adatas"
LOG_DIR="${MODEL_DIR}/inference_job_logs"
RUNNER="${WORK_DIR}/example/04_three_species/03_run_inference.sh"
BATCH_SIZE="${BATCH_SIZE:-4096}"

if [ ! -s "${CHECKPOINT_FILE}" ]; then
    echo "Checkpoint does not exist or is empty: ${CHECKPOINT_FILE}"
    echo "Submit inference after alignment training has finished."
    exit 1
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "04_three_species_infer_align_last" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=32;gpu=1;mem=180000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/inference_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/inference_${TIMESTAMP}.err" \
    bash "${RUNNER}" "${CONFIG_FILE}" "${CHECKPOINT_FILE}" "${OUTPUT_DIR}" "${BATCH_SIZE}"

echo "Alignment last.ckpt inference submitted; logs: ${LOG_DIR}"
