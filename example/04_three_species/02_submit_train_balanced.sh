#!/bin/bash
# dsub submission: species-balanced baseline without alignment loss

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${WORK_DIR}/out/04_three_species_mae_v1_train_balanced_v1/job_logs"
CONFIG_FILE="${WORK_DIR}/config/04_config_three_species_balanced.yaml"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "04_three_species_balanced_v1" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=128;gpu=4;mem=180000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/train_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/train_${TIMESTAMP}.err" \
    bash "${WORK_DIR}/train_pl.sh" "${CONFIG_FILE}"

echo "species-balanced baseline submitted; logs: ${LOG_DIR}"
