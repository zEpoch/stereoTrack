#!/bin/bash
# dsub submit script for Han mouse processed MAE training.
#
# Before training, build the cache once:
#   python example/03_han_mouse_processed/01_preprocess.py --config config/03_config_han_mouse_processed.yaml
#
# Submit training:
#   bash example/03_han_mouse_processed/02_submit_train.sh

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/03_config_han_mouse_processed.yaml"
LOG_DIR="${WORK_DIR}/out/03_han_mouse_processed_mae_v1_train/job_logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "03_han_mouse_processed_mae_v1" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=128;gpu=4;mem=180000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/train_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/train_${TIMESTAMP}.err" \
    bash "${WORK_DIR}/train_pl.sh" "${CONFIG_FILE}"

echo "作业已提交，日志目录: ${LOG_DIR}"
