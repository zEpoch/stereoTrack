#!/bin/bash
# dsub submit script for MERFISH mouse brain MAE training.
#
# Before training, build the cache once:
#   source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh
#   python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/05_merfish_mouseBrain/01_preprocess.py \
#     --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/05_config_merfish_mouseBrain.yaml
#
# Submit training:
#   bash /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/05_merfish_mouseBrain/02_submit_train.sh

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/05_config_merfish_mouseBrain.yaml"
LOG_DIR="${WORK_DIR}/out/05_merfish_mouseBrain_mae_v1_train/job_logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "05_merfish_mouseBrain_mae_v1" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=126;gpu=4;mem=180000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/train_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/train_${TIMESTAMP}.err" \
    bash "${WORK_DIR}/train_pl.sh" "${CONFIG_FILE}"

echo "作业已提交，日志目录: ${LOG_DIR}"
