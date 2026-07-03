#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# dsub 投递脚本：03_han_mouse_mae 训练任务
# 
# 使用方式：
#   bash submit_03_mae.sh
#
# 请根据实际情况修改以下参数：
#   -q    队列名称（如 root.default、root.gpu 等）
#   -A    资源账户（如 root.balong1）
#   -R    资源需求（cpu、mem、gpu 数量）
#   -T    作业超时时间（秒）
# ═══════════════════════════════════════════════════════════════
VERSION='v4'
WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${WORK_DIR}/out/01_merfish_pfc_${VERSION}/job_logs"
CONFIG_FILE="${WORK_DIR}/config/01_config_merfish_pfc_${VERSION}.yaml"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

dsub \
    -n "01_merfish_pfc_${VERSION}" \
    -A "root.project.P23Z10200N0876" \
    -R "cpu=30;gpu=1;mem=100000" \
    -T "86400" \
    -N 1 \
    -oo "${LOG_DIR}/mae_train_${TIMESTAMP}.out" \
    -eo "${LOG_DIR}/mae_train_${TIMESTAMP}.err" \
    bash "${WORK_DIR}/train_pl.sh" "${CONFIG_FILE}"

echo "作业已提交，日志目录: ${LOG_DIR}"


