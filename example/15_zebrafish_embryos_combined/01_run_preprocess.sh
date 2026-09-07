#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/15_config_zebrafish_embryos_combined.yaml"

python -u "${WORK_DIR}/example/07_zebrafish_embryos/01_preprocess.py" \
  --config "${CONFIG_FILE}" "$@"
