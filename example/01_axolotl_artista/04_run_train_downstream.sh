#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/01_config_axolotl_artista_downstream.yaml"

cd "${WORK_DIR}"
python -u train_pl.py --config "${CONFIG_FILE}" "$@"

