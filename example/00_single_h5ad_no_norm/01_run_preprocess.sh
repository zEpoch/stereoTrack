#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh
source /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/00_single_h5ad_no_norm/00_config.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

prepare_config
cd "${WORK_DIR}"
python -u example/07_zebrafish_embryos/01_preprocess.py --config "${CONFIG_FILE}" "$@"
