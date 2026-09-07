#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/rapids_sc.sh

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
SCRIPT="${WORK_DIR}/example/04_three_species/04_compare_balanced_align.py"

export PYTHONUNBUFFERED=1
COMPARE_THREADS="${STEREOTRACK_COMPARE_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${COMPARE_THREADS}"
export OMP_NUM_THREADS="${COMPARE_THREADS}"
export MKL_NUM_THREADS="${COMPARE_THREADS}"

python -u "${SCRIPT}" "$@"
