#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/rapids_sc.sh

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python -u "${WORK_DIR}/example/04_three_species/06_build_common_ontology.py" "$@"
