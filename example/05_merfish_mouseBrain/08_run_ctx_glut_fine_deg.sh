#!/bin/bash
set -euo pipefail

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

source /home/share/huadjyin/home/zhoutao3/envs/rapids_sc.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"

python -u "${WORK_DIR}/example/05_merfish_mouseBrain/08_ctx_glut_fine_deg.py" "$@"
