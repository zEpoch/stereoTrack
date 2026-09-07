#!/bin/bash
set -euo pipefail

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/12.1.0
module load libs/nccl/2.19.3_cuda12.0
module load libs/cudnn/9.8.0_cuda12
module load libs/openblas/0.3.26_gcc9.3.0
conda activate stereotrack_2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG_FILE="${WORK_DIR}/config/06_config_mouse_embryo_e115.yaml"

python -u "${WORK_DIR}/example/06_mouse_embryo_e115/01_preprocess.py" \
  --config "${CONFIG_FILE}" "$@"
