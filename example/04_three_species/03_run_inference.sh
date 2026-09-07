#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash 03_run_inference.sh CONFIG CHECKPOINT OUTPUT_DIR [BATCH_SIZE]"
    exit 2
fi

CONFIG_FILE="$1"
CHECKPOINT_FILE="$2"
OUTPUT_DIR="$3"
BATCH_SIZE="${4:-4096}"
WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/12.1.0
module load libs/nccl/2.19.3_cuda12.0
module load libs/cudnn/9.8.0_cuda12
module load libs/openblas/0.3.26_gcc9.3.0

export DGLBACKEND=pytorch
export OMP_NUM_THREADS=4

conda activate stereotrack_2
cd "${WORK_DIR}"

if [ ! -s "${CHECKPOINT_FILE}" ]; then
    echo "Checkpoint does not exist or is empty: ${CHECKPOINT_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"

python example/04_three_species/03_inference_v1.py \
    --config "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --device cuda:0 \
    --skip-existing
