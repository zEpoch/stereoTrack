#!/bin/bash
set -euo pipefail

source /home/share/huadjyin/home/zhoutao3/envs/rapids_sc.sh

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
SCRIPT="${WORK_DIR}/example/04_three_species/04_embedding_analysis.py"

INPUT_DIR="${INPUT_DIR:-${WORK_DIR}/out/04_three_species_mae_v1_train/inference_adatas}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/out/04_three_species_mae_v1_train/analysis_rapids_singlecell_balanced}"

MAX_CELLS_PER_FILE="${MAX_CELLS_PER_FILE:-3000}"
MAX_CELLS_PER_SPECIES="${MAX_CELLS_PER_SPECIES:-80000}"
MAX_TOTAL_CELLS="${MAX_TOTAL_CELLS:-240000}"
SAMPLING_MODE="${SAMPLING_MODE:-proportional}"
CLUSTER_METHOD="${CLUSTER_METHOD:-leiden}"
COMPUTE_BACKEND="${COMPUTE_BACKEND:-rapids}"
N_CLUSTERS="${N_CLUSTERS:-}"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export STEREOTRACK_ANALYSIS_THREADS="${STEREOTRACK_ANALYSIS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${STEREOTRACK_ANALYSIS_THREADS}"
export OMP_NUM_THREADS="${STEREOTRACK_ANALYSIS_THREADS}"
export MKL_NUM_THREADS="${STEREOTRACK_ANALYSIS_THREADS}"
export NUMEXPR_NUM_THREADS="${STEREOTRACK_ANALYSIS_THREADS}"

mkdir -p "${OUTPUT_DIR}"

N_CLUSTER_ARGS=()
if [[ -n "${N_CLUSTERS}" ]]; then
    N_CLUSTER_ARGS=(--n-clusters "${N_CLUSTERS}")
fi

echo "Python: $(which python)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Backend: ${COMPUTE_BACKEND}, cluster method: ${CLUSTER_METHOD}"
echo "Sampling: mode=${SAMPLING_MODE}, per-file=${MAX_CELLS_PER_FILE}, per-species=${MAX_CELLS_PER_SPECIES}, total=${MAX_TOTAL_CELLS}"

python -u "${SCRIPT}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --species-use marmoset macaque mouse \
    --layer-col layer \
    --region-col region \
    --celltype-col cell_type \
    --cluster-method "${CLUSTER_METHOD}" \
    --compute-backend "${COMPUTE_BACKEND}" \
    --max-cells-per-file "${MAX_CELLS_PER_FILE}" \
    --max-cells-per-species "${MAX_CELLS_PER_SPECIES}" \
    --max-total-cells "${MAX_TOTAL_CELLS}" \
    --sampling-mode "${SAMPLING_MODE}" \
    "${N_CLUSTER_ARGS[@]}" \
    "$@"
