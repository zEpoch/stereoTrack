#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
RUNNER="${WORK_DIR}/example/04_three_species/04_run_embedding_analysis_variant.sh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

submit_analysis() {
    local variant="$1"
    local model_dir="$2"
    local input_dir="${model_dir}/inference_adatas"
    local output_dir="${model_dir}/analysis_rapids_comparison"
    local log_dir="${output_dir}/job_logs"

    mkdir -p "${log_dir}"
    dsub \
        -n "04_embed_${variant}" \
        -A "root.project.P23Z10200N0876" \
        -R "cpu=32;gpu=1;mem=120000" \
        -T "86400" \
        -N 1 \
        -oo "${log_dir}/analysis_${TIMESTAMP}.out" \
        -eo "${log_dir}/analysis_${TIMESTAMP}.err" \
        bash "${RUNNER}" "${input_dir}" "${output_dir}"
}

submit_analysis \
    "balanced" \
    "${WORK_DIR}/out/04_three_species_mae_v1_train_balanced_v1"

submit_analysis \
    "align" \
    "${WORK_DIR}/out/04_three_species_mae_v1_train_align_v1"

echo "Submitted balanced and alignment RAPIDS embedding analyses."
