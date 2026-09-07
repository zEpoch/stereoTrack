#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash 04_run_embedding_analysis_variant.sh INPUT_DIR OUTPUT_DIR"
    exit 2
fi

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
INPUT_DIR="$1" \
OUTPUT_DIR="$2" \
bash "${WORK_DIR}/example/04_three_species/04_run_embedding_analysis_rapids.sh" \
    --celltype-col cell_subclass
