#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
ISH_DIR="${ROOT}/Imputation/00.ISH_Data/allen_3d_ish_data"
CONFIG="${ROOT}/config/03_config_han_mouse_processed.yaml"
MODEL_DIR="${ROOT}/out/03_han_mouse_processed_mae_v1_train"
OUT="${MODEL_DIR}/imputation_benchmark"
GENE_LIST="${OUT}/han_common_ish_genes.txt"
OUTPUT_DIR="${OUT}/stereotrack_gene_inference_adatas"
HAN_RAW="/home/share/huadjyin/home/zhoutao3/tracks/example_data/03_han_mouse_brain/processed"

CHECKPOINT="${1:-}"
if [ -z "${CHECKPOINT}" ]; then
  CHECKPOINT="$(find "${MODEL_DIR}/checkpoints" -maxdepth 1 -type f -name 'best*.ckpt' | sort | tail -n 1)"
fi
if [ -z "${CHECKPOINT}" ] || [ ! -s "${CHECKPOINT}" ]; then
  echo "Checkpoint not found. Pass it explicitly: bash $0 /path/to/best.ckpt" >&2
  exit 1
fi

mkdir -p "${OUT}" "${OUTPUT_DIR}"

python "${UTILS}/make_h5ad_ish_gene_list.py" \
  --input-dir "${HAN_RAW}" \
  --file-glob "*.h5ad" \
  --ish-dir "${ISH_DIR}" \
  --output "${GENE_LIST}"

python "${ROOT}/example/03_han_mouse_processed/03_inference.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --gene-file "${GENE_LIST}" \
  --write-cell-layer \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 4096 \
  --device cuda:3
