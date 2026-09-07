#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_ENV}/bin/python"
ISH_DIR="${ROOT}/Imputation/00.ISH_Data/allen_3d_ish_data"
CONFIG="${ROOT}/config/05_config_merfish_mouseBrain_3axes.yaml"
MODEL_DIR="${ROOT}/out/05_merfish_mouseBrain_mae_v1_3axes_train"
CHECKPOINT="${MODEL_DIR}/checkpoints/best-epoch=99-train_loss_epoch=0.3303.ckpt"
OUT="${MODEL_DIR}/imputation_benchmark"
GENE_LIST="${OUT}/merfish_common_ish_genes.txt"
OUTPUT_DIR="${OUT}/stereotrack_gene_inference_adatas"
MERFISH_RAW="/home/share/huadjyin/home/zhoutao3/tracks/example_data/14_merfish_mouseBrain/raw_data"

mkdir -p "${OUT}" "${OUTPUT_DIR}"

"${PYTHON}" -u "${UTILS}/make_h5ad_ish_gene_list.py" \
  --input-dir "${MERFISH_RAW}" \
  --file-glob "*.h5ad" \
  --ish-dir "${ISH_DIR}" \
  --gene-name-column gene_name \
  --output "${GENE_LIST}"

"${PYTHON}" -u "${ROOT}/example/05_merfish_mouseBrain/03_inference.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --gene-file "${GENE_LIST}" \
  --write-cell-layer \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 1024 \
  --device cuda:2
