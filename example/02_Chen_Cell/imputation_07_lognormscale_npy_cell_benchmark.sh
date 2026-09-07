#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
OUT="${MODEL_DIR}/imputation_lognormscale_benchmark"

STEREOTRACK_DIR="${MODEL_DIR}/imputation_benchmark/stereotrack_expression_npy"
SPALP_DIR="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/02_macaque_brain_mae_v5/inference_adatas"

source "/home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh"

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
PYTHON="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT}"

METHODS=(${METHODS:-StereoTrack SpaLP})
OUTPUT_PREFIX="${OUTPUT_PREFIX:-macaque_v1_lognormscale_cell_methods}"

echo "[model] ${MODEL_DIR}"
echo "[reference] ${MODEL_DIR}/cache/slice_*_full_graph.npz"
echo "[stereotrack] ${STEREOTRACK_DIR}"
echo "[spalp] ${SPALP_DIR}"
echo "[out] ${OUT}"

"${PYTHON}" -u "${UTILS}/macaque_lognormscale_npy_cell_correlation.py" \
  --model-dir "${MODEL_DIR}" \
  --stereotrack-dir "${STEREOTRACK_DIR}" \
  --spalp-dir "${SPALP_DIR}" \
  --methods "${METHODS[@]}" \
  --stereotrack-dtype "${STEREOTRACK_OUTPUT_DTYPE:-float16}" \
  --chunk-size "${CHUNK_SIZE:-64}" \
  --max-genes "${MAX_GENES:-0}" \
  --output "${OUT}/${OUTPUT_PREFIX}_correlation.csv" \
  --summary "${OUT}/${OUTPUT_PREFIX}_correlation.summary.csv"
