#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG="${ROOT}/config/02_config_Chen_Cell_v5.yaml"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
OUTPUT_DIR="${MODEL_DIR}/imputation_benchmark/stereotrack_expression_npy"

source "/home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh"

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
PYTHON="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CHECKPOINT="${1:-}"
if [ -z "${CHECKPOINT}" ]; then
  CHECKPOINT="$(find "${MODEL_DIR}/checkpoints" -maxdepth 1 -type f -name 'best*.ckpt' | sort | tail -n 1)"
fi
if [ -z "${CHECKPOINT}" ] || [ ! -s "${CHECKPOINT}" ]; then
  echo "[error] checkpoint not found. Pass it explicitly: bash $0 /path/to/best.ckpt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -u "${ROOT}/example/02_Chen_Cell/imputation_03_stereotrack_npy_inference.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${STEREOTRACK_INFER_BATCH_SIZE:-2048}" \
  --device "${STEREOTRACK_DEVICE:-cuda:0}" \
  --output-dtype "${STEREOTRACK_OUTPUT_DTYPE:-float16}" \
  --use-amp \
  --skip-existing
