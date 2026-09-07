#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_ENV}/bin/python"

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"

"${PYTHON}" "${ROOT}/example/03_Han_Neuron/imputation_05_plot_3d_match.py" \
  --genes Foxp2 Tcf7l2 Hes5 Slc1a3 Grik5 Lamp5 Cbln4 Rorb Mbp Trpc6 Scn4b Sox11 Tcf4 Sox4 Grik2 \
  --output-dir "${ROOT}/out/03_han_mouse_processed_mae_v1_train/imputation_benchmark/han_3d_match_html" \
  --with-mesh
# Foxp2 Tcf7l2 Hes5 Slc1a3 Grik5 Lamp5 Cbln4 Rorb Mbp Trpc6 Scn4b Sox11 Tcf4 Sox4 Grik2