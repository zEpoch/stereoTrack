#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_ENV}/bin/python"

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
OUT="${ROOT}/out/05_merfish_mouseBrain_mae_v1_3axes_train/imputation_benchmark/merfish_3d_match_html"

"${PYTHON}" -u "${ROOT}/example/05_merfish_mouseBrain/15_imputation_plot_3d_match.py" \
  --genes Tcf7l2 Lamp5 Cbln4 Rorb Trpc6 Scn4b Sox11 \
  --output-dir "${OUT}" \
  --mesh-path "${ROOT}/out/03_han_mouse_brain_mae/average_template_10_sur.stl" \
  --ccf-scale 1 \
  --method-plot-scale 10 \
  --ish-plot-scale 10 \
  --cell-point-size 4 \
  --voxel-point-size 180 \
  --with-mesh

# MERFISH present from your list:
#   Tcf7l2 Lamp5 Cbln4 Rorb Trpc6 Scn4b Sox11
#
# Han/StEReo-seq only from your list:
#   Foxp2 Hes5 Slc1a3 Grik5 Mbp Tcf4 Sox4 Grik2
#
# Present in both MERFISH and Han/StEReo-seq:
#   Tcf7l2 Lamp5 Cbln4 Rorb Trpc6 Scn4b Sox11
