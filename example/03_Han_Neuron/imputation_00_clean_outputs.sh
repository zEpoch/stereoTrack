#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
SPALP_ROOT="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP"

# Remove Han preprocessing cache, training checkpoints, inference outputs, and
# SpaLP outputs for a clean restart. Raw h5ad data are not touched.
TARGETS=(
  "${ROOT}/out/03_han_mouse_processed_mae_v1"
  "${ROOT}/out/03_han_mouse_processed_mae_v1_train"
  "${ROOT}/out/03_han_mouse_brain_mae"
  "${ROOT}/out/03_han_mouse_brain_mae_v3"
  "${ROOT}/out/03_han_mouse_brain_mae_v12_3"
  "${SPALP_ROOT}/out/03_han_mouse_brain_mae"
)

for path in "${TARGETS[@]}"; do
  if [ -e "${path}" ]; then
    echo "[remove] ${path}"
    rm -rf "${path}"
  else
    echo "[skip] ${path}"
  fi
done
