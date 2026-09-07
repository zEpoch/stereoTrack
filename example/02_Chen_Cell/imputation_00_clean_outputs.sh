#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
SPALP_ROOT="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP"

# Remove the current macaque v5 preprocessing cache, checkpoints, inference
# outputs, and SpaLP outputs. Raw h5ad data are not touched.
TARGETS=(
  "${ROOT}/out/02_macaque_brain_mae_v5"
  "${SPALP_ROOT}/out/02_macaque_brain_mae_v5"
)

for path in "${TARGETS[@]}"; do
  if [ -e "${path}" ]; then
    echo "[remove] ${path}"
    rm -rf "${path}"
  else
    echo "[skip] ${path}"
  fi
done
