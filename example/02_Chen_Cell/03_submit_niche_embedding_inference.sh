#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
LOG_DIR="${ROOT}/out/02_macaque_brain_mae_v5/job_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

dsub \
  -n "stereotrack_macaque_niche_embed" \
  -A "root.project.P23Z10200N0876" \
  -R "cpu=32;gpu=1;mem=220000" \
  -T "86400" \
  -N 1 \
  -oo "${LOG_DIR}/niche_embedding_infer_${TIMESTAMP}.out" \
  -eo "${LOG_DIR}/niche_embedding_infer_${TIMESTAMP}.err" \
  bash "${ROOT}/example/02_Chen_Cell/03_inference_niche_embedding.sh"

echo "Submitted StereoTrack macaque niche embedding inference; logs: ${LOG_DIR}/niche_embedding_infer_${TIMESTAMP}.{out,err}"
