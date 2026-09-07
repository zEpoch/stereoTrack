#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG="${ROOT}/config/03_config_han_mouse_processed.yaml"

python "${ROOT}/example/03_han_mouse_processed/01_preprocess.py" \
  --config "${CONFIG}" \
  --force
