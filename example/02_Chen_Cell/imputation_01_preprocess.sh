#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
CONFIG="${ROOT}/config/02_config_Chen_Cell_v5.yaml"

python "${ROOT}/example/02_Chen_Cell/03_inference.py" \
  --config "${CONFIG}" \
  --force
