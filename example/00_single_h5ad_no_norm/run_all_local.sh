#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/00_single_h5ad_no_norm"

bash "${SCRIPT_DIR}/01_run_preprocess.sh"
bash "${SCRIPT_DIR}/02_run_train.sh"
bash "${SCRIPT_DIR}/03_run_inference.sh"
