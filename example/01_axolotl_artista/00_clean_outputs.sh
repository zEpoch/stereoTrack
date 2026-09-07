#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
INPUT_DIR="${WORK_DIR}/out/01_axolotl_artista_mae_v1"
TRAIN_DIR="${WORK_DIR}/out/01_axolotl_artista_mae_v1_train"

echo "[clean] This will remove old axolotl ARTISTA preprocessing, checkpoints, logs, and inference outputs:"
echo "  ${INPUT_DIR}"
echo "  ${TRAIN_DIR}"
echo
read -r -p "Type DELETE to continue: " answer
if [[ "${answer}" != "DELETE" ]]; then
  echo "[clean] aborted"
  exit 0
fi

rm -rf "${INPUT_DIR}" "${TRAIN_DIR}"
echo "[clean] done"
