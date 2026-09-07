#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
EXAMPLE_DIR="${WORK_DIR}/example/01_axolotl_artista"

extract_jobid() {
  awk 'NR > 1 && $1 ~ /^[0-9]+$/ {print $1; exit}'
}

preprocess_log="$(mktemp)"
train_log="$(mktemp)"

bash "${EXAMPLE_DIR}/04_submit_preprocess_downstream.sh" "$@" | tee "${preprocess_log}"
preprocess_job="$(extract_jobid < "${preprocess_log}")"
if [[ -z "${preprocess_job}" ]]; then
  echo "Failed to parse downstream preprocess job id" >&2
  exit 1
fi

bash "${EXAMPLE_DIR}/04_submit_train_downstream.sh" "${preprocess_job}" | tee "${train_log}"
train_job="$(extract_jobid < "${train_log}")"
if [[ -z "${train_job}" ]]; then
  echo "Failed to parse downstream train job id" >&2
  exit 1
fi

bash "${EXAMPLE_DIR}/04_submit_inference_downstream.sh" "${train_job}"

rm -f "${preprocess_log}" "${train_log}"

