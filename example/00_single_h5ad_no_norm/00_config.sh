#!/bin/bash
set -euo pipefail

WORK_DIR="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
SCRIPT_DIR="${WORK_DIR}/example/00_single_h5ad_no_norm"

: "${DATA_PATH:?Please set DATA_PATH=/path/to/one_file.h5ad before running.}"

DATA_PATH="$(readlink -f "${DATA_PATH}")"
RUN_NAME="${RUN_NAME:-single_h5ad_no_norm}"
SOURCE_SPATIAL_KEY="${SOURCE_SPATIAL_KEY:-spatial}"
INTERNAL_SPATIAL_KEY="${INTERNAL_SPATIAL_KEY:-ccf}"
GRAPH_METHOD="${GRAPH_METHOD:-knn}"
N_NEIGHBORS="${N_NEIGHBORS:-12}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-100000}"
PATCH_MODE="${PATCH_MODE:-axis}"
PATCH_AXES="${PATCH_AXES:-3}"
PATCH_SIZE="${PATCH_SIZE:-4096}"
N_EPOCHS="${N_EPOCHS:-50}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1024}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INPUT_DIR="${WORK_DIR}/out/${RUN_NAME}"
SAVE_DIR="${WORK_DIR}/out/${RUN_NAME}_train"
CONFIG_FILE="${WORK_DIR}/config/${RUN_NAME}.yaml"

prepare_input_dir() {
  mkdir -p "${INPUT_DIR}/input_h5ad"
  if [ -f "${DATA_PATH}" ]; then
    local stem
    stem="$(basename "${DATA_PATH}" .h5ad)"
    ln -sfn "${DATA_PATH}" "${INPUT_DIR}/input_h5ad/${stem}.h5ad"
    DATA_DIR="${INPUT_DIR}/input_h5ad"
    INCLUDE_STEMS="${stem}"
  elif [ -d "${DATA_PATH}" ]; then
    DATA_DIR="${DATA_PATH}"
    INCLUDE_STEMS="${INCLUDE_STEMS:-}"
  else
    echo "[error] DATA_PATH is neither a file nor a directory: ${DATA_PATH}" >&2
    exit 1
  fi
}

write_config() {
  mkdir -p "$(dirname "${CONFIG_FILE}")" "${INPUT_DIR}" "${SAVE_DIR}"
  local include_yaml=""
  if [ -n "${INCLUDE_STEMS:-}" ]; then
    include_yaml="  include_stems:"
    for stem in ${INCLUDE_STEMS}; do
      include_yaml="${include_yaml}"$'\n'"    - ${stem}"
    done
  fi

  cat > "${CONFIG_FILE}" <<YAML
paths:
  data_path: ${DATA_DIR}
  input_dir: ${INPUT_DIR}
  save_dir: ${SAVE_DIR}

preprocess:
  spatial_key: ${INTERNAL_SPATIAL_KEY}
  source_spatial_key: ${SOURCE_SPATIAL_KEY}
  expression_layer:
  graph_method: ${GRAPH_METHOD}
  n_neighbors: ${N_NEIGHBORS}
  knn_chunk_size: ${KNN_CHUNK_SIZE}
  patch_axes: ${PATCH_AXES}
  patch_mode: ${PATCH_MODE}
  min_counts: 0
  skip_normalize_log_scale: true
  target_sum: 10000.0
  max_value: 10.0
  compress_npz: false
${include_yaml}

model:
  hidden_dim: 256
  latent_dim: 64
  dropout_rate: 0.1
  gene_mask_ratio: 0.5
  cell_mask_ratio: 0.2
  n_encoder_layers: 2
  n_decoder_layers: 2
  niche_self_weight: 0.1

training:
  patch_size: ${PATCH_SIZE}
  n_epochs: ${N_EPOCHS}
  learning_rate: 2e-4
  warmup_epochs: 5
  patience: 20
  grad_accum_steps: 2
  use_amp: true
  infer_batch_size: ${INFER_BATCH_SIZE}
  lambda_niche_recon: 1.0
  lambda_cell_recon: 1.0
  lambda_cell_mask: 1.0
  lambda_binary: 0.1

inference:
  output_genes: []

logging:
  use_tensorboard: false
  use_wandb: false
  wandb_project: StereoTrack
  wandb_name: ${RUN_NAME}
  wandb_save_dir: ${SAVE_DIR}/wandb
YAML
}

prepare_config() {
  prepare_input_dir
  write_config
  echo "[config] ${CONFIG_FILE}"
  echo "[data] ${DATA_DIR}"
  echo "[input] ${INPUT_DIR}"
  echo "[save] ${SAVE_DIR}"
}
