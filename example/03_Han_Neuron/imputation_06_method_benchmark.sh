#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
PYTHON="${CONDA_ENV}/bin/python"
OUT="${ROOT}/out/03_han_mouse_processed_mae_v1_train/imputation_benchmark"
GENE_LIST="${OUT}/han_common_ish_genes.txt"

HAN_RAW="/home/share/huadjyin/home/zhoutao3/tracks/example_data/03_han_mouse_brain/processed"
HAN_INFER="${OUT}/stereotrack_gene_inference_adatas"
SPALP_INFER="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/03_han_mouse_brain_mae/inference_adatas"

mkdir -p "${OUT}"
cp "${GENE_LIST}" "${OUT}/genes_used_for_method_correlation.txt"

"${PYTHON}" -u "${UTILS}/h5ad_voxel_method_correlation.py" \
  --gene-list "${GENE_LIST}" \
  --voxel-size 200 \
  --voxel-method floor \
  --positive-filter none \
  --chunk-size 256 \
  --norm-chunk-size 1024 \
  --normalize-scope all \
  --name Han_lognorm \
  --input-dir "${HAN_RAW}" \
  --file-glob "*.h5ad" \
  --source X \
  --layer "" \
  --normalize log1p_norm \
  --ccf-key "" \
  --coord-columns az,ay,ax \
  --gene-name-column "" \
  --ccf-scale 10 \
  --name StereoTrack \
  --input-dir "${HAN_INFER}" \
  --file-glob "*.h5ad" \
  --source layer \
  --layer niche_recon \
  --normalize none \
  --ccf-key "" \
  --coord-columns az,ay,ax \
  --gene-name-column "" \
  --ccf-scale 10 \
  --name SpaLP \
  --input-dir "${SPALP_INFER}" \
  --file-glob "*.h5ad" \
  --source X \
  --layer "" \
  --normalize none \
  --ccf-key "" \
  --coord-columns az,ay,ax \
  --gene-name-column "" \
  --ccf-scale 10 \
  --output "${OUT}/han_voxel_method_correlation.csv" \
  --summary "${OUT}/han_voxel_method_correlation.summary.csv"
