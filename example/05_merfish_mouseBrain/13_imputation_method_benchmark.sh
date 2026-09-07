#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
PYTHON="${CONDA_ENV}/bin/python"
MODEL_DIR="${ROOT}/out/05_merfish_mouseBrain_mae_v1_3axes_train"
OUT="${MODEL_DIR}/imputation_benchmark"
GENE_LIST="${OUT}/merfish_common_ish_genes.txt"

MERFISH_RAW="/home/share/huadjyin/home/zhoutao3/tracks/example_data/14_merfish_mouseBrain/raw_data"
MERFISH_INFER="${OUT}/stereotrack_gene_inference_adatas"
SPALP_INFER="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/05_merfish_mouseBrain/inference_adatas"

mkdir -p "${OUT}"
cp "${GENE_LIST}" "${OUT}/genes_used_for_method_correlation.txt"

"${PYTHON}" -u "${UTILS}/h5ad_voxel_method_correlation.py" \
  --gene-list "${GENE_LIST}" \
  --voxel-size 200 \
  --voxel-method floor \
  --positive-filter none \
  --chunk-size 128 \
  --norm-chunk-size 1024 \
  --normalize-scope all \
  --progress-every 1 \
  --name MERFISH_lognorm \
  --input-dir "${MERFISH_RAW}" \
  --file-glob "*.h5ad" \
  --source X \
  --layer "" \
  --normalize log1p_norm \
  --ccf-key X_CCF \
  --coord-columns - \
  --gene-name-column gene_name \
  --ccf-scale 1 \
  --name StereoTrack \
  --input-dir "${MERFISH_INFER}" \
  --file-glob "*.h5ad" \
  --source layer \
  --layer niche_recon \
  --normalize none \
  --ccf-key X_CCF \
  --coord-columns - \
  --gene-name-column gene_name \
  --ccf-scale 1 \
  --name SpaLP \
  --input-dir "${SPALP_INFER}" \
  --file-glob "*.h5ad" \
  --source X \
  --layer "" \
  --normalize none \
  --ccf-key X_CCF \
  --coord-columns - \
  --gene-name-column gene_name \
  --ccf-scale 1 \
  --output "${OUT}/merfish_voxel_method_correlation.csv" \
  --summary "${OUT}/merfish_voxel_method_correlation.summary.csv"
