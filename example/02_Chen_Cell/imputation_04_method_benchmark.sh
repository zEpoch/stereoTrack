#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
PYTHON="${CONDA_ENV}/bin/python"
OUT="${ROOT}/out/02_macaque_brain_mae_v5/imputation_benchmark"

MACAQUE_RAW="/home/share/huadjyin/home/zhoutao3/tracks/example_data/07_macaque_brain/cortex"
MACAQUE_INFER="${OUT}/stereotrack_all_gene_inference_adatas"
SPALP_INFER="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/02_macaque_brain_mae_v5/inference_adatas"
GENE_LIST="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/02_macaque_brain_mae_v5/preprocessed/genes_used_for_spalp_output.txt"

mkdir -p "${OUT}"
cp "${GENE_LIST}" "${OUT}/genes_used_for_method_correlation.txt"

"${PYTHON}" -u "${UTILS}/h5ad_voxel_method_correlation.py" \
  --gene-list "${GENE_LIST}" \
  --voxel-size 10 \
  --voxel-method floor \
  --positive-filter none \
  --chunk-size 128 \
  --norm-chunk-size 1024 \
  --normalize-scope all \
  --name Macaque_lognorm \
  --input-dir "${MACAQUE_RAW}" \
  --file-glob "*.h5ad" \
  --source X \
  --layer "" \
  --normalize log1p_norm \
  --ccf-key ccf \
  --coord-columns - \
  --gene-name-column "" \
  --ccf-scale 1 \
  --name StereoTrack \
  --input-dir "${MACAQUE_INFER}" \
  --file-glob "*.h5ad" \
  --source X \
  --layer "" \
  --normalize none \
  --ccf-key ccf \
  --coord-columns - \
  --gene-name-column "" \
  --ccf-scale 1 \
  --name SpaLP \
  --input-dir "${SPALP_INFER}" \
  --file-glob "*.h5ad" \
  --source X \
  --layer "" \
  --normalize none \
  --ccf-key ccf \
  --coord-columns - \
  --gene-name-column "" \
  --ccf-scale 1 \
  --output "${OUT}/macaque_voxel_method_correlation.csv" \
  --summary "${OUT}/macaque_voxel_method_correlation.summary.csv"
