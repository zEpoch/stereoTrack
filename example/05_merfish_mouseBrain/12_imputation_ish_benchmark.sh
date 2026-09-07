#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
PYTHON="${CONDA_ENV}/bin/python"
ISH_DIR="${ROOT}/Imputation/00.ISH_Data/allen_3d_ish_data"
MODEL_DIR="${ROOT}/out/05_merfish_mouseBrain_mae_v1_3axes_train"
OUT="${MODEL_DIR}/imputation_benchmark"
GENE_LIST="${OUT}/merfish_common_ish_genes.txt"

MERFISH_RAW="/home/share/huadjyin/home/zhoutao3/tracks/example_data/14_merfish_mouseBrain/raw_data"
MERFISH_INFER="${OUT}/stereotrack_gene_inference_adatas"
SPALP_INFER="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/05_merfish_mouseBrain/inference_adatas"

mkdir -p "${OUT}"

"${PYTHON}" -u "${UTILS}/make_h5ad_ish_gene_list.py" \
  --input-dir "${MERFISH_RAW}" \
  --file-glob "*.h5ad" \
  --ish-dir "${ISH_DIR}" \
  --gene-name-column gene_name \
  --output "${GENE_LIST}"

cp "${GENE_LIST}" "${OUT}/genes_used_for_ish_correlation.txt"

"${PYTHON}" -u "${UTILS}/h5ad_voxel_ish_correlation.py" \
  --input-dir "${MERFISH_RAW}" \
  --file-glob "*.h5ad" \
  --gene-list "${GENE_LIST}" \
  --ish-dir "${ISH_DIR}" \
  --source X \
  --normalize log1p_norm \
  --ccf-key X_CCF \
  --gene-name-column gene_name \
  --ccf-scale 1 \
  --chunk-size 256 \
  --progress-every 1 \
  --output "${OUT}/panelG_MERFISH_lognorm.csv"

"${PYTHON}" -u "${UTILS}/h5ad_voxel_ish_correlation.py" \
  --input-dir "${MERFISH_INFER}" \
  --file-glob "*.h5ad" \
  --gene-list "${GENE_LIST}" \
  --ish-dir "${ISH_DIR}" \
  --source layer \
  --layer niche_recon \
  --normalize none \
  --ccf-key X_CCF \
  --gene-name-column gene_name \
  --ccf-scale 1 \
  --chunk-size 256 \
  --progress-every 1 \
  --output "${OUT}/panelG_StereoTrack_imputed.csv"

"${PYTHON}" -u "${UTILS}/h5ad_voxel_ish_correlation.py" \
  --input-dir "${SPALP_INFER}" \
  --file-glob "*.h5ad" \
  --gene-list "${GENE_LIST}" \
  --ish-dir "${ISH_DIR}" \
  --source X \
  --normalize none \
  --ccf-key X_CCF \
  --gene-name-column gene_name \
  --ccf-scale 1 \
  --chunk-size 256 \
  --progress-every 1 \
  --output "${OUT}/panelG_SpaLP_imputed.csv"

"${PYTHON}" -u "${ROOT}/Imputation/03.1758_important_genes_test/panelG_plot_comparison.py" \
  --csv "${OUT}/panelG_MERFISH_lognorm.csv" \
  --csv "${OUT}/panelG_StereoTrack_imputed.csv" \
  --csv "${OUT}/panelG_SpaLP_imputed.csv" \
  --name MERFISH_lognorm --name StereoTrack --name SpaLP \
  --output "${OUT}/panelG_merfish_3methods"
