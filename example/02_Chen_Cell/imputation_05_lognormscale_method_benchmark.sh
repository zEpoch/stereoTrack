#!/bin/bash
set -euo pipefail

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
UTILS="${ROOT}/example/imputation_benchmark_utils"
OUT="${MODEL_DIR}/imputation_lognormscale_benchmark"

REF_CACHE="${MODEL_DIR}/cache/adatas"
STEREOTRACK_INFER="${MODEL_DIR}/imputation_benchmark/stereotrack_all_gene_inference_adatas"
SPALP_INFER="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/02_macaque_brain_mae_v5/inference_adatas"

source "/home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh"

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
PYTHON="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT}"

GENE_LIST_FULL="${OUT}/macaque_v1_common_genes_lognormscale.txt"
"${PYTHON}" - <<PY
import pickle
from pathlib import Path

meta_path = Path("${MODEL_DIR}") / "cache" / "meta.pkl"
out_path = Path("${GENE_LIST_FULL}")
with meta_path.open("rb") as handle:
    meta = pickle.load(handle)
genes = list(meta["common_genes"])
out_path.write_text("\\n".join(map(str, genes)) + "\\n")
print(f"[genes] {len(genes):,} -> {out_path}")
PY

GENE_LIST="${GENE_LIST_FULL}"
MAX_GENES="${MAX_GENES:-0}"
if [ "${MAX_GENES}" != "0" ]; then
  GENE_LIST="${OUT}/macaque_v1_common_genes_lognormscale_first_${MAX_GENES}.txt"
  head -n "${MAX_GENES}" "${GENE_LIST_FULL}" > "${GENE_LIST}"
  echo "[genes] using first ${MAX_GENES} genes for a test run: ${GENE_LIST}"
fi

count_files() {
  local dir="$1"
  local glob="$2"
  if [ ! -d "${dir}" ]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name "${glob}" | wc -l
}

REF_N="$(count_files "${REF_CACHE}" "slice_*.h5ad")"
SPALP_N="$(count_files "${SPALP_INFER}" "*.h5ad")"
STEREO_N="$(count_files "${STEREOTRACK_INFER}" "*.h5ad")"

echo "[input] reference cache h5ad: ${REF_N} (${REF_CACHE})"
echo "[input] SpaLP inference h5ad: ${SPALP_N} (${SPALP_INFER})"
echo "[input] StereoTrack inference h5ad: ${STEREO_N} (${STEREOTRACK_INFER})"

if [ "${REF_N}" -eq 0 ]; then
  echo "[error] missing reference cache h5ad: ${REF_CACHE}" >&2
  exit 1
fi
if [ "${SPALP_N}" -eq 0 ]; then
  echo "[error] missing SpaLP inference h5ad: ${SPALP_INFER}" >&2
  exit 1
fi

REQUIRE_STEREOTRACK="${REQUIRE_STEREOTRACK:-1}"
INCLUDE_STEREOTRACK=1
if [ "${STEREO_N}" -eq 0 ]; then
  INCLUDE_STEREOTRACK=0
  if [ "${REQUIRE_STEREOTRACK}" = "1" ]; then
    cat >&2 <<MSG
[error] StereoTrack all-gene inference h5ad not found.
Run this first if you want the full StereoTrack vs lognormscale benchmark:
  bash ${ROOT}/example/02_Chen_Cell/imputation_03_stereotrack_inference.sh

Warning: all-gene dense h5ad inference can be hundreds of GB. If you only want
to compute SpaLP vs lognormscale now, rerun with:
  REQUIRE_STEREOTRACK=0 bash $0
MSG
    exit 1
  fi
fi

METHOD_ARGS=(
  --name Macaque_lognormscale
  --input-dir "${REF_CACHE}"
  --file-glob "slice_*.h5ad"
  --source X
  --layer ""
  --normalize none
  --ccf-key ccf
  --coord-columns -
  --gene-name-column ""
  --ccf-scale 1
  --name SpaLP
  --input-dir "${SPALP_INFER}"
  --file-glob "*.h5ad"
  --source X
  --layer ""
  --normalize none
  --ccf-key ccf
  --coord-columns -
  --gene-name-column ""
  --ccf-scale 1
)

OUTPUT_PREFIX="macaque_v1_lognormscale_spalp"
if [ "${INCLUDE_STEREOTRACK}" -eq 1 ]; then
  OUTPUT_PREFIX="macaque_v1_lognormscale_3methods"
  METHOD_ARGS+=(
    --name StereoTrack
    --input-dir "${STEREOTRACK_INFER}"
    --file-glob "*.h5ad"
    --source X
    --layer ""
    --normalize none
    --ccf-key ccf
    --coord-columns -
    --gene-name-column ""
    --ccf-scale 1
  )
fi

OUTPUT_CSV="${OUT}/${OUTPUT_PREFIX}_voxel_method_correlation.csv"
SUMMARY_CSV="${OUT}/${OUTPUT_PREFIX}_voxel_method_correlation.summary.csv"

echo "[output] ${OUTPUT_CSV}"
echo "[summary] ${SUMMARY_CSV}"

CHECK_ONLY="${CHECK_ONLY:-0}"
if [ "${CHECK_ONLY}" = "1" ]; then
  echo "[check] CHECK_ONLY=1, stop before benchmark."
  exit 0
fi

"${PYTHON}" -u "${UTILS}/h5ad_voxel_method_correlation.py" \
  --gene-list "${GENE_LIST}" \
  --voxel-size "${VOXEL_SIZE:-10}" \
  --voxel-method "${VOXEL_METHOD:-floor}" \
  --positive-filter "${POSITIVE_FILTER:-none}" \
  --chunk-size "${CHUNK_SIZE:-128}" \
  --norm-chunk-size "${NORM_CHUNK_SIZE:-1024}" \
  --normalize-scope all \
  "${METHOD_ARGS[@]}" \
  --output "${OUTPUT_CSV}" \
  --summary "${SUMMARY_CSV}"
