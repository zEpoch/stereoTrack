#!/bin/bash
set -euo pipefail

# Rerun this same command after interruption to resume automatically.
# Each method/gene chunk updates the result and summary CSVs immediately.
# CHECKPOINT_EVERY=1 saves aggregation after every slice (default: 20).
# Keep the adjacent *.csv.checkpoints directory: it contains resume state.
# Changed inputs/settings require a new OUTPUT_PREFIX; old checkpoints are
# deliberately not reused across different benchmark configurations.

ROOT="/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack"
UTILS="${ROOT}/example/imputation_benchmark_utils"
MODEL_DIR="${ROOT}/out/02_macaque_brain_mae_v5"
OUT="${MODEL_DIR}/imputation_lognormscale_benchmark"

STEREOTRACK_DIR="${MODEL_DIR}/imputation_benchmark/stereotrack_expression_npy"
SPALP_DIR="/home/share/huadjyin/home/zhoutao3/tracks/SpaLP/out/02_macaque_brain_mae_v5/inference_adatas"

source "/home/share/huadjyin/home/zhoutao3/envs/env_dgl.sh"

CONDA_ENV="${CONDA_PREFIX:-/home/share/huadjyin/home/zhoutao3/.conda/envs/stereotrack_2}"
PY_SITE="${CONDA_ENV}/lib/python3.10/site-packages"
PYTHON="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT}"

read -r -a METHODS <<< "${METHODS:-StereoTrack SpaLP}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-macaque_v1_lognormscale_volume_methods}"
METHODS_PY="$(printf "'%s'," "${METHODS[@]}")"

echo "[model] ${MODEL_DIR}"
echo "[stereotrack] ${STEREOTRACK_DIR}"
echo "[spalp] ${SPALP_DIR}"
echo "[out] ${OUT}"
echo "[resume] Automatic; saved results are skipped and aggregation resumes from its last checkpoint."
echo "[checkpoint] Save every ${CHECKPOINT_EVERY:-20} slices; change OUTPUT_PREFIX for a separate run."
echo "[checkpoint-dir] ${OUT}/${OUTPUT_PREFIX}_voxel_method_correlation.csv.checkpoints"
echo "[note] StereoTrack input must be reconstructed expression npy (*.expression.float16.npy), not niche embedding npy."

"${PYTHON}" - <<PY
import pickle
from pathlib import Path
import numpy as np

model_dir = Path("${MODEL_DIR}")
stereotrack_dir = Path("${STEREOTRACK_DIR}")
meta = pickle.load(open(model_dir / "cache" / "meta.pkl", "rb"))
expected_genes = int(meta["input_dim"])
missing = []
bad_shape = []
for info in meta["slice_info"]:
    stem = Path(info["batch"]).stem
    path = stereotrack_dir / f"{stem}.expression.${STEREOTRACK_OUTPUT_DTYPE:-float16}.npy"
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(stem)
        continue
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != expected_genes:
        bad_shape.append((stem, tuple(arr.shape)))
methods = [${METHODS_PY}]
if "StereoTrack" in methods:
    if missing:
        raise SystemExit(f"[error] missing {len(missing)} StereoTrack expression npy files, first={missing[:10]}")
    if bad_shape:
        raise SystemExit(
            f"[error] {len(bad_shape)} StereoTrack npy files have wrong shape, first={bad_shape[:5]}. "
            "Do not use niche_embedding_by_sample for expression benchmark."
        )
    print(f"[check] StereoTrack expression npy complete: {len(meta['slice_info'])}/{len(meta['slice_info'])}, genes={expected_genes}")
else:
    print("[check] StereoTrack not selected")
PY

if [ "${CHECK_ONLY:-0}" = "1" ]; then
  echo "[check] CHECK_ONLY=1, stop before volume benchmark."
  exit 0
fi

"${PYTHON}" -u "${UTILS}/macaque_lognormscale_npy_method_correlation.py" \
  --model-dir "${MODEL_DIR}" \
  --stereotrack-dir "${STEREOTRACK_DIR}" \
  --spalp-dir "${SPALP_DIR}" \
  --methods "${METHODS[@]}" \
  --stereotrack-dtype "${STEREOTRACK_OUTPUT_DTYPE:-float16}" \
  --voxel-size "${VOXEL_SIZE:-200}" \
  --voxel-method "${VOXEL_METHOD:-floor}" \
  --positive-filter "${POSITIVE_FILTER:-none}" \
  --chunk-size "${CHUNK_SIZE:-64}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-20}" \
  --max-genes "${MAX_GENES:-0}" \
  --output "${OUT}/${OUTPUT_PREFIX}_voxel_method_correlation.csv" \
  --summary "${OUT}/${OUTPUT_PREFIX}_voxel_method_correlation.summary.csv"
