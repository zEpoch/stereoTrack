#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/share/huadjyin/home/zhoutao3"
DO_DELETE=0
INCLUDE_STEREOTRACK_BASE=0

usage() {
  cat <<'EOF'
Usage:
  cleanup_imputation_intermediates.sh [--execute] [--include-stereotrack-base]

Default is dry-run: print what would be removed, but do not delete anything.

This script keeps final benchmark/visualization files such as:
  *.csv, *.pdf, *.png, *.html

It also keeps trained weights/checkpoints:
  stereoTrack checkpoints, SpaLP models

It removes large intermediate imputation outputs for:
  - stereoTrack MERFISH / Han imputation h5ad inference folders
  - SpaLP MERFISH / Han inference/preprocessed folders
  - temporary benchmark folders such as .*.tmp

Optional:
  --include-stereotrack-base
      Also remove stereoTrack base preprocessing/inference artifacts:
        out/*/cache, inference_adatas, wandb, job_logs
      This still keeps checkpoints.
      Use this only if you are sure you can retrain/reinfer later.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      DO_DELETE=1
      shift
      ;;
    --include-stereotrack-base)
      INCLUDE_STEREOTRACK_BASE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

targets=(
  "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/imputation_benchmark/stereotrack_gene_inference_adatas"
  "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/imputation_benchmark/.MERFISH_lognorm.tmp"
  "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/imputation_benchmark/.StereoTrack.tmp"
  "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/imputation_benchmark/.SpaLP.tmp"
  "${WORK_ROOT}/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/imputation_benchmark/stereotrack_gene_inference_adatas"
  "${WORK_ROOT}/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/imputation_benchmark/.Han_lognorm.tmp"
  "${WORK_ROOT}/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/imputation_benchmark/.StereoTrack.tmp"
  "${WORK_ROOT}/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/imputation_benchmark/.SpaLP.tmp"
  "${WORK_ROOT}/tracks/SpaLP/out/05_merfish_mouseBrain/preprocessed"
  "${WORK_ROOT}/tracks/SpaLP/out/05_merfish_mouseBrain/inference_adatas"
  "${WORK_ROOT}/tracks/SpaLP/out/03_han_mouse_brain_mae/preprocessed"
  "${WORK_ROOT}/tracks/SpaLP/out/03_han_mouse_brain_mae/inference_adatas"
)

if [[ "${INCLUDE_STEREOTRACK_BASE}" -eq 1 ]]; then
  targets+=(
    "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes/cache"
    "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/inference_adatas"
    "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/wandb"
    "${WORK_ROOT}/tracks/stereoTrack/out/05_merfish_mouseBrain_mae_v1_3axes_train/job_logs"
    "${WORK_ROOT}/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1/cache"
    "${WORK_ROOT}/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/wandb"
    "${WORK_ROOT}/tracks/stereoTrack/out/03_han_mouse_processed_mae_v1_train/job_logs"
  )
fi

echo "[mode] $([[ "${DO_DELETE}" -eq 1 ]] && echo "execute" || echo "dry-run")"
echo "[keep] final *.csv, *.pdf, *.png, *.html files inside imputation_benchmark are not targeted"
echo "[keep] stereoTrack checkpoints and SpaLP models are not targeted"
echo

existing=()
for path in "${targets[@]}"; do
  if [[ -e "${path}" ]]; then
    existing+=("${path}")
  fi
done

if [[ "${#existing[@]}" -eq 0 ]]; then
  echo "[done] no cleanup targets found"
  exit 0
fi

echo "[targets]"
du -sh "${existing[@]}" 2>/dev/null || true
echo

if [[ "${DO_DELETE}" -ne 1 ]]; then
  echo "[dry-run] nothing was deleted"
  echo "[next] rerun with --execute to delete the targets above"
  echo "[optional] add --include-stereotrack-base to also remove base cache/inference_adatas"
  exit 0
fi

echo "[delete] removing ${#existing[@]} targets"
rm -rf -- "${existing[@]}"
echo "[done] cleanup finished"
