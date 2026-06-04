#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

CONFIGS=(
  paper_hybrid_1d
  paper_tcn_1d
  paper_mamba_1d
  paper_attention_1d
  paper_hybrid_no_features_1d
  paper_hybrid_no_action_1d
)
SEEDS=(23 29 37 41 53)

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${config}-seed${seed}-$(date +%Y%m%d-%H%M%S)"
    RUN_ROOT="${PROJECT_DIR}/runs/paper_alpha_seek" \
    RUN_NAME="${run_name}" \
    CONFIG_NAME="${config}" \
    TRAIN_EXTRA_ARGS="--seed ${seed}" \
    EARLY_STOPPING_PATIENCE="30" \
    EARLY_STOPPING_MIN_DELTA="0.0001" \
    CONDA_ENV="${CONDA_ENV}" \
    bash cluster/grove/submit_grove_pipeline.sh
  done
done
