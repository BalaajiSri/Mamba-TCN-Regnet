#!/bin/bash
# Group B – Component ablations on the full hybrid model
#
# Configs that train today (no code changes needed):
#   paper_hybrid_no_action_1d   – action-head loss disabled (weight=0)
#   paper_hybrid_no_features_1d – raw OHLCV only (no engineered features)
#   paper_hybrid_no_smooth_1d   – adaptive smoother disabled
#   paper_hybrid_low_action_1d  – action loss weight 0.01
#   paper_hybrid_high_action_1d – action loss weight 0.20

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

CONFIGS=(
  # Branch-level ablations (hybrid minus one component)
  paper_hybrid_no_tcn_1d
  paper_hybrid_no_mamba_1d
  paper_hybrid_no_attn_1d
  paper_hybrid_simple_fusion_1d
  # Training objective ablations
  paper_hybrid_no_action_1d
  paper_hybrid_low_action_1d
  paper_hybrid_high_action_1d
  # Input / signal ablations
  paper_hybrid_no_features_1d
  paper_hybrid_no_smooth_1d
)
SEEDS=(23 29 37 41 53)

PARTITION="${PARTITION:-compute}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
EMAIL="${EMAIL:-}"
LOGGER_TYPE="${LOGGER_TYPE:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
PREEMPT="${PREEMPT:-0}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_hybrid_ssm_trader}"

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${config}-seed${seed}-$(date +%Y%m%d-%H%M%S)"
    RUN_ROOT="${RUN_ROOT}" \
    RUN_NAME="${run_name}" \
    CONFIG_NAME="${config}" \
    TRAIN_EXTRA_ARGS="--seed ${seed}" \
    EARLY_STOPPING_PATIENCE="30" \
    EARLY_STOPPING_MIN_DELTA="0.0001" \
    PARTITION="${PARTITION}" \
    GRES="${GRES}" \
    CPUS_PER_TASK="${CPUS_PER_TASK}" \
    MEM="${MEM}" \
    EMAIL="${EMAIL}" \
    PREEMPT="${PREEMPT}" \
    LOGGER_TYPE="${LOGGER_TYPE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_OFFLINE="${WANDB_OFFLINE}" \
    CONDA_ENV="${CONDA_ENV}" \
    bash cluster/grove/submit_grove_pipeline.sh
    sleep 1
  done
done
