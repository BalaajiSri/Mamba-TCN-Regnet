#!/bin/bash
# Group C – Architectural complexity study
# Varies: hidden_dim, TCN depth, number of Mamba branches
# All configs use the same training settings as the default hybrid.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

CONFIGS=(
  paper_hybrid_small_1d       # hidden_dim=32
  paper_hybrid_1d             # hidden_dim=64  (default – reference)
  paper_hybrid_large_1d       # hidden_dim=128
  paper_hybrid_tcn_shallow_1d # tcn_channels=[64]   (1 layer)
  paper_hybrid_tcn_deep_1d    # tcn_channels=[64,64,64,64]  (4 layers)
  paper_hybrid_mamba_2_1d     # num_mamba_branches=2
  paper_hybrid_mamba_8_1d     # num_mamba_branches=8
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
