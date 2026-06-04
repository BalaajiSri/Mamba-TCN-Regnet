#!/bin/bash
# Group E – Negative controls (shuffled labels + lag-mismatch targets)
#
# REQUIRES: training.py must accept --negcontrol_mode {shuffled_labels,lag_mismatch}
# See negcontrol.sbatch for the implementation notes.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

CONFIGS=(paper_hybrid_1d paper_tcn_1d paper_mamba_1d)
NEGCONTROL_MODES=(shuffled_labels lag_mismatch)
SEEDS=(23 29 37 41 53)

PARTITION="${PARTITION:-compute}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
EMAIL="${EMAIL:-}"
PREEMPT="${PREEMPT:-0}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_hybrid_ssm_trader}"

for config in "${CONFIGS[@]}"; do
  for mode in "${NEGCONTROL_MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="${config}-negctrl-${mode}-seed${seed}-$(date +%Y%m%d-%H%M%S)"
      sbatch_args=(
        --parsable
        --partition="${PARTITION}"
        --gres="${GRES}"
        --cpus-per-task="${CPUS_PER_TASK}"
        --mem="${MEM}"
        --time="2-00:00:00"
        --job-name="negctrl-${config}-${mode}"
      )
      if [[ -n "${EMAIL}" ]]; then
        sbatch_args+=(--mail-user="${EMAIL}" --mail-type="END,FAIL")
      fi
      if [[ "${PREEMPT}" == "1" ]]; then
        sbatch_args+=(--qos=preempt --requeue)
      fi
      job_id=$(sbatch "${sbatch_args[@]}" \
        --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG_NAME=${config},NEGCONTROL_MODE=${mode},RUN_NAME=${run_name},RUN_ROOT=${RUN_ROOT},SEED=${seed},CONDA_ENV=${CONDA_ENV}" \
        "${SCRIPT_DIR}/negcontrol.sbatch")
      echo "  Submitted ${config} / ${mode} / seed${seed}: job ${job_id}"
      sleep 1
    done
  done
done
