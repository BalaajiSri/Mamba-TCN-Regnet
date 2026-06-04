#!/bin/bash
#SBATCH -J mambatcnregnet-walk-forward
#SBATCH -p compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o slurm-%x-%j.out

# Walk-forward evaluation — CPU only, runs in minutes.
# Depends on all validation jobs completing first.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_alpha_seek}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/experiments/paper_alpha_seek/metrics/walk_forward_metrics.json}"
WINDOW="${WINDOW:-63}"
STEP="${STEP:-21}"
LOG_WANDB="${LOG_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"

activate_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    set +u
    if command -v conda >/dev/null 2>&1; then
      source "$(conda info --base)/etc/profile.d/conda.sh"
    elif [[ -f "/home/nsribalaaji/anaconda3/etc/profile.d/conda.sh" ]]; then
      source "/home/nsribalaaji/anaconda3/etc/profile.d/conda.sh"
    else
      echo "ERROR: conda not found."; set -u; exit 1
    fi
    conda activate "${CONDA_ENV}"
    set -u; return
  fi
  if [[ -n "${VENV_PATH:-}" ]]; then source "${VENV_PATH}/bin/activate"; return; fi
  echo "ERROR: set CONDA_ENV or VENV_PATH."; exit 1
}

cd "${PROJECT_DIR}"
activate_python
[[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]] && \
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

EXTRA=()
[[ "${LOG_WANDB}" == "1" ]] && EXTRA+=(--log_wandb --wandb_project "${WANDB_PROJECT}")

python3 scripts/walk_forward_eval.py \
  --run_root "${RUN_ROOT}" \
  --output   "${OUTPUT}" \
  --window   "${WINDOW}" \
  --step     "${STEP}" \
  "${EXTRA[@]}"
