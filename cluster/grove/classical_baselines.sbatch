#!/bin/bash
#SBATCH -J mambatcnregnet-classical-baselines
#SBATCH -p compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%j.out

# Group A supplement – ARIMA and moving-average baselines (CPU only).
# No GPU required. ARIMA(2,1,2) fitting + rolling prediction over ~550 test days
# takes roughly 30–60 minutes depending on node speed.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

DATA_CONFIG="${DATA_CONFIG:-btc_1d_paper}"
ARIMA_ORDER="${ARIMA_ORDER:-2,1,2}"
MA_WINDOWS="${MA_WINDOWS:-5,10,20}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/runs/paper_alpha_seek/classical_baselines}"
LOG_WANDB="${LOG_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
TMP_BASE="${TMP_BASE:-${SCRATCH:-/tmp}/${USER}/mamba_tcn_regnet}"
TMPDIR="${TMPDIR:-${TMP_BASE}/tmp/${SLURM_JOB_ID:-$$}}"

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

mkdir -p "${TMPDIR}" "${RESULTS_DIR}"
export TMPDIR
cd "${PROJECT_DIR}"
activate_python
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

EXTRA_ARGS=()
[[ "${LOG_WANDB}" == "1" ]] && EXTRA_ARGS+=(--log_wandb --wandb_project "${WANDB_PROJECT}")

python3 scripts/baselines_classical.py \
  --data_config "${DATA_CONFIG}" \
  --results_dir "${RESULTS_DIR}" \
  --arima_order "${ARIMA_ORDER}" \
  --ma_windows "${MA_WINDOWS}" \
  "${EXTRA_ARGS[@]}"
