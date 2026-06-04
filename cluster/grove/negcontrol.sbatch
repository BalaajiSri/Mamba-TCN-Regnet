#!/bin/bash
#SBATCH -J mambatcnregnet-negcontrol
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out

# Group E – Negative control run.
# NEGCONTROL_MODE controls which control is applied:
#   shuffled_labels  – targets are randomly permuted within the training set
#   lag_mismatch     – targets shifted by +1 day (leakage check)
#
# Both require training.py to accept a --negcontrol_mode flag.
# Add the flag and corresponding logic in CMambaDataModule / dataset.py:
#
#   shuffled_labels:  after loading, call
#       targets = targets[torch.randperm(len(targets))]
#   lag_mismatch:     shift target tensor by +1 step before training.
#
# Once the flag exists, this sbatch will work without further changes.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONFIG_NAME="${CONFIG_NAME:-paper_hybrid_1d}"
NEGCONTROL_MODE="${NEGCONTROL_MODE:-shuffled_labels}"   # or lag_mismatch
RUN_NAME="${RUN_NAME:-${CONFIG_NAME}-negctrl-${NEGCONTROL_MODE}-${SLURM_JOB_ID}}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="${SCRATCH:-${PROJECT_DIR}/runs}/mamba_tcn_regnet"
fi
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${CONFIG_NAME}/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/results}"

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-23}"
TMP_BASE="${TMP_BASE:-${SCRATCH:-/tmp}/${USER}/mamba_tcn_regnet}"
TMPDIR="${TMPDIR:-${TMP_BASE}/tmp/${SLURM_JOB_ID:-$$}}"
MAMBA_CACHE_DIR="${MAMBA_CACHE_DIR:-${TMP_BASE}/mamba_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMP_BASE}/pip_cache}"

activate_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    set +u
    if command -v conda >/dev/null 2>&1; then
      source "$(conda info --base)/etc/profile.d/conda.sh"
    elif [[ -f "/home/nsribalaaji/anaconda3/etc/profile.d/conda.sh" ]]; then
      source "/home/nsribalaaji/anaconda3/etc/profile.d/conda.sh"
    else
      echo "ERROR: CONDA_ENV set but conda not found."
      set -u; exit 1
    fi
    conda activate "${CONDA_ENV}"
    set -u; return
  fi
  if [[ -n "${VENV_PATH:-}" ]]; then
    source "${VENV_PATH}/bin/activate"; return
  fi
  echo "ERROR: set CONDA_ENV or VENV_PATH."; exit 1
}

configure_python_runtime_libs() {
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${RESULTS_DIR}"
mkdir -p "${TMPDIR}" "${MAMBA_CACHE_DIR}" "${PIP_CACHE_DIR}"
export TMPDIR MAMBA_CACHE_DIR PIP_CACHE_DIR
cd "${PROJECT_DIR}"
activate_python
configure_python_runtime_libs

echo "Negative control mode: ${NEGCONTROL_MODE}"
echo "Config: ${CONFIG_NAME}  |  Seed: ${SEED}"
echo "Run directory: ${RUN_DIR}"

python3 scripts/training.py \
  --config "${CONFIG_NAME}" \
  --accelerator "${ACCELERATOR}" \
  --devices "${DEVICES}" \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --logger_type "tb" \
  --default_root_dir "${RUN_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --resume_mode "auto" \
  --strategy "auto" \
  --early_stopping_patience "30" \
  --early_stopping_min_delta "0.0001" \
  --early_stopping_monitor "val/rmse" \
  --early_stopping_mode "min" \
  --seed "${SEED}" \
  --negcontrol_mode "${NEGCONTROL_MODE}" \
  --save_checkpoints

python3 scripts/evaluation.py \
  --config "${CONFIG_NAME}" \
  --run_dir "${RUN_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --checkpoint_preference "best" \
  --accelerator "${ACCELERATOR}" \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}"
