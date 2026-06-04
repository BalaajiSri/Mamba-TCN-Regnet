#!/bin/bash
#SBATCH -J mambatcnregnet-validate-cpu
#SBATCH -p compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
RUN_DIR="${RUN_DIR:-${2:-}}"

if [[ -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: set CONFIG_NAME or pass it as the first argument."
  exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: set RUN_DIR or pass it as the second argument."
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/results}"
ACCELERATOR="cpu"
CKPT_SELECT="${CKPT_SELECT:-best}"
TRADE_SPLIT="${TRADE_SPLIT:-test}"
TRADE_MODE="${TRADE_MODE:-smart}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON_BIN="${PYTHON_BIN:-/home/nsribalaaji/anaconda3/envs/cryptomamba/bin/python}"
TMP_BASE="${TMP_BASE:-${SCRATCH:-/tmp}/${USER}/mamba_tcn_regnet}"
TMPDIR="${TMPDIR:-${TMP_BASE}/tmp/${SLURM_JOB_ID:-$$}}"
MAMBA_CACHE_DIR="${MAMBA_CACHE_DIR:-${TMP_BASE}/mamba_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMP_BASE}/pip_cache}"

activate_python() {
  if [[ -x "${PYTHON_BIN}" ]]; then
    return
  fi

  if [[ -n "${CONDA_ENV:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "ERROR: CONDA_ENV is set but conda is not available."
      exit 1
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  if [[ -n "${VENV_PATH:-}" ]]; then
    source "${VENV_PATH}/bin/activate"
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  echo "ERROR: set PYTHON_BIN, CONDA_ENV, or VENV_PATH before submitting the job."
  exit 1
}

configure_python_runtime_libs() {
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

mkdir -p "${RESULTS_DIR}"
mkdir -p "${TMPDIR}" "${MAMBA_CACHE_DIR}" "${PIP_CACHE_DIR}"
export TMPDIR
export MAMBA_CACHE_DIR
export PIP_CACHE_DIR

cd "${PROJECT_DIR}"
activate_python
configure_python_runtime_libs

echo "Validation config: ${CONFIG_NAME}"
echo "Run directory: ${RUN_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Checkpoint preference: ${CKPT_SELECT}"
echo "Trade split: ${TRADE_SPLIT}"
echo "Trade mode: ${TRADE_MODE}"
echo "Accelerator: ${ACCELERATOR}"
echo "Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/evaluation.py \
  --config "${CONFIG_NAME}" \
  --run_dir "${RUN_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --checkpoint_preference "${CKPT_SELECT}" \
  --accelerator "${ACCELERATOR}" \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}"

"${PYTHON_BIN}" scripts/simulate_trade.py \
  --config "${CONFIG_NAME}" \
  --run_dir "${RUN_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --checkpoint_preference "${CKPT_SELECT}" \
  --accelerator "${ACCELERATOR}" \
  --num_workers "${NUM_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --split "${TRADE_SPLIT}" \
  --trade_mode "${TRADE_MODE}"
