#!/bin/bash
#SBATCH -J mambatcnregnet-sweep-agent
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
SWEEP_ID="${SWEEP_ID:-${2:-}}"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: set CONFIG_NAME or pass it as first argument."
  exit 1
fi
if [[ -z "${SWEEP_ID}" ]]; then
  echo "ERROR: set SWEEP_ID or pass it as second argument."
  exit 1
fi

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SEED="${SEED:-23}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH:-${PROJECT_DIR}/runs}/mamba_tcn_regnet_sweeps}"
WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${CONFIG_NAME}-sweep}"
WANDB_AGENT_COUNT="${WANDB_AGENT_COUNT:-1}"
TMP_BASE="${TMP_BASE:-${SCRATCH:-/tmp}/${USER}/mamba_tcn_regnet}"
TMPDIR="${TMPDIR:-${TMP_BASE}/tmp/${SLURM_JOB_ID:-$$}}"
MAMBA_CACHE_DIR="${MAMBA_CACHE_DIR:-${TMP_BASE}/mamba_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMP_BASE}/pip_cache}"

activate_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "ERROR: CONDA_ENV is set but conda is not available."
      exit 1
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    return
  fi
  if [[ -n "${VENV_PATH:-}" ]]; then
    source "${VENV_PATH}/bin/activate"
    return
  fi
  echo "ERROR: set CONDA_ENV or VENV_PATH before submitting the Grove job."
  exit 1
}

configure_python_runtime_libs() {
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

mkdir -p "${TMPDIR}" "${MAMBA_CACHE_DIR}" "${PIP_CACHE_DIR}" "${RUN_ROOT}"
export TMPDIR
export MAMBA_CACHE_DIR
export PIP_CACHE_DIR

cd "${PROJECT_DIR}"
activate_python

configure_python_runtime_libs

echo "Sweep config: ${CONFIG_NAME}"
echo "Sweep id: ${SWEEP_ID}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B group: ${WANDB_GROUP}"
echo "Run root: ${RUN_ROOT}"

AGENT_CMD=(
  wandb agent
  --count "${WANDB_AGENT_COUNT}"
  "${SWEEP_ID}"
)

export RUN_ROOT
export ACCELERATOR
export DEVICES
export NUM_WORKERS
export SEED

"${AGENT_CMD[@]}"
