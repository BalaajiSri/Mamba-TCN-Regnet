#!/bin/bash
#SBATCH -J mambatcnregnet-train
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: set CONFIG_NAME or pass it as the first argument."
  exit 1
fi

RUN_NAME="${RUN_NAME:-${CONFIG_NAME}-${SLURM_JOB_ID}}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    RUN_ROOT="${SCRATCH}/mamba_tcn_regnet"
  else
    RUN_ROOT="${PROJECT_DIR}/runs"
  fi
fi

RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${CONFIG_NAME}/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/results}"
ACCELERATOR="${ACCELERATOR:-gpu}"
GRES="${GRES:-gpu:1}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOGGER_TYPE="${LOGGER_TYPE:-tb}"
RESUME_MODE="${RESUME_MODE:-auto}"
STRATEGY="${STRATEGY:-auto}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-val/rmse}"
EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-min}"
WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"
TMP_BASE="${TMP_BASE:-${SCRATCH:-/tmp}/${USER}/mamba_tcn_regnet}"
TMPDIR="${TMPDIR:-${TMP_BASE}/tmp/${SLURM_JOB_ID:-$$}}"
MAMBA_CACHE_DIR="${MAMBA_CACHE_DIR:-${TMP_BASE}/mamba_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMP_BASE}/pip_cache}"

infer_gpu_count() {
  local value="${1:-}"
  local last_field
  if [[ -z "${value}" ]]; then
    echo "1"
    return
  fi
  last_field="${value##*:}"
  if [[ "${last_field}" =~ ^[0-9]+$ ]]; then
    echo "${last_field}"
    return
  fi
  echo "1"
}

detect_allocated_gpu_count() {
  local count
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
    count="${#cuda_devices[@]}"
    if [[ "${count}" -gt 0 ]]; then
      echo "${count}"
      return
    fi
  fi
  infer_gpu_count "${GRES}"
}

DEVICES="${DEVICES:-$(detect_allocated_gpu_count)}"

activate_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    set +u
    if command -v conda >/dev/null 2>&1; then
      source "$(conda info --base)/etc/profile.d/conda.sh"
    elif [[ -f "/home/nsribalaaji/anaconda3/etc/profile.d/conda.sh" ]]; then
      source "/home/nsribalaaji/anaconda3/etc/profile.d/conda.sh"
    else
      echo "ERROR: CONDA_ENV is set but conda is not available."
      set -u
      exit 1
    fi
    conda activate "${CONDA_ENV}"
    set -u
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

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${RESULTS_DIR}"
mkdir -p "${TMPDIR}" "${MAMBA_CACHE_DIR}" "${PIP_CACHE_DIR}"
export TMPDIR
export MAMBA_CACHE_DIR
export PIP_CACHE_DIR
cd "${PROJECT_DIR}"
activate_python

configure_python_runtime_libs

echo "Training config: ${CONFIG_NAME}"
echo "Run directory: ${RUN_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Accelerator: ${ACCELERATOR}"
echo "Devices: ${DEVICES}"
echo "GRES: ${GRES}"

python3 scripts/preflight_grove.py \
  --config "${CONFIG_NAME}" \
  --accelerator "${ACCELERATOR}" \
  --devices "${DEVICES}" \
  --gres "${GRES}"

TRAIN_CMD=(
  python3 scripts/training.py
  --config "${CONFIG_NAME}"
  --accelerator "${ACCELERATOR}"
  --devices "${DEVICES}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --logger_type "${LOGGER_TYPE}"
  --default_root_dir "${RUN_DIR}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --results_dir "${RESULTS_DIR}"
  --resume_mode "${RESUME_MODE}"
  --strategy "${STRATEGY}"
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
  --early_stopping_min_delta "${EARLY_STOPPING_MIN_DELTA}"
  --early_stopping_monitor "${EARLY_STOPPING_MONITOR}"
  --early_stopping_mode "${EARLY_STOPPING_MODE}"
  --save_checkpoints
)

if [[ "${LOGGER_TYPE}" == "wandb" ]]; then
  TRAIN_CMD+=(--wandb_project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    TRAIN_CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_GROUP}" ]]; then
    TRAIN_CMD+=(--wandb_group "${WANDB_GROUP}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    TRAIN_CMD+=(--wandb_tags "${WANDB_TAGS}")
  fi
  if [[ "${WANDB_OFFLINE}" == "1" ]]; then
    TRAIN_CMD+=(--wandb_offline)
  fi
  if [[ "${WANDB_LOG_MODEL}" == "1" ]]; then
    TRAIN_CMD+=(--wandb_log_model)
  fi
fi

if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${TRAIN_EXTRA_ARGS}"
  TRAIN_CMD+=("${extra_args[@]}")
fi

"${TRAIN_CMD[@]}"
