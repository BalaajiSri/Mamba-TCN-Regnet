#!/bin/bash
#SBATCH -J mambatcnregnet-aggregate-paper
#SBATCH -p compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_hybrid_ssm_trader}"
AGGREGATE_OUTPUT="${AGGREGATE_OUTPUT:-${PROJECT_DIR}/experiments/paper_hybrid_ssm_trader/metrics/aggregate_metrics.json}"
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

  echo "ERROR: set CONDA_ENV or VENV_PATH before submitting the aggregate job."
  exit 1
}

configure_python_runtime_libs() {
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
}

mkdir -p "${TMPDIR}" "${MAMBA_CACHE_DIR}" "${PIP_CACHE_DIR}"
export TMPDIR
export MAMBA_CACHE_DIR
export PIP_CACHE_DIR

cd "${PROJECT_DIR}"
activate_python
configure_python_runtime_libs

echo "Aggregating paper runs from: ${RUN_ROOT}"
echo "Aggregate output: ${AGGREGATE_OUTPUT}"

python3 scripts/paper_aggregate.py \
  --run_root "${RUN_ROOT}" \
  --output "${AGGREGATE_OUTPUT}"

python3 experiments/paper_hybrid_ssm_trader/figures/make_poster_figures.py
