#!/bin/bash
#SBATCH -J mambatcnregnet-regime
#SBATCH -p compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%x-%j.out

# Group D – Market regime analysis
# Reads completed evaluation results from RUN_ROOT and produces
# per-regime tables (bull / bear / sideways / high-vol / low-vol).
#
# Prerequisite: scripts/regime_analysis.py must exist.
# See research plan for regime definitions and required output format.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_hybrid_ssm_trader}"
REGIME_OUTPUT="${REGIME_OUTPUT:-${PROJECT_DIR}/experiments/paper_hybrid_ssm_trader/metrics/regime_metrics.json}"
BULL_THRESHOLD="${BULL_THRESHOLD:-0.02}"
BEAR_THRESHOLD="${BEAR_THRESHOLD:--0.02}"
VOL_HIGH_PERCENTILE="${VOL_HIGH_PERCENTILE:-75}"
VOL_LOW_PERCENTILE="${VOL_LOW_PERCENTILE:-25}"
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

mkdir -p "${TMPDIR}"
export TMPDIR
cd "${PROJECT_DIR}"
activate_python
configure_python_runtime_libs

echo "Regime analysis: run_root=${RUN_ROOT}"
echo "Output: ${REGIME_OUTPUT}"

python3 scripts/regime_analysis.py \
  --run_root "${RUN_ROOT}" \
  --output "${REGIME_OUTPUT}" \
  --bull_threshold "${BULL_THRESHOLD}" \
  --bear_threshold "${BEAR_THRESHOLD}" \
  --vol_high_percentile "${VOL_HIGH_PERCENTILE}" \
  --vol_low_percentile "${VOL_LOW_PERCENTILE}"
