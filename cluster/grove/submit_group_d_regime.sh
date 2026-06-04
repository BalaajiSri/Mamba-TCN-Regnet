#!/bin/bash
# Group D – Submit regime analysis after Group A+B validations complete.
# Pass --dependency to chain after your last validation job IDs, e.g.:
#   AFTER_JOBS="12345:12346:12347" bash submit_group_d_regime.sh
#
# Or run standalone (no dependency) once all runs are done:
#   bash submit_group_d_regime.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_alpha_seek}"
AFTER_JOBS="${AFTER_JOBS:-}"   # colon-separated job IDs to wait for
PARTITION="${PARTITION:-compute}"
EMAIL="${EMAIL:-}"

dep_args=()
if [[ -n "${AFTER_JOBS}" ]]; then
  dep_args=(--dependency="afterok:${AFTER_JOBS}")
fi

mail_args=()
if [[ -n "${EMAIL}" ]]; then
  mail_args=(--mail-user="${EMAIL}" --mail-type="END,FAIL")
fi

job_id=$(sbatch \
  --parsable \
  --partition="${PARTITION}" \
  "${dep_args[@]}" \
  "${mail_args[@]}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV}" \
  "${SCRIPT_DIR}/regime_analysis.sbatch")

echo "Regime analysis job submitted: ${job_id}"
