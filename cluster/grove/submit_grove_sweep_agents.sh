#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
SWEEP_ID="${SWEEP_ID:-${2:-}}"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: set CONFIG_NAME or pass as first argument."
  exit 1
fi
if [[ -z "${SWEEP_ID}" ]]; then
  echo "ERROR: set SWEEP_ID or pass as second argument (e.g. team/project/sweepid)."
  exit 1
fi

AGENTS="${AGENTS:-3}"
PARTITION="${PARTITION:-compute}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH:-${PROJECT_DIR}/runs}/mamba_tcn_regnet_sweeps}"
WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${CONFIG_NAME}-sweep}"
WANDB_AGENT_COUNT="${WANDB_AGENT_COUNT:-1}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-${CPUS_PER_TASK}}"
SEED="${SEED:-23}"

mkdir -p "${RUN_ROOT}"

for ((i = 1; i <= AGENTS; i++)); do
  job_id="$(
    sbatch --parsable \
      --partition="${PARTITION}" \
      --gres="${GRES}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      --time="${TIME_LIMIT}" \
      --job-name="mambatcnregnet-sweep-${CONFIG_NAME}-${i}" \
      --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG_NAME=${CONFIG_NAME},SWEEP_ID=${SWEEP_ID},RUN_ROOT=${RUN_ROOT},WANDB_PROJECT=${WANDB_PROJECT},WANDB_ENTITY=${WANDB_ENTITY},WANDB_GROUP=${WANDB_GROUP},WANDB_AGENT_COUNT=${WANDB_AGENT_COUNT},ACCELERATOR=${ACCELERATOR},DEVICES=${DEVICES},NUM_WORKERS=${NUM_WORKERS},SEED=${SEED}" \
      "${SCRIPT_DIR}/sweep_agent_grove.sbatch"
  )"
  echo "Submitted sweep agent ${i}/${AGENTS}: ${job_id}"
done

echo "Submitted ${AGENTS} sweep agent jobs for sweep ${SWEEP_ID}."
