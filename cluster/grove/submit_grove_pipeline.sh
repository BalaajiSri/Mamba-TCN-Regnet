#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-${1:-}}"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "ERROR: set CONFIG_NAME or pass it as the first argument."
  exit 1
fi

RUN_NAME="${RUN_NAME:-${CONFIG_NAME}-$(date +%Y%m%d-%H%M%S)}"
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
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"

PARTITION="${PARTITION:-compute}"
NODELIST="${NODELIST:-}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
TIME_MIN="${TIME_MIN:-}"
CONSTRAINT="${CONSTRAINT:-}"
PREEMPT="${PREEMPT:-0}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL}"
EMAIL="${EMAIL:-}"

VALIDATION_PARTITION="${VALIDATION_PARTITION:-${PARTITION}}"
VALIDATION_NODELIST="${VALIDATION_NODELIST:-${NODELIST}}"
VALIDATION_GRES="${VALIDATION_GRES:-gpu:1}"
VALIDATION_CPUS_PER_TASK="${VALIDATION_CPUS_PER_TASK:-4}"
VALIDATION_MEM="${VALIDATION_MEM:-32G}"
VALIDATION_TIME_LIMIT="${VALIDATION_TIME_LIMIT:-04:00:00}"
VALIDATION_TIME_MIN="${VALIDATION_TIME_MIN:-}"
CKPT_SELECT="${CKPT_SELECT:-best}"
TRADE_SPLIT="${TRADE_SPLIT:-test}"
TRADE_MODE="${TRADE_MODE:-smart}"
LOGGER_TYPE="${LOGGER_TYPE:-tb}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-val/rmse}"
EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-min}"
WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-${CONFIG_NAME}}"
WANDB_TAGS="${WANDB_TAGS:-grove,${CONFIG_NAME}}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

train_cmd=(
  sbatch
  --parsable
  --partition="${PARTITION}"
  --gres="${GRES}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --time="${TIME_LIMIT}"
  --job-name="mambatcnregnet-train-${CONFIG_NAME}"
)

if [[ -n "${TIME_MIN}" ]]; then
  train_cmd+=(--time-min="${TIME_MIN}")
fi

if [[ -n "${CONSTRAINT}" ]]; then
  train_cmd+=(--constraint="${CONSTRAINT}")
fi

if [[ -n "${NODELIST}" ]]; then
  train_cmd+=(--nodelist="${NODELIST}")
fi

if [[ -n "${TRAIN_DEPENDENCY}" ]]; then
  train_cmd+=(--dependency="${TRAIN_DEPENDENCY}")
fi

if [[ -n "${EMAIL}" ]]; then
  train_cmd+=(--mail-user="${EMAIL}" --mail-type="${MAIL_TYPE}")
fi

if [[ "${PREEMPT}" == "1" ]]; then
  train_cmd+=(--qos=preempt --requeue)
fi

train_cmd+=(
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG_NAME=${CONFIG_NAME},RUN_NAME=${RUN_NAME},RUN_ROOT=${RUN_ROOT},RUN_DIR=${RUN_DIR},CHECKPOINT_DIR=${CHECKPOINT_DIR},RESULTS_DIR=${RESULTS_DIR},GRES=${GRES},LOGGER_TYPE=${LOGGER_TYPE},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA},EARLY_STOPPING_MONITOR=${EARLY_STOPPING_MONITOR},EARLY_STOPPING_MODE=${EARLY_STOPPING_MODE},WANDB_PROJECT=${WANDB_PROJECT},WANDB_ENTITY=${WANDB_ENTITY},WANDB_GROUP=${WANDB_GROUP},WANDB_TAGS=${WANDB_TAGS},WANDB_OFFLINE=${WANDB_OFFLINE},WANDB_LOG_MODEL=${WANDB_LOG_MODEL}"
  "${SCRIPT_DIR}/train_grove.sbatch"
)

train_job_id="$("${train_cmd[@]}")"

validate_cmd=(
  sbatch
  --parsable
  --dependency="afterok:${train_job_id}"
  --partition="${VALIDATION_PARTITION}"
  --gres="${VALIDATION_GRES}"
  --cpus-per-task="${VALIDATION_CPUS_PER_TASK}"
  --mem="${VALIDATION_MEM}"
  --time="${VALIDATION_TIME_LIMIT}"
  --job-name="mambatcnregnet-validate-${CONFIG_NAME}"
)

if [[ -n "${VALIDATION_TIME_MIN}" ]]; then
  validate_cmd+=(--time-min="${VALIDATION_TIME_MIN}")
fi

if [[ -n "${CONSTRAINT}" ]]; then
  validate_cmd+=(--constraint="${CONSTRAINT}")
fi

if [[ -n "${VALIDATION_NODELIST}" ]]; then
  validate_cmd+=(--nodelist="${VALIDATION_NODELIST}")
fi

if [[ -n "${EMAIL}" ]]; then
  validate_cmd+=(--mail-user="${EMAIL}" --mail-type="${MAIL_TYPE}")
fi

validate_cmd+=(
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG_NAME=${CONFIG_NAME},RUN_NAME=${RUN_NAME},RUN_ROOT=${RUN_ROOT},RUN_DIR=${RUN_DIR},RESULTS_DIR=${RESULTS_DIR},CKPT_SELECT=${CKPT_SELECT},TRADE_SPLIT=${TRADE_SPLIT},TRADE_MODE=${TRADE_MODE},THRESHOLD_CANDIDATES=${THRESHOLD_CANDIDATES:-0.0,0.001,0.002,0.003,0.005,0.01}"
  "${SCRIPT_DIR}/validate_grove.sbatch"
)

validate_job_id="$("${validate_cmd[@]}")"

echo "Training job submitted: ${train_job_id}"
echo "Validation job submitted: ${validate_job_id}"
echo "Run directory: ${RUN_DIR}"
