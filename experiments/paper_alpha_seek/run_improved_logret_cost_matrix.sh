#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

CONFIGS=(
  paper_hybrid_logret_cost_1d
  paper_mamba_logret_cost_1d
  paper_tcn_logret_cost_1d
)
SEEDS=(23 29 37)

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

# Default to a normal-QoS compute node with multiple GPUs.
# Override these if the scheduler state changes or you want A100/A6000 nodes.
PARTITION="${PARTITION:-compute}"
NODELIST="${NODELIST:-g108}"
GRES="${GRES:-gpu:rtx2080ti:1}"
VALIDATION_PARTITION="${VALIDATION_PARTITION:-${PARTITION}}"
VALIDATION_NODELIST="${VALIDATION_NODELIST:-${NODELIST}}"
VALIDATION_GRES="${VALIDATION_GRES:-${GRES}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
VALIDATION_CPUS_PER_TASK="${VALIDATION_CPUS_PER_TASK:-2}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_alpha_seek}"
THRESHOLD_CANDIDATES="${THRESHOLD_CANDIDATES:-0.0,0.001,0.002,0.003,0.005,0.01}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0001}"
SERIAL_SUBMIT="${SERIAL_SUBMIT:-1}"

validation_job_ids=()
previous_validation_job_id=""

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${config}-seed${seed}-$(date +%Y%m%d-%H%M%S)"
    echo "Submitting ${config} seed ${seed} on ${PARTITION}/${NODELIST} with ${GRES}"
    train_dependency=""
    if [[ "${SERIAL_SUBMIT}" == "1" && -n "${previous_validation_job_id}" ]]; then
      train_dependency="afterok:${previous_validation_job_id}"
    fi
    submit_output="$(
      RUN_ROOT="${RUN_ROOT}" \
      RUN_NAME="${run_name}" \
      CONFIG_NAME="${config}" \
      TRAIN_EXTRA_ARGS="--seed ${seed}" \
      TRAIN_DEPENDENCY="${train_dependency}" \
      PARTITION="${PARTITION}" \
      NODELIST="${NODELIST}" \
      GRES="${GRES}" \
      CPUS_PER_TASK="${CPUS_PER_TASK}" \
      VALIDATION_PARTITION="${VALIDATION_PARTITION}" \
      VALIDATION_NODELIST="${VALIDATION_NODELIST}" \
      VALIDATION_GRES="${VALIDATION_GRES}" \
      VALIDATION_CPUS_PER_TASK="${VALIDATION_CPUS_PER_TASK}" \
      THRESHOLD_CANDIDATES="${THRESHOLD_CANDIDATES}" \
      EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
      EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA}" \
      CONDA_ENV="${CONDA_ENV}" \
      bash cluster/grove/submit_grove_pipeline.sh
    )"
    echo "${submit_output}"
    validation_job_id="$(awk '/Validation job submitted:/ {print $4}' <<< "${submit_output}" | tail -n 1)"
    if [[ -z "${validation_job_id}" ]]; then
      echo "ERROR: failed to parse validation job id for ${config} seed ${seed}."
      exit 1
    fi
    validation_job_ids+=("${validation_job_id}")
    previous_validation_job_id="${validation_job_id}"
  done
done

dependency="$(IFS=:; echo "${validation_job_ids[*]}")"
aggregate_cmd=(
  sbatch
  --parsable
  --dependency="afterok:${dependency}"
  --partition="${AGGREGATE_PARTITION:-compute}"
  --cpus-per-task="${AGGREGATE_CPUS_PER_TASK:-2}"
  --mem="${AGGREGATE_MEM:-8G}"
  --time="${AGGREGATE_TIME_LIMIT:-01:00:00}"
  --job-name="mambatcnregnet-aggregate-improved"
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV}"
  cluster/grove/aggregate_paper.sbatch
)

aggregate_job_id="$("${aggregate_cmd[@]}")"

echo "Submitted ${#validation_job_ids[@]} validation jobs."
echo "Aggregate/figure job submitted after validations: ${aggregate_job_id}"
echo "Run root: ${RUN_ROOT}"
