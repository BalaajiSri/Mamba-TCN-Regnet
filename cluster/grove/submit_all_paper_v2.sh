#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Master submission script – full experiment suite for the paper
#
# Usage:
#   CONDA_ENV=cryptomamba bash cluster/grove/submit_all_paper_v2.sh
#
# Required:
#   CONDA_ENV            – conda environment name
#
# Key optional overrides:
#   EMAIL                – Slurm mail on END/FAIL
#   PARTITION            – default: compute
#   GRES                 – default: gpu:1
#   CPUS_PER_TASK        – default: 8 (training), 4 (validation), 2 (CPU jobs)
#   MEM                  – default: 64G (training), 32G (validation)
#   PREEMPT              – 1 = use preemptible QOS (Grove)
#   WANDB_PROJECT        – default: MambaTCNRegNet
#   WANDB_ENTITY         – your W&B team/user name (set if using team project)
#   WANDB_OFFLINE        – 1 = offline mode (sync later with `wandb sync`)
#   RUN_ROOT             – default: <project>/runs/paper_hybrid_ssm_trader
#   SKIP_GROUP_E         – 1 = skip negative controls
#   SKIP_CLASSICAL       – 1 = skip ARIMA/MA baselines
#   SKIP_WALK_FORWARD    – 1 = skip walk-forward eval
#
# Job dependency chain:
#   Group A ─┐
#   Group B ─┼──▶ Group D (regime + walk-forward, after all validations)
#   Group C  │
#   Group E ─┘  (submitted independently, after Group A training jobs)
#   Classical baselines (submitted independently, CPU only)
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

export CONDA_ENV
export PARTITION="${PARTITION:-compute}"
export GRES="${GRES:-gpu:1}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export MEM="${MEM:-64G}"
export EMAIL="${EMAIL:-}"
export PREEMPT="${PREEMPT:-0}"
export RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_hybrid_ssm_trader}"
export LOGGER_TYPE="${LOGGER_TYPE:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
SKIP_GROUP_E="${SKIP_GROUP_E:-0}"
SKIP_CLASSICAL="${SKIP_CLASSICAL:-0}"
SKIP_WALK_FORWARD="${SKIP_WALK_FORWARD:-0}"

SEEDS=(23 29 37 41 53)

echo "═══════════════════════════════════════════════════"
echo "  Mamba-TCN-RegNet full experiment suite"
echo "  RUN_ROOT  : ${RUN_ROOT}"
echo "  PARTITION : ${PARTITION}  GRES: ${GRES}"
echo "  LOGGER    : ${LOGGER_TYPE}  PROJECT: ${WANDB_PROJECT}"
echo "  SEEDS     : ${SEEDS[*]}"
echo "═══════════════════════════════════════════════════"

# ── helper ────────────────────────────────────────────────────────────────────
submit_one() {
  local config="$1" seed="$2" group_tag="$3"
  local run_name="${config}-seed${seed}-$(date +%Y%m%d-%H%M%S)"
  local output
  output=$(
    RUN_NAME="${run_name}" \
    CONFIG_NAME="${config}" \
    TRAIN_EXTRA_ARGS="--seed ${seed}" \
    EARLY_STOPPING_PATIENCE="30" \
    EARLY_STOPPING_MIN_DELTA="0.0001" \
    WANDB_GROUP="${group_tag}" \
    WANDB_TAGS="grove,${config},${group_tag},seed${seed}" \
    bash cluster/grove/submit_grove_pipeline.sh
  )
  echo "${output}" >&2
  awk '/Validation job submitted:/ {print $4}' <<< "${output}" | tail -n 1
}

# ── Group A ───────────────────────────────────────────────────────────────────
echo ""
echo "── Group A: core baselines ──"
GA_CONFIGS=(paper_hybrid_1d paper_tcn_1d paper_mamba_1d paper_attention_1d)
ga_val_ids=()
for config in "${GA_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    val_id=$(submit_one "${config}" "${seed}" "group_a")
    ga_val_ids+=("${val_id}")
    sleep 1
  done
done
echo "Group A: ${#ga_val_ids[@]} validation jobs queued."

# ── Group B ───────────────────────────────────────────────────────────────────
echo ""
echo "── Group B: ablations ──"
GB_CONFIGS=(
  paper_hybrid_no_tcn_1d
  paper_hybrid_no_mamba_1d
  paper_hybrid_no_attn_1d
  paper_hybrid_simple_fusion_1d
  paper_hybrid_no_action_1d
  paper_hybrid_low_action_1d
  paper_hybrid_high_action_1d
  paper_hybrid_no_features_1d
  paper_hybrid_no_smooth_1d
)
gb_val_ids=()
for config in "${GB_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    val_id=$(submit_one "${config}" "${seed}" "group_b")
    gb_val_ids+=("${val_id}")
    sleep 1
  done
done
echo "Group B: ${#gb_val_ids[@]} validation jobs queued."

# ── Group C ───────────────────────────────────────────────────────────────────
echo ""
echo "── Group C: complexity ──"
GC_CONFIGS=(
  paper_hybrid_small_1d
  paper_hybrid_large_1d
  paper_hybrid_tcn_shallow_1d
  paper_hybrid_tcn_deep_1d
  paper_hybrid_mamba_2_1d
  paper_hybrid_mamba_8_1d
)
gc_val_ids=()
for config in "${GC_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    val_id=$(submit_one "${config}" "${seed}" "group_c")
    gc_val_ids+=("${val_id}")
    sleep 1
  done
done
echo "Group C: ${#gc_val_ids[@]} validation jobs queued."

# ── Group D: regime + walk-forward (after A+B+C) ──────────────────────────────
echo ""
echo "── Group D: regime analysis + walk-forward ──"
all_ab_ids=("${ga_val_ids[@]}" "${gb_val_ids[@]}" "${gc_val_ids[@]}")
dep_str="$(IFS=:; echo "${all_ab_ids[*]}")"

# Regime analysis
regime_job_id=$(sbatch \
  --parsable \
  --partition="${PARTITION}" \
  --dependency="afterok:${dep_str}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV},LOG_WANDB=$([ "${LOGGER_TYPE}" = "wandb" ] && echo 1 || echo 0),WANDB_PROJECT=${WANDB_PROJECT}" \
  cluster/grove/regime_analysis.sbatch)
echo "Regime analysis: ${regime_job_id}"

# Walk-forward eval
if [[ "${SKIP_WALK_FORWARD}" != "1" ]]; then
  wf_job_id=$(sbatch \
    --parsable \
    --partition="${PARTITION}" \
    --cpus-per-task=4 \
    --mem=8G \
    --time=01:00:00 \
    --dependency="afterok:${dep_str}" \
    --job-name="mambatcnregnet-walk-forward" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV},LOG_WANDB=$([ "${LOGGER_TYPE}" = "wandb" ] && echo 1 || echo 0),WANDB_PROJECT=${WANDB_PROJECT}" \
    cluster/grove/walk_forward.sbatch)
  echo "Walk-forward eval: ${wf_job_id}"
fi

# ── Classical baselines (no dependency, CPU only) ─────────────────────────────
if [[ "${SKIP_CLASSICAL}" != "1" ]]; then
  echo ""
  echo "── Classical baselines (ARIMA + MA) ──"
  classical_job_id=$(sbatch \
    --parsable \
    --partition="${PARTITION}" \
    --cpus-per-task=8 \
    --mem=16G \
    --time=04:00:00 \
    --job-name="mambatcnregnet-classical" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONDA_ENV=${CONDA_ENV},LOG_WANDB=$([ "${LOGGER_TYPE}" = "wandb" ] && echo 1 || echo 0),WANDB_PROJECT=${WANDB_PROJECT}" \
    cluster/grove/classical_baselines.sbatch)
  echo "Classical baselines: ${classical_job_id}"
fi

# ── Group E: negative controls ────────────────────────────────────────────────
if [[ "${SKIP_GROUP_E}" != "1" ]]; then
  echo ""
  echo "── Group E: negative controls ──"
  bash cluster/grove/submit_group_e_negcontrol.sh
fi

# ── Aggregate (after all A+B+C+D) ────────────────────────────────────────────
echo ""
echo "── Aggregate paper metrics ──"
all_dep="${dep_str}:${regime_job_id}"
agg_job_id=$(sbatch \
  --parsable \
  --dependency="afterok:${all_dep}" \
  --partition="${PARTITION}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV}" \
  cluster/grove/aggregate_paper.sbatch)
echo "Aggregate: ${agg_job_id}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All jobs submitted."
echo "  Monitor:  squeue -u \$USER"
echo "  WandB:    https://wandb.ai/${WANDB_ENTITY:-}/${WANDB_PROJECT}"
echo "  Cancel all:  scancel \$(squeue -u \$USER -h -o %i | tr '\\n' ' ')"
echo "═══════════════════════════════════════════════════"
