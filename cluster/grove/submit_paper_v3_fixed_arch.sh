#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Paper v3 — full re-run on feature/arch-improvements branch
#
# Runs on A100s with the fixed architecture:
#   - Softmax 3-way gate (fixes broken sigmoid formulation)
#   - Single LayerNorm input preprocessing (removes double-norm)
#   - Vectorized causal EMA smoother
#   - GMADL + focal/ordinal/retce loss variants (new Group F)
#
# Groups:
#   A  — core baselines (hybrid + 3 single-branch)
#   B  — ablations (gate, action head, smoother, etc.)
#   F  — loss function variants (GMADL+focal, GMADL+ordinal, GMADL+retce)
#   Agg— aggregate metrics (after A+B+F)
#
# Usage (run from the feature/arch-improvements branch):
#   git checkout feature/arch-improvements
#   CONDA_ENV=cryptomamba bash cluster/grove/submit_paper_v3_fixed_arch.sh
#
# A100 partition override example:
#   CONDA_ENV=cryptomamba PARTITION=a100 GRES=gpu:a100:1 bash cluster/grove/submit_paper_v3_fixed_arch.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

# ── Verify we are on the fixed-arch branch ────────────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "DETACHED")
if [[ "${CURRENT_BRANCH}" != "feature/arch-improvements" ]]; then
  echo "ERROR: must run from feature/arch-improvements branch (currently on: ${CURRENT_BRANCH})"
  echo "  Run: git checkout feature/arch-improvements"
  exit 1
fi
COMMIT=$(git rev-parse --short HEAD)
echo "Branch: ${CURRENT_BRANCH}  Commit: ${COMMIT}"

export CONDA_ENV
export PARTITION="${PARTITION:-a100}"
export GRES="${GRES:-gpu:a100:1}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export MEM="${MEM:-80G}"
export EMAIL="${EMAIL:-}"
export PREEMPT="${PREEMPT:-0}"
export RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_v3_fixed_arch}"
export LOGGER_TYPE="${LOGGER_TYPE:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_OFFLINE="${WANDB_OFFLINE:-0}"

SEEDS=(23 29 37 41 53)

echo "═══════════════════════════════════════════════════"
echo "  Paper v3 — fixed arch + new losses"
echo "  Branch    : ${CURRENT_BRANCH} @ ${COMMIT}"
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
    WANDB_GROUP="${group_tag}_v3" \
    WANDB_TAGS="grove,${config},${group_tag},seed${seed},v3_fixed_arch,commit_${COMMIT}" \
    bash cluster/grove/submit_grove_pipeline.sh
  )
  echo "${output}" >&2
  awk '/Validation job submitted:/ {print $4}' <<< "${output}" | tail -n 1
}

# ── Group A: core baselines ───────────────────────────────────────────────────
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

# ── Group B: ablations ────────────────────────────────────────────────────────
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

# ── Group F: loss function variants (NEW) ─────────────────────────────────────
echo ""
echo "── Group F: loss function ablation ──"
GF_CONFIGS=(
  paper_hybrid_gmadl_focal_1d
  paper_hybrid_gmadl_ordinal_1d
  paper_hybrid_gmadl_retce_1d
)
gf_val_ids=()
for config in "${GF_CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    val_id=$(submit_one "${config}" "${seed}" "group_f")
    gf_val_ids+=("${val_id}")
    sleep 1
  done
done
echo "Group F: ${#gf_val_ids[@]} validation jobs queued."

# ── Aggregate (after A+B+F) ───────────────────────────────────────────────────
echo ""
echo "── Aggregate paper metrics ──"
all_val_ids=("${ga_val_ids[@]}" "${gb_val_ids[@]}" "${gf_val_ids[@]}")
dep_str="$(IFS=:; echo "${all_val_ids[*]}")"

agg_job_id=$(sbatch \
  --parsable \
  --dependency="afterok:${dep_str}" \
  --partition="${PARTITION}" \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --job-name="mambatcnregnet-aggregate-v3" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV}" \
  cluster/grove/aggregate_paper.sbatch)
echo "Aggregate: ${agg_job_id}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  All jobs submitted."
echo "  Total jobs : $((${#ga_val_ids[@]} + ${#gb_val_ids[@]} + ${#gf_val_ids[@]})) validation jobs"
echo "  Monitor    : squeue -u \$USER"
echo "  WandB      : https://wandb.ai/${WANDB_ENTITY:-}/${WANDB_PROJECT}"
echo "  Cancel all : scancel \$(squeue -u \$USER -h -o %i | tr '\\n' ' ')"
echo ""
echo "  Groups run:"
echo "    A  — core baselines (hybrid vs single-branch)"
echo "    B  — ablations (gate, action head, smoother, etc.)"
echo "    F  — loss variants (GMADL+focal/ordinal/retce)"
echo "  Groups skipped (not needed for paper story):"
echo "    C  — complexity variants"
echo "    D  — regime analysis (re-add after deadline if desired)"
echo "    E  — negative controls"
echo "═══════════════════════════════════════════════════"
