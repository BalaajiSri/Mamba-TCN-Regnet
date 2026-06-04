#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Paper v3 — full re-run on feature/arch-improvements (HybridSSM-Trader)
#
# BEFORE RUNNING:
#   1. python scripts/fetch_asset_data.py      # downloads ETH + BNB CSVs
#   2. git checkout feature/arch-improvements
#
# Groups:
#   A  — core baselines: hybrid vs single-branch (BTC)
#   B  — ablations: gate, action head, smoother, branches, features
#   F  — loss variants: GMADL+focal, GMADL+ordinal, GMADL+retce
#   G  — multi-asset: ETH + BNB (hybrid only, 5 seeds)
#   H  — sequence length: T=15, 30, 45, 60 (BTC hybrid)
#   Classical baselines: ARIMA + MA (CPU, no dependency)
#   Walk-forward: rolling eval (after A finishes)
#   Aggregate: paper metrics (after A+B+F+G+H)
#
# Usage:
#   CONDA_ENV=cryptomamba bash cluster/grove/submit_paper_v3_fixed_arch.sh
#   CONDA_ENV=cryptomamba PARTITION=a100 GRES=gpu:a100:1 bash cluster/grove/submit_paper_v3_fixed_arch.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CONDA_ENV:?Set CONDA_ENV, e.g. CONDA_ENV=cryptomamba}"

# ── Verify correct branch ────────────────────────────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "DETACHED")
if [[ "${CURRENT_BRANCH}" != "feature/arch-improvements" ]]; then
  echo "ERROR: must be on feature/arch-improvements (currently: ${CURRENT_BRANCH})"
  echo "  Run: git checkout feature/arch-improvements"
  exit 1
fi
COMMIT=$(git rev-parse --short HEAD)

# ── Verify asset data exists ─────────────────────────────────────────────────
for asset_csv in data/eth_daily_full.csv data/bnb_daily_full.csv data/sol_daily_full.csv; do
  if [[ ! -f "${asset_csv}" ]]; then
    echo "ERROR: ${asset_csv} not found. Run: python scripts/fetch_asset_data.py --all"
    exit 1
  fi
done

export CONDA_ENV
export PARTITION="${PARTITION:-a100}"
export GRES="${GRES:-gpu:a100:1}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export MEM="${MEM:-80G}"
export EMAIL="${EMAIL:-}"
export PREEMPT="${PREEMPT:-0}"
export RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/paper_v3}"
export LOGGER_TYPE="${LOGGER_TYPE:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-MambaTCNRegNet}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_OFFLINE="${WANDB_OFFLINE:-0}"

SEEDS=(23 29 37 41 53)

echo "═══════════════════════════════════════════════════"
echo "  HybridSSM-Trader  Paper v3"
echo "  Branch    : ${CURRENT_BRANCH} @ ${COMMIT}"
echo "  RUN_ROOT  : ${RUN_ROOT}"
echo "  PARTITION : ${PARTITION}  GRES: ${GRES}"
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
    WANDB_TAGS="grove,${config},${group_tag},seed${seed},v3,commit_${COMMIT}" \
    bash cluster/grove/submit_grove_pipeline.sh
  )
  echo "${output}" >&2
  awk '/Validation job submitted:/ {print $4}' <<< "${output}" | tail -n 1
}

submit_group() {
  local group_tag="$1"; shift
  local configs=("$@")
  local val_ids=()
  for config in "${configs[@]}"; do
    for seed in "${SEEDS[@]}"; do
      val_id=$(submit_one "${config}" "${seed}" "${group_tag}")
      val_ids+=("${val_id}")
      sleep 1
    done
  done
  echo "${val_ids[@]}"
}

# ── Group A: core baselines ───────────────────────────────────────────────────
echo ""; echo "── Group A: core baselines (BTC) ──"
read -ra ga_val_ids <<< "$(submit_group group_a \
  paper_hybrid_1d paper_tcn_1d paper_mamba_1d paper_attention_1d)"
echo "Group A: ${#ga_val_ids[@]} jobs queued."

# ── Group B: ablations ────────────────────────────────────────────────────────
echo ""; echo "── Group B: ablations ──"
read -ra gb_val_ids <<< "$(submit_group group_b \
  paper_hybrid_no_tcn_1d \
  paper_hybrid_no_mamba_1d \
  paper_hybrid_no_attn_1d \
  paper_hybrid_simple_fusion_1d \
  paper_hybrid_no_action_1d \
  paper_hybrid_low_action_1d \
  paper_hybrid_high_action_1d \
  paper_hybrid_no_features_1d \
  paper_hybrid_no_smooth_1d)"
echo "Group B: ${#gb_val_ids[@]} jobs queued."

# ── Group F: loss function variants ──────────────────────────────────────────
echo ""; echo "── Group F: loss function ablation ──"
read -ra gf_val_ids <<< "$(submit_group group_f \
  paper_hybrid_gmadl_focal_1d \
  paper_hybrid_gmadl_ordinal_1d \
  paper_hybrid_gmadl_retce_1d)"
echo "Group F: ${#gf_val_ids[@]} jobs queued."

# ── Group G: multi-asset generalisation ──────────────────────────────────────
echo ""; echo "── Group G: multi-asset (ETH + BNB) ──"
read -ra gg_val_ids <<< "$(submit_group group_g \
  paper_hybrid_eth_1d \
  paper_hybrid_bnb_1d)"
echo "Group G: ${#gg_val_ids[@]} jobs queued."

# ── Group H: sequence length ablation ────────────────────────────────────────
echo ""; echo "── Group H: sequence length (T=15,45,60; T=30 is Group A) ──"
read -ra gh_val_ids <<< "$(submit_group group_h \
  paper_hybrid_seq15_1d \
  paper_hybrid_seq45_1d \
  paper_hybrid_seq60_1d)"
echo "Group H: ${#gh_val_ids[@]} jobs queued."

# ── Group I: deep learning baselines (Transformer + LSTM) ────────────────────
echo ""; echo "── Group I: Transformer + LSTM baselines ──"
read -ra gi_val_ids <<< "$(submit_group group_i \
  paper_transformer_1d \
  paper_lstm_1d)"
echo "Group I: ${#gi_val_ids[@]} jobs queued."

# ── Group J: action threshold τ sensitivity ───────────────────────────────────
echo ""; echo "── Group J: action threshold τ sensitivity ──"
read -ra gj_val_ids <<< "$(submit_group group_j \
  paper_hybrid_tau0001_1d \
  paper_hybrid_tau005_1d \
  paper_hybrid_tau010_1d \
  paper_hybrid_tau020_1d)"
echo "Group J: ${#gj_val_ids[@]} jobs queued."

# ── Group G extended: add SOL ─────────────────────────────────────────────────
echo ""; echo "── Group G-SOL: SOL generalisation ──"
read -ra gsol_val_ids <<< "$(submit_group group_g \
  paper_hybrid_sol_1d)"
echo "Group G-SOL: ${#gsol_val_ids[@]} jobs queued."

# ── Classical baselines (CPU only, no dependency) ─────────────────────────────
echo ""; echo "── Classical baselines (ARIMA + MA, CPU) ──"
classical_job_id=$(sbatch \
  --parsable \
  --partition="${PARTITION}" \
  --cpus-per-task=8 \
  --mem=16G \
  --gres="" \
  --time=04:00:00 \
  --job-name="hybridssm-classical" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},CONDA_ENV=${CONDA_ENV},LOG_WANDB=$([ "${LOGGER_TYPE}" = "wandb" ] && echo 1 || echo 0),WANDB_PROJECT=${WANDB_PROJECT}" \
  cluster/grove/classical_baselines.sbatch)
echo "Classical baselines: ${classical_job_id}"

# ── Walk-forward evaluation (after Group A) ───────────────────────────────────
echo ""; echo "── Walk-forward eval (after Group A) ──"
ga_dep_str="$(IFS=:; echo "${ga_val_ids[*]}")"
wf_job_id=$(sbatch \
  --parsable \
  --partition="${PARTITION}" \
  --cpus-per-task=4 \
  --mem=8G \
  --time=01:00:00 \
  --dependency="afterok:${ga_dep_str}" \
  --job-name="hybridssm-walk-forward" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV},LOG_WANDB=$([ "${LOGGER_TYPE}" = "wandb" ] && echo 1 || echo 0),WANDB_PROJECT=${WANDB_PROJECT}" \
  cluster/grove/walk_forward.sbatch)
echo "Walk-forward: ${wf_job_id}"

# ── Aggregate (after A+B+F+G+H) ──────────────────────────────────────────────
echo ""; echo "── Aggregate paper metrics ──"
all_val_ids=("${ga_val_ids[@]}" "${gb_val_ids[@]}" "${gf_val_ids[@]}" "${gg_val_ids[@]}" "${gh_val_ids[@]}" "${gi_val_ids[@]}" "${gj_val_ids[@]}" "${gsol_val_ids[@]}")
dep_str="$(IFS=:; echo "${all_val_ids[*]}")"
agg_job_id=$(sbatch \
  --parsable \
  --dependency="afterok:${dep_str}" \
  --partition="${PARTITION}" \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --job-name="hybridssm-aggregate-v3" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV}" \
  cluster/grove/aggregate_paper.sbatch)
echo "Aggregate: ${agg_job_id}"

# ── Summary ───────────────────────────────────────────────────────────────────
total_gpu=$((${#ga_val_ids[@]} + ${#gb_val_ids[@]} + ${#gf_val_ids[@]} + ${#gg_val_ids[@]} + ${#gh_val_ids[@]} + ${#gi_val_ids[@]} + ${#gj_val_ids[@]} + ${#gsol_val_ids[@]}))
echo ""
echo "═══════════════════════════════════════════════════"
echo "  All jobs submitted."
printf "  GPU jobs  : %d validation jobs\n" "${total_gpu}"
echo "  CPU jobs  : classical baselines"
echo "  Monitor   : squeue -u \$USER"
echo "  WandB     : https://wandb.ai/${WANDB_ENTITY:-}/${WANDB_PROJECT}"
echo "  Cancel all: scancel \$(squeue -u \$USER -h -o %i | tr '\\n' ' ')"
echo ""
echo "  Group breakdown:"
echo "    A  (4 configs × 5 seeds = 20)  core BTC baselines"
echo "    B  (9 configs × 5 seeds = 45)  component ablations"
echo "    F  (3 configs × 5 seeds = 15)  loss function variants"
echo "    G  (2 configs × 5 seeds = 10)  ETH + BNB generalisation"
echo "    H  (3 configs × 5 seeds = 15)  sequence length T=15,45,60"
printf "    Total GPU: %d jobs\n" "${total_gpu}"
echo "═══════════════════════════════════════════════════"
