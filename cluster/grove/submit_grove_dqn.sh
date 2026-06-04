#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RL_CONFIG="${RL_CONFIG:-${1:-}}"
SIGNAL_CONFIG="${SIGNAL_CONFIG:-${2:-}}"
SIGNAL_CKPT="${SIGNAL_CKPT:-${3:-}}"

if [[ -z "${RL_CONFIG}" || -z "${SIGNAL_CONFIG}" || -z "${SIGNAL_CKPT}" ]]; then
  echo "Usage: submit_grove_dqn.sh <rl_config> <signal_config> <signal_ckpt>"
  exit 1
fi

PARTITION="${PARTITION:-compute}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

sbatch \
  --partition="${PARTITION}" \
  --gres="${GRES}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --job-name="mambatcnregnet-dqn-${RL_CONFIG}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},RL_CONFIG=${RL_CONFIG},SIGNAL_CONFIG=${SIGNAL_CONFIG},SIGNAL_CKPT=${SIGNAL_CKPT}" \
  "${SCRIPT_DIR}/train_dqn_grove.sbatch"

