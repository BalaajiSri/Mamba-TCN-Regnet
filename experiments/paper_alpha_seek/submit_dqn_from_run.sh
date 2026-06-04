#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <signal_run_dir> [signal_config] [rl_config]"
  exit 1
fi

SIGNAL_RUN_DIR="$1"
SIGNAL_CONFIG="${2:-paper_hybrid_1d}"
RL_CONFIG="${3:-alphaseek_dqn_paper_1d}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_INFO="${SIGNAL_RUN_DIR}/run_info.yaml"

if [[ ! -f "${RUN_INFO}" ]]; then
  echo "ERROR: missing run_info.yaml in ${SIGNAL_RUN_DIR}"
  exit 1
fi

SIGNAL_CKPT="$(RUN_INFO_PATH="${RUN_INFO}" python3 -c 'import os, yaml; from pathlib import Path; info = yaml.safe_load(Path(os.environ["RUN_INFO_PATH"]).read_text()) or {}; print(info.get("best_checkpoint") or info.get("last_checkpoint") or "")')"

if [[ -z "${SIGNAL_CKPT}" || ! -f "${SIGNAL_CKPT}" ]]; then
  echo "ERROR: unable to resolve checkpoint from ${RUN_INFO}"
  exit 1
fi

OUTPUT_DIR="${PROJECT_DIR}/runs/paper_alpha_seek_rl/$(basename "${SIGNAL_RUN_DIR}")"
CONDA_ENV="${CONDA_ENV:-cryptomamba}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${PROJECT_DIR}/cluster/grove/submit_grove_dqn.sh" "${RL_CONFIG}" "${SIGNAL_CONFIG}" "${SIGNAL_CKPT}"
