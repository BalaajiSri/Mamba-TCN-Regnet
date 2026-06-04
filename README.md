# Mamba-TCN-Regnet

`Mamba-TCN-Regnet` is a paper-inspired trading research repo built from the stable infrastructure patterns used in `CryptoMamba`, but kept as a separate standalone experiment.

## What Is Reused

- Groove cluster submission flow
- Lightning-based training and evaluation entry points
- YAML-driven configuration pattern
- BTC data loading and split caching flow
- Run directory structure with `run_info.yaml`
- Same conda environment and package stack

## What Is New

- HybridSSM-Trader-style hybrid backbone with:
  - input preprocessing and normalization
  - TCN local context blocks
  - parallel Mamba branches
  - multi-head attention global context
  - gated feature fusion and residual refinement
  - adaptive volatility-aware smoothing
- New experiment baselines:
  - TCN-only
  - Mamba-only
  - Attention-only
  - Hybrid
- Multi-task signal head that supports both:
  - next-step price prediction
  - action logits for `{sell, hold, buy}`
- First-pass RL scaffold:
  - trading environment
  - replay buffer
  - Double DQN / Dueling DQN-ready agent
  - Grove submission script for DQN training

## Data

This repo defaults to the same daily BTC source shape already used by the reference repo:

- [data/btc_daily_full.csv](/Users/sribalaajinatarajankalaivendan/Desktop/AdaptConference/Mamba-TCN-Regnet/data/btc_daily_full.csv)

The default data config is:

- [configs/data_configs/btc_1d_full.yaml](/Users/sribalaajinatarajankalaivendan/Desktop/AdaptConference/Mamba-TCN-Regnet/configs/data_configs/btc_1d_full.yaml)

## Signal Training

```bash
conda activate cryptomamba
python3 scripts/training.py --config hybrid_ssm_trader_hybrid_1d
```

Evaluate and backtest:

```bash
python3 scripts/evaluation.py --config hybrid_ssm_trader_hybrid_1d --run_dir runs/hybrid_ssm_trader_hybrid_1d/<run_name>
python3 scripts/simulate_trade.py --config hybrid_ssm_trader_hybrid_1d --run_dir runs/hybrid_ssm_trader_hybrid_1d/<run_name>
```

Use the learned action head instead of price heuristics:

```bash
python3 scripts/simulate_trade.py \
  --config hybrid_ssm_trader_hybrid_1d \
  --run_dir runs/hybrid_ssm_trader_hybrid_1d/<run_name> \
  --decision_source action_head
```

## Groove Runs

```bash
export CONDA_ENV=cryptomamba
bash cluster/grove/submit_grove_pipeline.sh hybrid_ssm_trader_hybrid_1d
```

W&B defaults use the new project name:

- `WANDB_PROJECT=MambaTCNRegNet`

## RL Skeleton

Train the first-pass DQN scaffold with the signal backbone as an encoder:

```bash
python3 scripts/train_dqn.py \
  --config hybrid_ssm_trader_dqn_1d \
  --signal_config hybrid_ssm_trader_hybrid_1d \
  --signal_ckpt runs/hybrid_ssm_trader_hybrid_1d/<run_name>/checkpoints/last.ckpt
```

Cluster submission:

```bash
export CONDA_ENV=cryptomamba
bash cluster/grove/submit_grove_dqn.sh hybrid_ssm_trader_dqn_1d hybrid_ssm_trader_hybrid_1d /absolute/path/to/last.ckpt
```

## Key Configs

- Hybrid model: [configs/models/HybridSSM-Trader/hybrid_v1.yaml](/Users/sribalaajinatarajankalaivendan/Desktop/AdaptConference/Mamba-TCN-Regnet/configs/models/HybridSSM-Trader/hybrid_v1.yaml)
- TCN baseline: [configs/models/Baselines/tcn_v1.yaml](/Users/sribalaajinatarajankalaivendan/Desktop/AdaptConference/Mamba-TCN-Regnet/configs/models/Baselines/tcn_v1.yaml)
- Mamba baseline: [configs/models/Baselines/mamba_v1.yaml](/Users/sribalaajinatarajankalaivendan/Desktop/AdaptConference/Mamba-TCN-Regnet/configs/models/Baselines/mamba_v1.yaml)
- Attention baseline: [configs/models/Baselines/attention_v1.yaml](/Users/sribalaajinatarajankalaivendan/Desktop/AdaptConference/Mamba-TCN-Regnet/configs/models/Baselines/attention_v1.yaml)
- RL config: [configs/rl/hybrid_ssm_trader_dqn_1d.yaml](/Users/sribalaajinatarajankalaivendan/Desktop/AdaptConference/Mamba-TCN-Regnet/configs/rl/hybrid_ssm_trader_dqn_1d.yaml)

