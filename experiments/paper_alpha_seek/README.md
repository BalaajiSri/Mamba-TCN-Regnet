# AlphaSeek Paper Evidence

## Paper Framing

This package supports the ADAPT poster abstract, **Learning Multi-Scale Market Representations for Short Horizon Cryptocurrency Trading**. The evidence should be presented as research in progress: the goal is to test whether combining TCN, Mamba state-space modules, attention, gated fusion, and adaptive smoothing produces useful market representations. It should not be written as investment advice or as a final deployment-ready trading system.

## Research Questions

- Does multi-scale feature learning improve short-horizon BTC signal quality under noisy market conditions?
- Which architectural components contribute most to forecasting and representation quality?
- Does a downstream sell/hold/buy DQN benefit from the learned latent state representation?
- How sensitive are results to validation window, volatility regime, trading costs, and turnover?

## Dataset And Split

- Raw data: `data/btc_daily_full.csv`
- Paper data config: `configs/data_configs/btc_1d_paper.yaml`
- Train: 2018-09-17 to 2024-01-01
- Validation: 2024-01-01 to 2025-01-01
- Frozen test: 2025-01-01 to 2026-03-13

Validation is used for checkpoint selection and hyperparameter decisions. The frozen test split is used only for final reporting.

## Representation Jobs

The Slurm matrix runs these configs across seeds `23`, `29`, and `37`:

- `paper_hybrid_1d`
- `paper_tcn_1d`
- `paper_mamba_1d`
- `paper_attention_1d`
- `paper_hybrid_no_features_1d`
- `paper_hybrid_no_action_1d`

Submit or resubmit the matrix:

```bash
cd /home/nsribalaaji/try/Mamba-TCN-Regnet
CONDA_ENV=cryptomamba bash experiments/paper_alpha_seek/run_paper_matrix.sh
```

Current submitted jobs are under `runs/paper_alpha_seek`. Dependent validation jobs run automatically after each training job succeeds.

## Downstream DQN Jobs

After a representation run finishes and has `run_info.yaml` with a checkpoint, submit a DQN job against that learned state encoder:

```bash
cd /home/nsribalaaji/try/Mamba-TCN-Regnet
CONDA_ENV=cryptomamba \
  experiments/paper_alpha_seek/submit_dqn_from_run.sh \
  runs/paper_alpha_seek/paper_hybrid_1d/<completed-run-dir>
```

This uses `configs/rl/alphaseek_dqn_paper_1d.yaml` and stores RL artifacts under `runs/paper_alpha_seek_rl/`.

## Metrics

Forecasting metrics: RMSE, MAE, MAPE, directional accuracy, and persistence-baseline deltas.

Trading metrics: total return, max drawdown, Sharpe, Sortino, final balance, action counts, and transaction-cost assumptions.

## Artifacts

Each evaluation writes:

- `metrics.txt` for quick inspection
- `metrics.json` for tables
- `predictions.jsonl` for audits and plotting
- `trading_metrics_*.json` for trading evidence

Aggregate completed run directories:

```bash
python3 scripts/paper_aggregate.py \
  --run_root runs/paper_alpha_seek \
  --output experiments/paper_alpha_seek/metrics/aggregate_metrics.json
```

## Reporting Guidance

Use careful language in the paper/poster: report this as planned and ongoing evaluation of representation quality, robustness, and downstream decision support. Emphasize limitations, transaction costs, turnover, drawdown, and frozen test evaluation.
