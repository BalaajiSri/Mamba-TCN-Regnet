# Paper Results Summary

Compiled for **Learning Multi-Scale Market Representations for Short Horizon Cryptocurrency Trading** from the latest completed `paper_alpha_seek` runs.

## Source And Protocol

- Aggregate source: `experiments/paper_alpha_seek/metrics/aggregate_metrics.json`
- Run root: `runs/paper_alpha_seek`
- Latest-run policy: one latest completed run per `(config, seed)` pair
- Seeds: `23`, `29`, `37`
- Dataset: `data/btc_daily_full.csv`
- Train split: 2018-09-17 to 2024-01-01
- Validation split: 2024-01-01 to 2025-01-01
- Frozen test split: 2025-01-01 to 2026-03-13
- Transaction cost used in threshold/action selection runs: `0.001`

## Frozen Test Forecasting Results

Values are mean +/- sample standard deviation over three seeds.

| Model | Test RMSE | Test MAE | Test MAPE | Directional Accuracy | RMSE Delta vs Persistence |
|---|---:|---:|---:|---:|---:|
| Mamba original | 2191.53 +/- 6.29 | 1572.81 +/- 3.86 | 0.01678 +/- 0.00004 | 0.5103 +/- 0.0100 | -6.80 +/- 6.29 |
| Mamba log-return+cost | 2191.62 +/- 6.70 | 1572.86 +/- 3.92 | 0.01678 +/- 0.00004 | 0.5203 +/- 0.0183 | -6.89 +/- 6.70 |
| Hybrid no action head | 2192.46 +/- 3.84 | 1576.88 +/- 4.26 | 0.01680 +/- 0.00003 | 0.5103 +/- 0.0062 | -7.73 +/- 3.84 |
| Hybrid original | 2193.44 +/- 6.38 | 1578.20 +/- 7.62 | 0.01684 +/- 0.00010 | 0.4963 +/- 0.0138 | -8.72 +/- 6.38 |
| Hybrid log-return+cost | 2195.66 +/- 11.52 | 1581.67 +/- 14.04 | 0.01688 +/- 0.00017 | 0.5012 +/- 0.0179 | -10.94 +/- 11.52 |
| TCN original | 2200.15 +/- 9.42 | 1586.87 +/- 7.22 | 0.01696 +/- 0.00006 | 0.4822 +/- 0.0014 | -15.42 +/- 9.42 |
| TCN log-return+cost | 2201.31 +/- 11.97 | 1587.19 +/- 10.48 | 0.01695 +/- 0.00012 | 0.4855 +/- 0.0165 | -16.58 +/- 11.97 |
| Hybrid OHLCV only | 2204.95 +/- 17.06 | 1591.35 +/- 20.31 | 0.01702 +/- 0.00028 | 0.4930 +/- 0.0158 | -20.22 +/- 17.06 |
| Attention original | 2590.72 +/- 686.76 | 1957.27 +/- 663.72 | 0.02038 +/- 0.00625 | 0.5070 +/- 0.0076 | -405.99 +/- 686.76 |

Interpretation: the Mamba and hybrid families are tightly clustered on frozen-test RMSE, with the Mamba runs marginally lowest in this aggregate. None of the learned models beats the persistence baseline on RMSE, so this should be framed as representation-learning evidence, not final trading-performance evidence.

## Latest Log-Return+Cost Runs

| Model | Validation RMSE | Validation Directional Accuracy | Test RMSE | Test Directional Accuracy |
|---|---:|---:|---:|---:|
| Hybrid log-return+cost | 1903.82 +/- 7.26 | 0.5174 | 2195.66 +/- 11.52 | 0.5012 +/- 0.0179 |
| Mamba log-return+cost | 1907.89 +/- 7.80 | 0.4866 | 2191.62 +/- 6.70 | 0.5203 +/- 0.0183 |
| TCN log-return+cost | 1906.71 +/- 5.10 | 0.5194 | 2201.31 +/- 11.97 | 0.4855 +/- 0.0165 |

Latest run directories:

- `paper_hybrid_logret_cost_1d`: `paper_hybrid_logret_cost_1d-seed23-20260427-153237`, `paper_hybrid_logret_cost_1d-seed29-20260427-153237`, `paper_hybrid_logret_cost_1d-seed37-20260427-153237`
- `paper_mamba_logret_cost_1d`: `paper_mamba_logret_cost_1d-seed23-20260427-153237`, `paper_mamba_logret_cost_1d-seed29-20260427-153237`, `paper_mamba_logret_cost_1d-seed37-20260427-153238`
- `paper_tcn_logret_cost_1d`: `paper_tcn_logret_cost_1d-seed23-20260427-153238`, `paper_tcn_logret_cost_1d-seed29-20260427-153238`, `paper_tcn_logret_cost_1d-seed37-20260427-153238`

## Threshold Action Signal Results

The latest log-return+cost runs selected threshold `0.0` on validation action accuracy with transaction cost `0.001`.

| Model | Test Action Accuracy | Turnover | Sell Fraction | Hold Fraction | Buy Fraction |
|---|---:|---:|---:|---:|---:|
| Hybrid log-return+cost | 0.3581 +/- 0.0686 | 0.7229 | 0.2250 | 0.2771 | 0.4979 |
| Mamba log-return+cost | 0.2175 +/- 0.1008 | 0.3474 | 0.0827 | 0.6526 | 0.2647 |
| TCN log-return+cost | 0.3333 +/- 0.1533 | 0.7014 | 0.0050 | 0.2986 | 0.6964 |

Interpretation: threshold-derived action labels are not yet strong enough for performance claims. The hybrid run produces more active trading signals; Mamba is more hold-biased and lower turnover. These numbers support the paper's current framing around stabilising the downstream decision process.

## Slice Findings

Directional accuracy remains regime-sensitive:

- Hybrid log-return+cost: low/mid/high-volatility directional accuracy is `0.4963`, `0.5199`, and `0.4876`.
- Mamba log-return+cost: low/mid/high-volatility directional accuracy is `0.5309`, `0.5124`, and `0.5174`.
- TCN log-return+cost: low/mid/high-volatility directional accuracy is `0.4593`, `0.5274`, and `0.4701`.
- Trend slices show stronger accuracy in up-trend segments and weaker accuracy in down-trend segments, especially for TCN. Treat these as diagnostic slices because class balance and sample counts can differ across regimes.

## Paper-Ready Wording

Current frozen-test results show that the Mamba and hybrid representation families produce comparable short-horizon BTC forecast errors across three seeds, while the attention-only baseline is substantially less stable. Directional accuracy remains close to the 0.50 reference line and all learned predictors remain slightly behind a persistence baseline on RMSE. These results support the research-in-progress framing: the implemented multi-scale backbone is suitable for controlled ablation and representation-quality analysis, but stronger claims require walk-forward validation with transaction costs, turnover, drawdown, and regime sensitivity.

