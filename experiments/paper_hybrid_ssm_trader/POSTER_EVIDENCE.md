# Poster Evidence Notes

These notes support the ADAPT poster section on current results for **Learning Multi-Scale Market Representations for Short Horizon Cryptocurrency Trading**. Treat the findings as research-in-progress evidence, not deployment or investment claims.

## Source Artifacts

- Experiment summary: `experiments/paper_hybrid_ssm_trader/metrics/aggregate_metrics.json`
- Dataset/split protocol: `experiments/paper_hybrid_ssm_trader/README.md`
- Architecture implementation: `models/hybrid_ssm_trader_hybrid.py`
- TCN layer: `models/layers/tcn.py`
- Mamba state-space module: `models/cmamba.py`
- Fusion layer: `models/layers/fusion.py`
- Adaptive smoothing: `models/layers/smoothing.py`
- Example convergence log: `slurm-mambatcnregnet-train-paper_hybrid_no_features_1d-282833.out`

## Poster Figures

- `figures/test_rmse_by_model.png`: frozen test RMSE by representation, averaged over seeds 23, 29, and 37 with standard deviation.
- `figures/test_directional_accuracy_by_model.png`: frozen test directional accuracy by representation, with 0.50 reference line.
- `figures/rmse_by_split.png`: train/validation/test RMSE comparison for the main representation families.
- `figures/validation_convergence_hybrid_no_features_seed37.png`: validation RMSE convergence example from the completed Slurm log.
- `figures/architecture_overview.png`: implemented multi-scale backbone for context.

## Quantitative Takeaways

- The aggregate now keeps the latest completed run per `(config, seed)`, avoiding duplicate weighting from earlier resubmissions.
- Frozen test RMSE is tightly clustered for Hybrid, Mamba, TCN, and hybrid ablations: approximately 2191 to 2205 mean RMSE over seeds 23, 29, and 37.
- Mamba has the lowest mean frozen test RMSE in the current aggregate (`2191.53 +/- 6.29`), with the latest log-return+cost Mamba run nearly identical (`2191.62 +/- 6.70`).
- The latest log-return+cost runs remain close together: Hybrid `2195.66 +/- 11.52`, Mamba `2191.62 +/- 6.70`, and TCN `2201.31 +/- 11.97` test RMSE.
- Attention-only is less stable in the current runs (`2590.72 +/- 686.76` test RMSE), supporting the poster framing that attention alone is not sufficient under noisy market conditions.
- Directional accuracy remains near the 0.50 reference line across models, so the poster should frame this as early signal-quality evidence rather than a strong trading-performance claim.
- The convergence example reaches its best validation RMSE at epoch 23 and early-stops after epoch 53, showing that checkpoint selection and early stopping are active in the pipeline.

## Suggested Caption Language

Current frozen-test results show comparable forecast error across multi-scale and single-family temporal backbones, with higher seed instability in the attention-only baseline. Directional accuracy remains close to chance-level, motivating the next evaluation phase focused on walk-forward validation, transaction costs, drawdown, turnover, and regime sensitivity.
