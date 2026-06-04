#!/usr/bin/env python3
"""Analyse and visualise the 3-way softmax gate weights of HybridSSMTrader.

For each test sample, records the mean gate weight assigned to the TCN (local),
Mamba (long-range), and Attention (global) branches as a function of market
volatility. Saves a CSV of per-sample weights and a matplotlib figure.

Usage:
    python scripts/gate_weight_analysis.py \
        --checkpoint runs/paper_v3/paper_hybrid_1d/<run>/checkpoints/best.ckpt \
        --data-config btc_1d_paper \
        --out-dir experiments/paper_hybrid_ssm_trader/figures/gate_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.hybrid_ssm_trader import HybridSSMTrader
from models.layers.fusion import GatedFeatureFusion


def _hook_gate_weights(module: GatedFeatureFusion, captured: list):
    """Register a forward hook that captures per-sample mean gate weights."""
    def hook(mod, inputs, output):
        # Recompute gate weights from the cached projections
        # We hook the GatedFeatureFusion and re-run just the gate
        local_h, long_h, global_h = inputs[0], inputs[1], inputs[2]
        import torch.nn.functional as F
        concat = torch.cat([
            mod.local_proj(local_h),
            mod.long_proj(long_h),
            mod.global_proj(global_h),
        ], dim=-1)
        gate_logits = mod.gate(concat)
        weights = F.softmax(gate_logits, dim=-1)  # (B, T, 3)
        # Mean over T → (B, 3)
        captured.append(weights.mean(dim=1).detach().cpu())
    return hook


def run_analysis(checkpoint: str, data_config: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from utils.io_tools import load_config_from_yaml
    from data_utils.dataset import DataConverter, CMambaDataset
    from data_transforms_helpers import build_transform  # project-specific helper
    import torch.utils.data as td

    # Load model
    from pl_modules.hybrid_ssm_trader_module import HybridSSMTraderModule
    module = HybridSSMTraderModule.load_from_checkpoint(checkpoint, map_location="cpu")
    model: HybridSSMTrader = module.model
    model.eval()

    # Register hook
    captured: list[torch.Tensor] = []
    hook_handle = model.fusion.register_forward_hook(
        _hook_gate_weights(model.fusion, captured)
    )

    # Load test data
    data_cfg_path = PROJECT_ROOT / "configs" / "data_configs" / f"{data_config}.yaml"
    data_cfg = load_config_from_yaml(str(data_cfg_path))
    dc = DataConverter(data_cfg)
    splits = dc.get_splits()
    transform = build_transform(data_cfg)
    test_ds = CMambaDataset(splits["test"], split="test",
                             window_size=module.hparams.window_size,
                             transform=transform)
    loader = td.DataLoader(test_ds, batch_size=64, shuffle=False)

    all_weights: list[np.ndarray] = []
    all_volatility: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"] if isinstance(batch, dict) else batch[0]
            _ = model(x)
            if captured:
                w = captured.pop().numpy()       # (B, 3)
                all_weights.append(w)
                # Approximate volatility: std of close returns in window
                close_col = 3
                close = x[:, close_col, :].numpy()
                ret = np.diff(np.log(np.clip(close, 1e-8, None)), axis=1)
                vol = ret.std(axis=1)
                all_volatility.append(vol)

    hook_handle.remove()

    weights = np.concatenate(all_weights, axis=0)    # (N, 3)
    volatility = np.concatenate(all_volatility, axis=0)  # (N,)

    df = pd.DataFrame({
        "volatility": volatility,
        "w_tcn": weights[:, 0],
        "w_mamba": weights[:, 1],
        "w_attn": weights[:, 2],
    })
    csv_path = out_path / "gate_weights.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples to {csv_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        quartiles = pd.qcut(df["volatility"], 4, labels=["Q1\n(low vol)", "Q2", "Q3", "Q4\n(high vol)"])
        grouped = df.assign(vol_quartile=quartiles).groupby("vol_quartile")[
            ["w_tcn", "w_mamba", "w_attn"]
        ].mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(4)
        width = 0.25
        ax.bar(x - width, grouped["w_tcn"],   width, label="TCN (local)",       color="#4C72B0")
        ax.bar(x,          grouped["w_mamba"], width, label="Mamba (long-range)", color="#DD8452")
        ax.bar(x + width, grouped["w_attn"],  width, label="Attention (global)", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.set_ylabel("Mean gate weight")
        ax.set_title("HybridSSM-Trader gate weights by volatility quartile")
        ax.legend()
        ax.set_ylim(0, 0.6)
        fig.tight_layout()
        fig_path = out_path / "gate_weights_by_volatility.pdf"
        fig.savefig(fig_path)
        print(f"Saved figure to {fig_path}")
    except ImportError:
        print("matplotlib not available — skipping figure generation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="btc_1d_paper")
    parser.add_argument("--out-dir",
                        default="experiments/paper_hybrid_ssm_trader/figures/gate_analysis")
    args = parser.parse_args()
    run_analysis(args.checkpoint, args.data_config, args.out_dir)


if __name__ == "__main__":
    main()
