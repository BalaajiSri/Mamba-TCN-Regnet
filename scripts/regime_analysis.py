"""
scripts/regime_analysis.py
──────────────────────────
Reads all predictions.jsonl files under --run_root and produces a full
regime × model metrics table:

    Regimes : bull, bear, sideways, high_vol, mid_vol, low_vol
    Models  : every (config, seed) pair found under run_root

Output
    <output>               — JSON with full regime table
    <output_dir>/regime_table.csv  — flat CSV for paper tables
    <output_dir>/regime_plots/     — per-model regime bar charts

Usage:
    python3 scripts/regime_analysis.py \\
        --run_root runs/paper_hybrid_ssm_trader \\
        --output   experiments/paper_hybrid_ssm_trader/metrics/regime_metrics.json

WandB (optional):
    python3 scripts/regime_analysis.py ... --log_wandb --wandb_project MambaTCNRegNet
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from utils import io_tools
from utils.metrics import (
    buy_and_hold_metrics,
    ic_metrics,
    label_regimes,
    per_class_metrics,
    sharpe_ratio,
)

ROOT = io_tools.get_root(__file__, num_returns=2)

REGIME_NAMES = ["bull", "bear", "sideways", "high_vol", "mid_vol", "low_vol", "all"]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = ArgumentParser()
    p.add_argument("--run_root", default=f"{ROOT}/runs/paper_hybrid_ssm_trader")
    p.add_argument("--output",   default=f"{ROOT}/experiments/paper_hybrid_ssm_trader/metrics/regime_metrics.json")
    p.add_argument("--split",    default="test", choices=["train", "val", "test"])
    p.add_argument("--bull_threshold",   type=float, default=0.02)
    p.add_argument("--bear_threshold",   type=float, default=-0.02)
    p.add_argument("--vol_high_percentile", type=float, default=75.0)
    p.add_argument("--vol_low_percentile",  type=float, default=25.0)
    p.add_argument("--trend_window", type=int, default=20)
    p.add_argument("--action_threshold", type=float, default=0.002)
    p.add_argument("--transaction_cost", type=float, default=0.001)
    p.add_argument("--log_wandb",       action="store_true")
    p.add_argument("--wandb_project",   default="MambaTCNRegNet")
    p.add_argument("--wandb_run_name",  default="regime_analysis")
    p.add_argument("--wandb_offline",   action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def find_prediction_files(run_root: pathlib.Path, split: str) -> list[tuple[str, str, pathlib.Path]]:
    """Return list of (config_name, run_name, predictions_jsonl_path)."""
    found = []
    for config_dir in sorted(run_root.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            pred_path = run_dir / "results" / "predictions.jsonl"
            if pred_path.exists():
                found.append((config_name, run_dir.name, pred_path))
    return found


def load_predictions(path: pathlib.Path, split: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") == split:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def extract_seed(run_name: str) -> str:
    """Extract seed from run name like 'paper_hybrid_1d-seed23-20260601-120000'."""
    for part in run_name.split("-"):
        if part.startswith("seed"):
            return part
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Per-regime metric computation
# ─────────────────────────────────────────────────────────────────────────────

def action_labels(current: np.ndarray, future: np.ndarray, threshold: float, tc: float) -> np.ndarray:
    ret = (future - current) / np.maximum(np.abs(current), 1e-8)
    eff = threshold + tc
    labels = np.ones_like(ret, dtype=np.int64)
    labels[ret > eff] = 2
    labels[ret < -eff] = 0
    return labels


def metrics_for_mask(
    preds: np.ndarray,
    targets: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
    action_threshold: float,
    tc: float,
) -> dict:
    p, t, c = preds[mask], targets[mask], current[mask]
    if len(p) == 0:
        return {"n": 0}
    errors = p - t
    pred_ret = (p - c) / np.maximum(np.abs(c), 1e-8)
    true_ret = (t - c) / np.maximum(np.abs(c), 1e-8)
    dir_acc  = float(np.mean(np.sign(true_ret) == np.sign(pred_ret)))

    m: dict = {
        "n":                    int(len(p)),
        "rmse":                 float(np.sqrt(np.mean(errors ** 2))),
        "mae":                  float(np.mean(np.abs(errors))),
        "directional_accuracy": dir_acc,
    }

    # IC
    try:
        ic = ic_metrics(pred_ret, true_ret)
        m["ic"]   = ic["ic"]
        m["icir"] = ic["icir"]
    except Exception:
        pass

    # Per-class F1 / action metrics
    try:
        pred_actions = action_labels(c, p, action_threshold, tc)
        true_actions = action_labels(c, t, action_threshold, tc)
        cls = per_class_metrics(true_actions, pred_actions)
        m["f1_macro"]    = cls.get("f1_macro",   float("nan"))
        m["f1_weighted"] = cls.get("f1_weighted", float("nan"))
        m["f1_buy"]      = cls.get("f1_buy",      float("nan"))
        m["f1_sell"]     = cls.get("f1_sell",     float("nan"))
        m["f1_hold"]     = cls.get("f1_hold",     float("nan"))
        m["action_acc"]  = float(np.mean(true_actions == pred_actions))
        m["turnover"]    = float(np.mean(pred_actions != 1))
    except Exception:
        pass

    # Sharpe (model signal)
    try:
        signal = np.where(pred_ret > 0, true_ret, -true_ret)
        m["sharpe_signal"] = sharpe_ratio(signal)
    except Exception:
        pass

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    run_root   = pathlib.Path(args.run_root)
    output     = pathlib.Path(args.output)
    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = find_prediction_files(run_root, args.split)
    if not pred_files:
        print(f"No predictions.jsonl files found under {run_root}")
        return

    print(f"Found {len(pred_files)} prediction files.")

    # Per-model results
    all_results: dict = {}

    for config_name, run_name, pred_path in pred_files:
        seed = extract_seed(run_name)
        key  = f"{config_name}/{seed}"
        print(f"  Processing {key} …")

        df = load_predictions(pred_path, args.split)
        if df.empty:
            print(f"    No {args.split} rows — skipping.")
            continue

        preds   = df["prediction"].to_numpy(dtype=np.float64)
        targets = df["target"].to_numpy(dtype=np.float64)
        current = df["current_price"].to_numpy(dtype=np.float64)

        # Build the full price series for regime labelling
        prices  = np.concatenate([current[:1], targets])

        masks = label_regimes(
            prices=targets,  # use target price as the "current" state index
            trend_window=args.trend_window,
            bull_threshold=args.bull_threshold,
            bear_threshold=args.bear_threshold,
            vol_high_pct=args.vol_high_percentile,
            vol_low_pct=args.vol_low_percentile,
        )

        regime_metrics: dict = {}
        for regime_name in REGIME_NAMES:
            mask = masks.get(regime_name, np.ones(len(preds), dtype=bool))
            if regime_name == "all":
                mask = np.ones(len(preds), dtype=bool)
            regime_metrics[regime_name] = metrics_for_mask(
                preds, targets, current, mask,
                args.action_threshold, args.transaction_cost,
            )

        all_results[key] = {
            "config": config_name,
            "seed":   seed,
            "split":  args.split,
            "regimes": regime_metrics,
        }

    # ── Aggregate across seeds ────────────────────────────────────────────────
    aggregated: dict = defaultdict(lambda: defaultdict(list))
    for key, res in all_results.items():
        config = res["config"]
        for regime, m in res["regimes"].items():
            for metric, val in m.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    aggregated[config][f"{regime}/{metric}"].append(val)

    agg_summary: dict = {}
    for config, metrics in aggregated.items():
        agg_summary[config] = {}
        for mk, vals in metrics.items():
            vals_arr = np.array([v for v in vals if np.isfinite(v)])
            if len(vals_arr) == 0:
                continue
            agg_summary[config][mk] = {
                "mean": float(vals_arr.mean()),
                "std":  float(vals_arr.std(ddof=1)) if len(vals_arr) > 1 else 0.0,
                "n_seeds": len(vals_arr),
            }

    full_output = {
        "split":       args.split,
        "regimes":     REGIME_NAMES,
        "per_run":     all_results,
        "aggregated":  agg_summary,
    }

    with open(output, "w") as f:
        json.dump(full_output, f, indent=2, sort_keys=True)
    print(f"\nRegime metrics → {output}")

    # ── Flat CSV for paper tables ─────────────────────────────────────────────
    rows = []
    for config, metrics in agg_summary.items():
        for mk, stats in metrics.items():
            regime, metric = mk.split("/", 1)
            rows.append({
                "config": config,
                "regime": regime,
                "metric": metric,
                "mean":   stats["mean"],
                "std":    stats["std"],
                "n_seeds": stats["n_seeds"],
            })
    csv_path = output_dir / "regime_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Regime CSV     → {csv_path}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    if args.log_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=["regime_analysis", "group_d"],
            config=vars(args),
            mode="offline" if args.wandb_offline else "online",
        )
        # Log regime × metric table per config
        for config, metrics in agg_summary.items():
            for mk, stats in metrics.items():
                wandb.log({f"regime/{config}/{mk}": stats["mean"]})

        # Upload JSON + CSV as artefacts
        art = wandb.Artifact("regime_analysis", type="analysis")
        art.add_file(str(output))
        art.add_file(str(csv_path))
        wandb.log_artifact(art)

        # WandB table
        table_data = [[row["config"], row["regime"], row["metric"],
                       round(row["mean"], 5), round(row["std"], 5)]
                      for row in rows]
        wandb.log({"regime_table": wandb.Table(
            columns=["config", "regime", "metric", "mean", "std"],
            data=table_data,
        )})
        wandb.finish()
        print("WandB run finished.")


if __name__ == "__main__":
    main()
