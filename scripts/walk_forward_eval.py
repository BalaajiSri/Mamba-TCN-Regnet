"""
scripts/walk_forward_eval.py
────────────────────────────
Rolling-window evaluation of existing model predictions.

For each model in --run_root, loads its predictions.jsonl and reports
metrics over rolling 3-month (63-day) windows across the test period.
Produces:
    walk_forward_metrics.json  — per-model per-window metrics
    walk_forward_table.csv     — flat CSV for plotting
    walk_forward_plots/        — per-model rolling RMSE and IC charts

Unlike re-training, this just slices existing predictions — it runs in
seconds and requires no GPU.

Usage:
    python3 scripts/walk_forward_eval.py \\
        --run_root runs/paper_alpha_seek \\
        --output   experiments/paper_alpha_seek/metrics/walk_forward_metrics.json \\
        --window   63
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from argparse import ArgumentParser
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from utils import io_tools
from utils.metrics import directional_accuracy, ic_metrics, rmse, mae

ROOT = io_tools.get_root(__file__, num_returns=2)


def get_args():
    p = ArgumentParser()
    p.add_argument("--run_root", default=f"{ROOT}/runs/paper_alpha_seek")
    p.add_argument("--output",   default=f"{ROOT}/experiments/paper_alpha_seek/metrics/walk_forward_metrics.json")
    p.add_argument("--split",    default="test")
    p.add_argument("--window",   type=int, default=63,
                   help="Rolling window size in trading days (default 63 ≈ 3 months).")
    p.add_argument("--step",     type=int, default=21,
                   help="Step size between windows (default 21 ≈ 1 month).")
    p.add_argument("--log_wandb", action="store_true")
    p.add_argument("--wandb_project", default="MambaTCNRegNet")
    p.add_argument("--wandb_run_name", default="walk_forward_eval")
    p.add_argument("--wandb_offline",  action="store_true")
    return p.parse_args()


def find_prediction_files(run_root: pathlib.Path, split: str):
    for config_dir in sorted(run_root.iterdir()):
        if not config_dir.is_dir():
            continue
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            p = run_dir / "results" / "predictions.jsonl"
            if p.exists():
                yield config_dir.name, run_dir.name, p


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
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def rolling_metrics(df: pd.DataFrame, window: int, step: int) -> list[dict]:
    n = len(df)
    windows = []
    for start in range(0, n - window + 1, step):
        end = start + window
        chunk = df.iloc[start:end]
        preds   = chunk["prediction"].to_numpy(dtype=np.float64)
        targets = chunk["target"].to_numpy(dtype=np.float64)
        current = chunk["current_price"].to_numpy(dtype=np.float64)
        pred_ret = (preds   - current) / np.maximum(np.abs(current), 1e-8)
        true_ret = (targets - current) / np.maximum(np.abs(current), 1e-8)
        m = {
            "window_start":          chunk["timestamp"].iloc[0].isoformat(),
            "window_end":            chunk["timestamp"].iloc[-1].isoformat(),
            "n":                     len(chunk),
            "rmse":                  rmse(targets, preds),
            "mae":                   mae(targets, preds),
            "directional_accuracy":  directional_accuracy(targets, preds, current),
        }
        try:
            ic = ic_metrics(pred_ret, true_ret)
            m["ic"]   = ic["ic"]
            m["icir"] = ic["icir"]
        except Exception:
            pass
        windows.append(m)
    return windows


def plot_rolling(windows: list[dict], config: str, out_dir: pathlib.Path) -> None:
    if not windows:
        return
    dates = [datetime.fromisoformat(w["window_end"]) for w in windows]
    metrics_to_plot = [
        ("rmse",                 "Rolling RMSE"),
        ("directional_accuracy", "Rolling Directional Accuracy"),
        ("ic",                   "Rolling IC"),
    ]
    for key, title in metrics_to_plot:
        vals = [w.get(key) for w in windows]
        if all(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, vals, marker="o", linewidth=1.5, markersize=3)
        if key == "directional_accuracy":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance")
        if key == "ic":
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{config} – {title}")
        ax.set_xlabel("Window end date")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        safe_name = config.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"{safe_name}_{key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    args = get_args()
    run_root   = pathlib.Path(args.run_root)
    output     = pathlib.Path(args.output)
    output_dir = output.parent
    plots_dir  = output_dir / "walk_forward_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    all_results: dict = {}
    flat_rows: list = []

    for config_name, run_name, pred_path in find_prediction_files(run_root, args.split):
        seed = next((p for p in run_name.split("-") if p.startswith("seed")), "unknown")
        key  = f"{config_name}/{seed}"
        print(f"  {key} …", end=" ")
        df = load_predictions(pred_path, args.split)
        if df.empty:
            print("no data")
            continue
        windows = rolling_metrics(df, args.window, args.step)
        all_results[key] = {"config": config_name, "seed": seed, "windows": windows}
        for w in windows:
            flat_rows.append({"config": config_name, "seed": seed, **w})
        print(f"{len(windows)} windows")
        plot_rolling(windows, key, plots_dir)

    with open(output, "w") as f:
        json.dump({"window": args.window, "step": args.step, "runs": all_results}, f, indent=2)
    print(f"\nWalk-forward metrics → {output}")

    csv_path = output_dir / "walk_forward_table.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(f"Walk-forward CSV    → {csv_path}")

    if args.log_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=["walk_forward", "group_d"],
            config=vars(args),
            mode="offline" if args.wandb_offline else "online",
        )
        table_data = [
            [r["config"], r["seed"], r["window_start"], r["window_end"],
             r.get("rmse"), r.get("directional_accuracy"), r.get("ic")]
            for r in flat_rows
        ]
        wandb.log({"walk_forward_table": wandb.Table(
            columns=["config", "seed", "window_start", "window_end", "rmse", "dir_acc", "ic"],
            data=table_data,
        )})
        art = wandb.Artifact("walk_forward_eval", type="analysis")
        art.add_file(str(output))
        art.add_file(str(csv_path))
        wandb.log_artifact(art)
        wandb.finish()
        print("WandB run finished.")


if __name__ == "__main__":
    main()
