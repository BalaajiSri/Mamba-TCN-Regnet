"""
scripts/baselines_classical.py
─────────────────────────────
Classical forecasting baselines for the paper:
  1. MA-5     – 5-day simple moving average of close price
  2. ARIMA    – ARIMA(p,d,q) fitted on training close prices (auto-order or fixed)

Both baselines output metrics.json in the same format as scripts/evaluation.py
so they can be directly compared in paper tables and WandB runs.

Usage (CPU node, no GPU needed):
    python3 scripts/baselines_classical.py \\
        --data_config btc_1d_paper \\
        --results_dir runs/paper_hybrid_ssm_trader/classical_baselines \\
        --arima_order 2,1,2

    # or auto-select ARIMA order (requires pmdarima):
    python3 scripts/baselines_classical.py \\
        --data_config btc_1d_paper \\
        --results_dir runs/paper_hybrid_ssm_trader/classical_baselines \\
        --arima_auto
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))
warnings.simplefilter(action="ignore", category=FutureWarning)

from utils import io_tools
from utils.metrics import (
    buy_and_hold_metrics,
    ic_metrics,
    max_drawdown,
    per_class_metrics,
    persistence_metrics,
    sharpe_ratio,
)

ROOT = io_tools.get_root(__file__, num_returns=2)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = ArgumentParser(description="Classical baseline forecasts for the paper.")
    p.add_argument("--data_config", default="btc_1d_paper",
                   help="Name of the data config YAML under configs/data_configs/")
    p.add_argument("--results_dir", default=None,
                   help="Where to write metrics JSONs and prediction CSVs.")
    p.add_argument("--arima_order", default="2,1,2",
                   help="ARIMA (p,d,q) as comma-separated ints, e.g. '2,1,2'.")
    p.add_argument("--arima_auto", action="store_true",
                   help="Auto-select ARIMA order using pmdarima.auto_arima (slower).")
    p.add_argument("--ma_windows", default="5,10,20",
                   help="Comma-separated MA window sizes to evaluate.")
    p.add_argument("--action_threshold", type=float, default=0.002,
                   help="Return threshold for buy/hold/sell labelling.")
    p.add_argument("--transaction_cost", type=float, default=0.001)
    p.add_argument("--wandb_project", default="MambaTCNRegNet")
    p.add_argument("--wandb_run_name", default="classical_baselines")
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--log_wandb", action="store_true",
                   help="Upload results to W&B after computing metrics.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(data_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test DataFrames using the DataConverter."""
    from data_utils.dataset import DataConverter
    converter = DataConverter(data_config)
    train, val, test = converter.get_data()
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def ma_forecast(prices: np.ndarray, window: int) -> np.ndarray:
    """Predict tomorrow's price = mean(last `window` closes)."""
    preds = np.full_like(prices, np.nan)
    for i in range(window, len(prices)):
        preds[i] = prices[i - window : i].mean()
    return preds


def arima_forecast(
    train_prices: np.ndarray,
    forecast_prices: np.ndarray,
    order: tuple[int, int, int] = (2, 1, 2),
    auto: bool = False,
) -> np.ndarray:
    """
    Fit ARIMA on train_prices, then produce one-step-ahead rolling forecasts
    over forecast_prices using an expanding window.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings as _w

    if auto:
        try:
            import pmdarima as pm
            print("Auto-selecting ARIMA order with pmdarima …")
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                model_auto = pm.auto_arima(
                    train_prices, seasonal=False, stepwise=True,
                    information_criterion="aic", max_p=4, max_q=4,
                    suppress_warnings=True, error_action="ignore",
                )
            order = model_auto.order
            print(f"  Selected ARIMA{order}")
        except ImportError:
            print("pmdarima not available; falling back to fixed order ARIMA(2,1,2).")

    history = list(train_prices)
    preds = []
    for obs in forecast_prices:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            try:
                fit = ARIMA(history, order=order).fit()
                forecast = float(fit.forecast(steps=1)[0])
            except Exception:
                forecast = history[-1]   # fallback: persistence
        preds.append(forecast)
        history.append(float(obs))       # expand window

    return np.array(preds, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def action_labels(current: np.ndarray, future: np.ndarray, threshold: float, tc: float) -> np.ndarray:
    returns = (future - current) / np.maximum(np.abs(current), 1e-8)
    eff = threshold + tc
    labels = np.ones_like(returns, dtype=np.int64)
    labels[returns > eff] = 2
    labels[returns < -eff] = 0
    return labels


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    current: np.ndarray,
    name: str,
    action_threshold: float = 0.002,
    tc: float = 0.001,
) -> dict:
    valid = np.isfinite(preds) & np.isfinite(targets) & np.isfinite(current)
    preds, targets, current = preds[valid], targets[valid], current[valid]

    errors = preds - targets
    pred_returns  = (preds   - current) / np.maximum(np.abs(current), 1e-8)
    true_returns  = (targets - current) / np.maximum(np.abs(current), 1e-8)
    directional_acc = float(np.mean(np.sign(true_returns) == np.sign(pred_returns)))

    per = persistence_metrics(np.concatenate([current[:1], targets]))
    bah = buy_and_hold_metrics(np.concatenate([current[:1], targets]))
    try:
        ic = ic_metrics(pred_returns, true_returns)
    except Exception:
        ic = {}

    true_actions = action_labels(current, targets, action_threshold, tc)
    pred_actions = action_labels(current, preds,   action_threshold, tc)
    try:
        cls_m = per_class_metrics(true_actions, pred_actions)
    except Exception:
        cls_m = {}

    return {
        "baseline": name,
        "n": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae":  float(np.mean(np.abs(errors))),
        "mape": float(np.mean(np.abs(errors) / np.maximum(np.abs(targets), 1e-8))),
        "directional_accuracy": directional_acc,
        "rmse_vs_persistence": float(np.sqrt(np.mean(errors ** 2))) - per["persistence_rmse"],
        **per,
        **bah,
        **{f"ic_{k}": v for k, v in ic.items()},
        **{f"cls_{k}": v for k, v in cls_m.items() if not isinstance(v, list)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    data_config = io_tools.load_config_from_yaml(
        f"{ROOT}/configs/data_configs/{args.data_config}.yaml"
    )
    train_df, val_df, test_df = load_splits(data_config)

    results_dir = pathlib.Path(
        args.results_dir or f"{ROOT}/runs/paper_hybrid_ssm_trader/classical_baselines"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train_close = train_df["Close"].to_numpy(dtype=np.float64)
    val_close   = val_df["Close"].to_numpy(dtype=np.float64)
    test_close  = test_df["Close"].to_numpy(dtype=np.float64)

    # Full concatenated array for sliding-window baselines
    all_close   = np.concatenate([train_close, val_close, test_close])
    val_start   = len(train_close)
    test_start  = len(train_close) + len(val_close)

    arima_order = tuple(int(x) for x in args.arima_order.split(","))
    ma_windows  = [int(w) for w in args.ma_windows.split(",")]

    results: dict = {"baselines": []}

    # ── MA baselines ─────────────────────────────────────────────────────────
    for window in ma_windows:
        print(f"MA-{window} …")
        all_preds = ma_forecast(all_close, window)
        for split_name, start, close_arr in [
            ("val",  val_start,  val_close),
            ("test", test_start, test_close),
        ]:
            preds   = all_preds[start : start + len(close_arr)]
            targets = close_arr
            current = all_close[start - 1 : start + len(close_arr) - 1]
            m = evaluate_predictions(preds, targets, current,
                                     name=f"MA{window}",
                                     action_threshold=args.action_threshold,
                                     tc=args.transaction_cost)
            m["split"] = split_name
            results["baselines"].append(m)
            print(f"  {split_name:5s} RMSE={m['rmse']:.2f}  DirAcc={m['directional_accuracy']:.4f}  IC={m.get('ic_ic', float('nan')):.4f}")

    # ── ARIMA baseline ───────────────────────────────────────────────────────
    print(f"ARIMA{arima_order} on val …")
    val_preds = arima_forecast(train_close, val_close, order=arima_order, auto=args.arima_auto)
    m_val = evaluate_predictions(val_preds, val_close,
                                 np.concatenate([train_close[-1:], val_close[:-1]]),
                                 name=f"ARIMA{arima_order}", tc=args.transaction_cost)
    m_val["split"] = "val"
    results["baselines"].append(m_val)
    print(f"  val   RMSE={m_val['rmse']:.2f}  DirAcc={m_val['directional_accuracy']:.4f}")

    print(f"ARIMA{arima_order} on test (expanding from val) …")
    test_preds = arima_forecast(
        np.concatenate([train_close, val_close]), test_close,
        order=arima_order, auto=False,
    )
    m_test = evaluate_predictions(test_preds, test_close,
                                  np.concatenate([val_close[-1:], test_close[:-1]]),
                                  name=f"ARIMA{arima_order}", tc=args.transaction_cost)
    m_test["split"] = "test"
    results["baselines"].append(m_test)
    print(f"  test  RMSE={m_test['rmse']:.2f}  DirAcc={m_test['directional_accuracy']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = results_dir / "classical_baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults written to {out_path}")

    # ── WandB upload ─────────────────────────────────────────────────────────
    if args.log_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=["classical_baseline", "group_a"],
            config=vars(args),
            mode="offline" if args.wandb_offline else "online",
        )
        for row in results["baselines"]:
            clean = {k: v for k, v in row.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
            wandb.log({f"{row['baseline']}/{row['split']}/{k}": v
                       for k, v in clean.items()})
        art = wandb.Artifact("classical_baselines", type="evaluation")
        art.add_file(str(out_path))
        wandb.log_artifact(art)
        wandb.finish()
        print("WandB run finished.")


if __name__ == "__main__":
    main()
