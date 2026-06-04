"""
utils/metrics.py
────────────────
Central metrics library for Mamba-TCN-RegNet paper experiments.

Functions are pure NumPy so they work in evaluation scripts, Lightning
module epoch hooks, and offline analysis notebooks without any additional
dependencies beyond scipy and sklearn (both standard on Grove).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Regression / forecasting
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, current: np.ndarray) -> float:
    """Fraction of steps where the model predicts the correct direction of price move."""
    return float(np.mean(np.sign(y_true - current) == np.sign(y_pred - current)))


# ─────────────────────────────────────────────────────────────────────────────
# Information Coefficient (IC) and ICIR
# ─────────────────────────────────────────────────────────────────────────────

def information_coefficient(
    pred_returns: np.ndarray,
    actual_returns: np.ndarray,
) -> dict:
    """
    Rank IC (Spearman correlation) between predicted and realized returns.

    Returns
    -------
    dict with keys: ic, p_value
    """
    from scipy.stats import spearmanr
    pred_returns = np.asarray(pred_returns, dtype=np.float64)
    actual_returns = np.asarray(actual_returns, dtype=np.float64)
    mask = np.isfinite(pred_returns) & np.isfinite(actual_returns)
    if mask.sum() < 4:
        return {"ic": float("nan"), "p_value": float("nan")}
    ic, p_value = spearmanr(pred_returns[mask], actual_returns[mask])
    return {"ic": float(ic), "p_value": float(p_value)}


def rolling_ic(
    pred_returns: np.ndarray,
    actual_returns: np.ndarray,
    window: int = 63,
) -> np.ndarray:
    """
    Rolling IC over a sliding window (default ≈ 3 months of daily data).
    Returns an array of length len(pred_returns) with NaN for the first
    (window-1) entries.
    """
    from scipy.stats import spearmanr
    n = len(pred_returns)
    ic_series = np.full(n, np.nan)
    for i in range(window - 1, n):
        p = pred_returns[i - window + 1 : i + 1]
        a = actual_returns[i - window + 1 : i + 1]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask.sum() < 4:
            continue
        ic, _ = spearmanr(p[mask], a[mask])
        ic_series[i] = ic
    return ic_series


def icir(ic_series: np.ndarray, eps: float = 1e-8) -> float:
    """IC Information Ratio = mean(IC) / std(IC). Higher is more consistent."""
    valid = ic_series[np.isfinite(ic_series)]
    if len(valid) < 2:
        return float("nan")
    return float(np.mean(valid) / (np.std(valid, ddof=1) + eps))


def ic_metrics(
    pred_returns: np.ndarray,
    actual_returns: np.ndarray,
    rolling_window: int = 63,
) -> dict:
    """Full IC report: overall IC, p-value, ICIR, rolling IC series."""
    base = information_coefficient(pred_returns, actual_returns)
    ic_roll = rolling_ic(pred_returns, actual_returns, window=rolling_window)
    return {
        **base,
        "icir": icir(ic_roll),
        "rolling_ic_mean": float(np.nanmean(ic_roll)),
        "rolling_ic_std": float(np.nanstd(ic_roll)),
        "rolling_ic_positive_frac": float(np.nanmean(ic_roll > 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classification / action head
# ─────────────────────────────────────────────────────────────────────────────

def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = ("sell", "hold", "buy"),
) -> dict:
    """
    Per-class precision, recall, F1.  Also returns macro/weighted averages
    and a flattened confusion matrix.

    Requires sklearn (available on Grove).
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(class_names)
    labels = list(range(n))

    out: dict = {}
    for avg in ("macro", "weighted"):
        out[f"f1_{avg}"] = float(f1_score(y_true, y_pred, labels=labels, average=avg, zero_division=0))
        out[f"precision_{avg}"] = float(precision_score(y_true, y_pred, labels=labels, average=avg, zero_division=0))
        out[f"recall_{avg}"] = float(recall_score(y_true, y_pred, labels=labels, average=avg, zero_division=0))

    for i, name in enumerate(class_names):
        out[f"f1_{name}"] = float(f1_score(y_true, y_pred, labels=[i], average="binary" if n == 2 else None,
                                            pos_label=i if n == 2 else None, zero_division=0)
                                   if n == 2 else
                                   f1_score(y_true == i, y_pred == i, zero_division=0))
        out[f"precision_{name}"] = float(precision_score(y_true == i, y_pred == i, zero_division=0))
        out[f"recall_{name}"] = float(recall_score(y_true == i, y_pred == i, zero_division=0))
        out[f"support_{name}"] = int(np.sum(y_true == i))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Expected Calibration Error (ECE) for a single class.

    probs  : (N,) predicted probability for the positive class
    labels : (N,) binary ground-truth labels
    Returns ECE scalar and reliability-diagram arrays (bin_confs, bin_accs, bin_counts).
    """
    probs = np.asarray(probs, dtype=np.float64).clip(0.0, 1.0)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = np.zeros(n_bins)
    bin_accs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for k in range(n_bins):
        mask = (probs >= bins[k]) & (probs < bins[k + 1])
        if k == n_bins - 1:
            mask |= probs == 1.0
        if mask.sum() == 0:
            continue
        bin_confs[k] = probs[mask].mean()
        bin_accs[k] = labels[mask].mean()
        bin_counts[k] = int(mask.sum())

    n_total = max(int(labels.shape[0]), 1)
    ece = float(np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / n_total)
    return {
        "ece": ece,
        "bin_confidences": bin_confs.tolist(),
        "bin_accuracies": bin_accs.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


def multiclass_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, ...] = ("sell", "hold", "buy"),
    n_bins: int = 10,
) -> dict:
    """One-vs-rest ECE for each class; also returns mean ECE."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    out: dict = {}
    eces = []
    for i, name in enumerate(class_names):
        cal = expected_calibration_error(probs[:, i], (labels == i).astype(float), n_bins)
        out[f"ece_{name}"] = cal["ece"]
        out[f"reliability_{name}"] = {k: v for k, v in cal.items() if k != "ece"}
        eces.append(cal["ece"])
    out["ece_mean"] = float(np.mean(eces))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Financial utility
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    annualization: int = 365,
) -> float:
    """Annualized Sharpe ratio from daily returns."""
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free / annualization
    std = excess.std(ddof=1)
    if std < 1e-12 or not np.isfinite(std):
        return float("nan")
    return float(np.sqrt(annualization) * excess.mean() / std)


def sortino_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    annualization: int = 365,
) -> float:
    """Annualized Sortino ratio (downside deviation denominator)."""
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free / annualization
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-12:
        return float("nan")
    return float(np.sqrt(annualization) * excess.mean() / downside_std)


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    prices = np.asarray(prices, dtype=np.float64)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / np.maximum(peak, 1e-8)
    return float(-drawdown.min())


def buy_and_hold_metrics(
    prices: np.ndarray,
    annualization: int = 365,
) -> dict:
    """
    Metrics for a passive buy-and-hold strategy.
    prices : daily close prices over the evaluation period
    """
    prices = np.asarray(prices, dtype=np.float64)
    daily_returns = np.diff(prices) / np.maximum(prices[:-1], 1e-8)
    total_return = float((prices[-1] - prices[0]) / max(prices[0], 1e-8))
    return {
        "bah_total_return": total_return,
        "bah_sharpe": sharpe_ratio(daily_returns, annualization=annualization),
        "bah_sortino": sortino_ratio(daily_returns, annualization=annualization),
        "bah_max_drawdown": max_drawdown(prices),
        "bah_annualized_return": float((1.0 + total_return) ** (annualization / max(len(prices) - 1, 1)) - 1.0),
    }


def persistence_metrics(
    prices: np.ndarray,
    annualization: int = 365,
) -> dict:
    """
    Metrics for the naive persistence (random-walk) forecast:
    tomorrow's price = today's price.
    """
    prices = np.asarray(prices, dtype=np.float64)
    targets = prices[1:]
    preds = prices[:-1]
    errors = preds - targets
    return {
        "persistence_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "persistence_mae": float(np.mean(np.abs(errors))),
        "persistence_directional_accuracy": 0.5,  # by definition
    }


# ─────────────────────────────────────────────────────────────────────────────
# Regime labelling
# ─────────────────────────────────────────────────────────────────────────────

def label_regimes(
    prices: np.ndarray,
    returns: Optional[np.ndarray] = None,
    trend_window: int = 20,
    bull_threshold: float = 0.02,
    bear_threshold: float = -0.02,
    vol_high_pct: float = 75.0,
    vol_low_pct: float = 25.0,
) -> dict[str, np.ndarray]:
    """
    Return boolean masks for each regime over an array of prices.

    Regimes:
        bull       – rolling_return_20d > bull_threshold
        bear       – rolling_return_20d < bear_threshold
        sideways   – |rolling_return_20d| <= thresholds
        high_vol   – realized_vol_20d > 75th percentile
        low_vol    – realized_vol_20d < 25th percentile
        mid_vol    – remainder

    All masks have the same length as `prices`.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    if returns is None:
        returns = np.concatenate([[0.0], np.diff(prices) / np.maximum(prices[:-1], 1e-8)])
    else:
        returns = np.asarray(returns, dtype=np.float64)

    # Rolling 20-day return
    rolling_ret = np.full(n, np.nan)
    for i in range(trend_window - 1, n):
        rolling_ret[i] = float(prices[i] / max(prices[i - trend_window + 1], 1e-8) - 1.0)

    # Rolling 20-day realized volatility (std of daily returns)
    rolling_vol = np.full(n, np.nan)
    for i in range(trend_window - 1, n):
        rolling_vol[i] = float(np.std(returns[i - trend_window + 1 : i + 1], ddof=1))

    valid = np.isfinite(rolling_ret) & np.isfinite(rolling_vol)
    vol_low_cut = np.nanpercentile(rolling_vol, vol_low_pct)
    vol_high_cut = np.nanpercentile(rolling_vol, vol_high_pct)

    masks = {
        "bull":     valid & (rolling_ret > bull_threshold),
        "bear":     valid & (rolling_ret < bear_threshold),
        "sideways": valid & (rolling_ret >= bear_threshold) & (rolling_ret <= bull_threshold),
        "high_vol": valid & (rolling_vol >= vol_high_cut),
        "low_vol":  valid & (rolling_vol <= vol_low_cut),
        "mid_vol":  valid & (rolling_vol > vol_low_cut) & (rolling_vol < vol_high_cut),
        "all":      np.ones(n, dtype=bool),
    }
    return masks
