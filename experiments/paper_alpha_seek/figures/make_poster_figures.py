"""Create poster-ready figures from paper AlphaSeek experiment artifacts.

The script uses only committed experiment summaries/logs:
- experiments/paper_alpha_seek/metrics/aggregate_metrics.json
- root-level Slurm training logs, when present
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = REPO_ROOT / "experiments/paper_alpha_seek/metrics/aggregate_metrics.json"
OUT_DIR = REPO_ROOT / "experiments/paper_alpha_seek/figures"

CONFIG_LABELS = {
    "paper_hybrid_1d": "Hybrid\nTCN+Mamba+Attn",
    "paper_hybrid_logret_cost_1d": "Hybrid\nlog-return+cost",
    "paper_mamba_1d": "Mamba",
    "paper_mamba_logret_cost_1d": "Mamba\nlog-return+cost",
    "paper_tcn_1d": "TCN",
    "paper_tcn_logret_cost_1d": "TCN\nlog-return+cost",
    "paper_attention_1d": "Attention",
    "paper_hybrid_no_action_1d": "Hybrid\nno action head",
    "paper_hybrid_no_features_1d": "Hybrid\nOHLCV only",
}

CONFIG_ORDER = [
    "paper_hybrid_1d",
    "paper_hybrid_logret_cost_1d",
    "paper_mamba_1d",
    "paper_mamba_logret_cost_1d",
    "paper_tcn_1d",
    "paper_tcn_logret_cost_1d",
    "paper_attention_1d",
    "paper_hybrid_no_action_1d",
    "paper_hybrid_no_features_1d",
]

LOGRET_COST_ORDER = [
    "paper_hybrid_logret_cost_1d",
    "paper_mamba_logret_cost_1d",
    "paper_tcn_logret_cost_1d",
]

LOGRET_COST_LABELS = {
    "paper_hybrid_logret_cost_1d": "Hybrid",
    "paper_mamba_logret_cost_1d": "Mamba",
    "paper_tcn_logret_cost_1d": "TCN",
}


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float
    count: int


def load_payload() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing aggregate metrics: {METRICS_PATH}")
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def load_runs() -> list[dict]:
    payload = load_payload()
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"No runs found in {METRICS_PATH}")
    return runs


def summarize(runs: Iterable[dict], split: str, metric: str) -> dict[str, Summary]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        if run.get("split") == split and metric in run:
            grouped[run["config"]].append(float(run[metric]))

    output: dict[str, Summary] = {}
    for config, values in grouped.items():
        output[config] = Summary(
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            count=len(values),
        )
    return output


def apply_poster_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_metric_bar(
    summary: dict[str, Summary],
    metric_label: str,
    title: str,
    output_name: str,
    reference_line: float | None = None,
    reference_label: str = "Reference",
    y_limits: tuple[float, float] | None = None,
) -> None:
    configs = [config for config in CONFIG_ORDER if config in summary]
    labels = [CONFIG_LABELS[config] for config in configs]
    means = [summary[config].mean for config in configs]
    stds = [summary[config].std for config in configs]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#2457A6", "#4E79A7", "#59A14F", "#F28E2B", "#8CD17D", "#B07AA1"]
    ax.bar(range(len(configs)), means, yerr=stds, capsize=4, color=colors[: len(configs)])
    if reference_line is not None:
        ax.axhline(reference_line, color="#333333", linestyle="--", linewidth=1.2, label=reference_label)
        ax.legend(frameon=False, loc="upper left")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    for index, value in enumerate(means):
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
        ax.text(
            index,
            value + offset,
            f"{value:.3f}" if value < 1 else f"{value:.0f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def save_poster_figure(fig, stem: str) -> None:
    """Save poster figures as high-resolution PNG and scalable SVG."""
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def values_for_configs(
    summary: dict[str, Summary],
    configs: list[str],
) -> tuple[list[str], list[float], list[float]]:
    labels = [LOGRET_COST_LABELS.get(config, CONFIG_LABELS.get(config, config)) for config in configs]
    means = [summary[config].mean for config in configs]
    stds = [summary[config].std for config in configs]
    return labels, means, stds


def save_poster_logret_cost_comparison(runs: list[dict]) -> None:
    rmse_summary = summarize(runs, split="test", metric="model_rmse")
    direction_summary = summarize(runs, split="test", metric="model_directional_accuracy")
    configs = [config for config in LOGRET_COST_ORDER if config in rmse_summary and config in direction_summary]
    if not configs:
        return

    labels, rmse_means, rmse_stds = values_for_configs(rmse_summary, configs)
    _, direction_means, direction_stds = values_for_configs(direction_summary, configs)
    colors = ["#2457A6", "#59A14F", "#F28E2B"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(labels, rmse_means, yerr=rmse_stds, capsize=5, color=colors[: len(configs)])
    axes[0].axhline(2184.73, color="#333333", linestyle="--", linewidth=1.2, label="Persistence")
    axes[0].set_ylabel("Frozen test RMSE")
    axes[0].set_title("Forecast Error")
    axes[0].set_ylim(2165, 2225)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].bar(labels, direction_means, yerr=direction_stds, capsize=5, color=colors[: len(configs)])
    axes[1].axhline(0.50, color="#333333", linestyle="--", linewidth=1.2, label="0.50 reference")
    axes[1].set_ylabel("Directional accuracy")
    axes[1].set_title("Directional Signal Quality")
    axes[1].set_ylim(0.45, 0.55)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")

    for ax, means in zip(axes, [rmse_means, direction_means]):
        for index, value in enumerate(means):
            label = f"{value:.0f}" if value > 1 else f"{value:.3f}"
            ax.text(index, value, label, ha="center", va="bottom", fontsize=9)

    fig.suptitle("Latest Log-Return + Cost-Aware Runs", y=1.03, fontsize=14)
    save_poster_figure(fig, "poster_logret_cost_forecast_direction")


def save_poster_action_behavior(runs: list[dict]) -> None:
    metrics = {
        "Action accuracy": summarize(runs, split="test", metric="threshold_action_action_accuracy"),
        "Turnover": summarize(runs, split="test", metric="threshold_action_turnover"),
    }
    configs = [
        config
        for config in LOGRET_COST_ORDER
        if all(config in summary for summary in metrics.values())
    ]
    if not configs:
        return

    labels = [LOGRET_COST_LABELS[config] for config in configs]
    x_positions = list(range(len(configs)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    accuracy = [metrics["Action accuracy"][config].mean for config in configs]
    accuracy_std = [metrics["Action accuracy"][config].std for config in configs]
    turnover = [metrics["Turnover"][config].mean for config in configs]
    turnover_std = [metrics["Turnover"][config].std for config in configs]

    ax.bar([x - width / 2 for x in x_positions], accuracy, width, yerr=accuracy_std, capsize=4, label="Action accuracy", color="#2457A6")
    ax.bar([x + width / 2 for x in x_positions], turnover, width, yerr=turnover_std, capsize=4, label="Turnover", color="#F28E2B")
    ax.axhline(0.50, color="#333333", linestyle="--", linewidth=1.0, label="0.50 reference")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Threshold-Derived Action Behaviour")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_poster_figure(fig, "poster_threshold_action_behavior")


def save_poster_volatility_direction(slice_rows: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    volatility_slices = ["vol_low", "vol_mid", "vol_high"]
    labels_by_slice = {"vol_low": "Low vol", "vol_mid": "Mid vol", "vol_high": "High vol"}
    for row in slice_rows:
        if row.get("split") != "test" or row.get("slice") not in volatility_slices:
            continue
        if row.get("config") in LOGRET_COST_ORDER and "directional_accuracy" in row:
            grouped[(row["config"], row["slice"])].append(float(row["directional_accuracy"]))
    configs = [
        config
        for config in LOGRET_COST_ORDER
        if all(grouped.get((config, slice_name)) for slice_name in volatility_slices)
    ]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.24
    x_positions = list(range(len(configs)))
    colors = {"vol_low": "#9ECAE1", "vol_mid": "#4292C6", "vol_high": "#084594"}
    for offset, slice_name in enumerate(volatility_slices):
        means = [statistics.mean(grouped[(config, slice_name)]) for config in configs]
        xs = [x + (offset - 1) * width for x in x_positions]
        ax.bar(xs, means, width=width, label=labels_by_slice[slice_name], color=colors[slice_name])

    ax.axhline(0.50, color="#333333", linestyle="--", linewidth=1.2, label="0.50 reference")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([LOGRET_COST_LABELS[config] for config in configs])
    ax.set_ylim(0.43, 0.55)
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Directional Accuracy by Volatility Regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_poster_figure(fig, "poster_volatility_direction")


def save_split_rmse(runs: list[dict]) -> None:
    splits = ["train", "val", "test"]
    configs = [
        config
        for config in [
            "paper_hybrid_1d",
            "paper_hybrid_logret_cost_1d",
            "paper_mamba_1d",
            "paper_mamba_logret_cost_1d",
            "paper_tcn_1d",
            "paper_tcn_logret_cost_1d",
            "paper_attention_1d",
        ]
        if any(run.get("config") == config for run in runs)
    ]
    summaries = {
        split: summarize(runs, split=split, metric="model_rmse")
        for split in splits
    }
    if not configs or any(config not in summaries["test"] for config in configs):
        return

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.18
    x_positions = list(range(len(configs)))
    colors = {"train": "#9ECAE1", "val": "#4292C6", "test": "#084594"}

    for offset, split in enumerate(splits):
        xs = [x + (offset - 1) * width for x in x_positions]
        means = [summaries[split][config].mean for config in configs]
        stds = [summaries[split][config].std for config in configs]
        ax.bar(xs, means, width=width, yerr=stds, capsize=3, color=colors[split], label=split.title())

    ax.set_xticks(x_positions)
    ax.set_xticklabels([CONFIG_LABELS[config] for config in configs])
    ax.set_ylabel("RMSE")
    ax.set_title("Generalization Across Train/Validation/Test Splits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmse_by_split.png", bbox_inches="tight")
    plt.close(fig)


def save_threshold_action_plot(runs: list[dict]) -> None:
    summary = summarize(runs, split="test", metric="threshold_action_action_accuracy")
    if not summary:
        return
    save_metric_bar(
        summary,
        metric_label="Test action accuracy after validation threshold selection",
        title="Validation-Selected Threshold Signal Quality",
        output_name="test_threshold_action_accuracy_by_model.png",
        reference_line=0.5,
        reference_label="0.50 reference",
        y_limits=(0.0, 1.0),
    )


def save_selected_threshold_plot(threshold_rows: list[dict]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in threshold_rows:
        if row.get("selected_threshold") is not None:
            grouped[row["config"]].append(float(row["selected_threshold"]))
    summary = {
        config: Summary(
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            count=len(values),
        )
        for config, values in grouped.items()
    }
    if not summary:
        return
    save_metric_bar(
        summary,
        metric_label="Validation-selected return threshold",
        title="Thresholds Selected Without Test Tuning",
        output_name="validation_selected_threshold_by_model.png",
    )


def save_volatility_slice_plot(slice_rows: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    volatility_slices = ["vol_low", "vol_mid", "vol_high"]
    configs = [
        config
        for config in CONFIG_ORDER
        if any(row.get("config") == config and row.get("split") == "test" for row in slice_rows)
    ]
    for row in slice_rows:
        if row.get("split") != "test" or row.get("slice") not in volatility_slices:
            continue
        if "directional_accuracy" in row:
            grouped[(row["config"], row["slice"])].append(float(row["directional_accuracy"]))
    if not configs or not grouped:
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    width = 0.22
    x_positions = list(range(len(configs)))
    colors = {"vol_low": "#9ECAE1", "vol_mid": "#4292C6", "vol_high": "#084594"}
    labels = {"vol_low": "Low vol", "vol_mid": "Mid vol", "vol_high": "High vol"}

    for offset, slice_name in enumerate(volatility_slices):
        xs = [x + (offset - 1) * width for x in x_positions]
        means = []
        stds = []
        for config in configs:
            values = grouped.get((config, slice_name), [])
            means.append(statistics.mean(values) if values else 0.0)
            stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        ax.bar(xs, means, width=width, yerr=stds, capsize=3, color=colors[slice_name], label=labels[slice_name])

    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.2, label="0.50 reference")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([CONFIG_LABELS.get(config, config) for config in configs])
    ax.set_ylabel("Test directional accuracy")
    ax.set_title("Directional Accuracy by Volatility Regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_directional_accuracy_by_volatility.png", bbox_inches="tight")
    plt.close(fig)


def parse_best_val_rmse(log_path: Path) -> list[tuple[int, float]]:
    pattern = re.compile(r"Epoch (\d+).*val/rmse.*best ([0-9.]+)")
    points: list[tuple[int, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(2))))
    return points


def save_convergence_plot() -> None:
    log_path = REPO_ROOT / "slurm-mambatcnregnet-train-paper_hybrid_no_features_1d-282833.out"
    if not log_path.exists():
        return
    points = parse_best_val_rmse(log_path)
    if not points:
        return

    epochs, best_rmse = zip(*points)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(epochs, best_rmse, marker="o", linewidth=2, color="#2457A6")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best validation RMSE")
    ax.set_title("Validation Convergence Example: Hybrid OHLCV-Only, Seed 37")
    ax.grid(alpha=0.25)
    ax.annotate(
        f"Early stop after epoch 53\nbest={min(best_rmse):.1f}",
        xy=(epochs[-1], best_rmse[-1]),
        xytext=(max(epochs) * 0.55, max(best_rmse) * 0.98),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_convergence_hybrid_no_features_seed37.png", bbox_inches="tight")
    plt.close(fig)


def add_box(ax, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", wrap=True)


def save_architecture_overview() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 0.03, 0.4, 0.13, 0.2, "OHLCV + engineered\nmarket features", "#DCEAF7")
    add_box(ax, 0.22, 0.68, 0.16, 0.18, "TCN\nlocal patterns", "#C7E9C0")
    add_box(ax, 0.22, 0.41, 0.16, 0.18, "Parallel Mamba\nlong-range state", "#C6DBEF")
    add_box(ax, 0.22, 0.14, 0.16, 0.18, "Multi-head attention\nsalient events", "#FDD0A2")
    add_box(ax, 0.48, 0.4, 0.16, 0.2, "Gated fusion\n+ residual state", "#E7D4E8")
    add_box(ax, 0.7, 0.4, 0.13, 0.2, "Adaptive\nsmoothing", "#F2F0A1")
    add_box(ax, 0.88, 0.57, 0.1, 0.16, "Price\nhead", "#FEE0D2")
    add_box(ax, 0.88, 0.27, 0.1, 0.16, "Sell / Hold /\nBuy head", "#FEE0D2")

    arrows = [
        ((0.16, 0.5), (0.22, 0.77)),
        ((0.16, 0.5), (0.22, 0.5)),
        ((0.16, 0.5), (0.22, 0.23)),
        ((0.38, 0.77), (0.48, 0.5)),
        ((0.38, 0.5), (0.48, 0.5)),
        ((0.38, 0.23), (0.48, 0.5)),
        ((0.64, 0.5), (0.7, 0.5)),
        ((0.83, 0.5), (0.88, 0.65)),
        ((0.83, 0.5), (0.88, 0.35)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 1.4})

    ax.set_title("Implemented Multi-Scale Market Representation Backbone", pad=8, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture_overview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_poster_style()
    payload = load_payload()
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in {METRICS_PATH}")

    save_metric_bar(
        summarize(runs, split="test", metric="model_rmse"),
        metric_label="Frozen test RMSE, mean +/- SD over 3 seeds",
        title="Short-Horizon BTC Forecast Error by Representation",
        output_name="test_rmse_by_model.png",
    )
    save_metric_bar(
        summarize(runs, split="test", metric="model_directional_accuracy"),
        metric_label="Frozen test directional accuracy",
        title="Directional Signal Quality by Representation",
        output_name="test_directional_accuracy_by_model.png",
        reference_line=0.5,
        reference_label="0.50 reference",
        y_limits=(0.45, 0.525),
    )
    save_split_rmse(runs)
    save_threshold_action_plot(runs)
    save_selected_threshold_plot(payload.get("threshold_selections", []))
    save_volatility_slice_plot(payload.get("slices", []))
    save_convergence_plot()
    save_architecture_overview()
    save_poster_logret_cost_comparison(runs)
    save_poster_action_behavior(runs)
    save_poster_volatility_direction(payload.get("slices", []))

    print(f"Wrote poster figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
