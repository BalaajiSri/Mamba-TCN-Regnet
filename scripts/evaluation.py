import json
import os
import pathlib
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from data_utils.data_transforms import DataTransform
from pl_modules.data_module import CMambaDataModule
from utils import io_tools
from utils.runtime import pl, resolve_checkpoint_path, resolve_inference_device

warnings.simplefilter(action="ignore", category=FutureWarning)

# Paper metrics library
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))
from utils.metrics import (
    ic_metrics,
    multiclass_calibration,
    per_class_metrics,
    buy_and_hold_metrics,
    persistence_metrics,
    sharpe_ratio,
    max_drawdown,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=3)

ROOT = io_tools.get_root(__file__, num_returns=2)


def infer_num_features(config):
    return 5 + int(bool(config.get("use_volume"))) + len(config.get("additional_features", []))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, help="Logging directory.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to store metrics and plots.")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory used to auto-resolve checkpoints.")
    parser.add_argument(
        "--checkpoint_preference",
        choices={"best", "last"},
        default="best",
        help="Checkpoint preference when resolving from --run_dir.",
    )
    parser.add_argument("--accelerator", type=str, default="gpu", help="The type of accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of computing devices.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument(
        "--expname",
        type=str,
        default="AlphaSeekHybrid",
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument("--config", type=str, default="alphaseek_hybrid_1d", help="Training config name.")
    parser.add_argument("--logger_type", default="tb", type=str, help="Logger type.")
    parser.add_argument("--use_volume", default=False, action="store_true")
    parser.add_argument("--ckpt_path", default=None, type=str, help="Checkpoint path.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--threshold_candidates",
        type=str,
        default="0.0,0.001,0.002,0.003,0.005,0.01",
        help="Comma-separated thresholds swept on validation predictions only.",
    )
    return parser.parse_args()


def print_and_write(file, txt, add_new_line=True):
    print(txt)
    if add_new_line:
        file.write(f"{txt}\n")
    else:
        file.write(txt)


def regression_metrics(targets, preds, current_prices=None):
    targets = np.asarray(targets, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    errors = preds - targets
    metrics = {
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": float(np.mean(np.abs(errors) / np.maximum(np.abs(targets), 1e-8))),
    }
    if current_prices is not None:
        current_prices = np.asarray(current_prices, dtype=np.float64)
        metrics["directional_accuracy"] = float(
            np.mean(np.sign(targets - current_prices) == np.sign(preds - current_prices))
        )
        # Persistence baseline
        persistence_errors = current_prices - targets
        metrics["persistence_rmse"] = float(np.sqrt(np.mean(persistence_errors ** 2)))
        metrics["persistence_mae"] = float(np.mean(np.abs(persistence_errors)))
        metrics["rmse_vs_persistence"] = metrics["rmse"] - metrics["persistence_rmse"]
        # IC / ICIR (return-space)
        try:
            pred_returns  = (preds  - current_prices) / np.maximum(np.abs(current_prices), 1e-8)
            true_returns  = (targets - current_prices) / np.maximum(np.abs(current_prices), 1e-8)
            ic = ic_metrics(pred_returns, true_returns)
            metrics.update({f"ic_{k}": v for k, v in ic.items()})
        except Exception:
            pass
        # Buy-and-hold baseline (requires price series, approximated from current_prices)
        try:
            bah = buy_and_hold_metrics(np.concatenate([current_prices[:1], targets]))
            metrics.update(bah)
        except Exception:
            pass
    return metrics


def parse_threshold_candidates(raw_value):
    if isinstance(raw_value, (list, tuple)):
        return [float(value) for value in raw_value]
    return [float(value.strip()) for value in raw_value.split(",") if value.strip()]


def action_labels(current_prices, future_prices, threshold, transaction_cost=0.0):
    current_prices = np.asarray(current_prices, dtype=np.float64)
    future_prices = np.asarray(future_prices, dtype=np.float64)
    returns = (future_prices - current_prices) / np.maximum(np.abs(current_prices), 1e-8)
    effective_threshold = float(threshold) + float(transaction_cost)
    labels = np.ones_like(returns, dtype=np.int64)
    labels[returns > effective_threshold] = 2
    labels[returns < -effective_threshold] = 0
    return labels


def action_metrics(targets, preds, current_prices, threshold, transaction_cost=0.0):
    true_actions = action_labels(current_prices, targets, threshold, transaction_cost)
    pred_actions = action_labels(current_prices, preds, threshold, transaction_cost)
    counts = np.bincount(pred_actions, minlength=3).astype(np.float64)
    fractions = counts / max(float(counts.sum()), 1.0)
    # Per-class F1 / precision / recall
    try:
        cls_m = per_class_metrics(true_actions, pred_actions)
    except Exception:
        cls_m = {}
    return {
        "threshold": float(threshold),
        "transaction_cost": float(transaction_cost),
        "action_accuracy": float(np.mean(true_actions == pred_actions)),
        "turnover": float(np.mean(pred_actions != 1)),
        "pred_sell_frac": float(fractions[0]),
        "pred_hold_frac": float(fractions[1]),
        "pred_buy_frac": float(fractions[2]),
        **{f"cls_{k}": v for k, v in cls_m.items()},
    }


def select_threshold(targets, preds, current_prices, candidates, transaction_cost=0.0):
    candidate_metrics = [
        action_metrics(targets, preds, current_prices, threshold, transaction_cost)
        for threshold in candidates
    ]
    best = max(
        candidate_metrics,
        key=lambda row: (
            row["action_accuracy"],
            -row["turnover"],
            -row["threshold"],
        ),
    )
    return {
        "selected_threshold": best["threshold"],
        "transaction_cost": float(transaction_cost),
        "selection_metric": "validation_action_accuracy",
        "candidates": candidate_metrics,
    }


def slice_metrics(timestamps, targets, preds, current_prices):
    targets = np.asarray(targets, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    current_prices = np.asarray(current_prices, dtype=np.float64)
    returns = (targets - current_prices) / np.maximum(np.abs(current_prices), 1e-8)
    abs_returns = np.abs(returns)
    slices = {}

    if len(targets) == 0:
        return slices

    low_cut, high_cut = np.quantile(abs_returns, [1 / 3, 2 / 3])
    masks = {
        "vol_low": abs_returns <= low_cut,
        "vol_mid": (abs_returns > low_cut) & (abs_returns <= high_cut),
        "vol_high": abs_returns > high_cut,
        "trend_down": returns < -0.001,
        "trend_flat": np.abs(returns) <= 0.001,
        "trend_up": returns > 0.001,
    }

    years = np.asarray([timestamp.year for timestamp in timestamps])
    for year in sorted(set(years.tolist())):
        masks[f"year_{year}"] = years == year

    for name, mask in masks.items():
        if not np.any(mask):
            continue
        metrics = regression_metrics(targets[mask], preds[mask], current_prices[mask])
        metrics["num_examples"] = int(np.sum(mask))
        slices[name] = metrics
    return slices


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def init_dirs(args, name):
    if args.results_dir is not None:
        path = pathlib.Path(args.results_dir).expanduser().resolve()
    elif args.run_dir is not None:
        path = pathlib.Path(args.run_dir).expanduser().resolve() / "results"
    else:
        path = pathlib.Path(f"{ROOT}/Results/{name}/{args.config}").resolve()
    path.mkdir(parents=True, exist_ok=True)
    txt_file = open(path / "metrics.txt", "w", encoding="utf-8")
    plot_path = path / "pred.jpg"
    return txt_file, plot_path


def load_model(config, checkpoint_path, device):
    arch_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/models/archs.yaml")
    model_arch = config.get("model")
    model_config_path = f"{ROOT}/configs/models/{arch_config.get(model_arch)}"
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get("normalize", False)
    model_config.get("params")["num_features"] = infer_num_features(config)
    model_class = io_tools.get_obj_from_str(model_config.get("target"))
    model = model_class.load_from_checkpoint(str(checkpoint_path), map_location=device, **model_config.get("params"))
    model.to(device)
    model.eval()
    return model, normalize


@torch.no_grad()
def run_model(model, dataloader, device, factors=None):
    target_list = []
    preds_list = []
    current_price_list = []
    timestamps = []
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get("Timestamp").numpy().reshape(-1)
            target = batch.get(model.y_key).numpy().reshape(-1)
            current_price = batch.get(f"{model.y_key}_old").numpy().reshape(-1)
            features = batch.get("features").to(device)
            y_old = batch.get(f"{model.y_key}_old").to(device)
            preds = model(features, y_old).detach().cpu().numpy().reshape(-1)
            target_list += [float(x) for x in list(target)]
            current_price_list += [float(x) for x in list(current_price)]
            preds_list += [float(x) for x in list(preds)]
            if factors is not None and "Timestamp_orig" in batch:
                timestamps += [float(x) for x in list(batch.get("Timestamp_orig").numpy().reshape(-1))]
            else:
                timestamps += [float(x) for x in list(ts)]

    if factors is not None:
        scale = factors.get(model.y_key).get("max") - factors.get(model.y_key).get("min")
        shift = factors.get(model.y_key).get("min")
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]
        current_price_list = [x * scale + shift for x in current_price_list]

    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)
    current_prices = np.asarray(current_price_list)
    timestamps = [datetime.fromtimestamp(int(x)) for x in timestamps]
    return timestamps, targets, preds, current_prices, regression_metrics(targets, preds, current_prices)


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    config = io_tools.load_config_from_yaml(f"{ROOT}/configs/training/{args.config}.yaml")
    name = config.get("name", args.expname)
    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

    checkpoint_path = resolve_checkpoint_path(
        ckpt_path=args.ckpt_path,
        run_dir=args.run_dir,
        preference=args.checkpoint_preference,
    )
    device = resolve_inference_device(args.accelerator)

    use_volume = args.use_volume or config.get("use_volume")
    train_transform = DataTransform(
        is_train=True,
        use_volume=use_volume,
        additional_features=config.get("additional_features", []),
    )
    val_transform = DataTransform(
        is_train=False,
        use_volume=use_volume,
        additional_features=config.get("additional_features", []),
    )
    test_transform = DataTransform(
        is_train=False,
        use_volume=use_volume,
        additional_features=config.get("additional_features", []),
    )

    model, normalize = load_model(config, checkpoint_path, device)
    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        distributed_sampler=False,
        num_workers=args.num_workers,
        normalize=normalize,
        window_size=model.window_size,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    dataloader_list = [train_loader, val_loader, test_loader]
    titles = ["Train", "Val", "Test"]
    colors = ["red", "green", "magenta"]

    factors = data_module.factors if normalize else None
    split_outputs = {}
    for key, dataloader in zip(titles, dataloader_list):
        timestamps, targets, preds, current_prices, metrics = run_model(model, dataloader, device, factors)
        split_outputs[key.lower()] = {
            "timestamps": timestamps,
            "targets": targets,
            "preds": preds,
            "current_prices": current_prices,
            "metrics": metrics,
        }

    threshold_candidates = parse_threshold_candidates(args.threshold_candidates)
    transaction_cost = float(getattr(model, "transaction_cost", 0.0))
    val_output = split_outputs["val"]
    threshold_selection = select_threshold(
        val_output["targets"],
        val_output["preds"],
        val_output["current_prices"],
        threshold_candidates,
        transaction_cost=transaction_cost,
    )
    selected_threshold = threshold_selection["selected_threshold"]

    metrics_file, plot_path = init_dirs(args, name)
    results_dir = pathlib.Path(metrics_file.name).parent
    metrics_json = {
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "data_config": config.get("data_config"),
        "seed": args.seed,
        "threshold_selection": threshold_selection,
        "splits": {},
        "slices": {},
    }
    prediction_path = results_dir / "predictions.jsonl"
    if prediction_path.exists():
        prediction_path.unlink()
    with open(results_dir / "threshold_selection.json", "w", encoding="utf-8") as handle:
        json.dump(threshold_selection, handle, indent=2, sort_keys=True)

    plt.figure(figsize=(20, 10))
    print_format = "{:^7} {:^15} {:^10} {:^7} {:^10} {:^9}"
    txt = print_format.format("Split", "MSE", "RMSE", "MAPE", "MAE", "DirAcc")
    print_and_write(metrics_file, txt)
    all_targets = []
    all_timestamps = []
    for key, color in zip(titles, colors):
        split_key = key.lower()
        output = split_outputs[split_key]
        timestamps = output["timestamps"]
        targets = output["targets"]
        preds = output["preds"]
        current_prices = output["current_prices"]
        metrics = output["metrics"]
        persistence = current_prices
        persistence_metrics = regression_metrics(targets, persistence, current_prices)
        threshold_metrics = action_metrics(
            targets,
            preds,
            current_prices,
            selected_threshold,
            transaction_cost=transaction_cost,
        )
        all_timestamps += timestamps
        all_targets += list(targets)
        txt = print_format.format(
            key,
            round(metrics["mse"], 3),
            round(metrics["rmse"], 3),
            round(metrics["mape"], 5),
            round(metrics["mae"], 3),
            round(metrics["directional_accuracy"], 4),
        )
        print_and_write(metrics_file, txt)
        metrics_json["splits"][split_key] = {
            "model": metrics,
            "persistence": persistence_metrics,
            "threshold_action": threshold_metrics,
            "num_examples": int(len(targets)),
        }
        metrics_json["slices"][split_key] = slice_metrics(timestamps, targets, preds, current_prices)
        append_jsonl(
            prediction_path,
            [
                {
                    "split": split_key,
                    "timestamp": ts.isoformat(),
                    "current_price": float(current),
                    "target": float(target),
                    "prediction": float(pred),
                    "persistence_prediction": float(base),
                    "selected_threshold": float(selected_threshold),
                }
                for ts, target, pred, current, base in zip(timestamps, targets, preds, current_prices, persistence)
            ],
        )
        sns.lineplot(x=timestamps, y=preds, color=color, linewidth=2.5, label=key)

    sns.lineplot(x=all_timestamps, y=all_targets, color="blue", zorder=0, linewidth=2.5, label="Target")
    plt.legend()
    plt.ylabel("Price ($)")
    plt.xlim([all_timestamps[0], all_timestamps[-1]])
    plt.xticks(rotation=30)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:,.0f}K".format(x / 1000)))
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    # ── Calibration per split ──────────────────────────────────────────────
    for split_key, output in split_outputs.items():
        if output is None:
            continue
        try:
            action_logits_np = np.array(output.get("action_logits", []), dtype=np.float64)
            action_targets_np = output.get("action_targets_np")
            if action_logits_np.ndim == 2 and action_targets_np is not None:
                import torch as _t
                probs_np = _t.from_numpy(action_logits_np).softmax(dim=-1).numpy()
                cal = multiclass_calibration(probs_np, np.array(action_targets_np, dtype=np.int64))
                if split_key in metrics_json["splits"]:
                    metrics_json["splits"][split_key]["calibration"] = cal
        except Exception:
            pass

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2, sort_keys=True)
    metrics_file.close()

    # ── WandB: upload metrics.json as artifact ──────────────────────────────
    try:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.log({"eval/metrics_summary": metrics_json.get("splits", {})})
            art = _wandb.Artifact(
                name=f"eval-{_wandb.run.name or _wandb.run.id}",
                type="evaluation",
                description="Evaluation metrics JSON",
            )
            art.add_file(str(results_dir / "metrics.json"))
            _wandb.log_artifact(art)
    except Exception:
        pass
