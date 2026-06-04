import json
import os
import pathlib
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from data_utils.data_transforms import DataTransform
from pl_modules.data_module import CMambaDataModule
from utils import io_tools
from utils.runtime import resolve_checkpoint_path, resolve_inference_device
from utils.trade import trade
from utils.actions import ACTION_TO_NAME

warnings.simplefilter(action="ignore", category=FutureWarning)

sns.set_theme(style="whitegrid", context="paper", font_scale=2)
palette = sns.color_palette("muted")

ROOT = io_tools.get_root(__file__, num_returns=2)


def infer_num_features(config):
    return 5 + int(bool(config.get("use_volume"))) + len(config.get("additional_features", []))

LABEL_DICT = {
    "hybrid_ssm_trader_hybrid_1d": "HybridSSM-Trader Hybrid",
    "hybrid_ssm_trader_tcn_1d": "TCN Baseline",
    "hybrid_ssm_trader_mamba_1d": "Mamba Baseline",
    "hybrid_ssm_trader_attention_1d": "Attention Baseline",
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="gpu", help="The type of accelerator.")
    parser.add_argument("--devices", type=int, default=1, help="Number of computing devices.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument(
        "--expname",
        type=str,
        default="HybridSSM-TraderHybrid",
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument("--config", type=str, default="hybrid_ssm_trader_hybrid_1d", help="Training config name.")
    parser.add_argument("--logger_type", default="tb", type=str, help="Logger type.")
    parser.add_argument("--ckpt_path", default=None, type=str, help="Checkpoint path.")
    parser.add_argument("--run_dir", default=None, type=str, help="Run directory used to auto-resolve checkpoints.")
    parser.add_argument("--results_dir", default=None, type=str, help="Directory to store plots.")
    parser.add_argument(
        "--checkpoint_preference",
        choices={"best", "last"},
        default="best",
        help="Checkpoint preference when resolving from --run_dir.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--balance", type=float, default=100, help="Initial balance.")
    parser.add_argument("--risk", type=float, default=2, help="Trading risk.")
    parser.add_argument("--split", type=str, default="test", choices={"test", "val", "train"})
    parser.add_argument(
        "--trade_mode",
        type=str,
        default="smart",
        choices={"smart", "smart_w_short", "vanilla", "no_strategy"},
    )
    parser.add_argument(
        "--decision_source",
        type=str,
        default="price",
        choices={"price", "action_head"},
        help="Use price heuristics or the model action head for trading decisions.",
    )
    return parser.parse_args()


def load_model(config, checkpoint_path, device, config_name=None):
    if checkpoint_path is None:
        checkpoint_path = f"{ROOT}/checkpoints/{config_name}.ckpt"
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


def init_dirs(args, name):
    if args.results_dir is not None:
        path = pathlib.Path(args.results_dir).expanduser().resolve()
    elif args.run_dir is not None:
        path = pathlib.Path(args.run_dir).expanduser().resolve() / "results"
    elif name == "all":
        path = pathlib.Path(f"{ROOT}/Results/all/").resolve()
    else:
        path = pathlib.Path(f"{ROOT}/Results/{name}/{args.config}").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def max_drawdown(prices):
    prices = np.array(prices, dtype=np.float64)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / np.maximum(peak, 1e-8)
    mdd = drawdown.min()
    return -mdd


def trading_summary(balance_in_time, initial_balance):
    equity = np.asarray(balance_in_time, dtype=np.float64)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-8)
    downside = returns[returns < 0]
    sharpe = 0.0 if returns.std() == 0 else float(np.sqrt(252) * returns.mean() / returns.std())
    sortino = 0.0 if downside.size == 0 or downside.std() == 0 else float(np.sqrt(252) * returns.mean() / downside.std())
    return {
        "initial_balance": float(initial_balance),
        "final_balance": float(equity[-1]),
        "total_return": float(equity[-1] / initial_balance - 1.0),
        "max_drawdown": float(max_drawdown(equity)),
        "sharpe": sharpe,
        "sortino": sortino,
        "num_steps": int(max(0, len(equity) - 1)),
    }


@torch.no_grad()
def run_model(model, dataloader, device, factors=None):
    target_list = []
    preds_list = []
    timetamps = []
    current_price_list = []
    action_list = []
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get("Timestamp").numpy().reshape(-1)
            target = batch.get(model.y_key).numpy().reshape(-1)
            current_price = batch.get(f"{model.y_key}_old").numpy().reshape(-1)
            features = batch.get("features").to(device)
            y_old = batch.get(f"{model.y_key}_old").to(device)
            preds = model(features, y_old).detach().cpu().numpy().reshape(-1)
            if hasattr(model, "predict_action_logits"):
                action_logits = model.predict_action_logits(features).detach().cpu().numpy()
                action_list += [int(x) for x in action_logits.argmax(axis=-1).tolist()]
            target_list += [float(x) for x in list(target)]
            current_price_list += [float(x) for x in list(current_price)]
            preds_list += [float(x) for x in list(preds)]
            if factors is not None:
                timetamps += [float(x) for x in list(batch.get("Timestamp_orig").numpy().reshape(-1))]
            else:
                timetamps += [float(x) for x in list(ts)]

    if factors is not None:
        scale = factors.get(model.y_key).get("max") - factors.get(model.y_key).get("min")
        shift = factors.get(model.y_key).get("min")
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]
        current_price_list = [x * scale + shift for x in current_price_list]

    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)
    current_prices = np.asarray(current_price_list)

    return timetamps, targets, preds, current_prices, action_list


if __name__ == "__main__":
    args = get_args()
    init_dir_flag = False
    colors = ["darkblue", "yellowgreen", "crimson", "darkviolet", "orange", "magenta"]
    device = resolve_inference_device(args.accelerator)

    if args.config == "all":
        config_list = [x.replace(".ckpt", "") for x in os.listdir(f"{ROOT}/checkpoints/") if "_nv.ckpt" in x]
    elif args.config == "all_v":
        config_list = [x.replace(".ckpt", "") for x in os.listdir(f"{ROOT}/checkpoints/") if "_v.ckpt" in x]
        init_dirs(args, "all")
    else:
        config_list = [args.config]
        colors = ["darkblue"]
        init_dir_flag = True

    plt.figure(figsize=(15, 10))
    results_path = None
    for conf, color in zip(config_list, colors):
        config = io_tools.load_config_from_yaml(f"{ROOT}/configs/training/{conf}.yaml")
        if init_dir_flag:
            init_dir_flag = False
            results_path = init_dirs(args, config.get("name", args.expname))
        data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

        checkpoint_path = resolve_checkpoint_path(
            ckpt_path=args.ckpt_path,
            run_dir=args.run_dir,
            preference=args.checkpoint_preference,
        )
        model, normalize = load_model(config, checkpoint_path, device, config_name=conf)

        use_volume = config.get("use_volume", False)
        test_transform = DataTransform(
            is_train=False,
            use_volume=use_volume,
            additional_features=config.get("additional_features", []),
        )
        data_module = CMambaDataModule(
            data_config,
            train_transform=test_transform,
            val_transform=test_transform,
            test_transform=test_transform,
            batch_size=args.batch_size,
            distributed_sampler=False,
            num_workers=args.num_workers,
            normalize=normalize,
            window_size=model.window_size,
        )

        if args.split == "test":
            test_loader = data_module.test_dataloader()
        if args.split == "val":
            test_loader = data_module.val_dataloader()
        if args.split == "train":
            test_loader = data_module.train_dataloader()

        factors = data_module.factors if normalize else None
        timstamps, targets, preds, current_prices, actions = run_model(model, test_loader, device, factors)

        data = test_loader.dataset.data.copy()
        time_key = "Timestamp"
        if normalize:
            time_key = "Timestamp_orig"
            scale = factors.get(model.y_key).get("max") - factors.get(model.y_key).get("min")
            shift = factors.get(model.y_key).get("min")
            data[model.y_key] = data[model.y_key] * scale + shift

        balance, balance_in_time = trade(
            data,
            time_key,
            timstamps,
            targets,
            preds,
            balance=args.balance,
            mode=args.trade_mode,
            risk=args.risk,
            y_key=model.y_key,
            step_seconds=data_config.get("jumps", 86400),
            actions=actions if args.decision_source == "action_head" else None,
            allow_short=args.trade_mode == "smart_w_short",
            current_prices=current_prices,
        )

        summary = trading_summary(balance_in_time, args.balance)
        summary.update({
            "config": conf,
            "split": args.split,
            "trade_mode": args.trade_mode,
            "decision_source": args.decision_source,
            "risk": float(args.risk),
            "final_balance": float(balance),
            "total_return": float(balance / args.balance - 1.0),
        })
        if args.decision_source == "action_head" and actions:
            summary["action_counts"] = {ACTION_TO_NAME[idx]: actions.count(idx) for idx in sorted(set(actions))}

        print(f"{conf} -- Final balance: {round(balance, 2)}")
        print(f"{conf} -- Maximum Draw Down : {round(summary['max_drawdown'] * 100, 2)}")
        if args.decision_source == "action_head" and actions:
            print(f"{conf} -- Action counts: {summary['action_counts']}")

        if results_path is not None:
            metrics_path = results_path / f"trading_metrics_{args.split}_{args.trade_mode}_{args.decision_source}.json"
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)

        label = LABEL_DICT.get(conf, conf)
        timestamps = [timstamps[0] - data_config.get("jumps", 86400)] + timstamps
        timestamps = [datetime.fromtimestamp(int(x)) for x in timestamps]
        sns.lineplot(
            x=timestamps,
            y=balance_in_time,
            color=color,
            zorder=0,
            linewidth=2.5,
            label=label,
        )

    name = config.get("name", args.expname)
    if args.trade_mode == "no_strategy":
        plot_path = pathlib.Path(f"./balance_{args.split}.jpg").resolve()
    else:
        if len(config_list) == 1:
            if results_path is None:
                results_path = init_dirs(args, name)
            plot_path = results_path / f"balance_{args.split}_{args.trade_mode}_{args.decision_source}.jpg"
        else:
            plot_path = init_dirs(args, "all") / f"balance_{args.config}_{args.split}_{args.trade_mode}_{args.decision_source}.jpg"
    plt.xticks(rotation=30)
    plt.axhline(y=100, color="r", linestyle="--")

    if len(config_list) == 1:
        ax = plt.gca()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        plt.title(f"Balance in time (final: {round(balance, 2)})")
    else:
        plt.title("Net Worth in Time")

    plt.xlim([timestamps[0], timestamps[-1]])
    plt.ylabel("Balance ($)")
    plt.xlabel("Date")
    plt.legend(loc="upper left")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
