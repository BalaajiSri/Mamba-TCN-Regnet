import argparse
import os
import pathlib
import subprocess
import sys
from datetime import datetime

import wandb
import yaml

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from utils import io_tools


ROOT = io_tools.get_root(__file__, num_returns=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Base training config name.")
    parser.add_argument("--run_root", default=None, type=str, help="Root dir for sweep run outputs.")
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--save_checkpoints", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="MambaTCNRegNet", type=str)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_group", default=None, type=str)
    return parser.parse_args()


def resolve_run_root(run_root_arg: str | None) -> pathlib.Path:
    if run_root_arg:
        return pathlib.Path(run_root_arg).expanduser().resolve()
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return pathlib.Path(scratch).expanduser().resolve() / "mamba_tcn_regnet" / "sweeps"
    return pathlib.Path(ROOT) / "runs" / "sweeps"


def main():
    args = parse_args()
    base_config_path = pathlib.Path(f"{ROOT}/configs/training/{args.config}.yaml").resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {base_config_path}")

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group)
    if run is None:
        raise RuntimeError("wandb.init returned None; cannot continue sweep run.")

    run_root = resolve_run_root(args.run_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.config}-sweep-{timestamp}-{run.id}"
    run_dir = run_root / args.config / run_name
    checkpoint_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_config = io_tools.load_config_from_yaml(str(base_config_path))
    resolved_config = dict(base_config)
    resolved_hparams = dict(base_config.get("hyperparams", {}))
    sweep_cfg = dict(run.config)

    for key in ("lr", "weight_decay", "lr_step_size", "lr_gamma"):
        if key in sweep_cfg:
            resolved_hparams[key] = sweep_cfg[key]
    resolved_config["hyperparams"] = resolved_hparams

    if "max_epochs" in sweep_cfg:
        resolved_config["max_epochs"] = int(sweep_cfg["max_epochs"])

    resolved_config_path = run_dir / "sweep_resolved_training_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    batch_size = int(sweep_cfg.get("batch_size", 32))
    early_stopping_patience = int(sweep_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(sweep_cfg.get("early_stopping_min_delta", 0.0))

    train_cmd = [
        "python3",
        "scripts/training.py",
        "--config",
        args.config,
        "--config_path",
        str(resolved_config_path),
        "--logger_type",
        "wandb",
        "--wandb_project",
        args.wandb_project,
        "--accelerator",
        args.accelerator,
        "--devices",
        str(args.devices),
        "--num_workers",
        str(args.num_workers),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(args.seed),
        "--default_root_dir",
        str(run_dir),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--results_dir",
        str(results_dir),
        "--resume_mode",
        "off",
        "--early_stopping_patience",
        str(early_stopping_patience),
        "--early_stopping_min_delta",
        str(early_stopping_min_delta),
        "--early_stopping_monitor",
        "val/rmse",
        "--early_stopping_mode",
        "min",
    ]
    if args.save_checkpoints:
        train_cmd.append("--save_checkpoints")
    if args.wandb_entity:
        train_cmd.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_group:
        train_cmd.extend(["--wandb_group", args.wandb_group])

    env = os.environ.copy()
    env["WANDB_RUN_ID"] = run.id
    env["WANDB_RESUME"] = "allow"
    env["SWEEP_RUN_DIR"] = str(run_dir)
    env["SWEEP_CONFIG_PATH"] = str(resolved_config_path)

    result = subprocess.run(train_cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
