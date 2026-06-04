import os
import pathlib
import subprocess
import sys
import warnings
from argparse import ArgumentParser

import yaml

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from data_utils.data_transforms import DataTransform
from pl_modules.data_module import CMambaDataModule
from utils import io_tools
from utils.runtime import (
    TensorBoardLogger,
    pl,
    prepare_run_paths,
    resolve_resume_checkpoint,
    resolve_trainer_strategy,
    save_run_info,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


ROOT = io_tools.get_root(__file__, num_returns=2)


def infer_num_features(config):
    return 5 + int(bool(config.get("use_volume"))) + len(config.get("additional_features", []))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, help="Legacy alias for --default_root_dir.")
    parser.add_argument("--default_root_dir", type=str, default=None, help="Root directory for this run.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to store checkpoints.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to store downstream outputs.")
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
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to a training YAML config. If set, overrides --config file lookup.",
    )
    parser.add_argument("--logger_type", default="tb", type=str, help="Logger type.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--save_checkpoints", default=False, action="store_true")
    parser.add_argument("--use_volume", default=False, action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--resume_mode",
        choices={"off", "auto", "path"},
        default="off",
        help="How to resume training. Use 'auto' to resume from last.ckpt in checkpoint_dir.",
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        type=str,
        help="Lightning strategy. 'auto' enables DDP only when devices > 1.",
    )
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Enable early stopping when >0 (patience in validation checks).",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum change in monitored metric to qualify as an improvement.",
    )
    parser.add_argument(
        "--early_stopping_monitor",
        type=str,
        default="val/rmse",
        help="Metric used by early stopping.",
    )
    parser.add_argument(
        "--early_stopping_mode",
        choices={"min", "max"},
        default="min",
        help="Optimization direction for early stopping monitor.",
    )
    # Weights & Biases (when --logger_type wandb)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MambaTCNRegNet",
        help="W&B project name (default: MambaTCNRegNet).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/user). Can also set WANDB_ENTITY env.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional W&B run group (e.g. config name for grouping).",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Optional comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        default=False,
        help="Run W&B in offline mode (no sync to cloud).",
    )
    parser.add_argument(
        "--wandb_log_model",
        action="store_true",
        default=False,
        help="Upload best/last checkpoints to W&B (can be slow for large models).",
    )
    # Negative-control experiments (Group E)
    parser.add_argument(
        "--negcontrol_mode",
        type=str,
        default=None,
        choices=["shuffled_labels", "lag_mismatch"],
        help=(
            "Corrupt training targets for negative-control validation. "
            "'shuffled_labels' randomly permutes targets; "
            "'lag_mismatch' shifts targets forward by one step."
        ),
    )
    parser.add_argument(
        "--negcontrol_seed",
        type=int,
        default=0,
        help="Seed for the negcontrol permutation (shuffled_labels mode).",
    )
    return parser.parse_args()


def load_model(config, logger_type):
    arch_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/models/archs.yaml")
    model_arch = config.get("model")
    model_config_path = f"{ROOT}/configs/models/{arch_config.get(model_arch)}"
    model_config = io_tools.load_config_from_yaml(model_config_path)

    normalize = model_config.get("normalize", False)
    hyperparams = config.get("hyperparams")
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get("params")[key] = hyperparams.get(key)

    model_config.get("params")["num_features"] = infer_num_features(config)
    model_config.get("params")["logger_type"] = logger_type
    model = io_tools.instantiate_from_config(model_config)
    model.train()
    return model, normalize


def resolve_paths(args):
    default_root_dir = args.default_root_dir or args.logdir or f"{ROOT}/runs/{args.config}"
    return prepare_run_paths(
        default_root_dir=default_root_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
    )


def get_git_commit_info() -> tuple[str | None, bool]:
    """Return (commit_hash, is_dirty). Returns (None, False) if not a git repo or git unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=pathlib.Path(__file__).resolve().parent.parent,
        )
        if result.returncode != 0:
            return None, False
        commit = result.stdout.strip() if result.stdout else None
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=pathlib.Path(__file__).resolve().parent.parent,
        )
        is_dirty = bool(dirty_result.stdout.strip()) if dirty_result.returncode == 0 else False
        return commit, is_dirty
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, False


if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    config_source = args.config_path or f"{ROOT}/configs/training/{args.config}.yaml"
    config = io_tools.load_config_from_yaml(config_source)
    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

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

    model, normalize = load_model(config, args.logger_type)
    run_paths = resolve_paths(args)

    combined_args = vars(args).copy()
    combined_args.update(config)

    name = config.get("name", args.expname)
    if args.logger_type == "tb":
        logger = TensorBoardLogger(save_dir=str(run_paths["logs_dir"]), name=name)
        logger.log_hyperparams(combined_args)
    elif args.logger_type == "wandb":
        wandb_config = dict(combined_args)
        wandb_config["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        git_commit, git_dirty = get_git_commit_info()
        wandb_config["git_commit"] = git_commit if git_commit else "n/a"
        wandb_config["git_dirty"] = git_dirty
        tags = [args.config, config.get("model", "")]
        if args.wandb_tags:
            tags.extend([token.strip() for token in args.wandb_tags.split(",") if token.strip()])
        tags = [t for t in tags if t]
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=name,
            save_dir=str(run_paths["run_dir"]),
            config=wandb_config,
            entity=entity,
            group=args.wandb_group or None,
            tags=tags if tags else None,
            offline=args.wandb_offline,
            log_model=("all" if args.wandb_log_model else False),
        )
    else:
        raise ValueError("Unknown logger type.")

    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        distributed_sampler=args.devices > 1,
        num_workers=args.num_workers,
        normalize=normalize,
        window_size=model.window_size,
        negcontrol_mode=getattr(args, "negcontrol_mode", None),
        negcontrol_seed=getattr(args, "negcontrol_seed", 0),
    )

    callbacks = []
    checkpoint_callback = None
    if args.save_checkpoints:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(run_paths["checkpoint_dir"]),
            save_top_k=1,
            verbose=True,
            monitor="val/rmse",
            mode="min",
            filename="best-epoch{epoch:03d}-val_rmse{val/rmse:.4f}",
            auto_insert_metric_name=False,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping_patience > 0:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=args.early_stopping_monitor,
                mode=args.early_stopping_mode,
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                verbose=True,
            )
        )

    max_epochs = config.get("max_epochs", args.max_epochs)
    model.set_normalization_coeffs(data_module.factors)

    resume_mode = args.resume_mode
    if args.resume_from_checkpoint and resume_mode == "off":
        resume_mode = "path"
    resume_checkpoint = resolve_resume_checkpoint(
        resume_mode=resume_mode,
        checkpoint_dir=run_paths["checkpoint_dir"],
        resume_path=args.resume_from_checkpoint,
    )

    save_run_info(
        run_paths["run_dir"],
        {
            "config_name": args.config,
            "config_source": config_source,
            "run_name": name,
            "run_dir": run_paths["run_dir"],
            "checkpoint_dir": run_paths["checkpoint_dir"],
            "results_dir": run_paths["results_dir"],
            "logger_type": args.logger_type,
            "resume_checkpoint": resume_checkpoint,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=max_epochs,
        enable_checkpointing=args.save_checkpoints,
        log_every_n_steps=10,
        logger=logger,
        callbacks=callbacks,
        strategy=resolve_trainer_strategy(args.strategy, args.devices),
        default_root_dir=str(run_paths["run_dir"]),
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint)

    best_checkpoint = None
    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_checkpoint = checkpoint_callback.best_model_path

    last_checkpoint = run_paths["checkpoint_dir"] / "last.ckpt"
    if not last_checkpoint.exists():
        last_checkpoint = None

    save_run_info(
        run_paths["run_dir"],
        {
            "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint,
            "checkpoint_dir": run_paths["checkpoint_dir"],
            "results_dir": run_paths["results_dir"],
        },
    )

    if args.save_checkpoints:
        test_checkpoint = best_checkpoint or (str(last_checkpoint) if last_checkpoint else None)
        if test_checkpoint is not None:
            trainer.test(model, datamodule=data_module, ckpt_path=test_checkpoint)
