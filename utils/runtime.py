from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies.ddp import DDPStrategy


RUN_INFO_NAME = "run_info.yaml"


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def prepare_run_paths(
    default_root_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Path]:
    run_dir = ensure_dir(default_root_dir)
    checkpoint_path = ensure_dir(checkpoint_dir or run_dir / "checkpoints")
    results_path = ensure_dir(results_dir or run_dir / "results")
    logs_path = ensure_dir(run_dir / "logs")
    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_path,
        "results_dir": results_path,
        "logs_dir": logs_path,
    }


def resolve_trainer_strategy(strategy: str | None, devices: int):
    if devices <= 1:
        return "auto"
    if strategy in (None, "auto", "ddp"):
        return DDPStrategy(find_unused_parameters=False)
    return strategy


def resolve_resume_checkpoint(
    resume_mode: str,
    checkpoint_dir: str | Path,
    resume_path: str | None = None,
) -> str | None:
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if resume_path:
        candidate = Path(resume_path).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"Requested resume checkpoint does not exist: {candidate}")
        return str(candidate)
    if resume_mode == "off":
        return None
    if resume_mode == "path":
        raise ValueError("--resume_mode=path requires --resume_from_checkpoint")
    if resume_mode == "auto":
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)
    return None


def resolve_inference_device(accelerator: str) -> torch.device:
    normalized = accelerator.lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU execution was requested, but CUDA is not available.")
        return torch.device("cuda:0")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS execution was requested, but MPS is not available.")
    raise ValueError(f"Unsupported accelerator '{accelerator}'.")


def parse_gres_gpu_count(gres: str | None) -> int | None:
    if not gres:
        return None
    parts = [part for part in gres.split(":") if part]
    if not parts:
        return None
    if parts[-1].isdigit():
        return int(parts[-1])
    if parts[0] == "gpu":
        return 1
    return None


def load_run_info(run_dir: str | Path) -> dict[str, Any]:
    info_path = Path(run_dir).expanduser().resolve() / RUN_INFO_NAME
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def save_run_info(run_dir: str | Path, info: dict[str, Any]) -> Path:
    target_dir = ensure_dir(run_dir)
    info_path = target_dir / RUN_INFO_NAME
    merged = load_run_info(target_dir)
    merged.update(_stringify_paths(info))
    with info_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=True)
    return info_path


def resolve_checkpoint_path(
    ckpt_path: str | None = None,
    run_dir: str | Path | None = None,
    preference: str = "best",
) -> Path:
    if ckpt_path:
        candidate = Path(ckpt_path).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"Checkpoint does not exist: {candidate}")
        return candidate

    if run_dir is None:
        raise ValueError("Either --ckpt_path or --run_dir must be provided.")

    run_path = Path(run_dir).expanduser().resolve()
    info = load_run_info(run_path)
    checkpoint_dir = Path(info.get("checkpoint_dir", run_path / "checkpoints")).expanduser().resolve()

    candidates: list[Path] = []
    best_from_info = info.get("best_checkpoint")
    last_from_info = info.get("last_checkpoint")
    best_candidates = []
    last_candidates = []

    if best_from_info:
        best_candidates.append(Path(best_from_info).expanduser().resolve())
    if last_from_info:
        last_candidates.append(Path(last_from_info).expanduser().resolve())

    if checkpoint_dir.exists():
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            last_candidates.append(last_ckpt)
        ranked_best = sorted(
            [path for path in checkpoint_dir.glob("*.ckpt") if path.name != "last.ckpt"],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        best_candidates.extend(ranked_best)

    if preference == "last":
        candidates.extend(last_candidates)
        candidates.extend(best_candidates)
    else:
        candidates.extend(best_candidates)
        candidates.extend(last_candidates)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    raise ValueError(
        f"Unable to resolve a checkpoint from run directory '{run_path}'. "
        "Expected metadata in run_info.yaml or checkpoint files in run_dir/checkpoints."
    )


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value
