import importlib
import importlib.util
import os
import pathlib
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from utils import io_tools


ROOT = io_tools.get_root(__file__, num_returns=2)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Training config name.")
    parser.add_argument("--accelerator", default="gpu", type=str, help="Requested accelerator.")
    parser.add_argument("--devices", default=1, type=int, help="Requested device count.")
    parser.add_argument("--gres", default=None, type=str, help="Requested Slurm GRES string.")
    return parser.parse_args()


def resolve_path(path_str):
    candidate = pathlib.Path(path_str)
    if candidate.is_absolute():
        return candidate.resolve()
    return (pathlib.Path(ROOT) / path_str).resolve()


def parse_gres_gpu_count(gres):
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


def require_module_spec(module_name, label):
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    except ImportError as exc:
        message = str(exc)
        hint = ""
        if "GLIBCXX_" in message and "libstdc++.so.6" in message:
            hint = (
                " This usually means Python loaded the system libstdc++ instead of the conda runtime. "
                "On Grove, ensure the active conda env has a recent libstdc++ package "
                "(for example, 'conda install -c conda-forge libstdcxx-ng') and that "
                "'${CONDA_PREFIX}/lib' is first on LD_LIBRARY_PATH."
            )
        raise RuntimeError(f"Failed to import-check '{label}' ({module_name}): {message}.{hint}") from exc
    if spec is None:
        raise RuntimeError(f"Missing required Python dependency '{label}' ({module_name}).")


def validate_imports(model_target):
    required = [
        ("torch", "PyTorch"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("yaml", "PyYAML"),
    ]
    for module_name, label in required:
        require_module_spec(module_name, label)

    lightning_missing = False
    try:
        require_module_spec("lightning.pytorch", "Lightning")
    except RuntimeError as exc:
        if str(exc).startswith("Missing required Python dependency"):
            lightning_missing = True
        else:
            raise

    if lightning_missing:
        try:
            require_module_spec("pytorch_lightning", "PyTorch Lightning")
        except RuntimeError as exc:
            if str(exc).startswith("Missing required Python dependency"):
                raise RuntimeError(
                    "Missing Lightning dependency. Install either 'lightning' or 'pytorch_lightning'."
                ) from exc
            raise

    module_name, _, _ = model_target.rpartition(".")
    require_module_spec(module_name, model_target)


if __name__ == "__main__":
    args = get_args()

    training_config_path = pathlib.Path(f"{ROOT}/configs/training/{args.config}.yaml").resolve()
    if not training_config_path.exists():
        raise SystemExit(f"Training config does not exist: {training_config_path}")

    training_config = io_tools.load_config_from_yaml(training_config_path)
    arch_config_path = pathlib.Path(f"{ROOT}/configs/models/archs.yaml").resolve()
    arch_config = io_tools.load_config_from_yaml(arch_config_path)

    model_name = training_config.get("model")
    model_relative_path = arch_config.get(model_name)
    if not model_relative_path:
        raise SystemExit(f"Model '{model_name}' is missing from {arch_config_path}")

    model_config_path = pathlib.Path(f"{ROOT}/configs/models/{model_relative_path}").resolve()
    if not model_config_path.exists():
        raise SystemExit(f"Model config does not exist: {model_config_path}")
    model_config = io_tools.load_config_from_yaml(model_config_path)

    data_config_name = training_config.get("data_config")
    data_config_path = pathlib.Path(f"{ROOT}/configs/data_configs/{data_config_name}.yaml").resolve()
    if not data_config_path.exists():
        raise SystemExit(f"Data config does not exist: {data_config_path}")
    data_config = io_tools.load_config_from_yaml(data_config_path)

    raw_data_path = resolve_path(data_config.get("data_path"))
    if not raw_data_path.exists():
        raise SystemExit(f"Raw data file does not exist: {raw_data_path}")

    requested_gpu_count = parse_gres_gpu_count(args.gres)
    if args.accelerator in {"gpu", "cuda"} and requested_gpu_count is not None and args.devices > requested_gpu_count:
        raise SystemExit(
            f"Requested devices ({args.devices}) exceed the GPU count implied by GRES '{args.gres}' ({requested_gpu_count})."
        )

    try:
        validate_imports(model_config.get("target"))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Preflight OK for config '{args.config}'.")
    print(f"Training config: {training_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Data config: {data_config_path}")
    print(f"Raw data: {raw_data_path}")
    if requested_gpu_count is not None:
        print(f"GRES '{args.gres}' resolves to {requested_gpu_count} GPU(s).")
