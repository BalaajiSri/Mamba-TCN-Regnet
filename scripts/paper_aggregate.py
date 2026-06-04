import argparse
import json
import re
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_seed(run_dir: Path, fallback):
    match = re.search(r"seed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    return fallback


def select_latest_metric_paths(run_root: Path):
    """Keep only the latest completed run for each config/seed pair."""
    latest = {}
    for metrics_path in sorted(run_root.glob("**/results/metrics.json")):
        metrics = load_json(metrics_path)
        run_dir = metrics_path.parent.parent
        config = metrics.get("config", run_dir.parent.name)
        seed = parse_seed(run_dir, metrics.get("seed"))
        key = (config, seed)
        if key not in latest or run_dir.name > latest[key][0].name:
            latest[key] = (run_dir, metrics_path, metrics)
    return [latest[key] for key in sorted(latest)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, help="Root containing completed paper run directories.")
    parser.add_argument("--output", required=True, help="Aggregate JSON output path.")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    rows = []
    slice_rows = []
    threshold_rows = []
    metric_paths = select_latest_metric_paths(run_root)
    for run_dir, metrics_path, metrics in metric_paths:
        config = metrics.get("config", run_dir.parent.name)
        seed = parse_seed(run_dir, metrics.get("seed"))
        threshold_selection = metrics.get("threshold_selection", {})
        if threshold_selection:
            threshold_rows.append(
                {
                    "run_dir": str(run_dir),
                    "config": config,
                    "seed": seed,
                    "selected_threshold": threshold_selection.get("selected_threshold"),
                    "transaction_cost": threshold_selection.get("transaction_cost"),
                    "selection_metric": threshold_selection.get("selection_metric"),
                }
            )
        for split, split_metrics in metrics.get("splits", {}).items():
            model = split_metrics.get("model", {})
            persistence = split_metrics.get("persistence", {})
            threshold_action = split_metrics.get("threshold_action", {})
            row = {
                "run_dir": str(run_dir),
                "config": config,
                "seed": seed,
                "split": split,
                "num_examples": split_metrics.get("num_examples"),
            }
            for key, value in model.items():
                row[f"model_{key}"] = value
            for key, value in persistence.items():
                row[f"persistence_{key}"] = value
            for key, value in threshold_action.items():
                row[f"threshold_action_{key}"] = value
            if "rmse" in model and "rmse" in persistence:
                row["rmse_improvement_vs_persistence"] = persistence["rmse"] - model["rmse"]
            rows.append(row)
        for split, split_slices in metrics.get("slices", {}).items():
            for slice_name, slice_metrics in split_slices.items():
                row = {
                    "run_dir": str(run_dir),
                    "config": config,
                    "seed": seed,
                    "split": split,
                    "slice": slice_name,
                }
                row.update(slice_metrics)
                slice_rows.append(row)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": rows,
                "slices": slice_rows,
                "threshold_selections": threshold_rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote {len(rows)} rows from {len(metric_paths)} latest config/seed runs to {output}")


if __name__ == "__main__":
    main()
