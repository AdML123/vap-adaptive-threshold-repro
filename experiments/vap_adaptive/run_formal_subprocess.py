"""Run the formal head-only matrix in isolated child processes.

Each child owns one eager feature cache and exits before the next run, which
prevents Windows virtual-memory growth from accumulating across seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"step=(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _is_complete(run_dir: Path, epochs: int, minimum_steps: int = 0) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    qa_path = run_dir / "qa.json"
    if not manifest_path.exists() or not qa_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        checkpoint = Path(manifest["checkpoint"])
        completed_steps = int(manifest.get("completed_steps", _checkpoint_steps(checkpoint)))
        return (
            int(manifest.get("max_epochs", 0)) >= epochs
            and completed_steps >= minimum_steps
            and qa.get("status") == "pass"
            and checkpoint.exists()
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def expected_training_steps(config: dict, config_path: Path, epochs: int) -> int:
    cache_dir = Path(config["feature_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (config_path.parent / cache_dir).resolve()
    manifest = json.loads((cache_dir / "train" / "manifest.json").read_text(encoding="utf-8"))
    samples = int(manifest["samples"])
    batch_size = int(config.get("feature_cache_train_batch_size", 256))
    return math.ceil(samples / batch_size) * epochs


def training_command(
    python: str,
    repo_root: Path,
    config_path: Path,
    setting: str,
    seed: int,
    epochs: int,
    fixed: tuple[float, ...] | list[float],
    lam: float,
) -> list[str]:
    command = [
        python,
        str(repo_root / "experiments/vap_adaptive/train_experiment.py"),
        "--config",
        str(config_path),
        "--setting",
        setting,
        "--seed",
        str(seed),
        "--max-epochs",
        str(epochs),
        "--limit-train-batches",
        "1.0",
        "--limit-val-batches",
        "1.0",
    ]
    if setting == "fixed_per_bin":
        command.extend(["--fixed-thresholds", *[str(value) for value in fixed]])
    elif setting in {"adaptive_forward", "adaptive_reverse"}:
        command.extend(["--lambda", str(lam)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    selection_path = (repo_root / args.selection_json).resolve() if not Path(args.selection_json).is_absolute() else Path(args.selection_json).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "formal_subprocess_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fixed = selection["selected_fixed_thresholds"]
    lam = float(selection["selected_lambda"])
    minimum_steps = expected_training_steps(config, config_path, args.epochs)
    seeds = [int(seed) for seed in config.get("formal_seeds", (1, 2, 3))]
    jobs: list[tuple[str, int, list[str]]] = []
    for setting in ("standard", "uniform_lower", "fixed_per_bin", "adaptive_forward", "adaptive_reverse", "extreme"):
        for seed in seeds:
            command = training_command(sys.executable, repo_root, config_path, setting, seed, args.epochs, fixed, lam)
            jobs.append((setting, seed, command))

    for index, (setting, seed, command) in enumerate(jobs, start=1):
        run_dir = output_dir / setting / str(seed)
        if _is_complete(run_dir, args.epochs, minimum_steps=minimum_steps):
            print(json.dumps({"index": index, "total": len(jobs), "setting": setting, "seed": seed, "status": "skip_complete"}), flush=True)
            continue
        stdout_path = log_dir / f"{setting}_{seed}.stdout.log"
        stderr_path = log_dir / f"{setting}_{seed}.stderr.log"
        child_env = os.environ.copy()
        child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(json.dumps({"index": index, "total": len(jobs), "setting": setting, "seed": seed, "status": "start"}), flush=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=repo_root, env=child_env, stdout=stdout, stderr=stderr)
        if result.returncode != 0:
            raise SystemExit(f"formal run failed: {setting}/{seed}; see {stderr_path}")
        print(json.dumps({"index": index, "total": len(jobs), "setting": setting, "seed": seed, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
