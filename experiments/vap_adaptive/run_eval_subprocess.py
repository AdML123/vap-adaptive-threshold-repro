"""Evaluate formal checkpoints in isolated processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--force", action="store_true", help="recompute metrics even when a QA-passing artifact exists")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config_path = (repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "eval_subprocess_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    settings = ("standard", "uniform_lower", "fixed_per_bin", "adaptive_forward", "adaptive_reverse", "extreme")
    jobs = []
    for setting in settings:
        for seed in (1, 2, 3):
            run_dir = output_dir / setting / str(seed)
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                raise SystemExit(f"missing manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = Path(manifest["checkpoint"])
            metrics_path = run_dir / f"{args.split}_metrics.json"
            if metrics_path.exists() and not args.force:
                try:
                    if json.loads(metrics_path.read_text(encoding="utf-8")).get("qa", {}).get("status") == "pass":
                        print(json.dumps({"setting": setting, "seed": seed, "status": "skip_complete"}), flush=True)
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            jobs.append((setting, seed, checkpoint, metrics_path))

    for index, (setting, seed, checkpoint, metrics_path) in enumerate(jobs, 1):
        stdout_path = log_dir / f"{setting}_{seed}.stdout.log"
        stderr_path = log_dir / f"{setting}_{seed}.stderr.log"
        command = [sys.executable, str(repo / "experiments/vap_adaptive/evaluate_experiment.py"), "--config", str(config_path), "--evaluate", "--setting", setting, "--checkpoint", str(checkpoint), "--split", args.split, "--output", str(metrics_path)]
        print(json.dumps({"index": index, "total": len(jobs), "setting": setting, "seed": seed, "status": "start"}), flush=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr)
        if result.returncode != 0:
            raise SystemExit(f"evaluation failed: {setting}/{seed}; see {stderr_path}")
        print(json.dumps({"index": index, "total": len(jobs), "setting": setting, "seed": seed, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
