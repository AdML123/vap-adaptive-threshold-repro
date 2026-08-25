"""Run floor-sensitivity selection runs in isolated child processes."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--floors", default="0.05,0.15,0.20")
    parser.add_argument("--max-epochs", type=int, default=20)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    lam = float(selection["selected_lambda"])
    rows = []
    for floor_text in args.floors.split(","):
        floor = float(floor_text)
        name = f"floor_{floor:.2f}"
        output = Path(json.loads(config_path.read_text(encoding="utf-8"))["output_dir"])
        if not output.is_absolute():
            output = (config_path.parent / output).resolve()
        result_path = output / "calibration" / f"{name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        code = (
            "import json; "
            "from pathlib import Path; "
            "from experiments.vap_adaptive.gates import load_config; "
            "from experiments.vap_adaptive.train_experiment import run_setting; "
            "from experiments.vap_adaptive.evaluate_experiment import evaluate_checkpoint; "
            f"config,_=load_config(r'{config_path}'); "
            f"manifest=run_setting(config,'adaptive_forward',int(config.get('selection_seed',1)),max_epochs={args.max_epochs},lam={lam!r},tau_floor={floor!r},run_name={name!r}); "
            f"metrics=evaluate_checkpoint(config,manifest['checkpoint'],split='val',setting={name!r},output=r'{result_path.parent / name / '1' / 'metrics.json'}'); "
            f"Path(r'{result_path}').write_text(json.dumps({{'tau_floor':{floor!r},'lambda_star':{lam!r},'metrics':metrics,'manifest':manifest}},indent=2,sort_keys=True),encoding='utf-8')"
        )
        print(json.dumps({"floor": floor, "status": "start"}), flush=True)
        proc = subprocess.run([sys.executable, "-c", code], cwd=repo)
        if proc.returncode != 0:
            raise SystemExit(f"floor run failed: {floor}")
        rows.append(json.loads(result_path.read_text(encoding="utf-8")))
        gc.collect()
        print(json.dumps({"floor": floor, "status": "complete"}), flush=True)
    output = Path(json.loads(config_path.read_text(encoding="utf-8"))["output_dir"])
    if not output.is_absolute():
        output = (config_path.parent / output).resolve()
    final = output / "calibration" / "floor_sensitivity.json"
    rows.insert(1, {"tau_floor": 0.10, "reused_formal": True, "lambda_star": lam})
    final.write_text(json.dumps({"rows": rows, "selection_seed": 1, "reused_floor": 0.10}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
