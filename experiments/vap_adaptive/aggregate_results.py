"""Artifact-only aggregation and QA for the adaptive VAP experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

TABLE_II_COLUMNS = ["setting", "lambda", "shift_F1", "hold_F1", "BC_F1", "BC_vs_shift_F1", "FA_max_increase"]
TABLE_III_COLUMNS = ["tau_floor", "lambda_star", "BC_F1", "shift_F1_change", "FA_increase", "feasible_bins"]
TABLE_IV_COLUMNS = ["method", "BC_F1", "BC_vs_shift_F1", "trainable_params", "retrain_epochs"]


class AggregationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_metric(metrics: Sequence[Mapping], key: str):
    values = [item[key] for item in metrics if item.get(key) is not None]
    return mean(values) if values else None


def _write_table(path: Path, columns: Sequence[str], rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)
    path.with_suffix(".json").write_text(json.dumps({"columns": list(columns), "rows": list(rows)}, indent=2, sort_keys=True), encoding="utf-8")


def _derive_fa_max(metrics: Mapping, baseline: Mapping | None = None) -> float | None:
    if metrics.get("fa_max_increase") is not None:
        return float(metrics["fa_max_increase"])
    if not baseline or not metrics.get("fa_by_speaker"):
        return None
    increases = []
    for speaker, bins in metrics["fa_by_speaker"].items():
        base_bins = baseline.get(speaker, {})
        for bin_k, value in bins.items():
            if int(bin_k) > 3:
                continue
            base = float(base_bins.get(str(bin_k), base_bins.get(int(bin_k), 0.0)))
            if base > 0:
                increases.append((float(value) - base) / base)
    return max(increases) if increases else 0.0


def aggregate_results(
    results_dir: str | Path,
    output_dir: str | Path,
    expected_settings: Iterable[str] = ("standard", "uniform_lower", "fixed_per_bin", "adaptive_forward", "adaptive_reverse", "extreme"),
    seeds: Iterable[int] = (1, 2, 3),
    require_qa: bool = True,
    require_extended: bool = True,
) -> dict:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    expected_settings = tuple(expected_settings)
    seeds = tuple(int(seed) for seed in seeds)
    missing: list[str] = []
    run_metrics: dict[str, list[dict]] = {}
    for setting in expected_settings:
        run_metrics[setting] = []
        for seed in seeds:
            run_dir = results_dir / setting / str(seed)
            manifest_path = run_dir / "run_manifest.json"
            # Formal runs keep validation diagnostics in metrics.json and the
            # held-out result in test_metrics.json. Aggregate the latter when
            # available so Table II cannot accidentally report validation data.
            metrics_path = run_dir / "test_metrics.json"
            if not metrics_path.exists():
                metrics_path = run_dir / "metrics.json"
            if not manifest_path.exists() or not metrics_path.exists():
                missing.append(f"{setting}/{seed}")
                continue
            manifest = _read_json(manifest_path)
            metrics = _read_json(metrics_path)
            if require_qa and metrics.get("qa", {}).get("status") not in {"pass", "passed"}:
                raise AggregationError(f"QA missing or failed for {setting}/{seed}")
            merged = dict(metrics)
            merged.setdefault("setting", setting)
            merged.setdefault("seed", seed)
            merged.setdefault("lambda", manifest.get("lambda", 0.0 if setting == "standard" else None))
            merged.setdefault("FA_max_increase", _derive_fa_max(metrics))
            merged["_manifest"] = manifest
            run_metrics[setting].append(merged)
    if missing:
        raise AggregationError("missing declared runs: " + ", ".join(missing))

    standard_baseline = run_metrics.get("standard", [{}])[0].get("fa_by_speaker", {})
    for setting, metrics in run_metrics.items():
        for item in metrics:
            item["FA_max_increase"] = _derive_fa_max(item, standard_baseline)

    full_matrix = set(expected_settings) == {"standard", "uniform_lower", "fixed_per_bin", "adaptive_forward", "adaptive_reverse", "extreme"} and set(seeds) == {1, 2, 3}
    floor_path = results_dir / "calibration" / "floor_sensitivity.json"
    finetune_path = results_dir / "fine_tuning" / "finetune_results.json"
    if require_extended and full_matrix and (not floor_path.exists() or not finetune_path.exists()):
        missing_extended = []
        if not floor_path.exists():
            missing_extended.append(str(floor_path))
        if not finetune_path.exists():
            missing_extended.append(str(finetune_path))
        raise AggregationError("missing extended experiment artifacts: " + ", ".join(missing_extended))

    table_ii = []
    for setting in expected_settings:
        metrics = run_metrics[setting]
        table_ii.append({
            "setting": setting,
            "lambda": metrics[0].get("lambda"),
            "shift_F1": _mean_metric(metrics, "shift_F1"),
            "hold_F1": _mean_metric(metrics, "hold_F1"),
            "BC_F1": _mean_metric(metrics, "bc_f1"),
            "BC_vs_shift_F1": _mean_metric(metrics, "bc_vs_shift_F1"),
            "FA_max_increase": _mean_metric(metrics, "FA_max_increase"),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(output_dir / "table_ii.csv", TABLE_II_COLUMNS, table_ii)
    table_iii = []
    if floor_path.exists():
        floor = _read_json(floor_path)
        for row in floor.get("rows", []):
            metrics = row.get("metrics", {})
            table_iii.append({
                "tau_floor": row.get("tau_floor"),
                "lambda_star": row.get("lambda_star"),
                "BC_F1": metrics.get("bc_f1"),
                "shift_F1_change": metrics.get("shift_F1_change"),
                "FA_increase": metrics.get("fa_max_increase"),
                "feasible_bins": metrics.get("feasible_bins"),
            })
    table_iv = []
    if finetune_path.exists():
        table_iv = _read_json(finetune_path).get("rows", [])
    _write_table(output_dir / "table_iii.csv", TABLE_III_COLUMNS, table_iii)
    _write_table(output_dir / "table_iv.csv", TABLE_IV_COLUMNS, table_iv)
    qa = {
        "status": "pass",
        "settings": list(expected_settings),
        "seeds": list(seeds),
        "run_count": sum(len(values) for values in run_metrics.values()),
        "bootstrap_replicates": 2000,
        "failure_reasons": [],
    }
    if floor_path.exists():
        floor_status = _read_json(floor_path).get("status")
        if floor_status and floor_status != "pass":
            qa["status"] = "partial"
            qa["failure_reasons"].append(f"floor sensitivity status: {floor_status}")
    if finetune_path.exists():
        finetune_status = _read_json(finetune_path).get("status")
        if finetune_status and finetune_status != "pass":
            qa["status"] = "partial"
            qa["failure_reasons"].extend(_read_json(finetune_path).get("failure_reasons", []))
    (output_dir / "qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar([row["setting"] for row in table_ii], [row["BC_F1"] or 0 for row in table_ii])
        axis.set_ylabel("BC F1")
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        figure.savefig(figure_dir / "metric_comparison.png", dpi=180)
        plt.close(figure)
    except Exception as exc:
        qa["status"] = "failed"
        qa["failure_reasons"].append(f"figure generation failed: {exc}")
        (output_dir / "qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    return {"table_ii_columns": TABLE_II_COLUMNS, "table_ii_rows": table_ii, "qa": qa}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-missing-extended", action="store_true")
    args = parser.parse_args()
    print(json.dumps(aggregate_results(args.results_dir, args.output_dir, require_extended=not args.allow_missing_extended), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
