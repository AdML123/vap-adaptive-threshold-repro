"""Metric and calibration contracts for adaptive VAP runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_limit(value: str | int | float):
    text = str(value)
    return int(text) if text.isdigit() else float(text)


def decode_state_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Decode 256-state logits into ``(..., speaker=2, bin=4)`` activity probabilities."""
    if logits.shape[-1] != 256:
        raise ValueError("VAP logits must have 256 states")
    probs = logits.softmax(dim=-1)
    indices = torch.arange(256, device=logits.device).view(-1, 1)
    bits = ((indices >> torch.arange(8, device=logits.device).view(1, -1)) & 1).float().view(256, 2, 4)
    return torch.einsum("...c,cxy->...xy", probs, bits)


def _non_bc_rows(rows: Iterable[Mapping]) -> list[Mapping]:
    return [row for row in rows if not row.get("has_bc", False)]


def bc_positive_windows(rows: Iterable[Mapping], bins: Sequence[int] = (1, 2, 3), threshold: float = 0.5) -> int:
    """Count listener-specific BC windows with any active bin in bins 1--3."""
    windows: dict[tuple[str, str, str], bool] = {}
    for row in rows:
        if not row.get("has_bc", False) or int(row["bin_k"]) not in bins:
            continue
        key = (str(row["conversation_id"]), str(row["window_id"]), str(row["speaker"]))
        windows[key] = windows.get(key, False) or bool(row.get("active", row.get("active_prob", 0) > threshold))
    return sum(windows.values())


def false_alarm_by_speaker(rows: Iterable[Mapping], bins: Sequence[int] = (1, 2, 3), threshold: float = 0.5) -> dict[str, dict[int, float]]:
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in _non_bc_rows(rows):
        bin_k = int(row["bin_k"])
        if bin_k not in bins:
            continue
        key = (str(row["speaker"]), bin_k)
        counts[key][1] += 1
        active = row.get("active")
        if active is None:
            active = float(row.get("active_prob", 0)) > threshold
        counts[key][0] += int(bool(active))
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for (speaker, bin_k), (active, total) in counts.items():
        result[speaker][bin_k] = active / total if total else 0.0
    return dict(result)


def max_false_alarm(fa_by_speaker: Mapping[str, Mapping[int, float]], baseline: Mapping[str, Mapping[int, float]]) -> float:
    increases = []
    for speaker, bins in fa_by_speaker.items():
        for bin_k, value in bins.items():
            base = float(baseline.get(speaker, {}).get(bin_k, 0.0))
            increases.append(float("inf") if base == 0 and value > 0 else (value - base) / base if base else 0.0)
    return max(increases, default=0.0)


def summarize_candidate(
    rows: Sequence[Mapping],
    baseline: Mapping[str, Mapping[int, float]],
    bins: Sequence[int] = (1, 2, 3),
    threshold: float = 0.5,
    max_relative_fa: float = 0.05,
) -> dict:
    fa = false_alarm_by_speaker(rows, bins=bins, threshold=threshold)
    fa_increase = max_false_alarm(fa, baseline)
    bc_total = len({(str(r["conversation_id"]), str(r["window_id"]), str(r["speaker"])) for r in rows if r.get("has_bc", False) and int(r["bin_k"]) in bins})
    bc_positive = bc_positive_windows(rows, bins=bins, threshold=threshold)
    return {
        "bc_positive": bc_positive,
        "bc_total": bc_total,
        "bc_recovery": bc_positive / bc_total if bc_total else 0.0,
        "fa_by_speaker": fa,
        "fa_max_increase": fa_increase,
        "feasible": fa_increase <= max_relative_fa,
    }


def rank_feasible_fixed(candidates: Iterable[Sequence[float]], summaries: Mapping[Sequence[float], Mapping], top_k: int | None = None) -> list[tuple[float, ...]]:
    feasible = []
    for candidate in candidates:
        key = tuple(float(value) for value in candidate)
        summary = summaries.get(candidate, summaries.get(key, {}))
        if summary.get("feasible", True):
            feasible.append((key, float(summary.get("bc_recovery", 0.0))))
    feasible.sort(key=lambda item: (-item[1], sum(item[0]), item[0]))
    ranked = [candidate for candidate, _ in feasible]
    return ranked if top_k is None else ranked[:top_k]


def _hist_fa(hist: np.ndarray, total: int, threshold: float, width: int | None = None) -> float:
    if not total:
        return 0.0
    width = int(width or (len(hist) - 1))
    active = sum(int(count) for voiced, count in enumerate(hist) if voiced / width > threshold)
    return active / total


def _array_fa(data: Mapping, thresholds: Sequence[float], bin_widths: Sequence[int] | None = None) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for speaker, bins in data["non_bc_hist"].items():
        for bin_k, hist in bins.items():
            width = bin_widths[int(bin_k) - 1] if bin_widths else None
            total = int(data["non_bc_totals"].get(speaker, {}).get(bin_k, sum(hist)))
            out[speaker][int(bin_k)] = _hist_fa(np.asarray(hist), total, float(thresholds[int(bin_k) - 1]), width)
    return dict(out)


def _array_candidate_summary(
    data: Mapping,
    thresholds: np.ndarray,
    baseline_fa: Mapping[str, Mapping[int, float]],
    bin_widths: Sequence[int] | None = None,
    max_relative_fa: float = 0.05,
) -> dict:
    fa = _array_fa(data, thresholds, bin_widths)
    constrained_fa = {speaker: {bin_k: value for bin_k, value in bins.items() if int(bin_k) <= 3} for speaker, bins in fa.items()}
    constrained_baseline = {speaker: {bin_k: value for bin_k, value in bins.items() if int(bin_k) <= 3} for speaker, bins in baseline_fa.items()}
    fa_increase = max_false_alarm(constrained_fa, constrained_baseline)
    listener = np.asarray(data["listener_ratios"])
    active = (listener[:, :3] > thresholds.reshape(1, -1)[:, :3]).any(axis=1)
    return {
        "bc_positive": int(active.sum()),
        "bc_total": int(len(listener)),
        "bc_recovery": float(active.mean()) if len(listener) else 0.0,
        "fa_by_speaker": fa,
        "fa_max_increase": fa_increase,
        "feasible": bool(fa_increase <= max_relative_fa),
        "thresholds": thresholds.tolist(),
    }


def _adaptive_fa(data: Mapping, lam: float, tau_floor: float) -> dict[str, dict[int, float]]:
    if "non_bc_listener_ratios" not in data or "non_bc_other_ratios" not in data:
        raise ValueError("adaptive FA requires non-BC listener and other ratios")
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for speaker, listener_ratios in data["non_bc_listener_ratios"].items():
        listener_ratios = np.asarray(listener_ratios)
        other_ratios = np.asarray(data["non_bc_other_ratios"][speaker])
        thresholds = np.maximum(float(tau_floor), 0.5 - float(lam) * other_ratios)
        for index, bin_k in enumerate((1, 2, 3, 4)):
            out[speaker][bin_k] = float((listener_ratios[:, index] > thresholds[:, index]).mean()) if len(listener_ratios) else 0.0
    return dict(out)


def label_prefilter(
    data: Mapping,
    fixed_values: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    lambda_grid: Sequence[float] = tuple(i / 20 for i in range(21)),
    tau_floor: float = 0.1,
    bin_widths: Sequence[int] = (10, 20, 30, 40),
    max_relative_fa: float = 0.05,
) -> dict:
    """Perform validation-label screening without loading a model or test rows."""
    baseline_fa = _array_fa(data, (0.5, 0.5, 0.5, 0.5), bin_widths)
    fixed = []
    for candidate in __import__("itertools").product(fixed_values, repeat=4):
        fixed.append(_array_candidate_summary(data, np.asarray(candidate, dtype=float), baseline_fa, bin_widths, max_relative_fa) | {"candidate": list(candidate)})
    other = np.asarray(data["other_ratios"])
    listener = np.asarray(data["listener_ratios"])
    forward = []
    for lam in lambda_grid:
        thresholds = np.maximum(float(tau_floor), 0.5 - float(lam) * other)
        fa = _adaptive_fa(data, float(lam), tau_floor) if "non_bc_listener_ratios" in data else _array_fa(data, (0.5, 0.5, 0.5, 0.5), bin_widths)
        constrained_fa = {speaker: {bin_k: value for bin_k, value in bins.items() if int(bin_k) <= 3} for speaker, bins in fa.items()}
        constrained_baseline = {speaker: {bin_k: value for bin_k, value in bins.items() if int(bin_k) <= 3} for speaker, bins in baseline_fa.items()}
        fa_increase = max_false_alarm(constrained_fa, constrained_baseline)
        active = (listener[:, :3] > thresholds[:, :3]).any(axis=1)
        forward.append({
            "lambda": float(lam),
            "bc_positive": int(active.sum()),
            "bc_total": int(len(listener)),
            "bc_recovery": float(active.mean()) if len(listener) else 0.0,
            "fa_by_speaker": fa,
            "fa_max_increase": fa_increase,
            "feasible": bool(fa_increase <= max_relative_fa),
        })
    return {"baseline_fa": baseline_fa, "fixed": fixed, "forward": forward}


def collect_label_data(config: Mapping, split: str, max_calls: int | None = None) -> dict:
    """Collect validation-only ratio arrays from proxy VAD and DAMSL intervals.

    The collector never opens audio and never reads test conversations. It keeps
    BC-containing rows in compact arrays and stores non-BC ratios per speaker so
    adaptive false-alarm rates can be evaluated without rescanning alignments.
    """
    from experiments.vap_adaptive.data import assign_conversation_split, build_proxy_vad, discover_alignment_files
    from experiments.bc_erase.analyze_bc_erase import discover_damsl_files, parse_word_da_csv

    alignment_calls = discover_alignment_files(config["alignment_root"])
    damsl_files = discover_damsl_files(config["damsl_root"])
    assignment = assign_conversation_split(alignment_calls, damsl_files, seed=int(config.get("split_seed", 1)))
    allowed = {split}
    if split == "train":
        allowed.add("train_extra")
    calls = [call_id for call_id in sorted(alignment_calls) if assignment.get(call_id) in allowed]
    if max_calls is not None:
        calls = calls[: int(max_calls)]
    frame_hz = int(config.get("frame_hz", 50))
    horizon = int(config.get("horizon_frames", 100))
    bin_ranges = ((0, 10), (10, 30), (30, 60), (60, 100))
    bc_listener, bc_other, bc_calls = [], [], []
    non_listener: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    non_other: dict[str, list[np.ndarray]] = {"A": [], "B": []}
    hist: dict[str, dict[int, np.ndarray]] = {"A": {}, "B": {}}
    totals: dict[str, dict[int, int]] = {"A": {}, "B": {}}
    for speaker in ("A", "B"):
        for bin_k, (start, end) in enumerate(bin_ranges, 1):
            hist[speaker][bin_k] = np.zeros(end - start + 1, dtype=np.int64)
            totals[speaker][bin_k] = 0
    for call_id in calls:
        vads = build_proxy_vad(alignment_calls[call_id], frame_hz=frame_hz, median_width=3, laughter_voiced=True)
        speakers = ("A", "B")
        n_frames = min(len(vads[speaker]) for speaker in speakers)
        n_windows = max(0, n_frames - horizon)
        if n_windows == 0:
            continue
        events_by_speaker = {speaker: [] for speaker in speakers}
        for path in damsl_files.get(call_id, []):
            for event in parse_word_da_csv(path, frame_hz=frame_hz):
                events_by_speaker[str(event["speaker"])].append(event)
        has_by_speaker = {}
        starts = np.arange(n_windows, dtype=np.int64) + 1
        for speaker in speakers:
            mask = np.zeros(n_frames, dtype=np.int8)
            for event in events_by_speaker[speaker]:
                lo = max(0, int(event["start_frame"]))
                hi = min(n_frames, int(event["end_frame"]))
                if hi > lo:
                    mask[lo:hi] = 1
            prefix = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
            has_by_speaker[speaker] = np.stack(
                [(prefix[starts + end] - prefix[starts + start]) > 0 for start, end in bin_ranges], axis=1
            )
        ratios_by_speaker = {}
        for speaker in speakers:
            values = np.asarray(vads[speaker], dtype=np.int8)
            prefix = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
            ratios_by_speaker[speaker] = np.stack(
                [(prefix[starts + end] - prefix[starts + start]) / float(end - start) for start, end in bin_ranges], axis=1
            )
        for speaker, other in (("A", "B"), ("B", "A")):
            is_bc = has_by_speaker[speaker][:, :3].any(axis=1)
            listener_ratios = ratios_by_speaker[speaker]
            other_ratios = ratios_by_speaker[other]
            if is_bc.any():
                bc_listener.append(listener_ratios[is_bc])
                bc_other.append(other_ratios[is_bc])
                bc_calls.extend([call_id] * int(is_bc.sum()))
            if (~is_bc).any():
                non_listener[speaker].append(listener_ratios[~is_bc])
                non_other[speaker].append(other_ratios[~is_bc])
                for index, (_start, _end) in enumerate(bin_ranges, 1):
                    width = _end - _start
                    counts = np.rint(listener_ratios[~is_bc, index - 1] * width).astype(int)
                    hist[speaker][index] += np.bincount(counts, minlength=width + 1)[: width + 1]
                    totals[speaker][index] += len(counts)
    empty = np.empty((0, 4), dtype=np.float32)
    bc_listener_array = np.concatenate(bc_listener, axis=0) if bc_listener else empty
    bc_other_array = np.concatenate(bc_other, axis=0) if bc_other else empty
    return {
        "listener_ratios": bc_listener_array,
        "other_ratios": bc_other_array,
        "bc_listener_ratios": bc_listener_array,
        "bc_other_ratios": bc_other_array,
        "bc_conversation_ids": bc_calls,
        "non_bc_listener_ratios": {speaker: np.concatenate(values, axis=0) if values else empty.copy() for speaker, values in non_listener.items()},
        "non_bc_other_ratios": {speaker: np.concatenate(values, axis=0) if values else empty.copy() for speaker, values in non_other.items()},
        "non_bc_hist": hist,
        "non_bc_totals": totals,
        "split": split,
        "calls": calls,
    }


def run_stage1(config: Mapping, split: str = "val", max_calls: int | None = None, output: str | Path | None = None) -> dict:
    data = collect_label_data(config, split, max_calls=max_calls)
    report = label_prefilter(
        data,
        fixed_values=config.get("fixed_threshold_grid", (0.1, 0.2, 0.3, 0.4, 0.5)),
        lambda_grid=config.get("lambda_grid", tuple(i / 20 for i in range(21))),
        tau_floor=float(config.get("tau_floor", 0.1)),
        bin_widths=config.get("bin_frames", (10, 20, 30, 40)),
    )
    fixed_map = {tuple(item["candidate"]): item for item in report["fixed"] if item["feasible"]}
    top_fixed = rank_feasible_fixed(fixed_map, fixed_map, top_k=5)
    feasible_forward = [item for item in report["forward"] if item["feasible"]]
    estimated_epoch_runs = 6 + len(top_fixed) + len(feasible_forward) + 18 + 9 + 3 + 12
    result = {
        "split": split,
        "calls": data["calls"],
        "call_count": len(data["calls"]),
        "bc_listener_instances": int(len(data["bc_listener_ratios"])),
        "non_bc_instances": {speaker: int(len(values)) for speaker, values in data["non_bc_listener_ratios"].items()},
        "baseline_fa": report["baseline_fa"],
        "fixed_candidates": report["fixed"],
        "fixed_feasible_top5": [fixed_map[candidate] for candidate in top_fixed],
        "forward_candidates": report["forward"],
        "forward_feasible": feasible_forward,
        "selection_seed": int(config.get("selection_seed", 1)),
        "fa_constraint_relative": 0.05,
        "reverse_lambda_policy": "reuse_forward_selected_lambda",
        "source_commit": _git_commit(),
        "preexperiment_manifest_sha256": _file_sha256(config.get("preexperiment_manifest")),
        "estimated_epoch_runs": estimated_epoch_runs,
        "estimated_epochs": estimated_epoch_runs * int(config.get("head_epochs", 20)),
        "approved_epoch_budget": config.get("approved_epoch_budget"),
        "budget_ok": config.get("approved_epoch_budget") is None or estimated_epoch_runs * int(config.get("head_epochs", 20)) <= int(config["approved_epoch_budget"]),
    }
    if output is None:
        output = Path(config["output_dir"]) / "calibration" / f"stage1_{split}.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _file_sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage2(
    config: Mapping,
    stage1_json: str | Path,
    max_epochs: int | None = None,
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    output: str | Path | None = None,
) -> dict:
    """Train only Stage-1-feasible candidates on selection seed 1 and select by val BC F1."""
    from experiments.vap_adaptive.train_experiment import run_setting

    stage1 = json.loads(Path(stage1_json).read_text(encoding="utf-8"))
    if stage1.get("budget_ok") is False:
        raise RuntimeError("approved epoch budget exceeded; refusing Stage 2 training")
    seed = int(config.get("selection_seed", 1))
    eval_max_batches = int(limit_val_batches) if isinstance(limit_val_batches, int) else None
    candidates = []
    for item in stage1.get("fixed_feasible_top5", []):
        tuple_values = tuple(float(value) for value in item["candidate"])
        name = "selection_fixed_" + "_".join(f"{value:.2f}" for value in tuple_values)
        manifest = run_setting(
            config,
            "fixed_per_bin",
            seed,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            fixed_thresholds=tuple_values,
            run_name=name,
        )
        metrics_path = Path(config["output_dir"]) / name / str(seed) / "metrics.json"
        metrics = evaluate_checkpoint(config, manifest["checkpoint"], split="val", setting=name, max_batches=eval_max_batches, selection_only=True, output=metrics_path)
        if metrics.get("qa", {}).get("status") != "pass":
            raise RuntimeError(f"Stage 2 candidate {name} failed validation QA: {metrics.get('failure_reasons')}")
        candidates.append({"kind": "fixed_per_bin", "candidate": list(tuple_values), "manifest": manifest, "metrics": metrics})
    for item in stage1.get("forward_feasible", []):
        lam = float(item["lambda"])
        name = f"selection_forward_{lam:.2f}"
        manifest = run_setting(
            config,
            "adaptive_forward",
            seed,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            lam=lam,
            run_name=name,
        )
        metrics_path = Path(config["output_dir"]) / name / str(seed) / "metrics.json"
        metrics = evaluate_checkpoint(config, manifest["checkpoint"], split="val", setting=name, max_batches=eval_max_batches, selection_only=True, output=metrics_path)
        if metrics.get("qa", {}).get("status") != "pass":
            raise RuntimeError(f"Stage 2 candidate {name} failed validation QA: {metrics.get('failure_reasons')}")
        candidates.append({"kind": "adaptive_forward", "lambda": lam, "manifest": manifest, "metrics": metrics})
    if not candidates:
        raise RuntimeError("Stage 2 has no feasible candidates")
    fixed_candidates = [item for item in candidates if item["kind"] == "fixed_per_bin"]
    forward_candidates = [item for item in candidates if item["kind"] == "adaptive_forward"]
    if not fixed_candidates or not forward_candidates:
        raise RuntimeError("Stage 2 must produce both fixed-per-bin and forward selections")
    selected_fixed = max(fixed_candidates, key=lambda item: float(item["metrics"].get("bc_f1") or 0.0))
    selected_forward = max(forward_candidates, key=lambda item: float(item["metrics"].get("bc_f1") or 0.0))
    result = {
        "selection_seed": seed,
        "selected_fixed_thresholds": selected_fixed["candidate"],
        "selected_fixed_bc_f1": selected_fixed["metrics"].get("bc_f1"),
        "selected_lambda": selected_forward["lambda"],
        "selected_forward_bc_f1": selected_forward["metrics"].get("bc_f1"),
        "selected_kind": "independent_fixed_and_forward",
        "candidates": candidates,
        "reverse_policy": "reuse_selected_lambda",
    }
    if output is None:
        output = Path(config["output_dir"]) / "calibration" / "stage2_selection.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_floor_sensitivity(
    config: Mapping,
    selection_json: str | Path,
    floors: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    max_epochs: int | None = None,
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    output: str | Path | None = None,
) -> dict:
    from experiments.vap_adaptive.train_experiment import run_setting

    selection = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    lam = selection.get("selected_lambda")
    if lam is None:
        raise RuntimeError("floor sensitivity requires selected forward lambda")
    rows = []
    for floor in floors:
        if abs(float(floor) - 0.10) < 1e-9:
            rows.append({"tau_floor": float(floor), "reused_formal": True, "lambda_star": float(lam)})
            continue
        name = f"floor_{float(floor):.2f}"
        manifest = run_setting(
            config,
            "adaptive_forward",
            int(config.get("selection_seed", 1)),
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            lam=float(lam),
            tau_floor=float(floor),
            run_name=name,
        )
        metrics = evaluate_checkpoint(config, manifest["checkpoint"], split="val", setting=name, output=Path(config["output_dir"]) / name / str(config.get("selection_seed", 1)) / "metrics.json")
        rows.append({"tau_floor": float(floor), "reused_formal": False, "lambda_star": float(lam), "metrics": metrics, "manifest": manifest})
    result = {"rows": rows, "selection_seed": int(config.get("selection_seed", 1)), "reused_floor": 0.10}
    output = Path(output or Path(config["output_dir"]) / "calibration" / "floor_sensitivity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_full_model_lr_scan(
    config: Mapping,
    learning_rates: Sequence[float] | None = None,
    max_epochs: int | None = None,
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    output: str | Path | None = None,
) -> dict:
    from experiments.vap_adaptive.train_experiment import run_setting

    learning_rates = tuple(learning_rates or config.get("full_model_learning_rates", (1e-5, 5e-5, 1e-4)))
    rows = []
    for learning_rate in learning_rates:
        name = f"lr_scan_{learning_rate:g}"
        manifest = run_setting(
            config,
            "full_standard",
            int(config.get("selection_seed", 1)),
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            learning_rate=float(learning_rate),
            run_name=name,
        )
        metrics = evaluate_checkpoint(config, manifest["checkpoint"], split="val", setting=name, output=Path(config["output_dir"]) / name / str(config.get("selection_seed", 1)) / "metrics.json")
        rows.append({"learning_rate": float(learning_rate), "metrics": metrics, "manifest": manifest})
    selected = max(rows, key=lambda row: float(row["metrics"].get("bc_f1") or 0.0)) if rows else None
    result = {"rows": rows, "selected_learning_rate": selected["learning_rate"] if selected else None, "selection_seed": int(config.get("selection_seed", 1))}
    output = Path(output or Path(config["output_dir"]) / "fine_tuning" / "lr_scan.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_finetune_comparison(
    config: Mapping,
    selection_json: str | Path,
    lr_scan_json: str | Path,
    max_epochs: int | None = None,
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    output: str | Path | None = None,
) -> dict:
    from experiments.vap_adaptive.train_experiment import run_setting

    selection = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    lr_scan = json.loads(Path(lr_scan_json).read_text(encoding="utf-8"))
    lam = selection.get("selected_lambda")
    learning_rate = lr_scan.get("selected_learning_rate")
    if lam is None or learning_rate is None:
        raise RuntimeError("fine-tuning comparison requires selected lambda and learning rate")
    seeds = tuple(int(seed) for seed in config.get("formal_seeds", (1, 2, 3)))
    conditions = {
        "standard_head": ("standard", False),
        "standard_full": ("full_standard", True),
        "adaptive_head": ("adaptive_forward", False),
        "adaptive_full": ("full_forward", True),
    }
    rows = []
    for method, (setting, _full) in conditions.items():
        for seed in seeds:
            # Head-only rows are the already validated formal checkpoints;
            # reuse them instead of spending six duplicate training runs.
            if method == "standard_head":
                manifest_path = Path(config["output_dir"]) / "standard" / str(seed) / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                checkpoint = manifest["checkpoint"]
            elif method == "adaptive_head":
                manifest_path = Path(config["output_dir"]) / "adaptive_forward" / str(seed) / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                checkpoint = manifest["checkpoint"]
            else:
                manifest = run_setting(
                    config,
                    setting,
                    seed,
                    max_epochs=max_epochs,
                    limit_train_batches=limit_train_batches,
                    limit_val_batches=limit_val_batches,
                    lam=float(lam) if "adaptive" in method else None,
                    learning_rate=float(learning_rate),
                    run_name=method,
                )
                checkpoint = manifest["checkpoint"]
            metrics_path = Path(config["output_dir"]) / method / str(seed) / "metrics.json"
            metrics = evaluate_checkpoint(config, checkpoint, split="test", setting=method, output=metrics_path)
            rows.append({
                "method": method,
                "seed": seed,
                "BC_F1": metrics.get("bc_f1"),
                "BC_vs_shift_F1": metrics.get("bc_vs_shift_F1"),
                "trainable_params": manifest["head"].get("trainable_parameters"),
                "retrain_epochs": manifest.get("max_epochs"),
                "metrics": metrics,
                "manifest": manifest,
            })
    result = {"rows": rows, "selected_lambda": lam, "selected_learning_rate": learning_rate, "selection_seed": int(config.get("selection_seed", 1))}
    output = Path(output or Path(config["output_dir"]) / "fine_tuning" / "finetune_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _event_masks(config: Mapping, call_id: str) -> dict[str, np.ndarray]:
    from experiments.vap_adaptive.data import build_proxy_vad, discover_alignment_files
    from experiments.bc_erase.analyze_bc_erase import discover_damsl_files, parse_word_da_csv

    alignments = discover_alignment_files(config["alignment_root"])[call_id]
    vads = build_proxy_vad(alignments, frame_hz=int(config.get("frame_hz", 50)), median_width=3, laughter_voiced=True)
    n_frames = max(len(values) for values in vads.values())
    events = discover_damsl_files(config["damsl_root"])
    masks = {"A": np.zeros(n_frames, dtype=bool), "B": np.zeros(n_frames, dtype=bool)}
    for path in events.get(call_id, []):
        for event in parse_word_da_csv(path, frame_hz=int(config.get("frame_hz", 50))):
            speaker = str(event["speaker"])
            lo = max(0, int(event["start_frame"]))
            hi = min(n_frames, int(event["end_frame"]))
            if speaker in masks and hi > lo:
                masks[speaker][lo:hi] = True
    return masks


def _projection_bin_presence(mask: np.ndarray) -> np.ndarray:
    """Return has-BC masks for bins 1-3 at every projection frame."""
    values = np.asarray(mask, dtype=np.int8)
    n_frames = int(values.shape[0])
    prefix = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
    result = np.zeros((n_frames, 3), dtype=bool)
    for bin_index, (bin_start, bin_end) in enumerate(((0, 10), (10, 30), (30, 60))):
        count = max(0, n_frames - bin_end)
        if count:
            lo = prefix[1 + bin_start : 1 + bin_start + count]
            hi = prefix[1 + bin_end : 1 + bin_end + count]
            result[:count, bin_index] = (hi - lo) > 0
    return result


def _f1_from_counts(counts: Mapping[str, int]) -> float:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def _load_run_model(checkpoint: str | Path, device: torch.device):
    from experiments.vap_adaptive.train_experiment import build_model

    model = build_model(None, full_model=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    translated = {}
    for key, value in state.items():
        if key.startswith("model."):
            translated[key[len("model."):]] = value
    if translated:
        model.load_state_dict(translated, strict=False)
    model.to(device).eval()
    return model


def evaluate_checkpoint(
    config: Mapping,
    checkpoint: str | Path,
    split: str = "test",
    setting: str = "standard",
    max_batches: int | None = None,
    selection_only: bool = False,
    output: str | Path | None = None,
) -> dict:
    """Evaluate BC/FA metrics with conversation-level accumulation."""
    from torch.utils.data import DataLoader
    from experiments.vap_adaptive.data import SwitchboardDataset
    from experiments.vap_adaptive.feature_cache import FeatureCacheDataset, cached_collate, cache_ready
    from vap.events import EventConfig, TurnTakingEvents

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_run_model(checkpoint, device)
    if cache_ready(config, split):
        dataset = FeatureCacheDataset(config, split)
    else:
        dataset = SwitchboardDataset(
            config,
            split,
            window_seconds=float(config.get("window_seconds", 20.0)),
            stride_seconds=float(config.get("stride_seconds", 20.0)),
        )
    eval_batch_size = (
        int(config.get("feature_cache_eval_batch_size", 256))
        if cache_ready(config, split)
        else int(config.get("eval_batch_size", config.get("batch_size", 1)))
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=cached_collate if cache_ready(config, split) else None,
    )
    conversation_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    fa_counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    bin_counts: dict[int, dict[str, int]] = {bin_k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for bin_k in range(1, 4)}
    bc_shift_rows: list[dict] = []
    event_values: dict[str, list[tuple[float, int]]] = {"shift": [], "hold": [], "long": [], "short": []}
    event_extractor = None if selection_only else TurnTakingEvents(EventConfig(frame_hz=int(config.get("frame_hz", 50))))
    mask_cache: dict[str, dict[str, np.ndarray]] = {}
    processed_batches = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and processed_batches >= max_batches:
                break
            processed_batches += 1
            if "features" in batch:
                logits = model.vap_head(batch["features"].to(device))
                output_model = {"logits": logits, "vad": None}
            else:
                waveform = batch["waveform"].to(device)
                output_model = model(waveform)
            active_probs = decode_state_probabilities(output_model["logits"])
            vap_probs = output_model["logits"].softmax(dim=-1)
            p_fut = model.objective.probs_next_speaker_aggregate(vap_probs, from_bin=2, to_bin=3)
            active_probs_np = active_probs.detach().cpu().numpy()
            p_fut_np = p_fut.detach().cpu().numpy()
            events = None
            if event_extractor is not None:
                try:
                    events = event_extractor(batch["vad"].to(device))
                    objective_probs = model.objective.get_probs(output_model["logits"])
                    event_preds, event_targets = model.objective.extract_prediction_and_targets(
                        p_now=objective_probs["p_now"], p_fut=objective_probs["p_future"], events=events
                    )
                    for key, metric_key in (("pred_shift", "shift"), ("hs", "hold"), ("ls", "long")):
                        if event_preds.get(key) is not None and event_targets.get(key) is not None:
                            event_values[metric_key].extend(
                                (float(pred), int(target))
                                for pred, target in zip(event_preds[key].detach().cpu().tolist(), event_targets[key].detach().cpu().tolist())
                            )
                except Exception:
                    events = None
            batch_size = batch["features"].shape[0] if "features" in batch else waveform.shape[0]
            for index in range(batch_size):
                call_id = str(batch["conversation_id"][index])
                if call_id not in mask_cache:
                    masks = _event_masks(config, call_id)
                    mask_cache[call_id] = {speaker: _projection_bin_presence(mask) for speaker, mask in masks.items()}
                masks = mask_cache[call_id]
                start_frame = int(batch["window_start_frame"][index])
                valid_frames = int(batch["valid_frames"][index])
                n_predictions = min(int(active_probs.shape[1]), max(0, valid_frames - int(config.get("horizon_frames", 100))))
                counts = conversation_counts[call_id]
                bc_row_candidates: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
                for speaker_index, speaker in enumerate(("A", "B")):
                    predicted_bins = active_probs_np[index, :n_predictions, speaker_index, :3] > 0.5
                    predicted_bc = predicted_bins.any(axis=1)
                    presence = masks[speaker]
                    has_bins = np.zeros((n_predictions, 3), dtype=bool)
                    if start_frame < len(presence):
                        available = min(n_predictions, len(presence) - start_frame)
                        has_bins[:available] = presence[start_frame : start_frame + available]
                    has_bc = has_bins.any(axis=1)
                    counts["tp"] += int(np.count_nonzero(has_bc & predicted_bc))
                    counts["fn"] += int(np.count_nonzero(has_bc & ~predicted_bc))
                    counts["fp"] += int(np.count_nonzero(~has_bc & predicted_bc))
                    counts["tn"] += int(np.count_nonzero(~has_bc & ~predicted_bc))
                    for bin_index in range(3):
                        has_bin = has_bins[:, bin_index]
                        predicted_bin = predicted_bins[:, bin_index]
                        fa_counts[(speaker, bin_index + 1)][1] += int(np.count_nonzero(~has_bc))
                        fa_counts[(speaker, bin_index + 1)][0] += int(np.count_nonzero((~has_bc) & predicted_bin))
                        bin_counts[bin_index + 1]["tp"] += int(np.count_nonzero(has_bin & predicted_bin))
                        bin_counts[bin_index + 1]["fn"] += int(np.count_nonzero(has_bin & ~predicted_bin))
                        bin_counts[bin_index + 1]["fp"] += int(np.count_nonzero(~has_bin & predicted_bin))
                        bin_counts[bin_index + 1]["tn"] += int(np.count_nonzero(~has_bin & ~predicted_bin))
                    for local_t in np.flatnonzero(has_bc):
                        bc_row_candidates[int(local_t)].append((speaker_index, float(active_probs_np[index, local_t, speaker_index, :3].max()), float(p_fut_np[index, local_t, speaker_index])))
                # Preserve the legacy frame-first, speaker-second ordering;
                # seeded BC/shift matching is intentionally order-sensitive.
                for local_t in sorted(bc_row_candidates):
                    for speaker_index, bc_score, shift_score in bc_row_candidates[local_t]:
                        bc_shift_rows.append({
                            "conversation_id": call_id,
                            "speaker": ("A", "B")[speaker_index],
                            "event": "bc",
                            "bc_score": bc_score,
                            "shift_score": shift_score,
                        })
                if events is not None:
                    for start, end, speaker_index in events["pred_shift"][index]:
                        for local_t in range(max(0, int(start)), min(int(end), int(p_fut.shape[1]))):
                            bc_shift_rows.append({
                                "conversation_id": call_id,
                                "speaker": ("A", "B")[int(speaker_index)],
                                "event": "shift",
                                "bc_score": float(active_probs_np[index, local_t, int(speaker_index), :3].max()),
                                "shift_score": float(p_fut_np[index, local_t, int(speaker_index)]),
                            })
    total_counts = {key: sum(counts[key] for counts in conversation_counts.values()) for key in ("tp", "fp", "fn", "tn")}
    conv_rows = [{"conversation_id": call_id, **counts} for call_id, counts in conversation_counts.items()]
    bootstrap = cluster_bootstrap(conv_rows, lambda rows: _f1_from_counts({key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")}), replicates=int(config.get("bootstrap_replicates", 2000)), seed=int(config.get("selection_seed", 1))) if conv_rows else None
    fa_by_speaker: dict[str, dict[str, float]] = defaultdict(dict)
    for (speaker, bin_k), (active, total) in fa_counts.items():
        fa_by_speaker[speaker][str(bin_k)] = active / total if total else 0.0
    def _binary_f1(values: list[tuple[float, int]]) -> float | None:
        if not values:
            return None
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for score, target in values:
            predicted = score > 0.5
            if predicted and target:
                counts["tp"] += 1
            elif predicted:
                counts["fp"] += 1
            elif target:
                counts["fn"] += 1
        return _f1_from_counts({**counts, "tn": 0})

    event_failure = [] if selection_only or any(event_values.values()) else ["existing event evaluator returned no evaluable regions"]
    matched_bc_shift_rows = match_bc_shift_rows(bc_shift_rows, seed=int(config.get("selection_seed", 1)))
    metrics = {
        "setting": setting,
        "split": split,
        "checkpoint": str(Path(checkpoint).resolve()),
        "selection_only": selection_only,
        "feature_cache_used": cache_ready(config, split),
        "feature_cache_dir": str(config.get("feature_cache_dir", "")) if cache_ready(config, split) else None,
        "bc_f1": _f1_from_counts(total_counts),
        "bc_counts": total_counts,
        "conversation_counts": conv_rows,
        "bc_bin_f1": {str(bin_k): _f1_from_counts(counts) for bin_k, counts in bin_counts.items()},
        "fa_by_speaker": dict(fa_by_speaker),
        "bootstrap": bootstrap,
        "processed_batches": processed_batches,
        "shift_F1": None if selection_only else _binary_f1(event_values["shift"]),
        "hold_F1": None if selection_only else _binary_f1(event_values["hold"]),
        "long_F1": None if selection_only else _binary_f1(event_values["long"]),
        "bc_vs_shift": None if selection_only else (bc_vs_shift_f1(matched_bc_shift_rows) if matched_bc_shift_rows else None),
        "bc_vs_shift_F1": None if selection_only else (bc_vs_shift_f1(matched_bc_shift_rows)["f1"] if matched_bc_shift_rows else None),
        "failure_reasons": event_failure,
    }
    metrics["qa"] = {
        "status": "pass" if not event_failure and all(np.isfinite(float(value)) for value in (metrics["bc_f1"],)) else "failed",
        "failure_reasons": event_failure,
        "checkpoint_exists": Path(checkpoint).exists(),
        "strict_comparison": config.get("comparison") == "gt",
    }
    if output is None:
        output = Path(config["output_dir"]) / setting / "metrics.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def bc_vs_shift_f1(rows: Iterable[Mapping]) -> dict[str, float | int]:
    actual = []
    predicted = []
    for row in rows:
        event = row.get("event")
        if event not in {"bc", "shift"}:
            continue
        score = float(row.get("bc_score", 0.0)) - float(row.get("shift_score", 0.0))
        actual.append(int(event == "bc"))
        predicted.append(int(score > 0.0))
    tp = sum(a and p for a, p in zip(actual, predicted))
    fp = sum((not a) and p for a, p in zip(actual, predicted))
    fn = sum(a and (not p) for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "positive_count": sum(actual), "negative_count": len(actual) - sum(actual), "tp": tp, "fp": fp, "fn": fn}


def match_bc_shift_rows(rows: Sequence[Mapping], seed: int = 1) -> list[dict]:
    """Balance BC positives and shift negatives within each conversation/speaker."""
    groups: dict[tuple[str, str], dict[str, list[Mapping]]] = defaultdict(lambda: {"bc": [], "shift": []})
    for row in rows:
        event = str(row.get("event", ""))
        if event in {"bc", "shift"}:
            groups[(str(row["conversation_id"]), str(row.get("speaker", "")))][event].append(row)
    rng = random.Random(seed)
    matched: list[dict] = []
    for group in groups.values():
        count = min(len(group["bc"]), len(group["shift"]))
        matched.extend(dict(row) for row in rng.sample(group["bc"], count))
        matched.extend(dict(row) for row in rng.sample(group["shift"], count))
    return matched


def cluster_bootstrap(
    rows: Sequence[Mapping],
    metric_fn: Callable[[list[Mapping]], float],
    replicates: int = 2000,
    seed: int = 1,
) -> dict[str, float | int]:
    """Resample conversation IDs with replacement, then recompute the global metric."""
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[str(row["conversation_id"])].append(row)
    conversation_ids = sorted(groups)
    if not conversation_ids:
        raise ValueError("cluster bootstrap requires at least one conversation")
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample: list[Mapping] = []
        for conversation_id in rng.choices(conversation_ids, k=len(conversation_ids)):
            sample.extend(groups[conversation_id])
        values.append(float(metric_fn(sample)))
    interval = np.percentile(np.asarray(values), [2.5, 97.5])
    return {
        "replicates": int(replicates),
        "conversation_count": len(conversation_ids),
        "estimate": float(metric_fn(list(rows))),
        "low": float(interval[0]),
        "high": float(interval[1]),
    }


def filter_candidates(candidates: Iterable, evaluate_fn: Callable[[object], Mapping], max_relative_fa: float = 0.05) -> list:
    feasible = []
    for candidate in candidates:
        result = evaluate_fn(candidate)
        if float(result.get("fa_max_increase", float("inf"))) <= max_relative_fa:
            feasible.append(candidate)
    return feasible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--stage1-json", default=None)
    parser.add_argument("--floor-sensitivity", action="store_true")
    parser.add_argument("--lr-scan", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--lr-scan-json", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--setting", default="standard")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=str, default="1.0")
    parser.add_argument("--limit-val-batches", type=str, default="1.0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"metric_contract": "bc-vs-shift score = bc_active_prob - shift_prob; positive iff > 0", "bootstrap_replicates": 2000}, indent=2))
        return
    if args.stage1:
        from experiments.vap_adaptive.gates import load_config

        config, _ = load_config(args.config)
        result = run_stage1(config, split=args.split, max_calls=args.max_calls, output=args.output)
        print(json.dumps({"output": str(args.output or Path(config["output_dir"]) / "calibration" / f"stage1_{args.split}.json"), "call_count": result["call_count"], "fixed_feasible_top5": len(result["fixed_feasible_top5"]), "forward_feasible": len(result["forward_feasible"])}, indent=2))
        return
    if args.stage2:
        from experiments.vap_adaptive.gates import load_config

        config, _ = load_config(args.config)
        stage1_json = args.stage1_json or str(Path(config["output_dir"]) / "calibration" / "stage1_val.json")
        result = run_stage2(config, stage1_json, max_epochs=args.max_epochs, limit_train_batches=parse_limit(args.limit_train_batches), limit_val_batches=parse_limit(args.limit_val_batches), output=args.output)
        print(json.dumps({"output": str(args.output or Path(config["output_dir"]) / "calibration" / "stage2_selection.json"), "selected_kind": result["selected_kind"], "selected_lambda": result["selected_lambda"], "selected_fixed_thresholds": result["selected_fixed_thresholds"]}, indent=2))
        return
    if args.floor_sensitivity:
        from experiments.vap_adaptive.gates import load_config

        config, _ = load_config(args.config)
        selection_json = args.stage1_json or str(Path(config["output_dir"]) / "calibration" / "stage2_selection.json")
        result = run_floor_sensitivity(config, selection_json, max_epochs=args.max_epochs, limit_train_batches=parse_limit(args.limit_train_batches), limit_val_batches=parse_limit(args.limit_val_batches), output=args.output)
        print(json.dumps({"output": str(args.output or Path(config["output_dir"]) / "calibration" / "floor_sensitivity.json"), "rows": len(result["rows"])}, indent=2))
        return
    if args.lr_scan:
        from experiments.vap_adaptive.gates import load_config

        config, _ = load_config(args.config)
        result = run_full_model_lr_scan(config, max_epochs=args.max_epochs, limit_train_batches=parse_limit(args.limit_train_batches), limit_val_batches=parse_limit(args.limit_val_batches), output=args.output)
        print(json.dumps({"output": str(args.output or Path(config["output_dir"]) / "fine_tuning" / "lr_scan.json"), "selected_learning_rate": result["selected_learning_rate"]}, indent=2))
        return
    if args.finetune:
        from experiments.vap_adaptive.gates import load_config

        config, _ = load_config(args.config)
        selection_json = args.stage1_json or str(Path(config["output_dir"]) / "calibration" / "stage2_selection.json")
        lr_scan_json = args.lr_scan_json or str(Path(config["output_dir"]) / "fine_tuning" / "lr_scan.json")
        result = run_finetune_comparison(config, selection_json, lr_scan_json, max_epochs=args.max_epochs, limit_train_batches=parse_limit(args.limit_train_batches), limit_val_batches=parse_limit(args.limit_val_batches), output=args.output)
        print(json.dumps({"output": str(args.output or Path(config["output_dir"]) / "fine_tuning" / "finetune_results.json"), "rows": len(result["rows"])}, indent=2))
        return
    if args.evaluate:
        from experiments.vap_adaptive.gates import load_config

        config, _ = load_config(args.config)
        if not args.checkpoint:
            raise SystemExit("--evaluate requires --checkpoint")
        result = evaluate_checkpoint(config, args.checkpoint, split=args.split, setting=args.setting, max_batches=args.max_batches, output=args.output)
        print(json.dumps({"output": str(args.output or Path(config["output_dir"]) / args.setting / "metrics.json"), "bc_f1": result["bc_f1"], "processed_batches": result["processed_batches"]}, indent=2))
        return
    raise SystemExit("evaluation requires trained predictions; use --dry-run for the contract check")


if __name__ == "__main__":
    main()
