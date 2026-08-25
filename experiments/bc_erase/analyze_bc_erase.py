"""Streaming Switchboard/VAP backchannel erasure pre-experiment.

The module intentionally uses only the Python standard library so that the
analysis can be audited and run in the project's conda environment without a
second data-processing stack.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


HORIZON_FRAMES = 100
BIN_RANGES = ((0, 10), (10, 30), (30, 60), (60, 100))
SILENCE_WORDS = {"[silence]", "[noise]", "[vocalized-noise]", "<sil>", "<noise>"}


def iter_projection_windows(n_frames: int, horizon: int = HORIZON_FRAMES) -> range:
    """Yield stride-one window starts whose [t+1, t+1+horizon) is complete."""
    return range(max(0, n_frames - horizon))


def median_filter_binary(values: Sequence[int], width: int = 3) -> list[int]:
    if width < 1 or width % 2 == 0:
        raise ValueError("median filter width must be a positive odd integer")
    radius = width // 2
    output: list[int] = []
    for i in range(len(values)):
        lo, hi = max(0, i - radius), min(len(values), i + radius + 1)
        window = values[lo:hi]
        output.append(int(sum(window) * 2 > len(window)))
    return output


def compute_bin_metrics(
    listener_vad: Sequence[int],
    other_vad: Sequence[int],
    bc_mask: Sequence[int],
    threshold: float,
) -> dict:
    if not (len(listener_vad) == len(other_vad) == len(bc_mask)):
        raise ValueError("VAD and BC mask slices must have equal lengths")
    width = len(listener_vad)
    if width == 0:
        raise ValueError("bin slice cannot be empty")
    listener_frames = int(sum(listener_vad))
    other_frames = int(sum(other_vad))
    bc_frames = int(sum(int(b and v) for b, v in zip(bc_mask, listener_vad)))
    ratio = listener_frames / width
    return {
        "listener_voiced_frames": listener_frames,
        "other_voiced_frames": other_frames,
        "bc_voiced_frames": bc_frames,
        "voiced_ratio": ratio,
        "label": int(ratio > threshold),
        "bc_dominant": listener_frames > other_frames,
    }


def _event_bin_intersects(event: Mapping, start: int, end: int) -> bool:
    return int(event["start_frame"]) < end and int(event["end_frame"]) > start


def iter_bc_bin_rows(
    call_id: str,
    vad_by_speaker: Mapping[str, Sequence[int]],
    events: Sequence[Mapping],
    thresholds: Sequence[float] = (0.5,),
) -> Iterator[dict]:
    """Enumerate listener-specific BC-containing rows for every complete window.

    Rows are deduplicated by (window, bin, listener), while overlapping BC
    events from that listener are unioned into the audit mask.
    """
    speakers = tuple(sorted(vad_by_speaker))
    if len(speakers) != 2:
        raise ValueError("vad_by_speaker must contain exactly two speakers")
    n_frames = min(len(v) for v in vad_by_speaker.values())
    events_by_listener: dict[str, list[Mapping]] = defaultdict(list)
    for event in events:
        speaker = str(event["speaker"])
        if speaker in vad_by_speaker and int(event["end_frame"]) > int(event["start_frame"]):
            events_by_listener[speaker].append(event)

    for listener in speakers:
        other = speakers[1] if listener == speakers[0] else speakers[0]
        listener_events = events_by_listener.get(listener, [])
        if not listener_events:
            continue
        # Candidate keys are generated analytically from BC/bin intersections;
        # this is equivalent to scanning all stride-one windows but avoids a
        # full call-by-call event cross-product.
        candidate_keys: set[tuple[int, int]] = set()
        for event in listener_events:
            for bin_k, (offset_start, offset_end) in enumerate(BIN_RANGES, 1):
                min_t = max(0, int(event["start_frame"]) - offset_end)
                max_t = min(n_frames - HORIZON_FRAMES - 1, int(event["end_frame"]) - offset_start - 2)
                if max_t >= min_t:
                    candidate_keys.update((t, bin_k) for t in range(min_t, max_t + 1))
        for window_t, bin_k in sorted(candidate_keys):
            offset_start, offset_end = BIN_RANGES[bin_k - 1]
            bin_start = window_t + 1 + offset_start
            bin_end = window_t + 1 + offset_end
            relevant = [e for e in listener_events if _event_bin_intersects(e, bin_start, bin_end)]
            if not relevant:
                continue
            bc_mask = [0] * (bin_end - bin_start)
            for event in relevant:
                start = max(bin_start, int(event["start_frame"]))
                end = min(bin_end, int(event["end_frame"]))
                for i in range(start - bin_start, max(start - bin_start, end - bin_start)):
                    bc_mask[i] = 1
            base = compute_bin_metrics(
                vad_by_speaker[listener][bin_start:bin_end],
                vad_by_speaker[other][bin_start:bin_end],
                bc_mask,
                thresholds[0],
            )
            row = {
                "call_id": call_id,
                "window_t": window_t,
                "bin_k": bin_k,
                "listener_speaker": listener,
                "other_speaker": other,
                "has_bc": True,
                "event_ids": ";".join(str(e.get("event_id", "")) for e in relevant),
                **base,
            }
            for threshold in thresholds:
                row[f"label_tau_{threshold:g}"] = compute_bin_metrics(
                    vad_by_speaker[listener][bin_start:bin_end],
                    vad_by_speaker[other][bin_start:bin_end],
                    bc_mask,
                    threshold,
                )["label"]
            yield row


def parse_isip_word_alignment(path: str | Path) -> list[dict]:
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) < 4 or fields[0].startswith("#"):
            continue
        try:
            start, end = float(fields[1]), float(fields[2])
        except ValueError:
            continue
        records.append({"utt_id": fields[0], "start": start, "end": end, "word": fields[3]})
    return records


def word_is_voiced(word: str, laughter_voiced: bool = True) -> bool:
    token = word.strip().lower()
    if token in SILENCE_WORDS or token in {"", "<silence>"}:
        return False
    if "laughter" in token or token in {"[laugh]", "<laughter>"}:
        return laughter_voiced
    return not (token.startswith("[") and token.endswith("]"))


def alignment_to_vad(
    records: Sequence[Mapping],
    frame_hz: int = 50,
    laughter_voiced: bool = True,
    median_width: int | None = 3,
) -> list[int]:
    max_end = max((float(r["end"]) for r in records), default=0.0)
    n_frames = max(1, math.ceil(max_end * frame_hz))
    vad = [0] * n_frames
    for record in records:
        if not word_is_voiced(str(record["word"]), laughter_voiced):
            continue
        start = max(0, int(float(record["start"]) * frame_hz))
        end = min(n_frames, int(float(record["end"]) * frame_hz))
        for frame in range(start, max(start, end)):
            vad[frame] = 1
    return median_filter_binary(vad, median_width) if median_width else vad


def parse_word_da_csv(path: str | Path, bc_labels: Sequence[str] = ("b", "bh"), frame_hz: int = 50) -> list[dict]:
    path = Path(path)
    events: dict[tuple[str, str], dict] = {}
    allowed = {label.lower() for label in bc_labels}
    for row in csv.reader(path.open(encoding="utf-8", newline="")):
        if len(row) < 7:
            continue
        utt_id, start, end, word, _boi, da, da_idx = row[:7]
        if da.strip().lower() not in allowed:
            continue
        match = re.match(r"sw(\d+)([AB])", utt_id)
        if not match:
            continue
        speaker = match.group(2)
        key = (speaker, da_idx)
        event = events.setdefault(key, {"event_id": f"{path.stem}:{speaker}:{da_idx}", "speaker": speaker, "da_idx": da_idx, "start": float(start), "end": float(end)})
        event["start"] = min(event["start"], float(start))
        event["end"] = max(event["end"], float(end))
    output = []
    for event in events.values():
        event = dict(event)
        event["start_frame"] = int(event["start"] * frame_hz)
        event["end_frame"] = int(event["end"] * frame_hz)
        output.append(event)
    return sorted(output, key=lambda e: (e["start_frame"], e["speaker"], e["event_id"]))


def discover_alignment_files(root: str | Path) -> dict[str, dict[str, Path]]:
    calls: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in Path(root).rglob("*-word.text"):
        match = re.match(r"(sw\d+)([AB])-ms98-.*-word\.text$", path.name)
        if match:
            calls[match.group(1)][match.group(2)] = path
    return dict(calls)


def discover_damsl_files(root: str | Path) -> dict[str, list[Path]]:
    """Group the two speaker-specific DAMSL files by conversation ID."""
    output: dict[str, list[Path]] = defaultdict(list)
    for path in Path(root).glob("sw*-word-da.csv"):
        match = re.match(r"(sw\d+)[AB]-word-da\.csv$", path.name)
        if match:
            output[match.group(1)].append(path)
    return {call_id: sorted(paths) for call_id, paths in output.items()}


def _word_da_utt_ids(path: Path) -> set[str]:
    ids = set()
    for row in csv.reader(path.open(encoding="utf-8", newline="")):
        if row:
            ids.add(row[0])
    return ids


def _alignment_utt_ids(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=1)
        if fields and fields[0].startswith("sw"):
            ids.add(fields[0])
    return ids


def make_split_assignments(call_ids: Sequence[str], seed: int = 1, fractions: Mapping[str, float] | None = None) -> dict[str, str]:
    fractions = fractions or {"train": 0.7, "val": 0.1, "test": 0.2}
    ids = sorted(set(call_ids))
    # Hash-based ordering is deterministic across Python versions and hosts.
    ids = sorted(ids, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    n_train = int(len(ids) * fractions["train"])
    n_val = int(len(ids) * fractions["val"])
    assignment = {}
    for i, call_id in enumerate(ids):
        assignment[call_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


def _resolve_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    for key in ("alignment_root", "word_da_root", "audio_root", "output_dir"):
        config[key] = (base / config[key]).resolve()
    return config, config_path


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _pad_vads(vads: Mapping[str, list[int]]) -> dict[str, list[int]]:
    n = max((len(v) for v in vads.values()), default=0)
    return {speaker: values + [0] * (n - len(values)) for speaker, values in vads.items()}


def build_proxy_vad(call_paths: Mapping[str, Path], frame_hz: int, median_width: int | None, laughter_voiced: bool) -> dict[str, list[int]]:
    vads = {}
    for speaker in ("A", "B"):
        records = parse_isip_word_alignment(call_paths[speaker])
        vads[speaker] = alignment_to_vad(records, frame_hz, laughter_voiced, median_width)
    return _pad_vads(vads)


def _variant_specs(config: Mapping) -> tuple[tuple[str, int | None, bool], ...]:
    width = int(config.get("median_filter_frames", 3))
    return (
        ("median_laughter_voiced", width, True),
        ("no_median_laughter_voiced", None, True),
        ("median_laughter_unvoiced", width, False),
    )


def _summary_row(variant: str, split_name: str, bin_k: int, counters: Mapping) -> dict:
    n = counters["N"]
    e = counters["E"]
    nd = counters["N_dom"]
    ed = counters["E_dom"]
    nc = counters["N_coex"]
    ec = counters["E_coex"]
    return {
        "variant": variant,
        "split": split_name,
        "bin_k": bin_k,
        "N_k": n,
        "E_k": e,
        "D_k": e / n if n else None,
        "N_k_dom": nd,
        "E_k_dom": ed,
        "D_k_dom": ed / nd if nd else None,
        "N_k_coex": nc,
        "E_k_coex": ec,
        "D_k_coex": ec / nc if nc else None,
    }


def classify_decision(d_total: float | None, d34: float | None, increase: float | None) -> str:
    if increase is not None and increase < 0.05:
        return "abandon_threshold_not_recovering"
    if d_total is not None and d_total > 0.15 and increase is not None and increase > 0.15:
        return "rewrite_paper_severe_erasure"
    if d_total is not None and d_total < 0.05 and d34 is not None and d34 > 0.20 and increase is not None and increase > 0.15:
        return "per_bin_long_bin_experiment"
    if d_total is not None and d_total < 0.05 and d34 is not None and d34 < 0.20:
        return "abandon_target_minor_erasure"
    if d_total is not None and 0.05 <= d_total <= 0.15 and increase is not None and increase > 0.10:
        return "calibrate_per_bin_moderate_erasure"
    return "gray_zone_review_per_bin"


def write_figures(output_dir: Path) -> list[str]:
    """Write compact QA figures using the non-interactive Agg backend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    rows = list(csv.DictReader((output_dir / "summary.csv").open(encoding="utf-8")))
    qa = json.loads((output_dir / "qa.json").read_text(encoding="utf-8"))
    variants = sorted({row["variant"] for row in rows})
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for variant in variants:
        values = []
        for k in range(1, 5):
            match = [r for r in rows if r["variant"] == variant and r["bin_k"] == str(k)]
            n = sum(int(r["N_k"]) for r in match)
            e = sum(int(r["E_k"]) for r in match)
            values.append(e / n if n else 0.0)
        ax.plot(range(1, 5), values, marker="o", label=variant)
    ax.set(xlabel="VAP bin", ylabel="Erasure rate D_k", xticks=[1, 2, 3, 4])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_bin_erasure.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = range(len(variants))
    tau05 = [qa["bc_positive"][v]["tau_0.5"] for v in variants]
    tau03 = [qa["bc_positive"][v]["tau_0.3"] for v in variants]
    width = 0.38
    ax.bar([i - width / 2 for i in x], tau05, width, label="tau=0.5")
    ax.bar([i + width / 2 for i in x], tau03, width, label="tau=0.3")
    ax.set_xticks(list(x), variants, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("BC-positive listener instances")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "threshold_comparison.png", dpi=180)
    plt.close(fig)
    return ["figures/per_bin_erasure.png", "figures/threshold_comparison.png"]


def inventory(config: Mapping) -> dict:
    alignments = discover_alignment_files(config["alignment_root"])
    damsl = discover_damsl_files(config["word_da_root"])
    aligned_calls = {call for call, speakers in alignments.items() if set(speakers) == {"A", "B"}}
    damsl_calls = set(damsl)
    missing_ids = []
    checked_files = 0
    for call_id, paths in damsl.items():
        for damsl_path in paths:
            match = re.match(r"sw\d+([AB])-word-da\.csv$", damsl_path.name)
            if not match or call_id not in alignments or match.group(1) not in alignments[call_id]:
                continue
            checked_files += 1
            missing = _word_da_utt_ids(damsl_path) - _alignment_utt_ids(alignments[call_id][match.group(1)])
            if missing:
                missing_ids.extend(sorted(missing)[:10])
    return {
        "alignment_calls": len(aligned_calls),
        "damsl_calls": len(damsl_calls),
        "damsl_aligned_calls": len(aligned_calls & damsl_calls),
        "damsl_without_alignment": sorted(damsl_calls - aligned_calls),
        "alignment_without_damsl": len(aligned_calls - damsl_calls),
        "id_mapping": "word-DAMSL utt_id prefixes match ISIP call/speaker IDs",
        "id_mapping_files_checked": checked_files,
        "id_mapping_missing_utt_ids": sorted(set(missing_ids)),
        "id_mapping_verified": not missing_ids and checked_files == sum(len(v) for v in damsl.values()),
    }


def run_analysis(config: Mapping) -> dict:
    alignments = discover_alignment_files(config["alignment_root"])
    damsl_files = discover_damsl_files(config["word_da_root"])
    calls = sorted(set(alignments) & set(damsl_files))
    sample_rate = float(config.get("sample_rate", 1.0))
    if sample_rate < 1:
        sample_count = max(1, math.ceil(len(calls) * sample_rate))
        calls = sorted(calls, key=lambda x: hashlib.sha256(f"{config.get('split_seed', 1)}:{x}".encode()).hexdigest())[:sample_count]
        calls.sort()
    split = make_split_assignments(sorted(set(damsl_files)), int(config.get("split_seed", 1)), config.get("split_fractions"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps({k: str(v) for k, v in config.items()}, sort_keys=True).encode()).hexdigest()
    manifest = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        "inventory": inventory(config),
        "calls_processed": calls,
        "split_counts": {s: sum(split.get(c) == s for c in calls) for s in ("train", "val", "test")},
        "projection": {"stride_frames": 1, "horizon_frames": HORIZON_FRAMES, "bin_ranges": BIN_RANGES},
        "bc_definition": "DAMSL b/bh; listener is event speaker; other is the other channel",
        "proxy_vad": {"frame_hz": config.get("frame_hz", 50), "variants": [name for name, _, _ in _variant_specs(config)]},
        "config_sha256": config_hash,
        "environment": {"python": __import__("sys").version.split()[0]},
    }
    event_path = output_dir / "event_windows.csv"
    summary_counters: dict[tuple[str, str, int], dict[str, int]] = defaultdict(lambda: {"N": 0, "E": 0, "N_dom": 0, "E_dom": 0, "N_coex": 0, "E_coex": 0})
    positive: dict[tuple[str, str, float], int] = defaultdict(int)
    positive_by_bin: dict[tuple[str, str, int, float], int] = defaultdict(int)
    instance_totals: dict[tuple[str, str], int] = defaultdict(int)
    fieldnames = ["variant", "split", "call_id", "window_t", "bin_k", "listener_speaker", "other_speaker", "event_ids", "event_start_frame", "event_end_frame", "event_duration_s", "listener_voiced_frames", "other_voiced_frames", "bc_voiced_frames", "voiced_ratio", "bc_dominant", "label_tau_0.5", "label_tau_0.3"]
    duration_counts = defaultdict(lambda: {"all": 0, "short_lt_750ms": 0})
    with event_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        frame_hz = int(config.get("frame_hz", 50))
        thresholds = tuple(float(x) for x in config.get("thresholds", [0.5, 0.3]))
        for variant, median_width, laughter_voiced in _variant_specs(config):
            for call_id in calls:
                events = []
                for damsl_path in damsl_files[call_id]:
                    events.extend(parse_word_da_csv(damsl_path, config.get("bc_labels", ["b", "bh"]), frame_hz))
                vads = build_proxy_vad(alignments[call_id], frame_hz, median_width, laughter_voiced)
                rows = list(iter_bc_bin_rows(call_id, vads, events, thresholds))
                split_name = split.get(call_id, "unsplit")
                duration_counts[(variant, split_name)]["all"] += len(events)
                duration_counts[(variant, split_name)]["short_lt_750ms"] += sum((e["end"] - e["start"]) < 0.75 for e in events)
                instances: dict[tuple[int, str], dict] = {}
                for row in rows:
                    row["variant"] = variant
                    row["split"] = split_name
                    row["event_start_frame"] = min(int(e["start_frame"]) for e in events if e["event_id"] in row["event_ids"].split(";"))
                    row["event_end_frame"] = max(int(e["end_frame"]) for e in events if e["event_id"] in row["event_ids"].split(";"))
                    row["event_duration_s"] = (row["event_end_frame"] - row["event_start_frame"]) / frame_hz
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
                    key = (int(row["window_t"]), row["listener_speaker"])
                    state = instances.setdefault(key, {"bins": set(), "labels": defaultdict(set)})
                    bin_k = int(row["bin_k"])
                    if bin_k <= 3:
                        state["bins"].add(bin_k)
                        for threshold in thresholds:
                            state["labels"][threshold].add((bin_k, int(row[f"label_tau_{threshold:g}"])))
                    counters = summary_counters[(variant, split_name, bin_k)]
                    counters["N"] += 1
                    counters["E"] += int(row["label_tau_0.5"] == 0)
                    if row["bc_dominant"]:
                        counters["N_dom"] += 1
                        counters["E_dom"] += int(row["label_tau_0.5"] == 0)
                    else:
                        counters["N_coex"] += 1
                        counters["E_coex"] += int(row["label_tau_0.5"] == 0)
                for state in instances.values():
                    if not state["bins"]:
                        continue
                    instance_totals[(variant, split_name)] += 1
                    for threshold in thresholds:
                        active = any(label for _bin, label in state["labels"][threshold])
                        positive[(variant, split_name, threshold)] += int(active)
                        for bin_k in state["bins"]:
                            positive_by_bin[(variant, split_name, bin_k, threshold)] += int((bin_k, 1) in state["labels"][threshold])

    summary_path = output_dir / "summary.csv"
    summary_fields = list(_summary_row("", "", 1, {"N": 0, "E": 0, "N_dom": 0, "E_dom": 0, "N_coex": 0, "E_coex": 0}).keys())
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for (variant, split_name, bin_k), counters in sorted(summary_counters.items()):
            writer.writerow(_summary_row(variant, split_name, bin_k, counters))
    qa = {"D1_is_lowest": {}, "bc_positive": {}, "bc_positive_by_bin": {}, "d_total": {}, "d_34_weight": {}}
    for variant in sorted({key[0] for key in summary_counters}):
        all_counts = {k: {field: 0 for field in ("N", "E")} for k in range(1, 5)}
        for (v, _split, k), counters in summary_counters.items():
            if v == variant:
                all_counts[k]["N"] += counters["N"]
                all_counts[k]["E"] += counters["E"]
        ds = {k: (all_counts[k]["E"] / all_counts[k]["N"] if all_counts[k]["N"] else None) for k in all_counts}
        qa["D1_is_lowest"][variant] = bool(ds[1] is not None and all(ds[1] <= value for value in ds.values() if value is not None))
        total_n = sum(x["N"] for x in all_counts.values())
        total_e = sum(x["E"] for x in all_counts.values())
        n34 = all_counts[3]["N"] + all_counts[4]["N"]
        e34 = all_counts[3]["E"] + all_counts[4]["E"]
        qa["d_total"][variant] = total_e / total_n if total_n else None
        qa["d_34_weight"][variant] = e34 / n34 if n34 else None
        base = positive.get((variant, "test", thresholds[0]), 0) + positive.get((variant, "train", thresholds[0]), 0) + positive.get((variant, "val", thresholds[0]), 0)
        low = positive.get((variant, "test", thresholds[1]), 0) + positive.get((variant, "train", thresholds[1]), 0) + positive.get((variant, "val", thresholds[1]), 0)
        qa["bc_positive"][variant] = {"tau_0.5": base, "tau_0.3": low, "increase_ratio": (low - base) / base if base else None, "instance_count": sum(instance_totals[(variant, s)] for s in ("train", "val", "test"))}
        qa["bc_positive_by_bin"][variant] = {
            str(bin_k): {
                "tau_0.5": sum(positive_by_bin[(variant, s, bin_k, thresholds[0])] for s in ("train", "val", "test")),
                "tau_0.3": sum(positive_by_bin[(variant, s, bin_k, thresholds[1])] for s in ("train", "val", "test")),
            }
            for bin_k in range(1, 4)
        }
    qa["bc_duration_sensitivity"] = {
        variant: {split_name: duration_counts[(variant, split_name)] for split_name in ("train", "val", "test")}
        for variant, _width, _laugh in _variant_specs(config)
    }
    qa["decision"] = {}
    for variant in qa["d_total"]:
        increase = qa["bc_positive"][variant]["increase_ratio"]
        qa["decision"][variant] = classify_decision(qa["d_total"][variant], qa["d_34_weight"][variant], increase)
    summary_payload = {
        "per_variant": {
            variant: {
                "D_total": qa["d_total"][variant],
                "D_34_weight": qa["d_34_weight"][variant],
                "BC_pos": qa["bc_positive"][variant],
                "BC_pos_by_bin": qa["bc_positive_by_bin"][variant],
                "decision": qa["decision"][variant],
            }
            for variant in qa["d_total"]
        }
    }
    _write_json(output_dir / "summary.json", summary_payload)
    _write_json(output_dir / "qa.json", qa)
    _write_json(output_dir / "run_manifest.json", manifest)
    manifest["artifacts"] = ["event_windows.csv", "summary.csv", "summary.json", "qa.json", "run_manifest.json"] + write_figures(output_dir)
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--sample-rate", type=float)
    args = parser.parse_args(argv)
    config, _ = _resolve_config(args.config)
    if args.sample_rate is not None:
        config["sample_rate"] = args.sample_rate
    if args.inventory_only:
        result = inventory(config)
        _write_json(Path(config["output_dir"]) / "inventory.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not args.run:
        parser.error("choose --inventory-only or --run")
    result = run_analysis(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
