"""Small, dependency-light adapters for Switchboard audio and splits."""

from __future__ import annotations

import audioop
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bc_erase.analyze_bc_erase import (  # noqa: E402
    alignment_to_vad,
    discover_damsl_files as _discover_damsl_files,
    discover_alignment_files as _discover_alignment_files,
    parse_isip_word_alignment,
)


def _parse_sphere_header(raw: bytes) -> dict[str, str]:
    if not raw.startswith(b"NIST_1A\n"):
        raise ValueError("not a NIST SPHERE file")
    try:
        header_size = int(raw.splitlines()[1].strip())
    except (IndexError, ValueError) as exc:
        raise ValueError("invalid NIST SPHERE header size") from exc
    text = raw[:header_size].decode("ascii", errors="replace")
    fields: dict[str, str] = {}
    for line in text.splitlines()[2:]:
        if line == "end_head":
            break
        match = re.match(r"^([^\s]+)\s+-[^\s]+\s+(.+)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    fields["_header_size"] = str(header_size)
    return fields


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform
    gcd = np.gcd(source_rate, target_rate)
    return resample_poly(waveform, target_rate // gcd, source_rate // gcd, axis=1).astype(np.float32)


def read_sphere(path: str | Path, target_sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
    """Read a NIST SPHERE recording as ``(channels, frames)`` float32 audio."""
    path = Path(path)
    raw = path.read_bytes()
    fields = _parse_sphere_header(raw[:4096])
    header_size = int(fields["_header_size"])
    channels = int(fields.get("channel_count", "1"))
    sample_count = int(fields["sample_count"])
    source_rate = int(float(fields["sample_rate"]))
    coding = fields.get("sample_coding", "pcm").lower()
    payload = raw[header_size:]
    if "ulaw" in coding or "mu-law" in coding:
        pcm = audioop.ulaw2lin(payload, 2)
        values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif "pcm" in coding:
        sample_bytes = int(fields.get("sample_n_bytes", "2"))
        if sample_bytes != 2:
            raise ValueError(f"unsupported SPHERE PCM width: {sample_bytes}")
        values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"unsupported SPHERE sample coding: {coding}")
    expected = sample_count * channels
    if values.size < expected:
        raise ValueError(f"truncated SPHERE payload: expected {expected} samples, got {values.size}")
    values = values[:expected].reshape(sample_count, channels).T
    values = np.clip(_resample(values, source_rate, target_sample_rate), -1.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(values)), target_sample_rate


def assign_conversation_split(
    call_ids: Iterable[str], damsl_ids: Iterable[str], seed: int = 1,
    fractions: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Assign DAMSL calls to deterministic train/val/test and others to train_extra."""
    fractions = fractions or {"train": 0.7, "val": 0.1, "test": 0.2}
    all_ids = sorted(set(call_ids))
    damsl = set(damsl_ids)
    ordered = sorted(damsl.intersection(all_ids), key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    n_train = int(len(ordered) * fractions["train"])
    n_val = int(len(ordered) * fractions["val"])
    assignment: dict[str, str] = {call_id: "train_extra" for call_id in all_ids if call_id not in damsl}
    for i, call_id in enumerate(ordered):
        assignment[call_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


def discover_alignment_files(root: str | Path) -> dict[str, dict[str, Path]]:
    return _discover_alignment_files(root)


def build_proxy_vad(
    call_paths: Mapping[str, str | Path],
    frame_hz: int = 50,
    median_width: int | None = 3,
    laughter_voiced: bool = True,
) -> dict[str, list[int]]:
    vads = {
        speaker: alignment_to_vad(
            parse_isip_word_alignment(path),
            frame_hz=frame_hz,
            laughter_voiced=laughter_voiced,
            median_width=median_width,
        )
        for speaker, path in call_paths.items()
    }
    n_frames = max((len(values) for values in vads.values()), default=0)
    return {speaker: values + [0] * (n_frames - len(values)) for speaker, values in vads.items()}


def build_data_manifest(config: Mapping) -> dict:
    alignment_calls = discover_alignment_files(config["alignment_root"])
    damsl_calls = _discover_damsl_files(config["damsl_root"])
    assignment = assign_conversation_split(alignment_calls, damsl_calls, seed=int(config.get("split_seed", 1)))
    split_counts = {name: sum(value == name for value in assignment.values()) for name in ("train", "val", "test", "train_extra")}
    return {
        "alignment_conversations": len(alignment_calls),
        "damsl_conversations": len(set(damsl_calls).intersection(alignment_calls)),
        "split_counts": split_counts,
        "missing_alignment_for_damsl": sorted(set(damsl_calls) - set(alignment_calls)),
        "call_ids": sorted(alignment_calls),
    }


def load_conversation(
    config: Mapping,
    call_id: str,
    median_width: int | None = 3,
    laughter_voiced: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one call and construct its two-channel proxy VAD."""
    audio, sample_rate = read_sphere(find_sphere_audio(config["audio_root"], call_id), target_sample_rate=16000)
    alignment = discover_alignment_files(config["alignment_root"])[call_id]
    vads = build_proxy_vad(alignment, int(config.get("frame_hz", 50)), median_width, laughter_voiced)
    speakers = sorted(vads)
    if speakers != ["A", "B"]:
        raise ValueError(f"expected A/B alignments for {call_id}, got {speakers}")
    audio_frames = int(round(audio.shape[1] * int(config.get("frame_hz", 50)) / sample_rate))
    n_frames = max(audio_frames, *(len(vads[speaker]) for speaker in speakers))
    vad = torch.zeros((n_frames, 2), dtype=torch.float32)
    for index, speaker in enumerate(speakers):
        values = torch.tensor(vads[speaker], dtype=torch.float32)
        vad[: values.shape[0], index] = values
    return audio, vad


def iter_samples(
    config: Mapping,
    split: str,
    call_ids: Iterable[str] | None = None,
    window_seconds: float = 20.0,
    stride_seconds: float | None = None,
    median_width: int | None = 3,
    laughter_voiced: bool = True,
):
    """Yield fixed-duration, padded stereo samples for one split."""
    frame_hz = int(config.get("frame_hz", 50))
    sample_rate = 16000
    window_frames = max(100, int(round(window_seconds * frame_hz)))
    stride_frames = max(1, int(round((stride_seconds or window_seconds) * frame_hz)))
    if call_ids is None:
        manifest = build_data_manifest(config)
        damsl_calls = set(_discover_damsl_files(config["damsl_root"]))
        assignment = assign_conversation_split(manifest["call_ids"], damsl_calls, seed=int(config.get("split_seed", 1)))
        allowed_splits = {split}
        if split == "train":
            allowed_splits.add("train_extra")
        call_ids = [call_id for call_id in manifest["call_ids"] if assignment.get(call_id) in allowed_splits]
    for call_id in call_ids:
        waveform, vad = load_conversation(config, str(call_id), median_width, laughter_voiced)
        total_frames = max(1, int(np.ceil(waveform.shape[1] / (sample_rate / frame_hz))))
        for start in range(0, total_frames, stride_frames):
            audio_start = start * sample_rate // frame_hz
            audio_end = audio_start + window_frames * sample_rate // frame_hz
            chunk = torch.zeros((2, audio_end - audio_start), dtype=torch.float32)
            source = waveform[:, audio_start:min(audio_end, waveform.shape[1])]
            chunk[:, : source.shape[1]] = source
            vad_chunk = torch.zeros((window_frames + 1, 2), dtype=torch.float32)
            source_vad = vad[start:min(start + window_frames + 1, vad.shape[0])]
            vad_chunk[: source_vad.shape[0]] = source_vad
            yield make_sample(
                chunk,
                vad_chunk,
                str(call_id),
                split,
                frame_hz,
                horizon_frames=100,
                window_start_frame=start,
                valid_frames=max(0, min(window_frames, total_frames - start)),
            )
            if audio_end >= waveform.shape[1]:
                break


class SwitchboardDataset(IterableDataset):
    def __init__(self, config: Mapping, split: str, **kwargs):
        super().__init__()
        self.config = config
        self.split = split
        self.kwargs = kwargs

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            return iter_samples(self.config, self.split, **self.kwargs)
        manifest = build_data_manifest(self.config)
        damsl_calls = set(_discover_damsl_files(self.config["damsl_root"]))
        assignment = assign_conversation_split(manifest["call_ids"], damsl_calls, seed=int(self.config.get("split_seed", 1)))
        allowed = {self.split}
        if self.split == "train":
            allowed.add("train_extra")
        calls = [call_id for call_id in manifest["call_ids"] if assignment.get(call_id) in allowed]
        calls = calls[worker.id :: worker.num_workers]
        return iter_samples(self.config, self.split, call_ids=calls, **self.kwargs)


def write_data_manifest(config: Mapping, output: str | Path | None = None) -> Path:
    manifest = build_data_manifest(config)
    manifest["training_budget"] = estimate_window_budget(config, "train")
    output = Path(output or Path(config["output_dir"]) / "calibration" / "data_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output


def estimate_window_budget(config: Mapping, split: str = "train") -> dict:
    """Estimate fixed-window counts from alignment durations without decoding audio."""
    alignment_calls = discover_alignment_files(config["alignment_root"])
    damsl_calls = _discover_damsl_files(config["damsl_root"])
    assignment = assign_conversation_split(alignment_calls, damsl_calls, seed=int(config.get("split_seed", 1)))
    allowed = {split}
    if split == "train":
        allowed.add("train_extra")
    window_frames = max(1, int(round(float(config.get("window_seconds", 20.0)) * int(config.get("frame_hz", 50)))))
    frame_counts = []
    for call_id, speakers in alignment_calls.items():
        if assignment.get(call_id) not in allowed:
            continue
        max_end = 0.0
        for path in speakers.values():
            records = parse_isip_word_alignment(path)
            max_end = max(max_end, *(float(record["end"]) for record in records), 0.0)
        frame_counts.append(int(np.ceil(max_end * int(config.get("frame_hz", 50)))))
    return {
        "split": split,
        "conversation_count": len(frame_counts),
        "total_frames": int(sum(frame_counts)),
        "audio_hours": float(sum(frame_counts) / int(config.get("frame_hz", 50)) / 3600.0),
        "window_frames": window_frames,
        "window_count_nonoverlap": int(sum((frames + window_frames - 1) // window_frames for frames in frame_counts)),
    }


def find_sphere_audio(audio_root: str | Path, call_id: str) -> Path:
    """Resolve a Switchboard call id to its SPHERE file, tolerating zero padding."""
    root = Path(audio_root)
    direct = list(root.rglob(f"{call_id}.sph"))
    if direct:
        return direct[0]
    match = re.search(r"(\d+)$", call_id)
    if match:
        number = int(match.group(1))
        matches = list(root.rglob(f"sw{number:05d}.sph"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no SPHERE file for {call_id} below {root}")


def make_sample(
    waveform: torch.Tensor,
    vad: torch.Tensor,
    conversation_id: str,
    split: str,
    frame_hz: int = 50,
    horizon_frames: int = 100,
    window_start_frame: int = 0,
    valid_frames: int | None = None,
) -> dict:
    """Validate and package one stereo training sample."""
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError("waveform must have two channels")
    if vad.ndim != 2 or vad.shape[1] != 2:
        raise ValueError("vad must have shape (frames, 2)")
    if vad.shape[0] < horizon_frames:
        raise ValueError(f"vad must contain at least {horizon_frames} future frames")
    if not torch.isfinite(waveform).all() or not torch.isfinite(vad).all():
        raise ValueError("waveform and vad must be finite")
    return {
        "waveform": waveform.float(),
        "vad": vad.to(dtype=torch.float32),
        "conversation_id": str(conversation_id),
        "split": str(split),
        "frame_hz": int(frame_hz),
        "window_start_frame": int(window_start_frame),
        "valid_frames": int(valid_frames if valid_frames is not None else vad.shape[0]),
    }
