"""Strict VAP target-label generation for adaptive-threshold experiments."""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import torch


BIN_RANGES = ((0, 10), (10, 30), (30, 60), (60, 100))
HORIZON_FRAMES = 100


def adaptive_threshold(
    other_ratio: float | torch.Tensor,
    lam: float,
    tau_floor: float,
    direction: str = "forward",
):
    if direction not in {"forward", "reverse"}:
        raise ValueError(f"unknown adaptive direction: {direction}")
    if lam < 0 or tau_floor < 0:
        raise ValueError("lambda and tau_floor must be non-negative")
    activity = other_ratio if direction == "forward" else 1 - other_ratio
    return torch.clamp(0.5 - lam * activity, min=tau_floor) if isinstance(activity, torch.Tensor) else max(tau_floor, 0.5 - lam * activity)


def _projection_ratios(vad: torch.Tensor) -> torch.Tensor:
    if vad.ndim != 3 or vad.shape[-1] != 2:
        raise ValueError("vad must have shape (batch, frames, 2)")
    if vad.shape[1] < HORIZON_FRAMES + 1:
        raise ValueError("vad must contain at least 101 frames")
    future = vad[:, 1:, :].unfold(dimension=1, size=HORIZON_FRAMES, step=1)
    # unfold returns (batch, windows, speakers, horizon) for this layout.
    ratios = []
    for start, end in BIN_RANGES:
        ratios.append(future[..., start:end].mean(dim=-1))
    return torch.stack(ratios, dim=-1)


def _thresholds(
    ratios: torch.Tensor,
    mode: str,
    lam: float,
    tau_floor: float,
    fixed_thresholds: Sequence[float] | None,
) -> torch.Tensor:
    if mode == "standard":
        return torch.full_like(ratios, 0.5)
    if mode == "uniform_lower":
        return torch.full_like(ratios, 0.3)
    if mode == "extreme":
        return torch.full_like(ratios, 0.1)
    if mode == "fixed_per_bin":
        if fixed_thresholds is None or len(fixed_thresholds) != 4:
            raise ValueError("fixed_per_bin requires four thresholds")
        return torch.tensor(fixed_thresholds, dtype=ratios.dtype, device=ratios.device).view(1, 1, 1, 4).expand_as(ratios)
    if mode not in {"forward", "reverse"}:
        raise ValueError(f"unknown label mode: {mode}")
    other = ratios.flip(dims=[2])
    return adaptive_threshold(other, lam, tau_floor, direction=mode)


def _encode_codebook(binary_bins: torch.Tensor) -> torch.Tensor:
    if binary_bins.shape[-2:] != (2, 4):
        raise ValueError("binary bins must have shape (..., 2, 4)")
    powers = (2 ** torch.arange(8, device=binary_bins.device)).view(1, 1, 2, 4)
    return (binary_bins.to(torch.long) * powers).sum(dim=(-1, -2))


def make_labels(
    vad: torch.Tensor,
    mode: str = "standard",
    lam: float = 0.0,
    tau_floor: float = 0.1,
    fixed_thresholds: Sequence[float] | None = None,
) -> torch.Tensor:
    ratios = _projection_ratios(vad)
    thresholds = _thresholds(ratios, mode, lam, tau_floor, fixed_thresholds)
    binary = (ratios > thresholds).to(vad.dtype)
    return _encode_codebook(binary)


def fixed_threshold_candidates(values: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5)):
    return list(product(values, repeat=4))


def rank_fixed_candidates(candidates: Sequence[Sequence[float]], recovery: dict) -> list[tuple[float, ...]]:
    normalized = [tuple(float(value) for value in candidate) for candidate in candidates]
    return sorted(normalized, key=lambda candidate: (-recovery[candidate], sum(candidate), candidate))

