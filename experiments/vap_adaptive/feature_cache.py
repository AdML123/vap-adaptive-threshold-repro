"""Frozen VAP projection-feature cache for head-only runs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cache_root(config: Mapping, split: str) -> Path:
    root = Path(config.get("feature_cache_dir", Path(config["output_dir"]) / "feature_cache"))
    return root / split


def cache_manifest(config: Mapping, split: str) -> Path:
    return cache_root(config, split) / "manifest.json"


def cache_ready(config: Mapping, split: str) -> bool:
    manifest_path = cache_manifest(config, split)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(config["checkpoint"])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        compatible = (
            manifest.get("checkpoint_sha256") == digest
            and float(manifest.get("window_seconds", -1)) == float(config.get("window_seconds", 20.0))
            and float(manifest.get("stride_seconds", -1)) == float(config.get("stride_seconds", 20.0))
        )
        return bool(manifest.get("status") == "pass" and manifest.get("shards") and compatible) and all(
            (cache_root(config, split) / shard).exists() for shard in manifest["shards"]
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return False


class FeatureCacheDataset(IterableDataset):
    """Yield cached feature/VAD samples with the same metadata contract as audio data."""

    def __init__(self, config: Mapping, split: str):
        super().__init__()
        manifest_path = cache_manifest(config, split)
        if not manifest_path.exists():
            raise FileNotFoundError(f"feature cache manifest missing: {manifest_path}")
        self.root = cache_root(config, split)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "pass":
            raise ValueError(f"feature cache is not valid: {manifest_path}")
        self._eager = bool(config.get("feature_cache_eager", True))
        self._features = None
        self._vad = None
        self._metadata = None
        if self._eager:
            self._load_eager()

    def _load_eager(self) -> None:
        """Load shards once so every epoch avoids repeated 20-GB disk scans."""
        first = torch.load(self.root / self.manifest["shards"][0], map_location="cpu", weights_only=False)
        sample_count = int(self.manifest.get("samples") or 0)
        if not sample_count:
            sample_count = sum(
                int(torch.load(self.root / relative, map_location="cpu", weights_only=False)["features"].shape[0])
                for relative in self.manifest["shards"]
            )
        self._features = torch.empty(
            (sample_count, *first["features"].shape[1:]), dtype=first["features"].dtype
        )
        self._vad = torch.empty((sample_count, *first["vad"].shape[1:]), dtype=first["vad"].dtype)
        self._metadata = []
        offset = 0
        for relative in self.manifest["shards"]:
            shard = torch.load(self.root / relative, map_location="cpu", weights_only=False)
            count = int(shard["features"].shape[0])
            self._features[offset : offset + count].copy_(shard["features"])
            self._vad[offset : offset + count].copy_(shard["vad"])
            self._metadata.extend(shard["metadata"])
            offset += count
        if offset != sample_count or len(self._metadata) != sample_count:
            raise ValueError(f"feature cache sample count mismatch for {self.root}")

    def __iter__(self):
        if self._eager:
            for index, metadata in enumerate(self._metadata):
                yield {
                    "features": self._features[index],
                    "vad": self._vad[index].float(),
                    **metadata,
                }
            return
        for relative in self.manifest["shards"]:
            shard = torch.load(self.root / relative, map_location="cpu", weights_only=False)
            features = shard["features"]
            vad = shard["vad"]
            for index, metadata in enumerate(shard["metadata"]):
                yield {
                    "features": features[index],
                    "vad": vad[index].float(),
                    **metadata,
                }


class CachedBatchDataset(Dataset):
    """Map-style training view that avoids one Python dict per cached sample."""

    def __init__(self, config: Mapping, split: str, batch_size: int):
        self._base = FeatureCacheDataset(config, split)
        self._features = self._base._features
        self._vad = self._base._vad
        self.batch_size = int(batch_size)
        self._lazy = not self._base._eager
        self._manifest = self._base.manifest
        self._root = self._base.root
        self._shard_counts = None
        if self._lazy:
            # Repacked caches have one shard per training batch. For legacy
            # caches, retain a generic offset index and assemble crossing
            # batches without retaining shard tensors in memory.
            self._shard_counts = []
            for relative in self._manifest["shards"]:
                shard = torch.load(self._root / relative, map_location="cpu", weights_only=False)
                self._shard_counts.append(int(shard["features"].shape[0]))
            self._offsets = [0]
            for count in self._shard_counts:
                self._offsets.append(self._offsets[-1] + count)

    def __len__(self) -> int:
        samples = int(self._features.shape[0]) if self._features is not None else int(self._manifest.get("samples", self._offsets[-1]))
        return (samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> dict:
        if self._lazy:
            start = int(index) * self.batch_size
            stop = min(start + self.batch_size, self._offsets[-1])
            features_parts = []
            vad_parts = []
            metadata_parts = []
            for shard_index, (shard_start, shard_stop) in enumerate(zip(self._offsets[:-1], self._offsets[1:])):
                if shard_stop <= start or shard_start >= stop:
                    continue
                shard = torch.load(self._root / self._manifest["shards"][shard_index], map_location="cpu", weights_only=False)
                lo = max(start, shard_start) - shard_start
                hi = min(stop, shard_stop) - shard_start
                features_parts.append(shard["features"][lo:hi])
                vad_parts.append(shard["vad"][lo:hi])
                metadata_parts.extend(shard["metadata"][lo:hi])
            if not features_parts:
                raise IndexError(index)
            result = {"features": torch.cat(features_parts, dim=0).float(), "vad": torch.cat(vad_parts, dim=0).float()}
            if metadata_parts:
                keys = sorted({key for item in metadata_parts for key in item})
                for key in keys:
                    result[key] = [item.get(key) for item in metadata_parts]
            return result
        start = int(index) * self.batch_size
        stop = min(start + self.batch_size, int(self._features.shape[0]))
        return {
            "features": self._features[start:stop].float(),
            "vad": self._vad[start:stop].float(),
        }
def cached_collate(batch: list[dict]) -> dict:
    """Stack half-precision cached features, then cast once per batch."""
    collated = default_collate(batch)
    collated["features"] = collated["features"].float()
    return collated


def _projection_features(model, waveform: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        x1, x2 = model.encode_audio(waveform)
        o1 = model.ar_channel(x1)
        o2 = model.ar_channel(x2)
        return model.ar(o1["x"], o2["x"])["x"]


def build_feature_cache(
    config: Mapping,
    split: str,
    batch_size: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Run the frozen backbone once and write auditable feature shards."""
    if cache_ready(config, split) and not overwrite:
        return json.loads(cache_manifest(config, split).read_text(encoding="utf-8"))

    from torch.utils.data import DataLoader
    from experiments.vap_adaptive.data import SwitchboardDataset
    from experiments.vap_adaptive.train_experiment import build_model

    root = cache_root(config, split)
    root.mkdir(parents=True, exist_ok=True)
    for path in root.glob("shard-*.pt"):
        path.unlink()
    model = build_model(config["checkpoint"], full_model=False).cuda() if torch.cuda.is_available() else build_model(config["checkpoint"], full_model=False)
    model.eval()
    batch_size = int(batch_size or config.get("feature_cache_batch_size", config.get("head_batch_size", 16)))
    dataset = SwitchboardDataset(
        config,
        split,
        window_seconds=float(config.get("window_seconds", 20.0)),
        stride_seconds=float(config.get("stride_seconds", 20.0)),
    )
    device = next(model.parameters()).device
    shards: list[str] = []
    shard_index = 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=int(config.get("feature_cache_workers", 0)),
    )
    for batch in loader:
        waveform = batch["waveform"].to(device)
        features = _projection_features(model, waveform).cpu().half()
        vad = batch["vad"].cpu().float()
        batch_count = int(features.shape[0])
        metadata = []
        for index in range(batch_count):
            row = {}
            for key, value in batch.items():
                if key in {"waveform", "vad"}:
                    continue
                item = value[index]
                if isinstance(item, torch.Tensor) and item.ndim == 0:
                    item = int(item)
                elif isinstance(item, torch.Tensor):
                    item = item.tolist()
                row[key] = item
            metadata.append(row)
        name = f"shard-{shard_index:06d}.pt"
        torch.save({"features": features, "vad": vad, "metadata": metadata}, root / name)
        shards.append(name)
        shard_index += 1
    digest = hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest()
    manifest = {
        "status": "pass" if shards else "failed",
        "split": split,
        "shards": shards,
        "samples": sum(int(torch.load(root / shard, map_location="cpu", weights_only=False)["features"].shape[0]) for shard in shards),
        "feature_dtype": "float16",
        "checkpoint": str(Path(config["checkpoint"]).resolve()),
        "checkpoint_sha256": digest,
        "window_seconds": float(config.get("window_seconds", 20.0)),
        "stride_seconds": float(config.get("stride_seconds", 20.0)),
        "frame_hz": int(config.get("frame_hz", 50)),
        "horizon_frames": int(config.get("horizon_frames", 100)),
        "head_batch_size": batch_size,
    }
    cache_manifest(config, split).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse
    from experiments.vap_adaptive.gates import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    for split in args.split:
        print(json.dumps(build_feature_cache(config, split, args.batch_size, args.overwrite), indent=2))
