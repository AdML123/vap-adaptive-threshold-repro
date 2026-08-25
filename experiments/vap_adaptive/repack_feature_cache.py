"""Repack small feature-cache shards into batch-sized lazy-load shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repack(source: Path, target: Path, batch_size: int) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    for path in target.glob("shard-*.pt"):
        path.unlink()
    feature_parts = []
    vad_parts = []
    metadata_parts = []
    samples = 0
    shards = []
    shard_index = 0

    def flush() -> None:
        nonlocal feature_parts, vad_parts, metadata_parts, samples, shard_index
        if not feature_parts:
            return
        features = torch.cat(feature_parts, dim=0)
        vad = torch.cat(vad_parts, dim=0)
        name = f"shard-{shard_index:06d}.pt"
        metadata = [item for part in metadata_parts for item in part]
        torch.save({"features": features, "vad": vad, "metadata": metadata}, target / name)
        shards.append(name)
        samples += int(features.shape[0])
        shard_index += 1
        feature_parts, vad_parts, metadata_parts = [], [], []

    pending = 0
    for relative in manifest["shards"]:
        shard = torch.load(source / relative, map_location="cpu", weights_only=False)
        offset = 0
        while offset < int(shard["features"].shape[0]):
            take = min(batch_size - pending, int(shard["features"].shape[0]) - offset)
            feature_parts.append(shard["features"][offset : offset + take])
            vad_parts.append(shard["vad"][offset : offset + take])
            metadata_parts.append(shard["metadata"][offset : offset + take])
            pending += take
            offset += take
            if pending == batch_size:
                flush()
                pending = 0
    if pending:
        flush()
    output = dict(manifest)
    output.update({"shards": shards, "samples": samples, "head_batch_size": batch_size, "repacked_from": str(source.resolve()), "status": "pass"})
    (target / "manifest.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(repack(Path(args.source), Path(args.target), args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
