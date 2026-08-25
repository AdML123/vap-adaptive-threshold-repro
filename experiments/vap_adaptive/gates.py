"""Pre-training provenance and model checkpoint gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


class GateError(RuntimeError):
    pass


def load_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    for key in ("checkpoint", "alignment_root", "audio_root", "damsl_root", "preexperiment_manifest", "tie_audit", "output_dir", "feature_cache_dir"):
        if key in config and config[key] is not None:
            config[key] = str((base / config[key]).resolve())
    return config, config_path


def validate_config(config: dict, base_dir: str | Path) -> None:
    if config.get("vad_source") not in {"original_vap", "model_vad_cache", "proxy_vad"}:
        raise GateError("vad_source must be explicitly selected")
    checkpoint = config.get("checkpoint")
    if not checkpoint or not (Path(base_dir) / checkpoint).exists():
        raise GateError("pretrained checkpoint is missing")
    budget = config.get("approved_epoch_budget")
    if not isinstance(budget, int) or budget <= 0:
        raise GateError("approved_epoch_budget must be a positive integer")
    if config["vad_source"] == "proxy_vad" and not config.get("proxy_vad_approved", False):
        raise GateError("proxy_vad requires explicit proxy_vad_approved=true")


def environment_metadata() -> dict:
    metadata = {
        "python": sys.version.split()[0],
        "vap_dataset_available": importlib.util.find_spec("vap_dataset") is not None,
        "pytorch_lightning_available": importlib.util.find_spec("pytorch_lightning") is not None,
    }
    try:
        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_device_count"] = int(torch.cuda.device_count())
        metadata["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        metadata["torch_error"] = str(exc)
    return metadata


def dry_run(path: str | Path) -> dict:
    config, config_path = load_config(path)
    validate_config(config, config_path.parent)
    checkpoint = inspect_checkpoint(config["checkpoint"])
    env = environment_metadata()
    if config["vad_source"] == "original_vap" and not env["vap_dataset_available"]:
        raise GateError("original_vap requires the external vap_dataset package")
    if config["vad_source"] == "model_vad_cache" and not Path(config.get("model_vad_cache", "")).exists():
        raise GateError("model_vad_cache path is missing")
    return {"config": config, "checkpoint": checkpoint, "environment": env}


def inspect_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise GateError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    weights = None
    for key, value in state.items():
        if key.endswith("vap_head.projection_head.weight") or key.endswith("vap_head.weight"):
            weights = value
            break
    if weights is None or weights.ndim != 2:
        raise GateError("checkpoint has no recognizable VAP head weight")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "head_in_features": int(weights.shape[1]),
        "head_out_features": int(weights.shape[0]),
        "head_parameters": int(weights.numel() + state[next(key for key in state if key.endswith("vap_head.projection_head.bias") or key.endswith("vap_head.bias"))].numel()),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("only --dry-run is supported by the provenance gate")
    try:
        print(json.dumps(dry_run(args.config), indent=2, sort_keys=True))
    except GateError as exc:
        print(f"GATE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
