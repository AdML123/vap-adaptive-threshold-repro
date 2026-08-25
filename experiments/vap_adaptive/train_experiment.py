"""Training wrapper for strict adaptive VAP targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

torch.set_float32_matmul_precision("medium")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vap.model import VapGPT, VapConfig
from experiments.vap_adaptive.labeling import make_labels


def _limit(value: str):
    return int(value) if value.isdigit() else float(value)


def training_batch_size(config: Mapping, full_model: bool) -> int:
    key = "full_model_batch_size" if full_model else "head_batch_size"
    return int(config.get(key, config.get("batch_size", 1)))


class AdaptiveVAPModule(pl.LightningModule):
    def __init__(
        self,
        model: VapGPT,
        mode: str = "head_only",
        label_mode: str = "standard",
        lam: float = 0.0,
        tau_floor: float = 0.1,
        fixed_thresholds=None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
    ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.label_mode = label_mode
        self.lam = float(lam)
        self.tau_floor = float(tau_floor)
        self.fixed_thresholds = fixed_thresholds
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.scheduler_factor = float(scheduler_factor)
        self.scheduler_patience = int(scheduler_patience)
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.mode == "full")
        for parameter in self.model.objective.parameters():
            parameter.requires_grad_(False)
        if self.mode == "head_only":
            for parameter in self.model.vap_head.parameters():
                parameter.requires_grad_(True)
        elif self.mode != "full":
            raise ValueError("mode must be 'head_only' or 'full'")

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]

    def measured_head_metadata(self) -> dict[str, int]:
        head = self.model.vap_head
        return {
            "head_in_features": int(head.in_features),
            "head_out_features": int(head.out_features),
            "head_parameters": int(sum(p.numel() for p in head.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }

    def forward(self, waveform: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.model(waveform)

    def target_labels(self, vad: torch.Tensor) -> torch.Tensor:
        return make_labels(
            vad,
            mode=self.label_mode,
            lam=self.lam,
            tau_floor=self.tau_floor,
            fixed_thresholds=self.fixed_thresholds,
        )

    def shared_step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "features" in batch:
            labels = self.target_labels(batch["vad"])
            logits = self.model.vap_head(batch["features"])
            vap_loss = self.model.objective.loss_vap(logits, labels)
            return {
                "loss": vap_loss,
                "vap_loss": vap_loss,
                "vad_loss": vap_loss.detach().new_zeros(()),
                "labels": labels,
                "logits": logits,
            }
        if self.mode == "head_only":
            # Frozen features need no autograd graph; reapply only the trainable VAP head.
            with torch.no_grad():
                x1, x2 = self.model.encode_audio(batch["waveform"])
                o1 = self.model.ar_channel(x1)
                o2 = self.model.ar_channel(x2)
                frozen_out = self.model.ar(o1["x"], o2["x"])
                vad = torch.cat((self.model.va_classifier(frozen_out["x1"]), self.model.va_classifier(frozen_out["x2"])), dim=-1)
            out = {"vad": vad, "logits": self.model.vap_head(frozen_out["x"].detach())}
        else:
            out = self.model(batch["waveform"])
        labels = self.target_labels(batch["vad"])
        vap_loss = self.model.objective.loss_vap(out["logits"], labels)
        vad_target = batch["vad"][:, : out["vad"].shape[1]]
        vad_loss = self.model.objective.loss_vad(out["vad"], vad_target)
        return {"loss": vap_loss + vad_loss, "vap_loss": vap_loss, "vad_loss": vad_loss, "labels": labels, "logits": out["logits"]}

    @staticmethod
    def state_diagnostics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = labels.detach().long().reshape(-1)
        counts = torch.bincount(flat, minlength=256).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log2()).sum()
        # Chunk over time to keep diagnostic softmax temporaries small on 8-GB GPUs.
        entropy_sum = logits.detach().new_zeros(())
        element_count = 0
        for chunk in logits.detach().float().split(32, dim=1):
            softmax = chunk.softmax(dim=-1)
            entropy_sum = entropy_sum - (softmax.clamp_min(1e-8) * softmax.clamp_min(1e-8).log2()).sum()
            element_count += int(softmax.numel() // softmax.shape[-1])
        softmax_entropy = entropy_sum / max(1, element_count)
        return {
            "target_entropy": entropy,
            "softmax_entropy": softmax_entropy,
            "unseen_states": (counts == 0).sum().float(),
        }

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        batch_size = batch["features"].shape[0] if "features" in batch else batch["waveform"].shape[0]
        self.log("train_loss", out["loss"], prog_bar=True, batch_size=batch_size)
        self.log("train_vap_loss", out["vap_loss"], batch_size=batch_size)
        self.log("train_vad_loss", out["vad_loss"], batch_size=batch_size)
        diagnostics = self.state_diagnostics(out["labels"], out["logits"])
        self.log("train_target_entropy", diagnostics["target_entropy"], batch_size=batch_size)
        self.log("train_softmax_entropy", diagnostics["softmax_entropy"], batch_size=batch_size)
        self.log("train_unseen_states", diagnostics["unseen_states"], batch_size=batch_size)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        batch_size = batch["features"].shape[0] if "features" in batch else batch["waveform"].shape[0]
        self.log("val_loss", out["loss"], prog_bar=True, batch_size=batch_size)
        self.log("val_vap_loss", out["vap_loss"], batch_size=batch_size)
        diagnostics = self.state_diagnostics(out["labels"], out["logits"])
        self.log("val_target_entropy", diagnostics["target_entropy"], batch_size=batch_size)
        self.log("val_softmax_entropy", diagnostics["softmax_entropy"], batch_size=batch_size)
        self.log("val_unseen_states", diagnostics["unseen_states"], batch_size=batch_size)
        return out

    def on_after_backward(self) -> None:
        norm = torch.zeros((), device=self.device)
        for parameter in self.parameters():
            if parameter.grad is not None:
                norm = norm + parameter.grad.detach().pow(2).sum()
        self.log("grad_norm", norm.sqrt(), on_step=True, on_epoch=False)

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("no trainable parameters")
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


def build_model(checkpoint: str | Path | None = None, full_model: bool = False) -> VapGPT:
    model = VapGPT(VapConfig(load_pretrained=0, freeze_encoder=0 if full_model else 1))
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        translated = {}
        for key, value in state.items():
            key = key.replace("net.", "").replace("vap_head.projection_head", "vap_head")
            if "objective.codebook" in key or "VAP.codebook" in key:
                continue
            translated[key] = value
        model.load_state_dict(translated, strict=False)
    return model


def validated_existing_run(existing: Mapping, qa: Mapping, requested_epochs: int) -> bool:
    checkpoint = existing.get("checkpoint")
    return bool(
        existing.get("max_epochs", 0) >= requested_epochs
        and int(existing.get("completed_steps", 0)) > requested_epochs
        and checkpoint
        and Path(checkpoint).exists()
        and qa.get("status") == "pass"
    )


def run_setting(
    config: Mapping,
    setting: str,
    seed: int,
    max_epochs: int | None = None,
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
    lam: float | None = None,
    fixed_thresholds=None,
    run_name: str | None = None,
    tau_floor: float | None = None,
    learning_rate: float | None = None,
) -> dict:
    from torch.utils.data import DataLoader
    from experiments.vap_adaptive.data import SwitchboardDataset
    from experiments.vap_adaptive.feature_cache import CachedBatchDataset, FeatureCacheDataset, cached_collate, cache_ready

    settings = {
        "standard": {"label_mode": "standard", "full_model": False},
        "uniform_lower": {"label_mode": "uniform_lower", "full_model": False},
        "fixed_per_bin": {"label_mode": "fixed_per_bin", "full_model": False},
        "adaptive_forward": {"label_mode": "forward", "full_model": False},
        "adaptive_reverse": {"label_mode": "reverse", "full_model": False},
        "extreme": {"label_mode": "extreme", "full_model": False},
        "full_standard": {"label_mode": "standard", "full_model": True},
        "full_forward": {"label_mode": "forward", "full_model": True},
    }
    if setting not in settings:
        raise ValueError(f"unknown setting: {setting}")
    spec = settings[setting]
    if spec["label_mode"] in {"forward", "reverse"} and lam is None:
        raise ValueError("adaptive settings require an explicitly selected lambda")
    if spec["label_mode"] == "fixed_per_bin" and fixed_thresholds is None:
        raise ValueError("fixed_per_bin requires an explicitly selected threshold tuple")
    run_name = run_name or setting
    run_dir = Path(config["output_dir"]) / run_name / str(seed)
    requested_epochs = int(max_epochs or config.get("head_epochs", 20))
    existing_manifest_path = run_dir / "run_manifest.json"
    if existing_manifest_path.exists():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if (run_dir / "qa.json").exists():
            existing_qa = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
            if validated_existing_run(existing, existing_qa, requested_epochs):
                return existing
    pl.seed_everything(seed, workers=True)
    model = build_model(config["checkpoint"], full_model=spec["full_model"])
    wrapper = AdaptiveVAPModule(
        model,
        mode="full" if spec["full_model"] else "head_only",
        label_mode=spec["label_mode"],
        lam=float(lam or 0.0),
        tau_floor=float(config.get("tau_floor", 0.1) if tau_floor is None else tau_floor),
        fixed_thresholds=fixed_thresholds,
        learning_rate=float(config.get("head_learning_rate", 1e-4) if learning_rate is None else learning_rate),
        weight_decay=float(config.get("weight_decay", 1e-3)),
        scheduler_factor=float(config.get("scheduler_factor", 0.5)),
        scheduler_patience=int(config.get("scheduler_patience", 2)),
    )
    data_kwargs = {
        "window_seconds": float(config.get("window_seconds", 20.0)),
        "stride_seconds": float(config.get("stride_seconds", 20.0)),
    }
    use_feature_cache = not spec["full_model"] and cache_ready(config, "train") and cache_ready(config, "val")
    batch_size = (
        int(config.get("feature_cache_train_batch_size", 256))
        if use_feature_cache
        else training_batch_size(config, spec["full_model"])
    )
    train_dataset = (
        CachedBatchDataset(config, "train", batch_size)
        if use_feature_cache
        else SwitchboardDataset(config, "train", **data_kwargs)
    )
    val_dataset = (
        CachedBatchDataset(config, "val", batch_size)
        if use_feature_cache
        else SwitchboardDataset(config, "val", **data_kwargs)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None if use_feature_cache else batch_size,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=None if use_feature_cache else batch_size,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=None,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=run_dir, save_last=True, monitor="val_loss", mode="min")
    logger = CSVLogger(save_dir=run_dir, name="logs", version="")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=requested_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        logger=logger,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)
    checkpoint_path = checkpoint.best_model_path or checkpoint.last_model_path
    completed_steps = int(trainer.global_step)
    head_metadata = wrapper.measured_head_metadata()
    trainable_parameter_names = wrapper.trainable_parameter_names()
    # Eager feature caches can occupy >20 GB; release each run before the next seed.
    del trainer, train_loader, val_loader, train_dataset, val_dataset, wrapper
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint_sha256 = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() if checkpoint_path else None
    tie_audit_path = config.get("tie_audit")
    tie_audit_sha256 = hashlib.sha256(Path(tie_audit_path).read_bytes()).hexdigest() if tie_audit_path and Path(tie_audit_path).exists() else None
    manifest = {
        "setting": setting,
        "run_name": run_name,
        "seed": int(seed),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "tie_audit": tie_audit_path,
        "tie_audit_sha256": tie_audit_sha256,
        "head": head_metadata,
        "trainable_parameter_names": trainable_parameter_names,
        "max_epochs": requested_epochs,
        "completed_steps": completed_steps,
        "vad_source": config["vad_source"],
        "comparison": config["comparison"],
        "selection_seed": int(config.get("selection_seed", 1)),
        "selection_provenance": "validation-only; freeze for formal seeds",
        "tau_floor": float(config.get("tau_floor", 0.1) if tau_floor is None else tau_floor),
        "learning_rate": float(config.get("head_learning_rate", 1e-4) if learning_rate is None else learning_rate),
        "batch_size": batch_size,
        "feature_cache_used": use_feature_cache,
        "feature_cache_dir": str(config.get("feature_cache_dir", "")) if use_feature_cache else None,
        "matmul_precision": config.get("matmul_precision", "medium"),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    }
    (run_dir / "config.json").write_text(json.dumps(dict(config), indent=2, sort_keys=True), encoding="utf-8")
    try:
        from experiments.vap_adaptive.gates import environment_metadata

        (run_dir / "environment.json").write_text(json.dumps(environment_metadata(), indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        manifest["environment_error"] = str(exc)
    (run_dir / "qa.json").write_text(json.dumps({"status": "pass" if checkpoint_path else "failed", "failure_reasons": [] if checkpoint_path else ["checkpoint was not written"]}, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_formal_matrix(
    config: Mapping,
    selection_json: str | Path,
    max_epochs: int | None = None,
    limit_train_batches: int | float = 1.0,
    limit_val_batches: int | float = 1.0,
) -> list[dict]:
    selection = json.loads(Path(selection_json).read_text(encoding="utf-8"))
    lam = selection.get("selected_lambda")
    fixed = selection.get("selected_fixed_thresholds")
    if lam is None or fixed is None:
        raise ValueError("formal matrix requires selected lambda and fixed thresholds")
    seeds = tuple(int(seed) for seed in config.get("formal_seeds", (1, 2, 3)))
    manifests = []
    for setting in ("standard", "uniform_lower", "fixed_per_bin", "adaptive_forward", "adaptive_reverse", "extreme"):
        for seed in seeds:
            manifest = run_setting(
                config,
                setting,
                seed,
                max_epochs=max_epochs,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                lam=float(lam) if setting in {"adaptive_forward", "adaptive_reverse"} else None,
                fixed_thresholds=tuple(fixed) if setting == "fixed_per_bin" else None,
            )
            manifests.append(manifest)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--setting", default="standard")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=_limit, default=None)
    parser.add_argument("--limit-val-batches", type=_limit, default=None)
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument("--fixed-thresholds", type=float, nargs=4, default=None)
    parser.add_argument("--tau-floor", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--formal-matrix", action="store_true")
    parser.add_argument("--selection-json", default=None)
    args = parser.parse_args()
    from experiments.vap_adaptive.gates import load_config

    config, _ = load_config(args.config)
    if args.dry_run:
        model = build_model(config["checkpoint"], full_model=False)
        print(json.dumps({"head": AdaptiveVAPModule(model).measured_head_metadata()}, indent=2))
        return
    if args.formal_matrix:
        selection_json = args.selection_json or str(Path(config["output_dir"]) / "calibration" / "stage2_selection.json")
        parse_limit = lambda value: int(value) if isinstance(value, str) and value.isdigit() else value
        train_limit = 1.0 if args.limit_train_batches is None else args.limit_train_batches
        val_limit = 1.0 if args.limit_val_batches is None else args.limit_val_batches
        manifests = run_formal_matrix(config, selection_json, max_epochs=args.max_epochs, limit_train_batches=parse_limit(train_limit), limit_val_batches=parse_limit(val_limit))
        print(json.dumps({"runs": len(manifests), "selection_json": selection_json}, indent=2))
        return
    train_limit = 1 if args.limit_train_batches is None else args.limit_train_batches
    val_limit = 1 if args.limit_val_batches is None else args.limit_val_batches
    result = run_setting(
        config,
        args.setting,
        args.seed,
        max_epochs=args.max_epochs,
        limit_train_batches=train_limit,
        limit_val_batches=val_limit,
        lam=args.lam,
        fixed_thresholds=args.fixed_thresholds,
        tau_floor=args.tau_floor,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
