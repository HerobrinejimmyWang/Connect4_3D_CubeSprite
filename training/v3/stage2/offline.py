"""Deterministic offline learner and cross-regime evaluator."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..config import ModelConfig, load_config, model_config_dict
from ..learner import OnlineD4Dataset, V3Learner, build_adamw
from ..model import build_model
from ..pipeline import _evaluate_validation
from ..replay import load_replay_shard, sha256_file
from .calibration import estimate_search_macs, parameter_count


REGIMES = {"standard_early", "standard_mid", "standard_late", "mixed_late"}
RECIPE_BY_REGIME = {
    "standard_early": "b10_standard_v1",
    "standard_mid": "b10_standard_v1",
    "standard_late": "b10_standard_v1",
    "mixed_late": "b10_mixed_opening_position_balanced_v1",
}


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_config(raw: Mapping[str, Any]) -> ModelConfig:
    allowed = {field.name for field in fields(ModelConfig)}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"unknown Stage 2 model fields: {sorted(unknown)}")
    return ModelConfig(**dict(raw))


def _validate_frozen_manifest(manifest: Mapping[str, Any], expected_regime: str) -> str:
    results = manifest.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Stage 2 replay lacks frozen-pool results metadata")
    if results.get("regime") != expected_regime:
        raise ValueError(f"Stage 2 replay regime differs from {expected_regime}")
    recipe = str(results.get("data_recipe_id", ""))
    if recipe != RECIPE_BY_REGIME[expected_regime]:
        raise ValueError(f"Stage 2 replay data recipe differs from {expected_regime}")
    return recipe


def _load_run_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "base_config",
        "model",
        "train_replay",
        "validation_replays",
        "output_dir",
        "seed",
        "target_positions",
        "train_regime",
    }
    missing = required.difference(raw)
    unknown = set(raw).difference(required | {"device", "resume", "warm_start_checkpoint"})
    if missing or unknown:
        raise ValueError(f"Stage 2 run config missing={sorted(missing)} unknown={sorted(unknown)}")
    if set(raw["validation_replays"]) != REGIMES:
        raise ValueError("validation_replays must contain the three standard pools and mixed_late")
    if raw["train_regime"] not in REGIMES:
        raise ValueError("train_regime is not a frozen Stage 2 regime")
    if raw["train_regime"] == "mixed_late" and not raw.get("warm_start_checkpoint"):
        raise ValueError("mixed_late is promotion-only and requires warm_start_checkpoint")
    if raw.get("warm_start_checkpoint") and raw["train_regime"] not in {
        "standard_late",
        "mixed_late",
    }:
        raise ValueError("promotion warm starts may train only standard_late or mixed_late")
    return raw


def train_offline(config_path: str | Path) -> dict[str, Any]:
    raw = _load_run_config(config_path)
    base = load_config(raw["base_config"])
    model_config = _model_config(raw["model"])
    device = str(raw.get("device", base.runtime.device))
    output = Path(raw["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    artifact_path = output / "model.pt"
    train_replay_path = Path(raw["train_replay"]).resolve()
    train_replay, train_manifest = load_replay_shard(train_replay_path, verify_checksum=True)
    train_recipe_id = _validate_frozen_manifest(train_manifest, str(raw["train_regime"]))
    train_sha256 = sha256_file(train_replay_path)
    dataset = OnlineD4Dataset(train_replay, augmentation_seed=int(raw["seed"]))
    torch.manual_seed(int(raw["seed"]))
    np.random.seed(int(raw["seed"]) & 0xFFFFFFFF)
    model = build_model(model_config)
    warm_start_path = str(raw.get("warm_start_checkpoint", ""))
    warm_start_sha256 = sha256_file(warm_start_path) if warm_start_path else ""
    if warm_start_path and not bool(raw.get("resume", False)):
        parent = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        if parent.get("format") not in {
            "connect4-v3-stage2-offline-v1",
            "connect4-v3-model",
        }:
            raise ValueError("unsupported Stage 2 warm-start checkpoint")
        if parent.get("model_config") != model_config_dict(model_config):
            raise ValueError("promotion warm start must keep the exact architecture and scale")
        parent_regime = (
            parent.get("train_regime")
            if parent.get("format") == "connect4-v3-stage2-offline-v1"
            else parent.get("metadata", {}).get("train_regime")
        )
        if parent_regime != "standard_late":
            raise ValueError("promotion warm start must come from standard_late")
        model.load_state_dict(parent["model_state"], strict=True)
    optimizer = build_adamw(
        model,
        learning_rate=base.learner.lr_schedule[0].learning_rate,
        weight_decay=base.learner.weight_decay,
    )
    learner = V3Learner(
        model,
        optimizer,
        device=device,
        batch_size=base.learner.batch_size,
        grad_clip_norm=base.learner.grad_clip_norm,
        sample_seed=int(raw["seed"]),
        amp=base.runtime.learner_amp,
        num_workers=base.runtime.num_workers,
        learning_rate_schedule=tuple(
            (row.start_train_positions, row.learning_rate) for row in base.learner.lr_schedule
        ),
        policy_loss_weight=base.learner.policy_loss_weight,
        wdl_loss_weight=base.learner.wdl_loss_weight,
        opponent_reply_loss_weight=base.learner.opponent_reply_loss_weight,
        future_occupancy_loss_weight=base.learner.future_occupancy_loss_weight,
        moves_left_loss_weight=base.learner.moves_left_loss_weight,
        future_occupancy_class_weights=base.learner.future_occupancy_class_weights,
    )
    if bool(raw.get("resume", False)):
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("format") != "connect4-v3-stage2-offline-v1":
            raise ValueError("unsupported Stage 2 offline checkpoint")
        if saved["model_config"] != model_config_dict(model_config):
            raise ValueError("Stage 2 checkpoint architecture differs from run config")
        if saved["train_replay_sha256"] != train_sha256:
            raise ValueError("Stage 2 checkpoint replay differs from run config")
        if saved.get("warm_start_sha256", "") != warm_start_sha256:
            raise ValueError("Stage 2 checkpoint warm-start source differs from run config")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        learner.load_state_dict(saved["learner_state"])
        learner.scaler.load_state_dict(saved["scaler_state"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if saved.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        np.random.set_state(saved["numpy_rng_state"])

    target_positions = int(raw["target_positions"])
    remaining = target_positions - learner.sample_cursor
    if remaining < 0:
        raise ValueError("target_positions precedes the resumed sample cursor")
    metrics = learner.train_steps(
        dataset,
        steps=(remaining + learner.batch_size - 1) // learner.batch_size,
        position_limit=remaining,
    )
    learner_state = learner.state_dict()
    # Keep the final batch IDs as a compact resume/sampling audit.  V3Learner
    # accumulates IDs per train_steps call, whose boundary is operational.
    learner_state["last_sample_ids"] = learner_state["last_sample_ids"][-learner.batch_size :]
    payload = {
        "format": "connect4-v3-stage2-offline-v1",
        "model_config": model_config_dict(model_config),
        "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "learner_state": learner_state,
        "scaler_state": learner.scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "train_replay_sha256": train_sha256,
        "train_regime": str(raw["train_regime"]),
        "train_data_recipe_id": train_recipe_id,
        "warm_start_checkpoint": warm_start_path,
        "warm_start_sha256": warm_start_sha256,
        "target_positions": target_positions,
        "train_metrics": asdict(metrics),
    }
    _atomic_torch_save(checkpoint_path, payload)
    _atomic_torch_save(
        artifact_path,
        {
            "format": "connect4-v3-model",
            "format_version": 1,
            "model_config": model_config_dict(model_config),
            "model_state": payload["model_state"],
            "metadata": {
                "model_id": (
                    f"stage2-offline-{model_config.architecture}-{raw['train_regime']}-"
                    f"s{int(raw['seed'])}-p{target_positions}"
                ),
                "lineage": "v3_stage2_offline",
                "train_regime": str(raw["train_regime"]),
                "seed": int(raw["seed"]),
                "train_positions": target_positions,
                "train_replay_sha256": train_sha256,
                "train_data_recipe_id": train_recipe_id,
                "warm_start_sha256": warm_start_sha256,
            },
        },
    )
    report = evaluate_checkpoint(config_path, checkpoint_path=checkpoint_path, model=model)
    report["train_metrics"] = asdict(metrics)
    report["model_artifact"] = {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
    }
    report["checkpoint_sha256"] = sha256_file(checkpoint_path)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _policy_metrics(model: torch.nn.Module, dataset: OnlineD4Dataset, device: str, batch_size: int) -> dict[str, float]:
    target_top1 = predicted_top1 = agreements = 0
    jsd_sum = 0.0
    policy_rows = 0
    model.eval()
    target_device = torch.device(device)
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            boards = torch.stack([item["board"] for item in items]).to(target_device)
            roles = torch.stack([item["role_to_play"] for item in items]).to(target_device)
            rules = torch.stack([item["rule_features"] for item in items]).to(target_device)
            legal = torch.stack([item["legal_mask"] for item in items]).to(target_device)
            targets = torch.stack([item["policy"] for item in items]).to(target_device)
            weights = torch.stack([item["policy_weight"] for item in items]).to(target_device)
            logits = model.forward_search(boards, role_to_play=roles, rule_features=rules).policy_logits
            predictions = torch.softmax(logits.float().masked_fill(~legal, -torch.inf), dim=1)
            active = weights > 0
            if not active.any():
                continue
            p = targets[active].clamp_min(1e-12)
            q = predictions[active].clamp_min(1e-12)
            midpoint = 0.5 * (p + q)
            jsd_sum += float((0.5 * (p * (p.log() - midpoint.log())).sum(1) + 0.5 * (q * (q.log() - midpoint.log())).sum(1)).sum().cpu())
            agreements += int(p.argmax(1).eq(q.argmax(1)).sum().cpu())
            policy_rows += int(active.sum().cpu())
    return {
        "policy_jsd": jsd_sum / max(policy_rows, 1),
        "policy_top1_agreement": agreements / max(policy_rows, 1),
    }


def evaluate_checkpoint(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    raw = _load_run_config(config_path)
    base = load_config(raw["base_config"])
    model_config = _model_config(raw["model"])
    device = str(raw.get("device", base.runtime.device))
    if model is None:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["model_config"] != model_config_dict(model_config):
            raise ValueError("checkpoint architecture differs from evaluation config")
        model = build_model(model_config)
        model.load_state_dict(saved["model_state"], strict=True)
        model.to(device)
    evaluation_config = replace(base, model=model_config, runtime=replace(base.runtime, device=device))
    reports = {}
    for regime, path in raw["validation_replays"].items():
        replay, manifest = load_replay_shard(path, verify_checksum=True)
        _validate_frozen_manifest(manifest, regime)
        dataset = OnlineD4Dataset(replay, augmentation_seed=int(raw["seed"]))
        metrics = _evaluate_validation(model, dataset, evaluation_config)
        metrics.update(_policy_metrics(model, dataset, device, base.learner.batch_size))
        metrics["replay_sha256"] = sha256_file(path)
        reports[regime] = metrics
    return {
        "schema": "connect4-v3-stage2-offline-report-v2",
        "model": model_config_dict(model_config),
        "seed": int(raw["seed"]),
        "target_positions": int(raw["target_positions"]),
        "train_regime": str(raw["train_regime"]),
        "train_data_recipe_id": RECIPE_BY_REGIME[str(raw["train_regime"])],
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "train_replay_sha256": sha256_file(raw["train_replay"]),
        "warm_start_checkpoint": str(raw.get("warm_start_checkpoint", "")),
        "efficiency": {
            "parameters": parameter_count(model),
            "search_macs_estimate": estimate_search_macs(model),
        },
        "cross_regime": reports,
    }
