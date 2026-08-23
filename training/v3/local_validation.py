"""Bounded local validation for the pre-Stage-1 P6/P7 contracts.

This module is deliberately diagnostic. It does not start self-play, update a
checkpoint, prune an artifact, or enable the guarded formal-run entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from connect4_core.rules import DEFAULT_RULE_REGISTRY

from .config import V3Config, config_hash
from .formal_state import FormalLoopState, PendingCandidateState
from .pipeline import formal_run_status
from .preflight import PreflightReport
from .replay import (
    SEARCH_FAST,
    SEARCH_FULL,
    TURN_FORCED_PASS,
    load_replay_shard,
)
from .retention import (
    ArchiveReceipt,
    ReceiptEntry,
    RetentionArtifact,
    RetentionPolicy,
    plan_retention,
)


LOCAL_VALIDATION_SCHEMA_VERSION = 1
P6_AUXILIARY_WEIGHTS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("baseline", (0.0, 0.0, 0.0)),
    ("opponent_reply", (0.15, 0.0, 0.0)),
    ("future_occupancy", (0.0, 0.15, 0.0)),
    ("moves_left", (0.0, 0.0, 0.05)),
    ("all_auxiliary", (0.15, 0.15, 0.05)),
)
_ALLOWED_ABLATION_DIFFS = frozenset(
    {
        "run.run_id",
        "run.run_dir",
        "learner.opponent_reply_loss_weight",
        "learner.future_occupancy_loss_weight",
        "learner.moves_left_loss_weight",
    }
)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(item, child))
    return result


def build_p6_ablation_configs(base: V3Config) -> tuple[tuple[str, V3Config], ...]:
    """Return the five fixed local P6 variants with an identical budget."""

    variants: list[tuple[str, V3Config]] = []
    for name, weights in P6_AUXILIARY_WEIGHTS:
        run_id = f"{base.run.run_id}_p6_{name}"
        learner = replace(
            base.learner,
            opponent_reply_loss_weight=weights[0],
            future_occupancy_loss_weight=weights[1],
            moves_left_loss_weight=weights[2],
        )
        run = replace(
            base.run,
            run_id=run_id,
            run_dir=str(Path(base.run.run_dir).parent / run_id),
            resume=False,
        )
        variants.append((name, replace(base, run=run, learner=learner)))
    return tuple(variants)


def validate_p6_ablation_matrix(
    base: V3Config, variants: Iterable[tuple[str, V3Config]]
) -> list[dict[str, Any]]:
    """Reject accidental optimizer, data, search, model, or topology drift."""

    base_flat = _flatten(base.to_dict())
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for name, variant in variants:
        if name in names:
            raise ValueError(f"duplicate P6 ablation name: {name}")
        names.add(name)
        variant_flat = _flatten(variant.to_dict())
        changed = sorted(
            key for key in base_flat if base_flat[key] != variant_flat.get(key)
        )
        unexpected = sorted(set(changed).difference(_ALLOWED_ABLATION_DIFFS))
        if unexpected:
            raise ValueError(
                f"P6 variant {name!r} changes non-ablation fields: {unexpected}"
            )
        rows.append(
            {
                "name": name,
                "run_id": variant.run.run_id,
                "config_hash": config_hash(variant),
                "changed_fields": changed,
                "loss_weights": {
                    "policy": variant.learner.policy_loss_weight,
                    "wdl": variant.learner.wdl_loss_weight,
                    "opponent_reply": variant.learner.opponent_reply_loss_weight,
                    "future_occupancy": variant.learner.future_occupancy_loss_weight,
                    "moves_left": variant.learner.moves_left_loss_weight,
                },
            }
        )
    expected_names = [name for name, _weights in P6_AUXILIARY_WEIGHTS]
    if [row["name"] for row in rows] != expected_names:
        raise ValueError("P6 ablation variants are incomplete or out of order")
    return rows


def write_p6_ablation_configs(
    base: V3Config, output_dir: str | Path
) -> tuple[Path, ...]:
    """Write explicit configs without replacing an existing experiment file."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, config in build_p6_ablation_configs(base):
        path = target / f"{name}.json"
        encoded = config.to_json()
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise FileExistsError(f"refusing to overwrite different P6 config: {path}")
        else:
            path.write_text(encoded, encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


def _balanced_class_weights(counts: np.ndarray) -> list[float] | None:
    if counts.shape != (3,) or np.any(counts <= 0):
        return None
    weights = counts.sum(dtype=np.float64) / (3.0 * counts.astype(np.float64))
    return [round(float(value), 8) for value in weights]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        label: float(np.percentile(values, percentile))
        for label, percentile in (("p50", 50), ("p90", 90), ("p99", 99))
    }


def inspect_locked_replay(
    replay_dir: str | Path,
    *,
    minimum_games: int | None = None,
    minimum_samples: int | None = None,
) -> dict[str, Any]:
    """Checksum Replay V2 shards and compute target-coverage calibration facts."""

    if (minimum_games is None) != (minimum_samples is None):
        raise ValueError("minimum_games and minimum_samples must be provided together")
    if minimum_games is not None and (
        type(minimum_games) is not int
        or type(minimum_samples) is not int
        or minimum_games < 1
        or minimum_samples < 1
    ):
        raise ValueError("minimum replay games and samples must be positive integers")
    root = Path(replay_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Replay V2 directory not found: {root}")
    paths = tuple(sorted(root.rglob("*.npz")))
    if not paths:
        raise ValueError(f"no Replay V2 NPZ shards found under {root}")

    fingerprint_rows: list[dict[str, Any]] = []
    sample_count = 0
    game_ids: set[int] = set()
    rule_codes: set[int] = set()
    producer_hashes: set[str] = set()
    registry_hashes: set[str] = set()
    wdl_counts = np.zeros((3,), dtype=np.int64)
    occupancy_counts = np.zeros((3,), dtype=np.int64)
    reply_valid = 0
    pass_count = 0
    full_count = 0
    fast_count = 0
    remaining_parts: list[np.ndarray] = []

    for path in paths:
        shard, manifest = load_replay_shard(path, verify_checksum=True)
        relative = path.relative_to(root).as_posix()
        fingerprint_rows.append(
            {
                "path": relative,
                "checksum_sha256": manifest["checksum_sha256"],
                "sample_count": int(manifest["sample_count"]),
            }
        )
        sample_count += len(shard)
        game_ids.update(int(value) for value in np.unique(shard.game_id))
        rule_codes.update(int(value) for value in np.unique(shard.rule_code))
        producer_hashes.add(str(manifest["config_hash"]))
        registry_hashes.add(str(manifest["rule_registry_hash"]))
        wdl_counts += np.bincount(shard.wdl.astype(np.int64), minlength=3)[:3]
        reply_valid += int(shard.opponent_reply_mask.sum(dtype=np.int64))
        pass_count += int(np.count_nonzero(shard.turn_kind == TURN_FORCED_PASS))
        full_count += int(np.count_nonzero(shard.search_kind == SEARCH_FULL))
        fast_count += int(np.count_nonzero(shard.search_kind == SEARCH_FAST))
        remaining_parts.append(shard.remaining_turns.astype(np.int64, copy=False))

        terminal_canonical = (
            shard.terminal_board * shard.player_to_move[:, np.newaxis, np.newaxis, np.newaxis]
        )
        current_empty = shard.board == 0
        occupancy = np.where(
            terminal_canonical > 0,
            0,
            np.where(terminal_canonical < 0, 1, 2),
        )
        occupancy_counts += np.bincount(
            occupancy[current_empty].astype(np.int64), minlength=3
        )[:3]

    if registry_hashes != {DEFAULT_RULE_REGISTRY.registry_hash}:
        raise ValueError(
            "locked replay uses a different or mixed rule registry: "
            f"{sorted(registry_hashes)}"
        )
    known_codes = {spec.rule_code for spec in DEFAULT_RULE_REGISTRY.specs}
    unknown_codes = sorted(rule_codes.difference(known_codes))
    if unknown_codes:
        raise ValueError(f"locked replay contains unknown rule codes: {unknown_codes}")
    canonical = json.dumps(
        fingerprint_rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    dataset_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    remaining = np.concatenate(remaining_parts)
    occupancy_total = int(occupancy_counts.sum())
    thresholds_configured = minimum_games is not None
    ready_for_screening = bool(
        thresholds_configured
        and len(game_ids) >= minimum_games
        and sample_count >= minimum_samples
        and full_count > 0
        and reply_valid > 0
        and np.all(occupancy_counts > 0)
    )
    return {
        "status": "locked_and_validated",
        "integrity_passed": True,
        "ready_for_p6_screening": ready_for_screening,
        "screening_thresholds": (
            {"minimum_games": minimum_games, "minimum_samples": minimum_samples}
            if thresholds_configured
            else None
        ),
        "screening_readiness_note": (
            "Explicit minimums were met and all auxiliary target families have coverage."
            if ready_for_screening
            else "Integrity passed, but explicit minimums are absent or not yet met."
        ),
        "root": str(root),
        "dataset_sha256": dataset_hash,
        "shards": len(paths),
        "samples": sample_count,
        "games": len(game_ids),
        "rule_codes": sorted(rule_codes),
        "producer_config_hashes": sorted(producer_hashes),
        "rule_registry_hash": next(iter(registry_hashes)),
        "coverage": {
            "full_policy_samples": full_count,
            "fast_policy_samples": fast_count,
            "forced_pass_samples": pass_count,
            "opponent_reply_samples": reply_valid,
            "wdl_counts": wdl_counts.tolist(),
            "future_occupancy_counts_own_opponent_empty": occupancy_counts.tolist(),
            "future_occupancy_fractions": (
                [round(float(value) / occupancy_total, 8) for value in occupancy_counts]
                if occupancy_total
                else [0.0, 0.0, 0.0]
            ),
            "remaining_turns": _percentiles(remaining),
        },
        "suggested_future_occupancy_class_weights": _balanced_class_weights(
            occupancy_counts
        ),
        "class_weight_note": (
            "Balanced inverse-frequency suggestion normalized to mean contribution 1; "
            "review clipping and validation behavior before freezing it in a formal config."
        ),
        "shard_fingerprint_rows": fingerprint_rows,
    }


def _validate_p7_local_primitives(config: V3Config) -> dict[str, Any]:
    state = FormalLoopState(train_positions_consumed=8).emit_candidate(
        PendingCandidateState(
            candidate_model_id="local-candidate",
            candidate_path="candidates/local-candidate.pt",
            incumbent_model_id="random",
            gate_path="metrics/local-gate.json",
            opening_manifest="manifests/local-openings.json",
            pairs_evaluated=2,
            max_pairs=4,
        )
    )
    state_round_trip = FormalLoopState.from_dict(state.to_dict()) == state

    checksum = "a" * 64
    artifact = RetentionArtifact(
        path="replay/raw/local.npz",
        kind="raw_replay",
        size_bytes=128,
        checksum_sha256=checksum,
        sequence=0,
        prunable=True,
        position_start=0,
        position_end=1,
    )
    bad_receipt = ArchiveReceipt(
        receipt_id="local-mismatch",
        archive_manifest_sha256="b" * 64,
        verified=True,
        entries=(ReceiptEntry(artifact.path, artifact.size_bytes + 1, checksum),),
    )
    retention = plan_retention(
        (artifact,),
        RetentionPolicy(keep_recent_by_kind=(), hard_free_bytes=0),
        active_window_start=2,
        active_window_end=2,
        receipts=(bad_receipt,),
    )
    checks = {
        "formal_state_round_trip": state_round_trip,
        "archive_receipt_mismatch_protected": not retention.eligible_paths,
        "formal_run_still_guarded": formal_run_status(config).get("production_ready") is False,
    }
    formal = formal_run_status(config)
    return {
        "local_primitives_passed": all(checks.values()),
        "checks": checks,
        "implemented_but_not_connected": [
            "candidate cadence and inconclusive-gate state transitions",
            "pure archive receipt and retention planning",
            "bounded hardware topology planning",
        ],
        "remaining_local_connection_work": [
            "atomic generation draft/commit/reconcile journal",
            "OS-level single-coordinator lock",
            "coordinated actor/inference/learner drain and restart",
            "archive catalog plus verified transfer receipt ingestion",
            "explicit revalidated prune command and disk-watermark backpressure",
        ],
        "formal_blockers": list(formal["blocking_items"]),
    }


def build_local_validation_report(
    config: V3Config,
    preflight: PreflightReport,
    *,
    replay_dir: str | Path | None = None,
    minimum_replay_games: int | None = None,
    minimum_replay_samples: int | None = None,
) -> dict[str, Any]:
    """Build a self-contained report without running training."""

    if replay_dir is None and (
        minimum_replay_games is not None or minimum_replay_samples is not None
    ):
        raise ValueError("replay minimums require replay_dir")
    if config.selfplay.rule_registry_hash != DEFAULT_RULE_REGISTRY.registry_hash:
        raise ValueError("config rule_registry_hash differs from the executable registry")
    DEFAULT_RULE_REGISTRY.get(config.selfplay.rule_id)
    ablation_rows = validate_p6_ablation_matrix(
        config, build_p6_ablation_configs(config)
    )
    dataset = (
        inspect_locked_replay(
            replay_dir,
            minimum_games=minimum_replay_games,
            minimum_samples=minimum_replay_samples,
        )
        if replay_dir is not None
        else {
            "status": "not_provided",
            "integrity_passed": False,
            "ready_for_p6_screening": False,
            "message": "Pass --replay-dir to lock and inspect a Replay V2 dataset.",
        }
    )
    p7 = _validate_p7_local_primitives(config)
    local_contract_passed = bool(ablation_rows) and p7["local_primitives_passed"]
    return {
        "schema_version": LOCAL_VALIDATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded_local_pre_stage1_validation",
        "safety": {
            "starts_training": False,
            "deletes_artifacts": False,
            "enables_formal_run": False,
        },
        "environment": {
            "python": preflight.python,
            "numpy": preflight.numpy,
            "torch": preflight.torch,
            "validated_device": preflight.device,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "contract": {
            "base_run_id": config.run.run_id,
            "base_config_hash": config_hash(config),
            "rule_id": config.selfplay.rule_id,
            "rule_registry_hash": DEFAULT_RULE_REGISTRY.registry_hash,
            "local_contract_passed": local_contract_passed,
        },
        "p6": {
            "ablation_matrix_valid": True,
            "identical_budget_and_topology": True,
            "variants": ablation_rows,
            "dataset": dataset,
            "local_acceptance_scope": [
                "finite forward/backward and overfit behavior via CPU smoke",
                "target masks, D4 transforms, Replay V2 checksums and coverage",
                "fair five-way auxiliary-head config matrix",
            ],
            "deferred_to_target_gpu": [
                "final batch size, throughput, peak CUDA memory, and queue sizing",
                "search-only actor throughput under target topology",
                "multi-seed closed-loop playing-strength comparison",
            ],
        },
        "p7": p7,
        "result": {
            "local_contract_passed": local_contract_passed,
            "dataset_integrity_passed": bool(dataset["integrity_passed"]),
            "dataset_ready_for_p6_screening": bool(
                dataset["ready_for_p6_screening"]
            ),
            "stage1_ready": False,
            "stage1_ready_reason": (
                "Target-GPU evidence and the remaining P7 connection work are not complete."
            ),
        },
    }


def write_local_validation_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    return target


__all__ = [
    "LOCAL_VALIDATION_SCHEMA_VERSION",
    "P6_AUXILIARY_WEIGHTS",
    "build_local_validation_report",
    "build_p6_ablation_configs",
    "inspect_locked_replay",
    "validate_p6_ablation_matrix",
    "write_local_validation_report",
    "write_p6_ablation_configs",
]
