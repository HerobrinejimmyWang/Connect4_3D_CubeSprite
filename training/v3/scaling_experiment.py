"""Strict plan-only contract for independent and replay-transfer scaling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from connect4_core.rules import DEFAULT_RULE_REGISTRY

from .cross_scale import STRATA, TransferSchedule, TransferStage


SCALING_EXPERIMENT_FORMAT = "dual_track_scaling_v1"
SCALING_EXPERIMENT_SCHEMA_VERSION = 1


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _exact_mapping(raw: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        missing = sorted(keys.difference(raw) if isinstance(raw, Mapping) else keys)
        extra = sorted(set(raw).difference(keys) if isinstance(raw, Mapping) else ())
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return raw


def load_scaling_experiment(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    top = _exact_mapping(
        raw,
        {
            "format",
            "schema_version",
            "experiment_id",
            "rule_id",
            "scale_levels",
            "research_track",
            "production_track",
            "shared_evaluation",
        },
        "scaling experiment",
    )
    if top["format"] != SCALING_EXPERIMENT_FORMAT or top["schema_version"] != 1:
        raise ValueError("unsupported dual-track scaling experiment")
    if not isinstance(top["experiment_id"], str) or not top["experiment_id"]:
        raise ValueError("experiment_id must be a non-empty string")
    rule = DEFAULT_RULE_REGISTRY.get(top["rule_id"])

    levels = top["scale_levels"]
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("scale_levels must contain at least two model sizes")
    level_ids: list[str] = []
    capacities: list[tuple[int, int]] = []
    for index, level_raw in enumerate(levels):
        level = _exact_mapping(
            level_raw,
            {"scale_id", "channels", "blocks", "v3_config", "purpose", "status"},
            f"scale_levels[{index}]",
        )
        if not isinstance(level["scale_id"], str) or not level["scale_id"]:
            raise ValueError("scale_id must be a non-empty string")
        if type(level["channels"]) is not int or type(level["blocks"]) is not int:
            raise TypeError("scale channels and blocks must be integers")
        if level["channels"] < 4 or level["blocks"] < 1:
            raise ValueError("scale channels/blocks are invalid")
        if level["v3_config"] is not None and not isinstance(level["v3_config"], str):
            raise TypeError("v3_config must be a path string or null")
        if level["status"] not in {"configured", "design_pending"}:
            raise ValueError("scale status must be configured or design_pending")
        level_ids.append(level["scale_id"])
        capacities.append((level["channels"], level["blocks"]))
    if len(set(level_ids)) != len(level_ids) or len(set(capacities)) != len(capacities):
        raise ValueError("scale levels must have unique IDs and capacities")
    if capacities != sorted(capacities, key=lambda item: item[0] * item[1]):
        raise ValueError("scale levels must be ordered by increasing channels*blocks")

    research = _exact_mapping(
        top["research_track"],
        {
            "seeds",
            "inherit_weights",
            "inherit_replay",
            "evaluation_milestones",
            "comparison_slices",
            "terminal_rule",
        },
        "research_track",
    )
    if research["inherit_weights"] is not False or research["inherit_replay"] is not False:
        raise ValueError("research scaling must not inherit weights or replay")
    if not isinstance(research["seeds"], list) or not research["seeds"] or any(
        type(seed) is not int or seed < 0 for seed in research["seeds"]
    ):
        raise ValueError("research seeds must be non-negative integers")
    if len(set(research["seeds"])) != len(research["seeds"]):
        raise ValueError("research seeds must be unique")
    if research["evaluation_milestones"] != ["early", "middle", "final"]:
        raise ValueError("research milestones are frozen to early/middle/final")
    required_slices = {"matched_train_positions", "matched_total_compute", "terminal_strength"}
    if set(research["comparison_slices"]) != required_slices:
        raise ValueError("research comparison slices do not match the V1 contract")

    production = _exact_mapping(
        top["production_track"],
        {
            "transfer_weights",
            "bundle_strata",
            "bundle_sampling_weights",
            "offline_bootstrap_train_tokens_per_donor_position",
            "online_transfer_schedule",
            "qualification",
        },
        "production_track",
    )
    if production["transfer_weights"] is not False:
        raise ValueError("production V1 transfers replay only, never weights")
    if production["bundle_strata"] != list(STRATA):
        raise ValueError("production bundle strata must be early/middle/late/strong")
    weights = production["bundle_sampling_weights"]
    if not isinstance(weights, list) or len(weights) != 4 or any(
        not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0
        for weight in weights
    ):
        raise ValueError("bundle sampling weights must contain four positive values")
    if abs(sum(float(weight) for weight in weights) - 1.0) > 1e-9:
        raise ValueError("bundle sampling weights must sum to one")
    token_ratio = production["offline_bootstrap_train_tokens_per_donor_position"]
    if not isinstance(token_ratio, (int, float)) or isinstance(token_ratio, bool) or token_ratio <= 0:
        raise ValueError("offline bootstrap token ratio must be positive")
    schedule_raw = production["online_transfer_schedule"]
    if not isinstance(schedule_raw, list):
        raise TypeError("online_transfer_schedule must be a list")
    decoded_stages: list[TransferStage] = []
    for index, row_raw in enumerate(schedule_raw):
        row = _exact_mapping(
            row_raw,
            {"start_own_positions", "donor_fraction"},
            f"online_transfer_schedule[{index}]",
        )
        if type(row["start_own_positions"]) is not int:
            raise TypeError("transfer start_own_positions must be an integer")
        if isinstance(row["donor_fraction"], bool) or not isinstance(
            row["donor_fraction"], (int, float)
        ):
            raise TypeError("transfer donor_fraction must be numeric")
        decoded_stages.append(
            TransferStage(
                start_own_positions=row["start_own_positions"],
                donor_fraction=float(row["donor_fraction"]),
            )
        )
    schedule = TransferSchedule(tuple(decoded_stages))
    qualification = _exact_mapping(
        production["qualification"],
        {
            "opening_manifest",
            "search_sims",
            "cpuct",
            "initial_pairs",
            "pair_increment",
            "max_pairs",
            "confidence",
            "bootstrap_samples",
            "role_floor",
            "accept_threshold",
        },
        "production_track.qualification",
    )
    if any(type(qualification[key]) is not int or qualification[key] < 1 for key in (
        "search_sims",
        "initial_pairs",
        "pair_increment",
        "max_pairs",
        "bootstrap_samples",
    )):
        raise ValueError("qualification integer budgets must be positive")
    if qualification["initial_pairs"] > qualification["max_pairs"]:
        raise ValueError("qualification initial pairs exceed max pairs")
    if not 0.0 < float(qualification["confidence"]) < 1.0:
        raise ValueError("qualification confidence must be in (0, 1)")
    if not 0.0 <= float(qualification["role_floor"]) <= 1.0:
        raise ValueError("qualification role floor must be in [0, 1]")
    if float(qualification["accept_threshold"]) != 0.5:
        raise ValueError("qualification accept threshold is frozen at 0.5")

    evaluation = _exact_mapping(
        top["shared_evaluation"],
        {"anchored_elo_config", "promotion_gate_separate", "report_role_splits"},
        "shared_evaluation",
    )
    if evaluation["promotion_gate_separate"] is not True or evaluation["report_role_splits"] is not True:
        raise ValueError("shared evaluation must keep Elo separate and report role splits")
    if not isinstance(evaluation["anchored_elo_config"], str) or not evaluation["anchored_elo_config"]:
        raise ValueError("anchored_elo_config must be a non-empty path")

    resolved = json.loads(json.dumps(raw))
    resolved["rule_version"] = rule.rule_version
    resolved["rule_registry_hash"] = DEFAULT_RULE_REGISTRY.registry_hash
    resolved["online_transfer_schedule_resolved"] = [as_stage(stage) for stage in schedule.stages]
    resolved["experiment_hash"] = _canonical_hash(raw)
    resolved["source_path"] = str(source.resolve())
    return resolved


def as_stage(stage: TransferStage) -> dict[str, Any]:
    return {
        "start_own_positions": stage.start_own_positions,
        "donor_fraction": stage.donor_fraction,
    }


def build_scaling_experiment_plan(
    spec: Mapping[str, Any], *, root: str | Path | None = None
) -> dict[str, Any]:
    base = Path(root).resolve() if root is not None else Path.cwd().resolve()
    levels = spec["scale_levels"]
    research_runs = [
        {
            "run_id": f"{spec['experiment_id']}-research-{level['scale_id']}-seed{seed}",
            "scale_id": level["scale_id"],
            "seed": seed,
            "initialization": "random",
            "bootstrap_producer": "random",
            "inherited_weights": False,
            "inherited_replay": False,
            "v3_config": level["v3_config"],
            "config_available": bool(
                level["v3_config"] and (base / level["v3_config"]).is_file()
            ),
            "evaluation_milestones": list(spec["research_track"]["evaluation_milestones"]),
        }
        for level in levels
        for seed in spec["research_track"]["seeds"]
    ]
    transitions = []
    for donor, target in zip(levels, levels[1:]):
        transitions.append(
            {
                "transition_id": f"{donor['scale_id']}-to-{target['scale_id']}",
                "donor_scale": donor["scale_id"],
                "target_scale": target["scale_id"],
                "target_initialization": "random",
                "weight_transfer": False,
                "ordered_phases": [
                    "freeze_accepted_donor_strata",
                    "build_and_verify_cross_scale_replay_bundle_v1",
                    "offline_bootstrap_target_from_donor_only",
                    "paired_role_swapped_donor_qualification",
                    "register_first_target_champion_only_if_accepted",
                    "start_target_champion_selfplay",
                    "mix_donor_and_own_replay_with_position_based_decay",
                    "finish_with_zero_to_five_percent_donor_sentinel",
                    "evaluate_early_middle_final_and_final_512",
                ],
                "qualification": dict(spec["production_track"]["qualification"]),
                "transfer_schedule": list(spec["online_transfer_schedule_resolved"]),
                "required_receipts": [
                    "bundle.ready.json",
                    "offline_bootstrap_checkpoint",
                    "donor_qualification.content_sha256",
                    "first_champion_commit",
                    "transfer_ledger_checkpoint_state",
                ],
            }
        )
    missing_configs = sorted(
        {run["scale_id"] for run in research_runs if not run["config_available"]}
    )
    return {
        "format": SCALING_EXPERIMENT_FORMAT,
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "experiment_hash": spec["experiment_hash"],
        "formal_run_enabled": False,
        "research_track": {
            "purpose": "capacity_and_selfplay_stability",
            "runs": research_runs,
            "comparison_slices": list(spec["research_track"]["comparison_slices"]),
            "terminal_rule": spec["research_track"]["terminal_rule"],
        },
        "production_track": {
            "purpose": "fastest_auditable_strong_model",
            "transitions": transitions,
            "bundle_sampling_weights": dict(
                zip(STRATA, spec["production_track"]["bundle_sampling_weights"], strict=True)
            ),
            "offline_bootstrap_train_tokens_per_donor_position": spec["production_track"][
                "offline_bootstrap_train_tokens_per_donor_position"
            ],
        },
        "shared_evaluation": dict(spec["shared_evaluation"]),
        "readiness": {
            "all_scale_configs_available": not missing_configs,
            "missing_scale_configs": missing_configs,
            "cross_scale_foundation_implemented": True,
            "formal_scheduler_integration_implemented": False,
            "p7_guard_remains": True,
        },
    }


__all__ = [
    "SCALING_EXPERIMENT_FORMAT",
    "build_scaling_experiment_plan",
    "load_scaling_experiment",
]
