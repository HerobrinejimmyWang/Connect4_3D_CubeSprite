"""Strict, plan-only contract for the Stage 1 independent scale screen."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from connect4_core.rules import DEFAULT_RULE_REGISTRY

from .config import config_hash, load_config


FORMAT = "stage1_scale_screen_v1"


def _exact(raw: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        missing = sorted(keys.difference(raw) if isinstance(raw, Mapping) else keys)
        extra = sorted(set(raw).difference(keys) if isinstance(raw, Mapping) else ())
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return raw


def load_scale_screen(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    project_root = Path(root).resolve() if root is not None else source.parents[3]
    raw = json.loads(source.read_text(encoding="utf-8"))
    top = _exact(
        raw,
        {
            "format",
            "schema_version",
            "experiment_id",
            "rule_id",
            "levels",
            "frozen_training_contract",
            "policy_target_quality",
            "continuation_rule",
            "readiness",
        },
        "scale screen",
    )
    if top["format"] != FORMAT or top["schema_version"] != 1:
        raise ValueError("unsupported scale-screen contract")
    if not isinstance(top["experiment_id"], str) or not top["experiment_id"]:
        raise ValueError("experiment_id must be non-empty")
    DEFAULT_RULE_REGISTRY.get(top["rule_id"])

    frozen = _exact(
        top["frozen_training_contract"],
        {
            "full_search_sims",
            "fast_search_sims",
            "full_probability",
            "cpuct",
            "virtual_loss",
            "exploration_phases",
            "train_tokens_per_raw_position",
            "learner_batch_size",
            "gate_search_sims",
        },
        "frozen_training_contract",
    )
    integer_fields = (
        "full_search_sims",
        "fast_search_sims",
        "learner_batch_size",
        "gate_search_sims",
    )
    if any(type(frozen[name]) is not int or frozen[name] < 1 for name in integer_fields):
        raise ValueError("frozen integer budgets must be positive integers")

    levels = top["levels"]
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("scale screen requires at least two levels")
    resolved_levels: list[dict[str, Any]] = []
    exploration_contract: list[dict[str, Any]] | None = None
    semantic_learner_contract: dict[str, Any] | None = None
    capacities: list[int] = []
    scale_ids: list[str] = []
    for index, level_raw in enumerate(levels):
        level = _exact(
            level_raw,
            {
                "scale_id",
                "channels",
                "blocks",
                "config",
                "seed_budgets",
                "evaluation_milestones",
                "purpose",
                "topology_status",
            },
            f"levels[{index}]",
        )
        if not isinstance(level["scale_id"], str) or not level["scale_id"]:
            raise ValueError("scale_id must be non-empty")
        if type(level["channels"]) is not int or type(level["blocks"]) is not int:
            raise TypeError("scale capacities must be integers")
        config_path = project_root / level["config"]
        config = load_config(config_path)
        if (config.model.channels, config.model.blocks) != (
            level["channels"],
            level["blocks"],
        ):
            raise ValueError(f"{level['scale_id']} model capacity differs from its config")
        if config.selfplay.rule_id != top["rule_id"]:
            raise ValueError(f"{level['scale_id']} rule differs from scale-screen rule")
        if len(config.selfplay.search_schedule) != 1:
            raise ValueError("scale-screen self-play search must stay fixed")
        search = config.selfplay.search_schedule[0]
        compared = {
            "full_search_sims": search.full_search_sims,
            "fast_search_sims": search.fast_search_sims,
            "full_probability": search.full_probability,
            "cpuct": config.selfplay.cpuct,
            "virtual_loss": config.selfplay.virtual_loss,
            "train_tokens_per_raw_position": config.replay.train_tokens_per_raw_position,
            "learner_batch_size": config.learner.batch_size,
            "gate_search_sims": config.gate.search_schedule[0].search_sims,
        }
        frozen_scalars = dict(frozen)
        declared_exploration = frozen_scalars.pop("exploration_phases")
        if compared != frozen_scalars or len(config.gate.search_schedule) != 1:
            raise ValueError(f"{level['scale_id']} violates the frozen training contract")

        exploration = [asdict(phase) for phase in config.selfplay.exploration_phases]
        if exploration != declared_exploration:
            raise ValueError(
                f"{level['scale_id']} exploration differs from the frozen training contract"
            )
        if exploration_contract is None:
            exploration_contract = exploration
        elif exploration != exploration_contract:
            raise ValueError("scale-screen exploration phases differ between sizes")
        learner = asdict(config.learner)
        learner.pop("max_optimizer_steps_per_cycle")
        if semantic_learner_contract is None:
            semantic_learner_contract = learner
        elif learner != semantic_learner_contract:
            raise ValueError("scale-screen learner semantics differ between sizes")

        seed_budgets = level["seed_budgets"]
        if not isinstance(seed_budgets, list) or not seed_budgets:
            raise ValueError("each scale needs at least one seed budget")
        seen_seeds: set[int] = set()
        for budget_index, budget_raw in enumerate(seed_budgets):
            budget = _exact(
                budget_raw,
                {"seed", "max_train_positions", "condition"},
                f"levels[{index}].seed_budgets[{budget_index}]",
            )
            if type(budget["seed"]) is not int or budget["seed"] < 0:
                raise ValueError("scale-screen seeds must be non-negative integers")
            if budget["seed"] in seen_seeds:
                raise ValueError("scale-screen seed budgets must be unique within a level")
            if type(budget["max_train_positions"]) is not int or budget["max_train_positions"] < 1:
                raise ValueError("seed max_train_positions must be positive")
            if not isinstance(budget["condition"], str) or not budget["condition"]:
                raise ValueError("seed condition must be non-empty")
            seen_seeds.add(budget["seed"])
        milestones = level["evaluation_milestones"]
        if (
            not isinstance(milestones, list)
            or not milestones
            or any(type(value) is not int or value < 1 for value in milestones)
            or milestones != sorted(set(milestones))
            or milestones[-1] > seed_budgets[0]["max_train_positions"]
        ):
            raise ValueError("evaluation milestones must be increasing within the primary budget")
        capacities.append(level["channels"] * level["blocks"])
        scale_ids.append(level["scale_id"])
        resolved_levels.append(
            {
                **dict(level),
                "config_path": str(config_path.resolve()),
                "config_hash": config_hash(config),
            }
        )
    if len(set(scale_ids)) != len(scale_ids) or capacities != sorted(set(capacities)):
        raise ValueError("scale levels must be unique and strictly capacity-ordered")

    quality = _exact(
        top["policy_target_quality"],
        {
            "primary_search_sims",
            "reference_search_sims",
            "repeat_primary_search",
            "fixed_position_source",
            "metrics",
            "escalation_rule",
        },
        "policy_target_quality",
    )
    if quality["primary_search_sims"] != frozen["gate_search_sims"]:
        raise ValueError("policy target primary sims must match the 256-sim evaluation budget")
    if type(quality["reference_search_sims"]) is not int or quality["reference_search_sims"] <= quality["primary_search_sims"]:
        raise ValueError("policy target reference budget must exceed the primary budget")
    if quality["repeat_primary_search"] is not True:
        raise ValueError("policy target diagnostics require a primary-budget repeat")

    return {
        **json.loads(json.dumps(raw)),
        "source_path": str(source),
        "project_root": str(project_root),
        "levels_resolved": resolved_levels,
        "formal_run_enabled": False,
        "run_count": sum(len(level["seed_budgets"]) for level in levels),
    }


__all__ = ["FORMAT", "load_scale_screen"]
