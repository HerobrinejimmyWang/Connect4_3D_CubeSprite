"""Deterministic, checkpointable exploration-window progression."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Iterable

from .config import ExplorationPhaseConfig, SelfPlayConfig
from .formal_state import FormalLoopState
from .stability import GenerationStabilityMetrics


def _relative_range(values: tuple[float, ...]) -> float:
    mean = sum(values) / len(values)
    return (max(values) - min(values)) / max(abs(mean), 1e-12)


def prepare_dynamic_exploration(
    selfplay: SelfPlayConfig,
    state: FormalLoopState,
    history: Iterable[GenerationStabilityMetrics],
) -> tuple[SelfPlayConfig, FormalLoopState, dict[str, Any]]:
    """Select the next generation's effective exploration phases.

    A transition is evaluated only at a committed generation boundary.  The
    resulting state is saved in that generation's checkpoint, so a retry or
    resume cannot make a different choice from partially written metrics.
    """

    dynamic = selfplay.dynamic_exploration
    if not dynamic.enabled:
        return selfplay, state, {
            "enabled": False,
            "stage_index": 0,
            "high_exploration_plies": selfplay.exploration_phases[1].start_ply,
            "transitioned": False,
            "reason": "disabled",
        }
    if state.exploration_stage_index >= len(dynamic.high_exploration_plies):
        raise ValueError("checkpoint dynamic exploration stage exceeds configured stages")

    stage_index = state.exploration_stage_index
    stage_age = state.next_generation - state.exploration_stage_started_generation
    stage_rows = tuple(
        row for row in history if row.generation >= state.exploration_stage_started_generation
    )
    window = stage_rows[-dynamic.stability_window_generations :]
    evidence: dict[str, Any] = {
        "stage_age_generations": stage_age,
        "window_generations": [row.generation for row in window],
    }
    transitioned = False
    reason = "final_stage"
    if stage_index < len(dynamic.high_exploration_plies) - 1:
        if stage_age < dynamic.min_generations_per_stage:
            reason = "minimum_stage_retention"
        elif len(window) < dynamic.stability_window_generations:
            reason = "insufficient_stability_window"
        elif any(row.mean_policy_entropy is None for row in window):
            reason = "missing_policy_entropy"
        else:
            mean_lengths = tuple(row.mean_game_length for row in window)
            short_rates = tuple(row.short_game_rate for row in window)
            entropies = tuple(float(row.mean_policy_entropy) for row in window)
            evidence.update(
                {
                    "mean_game_length_relative_range": _relative_range(mean_lengths),
                    "short_game_rate_absolute_range": max(short_rates) - min(short_rates),
                    "policy_entropy_relative_range": _relative_range(entropies),
                }
            )
            stable = (
                evidence["mean_game_length_relative_range"]
                <= dynamic.mean_game_length_relative_range
                and evidence["short_game_rate_absolute_range"]
                <= dynamic.short_game_rate_absolute_range
                and evidence["policy_entropy_relative_range"]
                <= dynamic.policy_entropy_relative_range
            )
            if stable:
                state = state.advance_exploration_stage()
                stage_index = state.exploration_stage_index
                transitioned = True
                reason = "stable_window"
            else:
                reason = "distribution_not_stable"

    high_plies = dynamic.high_exploration_plies[stage_index]
    phases = (
        selfplay.exploration_phases[0],
        replace(selfplay.exploration_phases[1], start_ply=high_plies),
        *selfplay.exploration_phases[2:],
    )
    effective = replace(selfplay, exploration_phases=phases)
    decision = {
        "enabled": True,
        "stage_index": stage_index,
        "high_exploration_plies": high_plies,
        "transitioned": transitioned,
        "reason": reason,
        "effective_phases": [asdict(phase) for phase in phases],
        "thresholds": asdict(dynamic),
        "evidence": evidence,
    }
    return effective, state, decision


__all__ = ["prepare_dynamic_exploration"]
