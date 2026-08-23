"""Diagnostics for judging whether an MCTS policy-target budget is adequate."""

from __future__ import annotations

from typing import Any

import numpy as np

from .replay import ReplayShard, SEARCH_FULL


def summarize_visit_targets(
    visit_counts: np.ndarray,
    *,
    expected_simulations: int | None = None,
) -> dict[str, Any]:
    counts = np.asarray(visit_counts)
    if counts.ndim != 2 or counts.shape[1] != 25:
        raise ValueError("visit_counts must have shape [N, 25]")
    if counts.shape[0] < 1 or np.any(counts < 0):
        raise ValueError("visit_counts must contain non-negative targets")
    totals = counts.sum(axis=1, dtype=np.float64)
    if np.any(totals <= 0.0):
        raise ValueError("every policy target must contain at least one visit")
    probabilities = counts.astype(np.float64) / totals[:, None]
    positive = probabilities > 0.0
    entropy = -np.sum(
        np.where(positive, probabilities * np.log(np.where(positive, probabilities, 1.0)), 0.0),
        axis=1,
    )
    ordered = np.sort(probabilities, axis=1)
    top1 = ordered[:, -1]
    top2 = ordered[:, -2]
    support = np.sum(counts > 0, axis=1)
    result: dict[str, Any] = {
        "positions": int(len(counts)),
        "visit_total": {
            "mean": float(totals.mean()),
            "min": int(totals.min()),
            "max": int(totals.max()),
        },
        "target_entropy": {
            "mean": float(entropy.mean()),
            "normalized_log25_mean": float((entropy / np.log(25.0)).mean()),
        },
        "effective_action_count_mean": float(np.exp(entropy).mean()),
        "positive_visit_actions_mean": float(support.mean()),
        "single_action_target_rate": float(np.mean(support == 1)),
        "top1_probability_mean": float(top1.mean()),
        "top1_top2_gap_mean": float((top1 - top2).mean()),
    }
    if expected_simulations is not None:
        if expected_simulations < 1:
            raise ValueError("expected_simulations must be positive")
        result["expected_simulations"] = int(expected_simulations)
        result["exact_budget_fraction"] = float(np.mean(totals == expected_simulations))
    return result


def compare_visit_targets(
    candidate_visit_counts: np.ndarray,
    reference_visit_counts: np.ndarray,
) -> dict[str, Any]:
    """Compare a proposed budget (for example 256) with a larger reference.

    This is diagnostic evidence, not an automatic strength decision.  The same
    fixed positions, rules, model checkpoint, cpuct and lane semantics must be
    used for both arrays.
    """

    candidate = np.asarray(candidate_visit_counts, dtype=np.float64)
    reference = np.asarray(reference_visit_counts, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1] != 25:
        raise ValueError("candidate and reference targets must share shape [N, 25]")
    if len(candidate) < 1 or np.any(candidate < 0.0) or np.any(reference < 0.0):
        raise ValueError("target comparisons require non-negative rows")
    candidate_totals = candidate.sum(axis=1)
    reference_totals = reference.sum(axis=1)
    if np.any(candidate_totals <= 0.0) or np.any(reference_totals <= 0.0):
        raise ValueError("target comparisons require non-empty rows")
    p = candidate / candidate_totals[:, None]
    q = reference / reference_totals[:, None]
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        positive = left > 0.0
        ratio = np.ones_like(left)
        np.divide(left, right, out=ratio, where=positive)
        return np.sum(
            np.where(positive, left * np.log(ratio), 0.0),
            axis=1,
        )

    js = 0.5 * (kl(p, midpoint) + kl(q, midpoint))
    candidate_top = np.argmax(p, axis=1)
    reference_top = np.argmax(q, axis=1)
    row_indices = np.arange(len(p))
    top1_reference_regret = q[row_indices, reference_top] - q[row_indices, candidate_top]
    return {
        "positions": int(len(p)),
        "top1_agreement": float(np.mean(candidate_top == reference_top)),
        "mean_total_variation": float((0.5 * np.abs(p - q).sum(axis=1)).mean()),
        "mean_jensen_shannon_divergence": float(js.mean()),
        "mean_reference_top1_regret": float(top1_reference_regret.mean()),
        "diagnostic_only": True,
    }


def summarize_replay_policy_targets(
    replay: ReplayShard,
    *,
    expected_simulations: int,
) -> dict[str, Any]:
    mask = (replay.search_kind == SEARCH_FULL) & (replay.policy_weight > 0.0)
    if not np.any(mask):
        raise ValueError("replay contains no weighted full-search policy targets")
    return summarize_visit_targets(
        replay.visit_counts[mask], expected_simulations=expected_simulations
    )


def compare_replay_policy_targets(
    primary: ReplayShard,
    reference: ReplayShard,
) -> dict[str, Any]:
    primary_mask = (primary.search_kind == SEARCH_FULL) & (primary.policy_weight > 0.0)
    reference_mask = (reference.search_kind == SEARCH_FULL) & (reference.policy_weight > 0.0)
    primary_indices = np.flatnonzero(primary_mask)
    reference_indices = np.flatnonzero(reference_mask)
    if len(primary_indices) != len(reference_indices) or len(primary_indices) < 1:
        raise ValueError("paired replay audits need the same non-empty full-search rows")
    identity_fields = (
        "game_id",
        "turn_index",
        "player_to_move",
        "rule_code",
        "turn_kind",
        "board",
    )
    for field in identity_fields:
        left = getattr(primary, field)[primary_indices]
        right = getattr(reference, field)[reference_indices]
        if not np.array_equal(left, right):
            raise ValueError(f"paired replay policy targets differ in {field}")
    return compare_visit_targets(
        primary.visit_counts[primary_indices],
        reference.visit_counts[reference_indices],
    )


__all__ = [
    "compare_replay_policy_targets",
    "compare_visit_targets",
    "summarize_replay_policy_targets",
    "summarize_visit_targets",
]
