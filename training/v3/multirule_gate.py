"""Equal-rule paired evidence for the BAL-5 R2 canary gate.

The caller supplies candidate-versus-incumbent and candidate-versus-each-rule's
retained peak games.  A peak may initially be the incumbent.  This module does
not route self-play or promote a model; the formal runner must publish the
evidence and state atomically before changing its accepted champion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from connect4_core.rules import BAL5_R2_RULE_REGISTRY

from .gate import GateSummary, summarize_paired_results
from .config import GateConfig


BAL5_R2_RULE_IDS = tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs)


@dataclass(frozen=True)
class MultiRuleGateSummary:
    verdict: str
    reason: str
    macro_point_score: float
    macro_ci: tuple[float, float]
    confidence: float
    per_rule: Mapping[str, GateSummary]
    versus_peak: Mapping[str, GateSummary] | None
    hard_regressions: tuple[str, ...] | None
    opening_pairs_per_rule: int
    regression_tolerance: float
    peak_evidence_status: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": (
                "connect4-v3-bal5-r2-multirule-gate-v1"
                if self.peak_evidence_status == "complete"
                else "connect4-v3-bal5-r2-multirule-gate-v2"
            ),
            "verdict": self.verdict,
            "reason": self.reason,
            "macro_point_score": self.macro_point_score,
            "macro_ci": list(self.macro_ci),
            "confidence": self.confidence,
            "per_rule": {key: value.to_dict() for key, value in self.per_rule.items()},
            "versus_peak": (
                None
                if self.versus_peak is None
                else {key: value.to_dict() for key, value in self.versus_peak.items()}
            ),
            "hard_regressions": (
                None if self.hard_regressions is None else list(self.hard_regressions)
            ),
            "opening_pairs_per_rule": self.opening_pairs_per_rule,
            "regression_tolerance": self.regression_tolerance,
        }
        if self.peak_evidence_status != "complete":
            result["peak_evidence_status"] = self.peak_evidence_status
        return result


def summarize_multirule_gate(
    incumbent_results: Mapping[str, Iterable[object]],
    peak_results: Mapping[str, Iterable[object]] | None,
    *,
    bootstrap_samples: int,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
    regression_tolerance: float = 0.05,
) -> MultiRuleGateSummary:
    """Bootstrap opening pairs within each rule and average five rule means."""

    expected = set(BAL5_R2_RULE_IDS)
    if set(incumbent_results) != expected or (
        peak_results is not None and set(peak_results) != expected
    ):
        raise ValueError("multi-rule gate needs evidence for all five rules")
    if bootstrap_samples < 1 or not 0.5 < confidence < 1.0:
        raise ValueError("invalid multi-rule gate bootstrap settings")
    if not 0.0 <= regression_tolerance <= 0.5:
        raise ValueError("regression_tolerance must be in [0, 0.5]")

    by_rule: dict[str, GateSummary] = {}
    by_peak: dict[str, GateSummary] = {}
    seen_opening_ids: set[str] = set()
    pair_counts: set[int] = set()
    for index, rule_id in enumerate(BAL5_R2_RULE_IDS):
        current = summarize_paired_results(
            incumbent_results[rule_id],
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 2 * index,
        )
        peak = (
            None
            if peak_results is None
            else summarize_paired_results(
                peak_results[rule_id],
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                bootstrap_seed=bootstrap_seed + 2 * index + 1,
            )
        )
        current_keys = {(pair.opening_id, pair.seed) for pair in current.pairs}
        peak_keys = (
            None if peak is None else {(pair.opening_id, pair.seed) for pair in peak.pairs}
        )
        if peak_keys is not None and current_keys != peak_keys:
            raise ValueError(f"rule {rule_id!r} peak evidence must use identical openings")
        opening_ids = {opening_id for opening_id, _seed in current_keys}
        if seen_opening_ids & opening_ids:
            raise ValueError("different rules must not share opening IDs")
        seen_opening_ids.update(opening_ids)
        pair_counts.add(len(current.pairs))
        by_rule[rule_id] = current
        if peak is not None:
            by_peak[rule_id] = peak
    if len(pair_counts) != 1:
        raise ValueError("each rule must contribute the same number of opening pairs")

    rng = np.random.default_rng(int(bootstrap_seed) + 1009)
    sample_means = []
    point_scores = []
    for rule_id in BAL5_R2_RULE_IDS:
        values = np.asarray(
            [pair.pair_score for pair in by_rule[rule_id].pairs], dtype=np.float64
        )
        indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        sample_means.append(values[indices].mean(axis=1))
        point_scores.append(float(values.mean()))
    macro_samples = np.mean(np.stack(sample_means, axis=0), axis=0)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(macro_samples, (tail, 1.0 - tail))
    hard_regressions = (
        None if peak_results is None else tuple(
            rule_id
            for rule_id in BAL5_R2_RULE_IDS
            if by_peak[rule_id].overall.point_score < 0.5 - regression_tolerance
        )
    )
    if hard_regressions:
        verdict = "reject"
        reason = "at least one rule exceeds the frozen regression tolerance versus its peak"
    elif float(lower) > 0.5:
        verdict = "accept" if peak_results is not None else "peak_pending"
        reason = (
            "equal-rule macro confidence lower bound exceeds 50%"
            if peak_results is not None else "macro improvement established; peak evidence pending"
        )
    elif float(upper) < 0.5:
        verdict = "reject"
        reason = "equal-rule macro confidence upper bound is below 50%"
    else:
        verdict = "inconclusive"
        reason = "equal-rule macro evidence does not separate from 50%"
    return MultiRuleGateSummary(
        verdict=verdict,
        reason=reason,
        macro_point_score=float(np.mean(point_scores)),
        macro_ci=(float(lower), float(upper)),
        confidence=float(confidence),
        per_rule=by_rule,
        versus_peak=by_peak if peak_results is not None else None,
        hard_regressions=hard_regressions,
        opening_pairs_per_rule=pair_counts.pop(),
        regression_tolerance=float(regression_tolerance),
        peak_evidence_status="complete" if peak_results is not None else "not_evaluated",
    )


def run_multirule_sequential_gate(
    openings_by_rule: Mapping[str, Sequence[Any]],
    *,
    gate: GateConfig,
    run_seed: int,
    incumbent_model_id: str,
    peak_model_ids: Mapping[str, str],
    evaluate_pairs: Callable[[str, Sequence[Any], str], Iterable[object]],
    regression_tolerance: float = 0.05,
) -> tuple[
    dict[str, list[object]],
    dict[str, list[object]],
    MultiRuleGateSummary,
    list[dict[str, Any]],
]:
    """Append equal opening-pair batches until the macro gate resolves."""

    expected = set(BAL5_R2_RULE_IDS)
    if set(openings_by_rule) != expected or set(peak_model_ids) != expected:
        raise ValueError("sequential gate needs openings and peak IDs for five rules")
    if not incumbent_model_id or any(not value for value in peak_model_ids.values()):
        raise ValueError("sequential gate needs committed incumbent and peak model IDs")
    if any(len(openings_by_rule[rule]) < gate.max_opening_pairs for rule in BAL5_R2_RULE_IDS):
        raise ValueError("opening manifests must cover the maximum pair budget")
    incumbent_results: dict[str, list[object]] = {rule: [] for rule in BAL5_R2_RULE_IDS}
    peak_results: dict[str, list[object]] = {rule: [] for rule in BAL5_R2_RULE_IDS}
    looks: list[dict[str, Any]] = []
    pair_start = 0
    target_pairs = gate.initial_opening_pairs
    while True:
        for rule_id in BAL5_R2_RULE_IDS:
            openings = openings_by_rule[rule_id][pair_start:target_pairs]
            candidate = list(evaluate_pairs(rule_id, openings, incumbent_model_id))
            expected_games = 2 * len(openings)
            if len(candidate) != expected_games:
                raise RuntimeError("candidate-incumbent gate returned incomplete paired evidence")
            incumbent_results[rule_id].extend(candidate)
            if gate.multirule_evaluation_mode == "complete":
                if peak_model_ids[rule_id] == incumbent_model_id:
                    peak_results[rule_id].extend(candidate)
                else:
                    peak = list(evaluate_pairs(rule_id, openings, peak_model_ids[rule_id]))
                    if len(peak) != expected_games:
                        raise RuntimeError("candidate-peak gate returned incomplete paired evidence")
                    peak_results[rule_id].extend(peak)
        summary = summarize_multirule_gate(
            incumbent_results,
            peak_results if gate.multirule_evaluation_mode == "complete" else None,
            bootstrap_samples=gate.bootstrap_samples,
            confidence=gate.decision_confidence(),
            bootstrap_seed=run_seed + 2701,
            regression_tolerance=regression_tolerance,
        )
        if summary.verdict == "peak_pending":
            for rule_id in BAL5_R2_RULE_IDS:
                if peak_model_ids[rule_id] == incumbent_model_id:
                    peak_results[rule_id].extend(incumbent_results[rule_id])
                else:
                    peak = list(
                        evaluate_pairs(
                            rule_id,
                            openings_by_rule[rule_id][:target_pairs],
                            peak_model_ids[rule_id],
                        )
                    )
                    if len(peak) != 2 * target_pairs:
                        raise RuntimeError("candidate-peak gate returned incomplete paired evidence")
                    peak_results[rule_id].extend(peak)
            summary = summarize_multirule_gate(
                incumbent_results,
                peak_results,
                bootstrap_samples=gate.bootstrap_samples,
                confidence=gate.decision_confidence(),
                bootstrap_seed=run_seed + 2701,
                regression_tolerance=regression_tolerance,
            )
        looks.append({"pairs_per_rule": target_pairs, **summary.to_dict()})
        if summary.verdict != "inconclusive":
            break
        if target_pairs >= gate.max_opening_pairs:
            summary = replace(
                summary,
                verdict="reject",
                reason="maximum paired opening budget exhausted without macro improvement",
            )
            looks[-1] = {"pairs_per_rule": target_pairs, **summary.to_dict()}
            break
        pair_start = target_pairs
        target_pairs += gate.pair_increment
    return incumbent_results, peak_results, summary, looks
