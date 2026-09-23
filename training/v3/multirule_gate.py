"""Equal-rule paired evidence for the BAL-5 R2 canary gate.

The caller supplies candidate-versus-incumbent and candidate-versus-each-rule's
retained peak games.  A peak may initially be the incumbent.  This module does
not route self-play or promote a model; the formal runner must publish the
evidence and state atomically before changing its accepted champion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from connect4_core.rules import BAL5_R2_RULE_REGISTRY

from .gate import GateSummary, summarize_paired_results


BAL5_R2_RULE_IDS = tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs)


@dataclass(frozen=True)
class MultiRuleGateSummary:
    verdict: str
    reason: str
    macro_point_score: float
    macro_ci: tuple[float, float]
    confidence: float
    per_rule: Mapping[str, GateSummary]
    versus_peak: Mapping[str, GateSummary]
    hard_regressions: tuple[str, ...]
    opening_pairs_per_rule: int
    regression_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "connect4-v3-bal5-r2-multirule-gate-v1",
            "verdict": self.verdict,
            "reason": self.reason,
            "macro_point_score": self.macro_point_score,
            "macro_ci": list(self.macro_ci),
            "confidence": self.confidence,
            "per_rule": {key: value.to_dict() for key, value in self.per_rule.items()},
            "versus_peak": {
                key: value.to_dict() for key, value in self.versus_peak.items()
            },
            "hard_regressions": list(self.hard_regressions),
            "opening_pairs_per_rule": self.opening_pairs_per_rule,
            "regression_tolerance": self.regression_tolerance,
        }


def summarize_multirule_gate(
    incumbent_results: Mapping[str, Iterable[object]],
    peak_results: Mapping[str, Iterable[object]],
    *,
    bootstrap_samples: int,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
    regression_tolerance: float = 0.05,
) -> MultiRuleGateSummary:
    """Bootstrap opening pairs within each rule and average five rule means."""

    expected = set(BAL5_R2_RULE_IDS)
    if set(incumbent_results) != expected or set(peak_results) != expected:
        raise ValueError("multi-rule gate needs incumbent and peak evidence for all five rules")
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
        peak = summarize_paired_results(
            peak_results[rule_id],
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 2 * index + 1,
        )
        current_keys = {(pair.opening_id, pair.seed) for pair in current.pairs}
        peak_keys = {(pair.opening_id, pair.seed) for pair in peak.pairs}
        if current_keys != peak_keys:
            raise ValueError(f"rule {rule_id!r} peak evidence must use identical openings")
        opening_ids = {opening_id for opening_id, _seed in current_keys}
        if seen_opening_ids & opening_ids:
            raise ValueError("different rules must not share opening IDs")
        seen_opening_ids.update(opening_ids)
        pair_counts.add(len(current.pairs))
        by_rule[rule_id] = current
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
    hard_regressions = tuple(
        rule_id
        for rule_id in BAL5_R2_RULE_IDS
        if by_peak[rule_id].overall.point_score < 0.5 - regression_tolerance
    )
    if hard_regressions:
        verdict = "reject"
        reason = "at least one rule exceeds the frozen regression tolerance versus its peak"
    elif float(lower) > 0.5:
        verdict = "accept"
        reason = "equal-rule macro confidence lower bound exceeds 50%"
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
        versus_peak=by_peak,
        hard_regressions=hard_regressions,
        opening_pairs_per_rule=pair_counts.pop(),
        regression_tolerance=float(regression_tolerance),
    )
