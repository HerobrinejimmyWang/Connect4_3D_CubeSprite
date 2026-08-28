"""Deterministic paired-opening gate summaries and promotion decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np


VALID_VERDICTS = frozenset({"accept", "reject", "inconclusive"})
VALID_ROLE_GUARD_MODES = frozenset({"absolute_floor", "relative_noninferiority"})


@dataclass(frozen=True)
class GateGameResult:
    opening_id: str
    seed: int
    candidate_is_first: bool
    candidate_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "opening_id", str(self.opening_id))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "candidate_is_first", bool(self.candidate_is_first))
        score = float(self.candidate_score)
        if not any(abs(score - valid) < 1e-12 for valid in (0.0, 0.5, 1.0)):
            raise ValueError("candidate_score must be 0, 0.5, or 1")
        object.__setattr__(self, "candidate_score", score)

    @classmethod
    def from_player_winner(
        cls,
        *,
        opening_id: str | int,
        seed: int,
        candidate_is_first: bool,
        winner: int,
        is_draw: bool = False,
    ) -> "GateGameResult":
        if is_draw or int(winner) == 0:
            score = 0.5
        else:
            candidate_player = 1 if candidate_is_first else -1
            score = 1.0 if int(winner) == candidate_player else 0.0
        return cls(str(opening_id), seed, candidate_is_first, score)


@dataclass(frozen=True)
class RoleStats:
    games: int
    wins: int
    draws: int
    losses: int
    points: float
    point_score: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PairStats:
    opening_id: str
    seed: int
    candidate_first_score: float
    candidate_second_score: float
    pair_score: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class GateSummary:
    overall: RoleStats
    candidate_as_first: RoleStats
    candidate_as_second: RoleStats
    pairs: tuple[PairStats, ...]
    confidence: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "candidate_as_first": self.candidate_as_first.to_dict(),
            "candidate_as_second": self.candidate_as_second.to_dict(),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "confidence": self.confidence,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "bootstrap_samples": self.bootstrap_samples,
        }


@dataclass(frozen=True)
class RoleNonInferiorityStats:
    role: str
    games: int
    candidate_point_score: float
    control_point_score: float
    point_delta: float
    margin: float
    confidence: float
    ci_lower: float
    ci_upper: float
    bootstrap_samples: int
    regression_established: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    verdict: str
    reason: str
    summary: GateSummary
    role_guard_mode: str = "absolute_floor"
    role_noninferiority: RoleNonInferiorityStats | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid gate verdict: {self.verdict!r}")
        if self.role_guard_mode not in VALID_ROLE_GUARD_MODES:
            raise ValueError(f"invalid role guard mode: {self.role_guard_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "summary": self.summary.to_dict(),
            "role_guard_mode": self.role_guard_mode,
            "role_noninferiority": (
                None if self.role_noninferiority is None else self.role_noninferiority.to_dict()
            ),
        }


def reject_terminal_inconclusive(decision: GateDecision) -> GateDecision:
    """Resolve exhausted evidence as no-promotion while retaining its meaning.

    Intermediate sequential looks remain ``inconclusive`` so the evaluator can
    append more paired openings.  Once the declared maximum pair budget is
    exhausted, however, training should not require an operator to clear a
    third promotion state.  The candidate is rejected for insufficient
    evidence, which is deliberately distinct from evidence that it is weaker.
    """

    if decision.verdict != "inconclusive":
        return decision
    return replace(
        decision,
        verdict="reject",
        reason=(
            "maximum pair budget exhausted without establishing promotion; "
            "candidate rejected for insufficient evidence"
        ),
    )


def _value(result: object, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _coerce_result(result: object) -> GateGameResult:
    if isinstance(result, GateGameResult):
        return result
    score = _value(result, "candidate_score")
    if score is not None:
        return GateGameResult(
            opening_id=_value(result, "opening_id"),
            seed=_value(result, "seed"),
            candidate_is_first=_value(result, "candidate_is_first"),
            candidate_score=score,
        )
    winner = _value(result, "winner")
    if winner is None:
        raise ValueError("gate result needs candidate_score or winner")
    return GateGameResult.from_player_winner(
        opening_id=_value(result, "opening_id"),
        seed=_value(result, "seed"),
        candidate_is_first=_value(result, "candidate_is_first"),
        winner=int(winner),
        is_draw=bool(_value(result, "is_draw", False)),
    )


def _role_stats(scores: Iterable[float]) -> RoleStats:
    values = tuple(float(value) for value in scores)
    games = len(values)
    wins = sum(value == 1.0 for value in values)
    draws = sum(value == 0.5 for value in values)
    losses = sum(value == 0.0 for value in values)
    points = float(sum(values))
    return RoleStats(
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        points=points,
        point_score=(points / games if games else 0.0),
    )


def summarize_paired_results(
    results: Iterable[object],
    *,
    bootstrap_samples: int,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> GateSummary:
    """Validate exact color swaps and bootstrap the mean score over pairs."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    grouped: dict[str, list[GateGameResult]] = {}
    for raw_result in results:
        result = _coerce_result(raw_result)
        grouped.setdefault(result.opening_id, []).append(result)
    if not grouped:
        raise ValueError("paired gate needs at least one opening pair")

    pairs: list[PairStats] = []
    first_scores: list[float] = []
    second_scores: list[float] = []
    for opening_id in sorted(grouped):
        games = grouped[opening_id]
        if len(games) != 2:
            raise ValueError(f"opening {opening_id!r} must have exactly two games")
        by_role = {game.candidate_is_first: game for game in games}
        if set(by_role) != {False, True}:
            raise ValueError(f"opening {opening_id!r} must swap candidate color exactly once")
        if len({game.seed for game in games}) != 1:
            raise ValueError(f"opening {opening_id!r} must use the same seed in both games")
        first = by_role[True].candidate_score
        second = by_role[False].candidate_score
        first_scores.append(first)
        second_scores.append(second)
        pairs.append(
            PairStats(
                opening_id=opening_id,
                seed=games[0].seed,
                candidate_first_score=first,
                candidate_second_score=second,
                pair_score=(first + second) / 2.0,
            )
        )

    pair_scores = np.asarray([pair.pair_score for pair in pairs], dtype=np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    sampled_indices = rng.integers(0, len(pair_scores), size=(bootstrap_samples, len(pair_scores)))
    bootstrap_means = pair_scores[sampled_indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    return GateSummary(
        overall=_role_stats((*first_scores, *second_scores)),
        candidate_as_first=_role_stats(first_scores),
        candidate_as_second=_role_stats(second_scores),
        pairs=tuple(pairs),
        confidence=float(confidence),
        ci_lower=float(lower),
        ci_upper=float(upper),
        bootstrap_samples=int(bootstrap_samples),
    )


def summarize_role_noninferiority(
    candidate: GateSummary,
    control: GateSummary,
    *,
    margin: float,
    bootstrap_samples: int,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> RoleNonInferiorityStats:
    """Compare candidate and accepted-control second-player scores by opening.

    The control is the accepted champion playing both colors under the exact
    opening, search, and seed contract used by the candidate gate.  A role
    regression is established only when the *upper* confidence bound of the
    paired candidate-minus-control delta is below ``-margin``.  Uncertain
    evidence therefore does not become a rejection by default.
    """

    if not 0.0 <= margin <= 1.0:
        raise ValueError("role non-inferiority margin must be in [0, 1]")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    candidate_pairs = {(pair.opening_id, pair.seed): pair for pair in candidate.pairs}
    control_pairs = {(pair.opening_id, pair.seed): pair for pair in control.pairs}
    if candidate_pairs.keys() != control_pairs.keys():
        raise ValueError("candidate and control role evidence must use identical openings and seeds")
    if not candidate_pairs:
        raise ValueError("role non-inferiority needs at least one opening pair")
    deltas = np.asarray(
        [
            candidate_pairs[key].candidate_second_score
            - control_pairs[key].candidate_second_score
            for key in sorted(candidate_pairs)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    sampled_indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
    bootstrap_means = deltas[sampled_indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    point_delta = float(deltas.mean())
    return RoleNonInferiorityStats(
        role="second",
        games=len(deltas),
        candidate_point_score=candidate.candidate_as_second.point_score,
        control_point_score=control.candidate_as_second.point_score,
        point_delta=point_delta,
        margin=float(margin),
        confidence=float(confidence),
        ci_lower=float(lower),
        ci_upper=float(upper),
        bootstrap_samples=int(bootstrap_samples),
        regression_established=bool(float(upper) < -float(margin)),
    )


def decide_gate(
    summary: GateSummary,
    *,
    accept_threshold: float = 0.5,
    role_floor: float = 0.45,
    role_hard_reject_floor: float | None = None,
    allow_role_extension: bool = False,
    role_guard_mode: str = "absolute_floor",
    role_noninferiority: RoleNonInferiorityStats | None = None,
) -> GateDecision:
    if not 0.0 <= accept_threshold <= 1.0:
        raise ValueError("accept_threshold must be in [0, 1]")
    if not 0.0 <= role_floor <= 1.0:
        raise ValueError("role_floor must be in [0, 1]")
    hard_floor = role_floor if role_hard_reject_floor is None else role_hard_reject_floor
    if not 0.0 <= hard_floor <= role_floor:
        raise ValueError("role_hard_reject_floor must be in [0, role_floor]")
    if role_guard_mode not in VALID_ROLE_GUARD_MODES:
        raise ValueError(f"invalid role guard mode: {role_guard_mode!r}")
    if role_guard_mode == "relative_noninferiority" and role_noninferiority is None:
        raise ValueError("relative_noninferiority requires accepted-control role evidence")
    first_score = summary.candidate_as_first.point_score
    second_score = summary.candidate_as_second.point_score
    roles_pass = first_score >= role_floor and second_score >= role_floor
    if summary.ci_upper < accept_threshold:
        return GateDecision(
            "reject",
            "pair-score confidence interval is below the acceptance threshold",
            summary,
            role_guard_mode,
            role_noninferiority,
        )
    if (
        role_guard_mode == "relative_noninferiority"
        and role_noninferiority is not None
        and role_noninferiority.regression_established
    ):
        return GateDecision(
            "reject",
            "accepted-control evidence establishes second-player regression beyond the margin",
            summary,
            role_guard_mode,
            role_noninferiority,
        )
    if summary.ci_lower > accept_threshold and (
        role_guard_mode == "relative_noninferiority" or roles_pass
    ):
        reason = (
            "pair-score confidence interval clears the threshold without established "
            "second-player regression"
            if role_guard_mode == "relative_noninferiority"
            else "pair-score confidence interval clears the threshold and both roles clear the floor"
        )
        return GateDecision(
            "accept",
            reason,
            summary,
            role_guard_mode,
            role_noninferiority,
        )
    if role_guard_mode == "relative_noninferiority":
        return GateDecision(
            "inconclusive",
            "available pairs do not establish overall improvement at the requested confidence",
            summary,
            role_guard_mode,
            role_noninferiority,
        )
    if first_score < role_floor or second_score < role_floor:
        if allow_role_extension and min(first_score, second_score) >= hard_floor:
            return GateDecision(
                "inconclusive",
                "role score is in the predeclared extension band",
                summary,
                role_guard_mode,
                role_noninferiority,
            )
        return GateDecision(
            "reject",
            "candidate failed the first-player or second-player score floor",
            summary,
            role_guard_mode,
            role_noninferiority,
        )
    return GateDecision(
        "inconclusive",
        "available pairs do not establish improvement at the requested confidence",
        summary,
        role_guard_mode,
        role_noninferiority,
    )


def evaluate_gate(
    results: Iterable[object],
    *,
    bootstrap_samples: int,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
    accept_threshold: float = 0.5,
    role_floor: float = 0.45,
    role_hard_reject_floor: float | None = None,
    allow_role_extension: bool = False,
    role_guard_mode: str = "absolute_floor",
    role_control_results: Iterable[object] | None = None,
    role_noninferiority_margin: float = 0.05,
) -> GateDecision:
    summary = summarize_paired_results(
        results,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    role_noninferiority = None
    if role_guard_mode == "relative_noninferiority":
        if role_control_results is None:
            raise ValueError("relative_noninferiority requires role_control_results")
        control_summary = summarize_paired_results(
            role_control_results,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 1,
        )
        role_noninferiority = summarize_role_noninferiority(
            summary,
            control_summary,
            margin=role_noninferiority_margin,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed + 2,
        )
    return decide_gate(
        summary,
        accept_threshold=accept_threshold,
        role_floor=role_floor,
        role_hard_reject_floor=role_hard_reject_floor,
        allow_role_extension=allow_role_extension,
        role_guard_mode=role_guard_mode,
        role_noninferiority=role_noninferiority,
    )


__all__ = [
    "GateDecision",
    "GateGameResult",
    "GateSummary",
    "PairStats",
    "RoleNonInferiorityStats",
    "RoleStats",
    "VALID_ROLE_GUARD_MODES",
    "VALID_VERDICTS",
    "decide_gate",
    "evaluate_gate",
    "reject_terminal_inconclusive",
    "summarize_role_noninferiority",
    "summarize_paired_results",
]
