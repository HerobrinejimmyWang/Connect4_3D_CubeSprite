"""Deterministic paired-opening gate summaries and promotion decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


VALID_VERDICTS = frozenset({"accept", "reject", "inconclusive"})


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
class GateDecision:
    verdict: str
    reason: str
    summary: GateSummary

    def __post_init__(self) -> None:
        if self.verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid gate verdict: {self.verdict!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "summary": self.summary.to_dict(),
        }


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


def decide_gate(
    summary: GateSummary,
    *,
    accept_threshold: float = 0.5,
    role_floor: float = 0.45,
    role_hard_reject_floor: float | None = None,
    allow_role_extension: bool = False,
) -> GateDecision:
    if not 0.0 <= accept_threshold <= 1.0:
        raise ValueError("accept_threshold must be in [0, 1]")
    if not 0.0 <= role_floor <= 1.0:
        raise ValueError("role_floor must be in [0, 1]")
    hard_floor = role_floor if role_hard_reject_floor is None else role_hard_reject_floor
    if not 0.0 <= hard_floor <= role_floor:
        raise ValueError("role_hard_reject_floor must be in [0, role_floor]")
    first_score = summary.candidate_as_first.point_score
    second_score = summary.candidate_as_second.point_score
    roles_pass = first_score >= role_floor and second_score >= role_floor
    if summary.ci_lower > accept_threshold and roles_pass:
        return GateDecision(
            "accept",
            "pair-score confidence interval clears the threshold and both roles clear the floor",
            summary,
        )
    if summary.ci_upper < accept_threshold:
        return GateDecision(
            "reject",
            "pair-score confidence interval is below the acceptance threshold",
            summary,
        )
    if first_score < role_floor or second_score < role_floor:
        if allow_role_extension and min(first_score, second_score) >= hard_floor:
            return GateDecision(
                "inconclusive",
                "role score is in the predeclared extension band",
                summary,
            )
        return GateDecision(
            "reject",
            "candidate failed the first-player or second-player score floor",
            summary,
        )
    return GateDecision(
        "inconclusive",
        "available pairs do not establish improvement at the requested confidence",
        summary,
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
) -> GateDecision:
    summary = summarize_paired_results(
        results,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    return decide_gate(
        summary,
        accept_threshold=accept_threshold,
        role_floor=role_floor,
        role_hard_reject_floor=role_hard_reject_floor,
        allow_role_extension=allow_role_extension,
    )


__all__ = [
    "GateDecision",
    "GateGameResult",
    "GateSummary",
    "PairStats",
    "RoleStats",
    "VALID_VERDICTS",
    "decide_gate",
    "evaluate_gate",
    "summarize_paired_results",
]
