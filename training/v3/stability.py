"""Behavioral stability checks for V3 self-play generations.

The thresholds are deliberately based on correlated, repeated symptoms.  A
single short or low-variance generation is evidence to inspect, not sufficient
evidence to stop a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class StabilityThresholds:
    mean_game_length_watch: float = 18.0
    game_length_variance_watch: float = 50.0
    game_length_variance_critical: float = 30.0
    short_game_rate_watch: float = 0.10
    value_loss_watch: float = 0.20
    consecutive_generations_to_pause: int = 2
    correlated_behavior_signals_to_pause: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.mean_game_length_watch,
            self.game_length_variance_watch,
            self.game_length_variance_critical,
            self.value_loss_watch,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("stability thresholds must be positive")
        if not 0.0 < self.short_game_rate_watch < 1.0:
            raise ValueError("short-game threshold must be in (0, 1)")
        if self.game_length_variance_critical >= self.game_length_variance_watch:
            raise ValueError("critical variance must be below the watch threshold")
        if self.consecutive_generations_to_pause < 2:
            raise ValueError("behavioral pause requires at least two generations")
        if self.correlated_behavior_signals_to_pause < 2:
            raise ValueError("behavioral pause requires at least two correlated signals")


@dataclass(frozen=True)
class GenerationStabilityMetrics:
    generation: int
    games: int
    mean_game_length: float
    game_length_variance: float
    short_game_rate: float
    mean_policy_entropy: float | None = None
    value_loss: float | None = None

    def __post_init__(self) -> None:
        if self.generation < 0 or self.games < 1:
            raise ValueError("generation must be non-negative and games positive")
        if self.mean_game_length <= 0.0 or self.game_length_variance < 0.0:
            raise ValueError("game-length statistics are invalid")
        if not 0.0 <= self.short_game_rate <= 1.0:
            raise ValueError("short_game_rate must be in [0, 1]")
        for label, value in (
            ("mean_policy_entropy", self.mean_policy_entropy),
            ("value_loss", self.value_loss),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{label} must be non-negative when present")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GenerationStabilityMetrics":
        expected = {
            "generation",
            "games",
            "mean_game_length",
            "game_length_variance",
            "short_game_rate",
            "mean_policy_entropy",
            "value_loss",
        }
        if set(raw) != expected:
            raise ValueError("stability metric fields do not match schema")
        return cls(**dict(raw))


@dataclass(frozen=True)
class StabilityAssessment:
    generation: int
    action: str
    behavioral_signals: tuple[str, ...]
    contextual_signals: tuple[str, ...]
    consecutive_behavioral_alerts: int
    first_pause_generation: int | None

    def __post_init__(self) -> None:
        if self.action not in {"continue", "watch", "pause"}:
            raise ValueError("stability action must be continue, watch, or pause")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _signals_for(
    row: GenerationStabilityMetrics,
    previous: GenerationStabilityMetrics | None,
    thresholds: StabilityThresholds,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    behavioral: list[str] = []
    contextual: list[str] = []
    if row.mean_game_length < thresholds.mean_game_length_watch:
        behavioral.append("mean_game_length_low")
        if previous is not None and row.mean_game_length < previous.mean_game_length:
            contextual.append("mean_game_length_falling")
    if row.game_length_variance < thresholds.game_length_variance_watch:
        behavioral.append("game_length_variance_low")
        if row.game_length_variance < thresholds.game_length_variance_critical:
            contextual.append("game_length_variance_critical")
    if row.short_game_rate > thresholds.short_game_rate_watch:
        behavioral.append("short_game_rate_high")
    # Low value loss can accompany a collapsed, easy replay distribution, but
    # it is never a stop signal without the behavioral evidence above.
    if row.value_loss is not None and row.value_loss < thresholds.value_loss_watch:
        contextual.append("value_loss_low")
    # Entropy is intentionally descriptive only.  The historical collapse had
    # high entropy, so neither high nor low entropy certifies target quality.
    return tuple(behavioral), tuple(contextual)


def assess_stability(
    history: Iterable[GenerationStabilityMetrics],
    *,
    thresholds: StabilityThresholds = StabilityThresholds(),
) -> StabilityAssessment:
    rows = tuple(history)
    if not rows:
        raise ValueError("stability assessment requires at least one generation")
    if any(current.generation <= previous.generation for previous, current in zip(rows, rows[1:])):
        raise ValueError("stability generations must be strictly increasing")

    consecutive = 0
    first_pause_generation: int | None = None
    terminal_behavioral: tuple[str, ...] = ()
    terminal_contextual: tuple[str, ...] = ()
    for index, row in enumerate(rows):
        previous = rows[index - 1] if index else None
        behavioral, contextual = _signals_for(row, previous, thresholds)
        correlated = len(behavioral) >= thresholds.correlated_behavior_signals_to_pause
        consecutive = consecutive + 1 if correlated else 0
        if (
            first_pause_generation is None
            and consecutive >= thresholds.consecutive_generations_to_pause
        ):
            first_pause_generation = row.generation
        terminal_behavioral = behavioral
        terminal_contextual = contextual

    if consecutive >= thresholds.consecutive_generations_to_pause:
        action = "pause"
    elif terminal_behavioral or terminal_contextual:
        action = "watch"
    else:
        action = "continue"
    return StabilityAssessment(
        generation=rows[-1].generation,
        action=action,
        behavioral_signals=terminal_behavioral,
        contextual_signals=terminal_contextual,
        consecutive_behavioral_alerts=consecutive,
        first_pause_generation=first_pause_generation,
    )


__all__ = [
    "GenerationStabilityMetrics",
    "StabilityAssessment",
    "StabilityThresholds",
    "assess_stability",
]
