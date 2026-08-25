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
    mean_game_length_drop_fraction_watch: float = 0.10
    short_game_rate_rise_watch: float = 0.10
    first_player_win_rate_watch: float = 1.0
    first_player_win_rate_rise_watch: float = 0.10
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
        if not 0.0 < self.mean_game_length_drop_fraction_watch < 1.0:
            raise ValueError("mean-game-length drop threshold must be in (0, 1)")
        if not 0.0 < self.short_game_rate_rise_watch < 1.0:
            raise ValueError("short-game rise threshold must be in (0, 1)")
        if not 0.5 <= self.first_player_win_rate_watch <= 1.0:
            raise ValueError("first-player win-rate threshold must be in [0.5, 1]")
        if not 0.0 < self.first_player_win_rate_rise_watch < 1.0:
            raise ValueError("first-player win-rate rise threshold must be in (0, 1)")
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
    train_positions: int | None = None
    first_player_win_rate: float | None = None

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
        if self.train_positions is not None and self.train_positions < 0:
            raise ValueError("train_positions must be non-negative when present")
        if self.first_player_win_rate is not None and not 0.0 <= self.first_player_win_rate <= 1.0:
            raise ValueError("first_player_win_rate must be in [0, 1] when present")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GenerationStabilityMetrics":
        required = {
            "generation",
            "games",
            "mean_game_length",
            "game_length_variance",
            "short_game_rate",
            "mean_policy_entropy",
            "value_loss",
        }
        optional = {"train_positions", "first_player_win_rate"}
        if not required.issubset(raw) or set(raw) - required - optional:
            raise ValueError("stability metric fields do not match schema")
        payload = dict(raw)
        payload.setdefault("train_positions", None)
        payload.setdefault("first_player_win_rate", None)
        return cls(**payload)


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
        if (
            previous is not None
            and row.mean_game_length
            < previous.mean_game_length
            * (1.0 - thresholds.mean_game_length_drop_fraction_watch)
        ):
            contextual.append("mean_game_length_falling")
    if row.game_length_variance < thresholds.game_length_variance_watch:
        behavioral.append("game_length_variance_low")
        if row.game_length_variance < thresholds.game_length_variance_critical:
            contextual.append("game_length_variance_critical")
    if row.short_game_rate > thresholds.short_game_rate_watch:
        behavioral.append("short_game_rate_high")
        if (
            previous is not None
            and row.short_game_rate - previous.short_game_rate
            >= thresholds.short_game_rate_rise_watch
        ):
            contextual.append("short_game_rate_rising")
    if (
        row.first_player_win_rate is not None
        and row.first_player_win_rate > thresholds.first_player_win_rate_watch
    ):
        behavioral.append("first_player_win_rate_high")
        if (
            previous is not None
            and previous.first_player_win_rate is not None
            and row.first_player_win_rate - previous.first_player_win_rate
            >= thresholds.first_player_win_rate_rise_watch
        ):
            contextual.append("first_player_win_rate_rising")
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
    game_length_pause_start_train_positions: int = 0,
) -> StabilityAssessment:
    rows = tuple(history)
    if not rows:
        raise ValueError("stability assessment requires at least one generation")
    if any(current.generation <= previous.generation for previous, current in zip(rows, rows[1:])):
        raise ValueError("stability generations must be strictly increasing")
    if game_length_pause_start_train_positions < 0:
        raise ValueError("game-length pause start must be non-negative")

    consecutive = 0
    first_pause_generation: int | None = None
    terminal_behavioral: tuple[str, ...] = ()
    terminal_contextual: tuple[str, ...] = ()
    for index, row in enumerate(rows):
        previous = rows[index - 1] if index else None
        behavioral, contextual = _signals_for(row, previous, thresholds)
        warmup_active = (
            row.train_positions is not None
            and row.train_positions < game_length_pause_start_train_positions
        )
        active_behavioral = behavioral
        active_contextual = contextual
        if warmup_active:
            game_length_signals = {
                "mean_game_length_low",
                "game_length_variance_low",
                "short_game_rate_high",
            }
            game_length_context = {
                "mean_game_length_falling",
                "game_length_variance_critical",
                "short_game_rate_rising",
            }
            active_behavioral = tuple(
                signal for signal in behavioral if signal not in game_length_signals
            )
            active_contextual = tuple(
                signal for signal in contextual if signal not in game_length_context
            )
            contextual = (*contextual, "game_length_pause_warmup")
        correlated = len(active_behavioral) >= thresholds.correlated_behavior_signals_to_pause
        distribution_deteriorating = {
            "mean_game_length_falling",
            "short_game_rate_rising",
        }.issubset(active_contextual)
        escalating = (
            distribution_deteriorating
            or "value_loss_low" in active_contextual
            or "first_player_win_rate_rising" in active_contextual
        )
        if not correlated:
            consecutive = 0
        elif consecutive == 0:
            consecutive = 1
        elif escalating:
            consecutive += 1
        else:
            # A frozen champion under a deliberately exploratory schedule can
            # repeatedly cross absolute game-length watch thresholds without
            # deteriorating. Keep the condition visible, but require a
            # joint material adverse trend (or collapsed value loss) before
            # an automatic pause is armed. A short-game-rate swing by itself
            # is too noisy in a 64-game monitoring window.
            consecutive = 1
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
