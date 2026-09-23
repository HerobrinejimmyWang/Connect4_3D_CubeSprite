"""BAL-5 R2 equal-game routing through the existing accepted-model actor pool."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from connect4_core.rules import BAL5_R2_RULE_REGISTRY

from .actor_runtime import ActorPoolResult, run_self_play_actor_pool
from .config import V3Config
from .selfplay import GameRecord


@dataclass(frozen=True)
class MultiRuleActorMetrics:
    per_rule: Mapping[str, Mapping[str, Any]]
    games: int
    raw_positions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "connect4-v3-bal5-r2-actor-routing-v1",
            "per_rule": {key: dict(value) for key, value in self.per_rule.items()},
            "games": self.games,
            "raw_positions": self.raw_positions,
        }


@dataclass(frozen=True)
class MultiRuleActorResult:
    games: tuple[GameRecord, ...]
    metrics: MultiRuleActorMetrics


def run_multirule_actor_pool(
    config: V3Config,
    *,
    accepted_model_state: Mapping[str, Any] | None,
    producer_model_id: str | None,
    start_game_id: int,
    generation: int,
    actor_pool: Callable[..., ActorPoolResult] = run_self_play_actor_pool,
) -> MultiRuleActorResult:
    """Produce equal games per rule using one committed producer and unique IDs."""

    rule_ids = config.selfplay.multi_rule_ids
    expected_ids = tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs)
    if rule_ids != expected_ids:
        raise ValueError("BAL-5 R2 actor routing requires the frozen five-rule config")
    if accepted_model_state is None or not producer_model_id or producer_model_id == "random":
        raise ValueError("BAL-5 R2 requires a committed warm-start champion")
    if start_game_id < 0 or generation < 0:
        raise ValueError("game ID and generation must be non-negative")
    stage = config.selfplay.stage_for_generation(generation)
    divisor = len(rule_ids) * (2 if config.selfplay.opening_temperature_mixture.enabled else 1)
    if stage.games % divisor:
        raise ValueError(
            "generation games must split equally across five rules and exploration variants"
        )
    games_per_rule = stage.games // len(rule_ids)
    games: list[GameRecord] = []
    metrics: dict[str, dict[str, Any]] = {}
    next_game_id = start_game_id
    for rule_id in rule_ids:
        rule = BAL5_R2_RULE_REGISTRY.get(rule_id)
        schedule = tuple(
            replace(row, games=games_per_rule) for row in config.selfplay.search_schedule
        )
        rule_config = replace(
            config,
            selfplay=replace(config.selfplay, rule_id=rule_id, search_schedule=schedule),
        )
        batch = actor_pool(
            rule_config,
            accepted_model_state=accepted_model_state,
            producer_model_id=producer_model_id,
            start_game_id=next_game_id,
            generation=generation,
        )
        rows = tuple(batch.games)
        if len(rows) != games_per_rule:
            raise RuntimeError(f"rule {rule_id} returned the wrong number of games")
        if any(
            game.game_id != next_game_id + offset
            or game.rule_id != rule_id
            or game.rule_code != rule.rule_code
            or game.producer_model_id != producer_model_id
            or any(sample.rule_code != rule.rule_code for sample in game.samples)
            for offset, game in enumerate(rows)
        ):
            raise RuntimeError(f"rule {rule_id} broke game, sample, or producer lineage")
        metrics[rule_id] = {
            "games": len(rows),
            "raw_positions": sum(len(game.samples) for game in rows),
            "game_id_start": next_game_id,
            "game_id_stop": next_game_id + len(rows),
            "producer_model_id": producer_model_id,
            "actor_runtime": batch.metrics.to_dict(),
        }
        games.extend(rows)
        next_game_id += len(rows)
    if len(games) != stage.games:
        raise RuntimeError("multi-rule actor routing lost generation games")
    return MultiRuleActorResult(
        games=tuple(games),
        metrics=MultiRuleActorMetrics(
            per_rule=metrics,
            games=len(games),
            raw_positions=sum(len(game.samples) for game in games),
        ),
    )
