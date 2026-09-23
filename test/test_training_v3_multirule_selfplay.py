from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from connect4_core.rules import BAL5_R2_RULE_REGISTRY
from training.v3.config import ModelConfig, V3Config
from training.v3.model import build_model
from training.v3.multirule_selfplay import (
    run_multirule_actor_pool,
    run_multirule_actor_pool_persistent,
)


class MultiRuleSelfPlayTests(unittest.TestCase):
    def test_persistent_pool_uses_one_service_lifetime_and_preserves_rule_blocks(self) -> None:
        calls = []

        def actor_pool(config, **kwargs):
            calls.append(kwargs)
            rows = []
            for offset, rule_id in enumerate(kwargs["game_rule_ids"]):
                rule = BAL5_R2_RULE_REGISTRY.get(rule_id)
                rows.append(SimpleNamespace(
                    game_id=kwargs["start_game_id"] + offset,
                    rule_id=rule_id,
                    rule_code=rule.rule_code,
                    producer_model_id=kwargs["producer_model_id"],
                    samples=(SimpleNamespace(rule_code=rule.rule_code),),
                ))
            return SimpleNamespace(
                games=tuple(rows), metrics=SimpleNamespace(to_dict=lambda: {"games": len(rows)})
            )

        result = run_multirule_actor_pool_persistent(
            self._config(10),
            accepted_model_state={"weight": object()},
            producer_model_id="accepted-g7",
            start_game_id=100,
            generation=0,
            actor_pool=actor_pool,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.metrics.to_dict()["shared_actor_runtime"], {"games": 10})
        self.assertEqual(
            tuple(calls[0]["game_rule_ids"]),
            tuple(rule for rule in self._config(10).selfplay.multi_rule_ids for _ in range(2)),
        )
        self.assertEqual([game.game_id for game in result.games], list(range(100, 110)))

    def test_persistent_cpu_pool_matches_sequential_game_outcomes(self) -> None:
        torch.manual_seed(7)
        config = self._config(10)
        config = replace(
            config,
            model=ModelConfig(architecture="gravity_resnet", channels=16, blocks=1),
            selfplay=replace(
                config.selfplay,
                search_schedule=(replace(config.selfplay.search_schedule[0], full_search_sims=2, fast_search_sims=1),),
            ),
            runtime=replace(
                config.runtime,
                device="cpu",
                selfplay_devices=(),
                actor_processes=2,
                mcts_lanes_per_actor=1,
                inference_batch_size=2,
                torch_threads=1,
            ),
        )
        model = build_model(config.model)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        kwargs = dict(
            accepted_model_state=state,
            producer_model_id="accepted-fixed",
            start_game_id=10,
            generation=0,
        )
        sequential = run_multirule_actor_pool(config, **kwargs)
        persistent = run_multirule_actor_pool_persistent(config, **kwargs)
        self.assertEqual(
            [(game.game_id, game.rule_id, game.rule_code, game.winner, game.moves)
             for game in sequential.games],
            [(game.game_id, game.rule_id, game.rule_code, game.winner, game.moves)
             for game in persistent.games],
        )

    def _config(self, games: int) -> V3Config:
        base = V3Config()
        return replace(
            base,
            selfplay=replace(
                base.selfplay,
                multi_rule_ids=tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs),
                rule_registry_hash=BAL5_R2_RULE_REGISTRY.registry_hash,
                search_schedule=(replace(base.selfplay.search_schedule[0], games=games),),
            ),
        )

    def test_routes_equal_games_with_one_producer_and_unique_ids(self) -> None:
        calls = []

        def actor_pool(config, **kwargs):
            rule = BAL5_R2_RULE_REGISTRY.get(config.selfplay.rule_id)
            start = kwargs["start_game_id"]
            count = config.selfplay.search_schedule[0].games
            calls.append((rule.rule_id, start, count, kwargs["producer_model_id"]))
            games = tuple(
                SimpleNamespace(
                    game_id=start + offset,
                    rule_id=rule.rule_id,
                    rule_code=rule.rule_code,
                    producer_model_id=kwargs["producer_model_id"],
                    samples=(SimpleNamespace(rule_code=rule.rule_code),),
                )
                for offset in range(count)
            )
            return SimpleNamespace(
                games=games, metrics=SimpleNamespace(to_dict=lambda: {"games": count})
            )

        result = run_multirule_actor_pool(
            self._config(10),
            accepted_model_state={"weight": object()},
            producer_model_id="accepted-g7",
            start_game_id=100,
            generation=0,
            actor_pool=actor_pool,
        )
        self.assertEqual(len(result.games), 10)
        self.assertEqual([game.game_id for game in result.games], list(range(100, 110)))
        self.assertEqual([item[2:] for item in calls], [(2, "accepted-g7")] * 5)
        self.assertEqual(result.metrics.to_dict()["raw_positions"], 10)

    def test_rejects_unequal_budget_or_uncommitted_producer(self) -> None:
        with self.assertRaisesRegex(ValueError, "split equally"):
            run_multirule_actor_pool(
                self._config(11),
                accepted_model_state={"weight": object()},
                producer_model_id="accepted-g7",
                start_game_id=0,
                generation=0,
            )
        with self.assertRaisesRegex(ValueError, "committed warm-start"):
            run_multirule_actor_pool(
                self._config(10),
                accepted_model_state=None,
                producer_model_id="random",
                start_game_id=0,
                generation=0,
            )


if __name__ == "__main__":
    unittest.main()
