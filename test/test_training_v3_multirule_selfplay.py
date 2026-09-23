from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from connect4_core.rules import BAL5_R2_RULE_REGISTRY
from training.v3.config import V3Config
from training.v3.multirule_selfplay import run_multirule_actor_pool


class MultiRuleSelfPlayTests(unittest.TestCase):
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
