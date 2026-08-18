from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from training.v3.actor_runtime import run_self_play_actor_pool
from training.v3.config import (
    ExplorationPhaseConfig,
    ModelConfig,
    SearchStageConfig,
    V3Config,
)
from training.v3.model import build_model
from training.v3.selfplay import run_self_play_games


def _actor_config(
    *, games: int = 4, actors: int = 2, lanes: int = 1, inference_batch_size: int = 4
) -> V3Config:
    config = V3Config()
    return replace(
        config,
        model=ModelConfig(architecture="gravity_resnet", channels=16, blocks=1),
        selfplay=replace(
            config.selfplay,
            search_schedule=(SearchStageConfig(0, games, 4, 2, 0.5),),
            exploration_phases=(ExplorationPhaseConfig(0, 1.0, 0.0, 0.0),),
        ),
        runtime=replace(
            config.runtime,
            device="cpu",
            selfplay_devices=(),
            actor_processes=actors,
            mcts_lanes_per_actor=lanes,
            inference_batch_size=inference_batch_size,
            learner_amp=False,
            torch_threads=1,
        ),
    )


class ActorPoolTests(unittest.TestCase):
    def test_random_actor_pool_matches_sequential_game_identity(self) -> None:
        config = _actor_config(games=4, actors=2)
        sequential = run_self_play_games(config, start_game_id=9, generation=0)
        pooled = run_self_play_actor_pool(
            config,
            start_game_id=9,
            generation=0,
        )

        self.assertEqual([game.game_id for game in pooled.games], [9, 10, 11, 12])
        self.assertEqual(
            [game.seed for game in pooled.games],
            [game.seed for game in sequential],
        )
        self.assertEqual(
            [game.moves for game in pooled.games],
            [game.moves for game in sequential],
        )
        self.assertEqual(pooled.metrics.actor_processes, 2)
        self.assertEqual(pooled.metrics.task_queue_capacity, 4)
        self.assertEqual(pooled.metrics.result_queue_capacity, 4)
        self.assertEqual(pooled.metrics.inference_services, ())

    def test_accepted_model_uses_one_shared_cpu_inference_service(self) -> None:
        config = _actor_config(games=2, actors=2)
        model = build_model(config.model)
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        pooled = run_self_play_actor_pool(
            config,
            accepted_model_state=state,
            producer_model_id="accepted-test",
            generation=0,
            inference_batch_timeout_s=0.005,
        )

        self.assertEqual(len(pooled.games), 2)
        self.assertTrue(all(game.producer_model_id == "accepted-test" for game in pooled.games))
        self.assertEqual(len(pooled.metrics.inference_services), 1)
        service = pooled.metrics.inference_services[0]
        self.assertEqual(service.device, "cpu")
        self.assertEqual(service.actor_count, 2)
        self.assertGreater(service.requests, 0)
        self.assertGreater(service.positions, 0)
        self.assertGreater(service.batches, 0)
        self.assertLessEqual(service.mean_batch, service.max_batch)
        self.assertTrue(np.isfinite(service.mean_batch))

    def test_shared_inference_batch_limit_is_hard(self) -> None:
        config = _actor_config(
            games=4,
            actors=4,
            lanes=3,
            inference_batch_size=4,
        )
        model = build_model(config.model)
        pooled = run_self_play_actor_pool(
            config,
            accepted_model_state=model.state_dict(),
            producer_model_id="accepted-hard-limit-test",
            inference_batch_timeout_s=0.01,
        )

        service = pooled.metrics.inference_services[0]
        self.assertGreater(service.batches, 0)
        self.assertLessEqual(service.max_batch, service.batch_limit)

    def test_model_state_requires_non_random_producer_identity(self) -> None:
        config = _actor_config(games=1, actors=1)
        model = build_model(config.model)
        with self.assertRaisesRegex(ValueError, "producer_model_id"):
            run_self_play_actor_pool(
                config,
                accepted_model_state=model.state_dict(),
                producer_model_id="random",
            )


if __name__ == "__main__":
    unittest.main()
