from __future__ import annotations

import queue
import unittest
from dataclasses import replace

import numpy as np

from training.v3.actor_runtime import RemotePredictor, run_self_play_actor_pool
from training.v3.config import (
    ExplorationPhaseConfig,
    ModelConfig,
    OpeningTemperatureMixtureConfig,
    SearchStageConfig,
    V3Config,
)
from training.v3.model import build_model, classic_rule_features
from training.v3.selfplay import SEARCH_FULL, run_self_play_games


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
    def test_opening_temperature_mixture_assigns_equal_games_and_only_changes_temperature(self) -> None:
        config = _actor_config(games=4, actors=2)
        mixture = OpeningTemperatureMixtureConfig(enabled=True)
        config = replace(
            config,
            selfplay=replace(
                config.selfplay,
                exploration_phases=(
                    ExplorationPhaseConfig(0, 1.0, 0.24, 0.06),
                    ExplorationPhaseConfig(28, 0.5, 0.5, 0.005),
                    ExplorationPhaseConfig(50, 0.0, 0.0, 0.0),
                ),
                opening_temperature_mixture=mixture,
            ),
        )
        lowered = config.selfplay.for_exploration_variant(
            "lowered_opening_temperature"
        )
        self.assertEqual(
            [phase.start_ply for phase in lowered.exploration_phases],
            [0, 8, 28, 50],
        )
        self.assertEqual(lowered.exploration_for_ply(7).temperature, 0.5)
        self.assertEqual(lowered.exploration_for_ply(8).temperature, 1.0)
        self.assertEqual(lowered.exploration_for_ply(7).dirichlet_alpha, 0.24)
        self.assertEqual(lowered.exploration_for_ply(7).dirichlet_epsilon, 0.06)

        pooled = run_self_play_actor_pool(config, start_game_id=10, generation=0)
        self.assertEqual(
            [game.exploration_variant for game in pooled.games],
            [
                "baseline",
                "lowered_opening_temperature",
                "baseline",
                "lowered_opening_temperature",
            ],
        )

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

    def test_pool_can_force_opening_positions_to_full_search(self) -> None:
        config = _actor_config(games=2, actors=2)
        pooled = run_self_play_actor_pool(
            config,
            force_full_search_before_ply=12,
        )
        for game in pooled.games:
            opening = [move for move in game.moves if move.ply < 12]
            self.assertTrue(opening)
            self.assertTrue(all(move.search_kind == SEARCH_FULL for move in opening))


class RemotePredictorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: queue.Queue = queue.Queue()
        self.responses: queue.Queue = queue.Queue()
        self.predictor = RemotePredictor(
            3,
            self.requests,
            self.responses,
            response_timeout_s=0.1,
        )

    def test_request_carries_aligned_board_role_and_rule_batches(self) -> None:
        boards = np.zeros((2, 6, 5, 5), dtype=np.int8)
        roles = np.asarray(((1, 0), (0, 1)), dtype=np.float32)
        rules = classic_rule_features(2).numpy()
        expected_policy = np.full((2, 25), 1.0 / 25.0, dtype=np.float32)
        expected_wdl = np.full((2, 3), 1.0 / 3.0, dtype=np.float32)
        self.responses.put((0, expected_policy, expected_wdl, None))

        policy, wdl = self.predictor.predict_batch(
            boards,
            role_to_play=roles,
            rule_features=rules,
        )

        actor_id, request_id, sent_boards, sent_roles, sent_rules = self.requests.get_nowait()
        self.assertEqual((actor_id, request_id), (3, 0))
        np.testing.assert_array_equal(sent_boards, boards)
        np.testing.assert_array_equal(sent_roles, roles)
        np.testing.assert_array_equal(sent_rules, rules)
        np.testing.assert_array_equal(policy, expected_policy)
        np.testing.assert_array_equal(wdl, expected_wdl)

    def test_context_is_required_and_batch_aligned_before_enqueue(self) -> None:
        boards = np.zeros((2, 6, 5, 5), dtype=np.int8)
        roles = np.asarray(((1, 0), (0, 1)), dtype=np.float32)
        rules = classic_rule_features(2).numpy()
        with self.assertRaises(TypeError):
            self.predictor.predict_batch(boards)  # type: ignore[call-arg]
        with self.assertRaisesRegex(ValueError, "role_to_play"):
            self.predictor.predict_batch(
                boards,
                role_to_play=roles[:1],
                rule_features=rules,
            )
        with self.assertRaisesRegex(ValueError, "rule_features"):
            self.predictor.predict_batch(
                boards,
                role_to_play=roles,
                rule_features=rules[:1],
            )
        self.assertTrue(self.requests.empty())


if __name__ == "__main__":
    unittest.main()
