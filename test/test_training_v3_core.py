from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from connect4_core import GameRules
from training.v3.config import (
    ExplorationPhaseConfig,
    ModelConfig,
    SearchStageConfig,
    V3Config,
    config_hash,
    load_config,
    resolve_config,
)
from training.v3.evaluation import build_openings
from training.v3.model import (
    COLUMN_COUNT,
    LEGACY_ACTION_COUNT,
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
    LegacyPolicyValueAdapter,
    TorchPredictor,
    build_model,
    column_policy_to_legacy,
    column_to_legacy_action,
    legacy_action_to_column,
    legacy_policy_to_columns,
    wdl_expected_value,
)
from training.v3.search import MCTS, RandomPredictor, SEARCH_FULL, TreeNode, puct_score
from training.v3.selfplay import run_self_play_games, wdl_target_for_player


FULL_DRAW_COLUMNS = (
    2, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 1, 1, 3, 1, 3, 2, 2, 3, 2, 3, 3, 4, 3,
    5, 4, 5, 4, 4, 5, 4, 4, 5, 5, 7, 5, 7, 6, 11, 6, 6, 6, 12, 6, 6, 7, 13, 7, 7,
    8, 7, 9, 8, 9, 8, 8, 8, 10, 8, 11, 9, 11, 9, 12, 9, 9, 10, 10, 10, 10, 10, 14,
    11, 15, 11, 17, 11, 17, 12, 18, 12, 19, 12, 12, 13, 13, 14, 13, 15, 13, 13, 14,
    15, 14, 14, 19, 14, 21, 15, 15, 16, 15, 18, 18, 18, 18, 20, 20, 16, 20, 16, 16,
    17, 16, 17, 16, 17, 17, 18, 20, 20, 20, 21, 21, 22, 22, 19, 22, 19, 19, 21, 19,
    22, 22, 23, 23, 23, 23, 21, 23, 21, 23, 24, 24, 22, 24, 24, 24, 24,
)


class ConfigTests(unittest.TestCase):
    def test_strict_config_round_trip_hash_and_overrides(self) -> None:
        config = V3Config()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(config.to_json(), encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(config, loaded)
        self.assertEqual(config_hash(config), config_hash(loaded))
        resolved = resolve_config(config, run_dir="somewhere", resume=True, device="cuda:0")
        self.assertEqual(resolved.run.run_dir, "somewhere")
        self.assertTrue(resolved.run.resume)
        self.assertEqual(resolved.runtime.device, "cuda:0")
        cuda_config = replace(
            config,
            runtime=replace(config.runtime, device="cuda:0", learner_amp=True),
        )
        cpu_config = resolve_config(cuda_config, device="cpu")
        self.assertEqual(cpu_config.runtime.device, "cpu")
        self.assertFalse(cpu_config.runtime.learner_amp)
        self.assertEqual(config_hash(config), config_hash(resolve_config(config, run_dir="moved", resume=True)))
        topology_change = replace(
            config,
            runtime=replace(config.runtime, actor_processes=3, inference_batch_size=3),
        )
        self.assertEqual(config_hash(config), config_hash(topology_change))
        lane_change = replace(
            config,
            runtime=replace(config.runtime, mcts_lanes_per_actor=2),
        )
        self.assertNotEqual(config_hash(config), config_hash(lane_change))
        shard_change = replace(
            config,
            replay=replace(config.replay, shard_games=2),
        )
        self.assertEqual(config_hash(config), config_hash(shard_change))
        self.assertEqual(config.selfplay.stage_for_generation(100).games, 4)
        self.assertEqual(config.selfplay.exploration_for_ply(20).temperature, 0.0)

    def test_single_gpu_preset_device_override_keeps_staged_topology(self) -> None:
        preset = (
            Path(__file__).resolve().parents[1]
            / "training"
            / "v3"
            / "configs"
            / "pilot_gpu_64x4.json"
        )
        config = load_config(preset)
        self.assertEqual(config.runtime.selfplay_devices, ())
        resolved = resolve_config(config, device="cuda:3")
        self.assertEqual(resolved.runtime.device, "cuda:3")
        self.assertEqual(resolved.runtime.selfplay_devices, ())

    def test_unknown_or_wrong_type_is_rejected(self) -> None:
        raw = V3Config().to_dict()
        raw["selfplay"]["mystery"] = 1
        with self.assertRaisesRegex(ValueError, "mystery"):
            V3Config.from_dict(raw)
        raw = V3Config().to_dict()
        raw["run"]["resume"] = "yes"
        with self.assertRaisesRegex(TypeError, "run.resume"):
            V3Config.from_dict(raw)
        with self.assertRaisesRegex(ValueError, "architecture"):
            ModelConfig(architecture="legacy")
        raw = V3Config().to_dict()
        raw["selfplay"]["search_schedule"][0]["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "unknown"):
            V3Config.from_dict(raw)


class ModelAndAdapterTests(unittest.TestCase):
    def test_model_outputs_column_policy_and_wdl_logits(self) -> None:
        model = build_model(ModelConfig(channels=8, blocks=1))
        policy, wdl = model(torch.zeros((2, 6, 5, 5), dtype=torch.float32))
        self.assertEqual(tuple(policy.shape), (2, 25))
        self.assertEqual(tuple(wdl.shape), (2, 3))
        self.assertFalse(any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()))
        policy_batch, wdl_batch = TorchPredictor(model).predict_batch(
            np.zeros((3, 6, 5, 5), dtype=np.int8)
        )
        self.assertEqual(policy_batch.shape, (3, 25))
        self.assertEqual(wdl_batch.shape, (3, 3))

    def test_column_legacy_mapping_round_trip(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0, 1, 2] = 1
        column = 1 * 5 + 2
        action = column_to_legacy_action(board, column)
        self.assertEqual(action, COLUMN_COUNT + column)
        self.assertEqual(legacy_action_to_column(action, board), column)

        policy = np.arange(1, COLUMN_COUNT + 1, dtype=np.float32)
        policy /= policy.sum()
        legacy = column_policy_to_legacy(policy, board)
        self.assertEqual(legacy.shape, (LEGACY_ACTION_COUNT,))
        np.testing.assert_allclose(legacy_policy_to_columns(legacy, board), policy)
        self.assertAlmostEqual(float(legacy.sum()), 1.0, places=6)

    def test_wdl_and_legacy_runtime_value(self) -> None:
        self.assertAlmostEqual(float(wdl_expected_value(np.asarray([0.7, 0.2, 0.1]))), 0.6)

        class FixedPredictor:
            def predict(self, board):
                del board
                return np.full(25, 1.0 / 25, dtype=np.float32), np.asarray([0.7, 0.2, 0.1])

        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[:, 0, 0] = np.asarray([1, -1, 1, -1, 1, -1], dtype=np.int8)
        policy, value = LegacyPolicyValueAdapter(FixedPredictor()).predict(board)
        self.assertEqual(policy.shape, (150,))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertEqual(float(policy[0]), 0.0)
        self.assertAlmostEqual(value, 0.6)


class SearchTests(unittest.TestCase):
    def test_virtual_loss_lowers_parent_perspective_score(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        parent = TreeNode(board=board, visit_count=10)
        child = TreeNode(board=board, prior=0.5, visit_count=2)
        before = puct_score(parent, child, cpuct=1.0)
        child.apply_virtual_loss(1.0)
        after = puct_score(parent, child, cpuct=1.0)
        self.assertLess(after, before)
        child.revert_virtual_loss(1.0)
        self.assertEqual(child.visit_count, 2)
        self.assertEqual(child.value_sum, 0.0)

    def test_single_and_virtual_lane_search_count_exact_simulations(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        for threads in (1, 4):
            result = MCTS(RandomPredictor(), num_threads=threads).search(
                board,
                8,
                rng=np.random.default_rng(123),
            )
            self.assertEqual(int(result.visit_counts.sum()), 8)
            self.assertEqual(result.simulations, 8)
            self.assertAlmostEqual(float(result.policy.sum()), 1.0, places=6)
            self.assertTrue(np.all(result.visit_counts >= 0))

    def test_virtual_lanes_use_batch_predictor(self) -> None:
        class BatchOnlyPredictor:
            def __init__(self):
                self.batch_sizes = []

            def predict(self, board):
                raise AssertionError("batch predictor should be preferred")

            def predict_batch(self, boards):
                self.batch_sizes.append(len(boards))
                return (
                    np.full((len(boards), 25), 1.0 / 25, dtype=np.float32),
                    np.tile(np.asarray([0.5, 0.0, 0.5], dtype=np.float32), (len(boards), 1)),
                )

        predictor = BatchOnlyPredictor()
        result = MCTS(predictor, num_threads=4).search(
            np.zeros((6, 5, 5), dtype=np.int8),
            8,
            rng=np.random.default_rng(9),
        )
        self.assertEqual(result.simulations, 8)
        self.assertEqual(result.inference_calls, len(predictor.batch_sizes))
        self.assertEqual(result.max_inference_batch, max(predictor.batch_sizes))
        self.assertGreater(result.max_inference_batch, 1)


class OpeningSuiteTests(unittest.TestCase):
    def test_openings_are_deterministic_and_d4_deduplicated(self) -> None:
        first = build_openings(20, run_seed=77, prefix_lengths=(0, 2, 4, 6, 8))
        second = build_openings(20, run_seed=77, prefix_lengths=(0, 2, 4, 6, 8))
        self.assertEqual(first, second)
        self.assertEqual(sum(not opening.columns for opening in first), 1)
        game = GameRules()
        keys = []
        for opening in first:
            board = game.get_init_board()
            player = 1
            for column in opening.columns:
                action = column_to_legacy_action(board, column)
                board, player = game.get_next_state(board, player, action)
            variants = []
            for transform in range(8):
                transformed = np.rot90(board, transform % 4, axes=(-2, -1))
                if transform >= 4:
                    transformed = np.flip(transformed, axis=-1)
                variants.append(np.ascontiguousarray(transformed).tobytes())
            keys.append((player, min(variants)))
        self.assertEqual(len(keys), len(set(keys)))
        maximum_suite = build_openings(
            200,
            run_seed=314159,
            prefix_lengths=(0, 2, 4, 6, 8, 10, 12),
        )
        self.assertEqual(len(maximum_suite), 200)


class SelfPlayTests(unittest.TestCase):
    def _tiny_config(self) -> V3Config:
        config = V3Config()
        return replace(
            config,
            selfplay=replace(
                config.selfplay,
                search_schedule=(SearchStageConfig(0, 1, 4, 2, 0.0),),
                exploration_phases=(ExplorationPhaseConfig(0, 1.0, 0.0, 0.0),),
            ),
            runtime=replace(config.runtime, mcts_lanes_per_actor=2),
        )

    def test_wdl_targets_use_original_player_and_draw_is_zero_class(self) -> None:
        self.assertEqual(wdl_target_for_player(1, 1), WDL_WIN)
        self.assertEqual(wdl_target_for_player(1, -1), WDL_LOSS)
        self.assertEqual(wdl_target_for_player(-1, -1), WDL_WIN)
        self.assertEqual(wdl_target_for_player(0, 1), WDL_DRAW)
        self.assertEqual(wdl_target_for_player(0, -1), WDL_DRAW)

    def test_handwritten_first_and_second_player_wins(self) -> None:
        game = GameRules()

        def play(columns):
            board = game.get_init_board()
            player = 1
            for column in columns:
                action = column_to_legacy_action(board, column)
                board, player = game.get_next_state(board, player, action)
            return board, player

        first_win, next_player = play([0, 20, 1, 21, 2, 22, 3])
        self.assertTrue(game.check_win(first_win, 1))
        self.assertEqual(game.get_game_ended(first_win, next_player), -1)
        self.assertEqual(
            [wdl_target_for_player(1, player) for player in (1, -1, 1, -1)],
            [WDL_WIN, WDL_LOSS, WDL_WIN, WDL_LOSS],
        )

        second_win, next_player = play([20, 0, 21, 1, 22, 2, 24, 3])
        self.assertTrue(game.check_win(second_win, -1))
        self.assertEqual(game.get_game_ended(second_win, next_player), -1)
        self.assertEqual(
            [wdl_target_for_player(-1, player) for player in (1, -1, 1, -1)],
            [WDL_LOSS, WDL_WIN, WDL_LOSS, WDL_WIN],
        )

    def test_fixed_full_board_draw_trajectory(self) -> None:
        game = GameRules()
        board = game.get_init_board()
        player = 1
        state_players = []
        self.assertEqual(len(FULL_DRAW_COLUMNS), 150)
        for ply, column in enumerate(FULL_DRAW_COLUMNS):
            state_players.append(player)
            action = column_to_legacy_action(board, column)
            board, player = game.get_next_state(board, player, action)
            terminal = game.get_game_ended(board, player)
            self.assertEqual(terminal, 1e-4 if ply == 149 else 0)
        self.assertFalse(game.check_win(board, 1))
        self.assertFalse(game.check_win(board, -1))
        self.assertTrue(all(wdl_target_for_player(0, side) == WDL_DRAW for side in state_players))

    def test_selfplay_is_reproducible_and_labels_full_and_fast_samples(self) -> None:
        config = self._tiny_config()
        first = run_self_play_games(config)[0]
        second = run_self_play_games(config)[0]
        self.assertEqual(first.seed, config.run.seed)
        self.assertEqual(first.moves, second.moves)
        self.assertEqual(first.full_search_positions, 1)
        self.assertEqual(first.fast_search_positions, len(first.moves) - 1)
        self.assertEqual(len(first.samples), len(first.moves))
        full_samples = [sample for sample in first.samples if sample.search_kind == SEARCH_FULL]
        self.assertEqual(len(full_samples), 1)
        self.assertGreater(full_samples[0].ply, 0)
        self.assertLess(full_samples[0].ply, 7)
        for first_sample, second_sample, move in zip(
            first.samples, second.samples, first.moves, strict=True
        ):
            self.assertEqual(first_sample.ply, move.ply)
            self.assertEqual(first_sample.search_kind, move.search_kind)
            np.testing.assert_array_equal(first_sample.board, second_sample.board)
            np.testing.assert_array_equal(first_sample.visit_counts, second_sample.visit_counts)
            expected = wdl_target_for_player(first.winner, first_sample.player)
            self.assertEqual(first_sample.wdl, expected)

    def test_accepted_predictor_requires_model_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "producer_model_id"):
            run_self_play_games(self._tiny_config(), accepted_predictor=RandomPredictor())


if __name__ == "__main__":
    unittest.main()
