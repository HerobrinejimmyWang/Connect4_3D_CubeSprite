from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for path in (PROJECT_ROOT / "training", PROJECT_ROOT / "train_features"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from connect4_core import BOARD_SHAPE, GameRules, action_to_coords, coords_to_action
from connect4_runtime import ModelRegistry, load_v22_agent, validate_v22_checkpoint
from game_client.app import GameClientApp
from game_client.state import HumanGameController
from training.experimental_models import GravityPolicyValueNet
from training.model import Connect4Net
from train_features.tiny_policy_model import TinyCandidatePolicyNet


class FakeAgent:
    def __init__(self, action=0):
        self.action = action
        self.closed = False

    def get_action(self, board, player, temp=0):
        return self.action, {}

    def close(self):
        self.closed = True


class SharedRulesTests(unittest.TestCase):
    def test_default_board_and_initial_moves(self):
        game = GameRules()
        board = game.get_init_board()
        self.assertEqual(board.shape, BOARD_SHAPE)
        self.assertEqual(board.dtype, np.int8)
        valid = game.get_valid_moves(board)
        self.assertEqual(int(valid.sum()), 25)
        self.assertTrue(np.all(valid[:25] == 1))
        self.assertTrue(np.all(valid[25:] == 0))

    def test_action_roundtrip_and_validation(self):
        for action in (0, 24, 25, 149):
            coords = action_to_coords(action)
            self.assertEqual(coords_to_action(*coords), action)
        with self.assertRaises(ValueError):
            action_to_coords(150)
        with self.assertRaises(ValueError):
            coords_to_action(6, 0, 0)

    def test_gravity_and_occupied_moves_raise(self):
        game = GameRules()
        board = game.get_init_board()
        with self.assertRaisesRegex(ValueError, "gravity"):
            game.get_next_state(board, 1, game.coords_to_action(1, 0, 0))
        board, _ = game.get_next_state(board, 1, game.coords_to_action(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "occupied"):
            game.get_next_state(board, -1, game.coords_to_action(0, 0, 0))
        invalid = game.get_init_board().astype(np.float32)
        invalid[0, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, r"only -1, 0, or \+1"):
            game.get_valid_moves(invalid)

    def test_horizontal_vertical_and_space_diagonal_wins(self):
        game = GameRules()
        horizontal = game.get_init_board()
        horizontal[0, 2, :4] = 1
        self.assertTrue(game.check_win(horizontal, 1))
        vertical = game.get_init_board()
        vertical[:4, 1, 1] = -1
        self.assertTrue(game.check_win(vertical, -1))
        diagonal = game.get_init_board()
        for index in range(4):
            diagonal[index, index, index] = 1
        self.assertTrue(game.check_win(diagonal, 1))

    def test_canonical_symmetries_and_shared_exports(self):
        game = GameRules()
        board = game.get_init_board()
        board[0, 0, 0] = -1
        np.testing.assert_array_equal(game.get_canonical_form(board, -1), -board)
        self.assertEqual(len(game.get_symmetries(board, np.zeros(150))), 8)
        from training.game_rules import GameRules as TrainingRules
        from arena.arena_game_rules import GameRules as ArenaRules
        self.assertIs(TrainingRules, GameRules)
        self.assertIs(ArenaRules, GameRules)


class ControllerTests(unittest.TestCase):
    def test_human_first_then_ai(self):
        game = GameRules()
        agent = FakeAgent(game.coords_to_action(0, 0, 1))
        controller = HumanGameController(agent, human_player=1, game=game)
        controller.human_move(0, 0, 0)
        self.assertTrue(controller.is_ai_turn)
        request = controller.begin_ai_turn()
        self.assertIsNotNone(request)
        self.assertTrue(controller.finish_ai_turn(agent.action))
        self.assertTrue(controller.is_human_turn)

    def test_ai_first_and_illegal_ai_action(self):
        controller = HumanGameController(FakeAgent(149), human_player=-1)
        self.assertTrue(controller.is_ai_turn)
        controller.begin_ai_turn()
        self.assertFalse(controller.finish_ai_turn(149))
        self.assertEqual(controller.status, "error")
        controller.reset()
        self.assertTrue(controller.is_ai_turn)

    def test_finished_game_rejects_more_human_moves_and_close(self):
        agent = FakeAgent()
        controller = HumanGameController(agent, human_player=1)
        controller.board[0, 0, :3] = 1
        controller.human_move(0, 0, 3)
        self.assertEqual(controller.status, "won")
        with self.assertRaises(RuntimeError):
            controller.human_move(0, 1, 0)
        controller.close()
        self.assertTrue(agent.closed)


class RegistryAndModelTests(unittest.TestCase):
    def test_registry_only_scans_explicit_roots_and_skips_temporary_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "good.pth").write_bytes(b"ok")
            (root / "writing.pth.tmp").write_bytes(b"partial")
            (root / "empty.pth").touch()
            items = ModelRegistry([root], settle_seconds=0).discover()
            self.assertEqual([item.path.name for item in items], ["good.pth"])

    def _save_checkpoints(self, root):
        standard_path = root / "standard.pth"
        standard = Connect4Net(board_layers=6, board_size=5, num_channels=4, num_res_blocks=1)
        torch.save(standard.state_dict(), standard_path)

        gravity_path = root / "gravity.pth.tar"
        gravity = GravityPolicyValueNet(num_channels=8, num_res_blocks=1, backbone_type="layer2d")
        torch.save({"state_dict": gravity.state_dict(), "student_model_config": gravity.model_config()}, gravity_path)

        tiny_path = root / "tiny.pth"
        tiny_config = {
            "architecture": "tiny-candidate-policy-v1", "board_layers": 6, "board_size": 5,
            "candidate_count": 25, "global_dim": 20, "candidate_dim": 28,
            "global_hidden": 4, "candidate_hidden": 4, "fusion_hidden": 4,
            "dropout": 0.0, "value_hidden": 4,
        }
        tiny = TinyCandidatePolicyNet(
            global_dim=20, candidate_dim=28, global_hidden=4, candidate_hidden=4,
            fusion_hidden=4, dropout=0.0, value_hidden=4,
        )
        torch.save({"model_state_dict": tiny.state_dict(), "model_config": tiny_config}, tiny_path)
        return standard_path, gravity_path, tiny_path

    def test_standard_gravity_and_tiny_models_validate_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._save_checkpoints(Path(temp_dir))
            game = GameRules()
            for path in paths:
                config = validate_v22_checkpoint(path, game)
                self.assertEqual(int(config["action_dim"]), 150)
                agent = load_v22_agent(game, path, device="cpu", num_mcts_sims=1, num_mcts_threads=1)
                agent.close()

    def test_eight_layer_and_corrupt_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.pth"
            torch.save(Connect4Net(board_layers=8, board_size=5, num_channels=4, num_res_blocks=1).state_dict(), old_path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_v22_checkpoint(old_path, GameRules())
            corrupt = root / "corrupt.pth"
            corrupt.write_bytes(b"not a checkpoint")
            with self.assertRaises(Exception):
                validate_v22_checkpoint(corrupt, GameRules())


class PygameSmokeTests(unittest.TestCase):
    def test_launcher_runs_with_dummy_video_driver(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        with tempfile.TemporaryDirectory() as temp_dir:
            app = GameClientApp([temp_dir], width=800, height=600)
            app.run(max_frames=2)
            self.assertFalse(app.running)


if __name__ == "__main__":
    unittest.main()
