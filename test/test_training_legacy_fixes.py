import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from game_rules import GameRules  # noqa: E402
from mcts import MCTS, TreeNode  # noqa: E402
from parallel_games import play_self_play_game  # noqa: E402
from trainer import Trainer  # noqa: E402


class UniformInference:
    def submit_and_wait(self, state):
        return np.full(150, 1.0 / 150, dtype=np.float32), 0.0


def _mcts_args(**overrides):
    values = {
        "cpuct": 1.0,
        "num_mcts_sims": 8,
        "num_mcts_threads": 1,
        "virtual_loss": 1.0,
        "inference_batch_size": 8,
        "inference_timeout_s": 0.001,
        "reuse_mcts_tree": True,
        "persistent_mcts_threads": True,
        "enable_mcts_search_stats": True,
        "dirichlet_alpha": 0.0,
        "dirichlet_epsilon": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LegacyMCTSVirtualLossTests(unittest.TestCase):
    def test_virtual_loss_lowers_parent_puct_and_reverts_exactly(self):
        game = GameRules()
        mcts = MCTS(game, UniformInference(), _mcts_args())
        try:
            root = TreeNode(game.get_init_board())
            root.visit_count = 2
            occupied = TreeNode(root.state.copy(), prior_probability=0.5, parent=root, action_from_parent=0)
            available = TreeNode(root.state.copy(), prior_probability=0.5, parent=root, action_from_parent=1)
            root.children = {0: occupied, 1: available}

            occupied.add_virtual_loss(1.0)
            self.assertGreater(occupied.q_value(), 0.0)
            self.assertIs(mcts._select_child_with_puct(root), available)

            occupied.revert_virtual_loss(1.0)
            self.assertEqual(occupied.virtual_loss_count, 0)
            self.assertEqual(occupied.visit_count, 0)
            self.assertEqual(occupied.value_sum, 0.0)
        finally:
            mcts.close()

    def test_single_and_multi_thread_search_complete_exact_budget(self):
        game = GameRules()
        for thread_count in (1, 4):
            with self.subTest(thread_count=thread_count):
                mcts = MCTS(game, UniformInference(), _mcts_args(num_mcts_threads=thread_count))
                try:
                    policy = np.asarray(mcts.get_action_prob(game.get_init_board(), temp=1, training=False))
                    self.assertEqual(mcts.get_search_stats()["simulations"], 8)
                    self.assertTrue(np.isfinite(policy).all())
                    self.assertAlmostEqual(float(policy.sum()), 1.0, places=7)
                    self.assertTrue(np.all(policy >= 0.0))
                finally:
                    mcts.close()


class _OneMoveDrawGame:
    def get_init_board(self):
        return np.zeros((6, 5, 5), dtype=np.int8)

    def get_canonical_form(self, board, player):
        return np.asarray(board) * int(player)

    def get_valid_moves(self, board):
        moves = np.zeros(150, dtype=np.int8)
        moves[0] = 1
        return moves

    def get_next_state(self, board, player, action):
        next_board = np.array(board, copy=True)
        next_board.flat[0] = int(player)
        return next_board, -int(player)

    def get_game_ended(self, board, player):
        return 1e-4 if np.count_nonzero(board) else 0.0

    def check_win(self, board, player):
        return False

    def get_symmetries(self, board, policy):
        return [(np.asarray(board), np.asarray(policy))]


class _OneActionMCTS:
    def __init__(self, game, predictor, args):
        pass

    def get_action_prob(self, board, **kwargs):
        policy = np.zeros(150, dtype=np.float64)
        policy[0] = 1.0
        return policy

    def get_search_stats(self):
        return {"simulations": 1}

    def close(self):
        pass


class LegacyTargetAndEvaluationTests(unittest.TestCase):
    def test_draw_sentinel_produces_exact_zero_value_target(self):
        args = SimpleNamespace(
            current_iteration=1,
            min_game_steps_start_iteration=999,
            min_game_steps=0,
            self_play_phase_schedule=[],
            exploration_iteration_schedule=[],
            self_play_exploration_strength=1.0,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            tactical_override_max_step=0,
        )
        with patch("parallel_games.MCTS", _OneActionMCTS):
            result = play_self_play_game(_OneMoveDrawGame(), object(), args, seed=7)

        self.assertTrue(result["trace"]["is_draw"])
        self.assertEqual(result["trace"]["result_code"], 1e-4)
        self.assertTrue(result["examples"])
        self.assertTrue(all(value == 0.0 for _, _, value in result["examples"]))

    def test_trainer_accepts_detailed_tuple_and_reports_side_wdl(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            eval_games=4,
            best_eval_games_per_generation=4,
            best_update_threshold=0.75,
            update_threshold=0.75,
            best_eval_required_generations=1,
            best_eval_parallelize_generations=False,
            enable_random_baseline_eval=False,
        )
        trainer.best_nnet = object()
        trainer.best_model_label = "incumbent"
        trainer.older_best_nnet = None
        trainer.eval_history = []
        trainer._evaluate_against_auxiliary_model = lambda iteration: None

        details = [
            {"game_index": 0, "outcome": "win", "candidate_player": 1},
            {"game_index": 1, "outcome": "loss", "candidate_player": -1},
            {"game_index": 2, "outcome": "draw", "candidate_player": 1},
            {"game_index": 3, "outcome": "win", "candidate_player": -1},
        ]

        def fake_evaluation(num_games, opponent_nnet=None, opponent_model_spec=None, return_details=False):
            self.assertTrue(return_details)
            return 2, 1, 1, details

        trainer.execute_evaluation_parallel = fake_evaluation
        result = trainer.evaluate_model(iteration=3)

        self.assertEqual(result["overall"], {
            "wins": 2, "losses": 1, "draws": 1, "games": 4, "win_rate": 0.625,
        })
        self.assertEqual(result["candidate_as_first"], {
            "wins": 1, "losses": 0, "draws": 1, "games": 2, "win_rate": 0.75,
        })
        self.assertEqual(result["candidate_as_second"], {
            "wins": 1, "losses": 1, "draws": 0, "games": 2, "win_rate": 0.5,
        })
        self.assertEqual(result["opponent_results"][0]["candidate_as_second"]["losses"], 1)


if __name__ == "__main__":
    unittest.main()
