import copy
import unittest

import numpy as np
import torch
from torch import nn

from connect4_core.rules import DEFAULT_RULE_REGISTRY
from training.v3.learner import OnlineD4Dataset, V3Learner, build_adamw
from training.v3.replay import ReplayShard, TrainTokenBucket, apply_d4


def _place_sample() -> dict:
    terminal = np.zeros((6, 5, 5), dtype=np.int8)
    terminal[0, 0, 0] = 1
    terminal[0, 0, 1] = -1
    visits = np.zeros((25,), dtype=np.uint32)
    visits[6] = 4
    return {
        "board": np.zeros((6, 5, 5), dtype=np.int8),
        "visit_counts": visits,
        "policy_weight": 1.0,
        "wdl": 2,
        "game_id": 1,
        "turn_index": 0,
        "player_to_move": -1,
        "search_kind": "full",
        "rule_code": 0,
        "turn_kind": "place",
        "placement_count": 0,
        "opponent_reply_column": 3,
        "opponent_reply_mask": 1,
        "terminal_board": terminal,
        "remaining_turns": 7,
    }


def _forced_pass_sample() -> dict:
    # Under Rule2, every next placement would complete a forbidden first-player
    # vertical four, so the only game action is a forced pass.
    board = np.zeros((6, 5, 5), dtype=np.int8)
    board[:3, :, :] = 1
    return {
        "board": board,
        "visit_counts": np.zeros((25,), dtype=np.uint32),
        "policy_weight": 0.0,
        "wdl": 1,
        "game_id": 2,
        "turn_index": 75,
        "player_to_move": 1,
        "search_kind": "none",
        "rule_code": 2,
        "turn_kind": "forced_pass",
        "placement_count": 75,
        "opponent_reply_column": -1,
        "opponent_reply_mask": 0,
        "terminal_board": board,
        "remaining_turns": 1,
    }


class _TinyAuxNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(150, 16), nn.Tanh())
        self.policy = nn.Linear(16, 25)
        self.wdl = nn.Linear(16, 3)
        self.reply = nn.Linear(16, 25)
        self.occupancy = nn.Linear(16, 3 * 6 * 5 * 5)
        self.moves_left = nn.Linear(16, 301)
        self.last_role = None
        self.last_rules = None

    def forward(self, board, *, role_to_play, rule_features):
        self.last_role = role_to_play.detach().cpu().clone()
        self.last_rules = rule_features.detach().cpu().clone()
        hidden = self.trunk(board)
        return {
            "policy_logits": self.policy(hidden),
            "wdl_logits": self.wdl(hidden),
            "opponent_reply_logits": self.reply(hidden),
            "future_occupancy_logits": self.occupancy(hidden).reshape(-1, 3, 6, 5, 5),
            "moves_left_logits": self.moves_left(hidden),
        }


class _OneShotSkippingScaler:
    def __init__(self) -> None:
        self.scale_value = 1024.0
        self.skip_next_step = True

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        if not self.skip_next_step:
            optimizer.step()

    def update(self):
        if self.skip_next_step:
            self.scale_value /= 2.0
            self.skip_next_step = False


def _learner(model: nn.Module) -> V3Learner:
    optimizer = build_adamw(model, learning_rate=1e-3, weight_decay=0.0)
    return V3Learner(
        model,
        optimizer,
        device="cpu",
        batch_size=2,
        grad_clip_norm=1.0,
        sample_seed=31,
        opponent_reply_loss_weight=0.15,
        future_occupancy_loss_weight=0.15,
        moves_left_loss_weight=0.05,
        future_occupancy_class_weights=(1.2, 1.3, 0.5),
    )


class ReplayV2LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_dataset_builds_global_inputs_and_d4_auxiliary_targets(self):
        sample = _place_sample()
        item = OnlineD4Dataset(
            ReplayShard.from_samples([sample]), augmentation_seed=99
        )[(0, 17)]
        self.assertTrue(torch.equal(item["role_to_play"], torch.tensor([0.0, 1.0])))
        self.assertTrue(
            torch.equal(
                item["rule_features"],
                torch.tensor(DEFAULT_RULE_REGISTRY.features(0), dtype=torch.float32),
            )
        )

        reply = np.zeros((5, 5), dtype=np.uint8)
        reply.reshape(25)[sample["opponent_reply_column"]] = 1
        expected_reply = int(np.argmax(apply_d4(reply, item["d4"]).reshape(25)))
        self.assertEqual(int(item["opponent_reply"]), expected_reply)

        transformed_terminal = apply_d4(sample["terminal_board"], item["d4"])
        canonical_terminal = transformed_terminal * sample["player_to_move"]
        expected_occupancy = np.where(
            canonical_terminal > 0, 0, np.where(canonical_terminal < 0, 1, 2)
        )
        np.testing.assert_array_equal(item["future_occupancy"].numpy(), expected_occupancy)
        self.assertTrue(
            torch.equal(item["future_occupancy_mask"], item["board"].eq(0))
        )

    def test_forced_pass_has_zero_policy_and_rule_aware_empty_legal_mask(self):
        item = OnlineD4Dataset(
            ReplayShard.from_samples([_forced_pass_sample()]), augmentation_seed=5
        )[0]
        self.assertEqual(float(item["policy"].sum()), 0.0)
        self.assertEqual(float(item["policy_weight"]), 0.0)
        self.assertFalse(bool(item["legal_mask"].any()))

    def test_auxiliary_losses_metrics_and_global_model_inputs(self):
        replay = ReplayShard.from_samples([_place_sample(), _forced_pass_sample()])
        dataset = OnlineD4Dataset(replay, augmentation_seed=7)
        model = _TinyAuxNet()
        learner = _learner(model)
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(2)
        metrics = learner.train_steps(dataset, steps=1, token_bucket=bucket)

        self.assertEqual(metrics.steps, 1)
        self.assertEqual(metrics.positions, 2)
        self.assertEqual(metrics.policy_positions, 1)
        self.assertEqual(metrics.opponent_reply_positions, 1)
        self.assertGreater(metrics.future_occupancy_cells, 0)
        self.assertEqual(metrics.moves_left_positions, 2)
        for value in (
            metrics.policy_loss,
            metrics.wdl_loss,
            metrics.opponent_reply_loss,
            metrics.future_occupancy_loss,
            metrics.moves_left_loss,
            metrics.total_loss,
        ):
            self.assertTrue(np.isfinite(value))
        self.assertEqual(tuple(model.last_role.shape), (2, 2))
        self.assertEqual(tuple(model.last_rules.shape), (2, 32))

    def test_amp_skip_preserves_cursor_tokens_and_model_with_auxiliary_losses(self):
        replay = ReplayShard.from_samples([_place_sample(), _forced_pass_sample()])
        dataset = OnlineD4Dataset(replay, augmentation_seed=7)
        model = _TinyAuxNet()
        learner = _learner(model)
        learner.amp_enabled = True
        learner.scaler = _OneShotSkippingScaler()
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(2)
        initial_bucket = bucket.state_dict()
        initial_model = copy.deepcopy(model.state_dict())

        metrics = learner.train_steps(dataset, steps=1, token_bucket=bucket)
        self.assertEqual(metrics.steps, 0)
        self.assertEqual(metrics.grad_norm, 0.0)
        self.assertEqual(learner.global_step, 0)
        self.assertEqual(learner.sample_cursor, 0)
        self.assertEqual(bucket.state_dict(), initial_bucket)
        for name, expected in initial_model.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
