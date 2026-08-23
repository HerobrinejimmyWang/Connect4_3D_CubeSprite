from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from training.v3.config import ModelConfig, V3Config, config_hash
from training.v3.model import (
    GLOBAL_INPUT_SCHEMA,
    MOVES_LEFT_CLASSES,
    OUTPUT_SCHEMA,
    ROLE_FEATURE_NAMES,
    RULE_FEATURE_NAMES,
    ModelOutput,
    SearchOutput,
    TorchPredictor,
    build_model,
    classic_rule_features,
)


class ModelContractConfigTests(unittest.TestCase):
    def test_old_config_shape_decodes_to_frozen_contract(self) -> None:
        config = V3Config.from_dict(
            {
                "model": {"architecture": "gravity_resnet", "channels": 8, "blocks": 1},
                "learner": {"batch_size": 4},
            }
        )
        self.assertEqual(config.model.global_input_schema, GLOBAL_INPUT_SCHEMA)
        self.assertEqual(config.model.output_schema, OUTPUT_SCHEMA)
        self.assertEqual(config.model.rule_feature_dim, 32)
        self.assertEqual(config.model.moves_left_classes, MOVES_LEFT_CLASSES)
        self.assertEqual(config.learner.opponent_reply_loss_weight, 0.15)

    def test_auxiliary_loss_weights_are_semantic(self) -> None:
        config = V3Config()
        changed = replace(
            config,
            learner=replace(config.learner, future_occupancy_loss_weight=0.2),
        )
        self.assertNotEqual(config_hash(config), config_hash(changed))
        with self.assertRaisesRegex(ValueError, "loss weights"):
            replace(config.learner, moves_left_loss_weight=-0.1)

    def test_fixed_schema_dimensions_are_rejected_if_changed(self) -> None:
        with self.assertRaisesRegex(ValueError, "rule_feature_dim"):
            ModelConfig(rule_feature_dim=31)
        with self.assertRaisesRegex(ValueError, "moves_left_classes"):
            ModelConfig(moves_left_classes=300)
        self.assertEqual(ROLE_FEATURE_NAMES, ("to_play_first", "to_play_second"))
        self.assertEqual(len(RULE_FEATURE_NAMES), 32)


class ModelContractForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = build_model(ModelConfig(channels=8, blocks=1))
        self.boards = torch.zeros((2, 6, 5, 5), dtype=torch.float32)
        self.roles = torch.tensor(((1.0, 0.0), (0.0, 1.0)))
        self.rules = classic_rule_features(2)

    def test_training_forward_has_named_fixed_shape_heads(self) -> None:
        output = self.model(
            self.boards,
            role_to_play=self.roles,
            rule_features=self.rules,
        )
        self.assertIsInstance(output, ModelOutput)
        self.assertEqual(tuple(output.policy_logits.shape), (2, 25))
        self.assertEqual(tuple(output.wdl_logits.shape), (2, 3))
        self.assertEqual(tuple(output.opponent_reply_logits.shape), (2, 25))
        self.assertEqual(tuple(output.future_occupancy_logits.shape), (2, 3, 6, 5, 5))
        self.assertEqual(tuple(output.moves_left_logits.shape), (2, 301))
        policy, wdl = output
        self.assertIs(policy, output.policy_logits)
        self.assertIs(wdl, output.wdl_logits)

    def test_context_is_mandatory_and_validated(self) -> None:
        with self.assertRaises(TypeError):
            self.model(self.boards)
        with self.assertRaisesRegex(ValueError, "one-hot"):
            self.model(
                self.boards,
                role_to_play=torch.ones((2, 2)),
                rule_features=self.rules,
            )
        with self.assertRaisesRegex(ValueError, "rule_features"):
            self.model(
                self.boards,
                role_to_play=self.roles,
                rule_features=torch.zeros((2, 31)),
            )

    def test_search_forward_does_not_execute_auxiliary_heads(self) -> None:
        calls: list[str] = []
        hooks = [
            head.register_forward_hook(lambda _module, _args, _output, name=name: calls.append(name))
            for name, head in (
                ("reply", self.model.opponent_reply_head),
                ("occupancy", self.model.future_occupancy_head),
                ("moves", self.model.moves_left_head),
            )
        ]
        try:
            output = self.model.forward_search(
                self.boards,
                role_to_play=self.roles,
                rule_features=self.rules,
            )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertIsInstance(output, SearchOutput)
        self.assertEqual(tuple(output.policy_logits.shape), (2, 25))
        self.assertEqual(tuple(output.wdl_logits.shape), (2, 3))
        self.assertEqual(calls, [])

    def test_predictor_requires_and_forwards_context(self) -> None:
        predictor = TorchPredictor(self.model)
        boards = np.zeros((2, 6, 5, 5), dtype=np.int8)
        policy, wdl = predictor.predict_batch(
            boards,
            role_to_play=self.roles.numpy(),
            rule_features=self.rules.numpy(),
        )
        self.assertEqual(policy.shape, (2, 25))
        self.assertEqual(wdl.shape, (2, 3))
        np.testing.assert_allclose(policy.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(wdl.sum(axis=1), 1.0, atol=1e-6)
        with self.assertRaises(TypeError):
            predictor.predict_batch(boards)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
