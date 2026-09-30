from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "desktop_app/backend")]

from connect4_core.rules import BAL5_R2_RULE_REGISTRY  # noqa: E402
from cubesprite_backend.model_runtime import ModelRegistry  # noqa: E402


class V4FlashRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ModelRegistry(ROOT / "desktop_app/src-tauri/resources")
        cls.predictor = cls.registry.predictor("cubesprite_v4_flash_preview1")

    def test_all_five_rule_features_and_both_roles_are_live(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0, 2, 2] = 1
        board[0, 1, 2] = -1
        board[1, 2, 2] = 1
        outputs = []
        for first_player in (1, -1):
            for spec in BAL5_R2_RULE_REGISTRY.specs:
                with self.subTest(first_player=first_player, rule=spec.rule_id):
                    policy, value = self.predictor.predict(
                        board,
                        first_player=first_player,
                        rule_features=np.asarray(BAL5_R2_RULE_REGISTRY.features(spec), dtype=np.float32),
                    )
                    self.assertEqual(policy.shape, (150,))
                    self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
                    self.assertTrue(np.all(np.isfinite(policy)))
                    self.assertTrue(-1.0 <= value <= 1.0)
                    outputs.append(np.r_[policy, value])
        for start in (0, 5):
            for left in range(start, start + 5):
                for right in range(left + 1, start + 5):
                    self.assertGreater(float(np.max(np.abs(outputs[left] - outputs[right]))), 1e-4)
        self.assertGreater(float(np.max(np.abs(outputs[0] - outputs[5]))), 1e-4)

    def test_context_is_required(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        with self.assertRaises(ValueError):
            self.predictor.predict(board)

    def test_registry_support_matches_stage3_rules(self) -> None:
        models = self.registry.list_models()
        self.assertEqual(models[0]["id"], "cubesprite_v4_flash_preview1")
        self.assertEqual(
            set(models[0]["supported_rule_ids"]),
            {spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs},
        )
        for model in models[1:]:
            self.assertEqual(model["supported_rule_ids"], ["classic"])


if __name__ == "__main__":
    unittest.main()
