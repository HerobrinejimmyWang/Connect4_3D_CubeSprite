from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "desktop_app/backend")]

from connect4_core import GameRules  # noqa: E402
from connect4_core.rules import BAL5_R2_RULE_REGISTRY, RuleEngine, RuleFeatureSchema  # noqa: E402
from cubesprite_backend.model_runtime import ModelRegistry  # noqa: E402
from cubesprite_backend.search import RuleMCTS  # noqa: E402


class V4SearchIntegrationTests(unittest.TestCase):
    def test_16_simulations_choose_legal_move_for_every_rule_and_role(self) -> None:
        game = GameRules()
        predictor = ModelRegistry(ROOT / "desktop_app/src-tauri/resources").predictor(
            "cubesprite_v4_flash_preview1"
        )
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0, 0, 0] = 1
        board[0, 4, 4] = -1
        board[0, 2, 2] = 1
        board[1, 0, 0] = -1
        for spec in BAL5_R2_RULE_REGISTRY.specs:
            engine = RuleEngine(spec, registry=BAL5_R2_RULE_REGISTRY)
            for player in (1, -1):
                with self.subTest(rule=spec.rule_id, player=player):
                    state = engine.state_from_board(board, player_to_move=player)
                    search = RuleMCTS(
                        game,
                        engine,
                        predictor,
                        simulations=16,
                        temperature=0.5,
                        seed=13,
                        rule_features=np.asarray(RuleFeatureSchema.encode(spec), dtype=np.float32),
                    )
                    result = search.run(state)
                    legal_columns = np.flatnonzero(engine.legal_column_mask(state))
                    self.assertIn(result.action % 25, legal_columns)
                    self.assertEqual(result.action, engine.legacy_action_for_column(state, result.action % 25))
                    self.assertAlmostEqual(sum(result.policy), 1.0, places=6)
                    self.assertTrue(np.isfinite(result.value))


if __name__ == "__main__":
    unittest.main()
