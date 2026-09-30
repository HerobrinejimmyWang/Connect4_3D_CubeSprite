from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "desktop_app" / "backend")]

from cubesprite_backend.service import CubeSpriteService, ServiceError  # noqa: E402
from cubesprite_backend.search import RuleMCTS  # noqa: E402
from connect4_core import GameRules  # noqa: E402
from connect4_core.rules import BAL5_R2_RULE_REGISTRY, RuleEngine  # noqa: E402


class V020IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.service = CubeSpriteService(ROOT / "desktop_app" / "src-tauri" / "resources", Path(self.temporary.name))

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    @staticmethod
    def token(state):
        return {"session_id": state["session_id"], "expected_revision": state["revision"]}

    def test_training_sample_import_and_continue(self):
        sample = Path(__file__).parent / "fixtures" / "training-v2-sample.c4replay.json"
        summary = self.service.handle("replay.import", {"path": str(sample)})["replay"]
        self.assertEqual(summary["turn_count"], 11)
        self.assertEqual(summary["rule_id"], "classic")
        ref = {"id": summary["id"], "expected_fingerprint": summary["fingerprint"]}
        opened = self.service.handle("replay.open", ref)
        self.assertEqual(len(opened["frames"]), 12)
        continued = self.service.handle("replay.continue", ref | {"step": 4})
        self.assertEqual(continued["rule_id"], "classic")
        self.assertEqual(continued["move_count"], 4)

    def test_layer0_rule_changes_first_player_win_only(self):
        sequence = [(0, 0), (4, 0), (0, 1), (4, 1), (0, 2), (4, 2), (0, 3)]
        for rule_id, expected in (("classic", "won"), ("p1_layer0_ignored", "playing"), ("p1_vertical_and_layer0_ignored", "playing")):
            state = self.service.handle("game.new", {"mode": "pvp", "rule_id": rule_id})
            for row, col in sequence:
                state = self.service.handle("game.move", self.token(state) | {"layer": 0, "row": row, "col": col})
            self.assertEqual(state["status"], expected)
            summary = self.service.handle("replay.save", self.token(state))["replay"]
            self.assertEqual(summary["rule_id"], rule_id)
            opened = self.service.handle("replay.open", {"id": summary["id"], "expected_fingerprint": summary["fingerprint"]})
            self.assertEqual(opened["frames"][-1]["status"], expected)

    def test_forbidden_column_hidden_and_rejected(self):
        state = self.service.handle("game.new", {"mode": "pvp", "rule_id": "p1_vertical_forbidden"})
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[:3, 0, 0] = 1
        self.service._state = self.service.engine.state_from_board(board, player_to_move=1)
        self.service.board = board
        self.service.current_player = 1
        shown = self.service.snapshot()
        self.assertNotIn(75, {move["action"] for move in shown["legal_moves"]})
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("game.move", self.token(state) | {"layer": 3, "row": 0, "col": 0})
        self.assertEqual(raised.exception.code, "ILLEGAL_MOVE")

    def test_nonclassic_rejects_legacy_model(self):
        self.service.handle("settings.update", self.token(self.service.snapshot()) | {"roles": {"combat": {"model_id": "v2.2_balance"}}})
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("game.new", {"mode": "pvai", "rule_id": "p1_vertical_ignored"})
        self.assertEqual(raised.exception.code, "MODEL_UNSUPPORTED_RULE")

    def test_new_game_ai_override_is_used_for_replay_provenance(self):
        state = self.service.handle("game.new", {"mode": "pvai", "human_player": 1, "ai": {"model_id": "v2.2_balance"}})
        state = self.service.handle("game.move", self.token(state) | {"layer": 0, "row": 0, "col": 0})
        summary = self.service.handle("replay.save", self.token(state))["replay"]
        opened = self.service.handle("replay.open", {"id": summary["id"], "expected_fingerprint": summary["fingerprint"]})
        second = opened["replay"]["participants"][1]
        self.assertEqual(second["model_id"], "v2.2_balance")
        self.assertEqual(len(second["artifact_sha256"]), 64)

    def test_forced_block_uses_model_value(self):
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0, 4, 1:4] = -1
        engine = RuleEngine("classic", registry=BAL5_R2_RULE_REGISTRY)
        state = engine.state_from_board(board, player_to_move=1)

        class Predictor:
            def predict(self, _board):
                return np.ones(150) / 150, -0.25

        result = RuleMCTS(GameRules(), engine, Predictor(), simulations=16).run(state)
        self.assertIn(result.action, {20, 24})
        self.assertAlmostEqual(result.value, -0.25)


if __name__ == "__main__":
    unittest.main()
