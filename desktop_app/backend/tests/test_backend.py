from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "desktop_app" / "backend"
RESOURCE_DIR = REPO_ROOT / "desktop_app" / "src-tauri" / "resources"
for import_root in (REPO_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from connect4_core import GameRules  # noqa: E402
from cubesprite_backend.model_runtime import ModelRegistry  # noqa: E402
from cubesprite_backend.search import NumpyMCTS, SearchResult  # noqa: E402
from cubesprite_backend.service import CubeSpriteService, ServiceError, find_winning_line  # noqa: E402


class ModelRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ModelRegistry(RESOURCE_DIR)
        cls.game = GameRules()

    def test_manifest_and_real_models_load(self) -> None:
        models = {item["id"]: item for item in self.registry.list_models()}
        expected_identities = {
            "cubesprite_v3": (
                "c6394b1ddcc7393fba5c30a83cffa5e21787be2d3ff1cd0c6848a7f4cdc95b76",
                192,
            ),
            "cubesprite_v3_mini": (
                "c991f73b241d67e7c2eea42812645e8335f952ba682b941f1113b63f5db1a94a",
                208,
            ),
            "v2.2_balance": (
                "bb8cc0c6042276dfa3954e67b71f1fd43f603f9d6d9a0492412726cc41d30712",
                None,
            ),
            "v2.1_high": (
                "d2b761e40bdccc40e8745589605dc46951cfb240ff357439a98c11035892bfa1",
                None,
            ),
        }
        for model_id in ("cubesprite_v3", "cubesprite_v3_mini", "v2.2_balance", "v2.1_high"):
            self.assertTrue(models[model_id]["available"], models[model_id]["unavailable_reason"])
            expected_hash, expected_iteration = expected_identities[model_id]
            self.assertEqual(models[model_id]["artifact_sha256"], expected_hash)
            self.assertEqual(models[model_id]["source_iteration"], expected_iteration)
            artifact = RESOURCE_DIR / models[model_id]["model_path"]
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), expected_hash)

        board = self.game.get_init_board()
        for model_id in ("cubesprite_v3", "cubesprite_v3_mini", "v2.2_balance", "v2.1_high"):
            with self.subTest(model_id=model_id):
                policy, value = self.registry.predictor(model_id).predict(board)
                self.assertEqual(policy.shape, (150,))
                self.assertTrue(np.all(np.isfinite(policy)))
                self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
                self.assertTrue(-1.0 <= value <= 1.0)

    def test_v21_adapter_pads_layers_and_crops_policy(self) -> None:
        predictor = self.registry.predictor("v2.1_high")
        board = self.game.get_init_board()
        board[0, 0, 0] = 1
        board[5, 4, 4] = -1
        encoded = predictor._encode(board)
        self.assertEqual(encoded.shape, (1, 1, 8, 5, 5))
        np.testing.assert_array_equal(encoded[0, 0, :6], board.astype(np.float32))
        np.testing.assert_array_equal(encoded[0, 0, 6:], np.zeros((2, 5, 5), dtype=np.float32))
        policy, _ = predictor.predict(board)
        self.assertEqual(policy.shape, (150,))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)

    def test_real_models_produce_legal_action_with_32_mcts(self) -> None:
        board = self.game.get_init_board()
        valid = self.game.get_valid_moves(board)
        for model_id in ("cubesprite_v3", "cubesprite_v3_mini", "v2.2_balance", "v2.1_high"):
            with self.subTest(model_id=model_id):
                result = NumpyMCTS(
                    self.game,
                    self.registry.predictor(model_id),
                    simulations=32,
                    temperature=0,
                    seed=7,
                ).run(board, 1)
                self.assertEqual(int(valid[result.action]), 1)
                self.assertAlmostEqual(sum(result.policy), 1.0, places=6)


class ServiceStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = CubeSpriteService(RESOURCE_DIR)

    @staticmethod
    def token(state: dict) -> dict:
        return {"session_id": state["session_id"], "expected_revision": state["revision"]}

    def fake_search(self, board: np.ndarray, _player: int, _ai: dict) -> SearchResult:
        action = int(np.flatnonzero(self.service.game.get_valid_moves(board) > 0)[0])
        policy = np.zeros(self.service.game.get_action_size(), dtype=float)
        policy[action] = 1.0
        return SearchResult(action=action, policy=policy.tolist(), value=0.0)

    def test_pvai_blue_cannot_undo_opening_ai_move(self) -> None:
        state = self.service.handle("game.new", {"mode": "pvai", "human_player": -1})
        self.assertEqual(state["current_player"], 1)
        self.assertFalse(state["can_undo"])
        self.service._search = self.fake_search

        state = self.service.handle("game.ai_move", self.token(state))
        self.assertEqual(state["move_count"], 1)
        self.assertEqual(state["current_player"], -1)
        self.assertFalse(state["can_undo"])
        unchanged = self.service.handle("game.undo", self.token(state))
        self.assertEqual(unchanged["revision"], state["revision"])
        self.assertEqual(unchanged["move_count"], 1)

        params = self.token(state) | {"layer": 0, "row": 0, "col": 1}
        state = self.service.handle("game.move", params)
        self.assertTrue(state["can_undo"])
        state = self.service.handle("game.ai_move", self.token(state))
        revision_before_undo = state["revision"]
        state = self.service.handle("game.undo", self.token(state))
        self.assertEqual(state["revision"], revision_before_undo + 1)
        self.assertEqual(state["move_count"], 1)
        self.assertEqual(state["current_player"], -1)
        self.assertFalse(state["can_undo"])
        self.assertEqual(state["board"][0][0][0], 1)

    def test_revision_rejects_stale_move_and_analysis(self) -> None:
        state = self.service.handle("game.new", {"mode": "pvp", "human_player": 1})
        stale_token = self.token(state)
        moved = self.service.handle(
            "game.move", stale_token | {"layer": 0, "row": 0, "col": 0}
        )
        self.assertGreater(moved["revision"], state["revision"])
        with self.assertRaises(ServiceError) as raised:
            self.service.handle(
                "game.move", stale_token | {"layer": 0, "row": 0, "col": 1}
            )
        self.assertEqual(raised.exception.code, "STALE_REVISION")

        board, player, revision, session_id, _ai = self.service._capture_analysis(
            self.token(moved), "hint"
        )
        self.assertEqual(board.shape, (6, 5, 5))
        self.assertEqual(player, -1)
        restarted = self.service.handle("game.restart", self.token(moved))
        self.assertEqual(restarted["session_id"], session_id)
        with self.assertRaises(ServiceError) as raised:
            self.service._assert_fresh(session_id, revision)
        self.assertEqual(raised.exception.code, "STALE_REVISION")

    def test_service_reports_stable_winning_line(self) -> None:
        state = self.service.handle("game.new", {"mode": "pvp", "human_player": 1})
        moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]
        for row, col in moves:
            state = self.service.handle(
                "game.move", self.token(state) | {"layer": 0, "row": row, "col": col}
            )
        self.assertEqual(state["status"], "won")
        self.assertEqual(state["winner"], 1)
        self.assertEqual(
            state["winning_line"],
            [
                {"layer": 0, "row": 0, "col": 0},
                {"layer": 0, "row": 0, "col": 1},
                {"layer": 0, "row": 0, "col": 2},
                {"layer": 0, "row": 0, "col": 3},
            ],
        )
        self.assertEqual(state["legal_moves"], [])

    def test_find_winning_line_covers_all_13_axes(self) -> None:
        directions = [
            (dz, dy, dx)
            for dz in (0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)
            and not (dz == 0 and dy < 0)
            and not (dz == 0 and dy == 0 and dx < 0)
        ]
        self.assertEqual(len(directions), 13)
        for direction in directions:
            with self.subTest(direction=direction):
                start = tuple(3 if delta < 0 else 0 if delta > 0 else 1 for delta in direction)
                expected = [
                    tuple(start[index] + step * direction[index] for index in range(3))
                    for step in range(4)
                ]
                board = np.zeros((6, 5, 5), dtype=np.int8)
                for coords in expected:
                    board[coords] = 1
                actual = find_winning_line(board, 1)
                self.assertEqual(
                    {(cell["layer"], cell["row"], cell["col"]) for cell in actual},
                    set(expected),
                )


class JsonProtocolTests(unittest.TestCase):
    def test_unhashable_request_ids_are_rejected_without_killing_sidecar(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(BACKEND_ROOT)))
        requests = [
            {"v": 1, "type": "request", "id": ["list"], "command": "system.ping", "params": {}},
            {"v": 1, "type": "request", "id": {"object": 1}, "command": "system.ping", "params": {}},
            {"v": 1, "type": "request", "id": "still-alive", "command": "system.ping", "params": {}},
        ]
        input_lines = [json.dumps(request) for request in requests]
        input_lines.insert(0, '{"v":1,"type":"request","id":NaN,"command":"system.ping"}')
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "cubesprite_backend.main",
                "--resource-dir",
                str(RESOURCE_DIR),
            ],
            input="".join(line + "\n" for line in input_lines),
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=30,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        messages = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(messages[0]["event"], "backend.ready")
        responses = messages[1:]
        malformed = [item for item in responses if item.get("error", {}).get("code") == "INVALID_JSON"]
        self.assertEqual(len(malformed), 1)
        invalid = [item for item in responses if item.get("error", {}).get("code") == "INVALID_REQUEST_ID"]
        self.assertEqual(len(invalid), 2)
        alive = next(item for item in responses if item.get("id") == "still-alive")
        self.assertTrue(alive["ok"])
        self.assertTrue(alive["result"]["pong"])


if __name__ == "__main__":
    unittest.main()
