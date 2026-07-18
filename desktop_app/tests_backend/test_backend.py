from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "desktop_app" / "backend"
RESOURCE_DIR = REPO_ROOT / "desktop_app" / "src-tauri" / "resources"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from connect4_core import GameRules
from cubesprite_backend.main import process_request
from cubesprite_backend.model_runtime import ModelRegistry, ModelRegistryError, ModelUnavailableError, OnnxPredictor
from cubesprite_backend.search import NumpyMCTS
from cubesprite_backend.service import CubeSpriteService, ServiceError, find_winning_line
from desktop_app.scripts.export_models import verify_source_sha256


class UniformPredictor:
    def __init__(self, value=0.0):
        self.value = value
        self.entered = None
        self.release = None

    def predict(self, board):
        if self.entered is not None:
            self.entered.set()
            self.release.wait(timeout=5)
        return np.full(150, 1.0 / 150.0), self.value


class FakeModels:
    def __init__(self, predictor=None):
        self.impl = predictor or UniformPredictor()
        self.requested = []

    def get(self, model_id):
        if model_id in {"cubesprite_v3", "cubesprite_v3_mini", "v2.2_balance", "v2.1_high"}:
            return SimpleNamespace(id=model_id, display_name=model_id, placeholder=False)
        raise ModelUnavailableError(f"Unknown model id: {model_id}")

    def predictor(self, model_id):
        self.get(model_id)
        self.requested.append(model_id)
        return self.impl

    def list_models(self):
        return []


def token(state):
    return {"session_id": state["session_id"], "expected_revision": state["revision"]}


def move(service, state, layer, row, col):
    return service.handle("game.move", {**token(state), "layer": layer, "row": row, "col": col})


class ExportSourceTests(unittest.TestCase):
    def test_source_hash_guard_accepts_exact_file_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "best_state.pth.tar"
            checkpoint.write_bytes(b"expected checkpoint")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(verify_source_sha256(checkpoint, expected.upper()), expected)

            checkpoint.write_bytes(b"replaced checkpoint")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_source_sha256(checkpoint, expected)


class ManifestAndAdapterTests(unittest.TestCase):
    def test_authoritative_manifest_has_four_clean_bilingual_entries(self):
        models = ModelRegistry(RESOURCE_DIR).list_models()
        self.assertEqual([item["id"] for item in models], [
            "cubesprite_v3", "cubesprite_v3_mini", "v2.2_balance", "v2.1_high"
        ])
        self.assertTrue(models[0]["available"])
        self.assertTrue(models[1]["available"])
        self.assertEqual(models[0]["architecture"], "gravity_resnet_v1")
        self.assertEqual(models[1]["architecture"], "gravity_resnet_v1")
        self.assertEqual(models[3]["architecture"], "legacy-v21-adapted-6-layer")
        self.assertEqual((models[3]["board_layers"], models[3]["action_dim"]), (8, 200))
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
        for item in models:
            expected_hash, expected_iteration = expected_identities[item["id"]]
            self.assertEqual(item["artifact_sha256"], expected_hash)
            self.assertEqual(item["source_iteration"], expected_iteration)
            artifact = RESOURCE_DIR / item["model_path"]
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), expected_hash)
            text = item["display_name"] + item["description"]["zh"] + item["description"]["en"]
            for mojibake in ("Ã", "é¢", "�"):
                self.assertNotIn(mojibake, text)

    def test_v22_two_channel_encoding(self):
        spec = ModelRegistry(RESOURCE_DIR).get("v2.2_balance")
        predictor = OnnxPredictor.__new__(OnnxPredictor)
        predictor.spec = spec
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0, 1, 2], board[0, 3, 4] = 1, -1
        encoded = predictor._encode(board)
        self.assertEqual(encoded.shape, (1, 2, 6, 5, 5))
        self.assertEqual(encoded[0, 0, 0, 1, 2], 1)
        self.assertEqual(encoded[0, 1, 0, 3, 4], 1)

    def test_v21_pads_two_empty_layers_and_crops_policy(self):
        spec = ModelRegistry(RESOURCE_DIR).get("v2.1_high")
        predictor = OnnxPredictor.__new__(OnnxPredictor)
        predictor.spec = spec
        predictor.input_name = "board"
        predictor.policy_output_name = "policy"
        predictor.value_output_name = "value"
        predictor._run_lock = threading.Lock()

        class Session:
            last_input = None

            def run(self, names, feeds):
                self.last_input = feeds["board"]
                logits = np.concatenate((np.zeros(150), np.full(50, 100.0))).reshape(1, 200)
                return logits, np.array([[0.25]], dtype=np.float32)

        predictor.session = Session()
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[5, 4, 3] = -1
        policy, value = predictor.predict(board)
        self.assertEqual(predictor.session.last_input.shape, (1, 1, 8, 5, 5))
        np.testing.assert_array_equal(predictor.session.last_input[0, 0, 6:], 0)
        self.assertEqual(predictor.session.last_input[0, 0, 5, 4, 3], -1)
        self.assertEqual(policy.shape, (150,))
        np.testing.assert_allclose(policy, np.full(150, 1.0 / 150.0))
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertAlmostEqual(value, 0.25)

    def test_manifest_rejects_escape_duplicate_and_wrong_dimensions(self):
        model = {
            "id": "bad", "display_name": "Bad", "model_path": "../escape.onnx",
            "architecture": "modern-v22", "board_layers": 6, "board_size": 5,
            "input_channels": 2, "action_dim": 150,
            "artifact_sha256": "0" * 64, "source_iteration": None,
            "defaults": {"mcts_sims": 128, "temperature": 1.0},
            "description": {"zh": "坏", "en": "Bad"},
            "placeholder": False,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry_file = root / "model_registry.json"
            registry_file.write_text(json.dumps({"models": [model]}), encoding="utf-8")
            with self.assertRaises(ModelRegistryError):
                ModelRegistry(root).list_models()
            safe = dict(model, model_path="models/a.onnx")
            registry_file.write_text(json.dumps({"models": [safe, safe]}), encoding="utf-8")
            with self.assertRaises(ModelRegistryError):
                ModelRegistry(root)
            registry_file.write_text(json.dumps({"models": [dict(safe, action_dim=149)]}), encoding="utf-8")
            with self.assertRaises(ModelRegistryError):
                ModelRegistry(root)
            for bad_hash in ("0" * 63, "A" * 64, "not-a-digest"):
                registry_file.write_text(
                    json.dumps({"models": [dict(safe, artifact_sha256=bad_hash)]}),
                    encoding="utf-8",
                )
                with self.subTest(bad_hash=bad_hash), self.assertRaisesRegex(
                    ModelRegistryError, "artifact_sha256"
                ):
                    ModelRegistry(root)
            for bad_iteration in (0, -1, True, "192"):
                registry_file.write_text(
                    json.dumps({"models": [dict(safe, source_iteration=bad_iteration)]}),
                    encoding="utf-8",
                )
                with self.subTest(bad_iteration=bad_iteration), self.assertRaisesRegex(
                    ModelRegistryError, "source_iteration"
                ):
                    ModelRegistry(root)

    def test_real_onnx_models_when_exported(self):
        models_dir = RESOURCE_DIR / "models"
        filenames = (
            "cubesprite_v3.onnx",
            "cubesprite_v3_mini.onnx",
            "v2.2_balance.onnx",
            "v2.1_high.onnx",
        )
        if not all((models_dir / name).is_file() for name in filenames):
            self.skipTest("Run export_models.py to create ONNX resources.")
        registry = ModelRegistry(RESOURCE_DIR)
        board = np.zeros((6, 5, 5), dtype=np.int8)
        for model_id in ("cubesprite_v3", "cubesprite_v3_mini", "v2.2_balance", "v2.1_high"):
            policy, value = registry.predictor(model_id).predict(board)
            self.assertEqual(policy.shape, (150,))
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
            self.assertTrue(-1 <= value <= 1)

    def test_corrupt_file_and_wrong_policy_dimension_are_rejected(self):
        model = {
            "id": "broken", "display_name": "Broken", "model_path": "models/broken.onnx",
            "architecture": "modern-v22", "board_layers": 6, "board_size": 5,
            "input_channels": 2, "action_dim": 150,
            "artifact_sha256": "0" * 64, "source_iteration": None,
            "defaults": {"mcts_sims": 128, "temperature": 1.0},
            "description": {"zh": "损坏", "en": "Broken"},
            "placeholder": False,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "models").mkdir()
            path = root / "models" / "broken.onnx"
            path.write_bytes(b"not an onnx model")
            model["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "model_registry.json").write_text(json.dumps({"models": [model]}), encoding="utf-8")
            registry = ModelRegistry(root)
            with self.assertRaises(ModelUnavailableError):
                registry.predictor("broken")
            listed = registry.list_models()[0]
            self.assertFalse(listed["available"])
            self.assertTrue(listed["unavailable_reason"].startswith("model_load_failed:"))

            spec = registry.get("broken")
            fake_session = mock.Mock()
            fake_session.get_inputs.return_value = [SimpleNamespace(name="board", shape=[None, 2, 6, 5, 5])]
            fake_session.get_outputs.return_value = [
                SimpleNamespace(name="policy", shape=[None, 149]),
                SimpleNamespace(name="value", shape=[None, 1]),
            ]
            with mock.patch("cubesprite_backend.model_runtime.ort.InferenceSession", return_value=fake_session):
                with self.assertRaisesRegex(ValueError, "policy"):
                    OnnxPredictor(spec, path)

    def test_model_artifact_replacement_is_rejected_before_onnx_loading(self):
        original = b"original model artifact"
        model = {
            "id": "guarded", "display_name": "Guarded", "model_path": "models/guarded.onnx",
            "architecture": "modern-v22", "board_layers": 6, "board_size": 5,
            "input_channels": 2, "action_dim": 150,
            "artifact_sha256": hashlib.sha256(original).hexdigest(), "source_iteration": 7,
            "defaults": {"mcts_sims": 128, "temperature": 1.0},
            "description": {"zh": "受保护", "en": "Guarded"},
            "placeholder": False,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models_dir = root / "models"
            models_dir.mkdir()
            artifact = models_dir / "guarded.onnx"
            artifact.write_bytes(b"replacement model artifact")
            (root / "model_registry.json").write_text(json.dumps({"models": [model]}), encoding="utf-8")
            registry = ModelRegistry(root)
            with mock.patch("cubesprite_backend.model_runtime.ort.InferenceSession") as load:
                with self.assertRaisesRegex(ModelUnavailableError, "SHA-256 mismatch"):
                    registry.predictor("guarded")
                load.assert_not_called()
            listed = registry.list_models()[0]
            self.assertFalse(listed["available"])
            self.assertIn("artifact SHA-256 mismatch", listed["unavailable_reason"])


class ServiceStateTests(unittest.TestCase):
    def setUp(self):
        self.service = CubeSpriteService(RESOURCE_DIR)
        self.service.models = FakeModels()

    def test_initial_state_and_settings(self):
        initialized = self.service.initialize()
        state = initialized["state"]
        self.assertEqual(np.asarray(state["board"]).shape, (6, 5, 5))
        self.assertEqual(len(state["legal_moves"]), 25)
        self.assertFalse(initialized["settings"]["preload_hint"])
        self.assertEqual(set(initialized["settings"]["roles"]), {"combat", "hint", "win_rate"})

    def test_mutations_require_and_check_concurrency_tokens(self):
        state = self.service.snapshot()
        with self.assertRaises(ServiceError) as caught:
            self.service.handle("game.move", {"layer": 0, "row": 0, "col": 0})
        self.assertEqual(caught.exception.code, "MISSING_CONCURRENCY_TOKEN")
        with self.assertRaises(ServiceError) as caught:
            self.service.handle("game.move", {**token(state), "expected_revision": state["revision"] - 1, "layer": 0, "row": 0, "col": 0})
        self.assertEqual(caught.exception.code, "STALE_REVISION")
        with self.assertRaises(ServiceError) as caught:
            self.service.handle("game.move", {**token(state), "session_id": "old", "layer": 0, "row": 0, "col": 0})
        self.assertEqual(caught.exception.code, "STALE_SESSION")

    def test_move_coordinates_reject_booleans_and_fractional_values(self):
        state = self.service.snapshot()
        for field, value in (("layer", 0.9), ("row", True)):
            params = {**token(state), "layer": 0, "row": 0, "col": 0, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ServiceError) as caught:
                self.service.handle("game.move", params)
            self.assertEqual(caught.exception.code, "INVALID_PARAMS")
            self.assertEqual(self.service.snapshot()["revision"], state["revision"])

    def test_pvp_gravity_win_terminal_guard_and_undo(self):
        state = self.service.handle("game.new", {"mode": "pvp"})
        sequence = [(0, 0, 0), (0, 4, 0), (0, 0, 1), (0, 4, 1), (0, 0, 2), (0, 4, 2), (0, 0, 3)]
        for coords in sequence:
            state = move(self.service, state, *coords)
        self.assertEqual((state["status"], state["winner"], len(state["winning_line"])), ("won", 1, 4))
        with self.assertRaises(ServiceError) as caught:
            move(self.service, state, 0, 2, 2)
        self.assertEqual(caught.exception.code, "GAME_FINISHED")
        state = self.service.handle("game.undo", token(state))
        self.assertEqual((state["status"], state["move_count"]), ("playing", 6))
        with self.assertRaises(ServiceError) as caught:
            move(self.service, state, 1, 2, 2)
        self.assertEqual(caught.exception.code, "ILLEGAL_MOVE")

    def test_pvai_ai_first_and_pair_undo(self):
        state = self.service.handle("game.new", {"mode": "pvai", "human_player": -1})
        self.assertFalse(state["can_undo"])
        state = self.service.handle("game.ai_move", {**token(state), "ai": {"mcts_sims": 32, "temperature": 0}})
        self.assertEqual(state["current_player"], -1)
        self.assertFalse(state["can_undo"])
        human_move = state["legal_moves"][0]
        state = move(self.service, state, human_move["layer"], human_move["row"], human_move["col"])
        state = self.service.handle("game.ai_move", {**token(state), "ai": {"mcts_sims": 32, "temperature": 0}})
        self.assertEqual(state["move_count"], 3)
        state = self.service.handle("game.undo", token(state))
        self.assertEqual((state["move_count"], state["current_player"]), (1, -1))

    def test_three_role_settings_and_preload_are_session_only(self):
        state = self.service.snapshot()
        result = self.service.handle("settings.update", {
            **token(state), "preload_hint": True,
            "roles": {"combat": {"model_id": "v2.1_high", "mcts_sims": 32, "temperature": 0.2}},
        })
        settings = result["settings"]
        self.assertTrue(settings["preload_hint"])
        self.assertEqual(settings["roles"]["combat"]["model_id"], "v2.1_high")
        self.assertEqual(settings["roles"]["hint"]["model_id"], "v2.2_balance")
        result = self.service.handle("settings.set_preload_hint", {**token(result["state"]), "enabled": False})
        self.assertFalse(result["settings"]["preload_hint"])
        state = self.service.handle("game.new", {"mode": "pvp"})
        self.assertEqual(self.service.handle("settings.get", {})["roles"]["combat"]["model_id"], "v2.1_high")
        updated = self.service.handle(
            "settings.update",
            {**token(state), "roles": {"combat": {"model_id": "cubesprite_v3"}}},
        )
        self.assertEqual(updated["settings"]["roles"]["combat"]["model_id"], "cubesprite_v3")

    def test_hint_and_win_rate_use_independent_roles(self):
        state = self.service.snapshot()
        updated = self.service.handle("settings.update", {
            **token(state), "roles": {
                "hint": {"model_id": "v2.1_high", "mcts_sims": 32, "temperature": 0},
                "win_rate": {"model_id": "v2.2_balance", "mcts_sims": 32, "temperature": 0},
            },
        })
        state = updated["state"]
        hint = self.service.handle("analysis.hint", token(state))
        rate = self.service.handle("analysis.win_rate", token(state))
        self.assertEqual(self.service.models.requested[-2:], ["v2.1_high", "v2.2_balance"])
        self.assertEqual(hint["for_revision"], state["revision"])
        self.assertAlmostEqual(rate["red"] + rate["blue"], 1.0)

    def test_background_analysis_is_rejected_after_board_change(self):
        slow = UniformPredictor()
        slow.entered, slow.release = threading.Event(), threading.Event()
        self.service.models = FakeModels(slow)
        state = self.service.snapshot()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.service.handle, "analysis.hint", {**token(state), "ai": {"mcts_sims": 32}})
            self.assertTrue(slow.entered.wait(timeout=5))
            newer = move(self.service, state, 0, 0, 0)
            slow.release.set()
            with self.assertRaises(ServiceError) as caught:
                future.result(timeout=10)
        self.assertEqual(caught.exception.code, "STALE_REVISION")
        self.assertEqual(newer["move_count"], 1)


class SearchAndLineTests(unittest.TestCase):
    def test_mcts_forces_immediate_win_and_block(self):
        game, predictor = GameRules(), UniformPredictor()
        winning = game.get_init_board()
        winning[0, 0, :3] = 1
        result = NumpyMCTS(game, predictor, simulations=1, temperature=0).run(winning, 1)
        self.assertEqual(game.action_to_coords(result.action), (0, 0, 3))
        self.assertEqual(result.value, 1.0)
        blocking = game.get_init_board()
        blocking[0, 1, :3] = -1
        result = NumpyMCTS(game, predictor, simulations=1, temperature=0).run(blocking, 1)
        self.assertEqual(game.action_to_coords(result.action), (0, 1, 3))

    def test_space_diagonal_line_is_reported(self):
        board = np.zeros((6, 5, 5), dtype=np.int8)
        for index in range(4):
            board[index, index, index] = 1
        line = find_winning_line(board, 1)
        self.assertEqual([(c["layer"], c["row"], c["col"]) for c in line], [
            (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)
        ])


class ProtocolTests(unittest.TestCase):
    def test_process_request_always_returns_json_error_envelope(self):
        service = CubeSpriteService(RESOURCE_DIR)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            process_request(service, ["not", "an", "object"])
        response = json.loads(stream.getvalue())
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "INVALID_ENVELOPE")

    def test_jsonl_subprocess_survives_bad_id_and_invalid_json(self):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(BACKEND_ROOT), env.get("PYTHONPATH", "")))
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", "cubesprite_backend.main", "--resource-dir", str(RESOURCE_DIR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=env,
        )
        try:
            self.assertEqual(json.loads(process.stdout.readline())["event"], "backend.ready")
            process.stdin.write("{broken json\n")
            process.stdin.flush()
            self.assertEqual(json.loads(process.stdout.readline())["error"]["code"], "INVALID_JSON")
            bad_id = {"v": 1, "type": "request", "id": {"bad": "id"}, "command": "system.ping", "params": {}}
            process.stdin.write(json.dumps(bad_id) + "\n")
            process.stdin.flush()
            self.assertEqual(json.loads(process.stdout.readline())["error"]["code"], "INVALID_REQUEST_ID")
            ping = {"v": 1, "type": "request", "id": "ping-1", "command": "system.ping", "params": {}}
            process.stdin.write(json.dumps(ping) + "\n")
            process.stdin.flush()
            self.assertTrue(json.loads(process.stdout.readline())["result"]["pong"])
            process.stdin.close()
            self.assertEqual(process.wait(timeout=10), 0)
            self.assertEqual(process.stderr.read(), "")
            process.stdout.close()
            process.stderr.close()
        finally:
            if process.poll() is None:
                process.kill()


if __name__ == "__main__":
    unittest.main()
