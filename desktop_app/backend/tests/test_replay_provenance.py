from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "desktop_app" / "backend")]

from cubesprite_backend.replay_store import participant_provenance_hash, validate_replay  # noqa: E402
from cubesprite_backend.search import SearchResult  # noqa: E402
from cubesprite_backend.service import CubeSpriteService  # noqa: E402


MODEL_A = "v2.2_balance"
MODEL_B = "cubesprite_v4_flash_preview1"


class ReplayProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.service = CubeSpriteService(ROOT / "desktop_app" / "src-tauri" / "resources", Path(self.temporary.name))
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.service.close)

    @staticmethod
    def token(state):
        return {"session_id": state["session_id"], "expected_revision": state["revision"]}

    def save(self, state):
        summary = self.service.handle("replay.save", self.token(state))["replay"]
        ref = {"id": summary["id"], "expected_fingerprint": summary["fingerprint"]}
        replay = self.service.handle("replay.open", ref)["replay"]
        exported = self.service.handle("replay.export", ref)
        self.assertEqual(validate_replay(json.loads(exported["content"]), self.service.game), replay)
        self.assertEqual(replay["protocol_version"], 2)
        self.assertNotIn("_participant", exported["content"])
        return replay

    def import_sample(self):
        sample = Path(__file__).parent / "fixtures" / "training-v2-sample.c4replay.json"
        summary = self.service.handle("replay.import", {"path": str(sample)})["replay"]
        ref = {"id": summary["id"], "expected_fingerprint": summary["fingerprint"]}
        return ref, self.service.handle("replay.open", ref)["replay"]

    def human_move(self, state):
        move = state["legal_moves"][-1]
        return self.service.handle("game.move", self.token(state) | {key: move[key] for key in ("layer", "row", "col")})

    def ai_move(self, state, model_id):
        # Isolate controller bookkeeping from MCTS quality and model runtime.
        action = state["legal_moves"][0]["action"]
        policy = np.zeros(150)
        policy[action] = 1.0
        with patch.object(self.service, "_search", return_value=SearchResult(action, policy.tolist(), 0.0)):
            return self.service.handle("game.ai_move", self.token(state) | {"ai": {"model_id": model_id}})

    def switch_model(self, state, model_id):
        return self.service.handle("settings.update", self.token(state) | {"roles": {"combat": {"model_id": model_id}}})["state"]

    def test_continuation_preserves_source_participants_before_new_moves(self):
        ref, original = self.import_sample()
        state = self.service.handle("replay.continue", ref | {"step": 4, "mode": "pvp"})
        replay = self.save(state)
        self.assertEqual(replay["turns"], original["turns"][:4])
        self.assertEqual(replay["participants"], original["participants"])
        self.assertEqual(replay["participant_provenance_hash"], original["participant_provenance_hash"])

    def test_new_human_controller_marks_only_its_seat_mixed_and_undo_restores_source(self):
        ref, original = self.import_sample()
        state = self.service.handle("replay.continue", ref | {"step": 4, "mode": "pvp"})
        state = self.human_move(state)
        self.assertNotIn("_participant", state["last_move"])
        replay = self.save(state)
        self.assertEqual(replay["participants"][0]["controller_type"], "external")
        for key in ("controller_id", "model_id", "lineage_hash", "artifact_sha256"):
            self.assertIsNone(replay["participants"][0][key])
        self.assertEqual(replay["participants"][1], original["participants"][1])
        state = self.service.handle("game.undo", self.token(state))
        self.assertEqual(self.save(state)["participants"], original["participants"])

    def test_restart_restores_continued_prefix_provenance(self):
        ref, original = self.import_sample()
        state = self.service.handle("replay.continue", ref | {"step": 4, "mode": "pvai", "human_player": 1})
        state = self.human_move(state)
        state = self.ai_move(state, MODEL_B)
        self.assertTrue(all(item["controller_type"] == "external" for item in self.save(state)["participants"]))
        state = self.service.handle("game.restart", self.token(state))
        replay = self.save(state)
        self.assertEqual(replay["turns"], original["turns"][:4])
        self.assertEqual(replay["participants"], original["participants"])

    def test_matching_model_identity_keeps_source_display_name(self):
        sample = Path(__file__).parent / "fixtures" / "training-v2-sample.c4replay.json"
        payload = json.loads(sample.read_text(encoding="utf-8"))
        payload["participants"][1].update(
            controller_id=MODEL_B,
            model_id=MODEL_B,
            artifact_sha256=self.service.models.get(MODEL_B).artifact_sha256,
            display_name="Training Flash",
        )
        payload["participant_provenance_hash"] = participant_provenance_hash(payload)
        summary = self.service.handle("replay.import", {"content": json.dumps(payload)})["replay"]
        ref = {"id": summary["id"], "expected_fingerprint": summary["fingerprint"]}
        state = self.service.handle("replay.continue", ref | {"step": 4, "mode": "pvai", "human_player": 1})
        state = self.ai_move(self.human_move(state), MODEL_B)
        self.assertEqual(self.save(state)["participants"][1], payload["participants"][1])

    def test_undone_model_does_not_contaminate_replacement_moves(self):
        state = self.service.handle("game.new", {"mode": "pvai", "ai": {"model_id": MODEL_A}})
        state = self.ai_move(self.human_move(state), MODEL_A)
        state = self.service.handle("game.undo", self.token(state))
        state = self.switch_model(state, MODEL_B)
        state = self.ai_move(self.human_move(state), MODEL_B)
        participant = self.save(state)["participants"][1]
        self.assertEqual(participant["controller_type"], "model")
        self.assertEqual(participant["model_id"], MODEL_B)
        self.assertEqual(participant["artifact_sha256"], self.service.models.get(MODEL_B).artifact_sha256)

    def test_retained_models_are_mixed_until_new_model_moves_are_undone(self):
        state = self.service.handle("game.new", {"mode": "pvai", "ai": {"model_id": MODEL_A}})
        state = self.ai_move(self.human_move(state), MODEL_A)
        state = self.switch_model(state, MODEL_B)
        self.assertEqual(self.save(state)["participants"][1]["model_id"], MODEL_A)
        state = self.ai_move(self.human_move(state), MODEL_B)
        self.assertEqual(self.save(state)["participants"][1]["controller_type"], "external")
        state = self.service.handle("game.undo", self.token(state))
        participant = self.save(state)["participants"][1]
        self.assertEqual(participant["controller_type"], "model")
        self.assertEqual(participant["model_id"], MODEL_A)

    def test_only_executed_model_counts_and_restart_discards_old_moves(self):
        state = self.service.handle("game.new", {"mode": "pvai", "ai": {"model_id": MODEL_A}})
        state = self.ai_move(self.human_move(state), MODEL_B)
        self.assertNotIn("_participant", state["last_move"])
        self.assertEqual(self.save(state)["participants"][1]["model_id"], MODEL_B)
        state = self.service.handle("game.restart", self.token(state))
        state = self.ai_move(self.human_move(state), MODEL_A)
        self.assertEqual(self.save(state)["participants"][1]["model_id"], MODEL_A)


if __name__ == "__main__":
    unittest.main()
