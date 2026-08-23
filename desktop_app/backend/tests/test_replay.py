from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "desktop_app" / "backend"
RESOURCE_DIR = REPO_ROOT / "desktop_app" / "src-tauri" / "resources"
for import_root in (REPO_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import cubesprite_backend.main as backend_main  # noqa: E402
from connect4_core.rules import RULE2, RuleEngine, TurnAction  # noqa: E402
from cubesprite_backend.replay_store import (  # noqa: E402
    MAX_REPLAY_BYTES,
    ReplayStore,
    ReplayStoreError,
    build_replay_frames,
    participant_provenance_hash,
    replay_fingerprint,
    validate_replay,
)
from cubesprite_backend.search import SearchResult  # noqa: E402
from cubesprite_backend.service import CubeSpriteService, ServiceError  # noqa: E402


FULL_DRAW_COLUMNS = (
    2, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 1, 1, 3, 1, 3, 2, 2, 3, 2, 3, 3, 4, 3,
    5, 4, 5, 4, 4, 5, 4, 4, 5, 5, 7, 5, 7, 6, 11, 6, 6, 6, 12, 6, 6, 7, 13, 7, 7,
    8, 7, 9, 8, 9, 8, 8, 8, 10, 8, 11, 9, 11, 9, 12, 9, 9, 10, 10, 10, 10, 10, 14,
    11, 15, 11, 17, 11, 17, 12, 18, 12, 19, 12, 12, 13, 13, 14, 13, 15, 13, 13, 14,
    15, 14, 14, 19, 14, 21, 15, 15, 16, 15, 18, 18, 18, 18, 20, 20, 16, 20, 16, 16,
    17, 16, 17, 16, 17, 17, 18, 20, 20, 20, 21, 21, 22, 22, 19, 22, 19, 19, 21, 19,
    22, 22, 23, 23, 23, 23, 21, 23, 21, 23, 24, 24, 22, 24, 24, 24, 24,
)


class ReplayServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cubesprite-replay-test-")
        self.data_dir = Path(self.temporary.name)
        self.service = CubeSpriteService(RESOURCE_DIR, data_dir=self.data_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def token(state: dict) -> dict:
        return {
            "session_id": state["session_id"],
            "expected_revision": state["revision"],
        }

    @staticmethod
    def replay_ref(replay: dict) -> dict:
        return {
            "id": replay["id"],
            "expected_fingerprint": replay["fingerprint"],
        }

    def play(self, coordinates: list[tuple[int, int, int]]) -> dict:
        state = self.service.handle("game.state", {})
        for layer, row, col in coordinates:
            state = self.service.handle(
                "game.move",
                self.token(state) | {"layer": layer, "row": row, "col": col},
            )
        return state

    def save(self, state: dict, name: str = "Test replay") -> dict:
        return self.service.handle(
            "replay.save",
            self.token(state) | {"name": name},
        )["replay"]

    def v2_payload(self) -> dict:
        rules = {
            "format": "connect4-3d-gravity",
            "version": 1,
            "board_layers": 6,
            "board_size": 5,
            "connect_n": 4,
            "gravity": "layer_ascending",
            "starting_player": 1,
        }
        participants = [
            {
                "seat": "FIRST",
                "player": 1,
                "controller_type": "model",
                "controller_id": "accepted:first",
                "display_name": "Red model",
                "model_id": "starter-v3",
                "lineage_hash": "1" * 64,
                "artifact_sha256": "2" * 64,
            },
            {
                "seat": "SECOND",
                "player": -1,
                "controller_type": "human",
                "controller_id": "local:second",
                "display_name": "Blue human",
                "model_id": None,
                "lineage_hash": None,
                "artifact_sha256": None,
            },
        ]
        turns = [
            {
                "ply": 1,
                "kind": "place",
                "player": 1,
                "column": 0,
                "action": 0,
                "layer": 0,
                "row": 0,
                "col": 0,
            },
            {
                "ply": 2,
                "kind": "place",
                "player": -1,
                "column": 5,
                "action": 5,
                "layer": 0,
                "row": 1,
                "col": 0,
            },
        ]
        payload = {
            "format": "cubesprite.replay",
            "protocol_version": 2,
            "id": "a" * 32,
            "name": "V2 portable",
            "saved_at": "2026-08-19T12:00:00Z",
            "rules": rules,
            "rule_id": "classic",
            "rule_version": 1,
            "participants": participants,
            "turns": turns,
            "turn_count": 2,
            "placement_count": 2,
            "status": "playing",
            "winner": None,
        }
        payload["fingerprint"] = replay_fingerprint(payload)
        payload["participant_provenance_hash"] = participant_provenance_hash(payload)
        return payload

    def v2_forced_pass_payload(self) -> dict:
        payload = self.v2_payload()
        payload["rule_id"] = RULE2.rule_id
        engine = RuleEngine(RULE2)
        state = engine.initial_state()
        turns = []
        for ply, column in enumerate(FULL_DRAW_COLUMNS, start=1):
            player = state.player_to_move
            action = engine.legacy_action_for_column(state, column)
            layer, row, col = self.service.game.action_to_coords(action)
            turns.append(
                {
                    "ply": ply,
                    "kind": "place",
                    "player": player,
                    "column": column,
                    "action": action,
                    "layer": layer,
                    "row": row,
                    "col": col,
                }
            )
            state = engine.step(state, TurnAction.place(column))
        for _ in range(2):
            turns.append(
                {
                    "ply": len(turns) + 1,
                    "kind": "forced_pass",
                    "player": state.player_to_move,
                }
            )
            state = engine.step(state, TurnAction.forced_pass())
        payload.update(
            {
                "turns": turns,
                "turn_count": len(turns),
                "placement_count": len(FULL_DRAW_COLUMNS),
                "status": "draw",
                "winner": 0,
            }
        )
        payload["fingerprint"] = replay_fingerprint(payload)
        payload["participant_provenance_hash"] = participant_provenance_hash(payload)
        return payload

    def install_fake_search(self, value: float) -> list[tuple[int, dict]]:
        calls: list[tuple[int, dict]] = []

        def fake_search(board: np.ndarray, player: int, ai: dict) -> SearchResult:
            calls.append((player, dict(ai)))
            action = int(np.flatnonzero(self.service.game.get_valid_moves(board) > 0)[0])
            policy = np.zeros(self.service.game.get_action_size(), dtype=float)
            policy[action] = 1.0
            return SearchResult(action=action, policy=policy.tolist(), value=value)

        self.service._search = fake_search
        return calls

    def test_sidecar_routes_replay_analysis_to_a_dedicated_single_worker(self) -> None:
        executors = []

        class RecordingExecutor:
            def __init__(self, max_workers: int, thread_name_prefix: str):
                self.max_workers = max_workers
                self.thread_name_prefix = thread_name_prefix
                self.commands: list[str] = []
                executors.append(self)

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _traceback) -> None:
                return None

            def submit(self, _callable, _service, request):
                self.commands.append(request.get("command", ""))
                future = Future()
                future.set_result(None)
                return future

        requests = (
            {"v": 1, "type": "request", "id": "state", "command": "game.state", "params": {}},
            {
                "v": 1,
                "type": "request",
                "id": "analysis",
                "command": "replay.analyze",
                "params": {},
            },
        )
        stdin = io.StringIO("".join(json.dumps(request) + "\n" for request in requests))
        backend_main.PENDING_IDS.clear()
        try:
            with (
                patch.object(backend_main, "ThreadPoolExecutor", RecordingExecutor),
                patch.object(backend_main, "CubeSpriteService", return_value=object()),
                patch.object(backend_main, "write_message"),
                patch.object(backend_main.sys, "stdin", stdin),
            ):
                exit_code = backend_main.main(
                    [
                        "--resource-dir",
                        str(RESOURCE_DIR),
                        "--data-dir",
                        str(self.data_dir),
                    ]
                )
        finally:
            backend_main.PENDING_IDS.clear()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [(executor.max_workers, executor.thread_name_prefix) for executor in executors],
            [(4, "cubesprite"), (1, "cubesprite-analysis")],
        )
        self.assertEqual(executors[0].commands, ["game.state"])
        self.assertEqual(executors[1].commands, ["replay.analyze"])

    def test_save_list_and_open_reconstruct_authoritative_frames(self) -> None:
        state = self.play([(0, 0, 0), (0, 1, 0)])
        summary = self.save(state, "Opening")

        self.assertEqual(
            set(summary),
            {"id", "name", "saved_at", "move_count", "status", "winner", "fingerprint"},
        )
        self.assertEqual(summary["move_count"], 2)
        self.assertEqual(self.service.handle("replay.list", {})["replays"], [summary])

        opened = self.service.handle("replay.open", self.replay_ref(summary))
        self.assertEqual(set(opened), {"replay", "frames", "analysis"})
        self.assertIsNone(opened["analysis"])
        self.assertEqual(len(opened["frames"]), 3)
        self.assertEqual(opened["replay"]["moves"][0]["ply"], 1)
        self.assertEqual(opened["replay"]["moves"][1]["player"], -1)
        self.assertNotIn("ai", opened["replay"])
        self.assertNotIn("mode", opened["replay"])

        initial, after_red, after_blue = opened["frames"]
        self.assertEqual(initial["mode"], "replay")
        self.assertEqual(initial["revision"], 0)
        self.assertEqual(initial["replay_total_steps"], 2)
        self.assertEqual(after_red["board"][0][0][0], 1)
        self.assertEqual(after_blue["board"][0][1][0], -1)
        self.assertEqual(after_blue["current_player"], 1)
        self.assertEqual(after_blue["replay_step"], 2)

        replay_path = self.data_dir / "replays" / f"{summary['id']}.c4replay.json"
        self.assertTrue(replay_path.is_file())
        self.assertEqual(list(self.data_dir.rglob("*.tmp")), [])

        reloaded_service = CubeSpriteService(RESOURCE_DIR, data_dir=self.data_dir)
        try:
            self.assertEqual(reloaded_service.handle("replay.list", {})["replays"], [summary])
            self.assertEqual(
                reloaded_service.handle("replay.open", self.replay_ref(summary))["frames"],
                opened["frames"],
            )
        finally:
            reloaded_service.close()

    def test_replay_hint_uses_selected_hint_ai_without_mutating_live_game(self) -> None:
        state = self.play([(0, 0, 0), (0, 0, 1)])
        summary = self.save(state, "Hint position")
        live_before = self.service.handle("game.state", {})
        calls = self.install_fake_search(0.35)

        result = self.service.handle(
            "replay.hint",
            self.replay_ref(summary)
            | {
                "step": 1,
                "ai": {
                    "model_id": "v2.2_balance",
                    "mcts_sims": 32,
                    "temperature": 0.2,
                },
            },
        )

        self.assertEqual(result["replay_id"], summary["id"])
        self.assertEqual(result["replay_fingerprint"], summary["fingerprint"])
        self.assertEqual(result["for_step"], 1)
        self.assertEqual(result["for_revision"], 1)
        self.assertEqual(result["value"], 0.35)
        self.assertEqual(calls, [(-1, {
            "model_id": "v2.2_balance",
            "mcts_sims": 32,
            "temperature": 0.2,
        })])
        frame = self.service.handle("replay.open", self.replay_ref(summary))["frames"][1]
        self.assertIn(result["move"], frame["legal_moves"])
        self.assertEqual(self.service.handle("game.state", {}), live_before)

    def test_import_is_strict_idempotent_and_round_trips_after_delete(self) -> None:
        state = self.play([(0, 0, 0)])
        summary = self.save(state, "Portable")
        replay_path = self.data_dir / "replays" / f"{summary['id']}.c4replay.json"
        content = replay_path.read_text(encoding="utf-8")

        duplicate = self.service.handle(
            "replay.import",
            {"content": content, "filename": "portable.c4replay.json"},
        )["replay"]
        self.assertEqual(duplicate, summary)

        alternate_metadata = json.loads(content)
        alternate_metadata["name"] = "Same moves, different imported title"
        alternate_metadata["saved_at"] = "2030-01-02T03:04:05Z"
        persisted = self.service.handle(
            "replay.import",
            {
                "content": json.dumps(alternate_metadata),
                "filename": "renamed-portable.c4replay.json",
            },
        )["replay"]
        self.assertEqual(persisted, summary)

        self.service.handle(
            "replay.delete",
            {
                "replay_id": summary["id"],
                "expected_fingerprint": summary["fingerprint"],
            },
        )
        imported = self.service.handle(
            "replay.import",
            {"content": content, "filename": "portable.c4replay.json"},
        )["replay"]
        self.assertEqual(imported, summary)
        self.assertEqual(
            self.service.handle("replay.open", self.replay_ref(summary))["replay"]["fingerprint"],
            summary["fingerprint"],
        )

        payload = json.loads(content)
        payload["unexpected"] = True
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.import", {"content": json.dumps(payload)})
        self.assertEqual(raised.exception.code, "INVALID_REPLAY")

        payload = json.loads(content)
        payload["rules"]["board_layers"] = 8
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.import", {"content": json.dumps(payload)})
        self.assertEqual(raised.exception.code, "UNSUPPORTED_REPLAY_RULES")

        with self.assertRaises(ServiceError) as raised:
            self.service.handle(
                "replay.import",
                {"content": '{"format":"cubesprite.replay","format":"cubesprite.replay"}'},
            )
        self.assertEqual(raised.exception.code, "INVALID_REPLAY_JSON")

        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.import", {"content": '{"value":NaN}'})
        self.assertEqual(raised.exception.code, "INVALID_REPLAY_JSON")

        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.import", {"content": "\ud800"})
        self.assertEqual(raised.exception.code, "INVALID_REPLAY_IMPORT")

        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.import", {"content": "x" * (MAX_REPLAY_BYTES + 1)})
        self.assertEqual(raised.exception.code, "REPLAY_TOO_LARGE")

    def test_import_rejects_tampered_move_and_fingerprint(self) -> None:
        state = self.play([(0, 0, 0)])
        summary = self.save(state)
        replay_path = self.data_dir / "replays" / f"{summary['id']}.c4replay.json"
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        payload["id"] = "1" * 32
        payload["moves"][0]["col"] = 4
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.import", {"content": json.dumps(payload)})
        self.assertEqual(raised.exception.code, "INVALID_REPLAY")

        payload["moves"][0]["col"] = 0
        payload["fingerprint"] = "0" * 64
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.import", {"content": json.dumps(payload)})
        self.assertEqual(raised.exception.code, "REPLAY_FINGERPRINT_MISMATCH")

    def test_v2_tagged_turns_and_participants_round_trip(self) -> None:
        payload = self.v2_payload()
        store = ReplayStore(self.data_dir / "v2-round-trip", self.service.game)
        imported = store.import_content(json.dumps(payload), filename="v2.c4replay.json")
        self.assertEqual(imported, validate_replay(payload, self.service.game))
        self.assertEqual(imported["turn_count"], 2)
        self.assertEqual(imported["placement_count"], 2)
        self.assertNotIn("moves", imported)

        frames = build_replay_frames(imported, self.service.game)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[1]["board"][0][0][0], 1)
        self.assertEqual(frames[2]["board"][0][1][0], -1)
        self.assertEqual(frames[2]["replay_total_steps"], 2)

        exported = store.export_replay(imported)
        destination = ReplayStore(self.data_dir / "v2-exported", self.service.game)
        round_tripped = destination.import_content(exported["content"], exported["filename"])
        self.assertEqual(round_tripped, imported)

    def test_v2_hashes_separate_game_semantics_from_participant_provenance(self) -> None:
        payload = self.v2_payload()
        original_game_hash = payload["fingerprint"]
        original_provenance = payload["participant_provenance_hash"]

        renamed = json.loads(json.dumps(payload))
        renamed["participants"][0]["display_name"] = "Localized red name"
        validated = validate_replay(renamed, self.service.game)
        self.assertEqual(validated["fingerprint"], original_game_hash)
        self.assertEqual(validated["participant_provenance_hash"], original_provenance)

        tampered = json.loads(json.dumps(payload))
        tampered["participants"][0]["controller_id"] = "different-controller"
        self.assertEqual(replay_fingerprint(tampered), original_game_hash)
        with self.assertRaises(ReplayStoreError) as raised:
            validate_replay(tampered, self.service.game)
        self.assertEqual(raised.exception.code, "PARTICIPANT_PROVENANCE_MISMATCH")

        store = ReplayStore(self.data_dir / "v2-provenance-conflict", self.service.game)
        store.import_content(json.dumps(payload))
        tampered["participant_provenance_hash"] = participant_provenance_hash(tampered)
        with self.assertRaises(ReplayStoreError) as raised:
            store.import_content(json.dumps(tampered))
        self.assertEqual(raised.exception.code, "REPLAY_ID_CONFLICT")

    def test_v2_forced_pass_is_exact_and_rule_validated(self) -> None:
        payload = self.v2_payload()
        payload["turns"] = [
            {"ply": 1, "kind": "forced_pass", "player": 1, "action": 0}
        ]
        payload["turn_count"] = 1
        payload["placement_count"] = 0
        payload["fingerprint"] = replay_fingerprint(payload)
        with self.assertRaises(ReplayStoreError) as raised:
            validate_replay(payload, self.service.game)
        self.assertEqual(raised.exception.code, "INVALID_REPLAY")

        del payload["turns"][0]["action"]
        payload["fingerprint"] = replay_fingerprint(payload)
        with self.assertRaises(ReplayStoreError) as raised:
            validate_replay(payload, self.service.game)
        self.assertEqual(raised.exception.code, "INVALID_REPLAY_MOVE")

    def test_v2_forced_pass_round_trip_keeps_board_unchanged(self) -> None:
        payload = self.v2_forced_pass_payload()
        validated = validate_replay(payload, self.service.game)
        self.assertEqual(validated["turn_count"], 152)
        self.assertEqual(validated["placement_count"], 150)
        self.assertEqual(validated["turns"][-2], {
            "ply": 151,
            "kind": "forced_pass",
            "player": 1,
        })
        self.assertEqual(validated["turns"][-1], {
            "ply": 152,
            "kind": "forced_pass",
            "player": -1,
        })

        frames = build_replay_frames(validated, self.service.game)
        self.assertEqual(frames[150]["board"], frames[151]["board"])
        self.assertEqual(frames[151]["board"], frames[152]["board"])
        self.assertEqual(frames[150]["current_player"], 1)
        self.assertEqual(frames[151]["current_player"], -1)
        self.assertEqual(frames[152]["status"], "draw")
        self.assertEqual(frames[152]["winner"], 0)

    def test_v2_gameplay_tamper_requires_a_new_game_fingerprint(self) -> None:
        payload = self.v2_payload()
        payload["turns"][1].update({"column": 6, "action": 6, "row": 1, "col": 1})
        with self.assertRaises(ReplayStoreError) as raised:
            validate_replay(payload, self.service.game)
        self.assertEqual(raised.exception.code, "REPLAY_FINGERPRINT_MISMATCH")

    def test_export_returns_a_portable_document_that_round_trips(self) -> None:
        state = self.play([(0, 0, 0), (0, 1, 0)])
        summary = self.save(state, "Shared position")
        exported = self.service.handle("replay.export", self.replay_ref(summary))

        self.assertEqual(set(exported), {"filename", "content"})
        self.assertEqual(
            exported["filename"],
            f"CubeSprite-{summary['id'][:12]}.c4replay.json",
        )
        self.assertTrue(exported["content"].endswith("\n"))
        payload = json.loads(exported["content"])
        self.assertEqual(payload["id"], summary["id"])
        self.assertEqual(payload["fingerprint"], summary["fingerprint"])
        self.assertNotIn("mode", payload)
        self.assertNotIn("ai", payload)

        imported_store = ReplayStore(self.data_dir / "export-round-trip", self.service.game)
        imported = imported_store.import_content(
            exported["content"],
            filename=exported["filename"],
        )
        self.assertEqual(imported, payload)
        self.assertEqual(imported_store.load_replay(summary["id"]), payload)

    def test_two_store_instances_serialize_conflicting_same_id_imports(self) -> None:
        state = self.play([(0, 0, 0)])
        summary = self.save(state, "Race source")
        source_path = self.data_dir / "replays" / f"{summary['id']}.c4replay.json"
        first_content = source_path.read_text(encoding="utf-8")

        second_payload = json.loads(first_content)
        second_payload["name"] = "Conflicting race source"
        second_payload["moves"][0].update(
            {
                "action": self.service.game.coords_to_action(0, 1, 0),
                "layer": 0,
                "row": 1,
                "col": 0,
            }
        )
        second_payload["fingerprint"] = replay_fingerprint(second_payload)
        second_content = json.dumps(second_payload, ensure_ascii=False)
        self.assertNotEqual(summary["fingerprint"], second_payload["fingerprint"])

        target_dir = self.data_dir / "same-id-import-race"
        first_store = ReplayStore(target_dir, self.service.game)
        second_store = ReplayStore(target_dir, self.service.game)
        first_write_entered = threading.Event()
        release_first_write = threading.Event()
        second_started = threading.Event()
        second_write_entered = threading.Event()
        original_first_write = first_store._write_json_atomic
        original_second_write = second_store._write_json_atomic

        def delayed_first_write(path: Path, payload: dict) -> None:
            first_write_entered.set()
            if not release_first_write.wait(timeout=5):
                raise AssertionError("Timed out waiting to release the first replay write.")
            original_first_write(path, payload)

        def observed_second_write(path: Path, payload: dict) -> None:
            second_write_entered.set()
            original_second_write(path, payload)

        def import_second() -> dict:
            second_started.set()
            return second_store.import_content(second_content, filename="second.c4replay.json")

        first_store._write_json_atomic = delayed_first_write
        second_store._write_json_atomic = observed_second_write
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                first_store.import_content,
                first_content,
                "first.c4replay.json",
            )
            self.assertTrue(first_write_entered.wait(timeout=5))
            second_future = executor.submit(import_second)
            self.assertTrue(second_started.wait(timeout=5))
            try:
                self.assertFalse(
                    second_write_entered.wait(timeout=0.25),
                    "A second ReplayStore entered its write while the first held the store lock.",
                )
            finally:
                release_first_write.set()

            imported = first_future.result(timeout=5)
            self.assertEqual(imported["fingerprint"], summary["fingerprint"])
            with self.assertRaises(ReplayStoreError) as raised:
                second_future.result(timeout=5)
            self.assertEqual(raised.exception.code, "REPLAY_ID_CONFLICT")

        stored = first_store.load_replay(summary["id"])
        self.assertEqual(stored["fingerprint"], summary["fingerprint"])
        self.assertEqual(len(first_store.list_replays()), 1)
        self.assertFalse(second_write_entered.is_set())

    def test_continue_creates_independent_live_session_from_cursor(self) -> None:
        state = self.play([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
        original_session = state["session_id"]
        summary = self.save(state, "Branch point")
        replay_path = self.data_dir / "replays" / f"{summary['id']}.c4replay.json"
        replay_bytes = replay_path.read_bytes()

        continued = self.service.handle(
            "replay.continue",
            {
                "id": summary["id"],
                "expected_fingerprint": summary["fingerprint"],
                "step": 2,
                "mode": "pvai",
                "human_player": 1,
            },
        )
        self.assertNotEqual(continued["session_id"], original_session)
        self.assertEqual(continued["mode"], "pvai")
        self.assertEqual(continued["human_player"], 1)
        self.assertEqual(continued["move_count"], 2)
        self.assertEqual(continued["current_player"], 1)
        self.assertEqual(continued["board"][0][0][0], 1)
        self.assertEqual(continued["board"][0][1][0], -1)
        self.assertEqual(continued["board"][0][0][1], 0)
        self.assertFalse(continued["can_undo"])

        blocked_undo = self.service.handle("game.undo", self.token(continued))
        self.assertEqual(blocked_undo, continued)

        moved = self.service.handle(
            "game.move",
            self.token(continued) | {"layer": 0, "row": 2, "col": 0},
        )
        self.assertEqual(moved["move_count"], 3)
        self.assertTrue(moved["can_undo"])

        undone = self.service.handle("game.undo", self.token(moved))
        self.assertEqual(undone["move_count"], 2)
        self.assertEqual(undone["board"], continued["board"])
        self.assertFalse(undone["can_undo"])

        second_undo = self.service.handle("game.undo", self.token(undone))
        self.assertEqual(second_undo, undone)

        moved_again = self.service.handle(
            "game.move",
            self.token(second_undo) | {"layer": 0, "row": 2, "col": 0},
        )
        restarted = self.service.handle("game.restart", self.token(moved_again))
        self.assertEqual(restarted["move_count"], 2)
        self.assertEqual(restarted["board"], continued["board"])
        self.assertEqual(restarted["current_player"], continued["current_player"])
        self.assertFalse(restarted["can_undo"])
        self.assertEqual(replay_path.read_bytes(), replay_bytes)

    def test_replay_operations_reject_missing_or_stale_fingerprint(self) -> None:
        state = self.play([(0, 0, 0), (0, 1, 0)])
        summary = self.save(state, "Fingerprint guard")
        stale = {
            "id": summary["id"],
            "expected_fingerprint": "0" * 64,
        }

        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.open", {"id": summary["id"]})
        self.assertEqual(raised.exception.code, "MISSING_REPLAY_FINGERPRINT")

        operations = (
            ("replay.open", stale),
            ("replay.analyze", stale),
            ("replay.export", stale),
            (
                "replay.continue",
                stale | {"step": 1, "mode": "pvp", "human_player": 1},
            ),
            ("replay.delete", stale),
        )
        for command, params in operations:
            with self.subTest(command=command):
                with self.assertRaises(ServiceError) as raised:
                    self.service.handle(command, params)
                self.assertEqual(raised.exception.code, "STALE_REPLAY")

        opened = self.service.handle("replay.open", self.replay_ref(summary))
        self.assertEqual(opened["replay"]["fingerprint"], summary["fingerprint"])
        self.assertEqual(self.service.handle("replay.list", {})["replays"], [summary])

    def test_terminal_replay_cannot_continue(self) -> None:
        state = self.play(
            [
                (0, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 0, 2),
                (0, 1, 2),
                (0, 0, 3),
            ]
        )
        summary = self.save(state, "Finished")
        opened = self.service.handle("replay.open", self.replay_ref(summary))
        self.assertEqual(opened["frames"][-1]["status"], "won")
        self.assertEqual(opened["frames"][-1]["winner"], 1)
        self.assertEqual(len(opened["frames"][-1]["winning_line"]), 4)

        with self.assertRaises(ServiceError) as raised:
            self.service.handle(
                "replay.continue",
                self.replay_ref(summary) | {"step": 7, "mode": "pvp", "human_player": 1},
            )
        self.assertEqual(raised.exception.code, "GAME_FINISHED")

    def test_analysis_uses_current_win_rate_settings_and_overwrites_cache(self) -> None:
        state = self.play([(0, 0, 0), (0, 1, 0)])
        summary = self.save(state, "Curve")
        updated = self.service.handle(
            "settings.update",
            self.token(state)
            | {
                "roles": {
                    "win_rate": {
                        "model_id": "v2.2_balance",
                        "mcts_sims": 32,
                        "temperature": 0.5,
                    }
                }
            },
        )
        calls = self.install_fake_search(0.2)
        analysis = self.service.handle("replay.analyze", self.replay_ref(summary))
        self.assertEqual(analysis["config"], updated["settings"]["roles"]["win_rate"])
        spec = self.service.models.get("v2.2_balance")
        self.assertEqual(
            analysis["model"],
            {
                "id": spec.id,
                "display_name": spec.display_name,
                "architecture": spec.architecture,
                "artifact_sha256": spec.artifact_sha256,
                "source_iteration": spec.source_iteration,
            },
        )
        self.assertRegex(analysis["model"]["artifact_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual([point["step"] for point in analysis["points"]], [0, 1, 2])
        self.assertAlmostEqual(analysis["points"][0]["red"], 0.6)
        self.assertAlmostEqual(analysis["points"][1]["red"], 0.4)
        self.assertAlmostEqual(analysis["points"][2]["red"], 0.6)
        self.assertEqual(len(calls), 3)

        analysis_path = self.data_dir / "replay_analysis" / f"{summary['id']}.winrate.json"
        first_bytes = analysis_path.read_bytes()
        calls = self.install_fake_search(-0.4)
        overwritten = self.service.handle(
            "replay.analyze",
            {
                "replay_id": summary["id"],
                "expected_fingerprint": summary["fingerprint"],
            },
        )
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(overwritten["points"][0]["red"], 0.3)
        self.assertNotEqual(analysis_path.read_bytes(), first_bytes)
        self.assertEqual(
            self.service.handle("replay.open", self.replay_ref(summary))["analysis"],
            overwritten,
        )
        analysis_dir = self.data_dir / "replay_analysis"
        self.assertEqual(len(list(analysis_dir.glob("*.winrate.json"))), 1)
        self.assertEqual(len(list(analysis_dir.glob(".*.generation.json"))), 1)
        self.assertEqual(list(self.data_dir.rglob("*.tmp")), [])

    def test_new_analysis_supersedes_an_in_flight_analysis(self) -> None:
        state = self.play([(0, 0, 0), (0, 1, 0)])
        summary = self.save(state, "Superseded analysis")
        first_search_entered = threading.Event()
        release_first_search = threading.Event()

        def controlled_search(board: np.ndarray, player: int, ai: dict) -> SearchResult:
            if ai["mcts_sims"] == 32:
                first_search_entered.set()
                if not release_first_search.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release the older analysis.")
                value = 0.2
            else:
                value = -0.4
            action = int(np.flatnonzero(self.service.game.get_valid_moves(board) > 0)[0])
            policy = np.zeros(self.service.game.get_action_size(), dtype=float)
            policy[action] = 1.0
            return SearchResult(action=action, policy=policy.tolist(), value=value)

        self.service._search = controlled_search
        with ThreadPoolExecutor(max_workers=2) as executor:
            older = executor.submit(
                self.service.handle,
                "replay.analyze",
                self.replay_ref(summary) | {"ai": {"mcts_sims": 32}},
            )
            self.assertTrue(first_search_entered.wait(timeout=5))
            newer = executor.submit(
                self.service.handle,
                "replay.analyze",
                self.replay_ref(summary) | {"ai": {"mcts_sims": 64}},
            )
            try:
                newest_analysis = newer.result(timeout=5)
            finally:
                release_first_search.set()

            with self.assertRaises(ServiceError) as raised:
                older.result(timeout=5)
            self.assertEqual(raised.exception.code, "ANALYSIS_SUPERSEDED")

        self.assertEqual(newest_analysis["config"]["mcts_sims"], 64)
        self.assertAlmostEqual(newest_analysis["points"][0]["red"], 0.3)
        cached = self.service.handle("replay.open", self.replay_ref(summary))["analysis"]
        self.assertEqual(cached, newest_analysis)

    def test_cross_instance_generation_prevents_an_older_analysis_overwrite(self) -> None:
        state = self.play([(0, 0, 0), (0, 1, 0)])
        summary = self.save(state, "Cross-instance analysis")
        newer_service = CubeSpriteService(RESOURCE_DIR, data_dir=self.data_dir)
        older_search_entered = threading.Event()
        release_older_search = threading.Event()

        def result_for(service: CubeSpriteService, board: np.ndarray, value: float) -> SearchResult:
            action = int(np.flatnonzero(service.game.get_valid_moves(board) > 0)[0])
            policy = np.zeros(service.game.get_action_size(), dtype=float)
            policy[action] = 1.0
            return SearchResult(action=action, policy=policy.tolist(), value=value)

        def older_search(board: np.ndarray, _player: int, _ai: dict) -> SearchResult:
            older_search_entered.set()
            if not release_older_search.wait(timeout=5):
                raise AssertionError("Timed out waiting to release the older cross-instance analysis.")
            return result_for(self.service, board, 0.2)

        def newer_search(board: np.ndarray, _player: int, _ai: dict) -> SearchResult:
            return result_for(newer_service, board, -0.4)

        self.service._search = older_search
        newer_service._search = newer_search
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                older = executor.submit(
                    self.service.handle,
                    "replay.analyze",
                    self.replay_ref(summary) | {"ai": {"mcts_sims": 32}},
                )
                self.assertTrue(older_search_entered.wait(timeout=5))
                newest_analysis = newer_service.handle(
                    "replay.analyze",
                    self.replay_ref(summary) | {"ai": {"mcts_sims": 64}},
                )
                release_older_search.set()
                with self.assertRaises(ServiceError) as raised:
                    older.result(timeout=5)
                self.assertEqual(raised.exception.code, "ANALYSIS_SUPERSEDED")

            self.assertEqual(newest_analysis["config"]["mcts_sims"], 64)
            self.assertEqual(newest_analysis["request_generation"], 2)
            cached = newer_service.handle("replay.open", self.replay_ref(summary))["analysis"]
            self.assertEqual(cached, newest_analysis)
        finally:
            release_older_search.set()
            newer_service.close()

    def test_terminal_analysis_uses_exact_result_and_delete_cascades(self) -> None:
        state = self.play(
            [
                (0, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 0, 2),
                (0, 1, 2),
                (0, 0, 3),
            ]
        )
        summary = self.save(state, "Analyzed finish")
        calls = self.install_fake_search(0.0)
        analysis = self.service.handle("replay.analyze", self.replay_ref(summary))
        self.assertEqual(len(calls), 7)
        self.assertEqual(
            analysis["points"][-1],
            {"step": 7, "red": 1.0, "blue": 0.0, "estimate": "terminal"},
        )

        replay_path = self.data_dir / "replays" / f"{summary['id']}.c4replay.json"
        analysis_path = self.data_dir / "replay_analysis" / f"{summary['id']}.winrate.json"
        generation_path = self.data_dir / "replay_analysis" / f".{summary['id']}.generation.json"
        self.assertTrue(replay_path.is_file())
        self.assertTrue(analysis_path.is_file())
        self.assertTrue(generation_path.is_file())
        self.assertEqual(
            self.service.handle("replay.delete", self.replay_ref(summary)),
            {"deleted": True},
        )
        self.assertFalse(replay_path.exists())
        self.assertFalse(analysis_path.exists())
        self.assertFalse(generation_path.exists())
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.open", self.replay_ref(summary))
        self.assertEqual(raised.exception.code, "REPLAY_NOT_FOUND")

    def test_save_requires_fresh_concurrency_token(self) -> None:
        state = self.service.handle("game.state", {})
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.save", {"name": "Missing token"})
        self.assertEqual(raised.exception.code, "MISSING_CONCURRENCY_TOKEN")

        stale = dict(self.token(state))
        moved = self.play([(0, 0, 0)])
        self.assertNotEqual(stale["expected_revision"], moved["revision"])
        with self.assertRaises(ServiceError) as raised:
            self.service.handle("replay.save", stale | {"name": "Stale"})
        self.assertEqual(raised.exception.code, "STALE_REVISION")


if __name__ == "__main__":
    unittest.main()
