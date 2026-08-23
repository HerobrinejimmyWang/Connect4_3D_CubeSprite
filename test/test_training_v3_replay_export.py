from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "desktop_app" / "backend"
for import_root in (REPO_ROOT, BACKEND_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from connect4_core import GameRules  # noqa: E402
from cubesprite_backend.replay_store import (  # noqa: E402
    ReplayStore,
    replay_fingerprint as app_replay_fingerprint,
    validate_replay as app_validate_replay,
)
from training.v3.replay_export import (  # noqa: E402
    PLACE_TURN_KEYS,
    REPLAY_KEYS,
    build_audit_index,
    game_record_to_replay,
    replay_fingerprint,
    select_representative_games,
    serialize_replay_document,
    write_audit_index_atomic,
    write_replay_atomic,
)
from training.v3.selfplay import GameRecord, MoveRecord  # noqa: E402


P1_WIN_ACTIONS = (0, 5, 1, 6, 2, 7, 3)


def make_game(
    game_id: int,
    *,
    actions: tuple[int, ...] = P1_WIN_ACTIONS,
    winner: int = 1,
    generation: int = 3,
) -> GameRecord:
    rules = GameRules()
    moves = []
    player = 1
    for ply, action in enumerate(actions):
        _layer, row, col = rules.action_to_coords(action)
        moves.append(
            MoveRecord(
                ply=ply,
                player=player,
                column=row * rules.board_size + col,
                legacy_action=action,
                search_kind="full" if ply % 2 == 0 else "fast",
                simulations=32 if ply % 2 == 0 else 8,
            )
        )
        player = -player
    full_positions = sum(move.search_kind == "full" for move in moves)
    fast_positions = len(moves) - full_positions
    return GameRecord(
        game_id=game_id,
        seed=9000 + game_id,
        generation=generation,
        producer_model_id="accepted-g000002",
        winner=winner,
        is_draw=winner == 0,
        moves=tuple(moves),
        samples=(),
        full_search_positions=full_positions,
        fast_search_positions=fast_positions,
        total_simulations=sum(move.simulations for move in moves),
    )


class V3ReplayExportTests(unittest.TestCase):
    def test_document_round_trips_through_desktop_app_validator_and_store(self) -> None:
        game = make_game(17)
        saved_at = datetime(2026, 8, 11, 12, 34, 56, tzinfo=timezone.utc)
        document = game_record_to_replay(game, run_id="audit-run", saved_at=saved_at)

        self.assertEqual(set(document), REPLAY_KEYS)
        self.assertTrue(
            all(set(turn) == PLACE_TURN_KEYS for turn in document["turns"])
        )
        self.assertEqual(
            [turn["ply"] for turn in document["turns"]], list(range(1, 8))
        )
        self.assertEqual(
            [turn["action"] for turn in document["turns"]], list(P1_WIN_ACTIONS)
        )
        self.assertEqual(document["saved_at"], "2026-08-11T12:34:56Z")
        self.assertEqual(document["status"], "won")
        self.assertEqual(document["winner"], 1)
        self.assertEqual(replay_fingerprint(document), app_replay_fingerprint(document))
        self.assertEqual(app_validate_replay(document, GameRules()), document)

        with tempfile.TemporaryDirectory(prefix="v3-replay-export-") as temporary:
            root = Path(temporary)
            replay_path = root / "sample.c4replay.json"
            write_replay_atomic(replay_path, document)
            write_replay_atomic(replay_path, document)
            content = replay_path.read_text(encoding="utf-8")
            stored = ReplayStore(root / "app-data", GameRules()).import_content(
                content,
                filename=replay_path.name,
            )
            self.assertEqual(stored, document)

            changed = dict(document)
            changed["name"] = "Different display name"
            with self.assertRaises(FileExistsError):
                write_replay_atomic(replay_path, changed)

    def test_export_rejects_nonterminal_or_inconsistent_game(self) -> None:
        valid = make_game(2)
        with self.assertRaisesRegex(ValueError, "completed"):
            game_record_to_replay(
                replace(valid, moves=valid.moves[:2]),
                run_id="audit-run",
                saved_at="2026-08-11T00:00:00Z",
            )
        with self.assertRaisesRegex(ValueError, "winner"):
            game_record_to_replay(
                replace(valid, winner=-1, is_draw=False),
                run_id="audit-run",
                saved_at="2026-08-11T00:00:00Z",
            )
        with self.assertRaisesRegex(ValueError, "timezone"):
            game_record_to_replay(
                valid,
                run_id="audit-run",
                saved_at=datetime(2026, 8, 11),
            )

    def test_selection_is_order_independent_and_covers_available_outcomes(self) -> None:
        lengths = (4, 5, 6, 7, 8, 9, 10, 11)
        winners = (1, 1, -1, 1, 0, 1, -1, 1)
        games = []
        for index, (length, winner) in enumerate(zip(lengths, winners), start=1):
            base = make_game(index)
            synthetic_moves = tuple(
                replace(base.moves[ply % len(base.moves)], ply=ply)
                for ply in range(length)
            )
            games.append(
                replace(
                    base,
                    moves=synthetic_moves,
                    winner=winner,
                    is_draw=winner == 0,
                )
            )

        forward = select_representative_games(games, limit=5)
        backward = select_representative_games(reversed(games), limit=5)
        self.assertEqual(
            [(selection.game_id, selection.reasons) for selection in forward],
            [(selection.game_id, selection.reasons) for selection in backward],
        )
        self.assertEqual(len(forward), 5)
        self.assertEqual({selection.game.winner for selection in forward}, {-1, 0, 1})
        selected_lengths = {len(selection.game.moves) for selection in forward}
        self.assertIn(sorted(lengths)[1], selected_lengths)
        self.assertIn(sorted(lengths)[-2], selected_lengths)

    def test_audit_index_keeps_training_metadata_outside_portable_replay(self) -> None:
        game = make_game(23)
        selection = select_representative_games([game], limit=1)
        document = game_record_to_replay(
            game,
            run_id="audit-run",
            saved_at="2026-08-11T00:00:00Z",
        )
        checkpoint_sha256 = "a" * 64
        index = build_audit_index(
            selection,
            {game.game_id: document},
            run_id="audit-run",
            generation=game.generation,
            checkpoint_id="checkpoint-g000003",
            checkpoint_sha256=checkpoint_sha256,
            created_at="2026-08-11T00:01:00Z",
        )

        self.assertNotIn("producer_model_id", document)
        self.assertNotIn("game_id", document)
        self.assertEqual(index["replays"][0]["producer_model_id"], game.producer_model_id)
        expected_file_hash = hashlib.sha256(
            serialize_replay_document(document).encode("utf-8")
        ).hexdigest()
        self.assertEqual(index["replays"][0]["file_sha256"], expected_file_hash)

        with tempfile.TemporaryDirectory(prefix="v3-audit-index-") as temporary:
            target = Path(temporary) / "audit_index.json"
            write_audit_index_atomic(target, index)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), index)


if __name__ == "__main__":
    unittest.main()
