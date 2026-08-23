import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.v3.replay import (
    ARRAY_SCHEMA,
    REPLAY_SCHEMA_VERSION,
    ReplayShard,
    SEARCH_FAST,
    SEARCH_FULL,
    SEARCH_NONE,
    TURN_FORCED_PASS,
    TURN_PLACE,
    load_replay_shard,
    replay_manifest_path,
    replay_ready_path,
    sha256_file,
    write_replay_shard,
)


def _manifest_metadata() -> dict:
    return {
        "run_id": "replay-v2-test",
        "generation": 0,
        "producer_model_id": "accepted-test-model",
        "seed_range": {"start": 10, "end": 10},
        "results": {"p1_wins": 0, "p2_wins": 0, "draws": 1},
        "search_config": {"full_search_sims": 8, "fast_search_sims": 2},
        "rule_registry_hash": "4c21f13e4e5c9529f0a2a3695bb70015893191bdad65a4208076438e79db90ca",
        "config_hash": "test-config-hash",
        "git_commit": "test-tree",
    }


def _sample(*, forced_pass: bool = False, search_kind: str = "full") -> dict:
    visits = np.zeros((25,), dtype=np.uint32)
    if not forced_pass:
        visits[7] = 8
    terminal_board = np.zeros((6, 5, 5), dtype=np.int8)
    terminal_board[0, 0, :4] = 1
    return {
        "board": np.zeros((6, 5, 5), dtype=np.int8),
        "visit_counts": visits,
        "policy_weight": 0.0 if forced_pass or search_kind == "fast" else 1.0,
        "wdl": 1 if forced_pass else 0,
        "game_id": 17,
        "turn_index": 12,
        "player_to_move": -1 if forced_pass else 1,
        "search_kind": "none" if forced_pass else search_kind,
        "rule_code": 2,
        "turn_kind": "forced_pass" if forced_pass else "place",
        "placement_count": 11,
        "opponent_reply_column": -1 if forced_pass else 4,
        "opponent_reply_mask": 0 if forced_pass else 1,
        "terminal_board": terminal_board,
        "remaining_turns": 3,
    }


class ReplayV2ContractTests(unittest.TestCase):
    def test_schema_uses_canonical_v2_names_and_compatibility_aliases(self):
        self.assertEqual(REPLAY_SCHEMA_VERSION, 2)
        self.assertIn("turn_index", ARRAY_SCHEMA)
        self.assertIn("player_to_move", ARRAY_SCHEMA)
        self.assertNotIn("ply", ARRAY_SCHEMA)
        self.assertNotIn("player", ARRAY_SCHEMA)

        shard = ReplayShard.from_samples([_sample(), _sample(forced_pass=True)])
        np.testing.assert_array_equal(shard.ply, shard.turn_index)
        np.testing.assert_array_equal(shard.player, shard.player_to_move)
        self.assertEqual(int(shard.turn_kind[0]), TURN_PLACE)
        self.assertEqual(int(shard.turn_kind[1]), TURN_FORCED_PASS)
        self.assertEqual(int(shard.search_kind[1]), SEARCH_NONE)
        self.assertEqual(int(shard.visit_counts[1].sum()), 0)

    def test_from_samples_accepts_old_attribute_names_but_requires_v2_fields(self):
        sample = _sample()
        sample["ply"] = sample.pop("turn_index")
        sample["player"] = sample.pop("player_to_move")
        shard = ReplayShard.from_samples([sample])
        self.assertEqual(int(shard.turn_index[0]), 12)
        self.assertEqual(int(shard.player_to_move[0]), 1)

        old_row = {
            "board": sample["board"],
            "visit_counts": sample["visit_counts"],
            "wdl": 0,
            "game_id": 17,
            "ply": 12,
            "player": 1,
            "search_kind": "fast",
        }
        with self.assertRaises((KeyError, AttributeError)):
            ReplayShard.from_samples([_sample(), old_row], full_only=True)

    def test_fast_placement_policy_can_be_masked_but_visits_are_required(self):
        fast = ReplayShard.from_samples([_sample(search_kind="fast")])
        self.assertEqual(int(fast.search_kind[0]), SEARCH_FAST)
        self.assertEqual(float(fast.policy_weight[0]), 0.0)
        self.assertGreater(int(fast.visit_counts[0].sum()), 0)

        invalid = _sample(search_kind="fast")
        invalid["visit_counts"] = np.zeros((25,), dtype=np.uint32)
        with self.assertRaisesRegex(ValueError, "placement.*at least one visit"):
            ReplayShard.from_samples([invalid])

    def test_forced_pass_invariants_are_enforced(self):
        invalid_search = _sample(forced_pass=True)
        invalid_search["search_kind"] = "fast"
        with self.assertRaisesRegex(ValueError, "forced-pass.*search_kind=none"):
            ReplayShard.from_samples([invalid_search])

        invalid_visits = _sample(forced_pass=True)
        invalid_visits["visit_counts"][0] = 1
        with self.assertRaisesRegex(ValueError, "forced-pass.*zero visits"):
            ReplayShard.from_samples([invalid_visits])

        invalid_weight = _sample(forced_pass=True)
        invalid_weight["policy_weight"] = 1.0
        with self.assertRaisesRegex(ValueError, "forced-pass.*policy_weight=0"):
            ReplayShard.from_samples([invalid_weight])

    def test_auxiliary_label_sentinels_and_turn_bounds_are_enforced(self):
        masked_reply = _sample()
        masked_reply["opponent_reply_mask"] = 0
        with self.assertRaisesRegex(ValueError, "masked.*-1 sentinel"):
            ReplayShard.from_samples([masked_reply])

        too_long = _sample()
        too_long["remaining_turns"] = 301
        with self.assertRaisesRegex(ValueError, r"outside \[0, 300\]"):
            ReplayShard.from_samples([too_long])

    def test_v2_round_trip_and_v1_marker_rejection(self):
        shard = ReplayShard.from_samples([_sample(), _sample(forced_pass=True)])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replay-v2.npz"
            manifest = write_replay_shard(path, shard, _manifest_metadata())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(set(manifest["arrays"]), set(ARRAY_SCHEMA))
            restored, _ = load_replay_shard(path)
            for name in ARRAY_SCHEMA:
                np.testing.assert_array_equal(getattr(restored, name), getattr(shard, name))

            manifest_path = replay_manifest_path(path)
            ready_path = replay_ready_path(path)
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            old_manifest["schema_version"] = 1
            manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
            old_ready = json.loads(ready_path.read_text(encoding="utf-8"))
            old_ready["schema_version"] = 1
            old_ready["manifest_sha256"] = sha256_file(manifest_path)
            ready_path.write_text(json.dumps(old_ready), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported replay schema: 1"):
                load_replay_shard(path)


if __name__ == "__main__":
    unittest.main()
