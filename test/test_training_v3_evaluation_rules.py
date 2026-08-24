from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from connect4_core.rules import RuleEngine, TurnAction, TurnKind
from training.v3.evaluation import (
    MAX_GATE_TURNS,
    OPENING_SCHEMA_VERSION,
    Opening,
    build_openings,
    load_opening_manifest,
    play_paired_game,
    play_paired_openings,
    write_opening_manifest,
)
from training.v3.gate import GateGameResult
from training.v3.search import RandomPredictor


class OpeningRuleContractTests(unittest.TestCase):
    def test_rule_context_round_trips_in_schema_v2_manifest(self) -> None:
        openings = build_openings(
            8,
            run_seed=819,
            rule_id="p1_vertical_forbidden",
            prefix_lengths=(0, 2, 4, 6),
        )
        self.assertTrue(
            all(
                opening.rule_id == "p1_vertical_forbidden" and opening.rule_version == 1
                for opening in openings
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "openings.json"
            write_opening_manifest(path, openings)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_opening_manifest(path)

        self.assertEqual(payload["schema_version"], OPENING_SCHEMA_VERSION)
        self.assertEqual(payload["rule_id"], "p1_vertical_forbidden")
        self.assertEqual(payload["rule_version"], 1)
        self.assertEqual(payload["openings"][0]["rule_id"], payload["rule_id"])
        self.assertEqual(loaded, openings)

    def test_manifest_rejects_a_row_with_mismatched_rule_context(self) -> None:
        openings = build_openings(2, run_seed=17, rule_id="classic", prefix_lengths=(0, 2))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "openings.json"
            write_opening_manifest(path, openings)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["openings"][0]["rule_id"] = "p1_layer0_ignored"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest rule context"):
                load_opening_manifest(path)

    def test_generated_columns_replay_through_the_explicit_rule_engine(self) -> None:
        openings = build_openings(
            20,
            run_seed=77,
            rule_id="p1_layer0_ignored",
            prefix_lengths=(0, 2, 4, 6, 8),
        )
        engine = RuleEngine("p1_layer0_ignored")
        keys = []
        for opening in openings:
            state = engine.initial_state()
            for column in opening.columns:
                required = engine.required_action(state)
                if required is not None:
                    state = engine.step(state, required)
                self.assertEqual(engine.legal_column_mask(state)[column], 1)
                state = engine.step(state, TurnAction.place(column))
                self.assertFalse(state.terminal)
            variants = []
            for transform in range(8):
                transformed = np.rot90(state.board, transform % 4, axes=(-2, -1))
                if transform >= 4:
                    transformed = np.flip(transformed, axis=-1)
                variants.append(np.ascontiguousarray(transformed).tobytes())
            keys.append((state.rule_id, state.player_to_move, min(variants)))
        self.assertEqual(len(keys), len(set(keys)))


class GateRuleContractTests(unittest.TestCase):
    @staticmethod
    def _forced_pass_root(engine: RuleEngine, _opening: Opening):
        board = np.ones((6, 5, 5), dtype=np.int8)
        board[3:, 0, 0] = 0
        return engine.state_from_board(board, player_to_move=1)

    def test_gate_advances_required_pass_without_policy_search(self) -> None:
        opening = Opening(
            "forced-pass",
            seed=4,
            columns=(),
            rule_id="p1_vertical_forbidden",
            rule_version=1,
        )
        actions = []
        original_step = RuleEngine.step

        def record_step(engine, state, action):
            actions.append(action)
            return original_step(engine, state, action)

        with (
            mock.patch("training.v3.evaluation._apply_opening", side_effect=self._forced_pass_root),
            mock.patch.object(RuleEngine, "step", autospec=True, side_effect=record_step),
        ):
            result = play_paired_game(
                opening,
                candidate_is_first=False,
                candidate_predictor=RandomPredictor(),
                incumbent_predictor=None,
                search_sims=1,
                cpuct=1.0,
            )

        self.assertIsInstance(result, GateGameResult)
        self.assertGreater(len(actions), 1)
        self.assertEqual(actions[0].kind, TurnKind.FORCED_PASS)
        self.assertTrue(all(action.kind != TurnKind.FORCED_PASS for action in actions[1:]))

    def test_color_swaps_reuse_the_same_opening_rule(self) -> None:
        opening = Opening(
            "same-rule",
            seed=5,
            columns=(),
            rule_id="p1_layer0_ignored",
            rule_version=1,
        )
        seen = []

        def fake_play(row, *, candidate_is_first, **_kwargs):
            seen.append((row.rule_id, row.rule_version, candidate_is_first))
            return GateGameResult(row.opening_id, row.seed, candidate_is_first, 0.5)

        with mock.patch("training.v3.evaluation.play_paired_game", side_effect=fake_play):
            results = play_paired_openings(
                (opening,),
                candidate_predictor=RandomPredictor(),
                incumbent_predictor=None,
                search_sims=1,
                cpuct=1.0,
            )
        self.assertEqual(len(results), 2)
        self.assertEqual(
            seen,
            [
                ("p1_layer0_ignored", 1, True),
                ("p1_layer0_ignored", 1, False),
            ],
        )

    def test_gate_turn_limit_is_explicitly_300(self) -> None:
        self.assertEqual(MAX_GATE_TURNS, 300)

    def test_asymmetric_search_budget_follows_model_identity_across_color_swap(self) -> None:
        opening = Opening("budget-identity", seed=9, columns=(), rule_id="classic", rule_version=1)
        candidate = RandomPredictor()
        incumbent = RandomPredictor()
        seen = []

        def fake_search(predictor, state, *, engine, simulations, **_kwargs):
            seen.append((predictor, simulations))
            return int(np.flatnonzero(engine.legal_column_mask(state))[0])

        with mock.patch("training.v3.evaluation._search_column", side_effect=fake_search):
            for candidate_is_first in (True, False):
                play_paired_game(
                    opening,
                    candidate_is_first=candidate_is_first,
                    candidate_predictor=candidate,
                    incumbent_predictor=incumbent,
                    search_sims=1,
                    candidate_search_sims=256,
                    incumbent_search_sims=512,
                    cpuct=1.0,
                )

        self.assertTrue(seen)
        self.assertTrue(
            all(simulations == (256 if predictor is candidate else 512) for predictor, simulations in seen)
        )


if __name__ == "__main__":
    unittest.main()
