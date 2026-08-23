from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from connect4_core import GameRules
from connect4_core.rules import (
    CLASSIC_RULE,
    DEFAULT_RULE_REGISTRY,
    RULE1,
    RULE2,
    RULE3,
    FEATURE_DIM,
    GameOutcome,
    NoLegalPlacementMode,
    RuleEngine,
    RuleFeatureSchema,
    RuleModifier,
    TurnAction,
    TurnKind,
    VerticalWinMode,
    compose_rule,
)


class RuleContractTests(unittest.TestCase):
    def test_specs_registry_and_features_are_stable_and_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            CLASSIC_RULE.rule_id = "changed"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            DEFAULT_RULE_REGISTRY._registry_hash = "changed"  # type: ignore[misc]

        self.assertIs(DEFAULT_RULE_REGISTRY.get("classic"), CLASSIC_RULE)
        self.assertIs(DEFAULT_RULE_REGISTRY.get(RULE2.rule_code), RULE2)
        self.assertIs(DEFAULT_RULE_REGISTRY.get(np.uint16(RULE2.rule_code)), RULE2)
        self.assertEqual(
            tuple(spec.rule_id for spec in DEFAULT_RULE_REGISTRY.specs),
            (
                "classic",
                "p1_vertical_ignored",
                "p1_vertical_forbidden",
                "p1_layer0_ignored",
            ),
        )
        self.assertEqual(len(DEFAULT_RULE_REGISTRY.registry_hash), 64)

        features = RuleFeatureSchema.encode(RULE2)
        self.assertEqual(len(features), FEATURE_DIM)
        self.assertEqual(features[0:3], (0.0, 0.0, 1.0))
        self.assertEqual(features[3:5], (1.0, 0.0))
        self.assertEqual(features[5:8], (0.0, 0.0, 1.0))
        self.assertTrue(all(value == 0.0 for value in features[8:]))

    def test_rule_modifiers_compose_across_independent_axes(self) -> None:
        combined = compose_rule(
            rule_id="vertical_and_layer0_ignored",
            rule_code=20,
            modifiers=(
                RuleModifier(p1_vertical_mode=VerticalWinMode.IGNORED),
                RuleModifier(p1_layer0_mode=RULE3.p1_layer0_mode),
            ),
        )
        features = RuleFeatureSchema.encode(combined)
        self.assertEqual(features[0:3], (0.0, 1.0, 0.0))
        self.assertEqual(features[3:5], (0.0, 1.0))
        self.assertEqual(combined.no_legal_placement_mode, NoLegalPlacementMode.DRAW)

    def test_classic_legal_columns_match_legacy_game_rules(self) -> None:
        legacy = GameRules()
        engine = RuleEngine(CLASSIC_RULE)
        rng = np.random.default_rng(20260819)
        board = legacy.get_init_board()
        player = 1

        for _ in range(60):
            state = engine.state_from_board(board, player_to_move=player)
            legacy_mask = legacy.get_valid_moves(board).reshape(6, 25).max(axis=0)
            np.testing.assert_array_equal(engine.legal_column_mask(state), legacy_mask)
            legal_actions = np.flatnonzero(legacy.get_valid_moves(board))
            if legal_actions.size == 0:
                break
            action = int(rng.choice(legal_actions))
            board, player = legacy.get_next_state(board, player, action)

    def test_rule1_ignores_first_player_vertical_win(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0:3, 0, 0] = 1
        engine = RuleEngine(RULE1)
        next_state = engine.step(
            engine.state_from_board(board, player_to_move=1),
            TurnAction.place(0),
        )
        self.assertEqual(next_state.outcome, GameOutcome.ONGOING)
        self.assertEqual(next_state.player_to_move, -1)

    def test_rule2_vertical_completion_is_illegal_even_if_other_line_wins(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0:3, 0, 0:3] = -1
        board[3, 0, 0:3] = 1
        board[0:3, 0, 3] = 1
        engine = RuleEngine(RULE2)
        state = engine.state_from_board(board, player_to_move=1)

        self.assertEqual(engine.legal_column_mask(state)[3], 0)
        with self.assertRaisesRegex(ValueError, "forbidden vertical"):
            engine.step(state, TurnAction.place(3))

    def test_rule3_ignores_first_player_layer0_win(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0, 0, 0:3] = 1
        engine = RuleEngine(RULE3)
        next_state = engine.step(
            engine.state_from_board(board, player_to_move=1),
            TurnAction.place(3),
        )
        self.assertEqual(next_state.outcome, GameOutcome.ONGOING)

    def test_rule1_ignored_line_does_not_hide_an_independent_scoring_line(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0:3, 0, 3] = 1
        board[0:3, 0, 0:3] = -1
        board[3, 0, 0:3] = 1
        engine = RuleEngine(RULE1)
        next_state = engine.step(
            engine.state_from_board(board, player_to_move=1),
            TurnAction.place(3),
        )
        self.assertEqual(next_state.outcome, GameOutcome.FIRST_PLAYER_WIN)

    def test_rule2_forces_pass_only_when_no_legal_placement_exists(self) -> None:
        board = np.ones((6, 5, 5), dtype=np.int8)
        board[3:, 0, 0] = 0
        engine = RuleEngine(RULE2)
        state = engine.state_from_board(board, player_to_move=1)

        self.assertFalse(engine.legal_column_mask(state).any())
        self.assertEqual(engine.required_action(state), TurnAction.forced_pass())
        passed = engine.step(state, TurnAction.forced_pass())
        self.assertEqual(passed.player_to_move, -1)
        self.assertEqual(passed.consecutive_passes, 1)
        self.assertEqual(passed.placement_count, state.placement_count)
        self.assertEqual(passed.turn_index, state.turn_index + 1)
        self.assertEqual(engine.legal_column_mask(passed)[0], 1)

        with self.assertRaisesRegex(ValueError, "only legal when no placement"):
            engine.step(passed, TurnAction.forced_pass())

    def test_two_consecutive_forced_passes_are_a_draw(self) -> None:
        board = np.ones((6, 5, 5), dtype=np.int8)
        engine = RuleEngine(RULE2)
        first = engine.state_from_board(board, player_to_move=1)
        second = engine.step(first, TurnAction.forced_pass())
        terminal = engine.step(second, TurnAction.forced_pass())

        self.assertEqual(terminal.outcome, GameOutcome.DRAW)
        self.assertEqual(terminal.consecutive_passes, 2)
        self.assertEqual(terminal.last_action.kind, TurnKind.FORCED_PASS)

    def test_real_placement_resets_consecutive_passes(self) -> None:
        board = np.ones((6, 5, 5), dtype=np.int8)
        board[3:, 0, 0] = 0
        engine = RuleEngine(RULE2)
        passed = engine.step(
            engine.state_from_board(board, player_to_move=1),
            TurnAction.forced_pass(),
        )
        placed = engine.step(passed, TurnAction.place(0))
        self.assertEqual(placed.consecutive_passes, 0)
        self.assertEqual(placed.placement_count, passed.placement_count + 1)

    def test_rule2_legal_mask_is_equivariant_under_d4_rotation_and_reflection(self) -> None:
        board = np.zeros((6, 5, 5), dtype=np.int8)
        board[0:3, 1, 3] = 1
        engine = RuleEngine(RULE2)
        mask = engine.legal_column_mask(engine.state_from_board(board, player_to_move=1)).reshape(5, 5)

        for rotations in range(4):
            rotated_board = np.rot90(board, rotations, axes=(1, 2))
            expected = np.rot90(mask, rotations, axes=(0, 1))
            actual = engine.legal_column_mask(
                engine.state_from_board(rotated_board, player_to_move=1)
            ).reshape(5, 5)
            np.testing.assert_array_equal(actual, expected)

            reflected_board = np.flip(rotated_board, axis=2)
            reflected_expected = np.flip(expected, axis=1)
            reflected_actual = engine.legal_column_mask(
                engine.state_from_board(reflected_board, player_to_move=1)
            ).reshape(5, 5)
            np.testing.assert_array_equal(reflected_actual, reflected_expected)


if __name__ == "__main__":
    unittest.main()
