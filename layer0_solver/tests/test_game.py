from __future__ import annotations

import unittest

from layer0_solver.game import (
    BLUE,
    DRAW,
    ONGOING,
    RED,
    SYMMETRY_MAPS,
    WINNING_LINES,
    Layer0State,
    canonical_pair,
    has_four,
    transform_bits,
    transform_position,
)


class Layer0GameTests(unittest.TestCase):
    def test_has_exactly_28_winning_segments(self) -> None:
        self.assertEqual(len(WINNING_LINES), 28)
        self.assertTrue(all(line.bit_count() == 4 for line in WINNING_LINES))

    def test_every_winning_segment_is_detected(self) -> None:
        for line in WINNING_LINES:
            with self.subTest(line=line):
                self.assertTrue(has_four(line))

    def test_numbering_is_row_major(self) -> None:
        state = Layer0State.from_moves((1, 25, 13))
        rows = state.rows()
        self.assertEqual(rows[0][0], RED)
        self.assertEqual(rows[4][4], BLUE)
        self.assertEqual(rows[2][2], RED)

    def test_known_v3_path_ends_in_red_diagonal_win(self) -> None:
        moves = (13, 9, 8, 18, 14, 12, 20, 2, 7, "pass", 19, 1, 25)
        state = Layer0State.from_moves(moves)
        self.assertEqual(state.outcome(), RED)
        winning = sum(1 << (position - 1) for position in (7, 13, 19, 25))
        self.assertEqual(state.red_bits & winning, winning)
        self.assertEqual(state.invisible_turns, 1)

    def test_invisible_turn_changes_player_not_layer0(self) -> None:
        state = Layer0State.from_moves((13, 9))
        passed = state.pass_invisible()
        self.assertEqual(passed.red_bits, state.red_bits)
        self.assertEqual(passed.blue_bits, state.blue_bits)
        self.assertEqual(passed.to_move, -state.to_move)
        self.assertEqual(passed.ply, state.ply + 1)

    def test_early_draw_when_both_sides_have_no_live_line(self) -> None:
        red = sum(1 << (p - 1) for p in (2, 5, 6, 7, 8, 10, 11, 14, 15, 18, 21, 24))
        blue = sum(1 << (p - 1) for p in (3, 4, 9, 12, 13, 16, 17, 19, 20, 22, 23, 25))
        state = Layer0State(red_bits=red, blue_bits=blue, to_move=RED)
        self.assertEqual(state.legal_positions, (1,))
        self.assertFalse(has_four(red))
        self.assertFalse(has_four(blue))
        self.assertEqual(state.outcome(), DRAW)

    def test_d4_maps_are_permutations_and_canonical_is_invariant(self) -> None:
        self.assertEqual(len(SYMMETRY_MAPS), 8)
        for mapping in SYMMETRY_MAPS:
            self.assertEqual(set(mapping), set(range(25)))
        state = Layer0State.from_moves((13, 9, 8, 18, 14))
        expected = canonical_pair(state.current_bits, state.opponent_bits)
        for symmetry in range(8):
            transformed = state.transformed(symmetry)
            self.assertEqual(
                canonical_pair(transformed.current_bits, transformed.opponent_bits),
                expected,
            )
            self.assertEqual(
                transform_bits(1 << 12, symmetry),
                1 << (transform_position(13, symmetry) - 1),
            )


if __name__ == "__main__":
    unittest.main()
