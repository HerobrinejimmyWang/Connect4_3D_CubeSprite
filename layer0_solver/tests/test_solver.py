from __future__ import annotations

import unittest

from layer0_solver.game import DRAW, RED, Layer0State, transform_position
from layer0_solver.solver import DRAW as SCORE_DRAW
from layer0_solver.solver import WIN, ExactSolver, HybridSolver


class ExactSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = ExactSolver(seed=7)
        self.hybrid = HybridSolver(seed=7, time_limit=0.15, max_depth=3, native_timeout=8.0)

    def test_takes_immediate_win(self) -> None:
        state = Layer0State.from_moves((7, 1, 13, 2, 19, 3))
        analysis = self.solver.analyze(state)
        self.assertEqual(analysis.outcome, WIN)
        self.assertEqual(analysis.distance, 1)
        self.assertIn(25, analysis.optimal_moves)
        self.assertEqual(state.play(25).outcome(), RED)

    def test_blocks_single_immediate_threat(self) -> None:
        state = Layer0State.from_moves((1, 7, 2, 13, 10, 19))
        analysis = self.hybrid.analyze(state)
        self.assertEqual(analysis.optimal_moves, (25,))

    def test_detects_unavoidable_double_threat(self) -> None:
        # Blue threatens both 10 (row 2) and 25 (diagonal 7-13-19-25).
        blue = sum(1 << (p - 1) for p in (6, 7, 8, 13, 19))
        red = sum(1 << (p - 1) for p in (1, 2, 11, 12, 16))
        state = Layer0State(red_bits=red, blue_bits=blue, to_move=RED)
        score = self.solver.solve_value(state)
        self.assertEqual(score.outcome, -1)
        self.assertEqual(score.distance, 2)

    def test_grabs_known_v3_mistake(self) -> None:
        before_19 = Layer0State.from_moves((13, 9, 8, 18, 14, 12, 20, 2, 7, "pass"))
        analysis = self.hybrid.analyze(before_19)
        self.assertEqual(analysis.outcome, WIN)
        self.assertTrue(analysis.proven)
        self.assertIn(19, analysis.optimal_moves)

        before_25 = Layer0State.from_moves(
            (13, 9, 8, 18, 14, 12, 20, 2, 7, "pass", 19, 1)
        )
        finish = self.hybrid.analyze(before_25)
        self.assertEqual(finish.outcome, WIN)
        self.assertEqual(finish.distance, 1)
        self.assertIn(25, finish.optimal_moves)

    def test_rotated_position_has_rotated_optimal_move(self) -> None:
        state = Layer0State.from_moves((7, 1, 13, 2, 19, 3))
        base = self.solver.analyze(state)
        self.assertIn(25, base.optimal_moves)
        for symmetry in range(8):
            rotated = self.solver.analyze(state.transformed(symmetry))
            self.assertIn(transform_position(25, symmetry), rotated.optimal_moves)

    def test_terminal_draw_has_no_move(self) -> None:
        # A full legal-looking position with no four is supplied as a terminal table state.
        red_positions = (1, 2, 4, 8, 10, 11, 15, 17, 18, 22, 24)
        red = sum(1 << (p - 1) for p in red_positions)
        blue = ((1 << 25) - 1) ^ red
        state = Layer0State(red_bits=red, blue_bits=blue, to_move=RED)
        if state.outcome() == DRAW:
            analysis = self.solver.analyze(state)
            self.assertEqual(analysis.outcome, SCORE_DRAW)
            self.assertEqual(analysis.optimal_moves, ())


if __name__ == "__main__":
    unittest.main()
