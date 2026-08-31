from __future__ import annotations

import unittest

from layer0_solver.game import DRAW, Layer0State
from layer0_solver.native import NativeSolver, PersistentNativeSolver


class ExactEquivalenceRegressionTests(unittest.TestCase):
    def test_old_random_losing_reply_is_not_equal_value(self) -> None:
        state = Layer0State.from_moves((13,))
        solver = NativeSolver(timeout=30.0)
        if not solver.available:
            self.skipTest("native solver is not built")
        result = solver.analyze(state)
        self.assertEqual(result.outcome, DRAW)
        self.assertEqual(result.optimal_moves, (7, 9, 17, 19))
        self.assertNotIn(18, result.optimal_moves)

    def test_persistent_backend_reuses_exact_midgame_cache(self) -> None:
        state = Layer0State.from_moves((13, 9, 8, 18, 14, 12, 20, 2, 7, "pass"))
        solver = PersistentNativeSolver(timeout=30.0)
        if not solver.available:
            self.skipTest("native solver is not built")
        try:
            first = solver.analyze(state)
            second = solver.analyze(state)
        finally:
            solver.close(force=True)
        self.assertEqual(first.outcome, 1)
        self.assertEqual(first.optimal_moves, (19,))
        self.assertEqual(second.optimal_moves, first.optimal_moves)
        self.assertLess(second.nodes, first.nodes)


if __name__ == "__main__":
    unittest.main()
