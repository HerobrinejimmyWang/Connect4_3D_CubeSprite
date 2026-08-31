from __future__ import annotations

import os
import random
import unittest

from layer0_solver.game import DRAW, Layer0State
from layer0_solver.solver import StrongSolver


@unittest.skipUnless(os.environ.get("LAYER0_RUN_SLOW") == "1", "set LAYER0_RUN_SLOW=1")
class NativeRootProofTests(unittest.TestCase):
    def test_random_exact_equal_value_self_play_stays_drawn(self) -> None:
        rng = random.Random(20260827)
        state = Layer0State()
        solver = StrongSolver(seed=20260827, timeout=180.0)
        try:
            while state.occupied.bit_count() < 25:
                if state.outcome(stop_when_dead=True) == DRAW:
                    move = rng.choice(state.legal_positions)
                else:
                    result = solver.analyze(state)
                    self.assertEqual(result.outcome, DRAW)
                    if state.ply == 0:
                        self.assertEqual(result.optimal_moves, tuple(range(1, 26)))
                    assert result.principal_move is not None
                    move = result.principal_move
                state = state.play(move, allow_after_dead_draw=True)
        finally:
            solver.close(force=True)
        self.assertEqual(state.outcome(stop_when_dead=False), DRAW)


if __name__ == "__main__":
    unittest.main()
