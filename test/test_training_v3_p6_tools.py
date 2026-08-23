from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tools.run_v3_p6_screen import _occupancy_weights


class P6ToolTests(unittest.TestCase):
    def test_occupancy_weights_are_derived_from_replay_v2_fields(self) -> None:
        board = np.zeros((1, 6, 5, 5), dtype=np.int8)
        terminal = board.copy()
        terminal[0, 0, 0, 0] = 1
        terminal[0, 0, 0, 1] = -1
        replay = SimpleNamespace(
            board=board,
            terminal_board=terminal,
            player_to_move=np.asarray([1], dtype=np.int8),
        )

        own, opponent, empty = _occupancy_weights(replay)

        self.assertEqual((own, opponent), (5.0, 5.0))
        self.assertAlmostEqual(empty, 150.0 / (3.0 * 148.0))


if __name__ == "__main__":
    unittest.main()
