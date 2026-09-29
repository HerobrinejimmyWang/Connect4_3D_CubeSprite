import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from stage2_fla_post3m_selection import NAMES, decide  # noqa: E402


class TerminalSelectionTest(unittest.TestCase):
    def evidence(self, means, elos, intervals):
        cpu = {name: {"mean_s": mean, "passes_3_3_s": mean <= 3.3}
               for name, mean in zip(NAMES, means)}
        ratings = {"ratings": {name: {"elo": elo, "ci95": ci}
                               for name, elo, ci in zip(NAMES, elos, intervals)}}
        inputs = {"models": {name: {"terminal_sha256": "a" * 64,
                                    "snapshot_sha256": "b" * 64} for name in NAMES}}
        return cpu, ratings, inputs

    @patch("stage2_fla_post3m_selection.sha256", return_value="c" * 64)
    def test_strong_slow_model_replaces_second(self, _):
        cpu, ratings, inputs = self.evidence(
            [2.5, 3.4, 2.9, 3.1],
            [20, 100, 10, 0],
            [[10, 30], [80, 120], [0, 20], [-10, 10]],
        )
        result = decide(cpu, ratings, inputs)
        self.assertEqual(result["selected"], [NAMES[0], NAMES[1]])
        self.assertEqual(result["exception_model"], NAMES[1])
        self.assertIn(NAMES[2], result["eliminated"])

    @patch("stage2_fla_post3m_selection.sha256", return_value="c" * 64)
    def test_overlap_does_not_trigger_exception(self, _):
        cpu, ratings, inputs = self.evidence(
            [2.5, 3.4, 2.9, 3.1],
            [20, 30, 10, 0],
            [[10, 30], [15, 45], [0, 20], [-10, 10]],
        )
        result = decide(cpu, ratings, inputs)
        self.assertEqual(result["selected"], [NAMES[0], NAMES[2]])
        self.assertIsNone(result["exception_model"])

    @patch("stage2_fla_post3m_selection.sha256", return_value="c" * 64)
    def test_one_latency_pass_does_not_force_second(self, _):
        cpu, ratings, inputs = self.evidence(
            [2.5, 3.4, 3.9, 4.0],
            [20, 30, 10, 0],
            [[10, 30], [15, 45], [0, 20], [-10, 10]],
        )
        result = decide(cpu, ratings, inputs)
        self.assertEqual(result["selected"], [NAMES[0]])
        self.assertEqual(result["status"], "insufficient_qualifiers_requires_review")


if __name__ == "__main__":
    unittest.main()
