from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.v3.stage2.metrics import prepare_trajectory_metrics


class Stage2MetricsTests(unittest.TestCase):
    def test_formal_events_are_flattened_with_bracketed_strength(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-metrics-") as temporary:
            root = Path(temporary)
            events = root / "events.jsonl"
            rows = []
            for generation in range(9):
                rows.extend(
                    (
                        {
                            "stage": "selfplay",
                            "generation": generation,
                            "health": {
                                "game_length": {"mean": 20 + generation, "short_le_12_rate": 0.2},
                                "mean_policy_entropy": {"full": 1.5},
                            },
                        },
                        {
                            "stage": "generation_commit",
                            "generation": generation,
                            "accepted_model_id": f"model-{generation // 3}",
                        },
                    )
                )
            events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            points = root / "points.json"
            points.write_text(
                json.dumps(
                    {
                        "schema": "connect4-v3-stage2-strength-points-v1",
                        "registry_hash": "806753498c10ce585a9b7586276eaa9037637be2b050072d0b684b9461773b79",
                        "profile_id": "primary_256",
                        "points": [
                            {"generation": 0, "anchored_strength": 0, "report_sha256": "a" * 64},
                            {"generation": 4, "anchored_strength": 40, "report_sha256": "b" * 64},
                            {"generation": 8, "anchored_strength": 80, "report_sha256": "c" * 64},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "flat.jsonl"
            report = prepare_trajectory_metrics(events, points, output, cadence_window=3)
            flattened = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["generation_count"], 9)
            self.assertEqual(flattened[2]["anchored_strength"], 20.0)
            self.assertEqual(flattened[4]["mean_game_length"], 24.0)

    def test_strength_points_must_bracket_trajectory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-metrics-invalid-") as temporary:
            root = Path(temporary)
            events = root / "events.jsonl"
            rows = []
            for generation in range(9):
                rows += [
                    {"stage": "selfplay", "generation": generation, "health": {"game_length": {"mean": 1, "short_le_12_rate": 1}, "mean_policy_entropy": {"full": 1}}},
                    {"stage": "generation_commit", "generation": generation, "accepted_model_id": "m"},
                ]
            events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            points = root / "points.json"
            points.write_text(json.dumps({"schema": "connect4-v3-stage2-strength-points-v1", "registry_hash": "806753498c10ce585a9b7586276eaa9037637be2b050072d0b684b9461773b79", "profile_id": "primary_256", "points": [{"generation": 1, "anchored_strength": 1, "report_sha256": "a" * 64}, {"generation": 4, "anchored_strength": 2, "report_sha256": "b" * 64}, {"generation": 7, "anchored_strength": 3, "report_sha256": "c" * 64}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bracket"):
                prepare_trajectory_metrics(events, points, root / "out.jsonl")


if __name__ == "__main__":
    unittest.main()
