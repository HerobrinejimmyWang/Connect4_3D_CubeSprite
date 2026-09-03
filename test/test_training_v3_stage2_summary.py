from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.v3.stage2.summary import summarize_reports


TRAIN_REGIMES = ("standard_early", "standard_mid", "standard_late")
EVAL_REGIMES = (*TRAIN_REGIMES, "mixed_late")


def metrics(loss: float) -> dict[str, object]:
    return {
        "status": "complete",
        "total_loss": loss,
        "policy_loss": loss + 0.1,
        "wdl_loss": loss + 0.2,
        "policy_jsd": 0.03,
        "policy_top1_agreement": 0.7,
        "brier_score": 0.2,
        "calibration_error": 0.04,
        "wdl_accuracy": 0.65,
        "opponent_reply_loss": 1.0,
        "future_occupancy_loss": 0.8,
        "moves_left_loss": 2.0,
        "opponent_reply_accuracy": 0.4,
        "future_occupancy_accuracy_unweighted": 0.75,
        "moves_left_accuracy": 0.1,
    }


class Stage2SummaryTests(unittest.TestCase):
    def test_summary_requires_and_aggregates_three_by_four_cells(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-summary-") as temporary:
            root = Path(temporary)
            paths = []
            for architecture, base_loss in (("gravity_resnet", 1.0), ("column_resnet", 0.9)):
                for train_index, train_regime in enumerate(TRAIN_REGIMES):
                    report = {
                        "model": {"architecture": architecture},
                        "train_regime": train_regime,
                        "seed": 271828,
                        "efficiency": {"parameters": 100, "search_macs_estimate": 200},
                        "cross_regime": {
                            regime: metrics(base_loss + train_index * 0.01 + eval_index * 0.02)
                            for eval_index, regime in enumerate(EVAL_REGIMES)
                        },
                    }
                    path = root / f"{architecture}-{train_regime}.json"
                    path.write_text(json.dumps(report), encoding="utf-8")
                    paths.append(path)

            summary = summarize_reports(paths)
            self.assertEqual(summary["schema"], "connect4-v3-stage2-round1-summary-v2")
            self.assertEqual(summary["ranking"][0]["architecture"], "column_resnet")
            self.assertAlmostEqual(
                summary["ranking"][0]["mixed_late_generalization_gap"], 0.02
            )
            self.assertIn("gravity_resnet", summary["promoted_architectures"])
            self.assertIn("future_occupancy_loss", summary["ranking"][0])

    def test_summary_rejects_missing_mixed_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-summary-invalid-") as temporary:
            root = Path(temporary)
            paths = []
            for train_regime in TRAIN_REGIMES:
                report = {
                    "model": {"architecture": "gravity_resnet"},
                    "train_regime": train_regime,
                    "seed": 1,
                    "cross_regime": {regime: metrics(1.0) for regime in TRAIN_REGIMES},
                }
                path = root / f"{train_regime}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "cross-regime"):
                summarize_reports(paths)


if __name__ == "__main__":
    unittest.main()
