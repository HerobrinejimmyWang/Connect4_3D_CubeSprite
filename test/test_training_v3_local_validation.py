from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from training.v3.config import load_config
from training.v3.local_validation import (
    build_local_validation_report,
    build_p6_ablation_configs,
    validate_p6_ablation_matrix,
    write_p6_ablation_configs,
)
from training.v3.preflight import PreflightReport


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "training" / "v3" / "configs" / "smoke_cpu.json"


class LocalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(SMOKE_CONFIG)
        self.preflight = PreflightReport(
            python="test",
            numpy="test",
            torch="test",
            device="cpu",
        )

    def test_ablation_matrix_has_five_variants_and_identical_budget(self) -> None:
        rows = validate_p6_ablation_matrix(
            self.config, build_p6_ablation_configs(self.config)
        )
        self.assertEqual(
            [row["name"] for row in rows],
            [
                "baseline",
                "opponent_reply",
                "future_occupancy",
                "moves_left",
                "all_auxiliary",
            ],
        )
        self.assertTrue(all(row["loss_weights"]["policy"] == 1.0 for row in rows))
        self.assertTrue(all(row["loss_weights"]["wdl"] == 1.0 for row in rows))

    def test_matrix_rejects_unintended_optimizer_drift(self) -> None:
        variants = list(build_p6_ablation_configs(self.config))
        name, variant = variants[0]
        variants[0] = (
            name,
            replace(
                variant,
                learner=replace(variant.learner, batch_size=variant.learner.batch_size + 1),
            ),
        )
        with self.assertRaisesRegex(ValueError, "non-ablation fields"):
            validate_p6_ablation_matrix(self.config, variants)

    def test_local_report_keeps_stage1_evidence_gated(self) -> None:
        report = build_local_validation_report(self.config, self.preflight)
        self.assertTrue(report["result"]["local_contract_passed"])
        self.assertFalse(report["result"]["dataset_integrity_passed"])
        self.assertFalse(report["result"]["dataset_ready_for_p6_screening"])
        self.assertFalse(report["result"]["stage1_ready"])
        self.assertFalse(report["safety"]["starts_training"])
        self.assertTrue(report["p7"]["checks"]["bounded_formal_runner_available"])
        self.assertFalse(report["p7"]["remaining_local_connection_work"])

    def test_explicit_config_writer_is_idempotent_but_never_clobbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_p6_ablation_configs(self.config, directory)
            self.assertEqual(len(paths), 5)
            self.assertEqual(paths, write_p6_ablation_configs(self.config, directory))
            paths[0].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_p6_ablation_configs(self.config, directory)


if __name__ == "__main__":
    unittest.main()
