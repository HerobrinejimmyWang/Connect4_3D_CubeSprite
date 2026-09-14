from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.v3.stage2.readiness import verify_round3_readiness
from training.v3.stage2.round3 import (
    ROUND3_DEPENDENCY_ID,
    ROUND3_PRIMARY_ARCHITECTURES,
    build_round3_design,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage2Round3DesignTest(unittest.TestCase):
    def test_design_is_additive_tiered_and_keeps_complete_scaling_cells(self) -> None:
        first = build_round3_design()
        second = build_round3_design()
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], "connect4-v3-stage2-round3-design-v1")
        followup = first["deployment_tiers"]["balance"]["frozen_3d_followup"]
        self.assertEqual(followup["base_model"]["architecture"], "column3d_fusion_v2")
        self.assertEqual(followup["encoder_width_screen"], [64, 96, 128])
        self.assertEqual(followup["new_training_cells"], ["T1", "T2", "E96", "E128"])
        self.assertEqual(
            [row["post_trunk_mode"] for row in followup["post_trunk_screen"]],
            ["none", "serial_attention", "parallel_attention"],
        )
        self.assertEqual(first["mode"], "preparation_only")
        self.assertFalse(first["execution_authorized"])
        self.assertEqual(first["dependency"]["id"], ROUND3_DEPENDENCY_ID)
        self.assertEqual(set(first["deployment_tiers"]), {"flash", "balance", "pro"})

        balance = first["deployment_tiers"]["balance"]
        candidates = balance["candidates"]
        self.assertGreaterEqual(
            sum(bool(row["uses_explicit_3d_information"]) for row in candidates),
            len(candidates) - 1,
        )
        cells = balance["scaling"]["factorial_cells"]
        self.assertEqual(
            {(row["parameter_anchor"], row["consumed_positions"]) for row in cells},
            {
                ("b6", 1_000_000),
                ("b6", 3_000_000),
                ("b8", 1_000_000),
                ("b8", 3_000_000),
            },
        )
        for tier in ("flash", "pro"):
            scaling = first["deployment_tiers"][tier]["promoted_scaling"]
            by_anchor: dict[str, set[int]] = {}
            for row in scaling["factorial_cells"]:
                by_anchor.setdefault(row["parameter_anchor"], set()).add(
                    row["consumed_positions"]
                )
            self.assertTrue(by_anchor)
            self.assertTrue(all(values == {1_000_000, 3_000_000} for values in by_anchor.values()))


class Stage2Round3ReadinessTest(unittest.TestCase):
    def _artifact(self, root: Path, name: str, content: str = "evidence\n") -> dict[str, str]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}

    def _evidence(self, root: Path) -> dict:
        queue = self._artifact(root, "queue/state.json", "{}\n")
        archive_index = self._artifact(root, "archive/index.json", "{}\n")
        archive_receipt = self._artifact(root, "archive/receipt.json", '{"verified":true}\n')
        lines = []
        for architecture in ROUND3_PRIMARY_ARCHITECTURES:
            for initialization in ("cold", "warm"):
                stem = f"{architecture}-{initialization}"
                guard = architecture == "plane3d_fusion_resnet" and initialization == "warm"
                consumed = 3_743_776 if guard else 5_000_000
                terminal = {
                    "outcome": "safe_guard_stop" if guard else "complete_at_bound",
                    "status": "stopped_at_safe_boundary",
                    "stop_reason": "stability_pause" if guard else "max_train_positions",
                    "train_positions_consumed": consumed,
                    "safe_boundary": True,
                }
                if guard:
                    terminal["guard_reason"] = "warm-line stability guard"
                lines.append(
                    {
                        "run_id": f"stage2b_{stem}_seed271828",
                        "architecture": architecture,
                        "initialization": initialization,
                        "seed": 271828,
                        "terminal": terminal,
                        "checkpoint": self._artifact(root, f"lines/{stem}.pt", stem),
                        "checkpoint_config_hash": "a" * 64,
                        "report": self._artifact(root, f"lines/{stem}.report.json", "{}\n"),
                        "report_status": "complete",
                        "elo": {
                            "artifact": self._artifact(root, f"lines/{stem}.elo.json", "{}\n"),
                            "status": "complete",
                            "fixed_openings": True,
                            "color_swapped": True,
                            "opening_pairs": 50,
                            "observed_terminal_positions": consumed,
                        },
                    }
                )
        return {
            "schema": "connect4-v3-stage2-r2-primary-evidence-v1",
            "dependency_id": ROUND3_DEPENDENCY_ID,
            "primary_seed": 271828,
            "target_positions": 5_000_000,
            "lines": lines,
            "queue_state": queue,
            "archive": {
                "status": "verified_receipts",
                "index": archive_index,
                "receipts": [{"artifact": archive_receipt, "verified": True}],
            },
        }

    def _write_evidence(self, root: Path, payload: dict, name: str = "evidence.json") -> Path:
        path = root / name
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    def test_complete_and_explicit_safe_guard_lines_produce_hash_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-r3-ready-") as temporary:
            root = Path(temporary)
            evidence_path = self._write_evidence(root, self._evidence(root))
            receipt = verify_round3_readiness(evidence_path, repo_root=root)
            self.assertEqual(receipt["status"], "ready")
            self.assertTrue(receipt["execution_authorized"])
            self.assertEqual(receipt["dependency_id"], ROUND3_DEPENDENCY_ID)
            self.assertEqual(len(receipt["lines"]), 6)
            self.assertEqual(len(receipt["content_sha256"]), 64)
            self.assertTrue(
                any(row["terminal"]["outcome"] == "safe_guard_stop" for row in receipt["lines"])
            )

    def test_missing_warm_line_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-r3-missing-") as temporary:
            root = Path(temporary)
            evidence = self._evidence(root)
            evidence["lines"] = evidence["lines"][:-1]
            with self.assertRaisesRegex(ValueError, "lacks required cold/warm"):
                verify_round3_readiness(
                    self._write_evidence(root, evidence), repo_root=root
                )

    def test_unqualified_guard_and_unpaired_elo_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-r3-guard-") as temporary:
            root = Path(temporary)
            evidence = self._evidence(root)
            guarded = next(
                row for row in evidence["lines"] if row["terminal"]["outcome"] == "safe_guard_stop"
            )
            guarded["terminal"].pop("guard_reason")
            with self.assertRaisesRegex(ValueError, "guard_reason"):
                verify_round3_readiness(
                    self._write_evidence(root, evidence, "bad-guard.json"), repo_root=root
                )

            evidence = self._evidence(root)
            evidence["lines"][0]["elo"]["color_swapped"] = False
            with self.assertRaisesRegex(ValueError, "colors swapped"):
                verify_round3_readiness(
                    self._write_evidence(root, evidence, "bad-elo.json"), repo_root=root
                )

    def test_artifact_checksum_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-r3-hash-") as temporary:
            root = Path(temporary)
            evidence = self._evidence(root)
            evidence_path = self._write_evidence(root, copy.deepcopy(evidence))
            checkpoint = root / evidence["lines"][0]["checkpoint"]["path"]
            checkpoint.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_round3_readiness(evidence_path, repo_root=root)


if __name__ == "__main__":
    unittest.main()
