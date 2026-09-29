"""Focused tests for the independent Stage 2 migration verifier.

The verifier exists because a migration receipt is a *claim*: it is written by
the process that performed the move.  These tests therefore concentrate on the
cases where a self-consistent receipt is still wrong -- content that drifted
after the copy, a file attributed to the wrong run, and a source that was
declared retired but survived.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

from stage2_scratch import make_scratch_dir, remove_scratch_dir
from tools.migrate_stage2_archive import (
    build_plan,
    execute_plan,
    _fs_path,
)
from tools.verify_stage2_archive_migration import (
    VerificationFailure,
    sha256_file,
    verify_migration,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Stage2ArchiveVerificationTests(unittest.TestCase):
    def setUp(self):
        self.root = make_scratch_dir("stage2-verify-")
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        self.source.mkdir(parents=True)
        self.manifest_path = self.root / "manifest.json"
        self.receipt_path = self.root / "receipt.json"

    def tearDown(self):
        remove_scratch_dir(self.root)

    # ---- fixtures --------------------------------------------------------

    def _write_run(self, name, files, archive_hash=None):
        """Create a run directory shaped like a real compact archive member."""

        run_root = self.source / name
        materialized = run_root / "materialized"
        entries = []
        for relative, data in files.items():
            target = materialized / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entries.append(
                {"path": relative, "size_bytes": len(data), "sha256": _sha(data)}
            )
        archive_hash = archive_hash or _sha(name.encode("utf-8"))
        compact = {
            "schema": "connect4-stage2-compact-run-v1",
            "run_id": name,
            "archive_sha256": archive_hash,
            "entries": entries,
        }
        manifest_file = run_root / "compact_manifest.json"
        manifest_file.write_text(json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8")
        receipt = {
            "schema": "connect4-stage2-compact-receipt-v1",
            "run_id": name,
            "archive_sha256": archive_hash,
            "manifest_sha256": sha256_file(manifest_file),
            "endpoints": {"terminal_checkpoint": entries[0]["path"] if entries else None},
            "entries": len(entries),
            "verified": True,
        }
        (run_root / "compact_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        return run_root

    def _two_run_archive(self):
        self._write_run("stage2b_alpha_cold_seed1", {"metrics/a.json": b"alpha\n",
                                                     "checkpoints/c.pt": b"pt-alpha"})
        self._write_run("stage2b_beta_warm_seed1", {"metrics/b.json": b"beta\n"})
        manifest = {
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.destination),
            "entries": [
                {
                    "run_id": "stage2b_alpha_cold_seed1",
                    "source": "stage2b_alpha_cold_seed1",
                    "destination": "experiments/stage2b/closed_loop/alpha/cold/seed1/"
                                   "stage2b_alpha_cold_seed1",
                },
                {
                    "run_id": "stage2b_beta_warm_seed1",
                    "source": "stage2b_beta_warm_seed1",
                    "destination": "experiments/stage2b/closed_loop/beta/warm/seed1/"
                                   "stage2b_beta_warm_seed1",
                },
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _migrate(self, remove_source=True):
        self._two_run_archive()
        plan = build_plan(self.manifest_path)
        receipt = execute_plan(plan, remove_source=remove_source)
        self.receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return receipt

    def _verify(self, **kwargs):
        return verify_migration(self.manifest_path, self.receipt_path, **kwargs)

    @staticmethod
    def _failed(report):
        return {row["check"] for row in report["findings"]}

    @staticmethod
    def _find(rows, posix_path):
        """Locate receipt inventory rows by path.

        The migration tool writes inventory paths with the native separator, so
        a literal forward-slash lookup silently matches nothing on Windows.
        """

        target = str(posix_path).replace("\\", "/")
        return [row for row in rows if str(row["path"]).replace("\\", "/") == target]

    # ---- tests -----------------------------------------------------------

    def test_clean_migration_verifies(self):
        self._migrate()
        report = self._verify()
        self.assertEqual(report["status"], "verified", report["findings"])
        self.assertEqual(report["entry_count"], 2)
        self.assertEqual(report["disk_file_count"], report["declared_file_count"])
        self.assertEqual(report["disk_bytes"], report["declared_bytes"])
        # 2 + 1 materialized files plus 2 metadata files per run.
        self.assertEqual(report["disk_file_count"], 3 + 4)
        self.assertEqual(report["oracle_entry_count"], 3)
        self.assertTrue(all(row["oracle"] == "ok" for row in report["runs"]))

    def test_reports_source_retained_when_not_removed(self):
        self._migrate(remove_source=False)
        report = self._verify(expect_remove_source=False)
        self.assertEqual(report["status"], "verified", report["findings"])
        checks = {row["check"]: row["passed"] for row in report["checks"]}
        self.assertNotIn("source.retired", checks)

        strict = self._verify()
        self.assertEqual(strict["status"], "failed")
        self.assertIn("source.retired", self._failed(strict))

    def test_detects_content_that_drifted_after_the_copy(self):
        receipt = self._migrate()
        victim = (
            self.destination
            / "experiments/stage2b/closed_loop/alpha/cold/seed1/"
              "stage2b_alpha_cold_seed1/materialized/metrics/a.json"
        )
        victim.write_bytes(b"tampered\n")
        report = self._verify()
        self.assertEqual(report["status"], "failed")
        self.assertIn("disk.matches_receipt_inventory", self._failed(report))
        self.assertEqual(receipt["status"], "completed")

    def test_detects_file_recorded_under_the_wrong_run(self):
        """A byte-perfect copy can still be attributed to the wrong lineage."""

        self._migrate()
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))

        alpha_prefix = "experiments/stage2b/closed_loop/alpha/cold/seed1/stage2b_alpha_cold_seed1"
        beta_prefix = "experiments/stage2b/closed_loop/beta/warm/seed1/stage2b_beta_warm_seed1"
        # Move a real file from beta into alpha's tree and rewrite the receipt so
        # the destination inventory still matches the disk exactly.
        old_rel = f"{beta_prefix}/materialized/metrics/b.json"
        new_rel = f"{alpha_prefix}/materialized/metrics/b.json"
        (self.destination / Path(old_rel)).rename(self.destination / Path(new_rel))
        moved = self._find(receipt["destination_inventory"], old_rel)
        self.assertEqual(len(moved), 1)
        moved[0]["path"] = new_rel
        self.receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        report = self._verify()
        checks = {row["check"]: row["passed"] for row in report["checks"]}
        self.assertTrue(checks["disk.matches_receipt_inventory"])
        self.assertFalse(checks["receipt.path_correspondence"])
        self.assertEqual(report["status"], "failed")

    def test_detects_oracle_drift_hidden_by_a_self_consistent_receipt(self):
        """The case the compact-manifest oracle exists for.

        A receipt can be internally consistent -- identical source and
        destination multisets, matching totals, correct paths -- while the
        archive no longer holds the content its own compact manifest declares.
        """

        self._migrate()
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        alpha_prefix = "experiments/stage2b/closed_loop/alpha/cold/seed1/stage2b_alpha_cold_seed1"
        rel = f"{alpha_prefix}/materialized/checkpoints/c.pt"
        replacement = b"pt-omega"
        (self.destination / Path(rel)).write_bytes(replacement)
        source_rel = "stage2b_alpha_cold_seed1/materialized/checkpoints/c.pt"

        for key in ("destination_inventory", "source_inventory"):
            for row in receipt[key]:
                if str(row["path"]).replace("\\", "/") in (rel, source_rel):
                    row["size"] = len(replacement)
                    row["sha256"] = _sha(replacement)
        for key in ("source_bytes", "destination_bytes"):
            receipt[key] = receipt[key] - len(b"pt-alpha") + len(replacement)
        self.receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        report = self._verify()
        failed = self._failed(report)
        self.assertEqual(report["status"], "failed")
        self.assertIn("oracle.compact_manifests", failed)
        self.assertNotIn("disk.matches_receipt_inventory", failed)
        self.assertNotIn("receipt.content_multiset", failed)
        self.assertNotIn("receipt.path_correspondence", failed)

    def test_detects_totals_that_disagree_with_the_inventories(self):
        self._migrate()
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        receipt["destination_file_count"] = receipt["destination_file_count"] + 1
        self.receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        report = self._verify()
        self.assertEqual(report["status"], "failed")
        failed = self._failed(report)
        self.assertIn("receipt.counts_agree", failed)
        self.assertIn("disk.file_count", failed)

    def test_rejects_an_unfinished_or_foreign_receipt(self):
        self._migrate()
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        receipt["status"] = "planned"
        receipt["source_deleted"] = False
        self.receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        report = self._verify()
        failed = self._failed(report)
        self.assertIn("receipt.status", failed)
        self.assertIn("receipt.source_deleted", failed)

        broken = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        broken["schema"] = "something-else"
        self.receipt_path.write_text(json.dumps(broken, indent=2), encoding="utf-8")
        with self.assertRaises(VerificationFailure):
            self._verify()

    def test_detects_missing_materialized_file_against_the_oracle(self):
        self._migrate()
        victim = (
            self.destination
            / "experiments/stage2b/closed_loop/beta/warm/seed1/"
              "stage2b_beta_warm_seed1/materialized/metrics/b.json"
        )
        victim.unlink()
        report = self._verify()
        self.assertEqual(report["status"], "failed")
        failed = self._failed(report)
        self.assertIn("disk.matches_receipt_inventory", failed)
        self.assertIn("oracle.compact_manifests", failed)

    def test_optional_oracle_mode_accepts_a_non_compact_subtree(self):
        """round3_balance-style groups have no compact manifest to reconcile."""

        (self.source / "plain").mkdir(parents=True)
        (self.source / "plain" / "results_matrix.json").write_bytes(b"matrix\n")
        manifest = {
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.destination),
            "entries": [
                {
                    "run_id": "stage2r3_round3_balance",
                    "source": "plain",
                    "destination": "experiments/stage2r3/round3_balance",
                }
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        receipt = execute_plan(build_plan(self.manifest_path), remove_source=True)
        self.receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        required = self._verify()
        self.assertEqual(required["status"], "failed")
        self.assertIn("oracle.compact_manifests", self._failed(required))

        optional = self._verify(oracle_mode="optional")
        self.assertEqual(optional["status"], "verified", optional["findings"])
        self.assertEqual(optional["runs"][0]["oracle"], "not_a_compact_archive")

        skipped = self._verify(oracle_mode="skip")
        self.assertEqual(skipped["status"], "verified", skipped["findings"])

        with self.assertRaises(VerificationFailure):
            self._verify(oracle_mode="nonsense")

    @unittest.skipUnless(os.name == "nt", "extended Windows paths are Windows-only")
    def test_verifies_paths_beyond_legacy_max_path(self):
        long_part = "long-directory-" + ("y" * 35)
        materialized = self.source / "stage2b_long_cold_seed1" / "materialized"
        for index in range(7):
            materialized = materialized / (("%02d-" % index) + long_part)
        # Path.mkdir cannot create this tree on Windows: it exceeds the legacy
        # MAX_PATH limit, so the extended-length form is required.
        os.makedirs(_fs_path(materialized), exist_ok=True)
        payload = b"deep\n"
        with open(_fs_path(materialized / "deep.txt"), "wb") as handle:
            handle.write(payload)
        self._write_run("stage2b_long_cold_seed1", {"dummy.json": b"x"})
        long_root = self.source / "stage2b_long_cold_seed1" / "materialized"
        # Re-point the run at the deep tree by declaring the deep file too.
        deep_relative = str((materialized / "deep.txt").relative_to(long_root)).replace("\\", "/")
        compact_file = self.source / "stage2b_long_cold_seed1" / "compact_manifest.json"
        compact = json.loads(compact_file.read_text(encoding="utf-8"))
        compact["entries"].append(
            {"path": deep_relative, "size_bytes": len(payload), "sha256": _sha(payload)}
        )
        compact_file.write_text(json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8")
        compact_receipt_file = self.source / "stage2b_long_cold_seed1" / "compact_receipt.json"
        compact_receipt = json.loads(compact_receipt_file.read_text(encoding="utf-8"))
        compact_receipt["manifest_sha256"] = sha256_file(compact_file)
        compact_receipt["entries"] = len(compact["entries"])
        compact_receipt_file.write_text(
            json.dumps(compact_receipt, indent=2, sort_keys=True), encoding="utf-8"
        )

        manifest = {
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.destination),
            "entries": [
                {
                    "run_id": "stage2b_long_cold_seed1",
                    "source": "stage2b_long_cold_seed1",
                    "destination": "experiments/stage2b/closed_loop/long/cold/seed1/"
                                   "stage2b_long_cold_seed1",
                }
            ],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        plan = build_plan(self.manifest_path)
        receipt = execute_plan(plan, remove_source=True)
        self.receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        report = self._verify()
        self.assertEqual(report["status"], "verified", report["findings"])
        self.assertEqual(report["oracle_entry_count"], 2)


if __name__ == "__main__":
    unittest.main()
