from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.v3.formal_journal import CoordinatorLock, GenerationJournal, reconcile_generation_drafts
from training.v3.layout import RunLayout


HASH = "a" * 64


class CoordinatorLockTests(unittest.TestCase):
    def test_lock_is_exclusive_and_owner_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = RunLayout.from_root(directory).create()
            first = CoordinatorLock(layout.coordinator_lock, run_id="run-a").acquire()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                CoordinatorLock(layout.coordinator_lock, run_id="run-a").acquire()
            first.release()
            with CoordinatorLock(layout.coordinator_lock, run_id="run-a"):
                self.assertTrue(layout.coordinator_lock.is_file())
            self.assertFalse(layout.coordinator_lock.exists())

    def test_changed_lock_is_not_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = RunLayout.from_root(directory).create()
            lock = CoordinatorLock(layout.coordinator_lock, run_id="run-a").acquire()
            payload = json.loads(layout.coordinator_lock.read_text(encoding="utf-8"))
            payload["nonce"] = "changed"
            layout.coordinator_lock.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                lock.release()
            self.assertTrue(layout.coordinator_lock.exists())


class GenerationJournalTests(unittest.TestCase):
    def test_staged_commit_can_be_published_after_ready_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = RunLayout.from_root(directory).create()
            journal = GenerationJournal.begin(
                layout, run_id="run-a", generation=2, config_hash=HASH
            )
            artifact = layout.checkpoints / "candidate.pt"
            artifact.write_bytes(b"checkpoint")
            journal.record_artifact(artifact, kind="checkpoint")
            journal.stage_commit(
                {
                    "schema_version": 1,
                    "run_id": "run-a",
                    "generation": 2,
                    "config_hash": HASH,
                }
            )
            self.assertEqual(reconcile_generation_drafts(layout)[0]["status"], "resume_precommit")
            restored = GenerationJournal.load(layout, 2)
            commit = restored.publish_commit()
            self.assertTrue(commit.is_file())
            pointer = json.loads(
                (layout.manifests / "latest_generation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pointer["generation"], 2)
            self.assertEqual(reconcile_generation_drafts(layout)[0]["status"], "committed")

    def test_ready_draft_reconciles_without_silent_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = RunLayout.from_root(directory).create()
            journal = GenerationJournal.begin(
                layout, run_id="run-a", generation=0, config_hash=HASH
            )
            artifact = layout.checkpoints / "candidate.pt"
            artifact.write_bytes(b"checkpoint")
            journal.record_artifact(artifact, kind="checkpoint")
            journal.mark_artifacts_ready()
            result = reconcile_generation_drafts(layout)
            self.assertEqual(result[0]["status"], "resume_precommit")
            with self.assertRaises(FileExistsError):
                GenerationJournal.begin(layout, run_id="run-a", generation=0, config_hash=HASH)

            (layout.generation_commits / "g000000.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(reconcile_generation_drafts(layout)[0]["status"], "committed")

    def test_missing_or_changed_artifact_blocks_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = RunLayout.from_root(directory).create()
            journal = GenerationJournal.begin(
                layout, run_id="run-a", generation=3, config_hash=HASH
            )
            artifact = layout.checkpoints / "candidate.pt"
            artifact.write_bytes(b"checkpoint")
            journal.record_artifact(artifact, kind="checkpoint")
            journal.mark_artifacts_ready()
            artifact.write_bytes(b"changed")
            result = reconcile_generation_drafts(layout)[0]
            self.assertEqual(result["status"], "blocked_partial_artifacts")
            self.assertEqual(result["failures"][0]["reason"], "checksum_mismatch")


if __name__ == "__main__":
    unittest.main()
