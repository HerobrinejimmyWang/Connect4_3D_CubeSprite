from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training.v3.archive import (
    create_archive_bundle,
    execute_prune,
    ingest_archive_receipt,
    plan_prune,
    verify_archive_bundle,
)


class ArchiveWorkflowTests(unittest.TestCase):
    def _run(self, root: Path) -> None:
        (root / "manifests" / "generations").mkdir(parents=True)
        (root / "checkpoints").mkdir()
        (root / "samples" / "g000001").mkdir(parents=True)
        (root / "archive_staging").mkdir()
        (root / "archive_receipts").mkdir()
        (root / "run_manifest.json").write_text(
            json.dumps({"run_id": "run-a", "config_hash": "a" * 64}), encoding="utf-8"
        )
        (root / "resolved_config.json").write_text(
            json.dumps(
                {
                    "runtime": {
                        "storage": {
                            "keep_checkpoints": 1,
                            "keep_accepted": 1,
                            "keep_rejected": 1,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        old = root / "checkpoints" / "g000000-s00000001.pt"
        current = root / "checkpoints" / "g000001-s00000002.pt"
        old.write_bytes(b"old-checkpoint")
        current.write_bytes(b"current-checkpoint")
        audit = root / "samples" / "g000001" / "audit_index.json"
        audit.write_text(json.dumps({"replays": []}), encoding="utf-8")
        commit = {
            "checkpoint": "checkpoints/g000001-s00000002.pt",
            "audit_index": "samples/g000001/audit_index.json",
            "accepted_model_path": None,
            "candidate_path": None,
            "replay_shards": [],
        }
        commit_path = root / "manifests" / "generations" / "g000001.json"
        commit_path.write_text(json.dumps(commit), encoding="utf-8")
        (root / "manifests" / "latest_generation.json").write_text(
            json.dumps({"commit": "manifests/generations/g000001.json"}), encoding="utf-8"
        )

    def test_archive_verify_receipt_and_prune_are_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as local:
            root = Path(directory)
            self._run(root)
            created = create_archive_bundle(root, bundle_target_bytes=1024**2)
            self.assertEqual(created["status"], "created")
            receipt_path = Path(local) / "bundle.receipt.json"
            receipt = verify_archive_bundle(
                created["archive"],
                created["manifest"],
                extract_to=Path(local) / "materialized",
                receipt_path=receipt_path,
            )
            self.assertTrue(receipt["verified"])
            incoming = root / "archive_receipts" / f"incoming-{receipt_path.name}"
            shutil.copy2(receipt_path, incoming)
            ingest_archive_receipt(root, incoming)
            self.assertFalse(incoming.exists())
            plan = plan_prune(root)
            self.assertIn("checkpoints/g000000-s00000001.pt", plan["eligible_paths"])
            self.assertNotIn("checkpoints/g000001-s00000002.pt", plan["eligible_paths"])
            result = execute_prune(root)
            self.assertIn("checkpoints/g000000-s00000001.pt", result["removed"])
            self.assertFalse((root / "checkpoints" / "g000000-s00000001.pt").exists())
            self.assertTrue((root / "checkpoints" / "g000001-s00000002.pt").is_file())


if __name__ == "__main__":
    unittest.main()
