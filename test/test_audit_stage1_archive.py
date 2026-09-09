import tempfile
import unittest
from pathlib import Path

from tools.audit_stage1_archive import local_inventory, reconcile


class Stage1ArchiveAuditTests(unittest.TestCase):
    def test_reconcile_marks_verified_remote_and_historical_states(self) -> None:
        local = {
            "records": [
                {
                    "source": "local",
                    "bundle_id": "run-a-aaa",
                    "run_id": "run-a",
                    "archive_sha256": "sha-a",
                    "archive_size_bytes": 10,
                    "manifest_sha256": "manifest-a",
                    "archive_path": "local/a.tar",
                    "manifest_path": "local/a.manifest.json",
                    "archive_exists": True,
                    "actual_archive_sha256": "sha-a",
                    "entries_signature": '[{"checksum_sha256":"entry","path":"x","size_bytes":1}]',
                    "receipts": [{
                        "verified": True,
                        "bundle_id": "run-a-aaa",
                        "archive_manifest_sha256": "manifest-a",
                        "archive_sha256": "sha-a",
                        "entries": [{"path": "x", "size_bytes": 1, "checksum_sha256": "entry"}],
                    }],
                }
            ],
            "materialized_only": [],
        }
        remote = {
            "records": [{
                "source": "remote",
                "bundle_id": "run-b-bbb",
                "run_id": "run-b",
                "archive_sha256": "sha-b",
                "archive_size_bytes": 20,
                "manifest_sha256": "manifest-b",
                "archive_path": "/remote/b.tar",
                "manifest_path": "/remote/b.manifest.json",
                "entries_signature": "[]",
            }],
            "receipts": [],
        }
        records = reconcile(local, remote, {"bbb"})
        by_id = {row["bundle_id"]: row for row in records}
        self.assertEqual(by_id["run-a-aaa"]["content_status"], "present_verified")
        self.assertEqual(by_id["run-b-bbb"]["content_status"], "remote_only")
        self.assertTrue(by_id["run-b-bbb"]["historical_image_seen"])

    def test_local_materialized_uncovered_file_is_not_called_a_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "materialized" / "accepted").mkdir(parents=True)
            (root / "materialized" / "accepted" / "unmatched.pt").write_bytes(b"x")
            inventory = local_inventory(root, hash_archives=False)
            self.assertEqual(inventory["materialized_file_count"], 1)
            self.assertEqual(inventory["materialized_only"][0]["relative_path"], "accepted/unmatched.pt")


if __name__ == "__main__":
    unittest.main()
