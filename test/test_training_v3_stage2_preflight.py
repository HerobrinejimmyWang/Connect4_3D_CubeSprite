from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.v3.stage2.preflight import preflight_stage1_sources


class Stage2PreflightTests(unittest.TestCase):
    def _run(self, root: Path, name: str, *, receipt_only: bool) -> Path:
        run = root / name
        (run / "manifests/generations").mkdir(parents=True)
        (run / "metrics").mkdir()
        replay = run / "replay/raw/g000000_s0000.npz"
        replay.parent.mkdir(parents=True)
        payload = b"replay"
        import hashlib

        checksum = hashlib.sha256(payload).hexdigest()
        if not receipt_only:
            replay.write_bytes(payload)
        commit = {
            "generation": 0,
            "run_id": name,
            "replay_cumulative_positions": 10,
            "replay_shards": [{"path": "replay/raw/g000000_s0000.npz", "checksum_sha256": checksum}],
        }
        (run / "manifests/generations/g000000.json").write_text(json.dumps(commit), encoding="utf-8")
        (run / "manifests/latest_generation.json").write_text(
            json.dumps({"generation": 0}), encoding="utf-8"
        )
        (run / "metrics/metrics.jsonl").write_text(
            json.dumps({"stage": "selfplay", "generation": 0, "health": {}}) + "\n",
            encoding="utf-8",
        )
        if receipt_only:
            receipts = run / "archive_receipts"
            receipts.mkdir()
            (receipts / "one.receipt.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "path": "replay/raw/g000000_s0000.npz",
                                "checksum_sha256": checksum,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
        return run

    def test_receipt_coverage_is_recoverable_but_not_materialized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-preflight-") as temporary:
            root = Path(temporary)
            standard = self._run(root, "standard", receipt_only=True)
            mixed = self._run(root, "mixed", receipt_only=False)
            result = preflight_stage1_sources(
                standard_run_dir=standard,
                mixed_run_dir=mixed,
                standard_required_positions=10,
                mixed_required_positions=10,
            )
            self.assertFalse(result["ready_for_remote_freeze"])
            self.assertTrue(result["sources"]["standard"]["provenance_recoverable"])
            self.assertEqual(result["sources"]["standard"]["receipt_backed_replay_shards"], 1)
            self.assertTrue(result["sources"]["mixed"]["remote_materialization_complete"])


if __name__ == "__main__":
    unittest.main()
