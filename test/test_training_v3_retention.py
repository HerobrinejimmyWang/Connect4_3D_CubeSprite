from __future__ import annotations

import unittest

from training.v3.retention import (
    ArchiveReceipt,
    ReceiptEntry,
    RetentionArtifact,
    RetentionPolicy,
    estimate_disk_usage,
    plan_retention,
)


def digest(character: str) -> str:
    return character * 64


def artifact(
    path: str,
    kind: str,
    sequence: int,
    *,
    checksum: str,
    start: int | None = None,
    end: int | None = None,
    size: int = 100,
    prunable: bool = True,
    pinned: bool = False,
) -> RetentionArtifact:
    return RetentionArtifact(
        path=path,
        kind=kind,
        size_bytes=size,
        checksum_sha256=checksum,
        sequence=sequence,
        prunable=prunable,
        pinned=pinned,
        position_start=start,
        position_end=end,
    )


def receipt_for(*artifacts: RetentionArtifact, verified: bool = True) -> ArchiveReceipt:
    return ArchiveReceipt(
        receipt_id="local-archive-001",
        archive_manifest_sha256=digest("f"),
        verified=verified,
        entries=tuple(
            ReceiptEntry(item.path, item.size_bytes, item.checksum_sha256)
            for item in artifacts
        ),
    )


class V3RetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = [
            artifact(
                f"replay/raw/shard-{index}.npz",
                "raw_replay",
                index,
                checksum=digest(str(index + 1)),
                start=index * 100,
                end=(index + 1) * 100,
            )
            for index in range(5)
        ]
        self.checkpoints = [
            artifact(
                f"checkpoints/g{index}.pt",
                "checkpoint",
                index,
                checksum=digest(chr(ord("a") + index)),
            )
            for index in range(3)
        ]
        self.policy = RetentionPolicy(
            active_window_margin=1.25,
            keep_recent_by_kind=(("raw_replay", 1), ("checkpoint", 2)),
            soft_used_fraction=0.80,
            hard_free_bytes=200,
        )

    def test_plan_requires_receipt_and_protects_margin_and_recent_artifacts(self) -> None:
        rows = self.raw + self.checkpoints
        archive_receipt = receipt_for(
            self.raw[0],
            self.raw[1],
            self.raw[3],
            self.raw[4],
            self.checkpoints[0],
        )
        plan = plan_retention(
            reversed(rows),
            self.policy,
            active_window_start=400,
            active_window_end=500,
            receipts=(archive_receipt,),
            capacity_bytes=2000,
            other_used_bytes=1000,
        )

        self.assertEqual(plan.expanded_window_start, 375)
        self.assertEqual(
            plan.eligible_paths,
            (
                "checkpoints/g0.pt",
                "replay/raw/shard-0.npz",
                "replay/raw/shard-1.npz",
            ),
        )
        decisions = {decision.artifact.path: decision for decision in plan.decisions}
        self.assertIn(
            "verified_archive_receipt_missing_or_mismatched",
            decisions[self.raw[2].path].reasons,
        )
        self.assertIn("active_window_margin", decisions[self.raw[3].path].reasons)
        self.assertIn("recent:raw_replay", decisions[self.raw[4].path].reasons)
        self.assertEqual(plan.eligible_bytes, 300)
        self.assertTrue(plan.soft_limit_exceeded)
        self.assertTrue(plan.hard_reserve_breached)
        self.assertEqual(plan.projected.total_used_bytes, plan.current.total_used_bytes - 300)

    def test_input_order_does_not_change_plan(self) -> None:
        rows = self.raw + self.checkpoints
        archive_receipt = receipt_for(*rows)
        forward = plan_retention(
            rows,
            self.policy,
            active_window_start=400,
            active_window_end=500,
            receipts=(archive_receipt,),
        )
        backward = plan_retention(
            reversed(rows),
            self.policy,
            active_window_start=400,
            active_window_end=500,
            receipts=(archive_receipt,),
        )
        self.assertEqual(forward.eligible_paths, backward.eligible_paths)
        self.assertEqual(forward.kept_paths, backward.kept_paths)

    def test_unverified_or_mismatched_receipt_never_allows_prune(self) -> None:
        target = self.raw[0]
        unverified = receipt_for(target, verified=False)
        mismatched = ArchiveReceipt(
            receipt_id="mismatch",
            archive_manifest_sha256=digest("e"),
            verified=True,
            entries=(ReceiptEntry(target.path, target.size_bytes + 1, target.checksum_sha256),),
        )
        for archive_receipt in (unverified, mismatched):
            with self.subTest(receipt=archive_receipt.receipt_id):
                plan = plan_retention(
                    (target,),
                    RetentionPolicy(keep_recent_by_kind=(), hard_free_bytes=0),
                    active_window_start=500,
                    active_window_end=500,
                    receipts=(archive_receipt,),
                )
                self.assertEqual(plan.eligible_paths, ())

    def test_pinned_or_not_explicitly_prunable_is_kept(self) -> None:
        pinned = artifact(
            "candidates/unresolved.pt",
            "candidate",
            1,
            checksum=digest("c"),
            pinned=True,
        )
        manifest = artifact(
            "manifests/run.json",
            "manifest",
            1,
            checksum=digest("d"),
            prunable=False,
        )
        plan = plan_retention(
            (pinned, manifest),
            RetentionPolicy(keep_recent_by_kind=(), hard_free_bytes=0),
            active_window_start=0,
            active_window_end=0,
            receipts=(receipt_for(pinned, manifest),),
        )
        self.assertEqual(plan.eligible_paths, ())
        decisions = {decision.artifact.path: decision for decision in plan.decisions}
        self.assertIn("pinned", decisions[pinned.path].reasons)
        self.assertIn("not_marked_prunable", decisions[manifest.path].reasons)

    def test_producer_shard_beyond_learner_cursor_is_never_pruned(self) -> None:
        pending = artifact(
            "replay/raw/pending.npz",
            "raw_replay",
            99,
            checksum=digest("9"),
            start=500,
            end=600,
        )
        plan = plan_retention(
            (pending,),
            RetentionPolicy(keep_recent_by_kind=(), hard_free_bytes=0),
            active_window_start=300,
            active_window_end=500,
            receipts=(receipt_for(pending),),
        )
        self.assertEqual(plan.eligible_paths, ())
        self.assertIn("beyond_active_cursor", plan.decisions[0].reasons)

    def test_disk_estimate_groups_kinds_without_reading_filesystem(self) -> None:
        estimate = estimate_disk_usage(
            self.raw[:2] + self.checkpoints[:1],
            capacity_bytes=1000,
            other_used_bytes=50,
        )
        self.assertEqual(estimate.artifact_bytes, 300)
        self.assertEqual(estimate.by_kind, {"checkpoint": 100, "raw_replay": 200})
        self.assertEqual(estimate.total_used_bytes, 350)
        self.assertEqual(estimate.free_bytes, 650)
        self.assertAlmostEqual(estimate.used_fraction or 0.0, 0.35)


if __name__ == "__main__":
    unittest.main()
