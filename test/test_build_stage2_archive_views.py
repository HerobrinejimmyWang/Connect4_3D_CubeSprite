"""Focused tests for the Stage 2 archive evidence-view builder.

The properties worth protecting are the two-layer path contract (a view entry
must expose both where an artifact is and where history said it was) and the
qualification boundary (no view entry may claim Flash qualification that was
never measured).
"""

import json
import os
import unittest
from pathlib import Path

from stage2_scratch import make_scratch_dir, remove_scratch_dir
from tools.build_stage2_archive_views import (
    SUBTREE_MOVES,
    apply_moves,
    artifact_entry,
    build_all,
    build_bal_view,
    build_fla_view,
    build_pro_view,
    historical_of,
    main,
)


class PathMoveTests(unittest.TestCase):
    def test_rewrites_only_a_path_segment_aligned_prefix(self):
        self.assertEqual(
            apply_moves("round3_balance/bal1/results_matrix.json"),
            "experiments/stage2r3/round3_balance/bal1/results_matrix.json",
        )
        self.assertEqual(
            apply_moves("round3_balance"), "experiments/stage2r3/round3_balance"
        )
        # A longer sibling name must not be captured by the prefix rule.
        self.assertEqual(
            apply_moves("round3_balance_extra/x.json"), "round3_balance_extra/x.json"
        )
        self.assertEqual(
            apply_moves("stage2_shared/materialized/round1/s.json"),
            "stage2_shared/materialized/round1/s.json",
        )

    def test_handles_native_separators_and_redundant_slashes(self):
        self.assertEqual(
            apply_moves("\\round3_balance\\bal2\\summary.json"),
            "experiments/stage2r3/round3_balance/bal2/summary.json",
        )
        self.assertEqual(apply_moves("/round3_balance/bal2/"), "experiments/stage2r3/round3_balance/bal2")

    def test_historical_of_inverts_apply_moves(self):
        for original in ("round3_balance/bal3/summary.json", "round3_balance"):
            self.assertEqual(historical_of(apply_moves(original)), original)
        self.assertIsNone(historical_of("stage2_shared/materialized/round1/s.json"))


class ArtifactEntryTests(unittest.TestCase):
    def setUp(self):
        self.root = make_scratch_dir("stage2-views-")
        self.repo = self.root
        self.archive = self.root / "training/runs/stage2/archive"
        self.archive.mkdir(parents=True)

    def tearDown(self):
        remove_scratch_dir(self.root)

    def test_reports_both_path_layers_when_already_moved(self):
        target = self.archive / "experiments/stage2r3/round3_balance/bal1/m.json"
        target.parent.mkdir(parents=True)
        target.write_text("matrix\n", encoding="utf-8")

        entry = artifact_entry(self.archive, self.repo, "round3_balance/bal1/m.json")
        self.assertEqual(entry["move_status"], "materialized_at_planned_path")
        self.assertTrue(entry["exists"])
        self.assertEqual(
            entry["materialized_path"],
            "training/runs/stage2/archive/experiments/stage2r3/round3_balance/bal1/m.json",
        )
        self.assertEqual(
            entry["historical_origin_path"],
            "training/runs/stage2/archive/round3_balance/bal1/m.json",
        )
        self.assertEqual(
            entry["planned_materialized_path"], entry["materialized_path"]
        )

    def test_labels_a_not_yet_moved_artifact_instead_of_losing_it(self):
        historical = self.archive / "round3_balance/bal1/m.json"
        historical.parent.mkdir(parents=True)
        historical.write_text("matrix\n", encoding="utf-8")

        entry = artifact_entry(self.archive, self.repo, "round3_balance/bal1/m.json")
        self.assertEqual(entry["move_status"], "pending_move")
        self.assertTrue(entry["exists"])
        self.assertEqual(
            entry["materialized_path"],
            "training/runs/stage2/archive/round3_balance/bal1/m.json",
        )
        self.assertIn("planned_materialized_path", entry)

    def test_reports_missing_when_neither_location_holds_the_file(self):
        entry = artifact_entry(self.archive, self.repo, "round3_balance/bal9/gone.json")
        self.assertEqual(entry["move_status"], "missing")
        self.assertFalse(entry["exists"])
        self.assertNotIn("sha256", entry)

    def test_artifacts_outside_the_moves_are_not_marked_as_relocated(self):
        target = self.archive / "stage2_shared/materialized/round1/s.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}\n", encoding="utf-8")
        entry = artifact_entry(
            self.archive, self.repo, "stage2_shared/materialized/round1/s.json"
        )
        self.assertEqual(entry["move_status"], "not_moved")
        self.assertNotIn("planned_materialized_path", entry)


class _ViewFixture(unittest.TestCase):
    def setUp(self):
        self.root = make_scratch_dir("stage2-views-")
        self.repo = self.root
        self.archive = self.root / "training/runs/stage2/archive"
        (self.archive / "flash").mkdir(parents=True)

    def tearDown(self):
        remove_scratch_dir(self.root)

    def _registry(self, owned, external):
        (self.archive / "flash/external_materials.json").write_text(
            json.dumps(
                {
                    "schema": "connect4-v3-stage2-fla-external-materials-v1",
                    "owned_materials": owned,
                    "external_materials": external,
                }
            ),
            encoding="utf-8",
        )

    def _file(self, archive_rel, text="x\n"):
        path = self.archive / Path(archive_rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class FlaViewTests(_ViewFixture):
    def test_current_registry_path_keeps_historical_origin(self):
        import hashlib

        current = self._file("experiments/stage2r3/round3_balance/bal1/results_matrix.json")
        digest = hashlib.sha256(current.read_bytes()).hexdigest()
        self._registry([], [{
            "path": "../experiments/stage2r3/round3_balance/bal1/results_matrix.json",
            "historical_origin_path": "../round3_balance/bal1/results_matrix.json",
            "sha256": digest,
            "origin": "stage2_r3_bal1",
            "fla_class": "extended",
        }])
        view = build_fla_view(self.archive, self.repo)
        self.assertEqual(view["integrity"]["missing_materialized_paths"], [])
        self.assertEqual(view["integrity"]["recorded_sha256_mismatches"], [])
        entry = view["entries"][0]
        self.assertIn("experiments/stage2r3/round3_balance", entry["materialized_path"])
        self.assertIn("archive/round3_balance", entry["historical_origin_path"])

    def test_never_claims_flash_qualification(self):
        self._file("flash/fla1_stage2a_round1/cpu_response_latency/summary.json", "{}\n")
        self._file("stage2_shared/materialized/round1/summary_v2_30.json", "{}\n")
        self._file("round3_balance/bal1/results_matrix.json", "{}\n")
        self._file("round3_balance/bal2/results_matrix.json", "{}\n")
        self._registry(
            [{"path": "fla1_stage2a_round1/cpu_response_latency/summary.json",
              "sha256": "0" * 64, "role": "latency"}],
            [
                {"path": "../stage2_shared/materialized/round1/summary_v2_30.json",
                 "sha256": "0" * 64, "origin": "stage2a_round1", "fla_class": "core"},
                {"path": "../round3_balance/bal1/results_matrix.json",
                 "sha256": "0" * 64, "origin": "stage2_r3_bal1", "fla_class": "extended"},
                {"path": "../round3_balance/bal2/results_matrix.json",
                 "sha256": "0" * 64, "origin": "stage2_r3_bal2", "fla_class": "boundary"},
            ],
        )

        view = build_fla_view(self.archive, self.repo)
        statuses = {row["qualification_status"] for row in view["entries"]}
        self.assertNotIn("flash_qualified", statuses)
        self.assertNotIn("flash_pareto", statuses)
        self.assertEqual(
            statuses, {"historical_evidence", "boundary_only", "latency_pending"}
        )
        # BAL-origin entries stay BAL-origin and are not copied into the view.
        bal1 = next(row for row in view["entries"] if "bal1" in row["historical_origin_path"])
        self.assertEqual(bal1["origin"], "stage2_r3_bal1")
        self.assertEqual(bal1["ownership"], "external_authoritative_elsewhere")
        self.assertIn("planned_materialized_path", bal1)

    def test_reports_a_recorded_hash_that_no_longer_matches_the_file(self):
        self._file("stage2_shared/materialized/round1/s.json", "actual\n")
        self._registry(
            [],
            [{"path": "../stage2_shared/materialized/round1/s.json",
              "sha256": "f" * 64, "fla_class": "core"}],
        )
        view = build_fla_view(self.archive, self.repo)
        self.assertEqual(len(view["integrity"]["recorded_sha256_mismatches"]), 1)
        self.assertEqual(view["integrity"]["missing_materialized_paths"], [])

    def test_verifies_against_a_correctly_recorded_hash(self):
        import hashlib

        payload = b"ok\n"
        path = self._file("stage2_shared/materialized/round1/s.json")
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self._registry(
            [],
            [{"path": "../stage2_shared/materialized/round1/s.json",
              "sha256": digest.upper(), "fla_class": "core"}],
        )
        view = build_fla_view(self.archive, self.repo)
        self.assertEqual(view["integrity"]["recorded_sha256_mismatches"], [])
        self.assertEqual(view["integrity"]["missing_materialized_paths"], [])

    def test_reports_a_missing_materialized_path(self):
        self._registry(
            [],
            [{"path": "../stage2_shared/materialized/round1/gone.json",
              "sha256": "a" * 64, "fla_class": "core"}],
        )
        view = build_fla_view(self.archive, self.repo)
        self.assertEqual(len(view["integrity"]["missing_materialized_paths"]), 1)


class BalViewTests(_ViewFixture):
    def test_groups_bal_members_and_marks_a_pending_move(self):
        self._file("round3_balance/bal1/results_matrix.json")
        self._file("round3_balance/bal2/results_matrix.json")
        self._file("round3_balance/bal2_3m_anchored_elo/summary.json")
        self._file("round3_balance/bal3_cpu_latency/summary.json")

        view = build_bal_view(self.archive, self.repo)
        by_lineage = {row["lineage"]: row for row in view["entries"]}
        self.assertEqual(set(by_lineage), {"BAL-1", "BAL-2", "BAL-3"})
        self.assertEqual(
            by_lineage["BAL-2"]["members"],
            ["bal2", "bal2_3m_anchored_elo"],
        )
        self.assertEqual(by_lineage["BAL-1"]["move_status"], "pending_atomic_move")
        self.assertEqual(view["status"], "populated_with_deferred_groups")
        self.assertEqual(by_lineage["BAL-1"]["file_count"], 1)

    def test_enumerates_groups_after_the_historical_root_is_gone(self):
        """The real post-move state: only the stage2r3 location exists.

        Reading only the historical path silently drops every BAL group.
        """

        self._file("experiments/stage2r3/round3_balance/bal1/results_matrix.json")
        self._file("experiments/stage2r3/round3_balance/bal2_3m_anchored_elo/summary.json")
        self._file("experiments/stage2r3/round3_balance/bal3/summary.json")
        self.assertFalse((self.archive / "round3_balance").exists())

        view = build_bal_view(self.archive, self.repo)
        by_lineage = {row["lineage"]: row for row in view["entries"]}
        self.assertEqual(set(by_lineage), {"BAL-1", "BAL-2", "BAL-3"})
        self.assertEqual(
            by_lineage["BAL-2"]["members"], ["bal2_3m_anchored_elo"]
        )
        for row in by_lineage.values():
            self.assertEqual(row["move_status"], "moved_atomically_under_stage2r3")
            self.assertEqual(
                row["materialized_path"],
                "training/runs/stage2/archive/experiments/stage2r3/round3_balance",
            )
            self.assertEqual(
                row["historical_origin_path"],
                "training/runs/stage2/archive/round3_balance",
            )
        self.assertEqual(view["status"], "populated")

    def test_an_empty_moved_root_does_not_count_as_a_completed_move(self):
        """A created-but-empty target means the move has not happened yet."""

        self._file("round3_balance/bal1/results_matrix.json")
        (self.archive / "experiments/stage2r3/round3_balance").mkdir(parents=True)
        view = build_bal_view(self.archive, self.repo)
        self.assertEqual(view["entries"][0]["move_status"], "pending_atomic_move")
        self.assertEqual(view["status"], "populated_with_deferred_groups")
        self.assertNotIn("location_conflict", view)

    def test_flags_both_locations_populated_as_a_conflict(self):
        self._file("round3_balance/bal1/results_matrix.json")
        self._file("experiments/stage2r3/round3_balance/bal2/results_matrix.json")
        view = build_bal_view(self.archive, self.repo)
        self.assertEqual(view["status"], "conflicting_locations")
        self.assertIn("location_conflict", view)
        # Enumerated from the historical location and labelled as such.
        self.assertEqual({row["lineage"] for row in view["entries"]}, {"BAL-1"})

    def test_registers_bal5_as_deferred_with_its_blockers(self):
        self._file("bal5/r1/summary.json")
        view = build_bal_view(self.archive, self.repo)
        bal5 = next(row for row in view["entries"] if row["lineage"] == "BAL-5")
        self.assertEqual(bal5["move_status"], "deferred_pending_current_state_review")
        self.assertTrue(bal5["move_blockers"])
        self.assertIn("experiments/stage2r3/bal5", bal5["planned_materialized_path"])

    def test_empty_archive_yields_no_bal_entries(self):
        view = build_bal_view(self.archive, self.repo)
        self.assertEqual(view["entries"], [])

    @unittest.skipUnless(os.name == "nt", "extended Windows paths are Windows-only")
    def test_counts_files_beyond_legacy_max_path(self):
        """A naive os.walk silently truncates BAL-5-style deep trees."""

        from tools.build_stage2_archive_views import _directory_totals, _fs_path

        deep = self.archive / "bal5"
        for index in range(7):
            deep = deep / (("%02d-" % index) + "long-directory-" + ("z" * 35))
        os.makedirs(_fs_path(deep), exist_ok=True)
        for name in ("a.json", "b.json"):
            with open(_fs_path(deep / name), "wb") as handle:
                handle.write(b"deep\n")
        files, total = _directory_totals(self.archive / "bal5")
        self.assertEqual(files, 2)
        self.assertEqual(total, 10)
        # The under-counting behaviour this guards against: a plain walk of the
        # same tree returns nothing at all on Windows.
        plain = sum(len(names) for _c, _d, names in os.walk(str(self.archive / "bal5")))
        self.assertLess(plain, 2)


class ProViewTests(_ViewFixture):
    def test_pro_stays_planned_and_empty(self):
        view = build_pro_view()
        self.assertEqual(view["status"], "planned_no_materials")
        self.assertEqual(view["entries"], [])


class BuildAllTests(_ViewFixture):
    def test_preview_writes_nothing(self):
        self._file("round3_balance/bal1/results_matrix.json")
        self._registry([], [])
        exit_code = main(["--repo-root", str(self.repo)])
        self.assertEqual(exit_code, 0)
        for name in ("fla", "bal", "pro"):
            self.assertFalse((self.archive / "views" / name / "index.json").exists())

    def test_execute_writes_all_three_view_indexes(self):
        self._file("round3_balance/bal1/results_matrix.json")
        self._registry([], [])
        exit_code = main(["--repo-root", str(self.repo), "--execute"])
        self.assertEqual(exit_code, 0)
        written = {
            name: json.loads(
                (self.archive / "views" / name / "index.json").read_text(encoding="utf-8")
            )
            for name in ("fla", "bal", "pro")
        }
        self.assertEqual(written["fla"]["schema"], "connect4-stage2-fla-view-index-v1")
        self.assertEqual(written["bal"]["schema"], "connect4-stage2-bal-view-index-v1")
        self.assertEqual(written["pro"]["schema"], "connect4-stage2-pro-view-index-v1")
        self.assertEqual(written["pro"]["entries"], [])

    def test_build_all_covers_every_view(self):
        self._registry([], [])
        views = build_all(self.archive, self.repo)
        self.assertEqual(set(views), {"fla", "bal", "pro"})
        self.assertEqual(SUBTREE_MOVES[0][0], "round3_balance")


if __name__ == "__main__":
    unittest.main()
