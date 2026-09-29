import json
import os
import unittest
from pathlib import Path

from stage2_scratch import make_scratch_dir, remove_scratch_dir
from tools.migrate_stage2_archive import (
    MigrationError,
    build_plan,
    cleanup_destination,
    execute_plan,
    _fs_path,
    load_manifest,
    write_plan_artifacts,
)


class Stage2ArchiveMigrationTests(unittest.TestCase):
    def setUp(self):
        self.root = make_scratch_dir()
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        (self.source / "nested").mkdir(parents=True)
        (self.source / "a.txt").write_text("alpha", encoding="utf-8")
        (self.source / "nested" / "b.bin").write_bytes(b"beta\x00")

    def tearDown(self):
        remove_scratch_dir(self.root)

    def manifest(self, entries=None):
        return {
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.destination),
            "entries": entries or [
                {"source": "a.txt", "destination": "one/a.txt"},
                {"source": "nested", "destination": "two/nested"},
            ],
        }

    def test_plan_is_read_only_and_writes_explicit_artifacts(self):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest()), encoding="utf-8")
        plan = build_plan(load_manifest(manifest_path))
        out = self.root / "plan"
        write_plan_artifacts(plan, out)

        self.assertFalse(self.destination.exists())
        self.assertEqual(plan.source_file_count, 2)
        self.assertEqual(plan.source_bytes, 10)
        self.assertTrue((out / "migration_plan.json").is_file())
        self.assertTrue((out / "legacy_path_map.json").is_file())
        self.assertTrue((out / "migration_receipt.json").is_file())
        receipt = json.loads((out / "migration_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "planned")
        self.assertFalse(receipt["source_deleted"])

    def test_rejects_destination_collision_before_copy(self):
        (self.destination / "one").mkdir(parents=True)
        (self.destination / "one" / "a.txt").write_text("old", encoding="utf-8")
        with self.assertRaises(MigrationError):
            build_plan(load_manifest(self.root_manifest_path()))

    def test_rejects_out_of_boundary_and_overlapping_entries(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        bad = self.manifest([{"source": str(outside), "destination": "outside.txt"}])
        with self.assertRaises(MigrationError):
            build_plan(bad)

        overlapping = self.manifest([
            {"source": "nested", "destination": "x"},
            {"source": "nested/b.bin", "destination": "y.bin"},
        ])
        with self.assertRaises(MigrationError):
            build_plan(overlapping)

    def test_finalize_verifies_hashes_and_keeps_source_by_default(self):
        plan = build_plan(self.manifest())
        receipt = execute_plan(plan)
        self.assertEqual(receipt["status"], "completed")
        self.assertFalse(receipt["source_deleted"])
        self.assertTrue((self.source / "a.txt").is_file())
        self.assertEqual((self.destination / "one" / "a.txt").read_text(encoding="utf-8"), "alpha")
        self.assertEqual((self.destination / "two" / "nested" / "b.bin").read_bytes(), b"beta\x00")
        self.assertEqual(receipt["source_file_count"], receipt["destination_file_count"])
        self.assertEqual(receipt["source_bytes"], receipt["destination_bytes"])
        self.assertEqual(
            sorted(item["sha256"] for item in receipt["source_inventory"]),
            sorted(item["sha256"] for item in receipt["destination_inventory"]),
        )

    def test_explicit_remove_source_requires_finalize_and_verification(self):
        plan = build_plan(self.manifest())
        receipt = execute_plan(plan, remove_source=True)
        self.assertTrue(receipt["source_deleted"])
        self.assertTrue(self.source.exists())
        self.assertFalse((self.source / "a.txt").exists())
        self.assertFalse((self.source / "nested").exists())
        self.assertTrue(self.destination.exists())

    def test_windows_alias_resolves_subst_style_root(self):
        aliases = {"X:\\": str(self.source)}
        aliased = self.manifest([
            {"source": "X:\\a.txt", "destination": "one/a.txt"},
        ])
        plan = build_plan(aliased, path_aliases=aliases)
        self.assertEqual(plan.source_file_count, 1)
        self.assertEqual(plan.file_pairs[0].source.name, "a.txt")

    def test_empty_directory_is_created_and_verified(self):
        empty = self.source / "empty"
        empty.mkdir()
        plan = build_plan(self.manifest([
            {"source": "empty", "destination": "empty-copy"},
        ]))
        receipt = execute_plan(plan)
        self.assertTrue((self.destination / "empty-copy").is_dir())
        self.assertEqual(receipt["source_file_count"], 0)
        self.assertEqual(receipt["destination_file_count"], 0)

    def test_reorganizes_entries_within_one_archive_root(self):
        plan = build_plan({
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.source / "experiments"),
            "entries": [{"source": "a.txt", "destination": "a.txt"}],
        })
        receipt = execute_plan(plan)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual((self.source / "experiments" / "a.txt").read_text(encoding="utf-8"), "alpha")
        self.assertTrue((self.source / "a.txt").exists())

    def test_rejects_destination_that_lands_inside_a_source_entry(self):
        manifest = {
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.source / "nested"),
            "entries": [{"source": "nested", "destination": "copy"}],
        }
        with self.assertRaises(MigrationError):
            build_plan(manifest)

    def test_rejects_cross_entry_source_destination_intersection(self):
        (self.source / "experiments" / "old").mkdir(parents=True)
        (self.source / "experiments" / "old" / "model.bin").write_bytes(b"model")
        manifest = {
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.source / "experiments"),
            "entries": [
                {"source": "a.txt", "destination": "old"},
                {"source": "experiments/old", "destination": "new"},
            ],
        }
        with self.assertRaises(MigrationError):
            build_plan(manifest)

    def test_partial_destination_cleanup_is_dry_run_until_explicit(self):
        manifest = self.manifest([
            {"source": "a.txt", "destination": "partial/a.txt"},
        ])
        plan = build_plan(manifest)
        target = self.destination / "partial" / "a.txt"
        target.parent.mkdir(parents=True)
        target.write_text("partial", encoding="utf-8")
        cleanup_plan = build_plan(manifest, allow_destination_collisions=True)
        report = cleanup_destination(cleanup_plan)
        self.assertEqual(report["status"], "planned")
        self.assertTrue(target.exists())
        report = cleanup_destination(cleanup_plan, execute=True)
        self.assertEqual(report["status"], "cleaned")
        self.assertFalse(target.exists())
        self.assertTrue((self.source / "a.txt").exists())

    def test_cleanup_rejects_tampered_out_of_boundary_target(self):
        plan = build_plan(self.manifest([
            {"source": "a.txt", "destination": "partial/a.txt"},
        ]))
        plan.entries.append((self.source / "a.txt", self.root / "outside.txt"))
        with self.assertRaises(MigrationError):
            cleanup_destination(plan, execute=True)

    @unittest.skipUnless(os.name == "nt", "extended Windows paths are Windows-only")
    def test_finalize_supports_more_than_260_characters(self):
        long_part = "long-directory-" + ("x" * 35)
        source = self.source
        for index in range(7):
            source = source / (("%02d-" % index) + long_part)
        source = source / "source-file.txt"
        os.makedirs(_fs_path(source.parent), exist_ok=True)
        with open(_fs_path(source), "w", encoding="utf-8") as handle:
            handle.write("long-path")
        destination = self.destination
        for index in range(7):
            destination = destination / (("%02d-" % index) + long_part)
        manifest = {
            "schema": "stage2-archive-migration-manifest-v1",
            "source_root": str(self.source),
            "destination_root": str(self.destination),
            "entries": [{
                "source": str(source.relative_to(self.source)),
                "destination": str(destination.relative_to(self.destination) / "copy.txt"),
            }],
        }
        plan = build_plan(manifest)
        execute_plan(plan)
        with open(_fs_path(destination / "copy.txt"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "long-path")

    def root_manifest_path(self):
        path = self.root / "collision_manifest.json"
        path.write_text(json.dumps(self.manifest()), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
