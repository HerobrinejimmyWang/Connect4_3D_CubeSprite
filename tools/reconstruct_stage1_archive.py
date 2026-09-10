"""Reconstruct a clear per-run Stage 1 tree from verified local bundles.

This is intentionally additive: it never edits the old materialized tree,
bundles, receipts, or any source run. Repeated paths are versioned under the
new tree's _reconstruction_conflicts directory, while the newest bundle wins
the canonical reconstructed path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import tarfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> str:
    normalized = posixpath.normpath(str(value).replace("\\", "/"))
    if normalized in ("", ".") or normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"unsafe archive member path: {value!r}")
    return normalized


def load_catalog(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def output_run_name(run_id: str) -> str:
    if len(run_id) <= 28:
        return run_id
    suffix = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    lowered = run_id.lower()
    if "b10c256_g487_mixed" in lowered:
        label = "b10_mixed"
    elif "b10c256_relative" in lowered:
        label = "b10_relative"
    elif "b8c192_role40" in lowered:
        label = "b8_role40"
    elif "b8c192_role30" in lowered:
        label = "b8_role30"
    elif "b8c192_v2" in lowered:
        label = "b8_v2"
    elif "b6c128_g343" in lowered:
        label = "b6_g343"
    elif "b6c128_dynamic" in lowered:
        label = "b6_dynamic"
    elif "scale_screen_b6" in lowered:
        label = "b6_scale"
    else:
        label = run_id.split("_", 1)[0]
    return f"{label}-{suffix}"


def extract_verified_bundle(
    bundle: dict[str, Any],
    output_root: Path,
    source_by_path: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = Path(bundle["local_manifest_path"])
    archive_path = Path(bundle["local_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest["run_id"])
    run_dir_name = output_run_name(run_id)
    run_root = output_root / ("p6_reconstructed" if run_id.startswith("p6_") else "runs_reconstructed") / run_dir_name
    expected = {
        safe_relative(row["path"]): {
            "size_bytes": int(row["size_bytes"]),
            "checksum_sha256": str(row["checksum_sha256"]),
        }
        for row in manifest.get("entries", [])
    }
    seen: set[str] = set()
    result = {
        "bundle_id": bundle["bundle_id"],
        "run_id": run_id,
        "reconstructed_run_dir": str(run_root),
        "archive": str(archive_path),
        "extracted_files": 0,
        "reused_identical_files": 0,
        "version_conflicts": 0,
        "unexpected_members": [],
        "missing_members": [],
        "validation_errors": [],
    }

    with tarfile.open(archive_path, mode="r") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            try:
                relative = safe_relative(member.name)
            except ValueError as exc:
                result["validation_errors"].append(str(exc))
                continue
            if relative not in expected:
                result["unexpected_members"].append(relative)
                continue
            if relative in seen:
                result["validation_errors"].append(f"duplicate tar member: {relative}")
                continue
            seen.add(relative)
            if not member.isfile():
                result["validation_errors"].append(f"non-regular tar member: {relative}")
                continue
            expected_row = expected[relative]
            target = run_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                actual_size = target.stat().st_size
                actual_sha = sha256_file(target)
                if actual_size == expected_row["size_bytes"] and actual_sha == expected_row["checksum_sha256"]:
                    result["reused_identical_files"] += 1
                    source_by_path[target] = {"bundle_id": bundle["bundle_id"], "run_id": run_id}
                    continue
                result["version_conflicts"] += 1
                prior = source_by_path.get(target, {"bundle_id": "prior_reconstruction", "run_id": run_id})
                short_bundle_id = hashlib.sha256(str(prior["bundle_id"]).encode("utf-8")).hexdigest()[:10]
                short_bundle_id += "-" + str(prior["bundle_id"]).rsplit("-", 1)[-1]
                prior_path = run_root / "_reconstruction_conflicts" / short_bundle_id / relative
                prior_path.parent.mkdir(parents=True, exist_ok=True)
                if not prior_path.exists():
                    target.replace(prior_path)
                else:
                    result["validation_errors"].append(f"prior conflict already exists: {prior_path}")
            stream = archive.extractfile(member)
            if stream is None:
                result["validation_errors"].append(f"cannot read tar member: {relative}")
                continue
            partial = target.with_name(target.name + ".partial")
            digest = hashlib.sha256()
            size = 0
            with partial.open("wb") as output:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != expected_row["size_bytes"] or digest.hexdigest() != expected_row["checksum_sha256"]:
                result["validation_errors"].append(f"entry checksum mismatch: {relative}")
                partial.unlink(missing_ok=True)
                continue
            partial.replace(target)
            source_by_path[target] = {"bundle_id": bundle["bundle_id"], "run_id": run_id}
            result["extracted_files"] += 1

    missing = sorted(set(expected) - seen)
    result["missing_members"] = missing
    return result


def reconstruct(local_root: Path, catalog_path: Path, report_dir: Path) -> dict[str, Any]:
    rows = [
        row for row in load_catalog(catalog_path)
        if row.get("record_type") == "bundle" and row.get("content_status") == "present_verified"
    ]
    rows.sort(key=lambda row: (str(row.get("run_id") or ""), str(row["bundle_id"])))
    source_by_path: dict[Path, dict[str, Any]] = {}
    bundle_results = []
    for row in rows:
        bundle_results.append(extract_verified_bundle(row, local_root, source_by_path))

    report = {
        "format": "connect4-v3-stage1-reconstruction-report",
        "destructive_actions_performed": [],
        "source_materialized_preserved": True,
        "source_bundles_preserved": True,
        "verified_bundle_count": len(rows),
        "run_directory_map": {
            str(row.get("run_id")): output_run_name(str(row.get("run_id")))
            for row in rows
        },
        "bundle_results": bundle_results,
        "summary": {
            "extracted_files": sum(int(r["extracted_files"]) for r in bundle_results),
            "reused_identical_files": sum(int(r["reused_identical_files"]) for r in bundle_results),
            "version_conflicts": sum(int(r["version_conflicts"]) for r in bundle_results),
            "unexpected_members": sum(len(r["unexpected_members"]) for r in bundle_results),
            "missing_members": sum(len(r["missing_members"]) for r in bundle_results),
            "validation_errors": sum(len(r["validation_errors"]) for r in bundle_results),
        },
        "next_step": "Review this report and the reconstructed trees before any duplicate cleanup.",
    }
    write_json(report_dir / "reconstruction_report.json", report)
    write_json(report_dir / "reconstruction_policy.json", {
        "canonical_roots": ["runs_reconstructed/<run_id>", "p6_reconstructed/<run_id>"],
        "source_roots_untouched": ["materialized", "bundles", "receipts"],
        "newer_bundle_wins_on_repeated_path": True,
        "older_versions_retained_under": "_reconstruction_conflicts",
        "cleanup_allowed": False,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("training/runs/stage1/archive"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    local_root = args.local_root.resolve()
    catalog_path = (args.catalog or local_root / "catalog" / "bundle_catalog.jsonl").resolve()
    report_dir = (args.report_dir or local_root / "reports").resolve()
    report = reconstruct(local_root, catalog_path, report_dir)
    print(json.dumps({
        "local_root": str(local_root),
        "verified_bundle_count": report["verified_bundle_count"],
        "summary": report["summary"],
        "reports": str(report_dir),
        "note": "Original materialized and bundle files were not modified.",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
