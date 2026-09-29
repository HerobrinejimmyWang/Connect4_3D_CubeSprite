"""Independently verify a completed Stage 2 archive migration.

`tools/migrate_stage2_archive.py` reports its own result.  This tool re-derives
that result from the filesystem and from a source of truth that predates the
migration, so a migration is never accepted on the strength of its own receipt
alone.

Three layers are checked:

1. **Receipt self-consistency** -- status, ``source_deleted``, and the declared
   file/byte totals must agree with each other and with the manifest.
2. **Path correspondence** -- the receipt inventories are keyed by different
   relative prefixes (source root vs destination root).  A migration may copy
   the right *multiset* of bytes into the right *files* while still placing a
   file under the wrong run directory.  Every destination record must therefore
   map back to exactly one source record by stripping the declared entry
   prefix, and the mapping must be a bijection.
3. **Independent re-derivation and a pre-migration oracle** -- every
   destination file is re-hashed from disk.  Each run's ``compact_manifest.json``
   was produced when the archive was created, before any migration, so it is an
   independent record of the intended content; every manifest entry must still
   match the materialized file byte-for-byte.

The oracle layer is what distinguishes "the copy succeeded" from "the archive
still holds the content it claims to hold".  They are different claims.

The tool is read-only.  It never deletes, moves, or repairs anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "stage2-archive-migration-verification-v1"
MIGRATION_RECEIPT_SCHEMA = "stage2-archive-migration-receipt-v1"
MIGRATION_MANIFEST_SCHEMA = "stage2-archive-migration-manifest-v1"


class VerificationFailure(RuntimeError):
    """Raised when a migration cannot be proven correct."""


# --------------------------------------------------------------------------
# filesystem helpers (Windows extended-length aware, symlink rejecting)
# --------------------------------------------------------------------------


def _fs_path(path: Path) -> str:
    """Return a filesystem-safe path, using Win32 extended syntax when needed."""
    raw = os.path.abspath(os.fspath(path))
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw[2:]
    return "\\\\?\\" + raw


def _is_file(path: Path) -> bool:
    return os.path.isfile(_fs_path(path))


def _is_dir(path: Path) -> bool:
    return os.path.isdir(_fs_path(path))


def _is_symlink(path: Path) -> bool:
    return os.path.islink(_fs_path(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_fs_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel_posix(path: Path, root: Path) -> str:
    """Relative path with forward slashes, so comparisons are separator-safe.

    ``os.walk`` is driven with extended-length paths, so its results carry the
    ``\\\\?\\`` prefix; ``Path.relative_to`` rejects that as a subpath of the
    unprefixed root.  Deriving the relative path from the OS-level strings is
    therefore both necessary and version-independent.
    """

    return os.path.relpath(_fs_path(path), _fs_path(root)).replace("\\", "/")


def load_json(path: Path) -> Any:
    try:
        with open(_fs_path(path), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise VerificationFailure(f"cannot read JSON: {path}") from exc


# --------------------------------------------------------------------------
# disk inventory
# --------------------------------------------------------------------------


def _walk_files(entry_root: Path) -> List[Path]:
    """Return every regular file below ``entry_root``, symlinks rejected."""

    root_fs = _fs_path(entry_root)
    if not _is_dir(entry_root):
        raise VerificationFailure(f"destination entry is not a directory: {entry_root}")
    files: List[Path] = []
    for current, dirs, names in os.walk(root_fs, followlinks=False):
        current_path = Path(current)
        for name in list(dirs) + list(names):
            candidate = current_path / name
            if _is_symlink(candidate):
                raise VerificationFailure(f"symlink present in migrated tree: {candidate}")
        for name in names:
            candidate = current_path / name
            if not _is_file(candidate):
                raise VerificationFailure(f"not a regular file: {candidate}")
            files.append(candidate)
    return files


def inventory_entries(
    entries: Iterable[Tuple[Path, str]],
    destination_root: Path,
) -> Tuple[List[Dict[str, Any]], int]:
    """Hash every file under each ``(root, prefix)`` pair into one inventory."""

    records: List[Dict[str, Any]] = []
    total = 0
    for entry_root, prefix in entries:
        for path in _walk_files(entry_root):
            rel = _rel_posix(path, destination_root)
            if prefix and not rel.startswith(prefix.rstrip("/") + "/"):
                raise VerificationFailure(f"entry escapes its declared prefix: {rel}")
            size = os.stat(_fs_path(path)).st_size
            records.append({"path": rel, "size": size, "sha256": sha256_file(path)})
            total += size
    records.sort(key=lambda row: row["path"])
    return records, total


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []
        self.findings: List[Dict[str, str]] = []

    def check(self, name: str, passed: bool, detail: str) -> bool:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.findings.append({"check": name, "detail": detail})
        return passed

    @property
    def ok(self) -> bool:
        return not self.findings and all(row["passed"] for row in self.checks)


def _normalise_inventory(rows: Any, label: str) -> List[Dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise VerificationFailure(f"{label} must be a non-empty list")
    normalised: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise VerificationFailure(f"{label}[{index}] must be an object")
        try:
            path = str(row["path"]).replace("\\", "/")
            size = int(row["size"])
            digest = str(row["sha256"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationFailure(f"{label}[{index}] is malformed") from exc
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise VerificationFailure(f"{label}[{index}] has an invalid sha256")
        normalised.append({"path": path, "size": size, "sha256": digest})
    return normalised


def verify_migration(
    manifest_path: Path,
    receipt_path: Path,
    *,
    expect_remove_source: bool = True,
    oracle_mode: str = "require",
) -> Dict[str, Any]:
    """Verify a completed migration.

    ``oracle_mode`` controls the pre-migration compact-manifest layer:

    ``require``
        every entry must carry a ``compact_manifest.json`` (Stage 2B runs do);
    ``optional``
        a missing compact manifest is recorded as not applicable -- used for
        subtrees such as ``round3_balance`` that are not compact archives;
    ``skip``
        the layer is not evaluated at all.
    """

    if oracle_mode not in ("require", "optional", "skip"):
        raise VerificationFailure(f"unknown oracle_mode: {oracle_mode!r}")

    manifest = load_json(manifest_path)
    receipt = load_json(receipt_path)
    report = Report()

    if not isinstance(manifest, Mapping) or manifest.get("schema") != MIGRATION_MANIFEST_SCHEMA:
        raise VerificationFailure("unexpected migration manifest schema")
    if not isinstance(receipt, Mapping) or receipt.get("schema") != MIGRATION_RECEIPT_SCHEMA:
        raise VerificationFailure("unexpected migration receipt schema")

    entries = manifest["entries"]
    source_root = Path(receipt["source_root"])
    destination_root = Path(receipt["destination_root"])

    # ---- layer 1: receipt self-consistency -------------------------------
    report.check(
        "receipt.status",
        receipt.get("status") == "completed",
        f"status={receipt.get('status')!r} (expected 'completed')",
    )
    if expect_remove_source:
        report.check(
            "receipt.source_deleted",
            receipt.get("source_deleted") is True,
            f"source_deleted={receipt.get('source_deleted')!r} (expected True)",
        )
    else:
        report.check(
            "receipt.source_deleted",
            receipt.get("source_deleted") is False,
            f"source_deleted={receipt.get('source_deleted')!r} "
            "(expected False for a source-retaining migration)",
        )
    report.check(
        "receipt.entry_count",
        receipt.get("entry_count") == len(entries),
        f"entry_count={receipt.get('entry_count')} manifest entries={len(entries)}",
    )
    source_count = receipt.get("source_file_count")
    destination_count = receipt.get("destination_file_count")
    pair_count = receipt.get("file_pair_count")
    source_bytes = receipt.get("source_bytes")
    destination_bytes = receipt.get("destination_bytes")
    report.check(
        "receipt.counts_agree",
        source_count == destination_count == pair_count,
        f"source={source_count} destination={destination_count} pairs={pair_count}",
    )
    report.check(
        "receipt.bytes_agree",
        source_bytes == destination_bytes,
        f"source={source_bytes} destination={destination_bytes}",
    )

    source_inventory = _normalise_inventory(receipt.get("source_inventory"), "source_inventory")
    destination_inventory = _normalise_inventory(
        receipt.get("destination_inventory"), "destination_inventory"
    )
    report.check(
        "receipt.inventory_lengths",
        len(source_inventory) == source_count == len(destination_inventory),
        f"source rows={len(source_inventory)} declared={source_count} "
        f"destination rows={len(destination_inventory)}",
    )
    report.check(
        "receipt.inventory_bytes",
        sum(r["size"] for r in source_inventory) == source_bytes
        and sum(r["size"] for r in destination_inventory) == destination_bytes,
        f"source sum={sum(r['size'] for r in source_inventory)} "
        f"destination sum={sum(r['size'] for r in destination_inventory)}",
    )
    content_match = Counter((r["size"], r["sha256"]) for r in source_inventory) == Counter(
        (r["size"], r["sha256"]) for r in destination_inventory
    )
    report.check(
        "receipt.content_multiset",
        content_match,
        "source and destination (size, sha256) multisets are identical"
        if content_match
        else "source and destination (size, sha256) multisets differ",
    )

    # ---- layer 2: path correspondence ------------------------------------
    source_by_path = {row["path"]: row for row in source_inventory}
    destination_by_path = {row["path"]: row for row in destination_inventory}
    report.check(
        "receipt.inventory_paths_unique",
        len(source_by_path) == len(source_inventory)
        and len(destination_by_path) == len(destination_inventory),
        f"source unique={len(source_by_path)} destination unique={len(destination_by_path)}",
    )

    consumed: set[str] = set()
    mismatches: List[str] = []
    for entry in entries:
        src_prefix = str(entry["source"]).replace("\\", "/").strip("/")
        dst_prefix = str(entry["destination"]).replace("\\", "/").strip("/")
        for dst_path, dst_row in destination_by_path.items():
            if not dst_path.startswith(dst_prefix + "/"):
                continue
            suffix = dst_path[len(dst_prefix) + 1 :]
            src_path = f"{src_prefix}/{suffix}"
            src_row = source_by_path.get(src_path)
            if src_row is None:
                mismatches.append(f"destination {dst_path} has no source counterpart {src_path}")
                continue
            if (src_row["size"], src_row["sha256"]) != (dst_row["size"], dst_row["sha256"]):
                mismatches.append(f"content drift for {dst_path} vs {src_path}")
            consumed.add(src_path)
    unmatched = sorted(set(source_by_path) - consumed)
    if unmatched:
        mismatches.append(f"{len(unmatched)} source record(s) matched no destination, e.g. {unmatched[0]}")
    report.check(
        "receipt.path_correspondence",
        not mismatches,
        "; ".join(mismatches[:5]) if mismatches else
        f"all {len(consumed)} destination records map 1:1 onto declared entry prefixes",
    )

    # ---- layer 3a: independent re-derivation from disk -------------------
    disk_entries: List[Tuple[Path, str]] = []
    oracle_roots: Dict[str, Path] = {}
    run_ids: Dict[str, str] = {}
    for entry in entries:
        dst_prefix = str(entry["destination"]).replace("\\", "/").strip("/")
        root = destination_root / Path(dst_prefix)
        disk_entries.append((root, dst_prefix))
        run_id = str(entry.get("run_id") or Path(dst_prefix).name)
        oracle_roots[run_id] = root
        run_ids[dst_prefix] = run_id

    disk_inventory, disk_bytes = inventory_entries(disk_entries, destination_root)
    disk_by_path = {row["path"]: row for row in disk_inventory}
    report.check(
        "disk.file_count",
        len(disk_inventory) == destination_count,
        f"disk files={len(disk_inventory)} receipt={destination_count}",
    )
    report.check(
        "disk.byte_total",
        disk_bytes == destination_bytes,
        f"disk bytes={disk_bytes} receipt={destination_bytes}",
    )
    differences = [
        path
        for path, row in disk_by_path.items()
        if destination_by_path.get(path) != row
    ]
    missing_on_disk = sorted(set(destination_by_path) - set(disk_by_path))
    if missing_on_disk:
        differences.extend(missing_on_disk[:5])
    report.check(
        "disk.matches_receipt_inventory",
        not differences,
        f"{len(differences)} file(s) disagree with the receipt, e.g. {differences[:3]}"
        if differences
        else f"all {len(disk_inventory)} files re-hashed and match the receipt",
    )

    # ---- layer 3b: pre-migration compact-manifest oracle -----------------
    per_run: List[Dict[str, Any]] = []
    oracle_problems: List[str] = []
    for entry in entries:
        dst_prefix = str(entry["destination"]).replace("\\", "/").strip("/")
        run_id = str(entry.get("run_id") or Path(dst_prefix).name)
        root = destination_root / Path(dst_prefix)
        manifest_file = root / "compact_manifest.json"
        receipt_file = root / "compact_receipt.json"
        row: Dict[str, Any] = {"run_id": run_id, "materialized_path": dst_prefix}
        if not _is_file(manifest_file):
            if oracle_mode == "require":
                oracle_problems.append(f"{run_id}: compact_manifest.json missing")
                row["oracle"] = "missing_manifest"
            else:
                row["oracle"] = "not_a_compact_archive"
            per_run.append(row)
            continue
        compact = load_json(manifest_file)
        row["manifest_sha256"] = sha256_file(manifest_file)
        row["archive_sha256"] = compact.get("archive_sha256")
        if _is_file(receipt_file):
            compact_receipt = load_json(receipt_file)
            if compact_receipt.get("manifest_sha256") != row["manifest_sha256"]:
                oracle_problems.append(f"{run_id}: compact manifest hash != compact receipt claim")
                row["oracle"] = "receipt_hash_mismatch"
            if compact_receipt.get("archive_sha256") != row["archive_sha256"]:
                oracle_problems.append(f"{run_id}: archive hash disagrees between compact receipt/manifest")
        else:
            oracle_problems.append(f"{run_id}: compact_receipt.json missing")
        oracle_entries = compact.get("entries") or []
        row["oracle_entries"] = len(oracle_entries)
        row["oracle_bytes"] = sum(int(item["size_bytes"]) for item in oracle_entries)
        checked = bad = 0
        for item in oracle_entries:
            materialised = f"{dst_prefix}/materialized/{str(item['path']).replace(chr(92), '/')}"
            disk_row = disk_by_path.get(materialised)
            checked += 1
            if disk_row is None:
                bad += 1
                if bad <= 3:
                    oracle_problems.append(f"{run_id}: materialized file missing: {item['path']}")
                continue
            if disk_row["size"] != int(item["size_bytes"]) or disk_row["sha256"] != str(
                item["sha256"]
            ).lower():
                bad += 1
                if bad <= 3:
                    oracle_problems.append(f"{run_id}: materialized content drift: {item['path']}")
        row["oracle_checked"] = checked
        row["oracle_bad"] = bad
        row["oracle"] = "ok" if bad == 0 else "failed"
        per_run.append(row)

    if oracle_mode == "skip":
        report.check(
            "oracle.compact_manifests",
            True,
            "oracle layer skipped by request",
        )
    else:
        report.check(
            "oracle.compact_manifests",
            not oracle_problems,
            "; ".join(oracle_problems[:5]) if oracle_problems
            else f"{len(per_run)} run(s) reconciled against their pre-migration compact manifests",
        )

    # ---- layer 3c: source retirement -------------------------------------
    if expect_remove_source:
        surviving = [
            str(entry["source"])
            for entry in entries
            if _is_file(source_root / Path(str(entry["source"]).replace("\\", "/")))
            or _is_dir(source_root / Path(str(entry["source"]).replace("\\", "/")))
        ]
        report.check(
            "source.retired",
            not surviving,
            f"source entries still present: {surviving}" if surviving
            else f"all {len(entries)} source entries are gone, as declared",
        )

    total_oracle_entries = sum(row.get("oracle_entries") or 0 for row in per_run)
    return {
        "schema": SCHEMA,
        "status": "verified" if report.ok else "failed",
        "manifest": str(manifest_path),
        "receipt": str(receipt_path),
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "entry_count": len(entries),
        "disk_file_count": len(disk_inventory),
        "disk_bytes": disk_bytes,
        "declared_file_count": destination_count,
        "declared_bytes": destination_bytes,
        "oracle_entry_count": total_oracle_entries,
        "runs": sorted(per_run, key=lambda row: row["run_id"]),
        "checks": report.checks,
        "findings": report.findings,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the full JSON report here")
    parser.add_argument("--source-retained", action="store_true",
                        help="the migration intentionally kept its sources")
    parser.add_argument("--oracle-mode", choices=("require", "optional", "skip"),
                        default="require",
                        help="compact-manifest oracle policy (default: require)")
    args = parser.parse_args(argv)

    report = verify_migration(
        args.manifest,
        args.receipt,
        expect_remove_source=not args.source_retained,
        oracle_mode=args.oracle_mode,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary = {
        "status": report["status"],
        "entry_count": report["entry_count"],
        "disk_file_count": report["disk_file_count"],
        "disk_bytes": report["disk_bytes"],
        "declared_file_count": report["declared_file_count"],
        "declared_bytes": report["declared_bytes"],
        "failed_checks": [row["check"] for row in report["findings"]],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
