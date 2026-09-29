"""Plan and execute explicit, hash-verified Stage 2 archive migrations.

The tool is intentionally conservative: a manifest is the only source of
truth, planning is read-only, destinations must not collide, and source files
are retained unless both ``finalize`` and ``--remove-source`` are explicit.
It is suitable for long Windows paths and for manifests written through a
``subst X:`` drive by accepting path aliases (for example ``X:\\=D:\\...``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "stage2-archive-migration-manifest-v1"


class MigrationError(RuntimeError):
    """Raised when a migration cannot be proven safe to perform."""


@dataclass(frozen=True)
class FilePair:
    source: Path
    destination: Path
    source_rel: str
    destination_rel: str


@dataclass
class MigrationPlan:
    manifest: Dict[str, Any]
    source_root: Path
    destination_root: Path
    entries: List[Tuple[Path, Path]]
    file_pairs: List[FilePair]
    source_inventory: List[Dict[str, Any]]
    source_file_count: int
    source_bytes: int


def _fs_path(path: Path) -> str:
    """Return a filesystem-safe path, using Win32 extended syntax when needed."""
    raw = os.path.abspath(os.fspath(path))
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw[2:]
    return "\\\\?\\" + raw


def _exists(path: Path) -> bool:
    return os.path.exists(_fs_path(path))


def _is_dir(path: Path) -> bool:
    return os.path.isdir(_fs_path(path))


def _is_file(path: Path) -> bool:
    return os.path.isfile(_fs_path(path))


def _is_symlink(path: Path) -> bool:
    return os.path.islink(_fs_path(path))


def _stat(path: Path) -> os.stat_result:
    return os.stat(_fs_path(path))


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    os.makedirs(_fs_path(path.parent), exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=_fs_path(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, _fs_path(path))
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_manifest(path_or_value: Any) -> Dict[str, Any]:
    if isinstance(path_or_value, (str, os.PathLike)):
        path = Path(path_or_value)
        try:
            with open(_fs_path(path), "r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError) as exc:
            raise MigrationError("cannot read migration manifest: %s" % path) from exc
    else:
        value = dict(path_or_value)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise MigrationError("manifest schema must be %s" % SCHEMA)
    if not value.get("source_root") or not value.get("destination_root"):
        raise MigrationError("manifest requires source_root and destination_root")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise MigrationError("manifest entries must be a non-empty list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("source") or not entry.get("destination"):
            raise MigrationError("entry %d requires source and destination" % index)
    return value


def _normalise_aliases(aliases: Optional[Mapping[str, str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in (aliases or {}).items():
        key = str(key)
        if len(key) == 2 and key[1] == ":":
            key += "\\"
        result[key] = str(value)
    return result


def _expand_alias(raw: str, aliases: Mapping[str, str]) -> str:
    for alias in sorted(aliases, key=len, reverse=True):
        if raw.lower().startswith(alias.lower()):
            remainder = raw[len(alias):]
            target = aliases[alias]
            if remainder and not target.endswith(("\\", "/")) and not remainder.startswith(("\\", "/")):
                target += os.sep
            return target + remainder
    return raw


def _resolved(path: Path) -> Path:
    # strict=False permits destination parents that do not exist yet.
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        return absolute.resolve(strict=False)
    except OSError:
        # Path.resolve can still hit legacy MAX_PATH behavior on older
        # Windows/Python combinations; retain the normalized absolute path.
        return absolute


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.normcase(os.path.commonpath([str(path), str(root)])) == os.path.normcase(str(root))
    except ValueError:
        return False


def _resolve_entry(raw: str, root: Path, aliases: Mapping[str, str], label: str) -> Path:
    expanded = _expand_alias(str(raw), aliases)
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = _resolved(candidate)
    if not _within(candidate, root):
        raise MigrationError("%s escapes its root: %s" % (label, raw))
    return candidate


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(_fs_path(path), "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MigrationError("cannot hash source=%s: %s" % (path, exc)) from exc
    return digest.hexdigest()


def _inventory(paths: Iterable[Tuple[Path, str]]) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    total = 0
    for path, relative in sorted(paths, key=lambda item: item[1].lower()):
        if not _is_file(path) or _is_symlink(path):
            raise MigrationError("inventory path is not a regular file: %s" % path)
        try:
            size = _stat(path).st_size
        except OSError as exc:
            raise MigrationError("cannot stat source=%s: %s" % (path, exc)) from exc
        record = {"path": relative, "size": size, "sha256": _hash_file(path)}
        records.append(record)
        total += size
    return records, total


def _directory_files(source: Path, destination: Path) -> List[FilePair]:
    result: List[FilePair] = []
    source_fs = _fs_path(source)
    for current, dirs, files in os.walk(source_fs, followlinks=False):
        relative_current = os.path.relpath(current, source_fs)
        current_path = source if relative_current == "." else source / relative_current
        for name in list(dirs) + list(files):
            if _is_symlink(current_path / name):
                raise MigrationError("symlinks are not allowed in archive migration: %s" % (current_path / name))
        relative = current_path.relative_to(source)
        for name in files:
            source_file = current_path / name
            dest_file = destination / relative / name
            result.append(FilePair(source_file, dest_file, str(source_file.relative_to(source.parent)), str(dest_file)))
    if not result and not _exists(source):
        raise MigrationError("source disappeared during inventory: %s" % source)
    return result


def _ancestor(a: Path, b: Path) -> bool:
    return _within(a, b) or _within(b, a)


def _validate_roots(source_root: Path, destination_root: Path) -> None:
    if not _is_dir(source_root):
        raise MigrationError("source_root is not a directory: %s" % source_root)
    if _exists(destination_root) and not _is_dir(destination_root):
        raise MigrationError("destination_root is not a directory: %s" % destination_root)


def build_plan(
    manifest_or_value: Any,
    path_aliases: Optional[Mapping[str, str]] = None,
    allow_destination_collisions: bool = False,
) -> MigrationPlan:
    manifest = load_manifest(manifest_or_value)
    raw_aliases = dict(manifest.get("path_aliases", {}))
    raw_aliases.update(path_aliases or {})
    aliases = _normalise_aliases(raw_aliases)
    source_raw = _expand_alias(str(manifest["source_root"]), aliases)
    destination_raw = _expand_alias(str(manifest["destination_root"]), aliases)
    source_root = _resolved(Path(source_raw))
    destination_root = _resolved(Path(destination_raw))
    _validate_roots(source_root, destination_root)

    entries: List[Tuple[Path, Path]] = []
    for index, entry in enumerate(manifest["entries"]):
        source = _resolve_entry(entry["source"], source_root, aliases, "source entry %d" % index)
        destination = _resolve_entry(entry["destination"], destination_root, aliases, "destination entry %d" % index)
        if not _exists(source) or _is_symlink(source):
            raise MigrationError("source entry is missing or symlink: %s" % source)
        if not _is_file(source) and not _is_dir(source):
            raise MigrationError("source entry is not a file or directory: %s" % source)
        if not allow_destination_collisions and _exists(destination):
            raise MigrationError("destination collision: %s" % destination)
        parent = destination.parent
        while _within(parent, destination_root) and parent != destination_root:
            if _exists(parent) and not _is_dir(parent):
                raise MigrationError("destination parent is not a directory: %s" % parent)
            parent = parent.parent
        entries.append((source, destination))

    for index, (source, destination) in enumerate(entries):
        for other_source, other_destination in entries[index + 1 :]:
            if _ancestor(source, other_source):
                raise MigrationError("overlapping source entries: %s and %s" % (source, other_source))
            if _ancestor(destination, other_destination):
                raise MigrationError("overlapping destination entries: %s and %s" % (destination, other_destination))

    # A nested destination root is safe only when no selected source entry
    # overlaps any selected destination entry. This prevents copying into a
    # source subtree that is itself being traversed or removed.
    for source, _ in entries:
        for _, destination in entries:
            if _ancestor(source, destination):
                raise MigrationError("source/destination entries intersect: %s and %s" % (source, destination))

    file_pairs: List[FilePair] = []
    for source, destination in entries:
        if _is_file(source):
            file_pairs.append(FilePair(source, destination, str(source.relative_to(source_root)), str(destination.relative_to(destination_root))))
        else:
            for pair in _directory_files(source, destination):
                file_pairs.append(FilePair(pair.source, pair.destination, str(pair.source.relative_to(source_root)), str(pair.destination.relative_to(destination_root))))

    destinations_seen = set()
    for pair in file_pairs:
        key = os.path.normcase(str(pair.destination))
        if key in destinations_seen:
            raise MigrationError("destination collision: %s" % pair.destination)
        destinations_seen.add(key)
        if not allow_destination_collisions and _exists(pair.destination):
            raise MigrationError("destination collision: %s" % pair.destination)

    source_inventory, source_bytes = _inventory((pair.source, pair.source_rel) for pair in file_pairs)
    return MigrationPlan(manifest, source_root, destination_root, entries, file_pairs, source_inventory, len(file_pairs), source_bytes)


def _destination_inventory(plan: MigrationPlan) -> Tuple[List[Dict[str, Any]], int]:
    return _inventory((pair.destination, pair.destination_rel) for pair in plan.file_pairs)


def _receipt(plan: MigrationPlan, status: str, source_deleted: bool, destination_inventory: Optional[List[Dict[str, Any]]] = None, destination_bytes: Optional[int] = None) -> Dict[str, Any]:
    return {
        "schema": "stage2-archive-migration-receipt-v1",
        "status": status,
        "source_root": str(plan.source_root),
        "destination_root": str(plan.destination_root),
        "entry_count": len(plan.entries),
        "file_pair_count": len(plan.file_pairs),
        "source_file_count": plan.source_file_count,
        "source_bytes": plan.source_bytes,
        "destination_file_count": len(destination_inventory) if destination_inventory is not None else None,
        "destination_bytes": destination_bytes,
        "source_inventory": plan.source_inventory,
        "destination_inventory": destination_inventory,
        "source_deleted": source_deleted,
    }


def _path_map(plan: MigrationPlan) -> Dict[str, Any]:
    return {
        "schema": "stage2-archive-legacy-path-map-v1",
        "source_root": str(plan.source_root),
        "destination_root": str(plan.destination_root),
        "mappings": [
            {
                "source": str(source),
                "destination": str(destination),
                "source_rel": str(source.relative_to(plan.source_root)),
                "destination_rel": str(destination.relative_to(plan.destination_root)),
            }
            for source, destination in plan.entries
        ],
    }


def write_plan_artifacts(plan: MigrationPlan, output_dir: Path, receipt: Optional[Dict[str, Any]] = None) -> None:
    output_dir = Path(output_dir)
    _json_dump(output_dir / "migration_plan.json", {
        "schema": "stage2-archive-migration-plan-v1",
        "source_root": str(plan.source_root),
        "destination_root": str(plan.destination_root),
        "file_pairs": [
            {"source": str(pair.source), "destination": str(pair.destination), "source_rel": pair.source_rel, "destination_rel": pair.destination_rel}
            for pair in plan.file_pairs
        ],
        "source_file_count": plan.source_file_count,
        "source_bytes": plan.source_bytes,
    })
    _json_dump(output_dir / "legacy_path_map.json", _path_map(plan))
    _json_dump(output_dir / "migration_receipt.json", receipt or _receipt(plan, "planned", False))


def execute_plan(plan: MigrationPlan, remove_source: bool = False, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    # Rebuild the source inventory immediately before writing, catching stale plans.
    current_inventory, current_bytes = _inventory((pair.source, pair.source_rel) for pair in plan.file_pairs)
    if current_inventory != plan.source_inventory or current_bytes != plan.source_bytes:
        raise MigrationError("source changed after planning; regenerate the plan")
    for pair in plan.file_pairs:
        if _exists(pair.destination):
            raise MigrationError("destination collision at finalize: %s" % pair.destination)

    try:
        for source, destination in plan.entries:
            if _is_dir(source):
                os.makedirs(_fs_path(destination), exist_ok=True)
        for pair in plan.file_pairs:
            os.makedirs(_fs_path(pair.destination.parent), exist_ok=True)
            try:
                shutil.copy2(_fs_path(pair.source), _fs_path(pair.destination))
            except OSError as exc:
                raise MigrationError(
                    "copy failed source=%s destination=%s: %s" % (pair.source, pair.destination, exc)
                ) from exc
        destination_inventory, destination_bytes = _destination_inventory(plan)
        expected = sorted((item["size"], item["sha256"]) for item in plan.source_inventory)
        actual = sorted((item["size"], item["sha256"]) for item in destination_inventory)
        if expected != actual or len(destination_inventory) != plan.source_file_count:
            raise MigrationError("destination verification failed")
    except Exception:
        # Leave a failed partial destination for inspection; never remove source.
        raise

    source_deleted = False
    if remove_source:
        # Verify source one last time before deleting anything.
        final_source_inventory, final_source_bytes = _inventory((pair.source, pair.source_rel) for pair in plan.file_pairs)
        if final_source_inventory != plan.source_inventory or final_source_bytes != plan.source_bytes:
            raise MigrationError("source changed before cleanup; source retained")
        for source, _ in plan.entries:
            if _is_dir(source):
                shutil.rmtree(_fs_path(source))
            else:
                os.unlink(_fs_path(source))
        source_deleted = all(not _exists(source) for source, _ in plan.entries)
        if not source_deleted:
            raise MigrationError("source cleanup did not complete")

    receipt = _receipt(plan, "completed", source_deleted, destination_inventory, destination_bytes)
    if output_dir is not None:
        write_plan_artifacts(plan, Path(output_dir), receipt)
    return receipt


def cleanup_destination(plan: MigrationPlan, execute: bool = False) -> Dict[str, Any]:
    """Remove only exact manifest destination entries, never the destination root.

    The default is a dry-run. ``execute=True`` is an explicit destructive
    operation intended only for a known partial finalize result.
    """
    root = _resolved(plan.destination_root)
    targets = [destination for _, destination in plan.entries]
    for target in targets:
        target = _resolved(target)
        if not _within(target, root) or target == root:
            raise MigrationError("cleanup target escapes destination root: %s" % target)

    existing = [target for target in targets if _exists(target)]
    removed: List[str] = []
    if execute:
        for target in existing:
            try:
                if _is_dir(target):
                    shutil.rmtree(_fs_path(target))
                else:
                    os.unlink(_fs_path(target))
            except OSError as exc:
                raise MigrationError("cleanup failed destination=%s: %s" % (target, exc)) from exc
            removed.append(str(target))
    return {
        "schema": "stage2-archive-cleanup-receipt-v1",
        "status": "cleaned" if execute else "planned",
        "destination_root": str(root),
        "targets": [str(target) for target in targets],
        "existing_targets": [str(target) for target in existing],
        "removed_targets": removed,
    }


def _parse_alias(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("alias must be KEY=PATH")
    key, target = value.split("=", 1)
    if not key or not target:
        raise argparse.ArgumentTypeError("alias must be KEY=PATH")
    return key, target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "finalize", "cleanup"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alias", action="append", default=[], type=_parse_alias, metavar="KEY=PATH")
    parser.add_argument("--dry-run", action="store_true", help="validate and write a plan without copying")
    parser.add_argument("--remove-source", action="store_true", help="delete selected sources after finalize verification")
    parser.add_argument("--execute-cleanup", action="store_true", help="explicitly remove manifest destination entries")
    args = parser.parse_args(argv)
    if args.action != "finalize" and args.remove_source:
        parser.error("--remove-source requires finalize")
    if args.action != "cleanup" and args.execute_cleanup:
        parser.error("--execute-cleanup requires cleanup")
    manifest = load_manifest(args.manifest)
    aliases = dict(args.alias)
    plan = build_plan(manifest, aliases, allow_destination_collisions=args.action == "cleanup")
    output_dir = args.output_dir or (plan.destination_root / "_migration")
    if args.action == "cleanup":
        report = cleanup_destination(plan, execute=args.execute_cleanup and not args.dry_run)
        if args.output_dir is not None:
            _json_dump(Path(args.output_dir) / "cleanup_receipt.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    if args.action == "plan" or args.dry_run:
        write_plan_artifacts(plan, output_dir)
        print(json.dumps({"status": "planned", "output_dir": str(output_dir)}, ensure_ascii=False))
        return 0
    receipt = execute_plan(plan, remove_source=args.remove_source, output_dir=output_dir)
    print(json.dumps({"status": receipt["status"], "source_deleted": receipt["source_deleted"], "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
