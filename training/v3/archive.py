"""Verified incremental archive bundles and receipt-gated pruning for V3 runs."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
GIB = 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _relative(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("archive paths must be non-empty and run-relative")
    return path.as_posix()


def _inside(root: Path, relative: str) -> Path:
    target = (root / _relative(relative)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("archive path escaped the run directory")
    return target


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON object expected: {path}")
    return raw


def _receipt_keys(receipt_dir: Path) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    if not receipt_dir.is_dir():
        return keys
    for path in sorted(receipt_dir.glob("*.receipt.json")):
        receipt = load_receipt(path)
        if not receipt["verified"]:
            continue
        for row in receipt["entries"]:
            keys.add((row["path"], int(row["size_bytes"]), row["checksum_sha256"]))
    return keys


def _candidate_files(run_root: Path) -> list[Path]:
    excluded_roots = {
        (run_root / "archive_staging").resolve(),
        (run_root / "archive_receipts").resolve(),
    }
    excluded_files = {(run_root / "manifests" / "coordinator.lock").resolve()}
    rows: list[Path] = []
    for path in run_root.rglob("*"):
        resolved = path.resolve()
        if not path.is_file() or resolved in excluded_files:
            continue
        if any(resolved.is_relative_to(root) for root in excluded_roots):
            continue
        if path.name.endswith((".partial", ".tmp")) or path.name.startswith("."):
            continue
        rows.append(path)
    return sorted(rows, key=lambda path: path.relative_to(run_root).as_posix())


def _group_key(path: Path) -> str:
    name = path.name
    for suffix in (".manifest.json", ".ready.json", ".npz"):
        if name.endswith(suffix):
            return str(path.with_name(name.removesuffix(suffix)))
    return str(path)


def _select_increment(
    rows: Iterable[dict[str, Any]], target_bytes: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        key = _group_key(Path(row["path"]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    selected: list[dict[str, Any]] = []
    size = 0
    for key in order:
        group = groups[key]
        group_size = sum(int(row["size_bytes"]) for row in group)
        if selected and size + group_size > target_bytes:
            break
        selected.extend(group)
        size += group_size
        if size >= target_bytes:
            break
    return selected


def create_archive_bundle(
    run_dir: str | Path,
    *,
    bundle_target_bytes: int,
) -> dict[str, Any]:
    """Create one immutable incremental tar after snapshotting stable files."""

    run_root = Path(run_dir).resolve()
    if not (run_root / "run_manifest.json").is_file():
        raise FileNotFoundError("V3 run_manifest.json is missing")
    if (run_root / "manifests" / "coordinator.lock").exists():
        raise RuntimeError("training coordinator is active; archive only at a safe boundary")
    if int(bundle_target_bytes) < 1:
        raise ValueError("bundle_target_bytes must be positive")
    staging = run_root / "archive_staging"
    staging.mkdir(parents=True, exist_ok=True)
    operation_lock = staging / "archive.lock"
    try:
        descriptor = os.open(operation_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("another archive operation is active") from exc
    os.close(descriptor)
    try:
        already_verified = _receipt_keys(run_root / "archive_receipts")
        catalog: list[dict[str, Any]] = []
        for path in _candidate_files(run_root):
            relative = path.relative_to(run_root).as_posix()
            size = path.stat().st_size
            checksum = _sha256_file(path)
            if (relative, size, checksum) in already_verified:
                continue
            catalog.append(
                {"path": relative, "size_bytes": size, "checksum_sha256": checksum}
            )
        configured_hard_bytes = 10 * GIB
        resolved_config = run_root / "resolved_config.json"
        if resolved_config.is_file():
            configured_storage = _load_json(resolved_config)["runtime"]["storage"]
            configured_hard_bytes = int(
                float(configured_storage.get("hard_free_gib", 10.0)) * GIB
            )
        disk = os.statvfs(run_root) if hasattr(os, "statvfs") else None
        if disk is None:
            import shutil

            free_bytes = shutil.disk_usage(run_root).free
        else:
            free_bytes = int(disk.f_bavail * disk.f_frsize)
        safe_bundle_bytes = max(0, free_bytes - configured_hard_bytes - 1024**2)
        if safe_bundle_bytes < 1:
            raise RuntimeError("insufficient staging space above the configured hard reserve")
        selected = _select_increment(
            catalog,
            min(int(bundle_target_bytes), safe_bundle_bytes),
        )
        if not selected:
            return {
                "status": "nothing_to_archive",
                "run_dir": str(run_root),
                "remaining_unarchived_files": 0,
            }
        selected_bytes = sum(int(row["size_bytes"]) for row in selected)
        if selected_bytes > safe_bundle_bytes:
            raise RuntimeError(
                "the next indivisible archive group cannot be staged above the hard reserve"
            )
        run_manifest = _load_json(run_root / "run_manifest.json")
        bundle_id = (
            f"{run_manifest.get('run_id', run_root.name)}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        )
        partial = staging / f"{bundle_id}.tar.partial"
        archive_path = staging / f"{bundle_id}.tar"
        with tarfile.open(partial, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for row in selected:
                source = _inside(run_root, row["path"])
                before = (source.stat().st_size, _sha256_file(source))
                if before != (row["size_bytes"], row["checksum_sha256"]):
                    raise RuntimeError(f"archive source changed before packing: {row['path']}")
                archive.add(source, arcname=row["path"], recursive=False)
                after = (source.stat().st_size, _sha256_file(source))
                if after != before:
                    raise RuntimeError(f"archive source changed while packing: {row['path']}")
        os.replace(partial, archive_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": "connect4-v3-incremental-archive",
            "bundle_id": bundle_id,
            "run_id": str(run_manifest.get("run_id", run_root.name)),
            "run_config_hash": run_manifest.get("config_hash"),
            "created_at": _utc_now(),
            "archive_file": archive_path.name,
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256_file(archive_path),
            "entries": selected,
        }
        manifest_path = staging / f"{bundle_id}.manifest.json"
        _atomic_json(manifest_path, manifest)
        return {
            "status": "created",
            "bundle_id": bundle_id,
            "run_id": manifest["run_id"],
            "archive": str(archive_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "archive_size_bytes": manifest["archive_size_bytes"],
            "entries": len(selected),
            "remaining_unarchived_files": len(catalog) - len(selected),
        }
    finally:
        operation_lock.unlink(missing_ok=True)
        for partial in staging.glob("*.partial"):
            partial.unlink(missing_ok=True)


def load_archive_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    manifest = _load_json(source)
    required = {
        "schema_version",
        "format",
        "bundle_id",
        "run_id",
        "run_config_hash",
        "created_at",
        "archive_file",
        "archive_size_bytes",
        "archive_sha256",
        "entries",
    }
    if set(manifest) != required or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("archive manifest schema mismatch")
    if manifest["format"] != "connect4-v3-incremental-archive":
        raise ValueError("unsupported archive format")
    paths: list[str] = []
    for row in manifest["entries"]:
        if set(row) != {"path", "size_bytes", "checksum_sha256"}:
            raise ValueError("archive entry schema mismatch")
        row["path"] = _relative(row["path"])
        if int(row["size_bytes"]) < 0 or len(str(row["checksum_sha256"])) != 64:
            raise ValueError("archive entry size or checksum is invalid")
        paths.append(row["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("archive manifest contains duplicate paths")
    return manifest


def verify_archive_bundle(
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    extract_to: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    archive_source = Path(archive_path).resolve()
    manifest_source = Path(manifest_path).resolve()
    manifest = load_archive_manifest(manifest_source)
    if archive_source.name != manifest["archive_file"]:
        raise ValueError("archive filename differs from manifest")
    if archive_source.stat().st_size != int(manifest["archive_size_bytes"]):
        raise ValueError("archive size differs from manifest")
    if _sha256_file(archive_source) != manifest["archive_sha256"]:
        raise ValueError("archive checksum differs from manifest")
    expected = {row["path"]: row for row in manifest["entries"]}
    destination = None if extract_to is None else Path(extract_to).resolve()
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_source, mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if set(names) != set(expected) or len(names) != len(expected):
            raise ValueError("archive members differ from manifest entries")
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("archive may contain regular files only")
            row = expected[member.name]
            if member.size != int(row["size_bytes"]):
                raise ValueError(f"archive member size differs: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            digest = hashlib.sha256()
            target = None if destination is None else _inside(destination, member.name)
            temporary = None
            handle = None
            try:
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                    handle = temporary.open("wb")
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(block)
                    if handle is not None:
                        handle.write(block)
                if handle is not None:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                    handle = None
                    os.replace(temporary, target)
            finally:
                if handle is not None:
                    handle.close()
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            if digest.hexdigest() != row["checksum_sha256"]:
                raise ValueError(f"archive member checksum differs: {member.name}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "format": "connect4-v3-archive-receipt",
        "receipt_id": f"receipt-{manifest['bundle_id']}-{uuid.uuid4().hex[:8]}",
        "bundle_id": manifest["bundle_id"],
        "run_id": manifest["run_id"],
        "archive_manifest_file": manifest_source.name,
        "archive_manifest_sha256": _sha256_file(manifest_source),
        "archive_sha256": manifest["archive_sha256"],
        "verified": True,
        "verified_at": _utc_now(),
        "receiver": socket.gethostname(),
        "entries": manifest["entries"],
    }
    if receipt_path is not None:
        _atomic_json(Path(receipt_path), receipt)
    return receipt


def load_receipt(path: str | Path) -> dict[str, Any]:
    receipt = _load_json(Path(path))
    required = {
        "schema_version",
        "format",
        "receipt_id",
        "bundle_id",
        "run_id",
        "archive_manifest_file",
        "archive_manifest_sha256",
        "archive_sha256",
        "verified",
        "verified_at",
        "receiver",
        "entries",
    }
    if set(receipt) != required or receipt["schema_version"] != SCHEMA_VERSION:
        raise ValueError("archive receipt schema mismatch")
    if receipt["format"] != "connect4-v3-archive-receipt" or receipt["verified"] is not True:
        raise ValueError("archive receipt is not a verified V3 receipt")
    for row in receipt["entries"]:
        row["path"] = _relative(row["path"])
    return receipt


def ingest_archive_receipt(run_dir: str | Path, receipt_path: str | Path) -> Path:
    run_root = Path(run_dir).resolve()
    receipt_source = Path(receipt_path).resolve()
    receipt = load_receipt(receipt_source)
    manifest_path = run_root / "archive_staging" / receipt["archive_manifest_file"]
    manifest = load_archive_manifest(manifest_path)
    if _sha256_file(manifest_path) != receipt["archive_manifest_sha256"]:
        raise ValueError("receipt manifest checksum differs from cloud manifest")
    if (
        receipt["bundle_id"] != manifest["bundle_id"]
        or receipt["run_id"] != manifest["run_id"]
        or receipt["archive_sha256"] != manifest["archive_sha256"]
        or receipt["entries"] != manifest["entries"]
    ):
        raise ValueError("receipt contents differ from the archived bundle")
    target = run_root / "archive_receipts" / f"{receipt['receipt_id']}.receipt.json"
    if target.exists():
        if _sha256_file(target) != _sha256_file(receipt_source):
            raise FileExistsError("receipt ID already exists with different content")
    else:
        _atomic_json(target, receipt)
    incoming_root = (run_root / "archive_receipts").resolve()
    if receipt_source != target.resolve() and receipt_source.is_relative_to(incoming_root):
        receipt_source.unlink()
    return target


def _latest_commit(run_root: Path) -> dict[str, Any]:
    pointer = _load_json(run_root / "manifests" / "latest_generation.json")
    return _load_json(_inside(run_root, pointer["commit"]))


def plan_prune(run_dir: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    resolved = _load_json(run_root / "resolved_config.json")
    storage = resolved["runtime"]["storage"]
    protected: set[str] = {
        "run_manifest.json",
        "resolved_config.json",
        "manifests/latest_generation.json",
    }
    keep_counts = {
        "checkpoints": int(storage["keep_checkpoints"]),
        "accepted": int(storage["keep_accepted"]),
        "rejected": int(storage["keep_rejected"]),
    }
    commit_paths = sorted(
        (run_root / "manifests" / "generations").glob("g*.json"),
        key=lambda path: path.name,
        reverse=True,
    )[: max(1, keep_counts["checkpoints"])]
    for commit_path in commit_paths:
        commit = _load_json(commit_path)
        protected.add(commit_path.relative_to(run_root).as_posix())
        protected.update({str(commit["checkpoint"]), str(commit["audit_index"])})
        for key in ("accepted_model_path", "candidate_path"):
            if commit.get(key):
                protected.add(str(commit[key]))
        for row in commit["replay_shards"]:
            shard = str(row["path"])
            protected.update(
                {
                    shard,
                    shard.removesuffix(".npz") + ".manifest.json",
                    shard.removesuffix(".npz") + ".ready.json",
                }
            )
        audit_index = _load_json(_inside(run_root, str(commit["audit_index"])))
        audit_dir = PurePosixPath(str(commit["audit_index"])).parent
        for row in audit_index.get("replays", []):
            protected.add((audit_dir / row["filename"]).as_posix())
    for directory, count in keep_counts.items():
        paths = sorted((run_root / directory).glob("*.pt"), key=lambda path: path.name, reverse=True)
        protected.update(path.relative_to(run_root).as_posix() for path in paths[:count])
    receipt_keys = _receipt_keys(run_root / "archive_receipts")
    prunable_roots = {"replay/raw", "checkpoints", "accepted", "rejected", "samples"}
    decisions: list[dict[str, Any]] = []
    for path in _candidate_files(run_root):
        relative = path.relative_to(run_root).as_posix()
        if not any(relative == root or relative.startswith(root + "/") for root in prunable_roots):
            continue
        size = path.stat().st_size
        checksum = _sha256_file(path)
        reasons: list[str] = []
        if relative in protected:
            reasons.append("current_resume_or_recent_retention")
        if (relative, size, checksum) not in receipt_keys:
            reasons.append("verified_local_receipt_missing_or_mismatched")
        decisions.append(
            {
                "path": relative,
                "size_bytes": size,
                "checksum_sha256": checksum,
                "action": "keep" if reasons else "eligible_to_prune",
                "reasons": reasons or ["verified_locally_and_not_required_for_resume"],
            }
        )
    staged_archives_seen: set[str] = set()
    for receipt_path in sorted((run_root / "archive_receipts").glob("*.receipt.json")):
        receipt = load_receipt(receipt_path)
        manifest_path = run_root / "archive_staging" / receipt["archive_manifest_file"]
        if not manifest_path.is_file():
            continue
        manifest = load_archive_manifest(manifest_path)
        archive_path = run_root / "archive_staging" / manifest["archive_file"]
        if not archive_path.is_file():
            continue
        relative = archive_path.relative_to(run_root).as_posix()
        if relative in staged_archives_seen:
            continue
        staged_archives_seen.add(relative)
        checksum = _sha256_file(archive_path)
        verified = (
            receipt["archive_manifest_sha256"] == _sha256_file(manifest_path)
            and receipt["archive_sha256"] == checksum == manifest["archive_sha256"]
        )
        decisions.append(
            {
                "path": relative,
                "size_bytes": archive_path.stat().st_size,
                "checksum_sha256": checksum,
                "action": "eligible_to_prune" if verified else "keep",
                "reasons": [
                    "verified_local_archive_receipt"
                    if verified
                    else "archive_receipt_missing_or_mismatched"
                ],
            }
        )
    eligible = [row for row in decisions if row["action"] == "eligible_to_prune"]
    if len({row["path"] for row in decisions}) != len(decisions):
        raise RuntimeError("prune plan contains duplicate artifact paths")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_root),
        "protected_paths": sorted(protected),
        "decisions": decisions,
        "eligible_paths": [row["path"] for row in eligible],
        "eligible_bytes": sum(int(row["size_bytes"]) for row in eligible),
        "deletion_enabled": False,
    }


def execute_prune(run_dir: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    if (run_root / "manifests" / "coordinator.lock").exists():
        raise RuntimeError("training coordinator is active; prune only at a safe boundary")
    before = plan_prune(run_root)
    removed: list[str] = []
    removed_bytes = 0
    decisions = {row["path"]: row for row in before["decisions"]}
    for relative in before["eligible_paths"]:
        row = decisions[relative]
        target = _inside(run_root, relative)
        if not target.is_file():
            raise RuntimeError(f"prune target disappeared during revalidation: {relative}")
        if target.stat().st_size != row["size_bytes"] or _sha256_file(target) != row["checksum_sha256"]:
            raise RuntimeError(f"prune target changed during revalidation: {relative}")
        target.unlink()
        removed.append(relative)
        removed_bytes += int(row["size_bytes"])
    after = plan_prune(run_root)
    return {
        "status": "complete",
        "removed": removed,
        "removed_bytes": removed_bytes,
        "remaining_eligible_paths": after["eligible_paths"],
    }


__all__ = [
    "create_archive_bundle",
    "execute_prune",
    "ingest_archive_receipt",
    "load_archive_manifest",
    "load_receipt",
    "plan_prune",
    "verify_archive_bundle",
]
