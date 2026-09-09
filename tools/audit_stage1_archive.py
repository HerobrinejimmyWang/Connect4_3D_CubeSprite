"""Read-only reconciliation for the Stage 1 V3 archive.

This tool deliberately does not move, delete, upload, prune, or ingest files.
It combines local bundles/receipts/materialized files with a read-only SSH
inventory and historical bundle-name clues from the attached image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STATUS_VALUES = {
    "present_verified",
    "remote_only",
    "historical_returned_unverified",
    "materialized_only",
    "conflict",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries_signature(entries: Any) -> str:
    if not isinstance(entries, list):
        return ""
    normalized = []
    for row in entries:
        if not isinstance(row, dict):
            return ""
        normalized.append(
            {
                "path": row.get("path"),
                "size_bytes": row.get("size_bytes"),
                "checksum_sha256": row.get("checksum_sha256"),
            }
        )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_clues(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    value = load_json(path) or {}
    fragments = value.get("bundle_id_fragments", [])
    return {str(item).lower() for item in fragments if str(item).strip()}


def historical_match(bundle_id: str, fragments: set[str]) -> bool:
    lowered = bundle_id.lower()
    return any(fragment in lowered for fragment in fragments)


def local_inventory(local_root: Path, hash_archives: bool) -> dict[str, Any]:
    bundles_dir = local_root / "bundles"
    receipts_dirs = [local_root / "receipts"]
    receipts_dirs.extend(path for path in local_root.rglob("archive_receipts") if path.is_dir())

    receipt_by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receipt_paths: list[str] = []
    for directory in sorted(set(receipts_dirs)):
        for path in sorted(directory.glob("*.receipt.json")):
            value = load_json(path)
            if value and value.get("bundle_id"):
                row = dict(value)
                row["_path"] = str(path.resolve())
                receipt_by_bundle[str(value["bundle_id"])].append(row)
                receipt_paths.append(str(path.resolve()))

    records: list[dict[str, Any]] = []
    if bundles_dir.is_dir():
        for manifest_path in sorted(bundles_dir.rglob("*.manifest.json")):
            manifest = load_json(manifest_path)
            if not manifest or not manifest.get("bundle_id"):
                continue
            bundle_id = str(manifest["bundle_id"])
            archive_name = str(manifest.get("archive_file") or f"{bundle_id}.tar")
            archive_path = manifest_path.parent / archive_name
            archive_sha = sha256_file(archive_path) if hash_archives and archive_path.is_file() else None
            records.append(
                {
                    "source": "local",
                    "bundle_id": bundle_id,
                    "run_id": manifest.get("run_id"),
                    "archive_sha256": manifest.get("archive_sha256"),
                    "archive_size_bytes": manifest.get("archive_size_bytes"),
                    "actual_archive_sha256": archive_sha,
                    "manifest_sha256": sha256_file(manifest_path),
                    "manifest_path": str(manifest_path.resolve()),
                    "archive_path": str(archive_path.resolve()) if archive_path.exists() else None,
                    "archive_exists": archive_path.is_file(),
                    "entries_signature": entries_signature(manifest.get("entries")),
                    "entry_count": len(manifest.get("entries", [])) if isinstance(manifest.get("entries"), list) else None,
                    "receipts": receipt_by_bundle.get(bundle_id, []),
                }
            )

    materialized = local_root / "materialized"
    materialized_files: list[dict[str, Any]] = []
    materialized_file_count = 0
    if materialized.is_dir():
        covered_paths: set[str] = set()
        for record in records:
            manifest = load_json(Path(record["manifest_path"]))
            for entry in (manifest or {}).get("entries", []):
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    covered_paths.add(entry["path"].replace("/", os.sep))
        for path in materialized.rglob("*"):
            if not path.is_file():
                continue
            materialized_file_count += 1
            relative = str(path.relative_to(materialized))
            if relative not in covered_paths:
                materialized_files.append(
                    {
                        "path": str(path.resolve()),
                        "relative_path": relative.replace(os.sep, "/"),
                        "size_bytes": path.stat().st_size,
                    }
                )

    return {
        "records": records,
        "receipt_paths": receipt_paths,
        "materialized_only": materialized_files,
        "materialized_file_count": materialized_file_count,
    }


REMOTE_SCRIPT = r'''
import hashlib, json, os, subprocess, sys
from pathlib import Path

repo = Path(sys.argv[2])
cold = Path(sys.argv[3])
orphans = Path(sys.argv[4])

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None

def sig(entries):
    if not isinstance(entries, list):
        return ""
    return json.dumps([
        {"path": r.get("path"), "size_bytes": r.get("size_bytes"),
         "checksum_sha256": r.get("checksum_sha256")}
        for r in entries if isinstance(r, dict)
    ], sort_keys=True, separators=(",", ":"))

records = []
receipts = []
seen_manifests = set()
roots = []
run_root = repo / "training" / "runs"
if run_root.is_dir():
    roots.extend(p for p in run_root.iterdir() if p.is_dir() and p.name != "stage2")
if cold.is_dir():
    roots.append(cold)
if orphans.is_dir():
    roots.append(orphans)

for root in roots:
    manifest_paths = set(root.rglob("archive_staging/*.manifest.json"))
    # The orphan handoff is intentionally outside a run's archive_staging dir.
    if root == orphans:
        manifest_paths.update(root.rglob("*.manifest.json"))
    for manifest_path in sorted(manifest_paths):
        if "archive_receipts" in manifest_path.parts:
            continue
        # Keep symlinked repo and cold-storage paths as separate locations. The
        # reconciliation layer groups them by bundle_id without treating equal
        # content at two locations as a conflict.
        key = str(manifest_path)
        if key in seen_manifests:
            continue
        seen_manifests.add(key)
        manifest = load(manifest_path)
        if not manifest or not manifest.get("bundle_id"):
            continue
        archive_path = manifest_path.parent / str(manifest.get("archive_file") or (manifest["bundle_id"] + ".tar"))
        records.append({
            "source": "remote",
            "bundle_id": manifest.get("bundle_id"),
            "run_id": manifest.get("run_id"),
            "archive_sha256": manifest.get("archive_sha256"),
            "archive_size_bytes": manifest.get("archive_size_bytes"),
            "manifest_sha256": digest(manifest_path),
            "manifest_path": str(manifest_path),
            "archive_path": str(archive_path) if archive_path.is_file() else None,
            "archive_exists": archive_path.is_file(),
            "archive_actual_size_bytes": archive_path.stat().st_size if archive_path.is_file() else None,
            "entries_signature": sig(manifest.get("entries")),
            "entry_count": len(manifest.get("entries", [])) if isinstance(manifest.get("entries"), list) else None,
        })
    for receipt_path in sorted(root.rglob("archive_receipts/*.receipt.json")):
        value = load(receipt_path)
        if value and value.get("bundle_id"):
            value = dict(value)
            value["_path"] = str(receipt_path)
            receipts.append(value)

def size(path):
    try:
        result = subprocess.run(["du", "-sb", str(path)], text=True, capture_output=True, check=False)
        return int(result.stdout.split()[0]) if result.returncode == 0 else None
    except Exception:
        return None

def usage(path):
    try:
        value = os.statvfs(path)
        return {"mount": path, "total_bytes": value.f_frsize * value.f_blocks,
                "free_bytes": value.f_frsize * value.f_bavail,
                "used_bytes": value.f_frsize * (value.f_blocks - value.f_bfree)}
    except Exception as exc:
        return {"mount": path, "error": repr(exc)}

stage2_root = run_root / "stage2"
try:
    lock_result = subprocess.run(["find", str(stage2_root), "-name", "coordinator.lock", "-print"], text=True, capture_output=True, check=False)
    locks = [line.strip() for line in lock_result.stdout.splitlines() if line.strip()]
except Exception:
    locks = []
try:
    ps = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False).stdout
    stage2_processes = [line.strip() for line in ps.splitlines() if "training.v3" in line and "stage2" in line.lower()][:30]
except Exception:
    stage2_processes = []

retention_plans = {}
plan_roots = []
if run_root.is_dir():
    plan_roots.extend(p for p in run_root.iterdir() if p.is_dir() and p.name != "stage2")
if cold.is_dir():
    plan_roots.extend(p.parent for p in cold.rglob("resolved_config.json"))

def lightweight_retention_plan(plan_root):
    """List retention candidates without hashing; deletion still needs plan_prune."""
    config = load(plan_root / "resolved_config.json") or {}
    storage = config.get("runtime", {}).get("storage", {})
    keep = {
        "checkpoints": int(storage.get("keep_checkpoints", 3)),
        "accepted": int(storage.get("keep_accepted", 2)),
        "rejected": int(storage.get("keep_rejected", 1)),
    }
    protected = {"run_manifest.json", "resolved_config.json", "manifests/latest_generation.json"}
    generation_dir = plan_root / "manifests" / "generations"
    commits = sorted(generation_dir.glob("g*.json"), key=lambda p: p.name, reverse=True)[:max(1, keep["checkpoints"])]
    for commit_path in commits:
        commit = load(commit_path) or {}
        protected.add(str(commit_path.relative_to(plan_root)).replace(os.sep, "/"))
        for key in ("checkpoint", "audit_index", "accepted_model_path", "candidate_path"):
            if commit.get(key):
                protected.add(str(commit[key]))
        for row in commit.get("replay_shards", []):
            if isinstance(row, dict) and row.get("path"):
                shard = str(row["path"])
                protected.update({shard, shard.removesuffix(".npz") + ".manifest.json", shard.removesuffix(".npz") + ".ready.json"})
    for directory, count in keep.items():
        paths = sorted((plan_root / directory).glob("*.pt"), key=lambda p: p.name, reverse=True)[:count]
        protected.update(str(p.relative_to(plan_root)).replace(os.sep, "/") for p in paths)
    candidates = []
    for path in plan_root.rglob("*"):
        if not path.is_file() or path.name.startswith(".") or path.name.endswith((".partial", ".tmp")):
            continue
        relative = str(path.relative_to(plan_root)).replace(os.sep, "/")
        if relative.startswith("archive_staging/") or relative.startswith("archive_receipts/"):
            continue
        if not (relative.startswith("replay/raw/") or relative.startswith("checkpoints/") or relative.startswith("accepted/") or relative.startswith("rejected/") or relative.startswith("samples/")):
            continue
        if relative in protected:
            continue
        candidates.append({"path": relative, "size_bytes": path.stat().st_size, "checksum_sha256": None,
                           "action": "preliminary_candidate_requires_hash_and_receipt_gate"})
    return {"mode": "metadata_only", "keep_counts": keep, "protected_path_count": len(protected),
            "candidate_paths": candidates, "eligible_bytes_pre_hash": sum(int(x["size_bytes"]) for x in candidates),
            "deletion_enabled": False}

for plan_root in sorted(set(plan_roots)):
    if "smoke" in plan_root.name.lower():
        continue
    if (plan_root / "manifests" / "coordinator.lock").exists():
        retention_plans[str(plan_root)] = {"skipped": "coordinator_lock_present"}
    elif (plan_root / "resolved_config.json").is_file():
        try:
            retention_plans[str(plan_root)] = lightweight_retention_plan(plan_root)
        except Exception as exc:
            retention_plans[str(plan_root)] = {"error": repr(exc)}

print(json.dumps({
    "records": records,
    "receipts": receipts,
    "disk": [usage("/"), usage("/root/autodl-tmp"), usage("/autodl-pub")],
    "sizes": {"repo_runs": size(run_root), "cold_storage": size(cold), "orphans": size(orphans)},
    "stage2_locks": locks,
    "stage2_root_exists": stage2_root.is_dir(),
    "stage2_lock_exists": bool(locks),
    "stage2_processes": stage2_processes,
    "retention_plans": retention_plans,
}, ensure_ascii=False))
'''


def remote_inventory(args: argparse.Namespace) -> dict[str, Any]:
    encoded = base64.b64encode(REMOTE_SCRIPT.encode("utf-8")).decode("ascii")
    command = "import base64,sys; exec(compile(base64.b64decode(sys.argv[1]), '<audit>', 'exec'))"
    remote_command = " ".join(
        shlex.quote(value)
        for value in (
            args.remote_python,
            "-c",
            command,
            encoded,
            args.remote_repo,
            args.remote_cold_storage,
            args.remote_orphans,
        )
    )
    ssh = ["ssh", args.remote, remote_command]
    result = subprocess.run(ssh, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"read-only SSH inventory failed ({result.returncode}): {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"remote inventory was not JSON: {result.stdout[-1000:]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("remote inventory has unexpected shape")
    return value


def _first_or_none(values: Iterable[Any]) -> Any:
    for value in values:
        return value


def reconcile(local: dict[str, Any], remote: dict[str, Any], clues: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in local.get("records", []):
        grouped[str(row["bundle_id"])].append(row)
    for row in remote.get("records", []):
        grouped[str(row["bundle_id"])].append(row)

    remote_receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in remote.get("receipts", []):
        if receipt.get("bundle_id"):
            remote_receipts[str(receipt["bundle_id"])].append(receipt)

    output: list[dict[str, Any]] = []
    for bundle_id in sorted(grouped):
        rows = grouped[bundle_id]
        local_rows = [r for r in rows if r.get("source") == "local"]
        remote_rows = [r for r in rows if r.get("source") == "remote"]
        archive_shas = {r.get("archive_sha256") for r in rows if r.get("archive_sha256")}
        archive_sizes = {r.get("archive_size_bytes") for r in rows if r.get("archive_size_bytes") is not None}
        manifest_shas = {r.get("manifest_sha256") for r in rows if r.get("manifest_sha256")}
        conflict_reasons: list[str] = []
        if len(archive_shas) > 1:
            conflict_reasons.append("archive_sha256_differs")
        if len(archive_sizes) > 1:
            conflict_reasons.append("archive_size_bytes_differs")
        if len(manifest_shas) > 1:
            conflict_reasons.append("manifest_sha256_differs")
        historical = historical_match(bundle_id, clues)
        local_row = _first_or_none(local_rows)
        remote_row = _first_or_none(remote_rows)
        local_receipts = (local_row or {}).get("receipts", [])
        remote_bundle_receipts = remote_receipts.get(bundle_id, [])

        local_verified = False
        if local_row:
            manifest_sha = local_row.get("manifest_sha256")
            for receipt in local_receipts:
                if (
                    receipt.get("verified") is True
                    and receipt.get("bundle_id") == bundle_id
                    and receipt.get("archive_manifest_sha256") == manifest_sha
                    and receipt.get("archive_sha256") == local_row.get("archive_sha256")
                    and entries_signature(receipt.get("entries")) == local_row.get("entries_signature")
                    and local_row.get("archive_exists")
                    and local_row.get("actual_archive_sha256") == local_row.get("archive_sha256")
                ):
                    local_verified = True
                    break
        remote_verified = any(
            r.get("verified") is True
            and r.get("archive_sha256") == (remote_row or {}).get("archive_sha256")
            and entries_signature(r.get("entries")) == (remote_row or {}).get("entries_signature")
            for r in remote_bundle_receipts
        )

        if conflict_reasons:
            status = "conflict"
        elif local_verified:
            status = "present_verified"
        elif not local_rows and remote_rows:
            status = "remote_only"
        elif historical and (local_rows or remote_rows):
            status = "historical_returned_unverified"
        elif not local_rows and not remote_rows:
            status = "materialized_only"
        else:
            status = "conflict"
            conflict_reasons.append("local_bundle_incomplete_or_unverified")
        assert status in STATUS_VALUES
        output.append(
            {
                "record_type": "bundle",
                "bundle_id": bundle_id,
                "run_id": (local_row or remote_row or {}).get("run_id"),
                "archive_sha256": _first_or_none(archive_shas),
                "archive_size_bytes": _first_or_none(archive_sizes),
                "manifest_sha256": _first_or_none(manifest_shas),
                "local_path": (local_row or {}).get("archive_path"),
                "local_manifest_path": (local_row or {}).get("manifest_path"),
                "remote_path": (remote_row or {}).get("archive_path") or (remote_row or {}).get("manifest_path"),
                "remote_manifest_path": (remote_row or {}).get("manifest_path"),
                "historical_image_seen": historical,
                "receipt_status": {
                    "local": "verified" if local_verified else ("present" if local_receipts else "missing"),
                    "remote": "verified" if remote_verified else ("present" if remote_bundle_receipts else "missing"),
                },
                "content_status": status,
                "verification_detail": {
                    "local_archive_hash_checked": bool(local_row and local_row.get("actual_archive_sha256")),
                    "local_verified": local_verified,
                    "remote_receipt_verified": remote_verified,
                    "historical_match_fragments": [f for f in sorted(clues) if f in bundle_id.lower()],
                    "conflict_reasons": conflict_reasons,
                },
            }
        )

    for item in local.get("materialized_only", []):
        output.append(
            {
                "record_type": "materialized_artifact",
                "bundle_id": f"materialized-only/{item['relative_path']}",
                "run_id": None,
                "archive_sha256": None,
                "archive_size_bytes": item["size_bytes"],
                "manifest_sha256": None,
                "local_path": item["path"],
                "remote_path": None,
                "historical_image_seen": False,
                "receipt_status": {"local": "missing", "remote": "missing"},
                "content_status": "materialized_only",
                "verification_detail": {"reason": "flat_materialized_file_not_covered_by_local_bundle_manifest"},
            }
        )
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_catalog(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")


def make_cleanup_report(records: list[dict[str, Any]], local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[str(row["content_status"])] += 1
    materialized = [row for row in records if row.get("record_type") == "materialized_artifact"]
    materialized_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"file_count": 0, "bytes": 0})
    for row in materialized:
        relative = str(row.get("bundle_id", "")).removeprefix("materialized-only/")
        group = relative.split("/", 1)[0] if relative else "<root>"
        materialized_groups[group]["file_count"] += 1
        materialized_groups[group]["bytes"] += int(row.get("archive_size_bytes") or 0)

    remote_receipt_ids = {
        str(row.get("bundle_id")) for row in records
        if row.get("record_type") == "bundle"
        and row.get("receipt_status", {}).get("remote") == "verified"
    }
    local_bundle_by_id = {
        str(row.get("bundle_id")): row
        for row in records
        if row.get("record_type") == "bundle"
    }
    remote_staging_candidates = []
    orphan_candidates = []
    cold_candidates = []
    for row in remote.get("records", []):
        if not row.get("archive_exists"):
            continue
        candidate = {
            "bundle_id": row.get("bundle_id"),
            "run_id": row.get("run_id"),
            "path": row.get("archive_path"),
            "size_bytes": row.get("archive_actual_size_bytes") or row.get("archive_size_bytes"),
            "archive_sha256": row.get("archive_sha256"),
            "manifest_sha256": row.get("manifest_sha256"),
            "receipt_verified_remote": row.get("bundle_id") in remote_receipt_ids,
            "action": "deferred_requires_local_cloud_verification_and_user_approval",
            "basis": ["remote manifest exists", "remote archive exists"],
        }
        local_record = local_bundle_by_id.get(str(row.get("bundle_id")))
        local_cloud_verified = bool(
            local_record
            and local_record.get("content_status") == "present_verified"
            and local_record.get("local_path")
            and local_record.get("receipt_status", {}).get("local") == "verified"
            and local_record.get("receipt_status", {}).get("remote") == "verified"
            and local_record.get("archive_sha256") == row.get("archive_sha256")
            and int(local_record.get("archive_size_bytes") or 0)
            == int(row.get("archive_size_bytes") or 0)
        )
        if local_cloud_verified:
            candidate["local_cloud_verified"] = True
            candidate["basis"].extend([
                "matching local verified archive",
                "matching local and remote receipt status",
                "matching archive SHA256 and size",
            ])
            candidate["action"] = "ready_after_user_approval"
        else:
            candidate["local_cloud_verified"] = False
        if candidate["receipt_verified_remote"]:
            candidate["basis"].append("matching remote verified receipt")
        else:
            candidate["basis"].append("remote verified receipt missing; do not delete")
        path = str(candidate["path"] or "")
        if "/stage1_cold_storage/" in path:
            cold_candidates.append(candidate)
        elif "/connect4_archive_orphans/" in path:
            orphan_candidates.append(candidate)
        else:
            remote_staging_candidates.append(candidate)

    v3_retention_candidates = []
    for run_path, plan in sorted(remote.get("retention_plans", {}).items()):
        if not isinstance(plan, dict):
            continue
        for decision in plan.get("candidate_paths", []):
            v3_retention_candidates.append({
                "run_dir": run_path,
                "path": decision.get("path"),
                "size_bytes": decision.get("size_bytes"),
                "checksum_sha256": decision.get("checksum_sha256"),
                "action": "deferred_requires_plan_prune_hash_receipt_gate_and_user_approval",
                "basis": ["metadata-only V3 retention candidate", "exact plan_prune hash and receipt gate still required", "not Stage 2"],
            })

    all_targets = v3_retention_candidates + remote_staging_candidates + cold_candidates + orphan_candidates
    conditional_bytes = sum(int(row.get("size_bytes") or 0) for row in all_targets)
    locally_verified_remote_targets = [
        row for row in all_targets if row.get("action") == "ready_after_user_approval"
    ]
    locally_verified_remote_bytes = sum(
        int(row.get("size_bytes") or 0) for row in locally_verified_remote_targets
    )
    return {
        "format": "connect4-v3-stage1-cleanup-candidate-report",
        "deletion_enabled": False,
        "generated_by": "tools/audit_stage1_archive.py",
        "policy": {
            "no_delete_or_move_performed": True,
            "stage2_excluded": True,
            "public_data_excluded": True,
            "cleanup_requires_explicit_user_approval": True,
            "remote_prune_requires_receipt_gated_plan_prune_execute_prune": True,
            "external_storage_verification_required": False,
            "local_cloud_sha_receipt_verification_required": True,
        },
        "bundle_status_counts": dict(sorted(counts.items())),
        "unresolved_historical_bundles": [
            row for row in records
            if row.get("historical_image_seen") and row.get("content_status") != "present_verified"
        ],
        "materialized_only": {
            "file_count": len(materialized),
            "total_bytes": sum(int(row.get("archive_size_bytes") or 0) for row in materialized),
            "by_top_level_directory": dict(sorted(materialized_groups.items())),
            "catalog_records": "bundle_catalog.jsonl record_type=materialized_artifact",
        },
        "v3_retention_candidates": v3_retention_candidates,
        "remote_staging_candidates": remote_staging_candidates,
        "cold_storage_candidates": cold_candidates,
        "orphan_candidates": orphan_candidates,
        "cleanup_candidates": all_targets,
        "local_cloud_verified_candidates": locally_verified_remote_targets,
        "estimated_bytes_local_cloud_verified_pending_approval": locally_verified_remote_bytes,
        "estimated_bytes_if_all_deferred_targets_are_later_verified": conditional_bytes,
        "estimated_bytes_safe_to_delete_now": 0,
        "cleanup_candidate_note": "Local archive plus cloud manifest/receipt/SHA256 verification is sufficient for candidate selection. Explicit user approval is still required; V3 execute_prune remains the only deletion path for formal runs.",
        "remote_disk": remote.get("disk", []),
        "remote_sizes": remote.get("sizes", {}),
        "stage2_safety": {
            "root_exists": remote.get("stage2_root_exists", False),
            "lock_exists": remote.get("stage2_lock_exists", bool(remote.get("stage2_locks", []))),
            "locks": remote.get("stage2_locks", []),
            "processes": remote.get("stage2_processes", []),
        },
        "local_materialized_file_count": local.get("materialized_file_count", 0),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", nargs="?", default="scan")
    parser.add_argument("--local-root", type=Path, default=Path("training/runs/local_archive_validation"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--historical-clues", type=Path, default=Path("tools/stage1_historical_bundle_clues.json"))
    parser.add_argument("--remote", default="connect4_gpu_2608")
    parser.add_argument("--remote-repo", default="/root/autodl-tmp/Connect4_3D_game_refactor")
    parser.add_argument("--remote-python", default="/root/miniconda3/bin/python")
    parser.add_argument("--remote-cold-storage", default="/root/stage1_cold_storage")
    parser.add_argument("--remote-orphans", default="/root/connect4_archive_orphans")
    parser.add_argument("--no-remote", action="store_true", help="Only run the local side; useful for tests and offline inspection.")
    parser.add_argument("--hash-archives", action="store_true", help="Hash local tar files before marking present_verified.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    local_root = args.local_root.resolve()
    output = (args.output or (local_root / "catalog")).resolve()
    local = local_inventory(local_root, hash_archives=args.hash_archives)
    remote = {"records": [], "receipts": [], "disk": [], "sizes": {}, "stage2_locks": [], "stage2_processes": [], "stage2_root_exists": False, "stage2_lock_exists": False}
    if not args.no_remote:
        remote = remote_inventory(args)
    clues = load_clues(args.historical_clues.resolve() if args.historical_clues else None)
    records = reconcile(local, remote, clues)
    write_catalog(output / "bundle_catalog.jsonl", records)
    write_json(output / "remote_inventory.json", remote)
    write_json(output / "local_inventory.json", {k: v for k, v in local.items() if k != "records"})
    cleanup_report = make_cleanup_report(records, local, remote)
    write_json(output / "cleanup_candidate_report.json", cleanup_report)
    write_json(local_root / "reports" / "cleanup_candidate_report.json", cleanup_report)
    write_json(output / "catalog_metadata.json", {
        "format": "connect4-v3-stage1-bundle-catalog",
        "matching_precedence": ["bundle_id", "manifest archive_sha256", "receipt manifest SHA256 and entries", "file size hint only"],
        "historical_clues": str(args.historical_clues.resolve()) if args.historical_clues else None,
        "hash_archives": args.hash_archives,
        "remote_read_only": not args.no_remote,
        "status_values": sorted(STATUS_VALUES),
        "record_count": len(records),
    })
    write_json(local_root / "reports" / "catalog_summary.json", {
        "format": "connect4-v3-stage1-catalog-summary",
        "catalog": str((output / "bundle_catalog.jsonl").resolve()),
        "cleanup_report": str((output / "cleanup_candidate_report.json").resolve()),
        "status_counts": {
            status: sum(1 for row in records if row["content_status"] == status)
            for status in sorted(STATUS_VALUES)
        },
    })
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[row["content_status"]] += 1
    print(json.dumps({"output": str(output), "record_count": len(records), "status_counts": dict(sorted(counts.items()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
