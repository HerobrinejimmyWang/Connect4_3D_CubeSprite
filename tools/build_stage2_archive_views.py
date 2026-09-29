"""Build the Stage 2 archive FLA/BAL/PRO evidence views.

The archive follows a two-layer contract:

* physical artifacts live **once**, under an experiment-lineage directory;
* FLA, BAL, and PRO are *evidence views* over those artifacts -- they never
  hold a second copy of a checkpoint, model, or result matrix.

Every entry this tool writes therefore carries both path layers:

``materialized_path``
    where the artifact actually is now, resolved against the filesystem;
``historical_origin_path``
    where the pre-reorganization report, manifest, or receipt said it was.

The historical string is never rewritten.  Reports written before the
reorganization keep quoting the old paths, and a reader needs to be able to
reconcile the two.  ``sha256`` is always recomputed from the materialized file,
so a view entry cannot describe content that is no longer there.

The tool is read-only unless ``--execute`` is given.  It does not copy, move, or
delete artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FLA_SCHEMA = "connect4-stage2-fla-view-index-v1"
BAL_SCHEMA = "connect4-stage2-bal-view-index-v1"
PRO_SCHEMA = "connect4-stage2-pro-view-index-v1"

ARCHIVE_REL = "training/runs/stage2/archive"

# Physical subtree relocations performed by the archive reorganization, as
# (historical archive-relative prefix, materialized archive-relative prefix).
# A run moved by tools/migrate_stage2_archive.py is *not* listed here: its two
# paths are read from archive/index.json, which is the owner of that mapping.
SUBTREE_MOVES: Tuple[Tuple[str, str], ...] = (
    ("round3_balance", "experiments/stage2r3/round3_balance"),
)

# FLA classification, transcribed from flash/README.md section 5.  Nothing in
# the current archive is a formal Flash qualification: the only latency sweep
# stops at 256 simulations and the deployment target is a 512-simulation CPU
# response, so no 512-simulation value may be inferred from it.
FLA_CLASS_QUALIFICATION = {
    "core": "historical_evidence",
    "extended": "historical_evidence",
    "boundary": "boundary_only",
}
FLA_LATENCY_QUALIFICATION = "latency_pending"

FLA_LINEAGE_POLICY = (
    "FLA is a non-duplicating evidence view. Original run IDs, configuration "
    "hashes, checkpoint hashes, and archive locations remain authoritative; "
    "a BAL-origin artifact stays BAL-origin here."
)

BAL_LINEAGE_POLICY = (
    "BAL remains the authoritative lineage for BAL-origin experiments. Later "
    "BAL rounds are not automatically Flash qualified."
)

PRO_LINEAGE_POLICY = (
    "PRO is a deployment/evidence view and does not promote historical "
    "artifacts by name alone."
)

BAL_GROUPS: Tuple[Dict[str, Any], ...] = (
    {"lineage": "BAL-1", "prefixes": ("bal1",), "role": "roughly-2M representation extension"},
    {"lineage": "BAL-2", "prefixes": ("bal2",), "role": "Flash-to-Balance capacity boundary"},
    {"lineage": "BAL-3", "prefixes": ("bal3",), "role": "scaled architecture evaluation"},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path, repo_root: Path) -> str:
    return os.path.relpath(str(path), str(repo_root)).replace("\\", "/")


def apply_moves(
    archive_rel: str, moves: Sequence[Tuple[str, str]] = SUBTREE_MOVES
) -> str:
    """Rewrite a historical archive-relative path to its materialized form."""

    candidate = str(archive_rel).replace("\\", "/").strip("/")
    for historical, materialized in moves:
        historical = historical.strip("/")
        if candidate == historical or candidate.startswith(historical + "/"):
            remainder = candidate[len(historical):].lstrip("/")
            return f"{materialized.strip('/')}/{remainder}" if remainder else materialized.strip("/")
    return candidate


def historical_of(
    archive_rel: str, moves: Sequence[Tuple[str, str]] = SUBTREE_MOVES
) -> Optional[str]:
    """Return the pre-move path for a materialized archive-relative path."""

    candidate = str(archive_rel).replace("\\", "/").strip("/")
    for historical, materialized in moves:
        materialized = materialized.strip("/")
        if candidate == materialized or candidate.startswith(materialized + "/"):
            remainder = candidate[len(materialized):].lstrip("/")
            return f"{historical.strip('/')}/{remainder}" if remainder else historical.strip("/")
    return None


def _fs_path(path: Path) -> str:
    """Return a filesystem-safe path, using Win32 extended syntax when needed."""

    raw = os.path.abspath(os.fspath(path))
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw[2:]
    return "\\\\?\\" + raw


def _directory_totals(path: Path) -> Tuple[int, int]:
    """Count files and bytes below ``path``.

    ``os.walk`` ignores enumeration errors unless ``onerror`` is supplied, and
    its results are silently truncated by the legacy ``MAX_PATH`` limit.  BAL-5
    contains 312-character paths, so a naive walk under-reported it by 2,196
    files.  Extended-length paths are used and any residual error is raised
    rather than reported as a smaller, wrong inventory.
    """

    errors: List[BaseException] = []
    files = 0
    total = 0
    for current, _dirs, names in os.walk(_fs_path(path), onerror=errors.append):
        for name in names:
            candidate = os.path.join(current, name)
            try:
                stat = os.stat(candidate)
            except OSError as exc:
                errors.append(exc)
                continue
            files += 1
            total += stat.st_size
    if errors:
        raise RuntimeError(
            f"incomplete inventory for {path}: {len(errors)} error(s); first={errors[0]}"
        )
    return files, total


def artifact_entry(
    archive_root: Path,
    repo_root: Path,
    historical_archive_rel: str,
    *,
    moves: Sequence[Tuple[str, str]] = SUBTREE_MOVES,
    **fields: Any,
) -> Dict[str, Any]:
    """Describe one artifact with both path layers and a verified hash.

    The entry stays truthful whether or not the planned move has happened yet:
    the materialized path is whichever location actually holds the file, and a
    not-yet-moved artifact is labelled instead of being reported as missing.
    """

    historical_rel = str(historical_archive_rel).replace("\\", "/").strip("/")
    planned_rel = apply_moves(historical_rel, moves)
    historical_path = archive_root / Path(historical_rel)
    planned_path = archive_root / Path(planned_rel)
    relocated = planned_rel != historical_rel

    if planned_path.is_file():
        path = planned_path
        move_status = "materialized_at_planned_path" if relocated else "not_moved"
    elif historical_path.is_file():
        path = historical_path
        move_status = "pending_move"
    else:
        path = planned_path
        move_status = "missing"

    entry: Dict[str, Any] = {
        "materialized_path": repo_relative(path, repo_root),
        "historical_origin_path": repo_relative(historical_path, repo_root),
        "move_status": move_status,
        "exists": path.is_file(),
    }
    if relocated:
        entry["planned_materialized_path"] = repo_relative(planned_path, repo_root)
    if path.is_file():
        entry["sha256"] = sha256_file(path)
        entry["size_bytes"] = path.stat().st_size
    entry.update(fields)
    return entry


def _run_index(archive_root: Path) -> List[Mapping[str, Any]]:
    index_path = archive_root / "index.json"
    if not index_path.is_file():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return list(index.get("runs", []))


def build_fla_view(archive_root: Path, repo_root: Path) -> Dict[str, Any]:
    """Derive the FLA view from the curated external-material registry."""

    registry_path = archive_root / "flash" / "external_materials.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = []

    for row in registry.get("owned_materials", []):
        target = (registry_path.parent / row["path"]).resolve()
        archive_rel = os.path.relpath(str(target), str(archive_root)).replace("\\", "/")
        entries.append(
            artifact_entry(
                archive_root,
                repo_root,
                archive_rel,
                fla_class="core",
                qualification_status=FLA_LATENCY_QUALIFICATION,
                qualification_limit=(
                    "16/64/256 simulations only; formal 512-simulation Flash "
                    "qualification is not measured"
                ),
                ownership="fla_view_owned",
                role=row.get("role"),
                recorded_sha256=row.get("sha256"),
            )
        )

    for row in registry.get("external_materials", []):
        target = (registry_path.parent / row["path"]).resolve()
        archive_rel = os.path.relpath(str(target), str(archive_root)).replace("\\", "/")
        fla_class = row.get("fla_class", "core")
        entry = artifact_entry(
            archive_root,
            repo_root,
            archive_rel,
            fla_class=fla_class,
            qualification_status=FLA_CLASS_QUALIFICATION.get(
                fla_class, "historical_evidence"
            ),
            qualification_limit=row.get("qualification_limit"),
            ownership="external_authoritative_elsewhere",
            origin=row.get("origin"),
            role=row.get("role"),
            recorded_sha256=row.get("sha256"),
        )
        if row.get("historical_origin_path"):
            historical = (registry_path.parent / row["historical_origin_path"]).resolve()
            entry["historical_origin_path"] = repo_relative(historical, repo_root)
            entry["move_status"] = "materialized_at_planned_path" if entry["exists"] else "missing"
            entry["planned_materialized_path"] = entry["materialized_path"]
        entries.append(entry)

    mismatches = [
        row["materialized_path"]
        for row in entries
        if row.get("exists")
        and row.get("recorded_sha256")
        and row["sha256"].lower() != str(row["recorded_sha256"]).lower()
    ]
    missing = [row["materialized_path"] for row in entries if not row.get("exists")]
    return {
        "schema": FLA_SCHEMA,
        "status": "populated",
        "lineage_policy": FLA_LINEAGE_POLICY,
        "qualification_note": (
            "No entry carries flash_qualified or flash_pareto. The 256-simulation "
            "latency sweep is historical evidence and cannot be doubled into a "
            "512-simulation result."
        ),
        "existing_view": "training/runs/stage2/archive/flash",
        "entries": sorted(entries, key=lambda row: row["materialized_path"]),
        "integrity": {
            "recorded_sha256_mismatches": mismatches,
            "missing_materialized_paths": missing,
        },
    }


def build_bal_view(archive_root: Path, repo_root: Path) -> Dict[str, Any]:
    """Describe BAL lineage groups, including those not yet physically moved."""

    entries: List[Dict[str, Any]] = []
    historical_root = archive_root / "round3_balance"
    moved_root = archive_root / "experiments" / "stage2r3" / "round3_balance"

    def _child_dirs(root: Path) -> List[Path]:
        if not root.is_dir():
            return []
        return sorted(child for child in root.iterdir() if child.is_dir())

    historical_children = _child_dirs(historical_root)
    moved_children = _child_dirs(moved_root)
    # The subtree must be enumerated from whichever location actually holds it.
    # Reading only the historical path silently drops every BAL-1/2/3 group the
    # moment the move completes; blindly preferring the moved path would drop
    # them in a partial copy.  Both populated is a conflict worth surfacing.
    conflict = bool(historical_children and moved_children)
    if moved_children and not historical_children:
        active_root, root_moved = moved_root, True
    elif historical_children:
        active_root, root_moved = historical_root, False
    else:
        active_root, root_moved = moved_root, False

    if active_root.is_dir():
        children = _child_dirs(active_root)
        for group in BAL_GROUPS:
            members = [
                child
                for child in children
                if any(child.name == p or child.name.startswith(p + "_") for p in group["prefixes"])
            ]
            if not members:
                continue
            files = total = 0
            for member in members:
                member_files, member_bytes = _directory_totals(member)
                files += member_files
                total += member_bytes
            entry = {
                "lineage": group["lineage"],
                "role": group["role"],
                "members": sorted(member.name for member in members),
                "file_count": files,
                "total_bytes": total,
                "historical_origin_path": repo_relative(historical_root, repo_root),
                "origin_lineage": "BAL",
            }
            if root_moved:
                entry["materialized_path"] = repo_relative(moved_root, repo_root)
                entry["move_status"] = "moved_atomically_under_stage2r3"
            else:
                entry["materialized_path"] = repo_relative(historical_root, repo_root)
                entry["planned_materialized_path"] = repo_relative(moved_root, repo_root)
                entry["move_status"] = "pending_atomic_move"
            entries.append(entry)

    bal5 = archive_root / "bal5"
    if bal5.is_dir():
        files, total = _directory_totals(bal5)
        entries.append(
            {
                "lineage": "BAL-5",
                "role": "final scale-up round; not yet migrated",
                "members": sorted(child.name for child in bal5.iterdir()),
                "file_count": files,
                "total_bytes": total,
                "materialized_path": repo_relative(bal5, repo_root),
                "historical_origin_path": repo_relative(bal5, repo_root),
                "planned_materialized_path": repo_relative(
                    archive_root / "experiments" / "stage2r3" / "bal5", repo_root
                ),
                "move_status": "deferred_pending_current_state_review",
                "move_blockers": [
                    "active/recent queue and watch dependencies",
                    "about 14.98 GiB (final and largest batch)",
                    "paths up to roughly 308 characters require a deep X: mapping",
                ],
                "origin_lineage": "BAL",
            }
        )

    deferred_states = {"deferred_pending_current_state_review", "pending_atomic_move"}
    status = "populated"
    if conflict:
        # A partial move is the more serious condition and must not be masked
        # by the milder "deferred" label.
        status = "conflicting_locations"
    elif any(row.get("move_status") in deferred_states for row in entries):
        status = "populated_with_deferred_groups"
    view = {
        "schema": BAL_SCHEMA,
        "status": status,
        "lineage_policy": BAL_LINEAGE_POLICY,
        "entries": entries,
    }
    if conflict:
        view["location_conflict"] = {
            "historical_origin_path": repo_relative(historical_root, repo_root),
            "moved_path": repo_relative(moved_root, repo_root),
            "note": (
                "Both the historical and the moved round3_balance locations hold "
                "subdirectories. Enumerated from the historical location; resolve "
                "the partial move before trusting either inventory."
            ),
        }
    return view


def build_pro_view() -> Dict[str, Any]:
    return {
        "schema": PRO_SCHEMA,
        "status": "planned_no_materials",
        "lineage_policy": PRO_LINEAGE_POLICY,
        "entries": [],
        "note": (
            "Reserved for later high-capacity, fixed-architecture and multi-rule "
            "design evidence. A Transformer name or a large historical checkpoint "
            "does not by itself constitute a PRO result."
        ),
    }


def build_all(archive_root: Path, repo_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "fla": build_fla_view(archive_root, repo_root),
        "bal": build_bal_view(archive_root, repo_root),
        "pro": build_pro_view(),
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--execute", action="store_true",
                        help="write the view indexes (default is a read-only preview)")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    archive_root = repo_root / Path(ARCHIVE_REL)
    views = build_all(archive_root, repo_root)
    summary = {
        name: {
            "status": view["status"],
            "entries": len(view["entries"]),
            "integrity": view.get("integrity", {}),
        }
        for name, view in views.items()
    }
    if args.execute:
        for name, view in views.items():
            _write(archive_root / "views" / name / "index.json", view)
        summary["written"] = str(archive_root / "views")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
