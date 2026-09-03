"""Read-only Stage 1 archive/materialization checks before Stage 2 freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..replay import sha256_file


def _path_digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()


def _inspect_run(run_dir: Path, *, recipe_id: str, required_positions: int) -> dict[str, Any]:
    pointer_path = run_dir / "manifests/latest_generation.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    latest = int(pointer["generation"])
    commits = []
    expected: dict[str, str] = {}
    for generation in range(latest + 1):
        path = run_dir / "manifests/generations" / f"g{generation:06d}.json"
        if not path.is_file():
            raise ValueError(f"missing committed generation {generation} under {run_dir}")
        commit = json.loads(path.read_text(encoding="utf-8"))
        if int(commit.get("generation", -1)) != generation:
            raise ValueError(f"generation manifest identity mismatch: {path}")
        commits.append(commit)
        for row in commit.get("replay_shards", ()):
            relative = str(row["path"])
            checksum = str(row["checksum_sha256"])
            previous = expected.setdefault(relative, checksum)
            if previous != checksum:
                raise ValueError(f"replay checksum changed across commits: {relative}")

    receipt_entries: dict[str, str] = {}
    for path in sorted((run_dir / "archive_receipts").glob("*.receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        for row in receipt.get("entries", ()):
            relative = str(row["path"])
            checksum = str(row["checksum_sha256"])
            previous = receipt_entries.setdefault(relative, checksum)
            if previous != checksum:
                raise ValueError(f"archive receipt checksum conflict: {relative}")

    materialized = []
    receipt_backed = []
    missing = []
    for relative, checksum in sorted(expected.items()):
        path = run_dir / relative
        if path.is_file():
            if sha256_file(path) != checksum:
                raise ValueError(f"materialized replay checksum mismatch: {path}")
            materialized.append(relative)
        elif receipt_entries.get(relative) == checksum:
            receipt_backed.append(relative)
        else:
            missing.append(relative)

    selfplay_generations = set()
    metric_path = run_dir / "metrics/metrics.jsonl"
    for line in metric_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") == "selfplay" and "health" in row:
            selfplay_generations.add(int(row["generation"]))
    missing_health = sorted(set(range(latest + 1)).difference(selfplay_generations))
    cumulative_positions = int(commits[-1]["replay_cumulative_positions"])
    blockers = []
    if receipt_backed:
        blockers.append(
            f"restore {len(receipt_backed)} receipt-backed replay shards before audit/freeze"
        )
    if missing:
        blockers.append(f"recover {len(missing)} replay shards without matching local materialization/receipt")
    if missing_health:
        blockers.append(f"recover health metrics for {len(missing_health)} generations")
    if cumulative_positions < required_positions:
        blockers.append(
            f"unique raw positions {cumulative_positions} are below required {required_positions}"
        )
    return {
        "run_dir": str(run_dir),
        "run_id": str(commits[-1]["run_id"]),
        "data_recipe_id": recipe_id,
        "latest_generation": latest,
        "committed_generations": len(commits),
        "cumulative_raw_positions": cumulative_positions,
        "required_raw_positions": required_positions,
        "expected_replay_shards": len(expected),
        "materialized_replay_shards": len(materialized),
        "receipt_backed_replay_shards": len(receipt_backed),
        "uncovered_replay_shards": len(missing),
        "receipt_backed_path_digest": _path_digest(receipt_backed),
        "uncovered_path_digest": _path_digest(missing),
        "health_metric_generations": len(selfplay_generations),
        "missing_health_generations": missing_health,
        "remote_materialization_complete": not receipt_backed and not missing,
        "provenance_recoverable": not missing,
        "blockers": blockers,
    }


def preflight_stage1_sources(
    *,
    standard_run_dir: str | Path,
    mixed_run_dir: str | Path,
    standard_required_positions: int = 3_150_000,
    mixed_required_positions: int = 1_050_000,
) -> dict[str, Any]:
    standard = _inspect_run(
        Path(standard_run_dir).resolve(),
        recipe_id="b10_standard_v1",
        required_positions=standard_required_positions,
    )
    mixed = _inspect_run(
        Path(mixed_run_dir).resolve(),
        recipe_id="b10_mixed_opening_position_balanced_v1",
        required_positions=mixed_required_positions,
    )
    blockers = [
        f"{name}: {blocker}"
        for name, result in (("standard", standard), ("mixed", mixed))
        for blocker in result["blockers"]
    ]
    return {
        "schema": "connect4-v3-stage2-source-preflight-v1",
        "ready_for_remote_freeze": not blockers,
        "sources": {"standard": standard, "mixed": mixed},
        "blockers": blockers,
        "note": (
            "Archive receipts establish recoverability but do not replace materialized Replay V2 "
            "files for Stage 2 audit and freeze."
        ),
    }


__all__ = ["preflight_stage1_sources"]
