"""Read-only re-audit of the Stage 2B discovery helpers after Batch 1.

Required post-migration check: both bootstrap helpers must still discover
exactly twelve verified Stage 2B runs, now from the nested layout, and the
practical-prune plan they derive must be unchanged by the reorganization
(it addresses runs by ID and by run-relative endpoint path, not by absolute
archive path).

This script never writes: it validates receipts, re-derives what
``build_practical_prune_plan.main`` would emit, and reports the result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "training" / "runs" / "stage2" / "archive"
PRUNE_PLAN = REPO / "training" / "runs" / "stage2" / "bootstrap" / "practical_prune_plan.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the JSON report here (UTF-8); otherwise print to stdout",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    readiness = importlib.import_module(
        "training.runs.stage2.bootstrap.build_primary_readiness"
    )
    prune = importlib.import_module(
        "training.runs.stage2.bootstrap.build_practical_prune_plan"
    )

    report: dict = {"schema": "stage2-stage2b-discovery-reaudit-v1"}

    # Helper 1: full validation (receipt metadata, compact manifest hash,
    # archive hash agreement, and both materialized terminal endpoints).
    discovered = readiness._stage2b_receipts()
    readiness_runs = sorted(receipt["run_id"] for _path, receipt in discovered)
    report["build_primary_readiness"] = {
        "validated_receipts": len(discovered),
        "expected": 12,
        "passed": len(discovered) == 12,
        "run_ids": readiness_runs,
        "paths": [str(path.relative_to(REPO)) for path, _ in discovered],
        "layouts": sorted(
            {"old_root" if "experiments" not in str(path) else "nested"
             for path, _ in discovered}
        ),
    }

    # Helper 2: same validation through the second helper, so the two
    # discovery implementations are proven to agree.
    prune_discovered = prune._stage2b_receipts()
    prune_runs = sorted(receipt["run_id"] for _path, receipt in prune_discovered)
    report["build_practical_prune_plan"] = {
        "validated_receipts": len(prune_discovered),
        "expected": 12,
        "passed": len(prune_discovered) == 12,
        "agrees_with_readiness_helper": prune_runs == readiness_runs,
        "layouts": sorted(
            {"old_root" if "experiments" not in str(path) else "nested"
             for path, _ in prune_discovered}
        ),
    }

    # Re-derive the prune plan content without writing it: a layout change must
    # not alter the remote-prune allowlist.
    runs = []
    for receipt_path, receipt in prune_discovered:
        keep = []
        for role, relative in receipt["endpoints"].items():
            if not relative:
                continue
            local = receipt_path.parent / "materialized" / relative
            if not local.is_file():
                raise FileNotFoundError(local)
            keep.append({"role": role, "path": relative, "sha256": sha256_file(local)})
        runs.append(
            {
                "run_id": receipt["run_id"],
                "receipt_sha256": sha256_file(receipt_path),
                "keep": keep,
            }
        )
    derived = {
        "schema": "connect4-stage2-practical-prune-plan-v1",
        "remote_repo_root": "/root/autodl-tmp/Connect4_3D_game_refactor",
        "delete_working_directories": [
            "training/runs/stage2/round1",
            "training/runs/stage2/round2",
            "training/runs/stage2/round1_incomplete_attempt",
        ],
        "selfplay_root": "training/runs/stage2/selfplay",
        "runs": runs,
        "preserve": [
            "training/runs/stage2/pools",
            "training/runs/stage2/round3",
            "all non-weight selfplay files",
            "each listed terminal checkpoint and terminal accepted model",
        ],
        "accepted_loss": "intermediate optimizer checkpoints and historical accepted/rejected/candidate weights",
    }
    on_disk = json.loads(PRUNE_PLAN.read_text(encoding="utf-8"))
    report["practical_prune_plan_unchanged_by_reorganization"] = derived == on_disk
    report["practical_prune_plan_path"] = str(PRUNE_PLAN.relative_to(REPO))
    if derived != on_disk:
        report["practical_prune_plan_diff_keys"] = sorted(
            key for key in set(derived) | set(on_disk) if derived.get(key) != on_disk.get(key)
        )

    # The root layout must be fully retired; provenance snapshots and the Elo
    # root legitimately remain.
    remaining = sorted(
        child.name
        for child in ARCHIVE.iterdir()
        if child.is_dir() and child.name.startswith("stage2b_") and "_seed" in child.name
    )
    report["remaining_root_stage2b_dirs"] = remaining
    report["root_layout_retired"] = not any(
        (ARCHIVE / name / "compact_receipt.json").is_file() for name in remaining
    )

    report["status"] = (
        "passed"
        if report["build_primary_readiness"]["passed"]
        and report["build_practical_prune_plan"]["passed"]
        and report["build_practical_prune_plan"]["agrees_with_readiness_helper"]
        and report["practical_prune_plan_unchanged_by_reorganization"]
        and report["root_layout_retired"]
        else "failed"
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(payload, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
