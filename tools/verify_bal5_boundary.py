"""Verify the paused run sits on a complete, checksum-consistent atomic boundary.

Runs after a planned drain and BEFORE any calibration or resume. Fails closed: any
missing manifest, mismatched checksum, unreleased lock or incomplete journal
blocks both calibration and resume.

Produces boundary_verification.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from training.v3.checkpoint import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # 1. manifest exists and records a safe boundary
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    check("run_manifest present", manifest_path.is_file(), str(manifest_path))
    check(
        "status is stopped_at_safe_boundary",
        manifest.get("status") == "stopped_at_safe_boundary",
        manifest.get("status"),
    )
    loop = manifest.get("formal_loop_state", {})

    # 2. pointer names a real generation commit and its checksum matches
    pointer_path = run_dir / "manifests" / "latest_generation.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8")) if pointer_path.is_file() else {}
    commit_path = run_dir / str(pointer.get("commit", ""))
    check("latest_generation pointer present", pointer_path.is_file(), str(pointer_path))
    check("pointed commit exists", commit_path.is_file(), str(commit_path))
    commit_sha_ok = False
    if commit_path.is_file() and pointer.get("commit_sha256"):
        commit_sha_ok = _sha256(commit_path) == pointer["commit_sha256"]
    check("commit checksum matches pointer", commit_sha_ok, pointer.get("commit_sha256"))
    commit = json.loads(commit_path.read_text(encoding="utf-8")) if commit_path.is_file() else {}

    # 3. pointer generation == next_generation - 1 (atomic boundary, not mid-flight)
    gen = pointer.get("generation")
    nxt = loop.get("next_generation")
    check(
        "pointer generation == next_generation - 1",
        isinstance(gen, int) and isinstance(nxt, int) and gen == nxt - 1,
        {"pointer_generation": gen, "next_generation": nxt},
    )

    # 4. the generation's checkpoint exists, loads, and agrees with the commit
    checkpoint_rel = commit.get("checkpoint")
    checkpoint_path = run_dir / str(checkpoint_rel) if checkpoint_rel else None
    check("commit names a checkpoint", bool(checkpoint_rel), checkpoint_rel)
    checkpoint_ok = False
    checkpoint_info: dict[str, Any] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            saved = load_checkpoint(checkpoint_path, map_location="cpu")
            checkpoint_info = {
                "generation": saved.generation,
                "global_step": saved.global_step,
                "config_hash": saved.config_hash,
                "accepted_model_id": saved.accepted_model_id,
                "candidate_model_id": saved.candidate_model_id,
                "sha256": _sha256(checkpoint_path),
            }
            checkpoint_ok = saved.generation == commit.get("generation")
        except Exception as error:  # noqa: BLE001 - report, never raise past the summary
            checkpoint_info = {"error": f"{type(error).__name__}: {error}"}
    check("checkpoint loads and matches commit generation", checkpoint_ok, checkpoint_info)

    # 5. coordinator lock released (no concurrent coordinator once we resume)
    lock_path = run_dir / "manifests" / "coordinator.lock"
    check("coordinator lock released", not lock_path.exists(), str(lock_path))

    # 6. no draft journal left in an uncommitted state
    drafts_dir = run_dir / "manifests" / "generation_drafts"
    drafts = sorted(p.name for p in drafts_dir.glob("*.json")) if drafts_dir.is_dir() else []
    commits = sorted(p.name for p in (run_dir / "manifests" / "generations").glob("g*.json"))
    uncommitted = [name for name in drafts if name not in commits]
    check("no uncommitted generation drafts", not uncommitted, uncommitted[:5])

    # 7. Continuity numbers. NOTE: a generation commit does not carry
    #    train_positions_consumed; it carries replay counters instead. The correct
    #    cross-check is therefore against the pre-pause checkpoint (the saved
    #    formal_loop_state) plus the commit's own replay/game-id counters.
    manifest_positions = loop.get("train_positions_consumed")
    saved_positions = None
    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            prior = load_checkpoint(checkpoint_path, map_location="cpu")
            prior_loop = prior.extra_state.get("formal_loop_state", {}) or {}
            saved_positions = prior_loop.get("train_positions_consumed")
        except Exception:  # noqa: BLE001 - already reported by the checkpoint check
            saved_positions = None
    check(
        "train_positions_consumed is a positive int",
        isinstance(manifest_positions, int) and int(manifest_positions) > 0,
        manifest_positions,
    )
    check(
        "train_positions_consumed >= paused checkpoint value",
        saved_positions is None
        or (isinstance(manifest_positions, int) and int(manifest_positions) >= int(saved_positions)),
        {"manifest": manifest_positions, "checkpoint": saved_positions},
    )
    check(
        "next_game_id agrees between manifest and commit",
        loop.get("next_game_id") is not None
        and commit.get("next_game_id") is not None
        and int(loop["next_game_id"]) == int(commit["next_game_id"]),
        {"manifest": loop.get("next_game_id"), "commit": commit.get("next_game_id")},
    )
    check(
        "commit config_hash equals checkpoint config_hash",
        checkpoint_info.get("config_hash") is not None
        and commit.get("config_hash") == checkpoint_info.get("config_hash"),
        {"commit": commit.get("config_hash"), "checkpoint": checkpoint_info.get("config_hash")},
    )
    check(
        "commit carries replay counters",
        isinstance(commit.get("replay_cumulative_positions"), int),
        {
            "replay_cumulative_positions": commit.get("replay_cumulative_positions"),
            "replay_raw_positions": commit.get("replay_raw_positions"),
        },
    )

    payload = {
        "schema": "connect4-bal5-boundary-verification-v1",
        "run_dir": str(run_dir),
        "stop_reason": manifest.get("stop_reason"),
        "manifest_updated_at": manifest.get("updated_at"),
        "latest_generation": gen,
        "next_generation": nxt,
        "train_positions_consumed": loop.get("train_positions_consumed"),
        "cumulative_raw_positions": loop.get("cumulative_raw_positions") or commit.get("cumulative_raw_positions"),
        "accepted_model_id": loop.get("accepted_model_id"),
        "producer_model_id": commit.get("producer_model_id"),
        "pending_candidate": loop.get("pending_candidate"),
        "next_game_id": loop.get("next_game_id"),
        "checkpoint": checkpoint_info,
        "checks": checks,
        "passed": all(row["ok"] for row in checks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = [row["check"] for row in checks if not row["ok"]]
    print(json.dumps({
        "passed": payload["passed"],
        "latest_generation": gen,
        "next_generation": nxt,
        "train_positions_consumed": payload["train_positions_consumed"],
        "failed_checks": failed,
    }, sort_keys=True))
    if not payload["passed"]:
        print("BOUNDARY VERIFICATION FAILED - calibration and resume must not proceed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
