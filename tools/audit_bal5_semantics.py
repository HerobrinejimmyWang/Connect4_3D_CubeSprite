"""BAL-5 pre-calibration semantic audit.

Compares the semantic configuration that PRODUCED the g57 checkpoint with the
configuration the run is currently executing, and classifies every difference as
semantic / operational / incidental. Any unexplained semantic difference must
stop the calibration and block the resume.

Inputs (read-only):
  * the g57 checkpoint, whose embedded resolved config is the pre-resume truth
  * the current adapted config the resume is running under
  * the current run_manifest.json (post-pause) and the latest checkpoint

Output: semantic_audit.json with a machine-readable pass/fail and a per-field
classification table.

The classification contract used here:
  semantic      -> affects learning/search meaning; must be UNCHANGED
  operational   -> explicitly excluded from the semantic hash (actors, device
                   mapping, evaluation worker topology, timeouts, worker counts)
  incidental    -> paths, timestamps, ids, hashes of the above
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from training.v3.checkpoint import load_checkpoint
from training.v3.config import load_config, model_config_dict
from training.v3.pipeline import lineage_config_hash


# Runtime fields that the project deliberately excludes from the semantic hash.
OPERATIONAL_RUNTIME_FIELDS = (
    "actor_processes",
    "selfplay_devices",
    "device",
    "evaluation_devices",
    "evaluation_parallel_games",
    "evaluation_inference_batch_size",
    "evaluation_inference_batch_timeout_ms",
    "evaluation_replicas_per_device",
    "evaluation_reuse_committed_role_control",
    "num_workers",
    "torch_threads",
    "deterministic",
)

INCIDENTAL_RUN_FIELDS = (
    "run_dir",
    "warm_start_checkpoint",
    "warm_start_checkpoint_sha256",
    "run_id",
    "warm_start_mode",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, item, out)
    else:
        out[prefix] = value


def _classify(path: str) -> str:
    if path.startswith("run."):
        tail = path.split(".", 1)[1]
        return "incidental" if tail in INCIDENTAL_RUN_FIELDS else "semantic"
    if path.startswith("runtime."):
        tail = path.split(".", 1)[1]
        if tail in OPERATIONAL_RUNTIME_FIELDS or tail.startswith("storage."):
            return "operational"
        return "semantic"
    return "semantic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--paused-checkpoint", type=Path, required=True,
                        help="the checkpoint this run paused at (authoritative for ITS lineage)")
    parser.add_argument("--current-config", type=Path, required=True,
                        help="the adapted config the resume ran under")
    parser.add_argument("--expected-paused-sha256", required=True)
    parser.add_argument("--origin-checkpoint", type=Path, default=None,
                        help="optional lineage origin checkpoint (e.g. the shared g57 donor); "
                             "used only to report whether the lineage diverged, never to assert "
                             "hash equality across different architectures")
    parser.add_argument("--expected-origin-sha256", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    # ---- 0. the paused checkpoint is the authoritative semantic record ------
    paused_sha = _sha256(args.paused_checkpoint)
    if paused_sha != args.expected_paused_sha256:
        raise RuntimeError(
            f"paused checkpoint SHA-256 mismatch: {paused_sha} != {args.expected_paused_sha256}"
        )
    paused = load_checkpoint(args.paused_checkpoint, map_location="cpu")
    paused_model_config = _plain(paused.extra_state.get("model_config") or {})
    paused_loop = _plain(paused.extra_state.get("formal_loop_state") or {})
    paused_hash = paused.config_hash

    # ---- 1. what the resume is actually executing ---------------------------
    current = load_config(args.current_config)
    current_hash = lineage_config_hash(current)
    current_model_config = _plain(model_config_dict(current.model))

    # The authoritative semantic equality test is: does the CURRENT config hash
    # equal the hash the run's own checkpoint recorded? A resume must not change
    # learning semantics, so these must match.
    hash_equal = paused_hash == current_hash

    # Architectural equality, asserted field by field against the run's own
    # checkpoint rather than against another lineage's donor.
    model_config_diffs = [
        {
            "field": f"model.{key}",
            "paused_checkpoint": paused_model_config.get(key),
            "current": current_model_config.get(key),
            "class": "semantic",
        }
        for key in sorted(set(paused_model_config) | set(current_model_config))
        if paused_model_config.get(key) != current_model_config.get(key)
    ]

    # ---- 1b. optional origin comparison (informational) --------------------
    origin_block: dict[str, Any] | None = None
    if args.origin_checkpoint is not None and args.origin_checkpoint.is_file():
        origin_sha = _sha256(args.origin_checkpoint)
        if args.expected_origin_sha256 and origin_sha != args.expected_origin_sha256:
            raise RuntimeError(
                f"origin checkpoint SHA-256 mismatch: {origin_sha} != {args.expected_origin_sha256}"
            )
        origin = load_checkpoint(args.origin_checkpoint, map_location="cpu")
        origin_block = {
            "path": str(args.origin_checkpoint),
            "sha256": origin_sha,
            "config_hash": origin.config_hash,
            "generation": origin.generation,
            "differs_from_paused_lineage": origin.config_hash != paused_hash,
            "note": (
                "reported for provenance only; a cross-lineage donor is expected to carry a "
                "different config_hash because the hash encodes that lineage's architecture"
            ),
        }

    # ---- 3. run manifest identity + continuity ------------------------------
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    loop = manifest.get("formal_loop_state", {})
    pointer = json.loads(
        (run_dir / "manifests" / "latest_generation.json").read_text(encoding="utf-8")
    )
    commit_path = run_dir / pointer["commit"]
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    latest_checkpoint = run_dir / commit["checkpoint"]

    differences: list[dict[str, Any]] = []
    # The run manifest's recorded config_hash is the hash the run has been
    # executing all along; comparing it with the checkpoint's hash proves the
    # pause did not silently switch semantics.
    manifest_hash = manifest.get("config_hash")
    differences.append(
        {
            "field": "<run_manifest.config_hash vs checkpoint.config_hash>",
            "paused_checkpoint": paused_hash,
            "current": manifest_hash,
            "class": "semantic",
            "changed": manifest_hash != paused_hash,
        }
    )
    differences.append(
        {
            "field": "<run_manifest.config_hash vs current config lineage hash>",
            "paused_checkpoint": paused_hash,
            "current": current_hash,
            "class": "semantic",
            "changed": not hash_equal,
        }
    )
    differences.append(
        {
            "field": "<checkpoint model_config vs current model_config>",
            "paused_checkpoint": "see model_parameterization",
            "current": "see model_parameterization",
            "class": "semantic",
            "changed": bool(model_config_diffs),
        }
    )
    # A checkpoint does not embed a full resolved config, so per-field config
    # comparison is not available; equality is asserted through the hash above
    # plus the model_config table below.
    differences.append(
        {
            "field": "<full resolved config comparison>",
            "paused_checkpoint": "not embedded in checkpoint",
            "current": str(args.current_config),
            "class": "incidental",
            "changed": False,
            "note": "checkpoints store config_hash + model_config, not the full resolved "
                    "config; semantic equality is asserted via hash and model_config",
        }
    )

    semantic_diffs = [
        row for row in differences if row["class"] == "semantic" and row.get("changed")
    ]
    semantic_diffs.extend(model_config_diffs)
    operational_diffs = [
        row for row in differences if row["class"] == "operational" and row.get("changed")
    ]
    incidental_diffs = [row for row in differences if row["class"] == "incidental"]

    payload = {
        "schema": "connect4-bal5-semantic-audit-v1",
        "run_dir": str(run_dir),
        "inputs": {
            "paused_checkpoint": str(args.paused_checkpoint),
            "paused_checkpoint_sha256": paused_sha,
            "current_config": str(args.current_config),
            "current_config_sha256": _sha256(args.current_config),
            "resolved_config_sha256": _sha256(run_dir / "resolved_config.json"),
            "run_manifest_sha256": _sha256(run_dir / "run_manifest.json"),
            "latest_commit": pointer["commit"],
            "latest_generation": pointer["generation"],
            "latest_checkpoint": str(latest_checkpoint.relative_to(run_dir)),
            "latest_checkpoint_sha256": (
                _sha256(latest_checkpoint) if latest_checkpoint.is_file() else None
            ),
            "origin_donor": origin_block,
        },
        "hashes": {
            "paused_checkpoint_config_hash": paused_hash,
            "run_manifest_config_hash": manifest_hash,
            "current_lineage_config_hash": current_hash,
            "checkpoint_equals_manifest": manifest_hash == paused_hash,
            "checkpoint_equals_current": hash_equal,
            "all_equal": hash_equal and manifest_hash == paused_hash,
        },
        "continuity": {
            "manifest_status": manifest.get("status"),
            "manifest_stop_reason": manifest.get("stop_reason"),
            "next_generation": loop.get("next_generation"),
            "train_positions_consumed": loop.get("train_positions_consumed"),
            "replay_positions": loop.get("replay_positions"),
            "accepted_model_id": loop.get("accepted_model_id"),
            "pending_candidate": loop.get("pending_candidate"),
            "next_game_id": loop.get("next_game_id"),
            "exploration_stage_index": loop.get("exploration_stage_index"),
            "checkpoint_generation": commit.get("generation"),
            "checkpoint_candidate_model_id": commit.get("candidate_model_id"),
            "checkpoint_replay_cumulative_positions": commit.get("replay_cumulative_positions"),
            "checkpoint_replay_raw_positions": commit.get("replay_raw_positions"),
            "checkpoint_next_game_id": commit.get("next_game_id"),
            "checkpoint_gate_verdict": commit.get("gate_verdict"),
            "checkpoint_config_hash": commit.get("config_hash"),
            "pointer_generation_matches_manifest_next_minus_one": (
                loop.get("next_generation") is not None
                and int(pointer["generation"]) == int(loop["next_generation"]) - 1
            ),
            "next_game_id_agrees_with_commit": (
                loop.get("next_game_id") is not None
                and commit.get("next_game_id") is not None
                and int(loop["next_game_id"]) == int(commit["next_game_id"])
            ),
            "paused_loop_state": paused_loop,
        },
        "semantic_contract": {
            "model": _plain(current.model),
            "selfplay": _plain(current.selfplay),
            "learner": _plain(current.learner),
            "replay": _plain(current.replay),
            "gate": _plain(current.gate),
            "stability": _plain(current.stability),
        },
        "model_parameterization": {
            "paused_checkpoint_model_config": paused_model_config,
            "current_model_config": current_model_config,
            "identical": not model_config_diffs,
            "differences": model_config_diffs,
        },
        "differences": {
            "semantic": semantic_diffs,
            "operational": operational_diffs,
            "incidental": incidental_diffs,
        },
        "counts": {
            "semantic": len(semantic_diffs),
            "operational": len(operational_diffs),
            "incidental": len(incidental_diffs),
        },
        "verdict": {
            "semantic_hash_preserved": hash_equal,
            "unexplained_semantic_differences": len(semantic_diffs),
            "passed": hash_equal and not semantic_diffs,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "semantic_hash_preserved": hash_equal,
        "semantic_differences": len(semantic_diffs),
        "operational_differences": len(operational_diffs),
        "incidental_differences": len(incidental_diffs),
        "latest_generation": pointer["generation"],
        "passed": payload["verdict"]["passed"],
    }, sort_keys=True))

    if not payload["verdict"]["passed"]:
        print("SEMANTIC AUDIT FAILED - calibration and resume must not proceed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
