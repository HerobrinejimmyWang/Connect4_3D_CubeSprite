"""Replay committed V3 gates through the role-control reuse path.

This is an operational validation tool.  It never writes into a source run;
each replay and its strict comparison are written below ``--output-root``.

The replay runs in V3's replicated evaluation mode on a single GPU: every
concurrent game gets its own worker process on ``cuda:0`` and loads its own
predictor from an ``EvaluationModelSource``.  The central-batched mode is not
usable here because this tool hands the gate model references rather than live
predictor objects.  Requested parallelism is preserved, so ``--evaluation-parallel-games``
matches the concurrency of the multi-device reference protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.v3.config import load_config
from training.v3.evaluation import load_opening_manifest
from training.v3.evaluation_runtime import EvaluationModelSource
from training.v3.gate import GateGameResult
from training.v3.pipeline import _run_sequential_gate


STRICT_KEYS = (
    "games",
    "role_control_games",
    "looks",
    "summary",
    "role_guard_mode",
    "role_control_summary",
    "role_noninferiority",
    "verdict",
    "reason",
)


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "9fcb1bfdd6154fd1e6b5f4131961e182ef67cf7e"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_artifact(run_dir: Path, relative: str, expected_sha256: str) -> Path:
    path = (run_dir / relative).resolve()
    if run_dir.resolve() not in path.parents:
        raise ValueError(f"artifact escapes run directory: {relative}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"artifact checksum mismatch: {path}: {actual}")
    return path


def _commits(run_dir: Path, through_generation: int) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "manifests" / "generations").glob("g*.json")):
        row = _read_json(path)
        if int(row["generation"]) <= through_generation:
            rows.append(row)
    return rows


def _model_sources(
    run_dir: Path, gate: dict[str, Any], generation: int
) -> tuple[EvaluationModelSource, EvaluationModelSource]:
    commits = _commits(run_dir, generation)
    current = next(row for row in commits if int(row["generation"]) == generation)
    if current.get("candidate_model_id") != gate.get("candidate_model_id"):
        raise RuntimeError("generation commit does not bind the requested candidate")
    candidate = _checked_artifact(
        run_dir, str(current["candidate_path"]), str(current["candidate_sha256"])
    )

    incumbent_id = str(gate["incumbent_model_id"])
    incumbent_commit = next(
        (
            row
            for row in reversed(commits)
            if int(row["generation"]) < generation
            and row.get("accepted_model_id") == incumbent_id
        ),
        None,
    )
    if incumbent_commit is None:
        raise RuntimeError(f"cannot resolve committed incumbent {incumbent_id}")
    incumbent = _checked_artifact(
        run_dir,
        str(incumbent_commit["accepted_model_path"]),
        str(incumbent_commit["accepted_model_sha256"]),
    )
    return (
        EvaluationModelSource("v3_artifact", str(candidate), str(gate["candidate_model_id"])),
        EvaluationModelSource("v3_artifact", str(incumbent), incumbent_id),
    )


def _replica_devices(device_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the evaluation device tuple to use for a gate replay.

    ``evaluation_devices`` rejects duplicates, so concurrency comes from
    ``evaluation_replicas_per_device`` rather than from repeating one device.
    Replicas are excluded from the semantic config hash, so the same concurrent
    game count can be reproduced on one card (1 device x 8 replicas) or on the
    multi-card topology the reference run used (2 devices x 4 replicas) without
    changing lineage. Keeping concurrent games identical is what makes the two
    machines' gate timings comparable.
    """

    if not device_names:
        raise ValueError("at least one evaluation device is required")
    for device in device_names:
        if re.fullmatch(r"cuda:\d+", device) is None:
            raise ValueError(f"invalid evaluation device name: {device}")
    if len(set(device_names)) != len(device_names):
        raise ValueError("evaluation devices cannot contain duplicates")
    return tuple(device_names)


def _replicas_per_device(parallel_games: int, device_names: tuple[str, ...]) -> int:
    devices = _replica_devices(device_names)
    if parallel_games % len(devices) != 0:
        raise ValueError(
            "evaluation parallel games must divide evenly over the evaluation devices"
        )
    return parallel_games // len(devices)


def replay_case(
    run_dir: Path,
    generation: int,
    output_root: Path,
    *,
    evaluation_parallel_games: int,
    evaluation_inference_batch_size: int,
    evaluation_inference_batch_timeout_ms: float,
    evaluation_devices: tuple[str, ...] = ("cuda:0",),
) -> bool:
    run_dir = run_dir.resolve()
    gate_path = run_dir / "metrics" / f"gate_g{generation:06d}.json"
    gate = _read_json(gate_path)
    config = load_config(run_dir / "resolved_config.json")
    # V3 keeps two mutually exclusive evaluation modes: replicated workers, which
    # load their own predictor from an ``EvaluationModelSource``, and the
    # central-batched mode, which requires live predictor objects.  This tool only
    # ever supplies sources, so it must select the replicated mode even on one
    # GPU.  Leaving ``evaluation_devices`` empty silently lands in the
    # central-batched branch, where the missing predictor fails inside the
    # inference worker.  Parallelism is preserved to stay comparable with the
    # multi-device reference protocol, where 2 devices x 4 replicas gave 8
    # concurrent games.
    replica_devices = _replica_devices(evaluation_devices)
    replicas_per_device = _replicas_per_device(evaluation_parallel_games, evaluation_devices)
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            device="cuda:0",
            evaluation_devices=replica_devices,
            evaluation_replicas_per_device=replicas_per_device,
            evaluation_parallel_games=1,
            evaluation_inference_batch_size=1,
            evaluation_inference_batch_timeout_ms=evaluation_inference_batch_timeout_ms,
            evaluation_reuse_committed_role_control=True,
        ),
    )
    opening_path = run_dir / str(gate["opening_manifest"])
    openings = list(load_opening_manifest(opening_path))
    candidate_source, incumbent_source = _model_sources(run_dir, gate, generation)
    cached_controls = [GateGameResult(**row) for row in gate["role_control_games"]]
    runtime: list[dict[str, Any]] = []

    started = time.perf_counter()
    games, controls, decision, looks = _run_sequential_gate(
        config,
        generation=generation,
        openings=openings,
        candidate_predictor=None,
        incumbent_predictor=None,
        cached_control_results=cached_controls,
        cached_control_provenance={
            "validation_source_gate": str(gate_path),
            "validation_source_gate_sha256": _sha256(gate_path),
        },
        runtime_records=runtime,
        candidate_source=candidate_source,
        incumbent_source=incumbent_source,
    )
    elapsed = time.perf_counter() - started
    replay = {
        "games": [asdict(row) for row in games],
        "role_control_games": [asdict(row) for row in controls],
        "looks": looks,
        **decision.to_dict(),
    }
    differences = {
        key: {"archived": gate.get(key), "replayed": replay.get(key)}
        for key in STRICT_KEYS
        if gate.get(key) != replay.get(key)
    }
    case_dir = output_root / run_dir.name / f"g{generation:06d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": "connect4-v3-gate-control-reuse-validation-v1",
        "run_dir": str(run_dir),
        "generation": generation,
        "source_gate": str(gate_path),
        "source_gate_sha256": _sha256(gate_path),
        "candidate": asdict(candidate_source),
        "incumbent": asdict(incumbent_source),
        "opening_manifest": str(opening_path),
        "opening_manifest_sha256": _sha256(opening_path),
        "elapsed_seconds": elapsed,
        "single_gpu_runtime": {
            "device": "cuda:0",
            "evaluation_mode": "replicated",
            "evaluation_devices": list(replica_devices),
            "evaluation_replicas_per_device": replicas_per_device,
            "evaluation_parallel_games": evaluation_parallel_games,
            "evaluation_inference_batch_size": evaluation_inference_batch_size,
            "evaluation_inference_batch_timeout_ms": evaluation_inference_batch_timeout_ms,
        },
        "strict_keys": list(STRICT_KEYS),
        "passed": not differences,
        "differences": differences,
        "evaluation_runtime": runtime,
    }
    (case_dir / "replayed_gate.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (case_dir / "comparison.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return not differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, metavar="RUN_DIR:GENERATION")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-parallel-games", type=int, default=8)
    parser.add_argument("--evaluation-inference-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-inference-batch-timeout-ms", type=float, default=1.0)
    parser.add_argument(
        "--evaluation-devices",
        default="cuda:0",
        help=(
            "comma-separated evaluation GPUs; concurrent games are split as "
            "replicas x devices (e.g. cuda:0 for 1x4090, cuda:0,cuda:1 for 2x3080Ti)"
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="continue to later cases after a mismatch instead of stopping at the first one",
    )
    args = parser.parse_args()
    if (
        args.evaluation_parallel_games < 1
        or args.evaluation_inference_batch_size < 1
        or args.evaluation_inference_batch_timeout_ms < 0.0
    ):
        raise ValueError("single-GPU evaluation runtime values are invalid")
    evaluation_devices = tuple(
        part.strip() for part in args.evaluation_devices.split(",") if part.strip()
    )
    _replica_devices(evaluation_devices)  # validate early, before any GPU work
    _replicas_per_device(args.evaluation_parallel_games, evaluation_devices)
    args.output_root.mkdir(parents=True, exist_ok=False)
    summary: list[dict[str, Any]] = []
    all_passed = True
    for raw in args.case:
        run_text, generation_text = raw.rsplit(":", 1)
        passed = replay_case(
            Path(run_text),
            int(generation_text),
            args.output_root,
            evaluation_parallel_games=args.evaluation_parallel_games,
            evaluation_inference_batch_size=args.evaluation_inference_batch_size,
            evaluation_inference_batch_timeout_ms=(
                args.evaluation_inference_batch_timeout_ms
            ),
            evaluation_devices=evaluation_devices,
        )
        summary.append({"run_dir": run_text, "generation": int(generation_text), "passed": passed})
        all_passed &= passed
        if not passed and not args.allow_partial:
            break
    payload = {
        "schema": "connect4-v3-gate-control-reuse-validation-summary-v1",
        "git_commit": _git_revision(),
        "all_passed": all_passed,
        "cases_attempted": len(summary),
        "single_gpu_runtime": {
            "device": evaluation_devices[0],
            "evaluation_mode": "replicated",
            "evaluation_devices": list(evaluation_devices),
            "evaluation_replicas_per_device": _replicas_per_device(
                args.evaluation_parallel_games, evaluation_devices
            ),
            "evaluation_parallel_games": args.evaluation_parallel_games,
            "evaluation_inference_batch_size": args.evaluation_inference_batch_size,
            "evaluation_inference_batch_timeout_ms": (
                args.evaluation_inference_batch_timeout_ms
            ),
        },
        "cases": summary,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
