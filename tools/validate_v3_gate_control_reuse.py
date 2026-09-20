"""Replay committed V3 gates through the role-control reuse path.

This is an operational validation tool.  It never writes into a source run;
each replay and its strict comparison are written below ``--output-root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def replay_case(run_dir: Path, generation: int, output_root: Path) -> bool:
    run_dir = run_dir.resolve()
    gate_path = run_dir / "metrics" / f"gate_g{generation:06d}.json"
    gate = _read_json(gate_path)
    config = load_config(run_dir / "resolved_config.json")
    config = replace(
        config,
        runtime=replace(config.runtime, evaluation_reuse_committed_role_control=True),
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
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    summary: list[dict[str, Any]] = []
    all_passed = True
    for raw in args.case:
        run_text, generation_text = raw.rsplit(":", 1)
        passed = replay_case(Path(run_text), int(generation_text), args.output_root)
        summary.append({"run_dir": run_text, "generation": int(generation_text), "passed": passed})
        all_passed &= passed
        if not passed:
            break
    payload = {
        "schema": "connect4-v3-gate-control-reuse-validation-summary-v1",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "all_passed": all_passed,
        "cases": summary,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
