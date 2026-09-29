"""Watch the four FLA child-2M forks, then run a one-card terminal round robin.

The parent 1M plus child 2M is cumulative exposure, not a continuous V3 run.
All six Classic edges use the same 64 paired openings and 256 simulations.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage2_fla_bal5r1_dual_gpu import DUAL, PLAN, verify_complete
from stage2_fla_bal5r1_fork4_2m import BASE, NAMES, latest_commit, read
from stage2_fla_selfplay_queue import ROOT, atomic_json, sha256
from training.v3.anchored_elo import fit_anchor_scale
from training.v3.evaluation import build_openings, load_opening_manifest, write_opening_manifest
from training.v3.evaluation_runtime import EvaluationModelSource, play_paired_openings_replicated
from training.v3.evaluation_snapshot import export_evaluation_snapshot
from training.v3.gate import summarize_paired_results

MATCH = ROOT / "training/runs/stage2/fla/direct_3m/r2_gpu1_round_robin"
STATE = MATCH / "watcher_state.json"
PHYSICAL_GPU = 1
OPENINGS = 64
SIMS = 256
WORKERS = 4
CPU_LEN = 3.3
CPUCT = 1.5
SEED = 271828
EDGES = tuple((NAMES[i], NAMES[j]) for i in range(len(NAMES)) for j in range(i + 1, len(NAMES)))
EARLY_NAMES = NAMES[:3]
EARLY_EDGES = tuple((EARLY_NAMES[i], EARLY_NAMES[j])
                    for i in range(len(EARLY_NAMES)) for j in range(i + 1, len(EARLY_NAMES)))
EARLY_STATE = MATCH / "partial_three_state.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lane_states() -> dict[str, dict]:
    return {lane: read(DUAL / f"queue_{lane}.json") for lane in ("gpu0", "gpu1")}


def verify_inputs(names: tuple[str, ...] = NAMES, *, require_all: bool = True) -> dict[str, dict]:
    plan = read(PLAN)
    states = lane_states()
    if require_all and any(state["status"] != "complete" for state in states.values()):
        raise ValueError("both serial training lanes must finish before terminal evaluation")
    rows = {row["name"]: (lane, row)
            for lane, detail in plan["lanes"].items() for row in detail["rows"]}
    if set(rows) != set(NAMES):
        raise ValueError("four-model dual-GPU plan changed")
    verified = {}
    for name in names:
        lane, row = rows[name]
        receipt = next((item for item in states[lane]["completed"] if item["name"] == name), None)
        if receipt is None:
            raise ValueError(f"missing completed lane receipt: {name}")
        actual = verify_complete(row, lane=lane)
        if receipt != actual:
            raise ValueError(f"completed lane receipt changed: {name}")
        run_dir = Path(row["run_dir"])
        pointer, commit = latest_commit(run_dir)
        if (pointer["commit_sha256"] != receipt["generation_commit_sha256"]
                or commit["checkpoint_sha256"] != receipt["terminal_sha256"]
                or commit["accepted_model_sha256"] != receipt["accepted_sha256"]):
            raise ValueError(f"terminal identity drift: {name}")
        verified[name] = {
            "name": name, "lane": lane, "run_dir": str(run_dir),
            "config": row["config"], "config_sha256": row["config_sha256"],
            "lineage_hash": row["lineage_hash"],
            "child_positions": 2_000_000, "parent_positions": 1_000_000,
            "cumulative_exposure_positions": 3_000_000,
            "generation": pointer["generation"],
            "generation_commit_sha256": pointer["commit_sha256"],
            "terminal": str(run_dir / commit["checkpoint"]),
            "terminal_sha256": commit["checkpoint_sha256"],
            "accepted": str(run_dir / commit["accepted_model_path"]),
            "accepted_sha256": commit["accepted_model_sha256"],
        }
    return verified


def selected_gpu_idle() -> bool:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20, check=True)
    selected = next(line.split(",", 1)[1].strip() for line in gpu.stdout.splitlines()
                    if line.split(",", 1)[0].strip() == str(PHYSICAL_GPU))
    return not any(line.split(",", 1)[-1].strip() == selected
                   for line in apps.stdout.splitlines() if "," in line)


def snapshots(rows: dict[str, dict]) -> None:
    folder = MATCH / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    for name in rows:
        row = rows[name]
        target = folder / f"{name}.pt"
        receipt_path = folder / f"{name}.json"
        if target.exists() != receipt_path.exists():
            raise ValueError(f"incomplete immutable snapshot: {name}")
        if target.exists():
            receipt = read(receipt_path)
            if (sha256(target) != receipt["output_sha256"]
                    or receipt["source_checkpoint_sha256"] != row["terminal_sha256"]):
                raise ValueError(f"snapshot receipt drift: {name}")
        else:
            receipt = export_evaluation_snapshot(
                row["terminal"], target, model_id=f"fla-child2m-terminal-{name}")
            if (receipt["source_checkpoint_sha256"] != row["terminal_sha256"]
                    or receipt["train_positions_consumed"] != 2_000_000):
                raise ValueError(f"snapshot training boundary mismatch: {name}")
            atomic_json(receipt_path, receipt)
        row["snapshot"] = str(target)
        row["snapshot_sha256"] = receipt["output_sha256"]


def freeze_inputs(rows: dict[str, dict]) -> None:
    path = MATCH / "inputs.json"
    payload = {"schema": "connect4-stage2-fla-terminal-four-v1",
               "created_at_utc": now(), "models": rows, "openings": OPENINGS,
               "search_sims": SIMS, "cpuct": CPUCT, "worker_processes": WORKERS,
               "physical_gpu": PHYSICAL_GPU, "cpu_latency_gate_mean_s": CPU_LEN,
               "training_interpretation": "parent 1M plus semantic-fork child 2M"}
    if path.exists():
        existing = read(path)
        existing.pop("created_at_utc", None)
        payload.pop("created_at_utc", None)
        if existing != payload:
            raise ValueError("frozen four-model inputs changed")
    else:
        atomic_json(path, payload)


def match_all(rows: dict[str, dict], *,
              edges: tuple[tuple[str, str], ...] = EDGES) -> tuple[Path, list[dict]]:
    opening_path = MATCH / "classic_openings_64.json"
    if not opening_path.exists():
        write_opening_manifest(opening_path, build_openings(
            OPENINGS, run_seed=20260926, opening_id_prefix="fla3m"))
    openings = load_opening_manifest(opening_path)
    if len(openings) != OPENINGS or any(item.rule_id != "classic" for item in openings):
        raise ValueError("opening manifest is not 64 Classic positions")
    opening_sha = sha256(opening_path)
    match_dir = MATCH / "matches"
    match_dir.mkdir(parents=True, exist_ok=True)
    matches = []
    for name_a, name_b in edges:
        row_a, row_b = rows[name_a], rows[name_b]
        output = match_dir / f"{name_a}__vs__{name_b}.json"
        expected = {"a": row_a["snapshot_sha256"], "b": row_b["snapshot_sha256"]}
        if output.exists():
            payload = read(output)
            if (payload["model_sha256"] != expected
                    or payload["opening_sha256"] != opening_sha
                    or payload["search_sims"] != SIMS
                    or len(payload["games"]) != 2 * OPENINGS):
                raise ValueError(f"existing match protocol/hash differs: {output}")
        else:
            evaluated = play_paired_openings_replicated(
                openings,
                candidate_source=EvaluationModelSource(
                    "v3_artifact", row_a["snapshot"], name_a),
                incumbent_source=EvaluationModelSource(
                    "v3_artifact", row_b["snapshot"], name_b),
                search_sims=SIMS, cpuct=CPUCT,
                worker_devices=("cuda:0",) * WORKERS,
            )
            summary = summarize_paired_results(
                evaluated.games, bootstrap_samples=2000, bootstrap_seed=SEED)
            payload = {
                "schema": "connect4-stage2-fla-3m-round-robin-edge-v1",
                "created_at_utc": now(), "model_a": name_a, "model_b": name_b,
                "model_sha256": expected, "opening_sha256": opening_sha,
                "search_sims": SIMS, "cpuct": CPUCT, "physical_gpu": PHYSICAL_GPU,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "runtime": evaluated.metrics.to_dict(),
                "games": [asdict(game) for game in evaluated.games],
                "summary": asdict(summary),
            }
            atomic_json(output, payload)
        matches.append(payload)
        atomic_json(STATE, {"status": "matching", "last_completed": str(output),
                            "completed_edges": len(matches), "updated_at_utc": now()})
    return opening_path, matches


def ratings(opening_path: Path, matches: list[dict]) -> None:
    observations = [(m["model_a"], m["model_b"], game["candidate_score"])
                    for m in matches for game in m["games"]]
    point = fit_anchor_scale(observations, model_ids=NAMES,
                             reference_model_id=NAMES[0], prior_sigma=400.0)
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name in NAMES}
    for _ in range(1000):
        sampled = []
        for match in matches:
            for index in rng.integers(0, OPENINGS, size=OPENINGS):
                for game in match["games"][2 * index:2 * index + 2]:
                    sampled.append((match["model_a"], match["model_b"],
                                    game["candidate_score"]))
        fitted = fit_anchor_scale(sampled, model_ids=NAMES,
                                  reference_model_id=NAMES[0], prior_sigma=400.0)
        for name in NAMES:
            draws[name].append(fitted[name])
    atomic_json(MATCH / "ratings.json", {
        "schema": "connect4-stage2-fla-3m-round-robin-ratings-v1",
        "created_at_utc": now(), "reference": NAMES[0],
        "search_sims": SIMS, "opening_pairs_per_edge": OPENINGS,
        "opening_sha256": sha256(opening_path),
        "ratings": {name: {
            "elo": point[name],
            "ci95": list(map(float, np.quantile(draws[name], (0.025, 0.975)))),
        } for name in NAMES},
        "edges": [{
            "a": m["model_a"], "b": m["model_b"],
            "w_d_l": [m["summary"]["overall"][key]
                      for key in ("wins", "draws", "losses")],
            "a_first": m["summary"]["candidate_as_first"],
            "a_second": m["summary"]["candidate_as_second"],
            "sha256": sha256(MATCH / "matches" /
                             f"{m['model_a']}__vs__{m['model_b']}.json"),
        } for m in matches],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    if args.poll_seconds < 30:
        parser.error("poll interval must be at least 30 seconds")
    if not args.execute:
        print(json.dumps({"models": NAMES, "edges": EDGES, "opening_pairs": OPENINGS,
                          "search_sims": SIMS, "physical_gpu": PHYSICAL_GPU,
                          "watch": args.watch}, indent=2))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError(f"matches require physical GPU{PHYSICAL_GPU} only")
    MATCH.mkdir(parents=True, exist_ok=True)
    with (MATCH / "watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                states = lane_states()
                if any(state["status"] == "failed" for state in states.values()):
                    raise RuntimeError(f"training lane failed: {states}")
                if all(state["status"] == "complete" for state in states.values()):
                    break
                if not args.watch:
                    raise RuntimeError("training lanes not complete")
                early_done = (EARLY_STATE.exists()
                              and read(EARLY_STATE).get("status") == "complete")
                completed = {item["name"] for state in states.values()
                             for item in state["completed"]}
                if not early_done and set(EARLY_NAMES).issubset(completed) and selected_gpu_idle():
                    early_rows = verify_inputs(EARLY_NAMES, require_all=False)
                    snapshots(early_rows)
                    atomic_json(EARLY_STATE, {"status": "matching", "updated_at_utc": now(),
                                              "edges": EARLY_EDGES})
                    match_all(early_rows, edges=EARLY_EDGES)
                    atomic_json(EARLY_STATE, {"status": "complete", "updated_at_utc": now(),
                                              "edges": EARLY_EDGES,
                                              "opening_sha256": sha256(MATCH / "classic_openings_64.json")})
                    continue
                atomic_json(STATE, {"status": "waiting_training", "lanes": {
                    lane: state["status"] for lane, state in states.items()},
                    "early_three_complete": early_done,
                    "updated_at_utc": now()})
                time.sleep(args.poll_seconds)
            rows = verify_inputs()
            snapshots(rows)
            freeze_inputs(rows)
            while not selected_gpu_idle():
                if not args.watch:
                    raise RuntimeError(f"physical GPU{PHYSICAL_GPU} occupied")
                atomic_json(STATE, {"status": "waiting_selected_gpu",
                                    "physical_gpu": PHYSICAL_GPU,
                                    "updated_at_utc": now()})
                time.sleep(args.poll_seconds)
            atomic_json(STATE, {"status": "matching", "updated_at_utc": now()})
            opening_path, matches = match_all(rows)
            ratings(opening_path, matches)
            atomic_json(STATE, {"status": "complete", "ratings_sha256":
                                sha256(MATCH / "ratings.json"), "updated_at_utc": now()})
        except Exception as exc:
            atomic_json(STATE, {"status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "updated_at_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
