"""Run two selected FLA architectures serially on physical GPU0.

Waits for the verified CPU/round-robin handoff, the 2026-09-27 07:30 China
time floor, and an idle physical GPU0. Each run has a bounded canary and
then resumes to exactly 2M positions. No automatic promotion is performed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_core.rules import BAL5_R2_RULE_REGISTRY  # noqa: E402
from stage2_fla_bal5r1_dual_gpu import PLAN, verify_complete  # noqa: E402
from stage2_fla_bal5r1_fork4_2m import BASE as CLASSIC_BASE, latest_commit, read  # noqa: E402
from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402
from training.v3.config import V3Config  # noqa: E402
from training.v3.pipeline import lineage_config_hash  # noqa: E402

BASE = ROOT / "training/runs/stage2/fla2_r2"
HANDOFF = BASE / "handoff"
MATCH = ROOT / "training/runs/stage2/fla/direct_3m/r2_gpu1_round_robin"
BAL = ROOT / ("training/runs/stage2/bal5/r2_pre/configs/"
              "bal5_r2_column_no_tail_classic_canary_seed271828_resume_persistent.json")
STATE = BASE / "watcher_state.json"
PLAN_PATH = BASE / "plan.json"
EARLIEST_UTC = datetime(2026, 9, 26, 23, 30, tzinfo=timezone.utc)
CANARY_POSITIONS = 80_000
TARGET_POSITIONS = 2_000_000
ACTORS_PER_GPU = 16
RULE_IDS = tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs)
RULE_CODES = {int(spec.rule_code) for spec in BAL5_R2_RULE_REGISTRY.specs}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_idle(index: int) -> bool:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20, check=True)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20, check=True)
    selected = next(line.split(",", 1)[1].strip() for line in gpu.stdout.splitlines()
                    if line.split(",", 1)[0].strip() == str(index))
    return not any(line.split(",", 1)[-1].strip() == selected
                   for line in apps.stdout.splitlines() if "," in line)


def verify_handoff() -> tuple[dict, dict]:
    marker = read(HANDOFF / "ready.json")
    selection_path = HANDOFF / "selection.json"
    cpu_path = HANDOFF / "cpu_summary.json"
    ratings_path = MATCH / "ratings.json"
    match_state = read(MATCH / "watcher_state.json")
    if marker.get("schema") != "connect4-stage2-fla2-r2-handoff-v1":
        raise ValueError("unexpected handoff schema")
    if (sha256(selection_path) != marker["selection_sha256"]
            or sha256(cpu_path) != marker["cpu_summary_sha256"]
            or sha256(ratings_path) != marker["ratings_sha256"]
            or match_state["status"] != "complete"
            or match_state["ratings_sha256"] != marker["ratings_sha256"]):
        raise ValueError("handoff artifacts or official GPU1 match have different hashes")
    selection, cpu, ratings = read(selection_path), read(cpu_path), read(ratings_path)
    selected = selection.get("selected", [])
    if (selection.get("status") != "provisional_two_architectures_not_flash_qualified"
            or len(selected) != 2 or len(set(selected)) != 2
            or selected != marker["selected"]):
        raise ValueError("handoff does not contain exactly two provisional finalists")
    if (cpu["schema"] not in {
                "connect4-stage2-fla-cpu-terminal-four-v1",
                "connect4-stage2-fla-cpu-terminal-top3-recheck-v1",
            }
            or cpu["simulations"] != 512 or cpu["repeats"] != 3
            or cpu["idle_s"] != 10 or cpu["group_idle_s"] != 60
            or cpu["cpu_gate_mean_s"] != 3.3
            or sha256(cpu_path) != selection["cpu_summary_sha256"]):
        raise ValueError("CPU handoff is not the agreed 512/3/10/60 protocol")
    if (ratings["schema"] != "connect4-stage2-fla-3m-round-robin-ratings-v1"
            or ratings["search_sims"] != 256
            or ratings["opening_pairs_per_edge"] != 64
            or len(ratings["edges"]) != 6
            or sha256(ratings_path) != selection["ratings_sha256"]):
        raise ValueError("GPU1 round-robin handoff differs from frozen protocol")
    inputs = read(MATCH / "inputs.json")
    if inputs["physical_gpu"] != 1:
        raise ValueError("formal four-model matches were not assigned to GPU1")
    if cpu["schema"] == "connect4-stage2-fla-cpu-terminal-top3-recheck-v1":
        bracket_path = HANDOFF / "bracket_summary.json"
        bracket_hash = sha256(bracket_path)
        bracket = read(bracket_path)
        if (marker.get("bracket_summary_sha256") != bracket_hash
                or cpu.get("bracket_summary_sha256") != bracket_hash
                or selection.get("bracket_summary_sha256") != bracket_hash
                or bracket["status"] != "stable_control_pass"
                or bracket["max_control_ratio"] != 1.15
                or bracket["max_bracket_ratio"] != 1.10
                or bracket["protocol"] != {
                    "simulations": 512, "repeats": 3,
                    "idle_s": 10.0, "group_idle_s": 60.0,
                    "searched_measurements_per_model": 45,
                }
                or bracket["terminal_inputs_sha256"] != sha256(MATCH / "inputs.json")
                or cpu["source_inputs_sha256"] != bracket["terminal_inputs_sha256"]):
            raise ValueError("stable top-three CPU bracket identity/protocol differs")
        old = bracket["old_control_mean_s"]
        before = bracket["control_before"]["mean_s"]
        after = bracket["control_after"]["mean_s"]
        if (before > bracket["max_control_ratio"] * old
                or after > bracket["max_control_ratio"] * old
                or max(before, after) / min(before, after) > bracket["max_bracket_ratio"]):
            raise ValueError("top-three CPU bracket controls do not pass")
        names = set(bracket["variants"])
        if (names != {row["name"] for row in cpu["results"]}
                or names != set(bracket["results"])):
            raise ValueError("top-three CPU bracket model set differs")
        for row in cpu["results"]:
            measured = bracket["results"][row["name"]]
            if (row["mean_s"] != measured["mean_s"]
                    or row["count"] != measured["count"]
                    or row["model_sha256"] != measured["artifact_sha256"]
                    or row["config_sha256"] != measured["config_sha256"]
                    or row["result_sha256"] != measured["sha256"]):
                raise ValueError(f"CPU bracket result differs: {row['name']}")
        eligible = sorted(
            (row["name"] for row in cpu["results"] if row["passes_3_3_s"]),
            key=lambda name: (-ratings["ratings"][name]["elo"], name),
        )
        if selected != eligible[:2]:
            raise ValueError("top-three finalist selection differs from GPU1 Elo order")
    cpu_rows = {row["name"]: row for row in cpu["results"]}
    ranked = {row["name"]: row for row in selection["ranking"]}
    for name in selected:
        if name not in inputs["models"] or name not in cpu_rows or name not in ranked:
            raise ValueError(f"finalist missing from match or CPU inputs: {name}")
        source = inputs["models"][name]
        if (ranked[name]["terminal_checkpoint_sha256"] != source["terminal_sha256"]
                or ranked[name]["terminal_snapshot_sha256"] != source["snapshot_sha256"]
                or cpu_rows[name]["model_sha256"] != source["snapshot_sha256"]
                or not cpu_rows[name]["passes_3_3_s"]
                or ranked[name]["cpu_mean_s"] != cpu_rows[name]["mean_s"]):
            raise ValueError(f"finalist identity differs across selection evidence: {name}")
    return selection, inputs


def prepare(selection: dict, inputs: dict) -> dict:
    if PLAN_PATH.exists():
        raise FileExistsError(f"existing FLA2-R2 plan must be inspected: {PLAN_PATH}")
    reference = read(BAL)
    dual = read(PLAN)
    source_rows = {row["name"]: (lane, row)
                   for lane, entry in dual["lanes"].items() for row in entry["rows"]}
    results = []
    for index, name in enumerate(selection["selected"]):
        source_lane, source = source_rows[name]
        actual = verify_complete(source, lane=source_lane)
        if (actual["terminal_sha256"] != inputs["models"][name]["terminal_sha256"]
                or actual["accepted_sha256"] != inputs["models"][name]["accepted_sha256"]):
            raise ValueError(f"selected Classic child receipt changed: {name}")
        run_dir = Path(source["run_dir"])
        _, commit = latest_commit(run_dir)
        accepted = run_dir / commit["accepted_model_path"]
        raw = copy.deepcopy(read(Path(source["config"])))
        accepted_payload = torch.load(accepted, map_location="cpu", weights_only=True)
        accepted_meta = accepted_payload.get("metadata", {})
        if (isinstance(accepted_meta.get("candidate_model_id"), str)
                and accepted_meta["candidate_model_id"].startswith("candidate-g")):
            warm_start = accepted
            warm_mode = "accepted_artifact_fresh_optimizer_replay_v1"
        else:
            donor = Path(raw["run"]["warm_start_checkpoint"])
            if sha256(donor) != raw["run"]["warm_start_checkpoint_sha256"]:
                raise ValueError(f"original Classic warm-start source hash changed: {name}")
            donor_payload = torch.load(donor, map_location="cpu", weights_only=True)
            donor_meta = donor_payload.get("metadata", {})
            donor_mode = raw["run"]["warm_start_mode"]
            if donor_mode == "model_only_fresh_optimizer_replay_v1":
                valid_source = (donor_meta.get("lineage") == "v3_stage2_offline"
                                and donor_meta.get("train_regime") == "standard_late")
            elif donor_mode == "accepted_artifact_fresh_optimizer_replay_v1":
                candidate = donor_meta.get("candidate_model_id")
                valid_source = (isinstance(candidate, str)
                                and candidate.startswith("candidate-g")
                                and isinstance(donor_meta.get("config_hash"), str)
                                and len(donor_meta["config_hash"]) == 64)
            else:
                valid_source = False
            if (not valid_source
                    or donor_payload.get("model_config") != accepted_payload.get("model_config")):
                raise ValueError(f"unaccepted root has no compatible prior donor: {name}")
            donor_state = donor_payload["model_state"]
            accepted_state = accepted_payload["model_state"]
            if donor_state.keys() != accepted_state.keys() or any(
                not torch.equal(donor_state[key], accepted_state[key])
                for key in donor_state
            ):
                raise ValueError(f"unaccepted root differs from its prior donor: {name}")
            warm_start = donor
            warm_mode = donor_mode
        if raw["gate"] != reference["gate"] or raw["replay"] != reference["replay"]:
            raise ValueError(f"FLA and BAL5-R2 gate/replay contract differs: {name}")
        run_id = f"fla2_r2_{name}_five_rule_seed271828"
        target_dir = BASE / "runs" / run_id
        if target_dir.exists():
            raise FileExistsError(f"FLA2-R2 run already exists: {target_dir}")
        raw["run"].update({
            "run_id": run_id, "run_dir": str(target_dir), "seed": 271828,
            "resume": False, "warm_start_checkpoint": str(warm_start),
            "warm_start_checkpoint_sha256": sha256(warm_start),
            "warm_start_mode": warm_mode,
        })
        for key in ("multi_rule_ids", "rule_registry_hash", "search_schedule",
                    "opening_temperature_mixture"):
            raw["selfplay"][key] = reference["selfplay"][key]
        raw["selfplay"]["rule_id"] = "classic"
        if tuple(raw["selfplay"]["multi_rule_ids"]) != RULE_IDS:
            raise ValueError("BAL5-R2 five-rule order changed")
        raw["learner"]["max_optimizer_steps_per_cycle"] = (
            reference["learner"]["max_optimizer_steps_per_cycle"])
        raw["runtime"].update({
            "device": "cuda:0", "selfplay_devices": ["cuda:0"],
            "evaluation_devices": ["cuda:0"], "actor_processes": ACTORS_PER_GPU,
            "mcts_lanes_per_actor": 4, "inference_batch_size": 32,
            "multi_rule_actor_pool_mode": "persistent",
        })
        config = V3Config.from_dict(raw)
        if lineage_config_hash(config) == source["lineage_hash"]:
            raise ValueError("five-rule run must form a new semantic V3 lineage")
        config_path = BASE / "configs" / f"{run_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = config.to_json()
        if config_path.exists():
            if config_path.read_text(encoding="utf-8") != encoded:
                raise FileExistsError(f"existing FLA2-R2 config differs: {config_path}")
        else:
            config_path.write_text(encoded, encoding="utf-8")
        results.append({
            "name": name, "physical_gpu": 0, "run_dir": str(target_dir),
            "run_id": run_id, "config": str(config_path),
            "config_sha256": sha256(config_path),
            "lineage_hash": lineage_config_hash(config),
            "source_classic_terminal_sha256": commit["checkpoint_sha256"],
            "source_classic_accepted_sha256": commit["accepted_model_sha256"],
            "warm_start_artifact": str(warm_start),
            "warm_start_sha256": sha256(warm_start),
            "warm_start_mode": warm_mode,
        })
    plan = {
        "schema": "connect4-stage2-fla2-r2-single-card-serial-v1",
        "created_at_utc": now(), "earliest_utc": EARLIEST_UTC.isoformat(),
        "selection_sha256": sha256(HANDOFF / "selection.json"),
        "cpu_summary_sha256": sha256(HANDOFF / "cpu_summary.json"),
        "ratings_sha256": sha256(MATCH / "ratings.json"),
        "bal5_r2_reference_sha256": sha256(BAL),
        "canary_positions": CANARY_POSITIONS,
        "target_positions": TARGET_POSITIONS,
        "rule_ids": RULE_IDS,
        "rule_registry_hash": BAL5_R2_RULE_REGISTRY.registry_hash,
        "games_per_generation": 800, "games_per_rule": 160,
        "hard_regression_tolerance": 0.05,
        "rows": results,
    }
    atomic_json(PLAN_PATH, plan)
    return plan


def verify_boundary(row: dict, positions: int, *, canary: bool) -> dict:
    run_dir = Path(row["run_dir"])
    manifest = read(run_dir / "run_manifest.json")
    consumed = int(manifest["formal_loop_state"]["train_positions_consumed"])
    if (consumed != positions or manifest.get("stop_reason") != "max_train_positions"
            or manifest["config_hash"] != row["lineage_hash"]):
        raise ValueError(f"{row['name']} boundary is not committed at {positions}: {consumed}")
    pointer, commit = latest_commit(run_dir)
    metrics_path = run_dir / "metrics/metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    commits = [item for item in metrics if item.get("stage") == "generation_commit"]
    selfplays = [item for item in metrics if item.get("stage") == "selfplay"]
    learners = [item for item in metrics if item.get("stage") == "learner"]
    if (not commits or commits[-1]["train_positions_consumed"] != positions
            or not learners or not selfplays
            or not commit["replay_shards"]):
        raise ValueError(f"{row['name']} missing committed replay or learner evidence")
    if canary:
        for item in selfplays:
            per_rule = item["actor_runtime"]["per_rule"]
            if (item["games"] != 800 or set(per_rule) != set(RULE_IDS)
                    or any(per_rule[rule]["games"] != 160 for rule in RULE_IDS)):
                raise ValueError(f"{row['name']} canary violated 160 games per rule")
        observed_codes: set[int] = set()
        for shard in commit["replay_shards"]:
            path = run_dir / shard["path"]
            with np.load(path, allow_pickle=False) as data:
                observed_codes.update(map(int, np.unique(data["rule_code"])))
        if observed_codes != RULE_CODES:
            raise ValueError(f"{row['name']} canary replay rule codes differ: {observed_codes}")
    return {
        "name": row["name"], "positions": positions,
        "generation": pointer["generation"],
        "generation_commit_sha256": pointer["commit_sha256"],
        "checkpoint_sha256": commit["checkpoint_sha256"],
        "accepted_sha256": commit["accepted_model_sha256"],
        "replay_shards": len(commit["replay_shards"]),
        "selfplay_generations": len(selfplays),
        "learner_records": len(learners),
        "config_sha256": row["config_sha256"],
        "lineage_hash": row["lineage_hash"],
    }


def run_lane(index: int) -> None:
    plan = read(PLAN_PATH)
    row = plan["rows"][index]
    if row["physical_gpu"] != 0 or sha256(Path(row["config"])) != row["config_sha256"]:
        raise ValueError("lane/config identity differs from frozen plan")
    state_path = BASE / f"queue_row{index}.json"
    if state_path.exists():
        raise FileExistsError(f"lane state already exists: {state_path}")
    atomic_json(state_path, {"status": "planning", "name": row["name"],
                             "physical_gpu": 0, "updated_at_utc": now()})
    command = [sys.executable, "-u", "-B", "-m", "training.v3", "run",
               "--config", row["config"]]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    try:
        preview = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                 text=True, timeout=120)
        if preview.returncode:
            raise RuntimeError(f"guarded V3 plan failed: {preview.stderr[-1500:]}")
        log_dir = BASE / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for phase, bound, resume in (
            ("canary", CANARY_POSITIONS, False),
            ("to2m", TARGET_POSITIONS, True),
        ):
            atomic_json(state_path, {"status": f"running_{phase}", "name": row["name"],
                                     "physical_gpu": 0, "updated_at_utc": now()})
            log_path = log_dir / f"{row['name']}_{phase}.log"
            with log_path.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    [*command, *(["--resume"] if resume else []), "--execute",
                     "--max-train-positions", str(bound)],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"V3 {phase} exited {result.returncode}: {log_path}")
            receipt = verify_boundary(row, bound, canary=not resume)
            atomic_json(BASE / "receipts" / f"{row['name']}_{phase}.json", receipt)
        atomic_json(state_path, {"status": "complete", "name": row["name"],
                                 "physical_gpu": 0, "updated_at_utc": now()})
    except Exception as exc:
        atomic_json(state_path, {"status": "failed", "name": row["name"],
                                 "physical_gpu": 0,
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "updated_at_utc": now()})
        raise


def watch_and_launch(poll_seconds: int) -> None:
    import fcntl

    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((BASE / f"queue_row{i}.json").exists() for i in (0, 1)):
            raise FileExistsError("FLA2-R2 already launched; inspect before retry")
        prepared = read(PLAN_PATH) if PLAN_PATH.exists() else None
        try:
            while True:
                if not (HANDOFF / "ready.json").exists():
                    reason = "waiting_verified_finalists"
                elif datetime.now(timezone.utc) < EARLIEST_UTC:
                    reason = "waiting_2026_09_27_0730_local"
                elif not gpu_idle(0):
                    reason = "waiting_gpu0_idle"
                elif shutil.disk_usage(BASE).free < 30 * 1024 ** 3:
                    reason = "waiting_30gib_data_volume_headroom"
                else:
                    break
                atomic_json(STATE, {"status": reason, "updated_at_utc": now()})
                time.sleep(poll_seconds)
            selection, inputs = verify_handoff()
            if prepared is None:
                plan = prepare(selection, inputs)
            else:
                plan = prepared
                if (plan["schema"] != "connect4-stage2-fla2-r2-single-card-serial-v1"
                        or plan["selection_sha256"] != sha256(HANDOFF / "selection.json")
                        or plan["cpu_summary_sha256"] != sha256(HANDOFF / "cpu_summary.json")
                        or plan["ratings_sha256"] != sha256(MATCH / "ratings.json")
                        or [row["name"] for row in plan["rows"]] != selection["selected"]
                        or any(row["physical_gpu"] != 0 or
                               sha256(Path(row["config"])) != row["config_sha256"] or
                               Path(row["run_dir"]).exists() for row in plan["rows"])):
                    raise ValueError("prepared single-card plan or run boundary differs")
            atomic_json(STATE, {"status": "prepared", "plan_sha256": sha256(PLAN_PATH),
                                "selected": selection["selected"],
                                "updated_at_utc": now()})
            for index in (0, 1):
                atomic_json(STATE, {"status": "running", "row_index": index,
                                    "plan_sha256": sha256(PLAN_PATH),
                                    "updated_at_utc": now()})
                run_lane(index)
            states = [read(BASE / f"queue_row{i}.json") for i in (0, 1)]
            if any(row["status"] != "complete" for row in states):
                raise ValueError("FLA2-R2 lane states do not both show complete")
            atomic_json(STATE, {"status": "complete", "plan_sha256": sha256(PLAN_PATH),
                                "selected": [row["name"] for row in plan["rows"]],
                                "updated_at_utc": now()})
        except Exception as exc:
            atomic_json(STATE, {"status": "failed", "updated_at_utc": now(),
                                "error": f"{type(exc).__name__}: {exc}"})
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--row-index", type=int, choices=(0, 1))
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("poll interval must be at least 30 seconds")
    if not args.execute:
        print(json.dumps({"earliest_utc": EARLIEST_UTC.isoformat(),
                          "source_handoff": str(HANDOFF / "ready.json"),
                          "physical_gpus": [0], "execution": "serial",
                          "actors_per_gpu": ACTORS_PER_GPU,
                          "rule_ids": RULE_IDS, "games_per_rule": 160,
                          "canary_positions": CANARY_POSITIONS,
                          "target_positions": TARGET_POSITIONS}, indent=2))
        return 0
    if args.row_index is not None:
        run_lane(args.row_index)
    elif args.watch:
        watch_and_launch(args.poll_seconds)
    else:
        parser.error("--execute requires --watch or --lane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
