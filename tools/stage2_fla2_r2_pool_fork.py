"""Bounded GPU1 test of the 120 x 2 x 5 five-rule self-play pool.

The fork starts from the committed g16 accepted model (1,025,352 parent
training positions) with fresh optimizer and replay. It is not a continuation
of the parent run or a 1M-exact checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402
from training.v3.config import V3Config  # noqa: E402
from training.v3.pipeline import lineage_config_hash  # noqa: E402

BASE = ROOT / "training/runs/stage2/fla2_r2/pool120x2x5_g16_gpu1"
PARENT = ROOT / ("training/runs/stage2/fla2_r2/runs/"
                 "fla2_r2_raw3d_to2d_b8_five_rule_seed271828")
PARENT_CONFIG = ROOT / ("training/runs/stage2/fla2_r2/configs/"
                        "fla2_r2_raw3d_to2d_b8_five_rule_seed271828.json")
RUN_ID = "fla2_r2_raw3d_to2d_b8_pool120x2x5_from_g16_seed271828"
RUN_DIR = BASE / "runs" / RUN_ID
CONFIG = BASE / "configs" / f"{RUN_ID}.json"
PLAN = BASE / "fork_manifest.json"
STATE = BASE / "queue_state.json"
CANARY = 80_000
TARGET = 1_000_000
PARENT_POSITIONS = 1_025_352
GAMES_PER_RULE = 240
RULES = ("classic", "p1_vertical_ignored", "p1_vertical_forbidden",
         "p1_layer0_ignored", "p1_vertical_and_layer0_ignored")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generation_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics/metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def source() -> dict:
    commit_path = PARENT / "manifests/generations/g000016.json"
    commit = read(commit_path)
    accepted = PARENT / commit["accepted_model_path"]
    checkpoint = PARENT / commit["checkpoint"]
    gate = PARENT / commit["gate_path"]
    rows = [row for row in generation_rows(PARENT)
            if row.get("stage") == "generation_commit" and row.get("generation") == 16]
    if (len(rows) != 1 or rows[0]["train_positions_consumed"] != PARENT_POSITIONS
            or rows[0]["gate_verdict"] != "accept"
            or rows[0]["accepted_model_id"] != commit["accepted_model_id"]
            or commit["generation"] != 16 or commit["gate_verdict"] != "accept"
            or sha256(accepted) != commit["accepted_model_sha256"]
            or sha256(checkpoint) != commit["checkpoint_sha256"]
            or sha256(gate) != commit["gate_sha256"]):
        raise ValueError("g16 parent commit, gate, or accepted artifact differs")
    payload = torch.load(accepted, map_location="cpu", weights_only=True)
    if payload.get("metadata", {}).get("candidate_model_id") != commit["accepted_model_id"]:
        raise ValueError("g16 accepted model metadata differs")
    return {
        "parent_run_dir": str(PARENT), "parent_generation": 16,
        "parent_train_positions": PARENT_POSITIONS,
        "parent_config_sha256": sha256(PARENT_CONFIG),
        "parent_generation_commit_sha256": sha256(commit_path),
        "parent_checkpoint_sha256": sha256(checkpoint),
        "parent_gate_sha256": sha256(gate),
        "warm_start_checkpoint": str(accepted),
        "warm_start_sha256": sha256(accepted),
        "warm_start_model_id": commit["accepted_model_id"],
    }


def prepare() -> dict:
    if PLAN.exists() or STATE.exists() or RUN_DIR.exists():
        raise FileExistsError("fork already prepared or started; inspect before retry")
    parent = source()
    raw = copy.deepcopy(read(PARENT_CONFIG))
    raw["run"].update({
        "run_id": RUN_ID, "run_dir": str(RUN_DIR), "resume": False,
        "warm_start_checkpoint": parent["warm_start_checkpoint"],
        "warm_start_checkpoint_sha256": parent["warm_start_sha256"],
        "warm_start_mode": "accepted_artifact_fresh_optimizer_replay_v1",
    })
    raw["selfplay"]["search_schedule"][0]["games"] = 1200
    raw["selfplay"]["opening_temperature_mixture"]["start_train_positions"] = 0
    raw["runtime"]["actor_processes"] = 12
    config = V3Config.from_dict(raw)
    if (len(config.selfplay.search_schedule) != 1
            or config.selfplay.search_schedule[0].games != 1200
            or tuple(config.selfplay.multi_rule_ids) != RULES
            or config.selfplay.opening_temperature_mixture.lowered_temperature_plies != 8
            or config.selfplay.opening_temperature_mixture.lowered_game_fraction != 0.5
            or config.selfplay.opening_temperature_mixture.lowered_train_position_fraction != 0.5
            or config.selfplay.opening_temperature_mixture.start_train_positions != 0
            or config.selfplay.exploration_phases[0].temperature != 1.0
            or config.selfplay.exploration_phases[1].start_ply != 28
            or config.gate.candidate_train_positions != 300_000
            or config.gate.bootstrap_candidate_train_positions != 150_000):
        raise ValueError("pool or gate contract differs from requested fork")
    parent_hash = read(PARENT / "manifests/generations/g000016.json")["config_hash"]
    child_hash = lineage_config_hash(config)
    if child_hash == parent_hash:
        raise ValueError("fork must create a new V3 semantic lineage")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(config.to_json(), encoding="utf-8")
    plan = subprocess.run([sys.executable, "-B", "-m", "training.v3", "run",
                           "--config", str(CONFIG)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    if plan.returncode:
        raise RuntimeError(f"V3 guarded plan failed: {plan.stderr[-2000:]}")
    manifest = {
        "schema": "connect4-stage2-fla2-r2-pool120x2x5-fork-v1",
        "created_at_utc": now(), **parent,
        "child_run_dir": str(RUN_DIR), "child_config": str(CONFIG),
        "child_config_sha256": sha256(CONFIG), "child_lineage_hash": child_hash,
        "parent_lineage_hash": parent_hash,
        "physical_gpu": 1, "actor_processes": 12,
        "games_per_generation": 1200, "games_per_rule": GAMES_PER_RULE,
        "games_per_rule_per_variant": 120,
        "variant_assignment": "alternating_game_id_v1",
        "child_uses_fresh_optimizer_and_replay": True,
        "gate_cadence_positions_unchanged": True,
        "canary_train_positions": CANARY,
        "child_target_train_positions": TARGET,
        "cumulative_exposure_positions": PARENT_POSITIONS + TARGET,
    }
    atomic_json(PLAN, manifest)
    return manifest


def gpu1_idle() -> bool:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, check=True, timeout=20)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                           "--format=csv,noheader,nounits"], capture_output=True,
                          text=True, check=True, timeout=20)
    uuid = next(line.split(",", 1)[1].strip() for line in gpu.stdout.splitlines()
                if line.split(",", 1)[0].strip() == "1")
    return not any(line.split(",", 1)[1].strip() == uuid
                   for line in apps.stdout.splitlines() if "," in line)


def verify_boundary(bound: int, *, canary: bool) -> dict:
    manifest = read(RUN_DIR / "run_manifest.json")
    commit_path = sorted((RUN_DIR / "manifests/generations").glob("g*.json"))[-1]
    commit = read(commit_path)
    rows = generation_rows(RUN_DIR)
    commits = [row for row in rows if row.get("stage") == "generation_commit"]
    selfplays = [row for row in rows if row.get("stage") == "selfplay"]
    learners = [row for row in rows if row.get("stage") == "learner"]
    if (manifest["formal_loop_state"]["train_positions_consumed"] != bound
            or manifest["stop_reason"] != "max_train_positions"
            or manifest["config_hash"] != read(PLAN)["child_lineage_hash"]
            or not commits or commits[-1]["train_positions_consumed"] != bound
            or not selfplays or not learners or not commit["replay_shards"]
            or sha256(RUN_DIR / commit["checkpoint"]) != commit["checkpoint_sha256"]
            or sha256(RUN_DIR / commit["accepted_model_path"]) != commit["accepted_model_sha256"]):
        raise ValueError(f"fork boundary {bound} lacks committed V3 evidence")
    for row in selfplays:
        per_rule = row["actor_runtime"]["per_rule"]
        mixture = row["health"]["opening_temperature_mixture"]["variants"]
        if (row["games"] != 1200 or set(per_rule) != set(RULES)
                or any(per_rule[rule]["games"] != GAMES_PER_RULE for rule in RULES)
                or mixture["baseline"]["games"] != 600
                or mixture["lowered_opening_temperature"]["games"] != 600
                or any((per_rule[rule]["game_id_stop"] -
                        per_rule[rule]["game_id_start"]) != GAMES_PER_RULE
                       for rule in RULES)):
            raise ValueError("fork generation violated 120 x 2 x 5 assignment")
    if canary and not any(
        row.get("sampling_group_positions", {}).get("baseline", 0) > 0
        and row.get("sampling_group_positions", {}).get("lowered_opening_temperature", 0) > 0
        for row in learners
    ):
        raise ValueError("fork learner did not consume both exploration variants")
    return {
        "schema": "connect4-stage2-fla2-r2-pool120x2x5-boundary-v1",
        "verified_at_utc": now(), "train_positions_consumed": bound,
        "generation": commit["generation"],
        "generation_commit_sha256": sha256(commit_path),
        "checkpoint_sha256": commit["checkpoint_sha256"],
        "accepted_model_sha256": commit["accepted_model_sha256"],
        "selfplay_generations": len(selfplays), "learner_records": len(learners),
        "config_sha256": sha256(CONFIG),
    }


def execute() -> None:
    if STATE.exists() or RUN_DIR.exists():
        raise FileExistsError("fork already launched; inspect before retry")
    plan = read(PLAN)
    parent = source()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
            or any(plan[key] != parent[key] for key in parent)
            or plan["child_config_sha256"] != sha256(CONFIG)
            or not gpu1_idle()
            or shutil.disk_usage(BASE).free < 30 * 1024 ** 3):
        raise ValueError("fork source/config/GPU1/disk preflight differs")
    command = [sys.executable, "-u", "-B", "-m", "training.v3", "run",
               "--config", str(CONFIG)]
    atomic_json(STATE, {"status": "starting", "updated_at_utc": now(),
                        "fork_manifest_sha256": sha256(PLAN)})
    try:
        for phase, bound, resume in (("canary", CANARY, False), ("to1m", TARGET, True)):
            atomic_json(STATE, {"status": f"running_{phase}",
                                "updated_at_utc": now(),
                                "fork_manifest_sha256": sha256(PLAN)})
            log = BASE / "logs" / f"{phase}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run([*command, *(["--resume"] if resume else []),
                                         "--execute", "--max-train-positions", str(bound)],
                                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"V3 {phase} exit {result.returncode}: {log}")
            receipt = verify_boundary(bound, canary=not resume)
            atomic_json(BASE / "receipts" / f"{phase}.json", receipt)
        atomic_json(STATE, {"status": "complete", "updated_at_utc": now(),
                            "fork_manifest_sha256": sha256(PLAN)})
    except Exception as exc:
        atomic_json(STATE, {"status": "failed", "updated_at_utc": now(),
                            "error": f"{type(exc).__name__}: {exc}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error("choose exactly one of --prepare or --execute")
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
    else:
        execute()


if __name__ == "__main__":
    main()
