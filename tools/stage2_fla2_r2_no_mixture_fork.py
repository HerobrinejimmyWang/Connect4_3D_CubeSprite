"""GPU1 near-1M fork retaining 160 games/rule and the original temperature.

Waits for the superseded 120 x 2 x 5 canary to drain at a generation boundary.
Uses the same committed g16 accepted model, with fresh optimizer and replay.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage2_fla2_r2_pool_fork as prior  # noqa: E402
from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402
from training.v3.config import V3Config  # noqa: E402
from training.v3.pipeline import lineage_config_hash  # noqa: E402

BASE = ROOT / "training/runs/stage2/fla2_r2/no_mixture_g16_gpu1"
RUN_ID = "fla2_r2_raw3d_to2d_b8_no_mixture_from_g16_seed271828"
RUN_DIR = BASE / "runs" / RUN_ID
CONFIG = BASE / "configs" / f"{RUN_ID}.json"
PLAN = BASE / "fork_manifest.json"
STATE = BASE / "queue_state.json"
CANARY = 80_000
TARGET = 1_000_000


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare() -> dict:
    if PLAN.exists() or STATE.exists() or RUN_DIR.exists():
        raise FileExistsError("no-mixture fork already prepared or started")
    source = prior.source()
    raw = copy.deepcopy(read(prior.PARENT_CONFIG))
    raw["run"].update({
        "run_id": RUN_ID, "run_dir": str(RUN_DIR), "resume": False,
        "warm_start_checkpoint": source["warm_start_checkpoint"],
        "warm_start_checkpoint_sha256": source["warm_start_sha256"],
        "warm_start_mode": "accepted_artifact_fresh_optimizer_replay_v1",
    })
    raw["selfplay"]["opening_temperature_mixture"]["enabled"] = False
    raw["selfplay"]["opening_temperature_mixture"]["start_train_positions"] = 0
    raw["runtime"]["actor_processes"] = 12
    config = V3Config.from_dict(raw)
    stage = config.selfplay.search_schedule[0]
    if (len(config.selfplay.search_schedule) != 1 or stage.games != 800
            or stage.full_search_sims != 256 or stage.fast_search_sims != 32
            or stage.full_probability != 0.5
            or tuple(config.selfplay.multi_rule_ids) != prior.RULES
            or config.selfplay.opening_temperature_mixture.enabled
            or config.selfplay.exploration_phases[0].temperature != 1.0
            or config.selfplay.exploration_phases[1].start_ply != 28
            or config.gate.bootstrap_candidate_train_positions != 150_000
            or config.gate.candidate_train_positions != 300_000):
        raise ValueError("160 x 5 no-mixture/search/gate contract differs")
    parent_hash = read(prior.PARENT / "manifests/generations/g000016.json")["config_hash"]
    child_hash = lineage_config_hash(config)
    if child_hash == parent_hash:
        raise ValueError("temperature change must create a new V3 lineage")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(config.to_json(), encoding="utf-8")
    guarded = subprocess.run([sys.executable, "-B", "-m", "training.v3", "run",
                              "--config", str(CONFIG)], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
    if guarded.returncode:
        raise RuntimeError(f"guarded V3 plan failed: {guarded.stderr[-2000:]}")
    manifest = {
        "schema": "connect4-stage2-fla2-r2-no-mixture-g16-fork-v1",
        "created_at_utc": prior.now(), **source,
        "child_run_dir": str(RUN_DIR), "child_config": str(CONFIG),
        "child_config_sha256": sha256(CONFIG),
        "child_lineage_hash": child_hash,
        "parent_lineage_hash": parent_hash,
        "superseded_experiment": str(prior.BASE),
        "physical_gpu": 1, "actor_processes": 12,
        "games_per_generation": 800, "games_per_rule": 160,
        "opening_temperature_mixture_enabled": False,
        "gate_cadence_positions_unchanged": True,
        "child_uses_fresh_optimizer_and_replay": True,
        "canary_train_positions": CANARY,
        "child_target_train_positions": TARGET,
        "cumulative_exposure_positions": prior.PARENT_POSITIONS + TARGET,
    }
    atomic_json(PLAN, manifest)
    return manifest


def old_drained() -> bool:
    old_state = prior.BASE / "queue_state.json"
    old_run = prior.RUN_DIR / "run_manifest.json"
    if not old_state.exists() or not old_run.exists():
        return False
    status = read(old_state).get("status")
    manifest = read(old_run)
    return (status == "failed"
            and manifest.get("stop_reason") == "drained_after_signal_15"
            and manifest.get("status") == "stopped_at_safe_boundary"
            and int(manifest["formal_loop_state"]["train_positions_consumed"]) > 0)


def verify_boundary(bound: int, *, canary: bool) -> dict:
    manifest = read(RUN_DIR / "run_manifest.json")
    commit_path = sorted((RUN_DIR / "manifests/generations").glob("g*.json"))[-1]
    commit = read(commit_path)
    rows = prior.generation_rows(RUN_DIR)
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
        raise ValueError(f"no-mixture boundary {bound} lacks committed V3 evidence")
    for row in selfplays:
        per_rule = row["actor_runtime"]["per_rule"]
        if (row["games"] != 800 or set(per_rule) != set(prior.RULES)
                or any(per_rule[rule]["games"] != 160 for rule in prior.RULES)
                or "opening_temperature_mixture" in row["health"]):
            raise ValueError("fork generated a different pool or temperature mixture")
    return {
        "schema": "connect4-stage2-fla2-r2-no-mixture-boundary-v1",
        "verified_at_utc": prior.now(), "train_positions_consumed": bound,
        "generation": commit["generation"],
        "generation_commit_sha256": sha256(commit_path),
        "checkpoint_sha256": commit["checkpoint_sha256"],
        "accepted_model_sha256": commit["accepted_model_sha256"],
        "selfplay_generations": len(selfplays), "learner_records": len(learners),
        "config_sha256": sha256(CONFIG),
    }


def execute() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("no-mixture fork requires physical GPU1")
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists() or RUN_DIR.exists():
            raise FileExistsError("no-mixture fork already launched")
        plan = read(PLAN)
        source = prior.source()
        if (any(plan[key] != source[key] for key in source)
                or plan["child_config_sha256"] != sha256(CONFIG)):
            raise ValueError("no-mixture source or config differs")
        deadline = time.monotonic() + 4 * 3600
        while True:
            if old_drained() and prior.gpu1_idle():
                break
            if time.monotonic() > deadline:
                raise TimeoutError("superseded fork did not drain safely within four hours")
            atomic_json(STATE, {"status": "waiting_old_generation_drain",
                                "updated_at_utc": prior.now(),
                                "fork_manifest_sha256": sha256(PLAN)})
            time.sleep(30)
        if shutil.disk_usage(BASE).free < 30 * 1024 ** 3:
            raise RuntimeError("less than 30 GiB data-volume headroom")
        command = [sys.executable, "-u", "-B", "-m", "training.v3", "run",
                   "--config", str(CONFIG)]
        try:
            for phase, bound, resume in (("canary", CANARY, False), ("to1m", TARGET, True)):
                atomic_json(STATE, {"status": f"running_{phase}",
                                    "updated_at_utc": prior.now(),
                                    "fork_manifest_sha256": sha256(PLAN)})
                log = BASE / "logs" / f"{phase}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run([*command, *(["--resume"] if resume else []),
                                             "--execute", "--max-train-positions", str(bound)],
                                            cwd=ROOT, stdout=stream,
                                            stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"V3 {phase} exit {result.returncode}: {log}")
                atomic_json(BASE / "receipts" / f"{phase}.json",
                            verify_boundary(bound, canary=not resume))
            atomic_json(STATE, {"status": "complete", "updated_at_utc": prior.now(),
                                "fork_manifest_sha256": sha256(PLAN)})
        except Exception as exc:
            atomic_json(STATE, {"status": "failed", "updated_at_utc": prior.now(),
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
