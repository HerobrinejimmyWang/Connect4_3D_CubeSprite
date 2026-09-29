"""Split the FLA child-2M queue across two single-card V3 lanes.

GPU0 inherits the already running gravity child, then runs full raw3d-to2d B8.
GPU1 runs thin raw3d-to2d B8 and then thin B6. Each lane is serial. CUDA
device 0 inside a lane is mapped to that lane's physical GPU by visibility.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_fla_bal5r1_fork4_2m import BASE, CHILD_POSITIONS, NAMES, STATE, latest_commit, read  # noqa: E402
from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402
from training.v3.config import V3Config  # noqa: E402
from training.v3.pipeline import lineage_config_hash  # noqa: E402

DUAL = BASE / "dual_gpu"
PLAN = DUAL / "plan.json"
LANES = {"gpu0": ("gravity_b8", "raw3d_to2d_b8"),
         "gpu1": ("raw3d_to2d_thin_b8c192", "raw3d_to2d_thin_b6c128")}
ACTORS = {"gpu0": 20, "gpu1": 16}
INHERITED_PID = 637812


def worker_alive(pid: int) -> bool:
    proc = Path(f"/proc/{pid}/stat")
    if not proc.exists():
        return False
    return proc.read_text().split()[2] != "Z"


def gpu_idle(index: int) -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr[-500:]}")
    for line in result.stdout.splitlines():
        gpu, memory = line.split(",", 1)
        if int(gpu.strip()) == index:
            return int(memory.strip().split()[0]) < 500
    raise RuntimeError(f"physical GPU{index} missing")


def wait_gpu_idle(index: int, *, hours: int = 12) -> None:
    deadline = time.monotonic() + hours * 3600
    while not gpu_idle(index):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU{index} did not become idle within {hours} hours")
        time.sleep(30)


def source_rows() -> list[dict]:
    original = read(STATE)
    manifest = BASE / "fork_manifest.json"
    if sha256(manifest) != original["fork_manifest_sha256"]:
        raise ValueError("fork manifest checksum changed")
    rows = read(manifest)["rows"]
    if [row["name"] for row in rows] != list(NAMES):
        raise ValueError("fork manifest candidate order changed")
    for row in rows:
        config = Path(row["child_config"])
        if sha256(config) != row["child_config_sha256"]:
            raise ValueError(f"source config hash changed: {row['name']}")
        if lineage_config_hash(V3Config.from_dict(read(config))) != row["child_lineage_hash"]:
            raise ValueError(f"source semantic hash changed: {row['name']}")
    return rows


def make_overlay(row: dict, *, lane: str) -> dict:
    source = Path(row["child_config"])
    raw = read(source)
    raw["runtime"]["actor_processes"] = ACTORS[lane]
    # Logical cuda:0 maps to physical GPU1 under CUDA_VISIBLE_DEVICES=1.
    raw["runtime"]["device"] = "cuda:0"
    raw["runtime"]["selfplay_devices"] = ["cuda:0"]
    raw["runtime"]["evaluation_devices"] = ["cuda:0"]
    config = V3Config.from_dict(raw)
    if lineage_config_hash(config) != row["child_lineage_hash"]:
        raise ValueError(f"operational overlay changed model lineage: {row['name']}")
    target = DUAL / "configs" / f"{row['name']}_{lane}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = config.to_json()
    if target.exists() and target.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"overlay config differs: {target}")
    if not target.exists():
        target.write_text(encoded, encoding="utf-8")
    return {"name": row["name"], "source_config_sha256": row["child_config_sha256"],
            "config": str(target), "config_sha256": sha256(target),
            "lineage_hash": row["child_lineage_hash"],
            "run_dir": row["child_run_dir"], "actor_processes": ACTORS[lane]}


def prepare() -> dict:
    if PLAN.exists():
        raise FileExistsError(f"dual-GPU plan already exists: {PLAN}")
    queue = read(STATE)
    if (queue.get("status") != "running" or queue.get("active") != "gravity_b8"
            or queue.get("completed") or queue.get("planned") != list(NAMES)):
        raise ValueError("original queue no longer at inherited gravity boundary")
    rows = source_rows()
    gravity = Path(rows[0]["child_run_dir"])
    manifest = read(gravity / "run_manifest.json")
    # A resumed V3 invocation preserves the previous safe-boundary manifest
    # status until its next commit; the live PID and coordinator lock identify
    # the active invocation here.
    if manifest.get("status") not in ("running", "stopped_at_safe_boundary") or not worker_alive(INHERITED_PID):
        raise ValueError("inherited gravity worker is not active")
    command = Path(f"/proc/{INHERITED_PID}/cmdline").read_bytes().replace(b"\x00", b" ")
    if gravity.name.encode() not in command or b"--max-train-positions 2000000" not in command:
        raise ValueError("inherited PID is not the expected bounded gravity worker")
    if not gpu_idle(1):
        raise RuntimeError("physical GPU1 is not idle")
    lane_rows = {
        "gpu0": [
            {"name": "gravity_b8", "source_config_sha256": rows[0]["child_config_sha256"],
             "config": rows[0]["child_config"], "config_sha256": rows[0]["child_config_sha256"],
             "lineage_hash": rows[0]["child_lineage_hash"],
             "run_dir": rows[0]["child_run_dir"], "actor_processes": 24,
             "inherited_worker_pid": INHERITED_PID},
            make_overlay(rows[1], lane="gpu0"),
        ],
        "gpu1": [make_overlay(rows[2], lane="gpu1"), make_overlay(rows[3], lane="gpu1")],
    }
    plan = {"schema": "connect4-stage2-fla-dual-gpu-v1",
            "source_queue_sha256": sha256(STATE),
            "source_fork_manifest_sha256": sha256(BASE / "fork_manifest.json"),
            "child_target_positions": CHILD_POSITIONS,
            "each_lane_serial": True,
            "lanes": {
                lane: {"physical_gpu": int(lane[-1]),
                       "cuda_visible_devices": lane[-1], "rows": lane_rows[lane]}
                for lane in LANES
            }}
    atomic_json(PLAN, plan)
    queue.update(status="split_dual_gpu", split_plan=str(PLAN),
                 split_plan_sha256=sha256(PLAN), lane_states={
                     lane: str(DUAL / f"queue_{lane}.json") for lane in LANES
                 })
    atomic_json(STATE, queue)
    return plan


def verify_complete(row: dict, *, lane: str) -> dict:
    run_dir = Path(row["run_dir"])
    manifest = read(run_dir / "run_manifest.json")
    positions = int(manifest["formal_loop_state"]["train_positions_consumed"])
    if manifest.get("stop_reason") != "max_train_positions" or positions != CHILD_POSITIONS:
        raise ValueError(f"{row['name']} did not reach exact 2M boundary: {positions}")
    if manifest.get("config_hash") != row["lineage_hash"]:
        raise ValueError(f"{row['name']} semantic hash differs from fork design")
    pointer, commit = latest_commit(run_dir)
    if not commit["replay_shards"] or any(
        not (run_dir / shard["path"]).is_file() for shard in commit["replay_shards"]
    ):
        raise ValueError(f"{row['name']} replay inventory incomplete")
    metrics = (run_dir / "metrics/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    commits = [json.loads(line) for line in metrics if '"stage": "generation_commit"' in line]
    if not commits or commits[-1]["train_positions_consumed"] != CHILD_POSITIONS:
        raise ValueError(f"{row['name']} learner metric does not reach 2M")
    receipt = {"name": row["name"], "lane": lane, "physical_gpu": int(lane[-1]),
               "child_positions": positions, "cumulative_exposure_positions": 3_000_000,
               "generation": pointer["generation"],
               "generation_commit_sha256": pointer["commit_sha256"],
               "terminal_sha256": commit["checkpoint_sha256"],
               "accepted_sha256": commit["accepted_model_sha256"],
               "source_config_sha256": row["source_config_sha256"],
               "active_config_sha256": row["config_sha256"],
               "lineage_hash": row["lineage_hash"]}
    atomic_json(BASE / "receipts" / f"{row['name']}_child2m.json", receipt)
    return receipt


def run_lane(lane: str) -> None:
    physical = int(lane[-1])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError(f"{lane} requires CUDA_VISIBLE_DEVICES={physical}")
    plan = read(PLAN)
    if sha256(PLAN) != read(STATE)["split_plan_sha256"]:
        raise ValueError("dual-GPU split plan checksum changed")
    rows = plan["lanes"][lane]["rows"]
    state_path = DUAL / f"queue_{lane}.json"
    if state_path.exists():
        raise FileExistsError(f"{lane} controller already started")
    state = {"schema": "connect4-stage2-fla-dual-gpu-lane-v1",
             "status": "running", "lane": lane, "physical_gpu": physical,
             "plan_sha256": sha256(PLAN), "planned": list(LANES[lane]),
             "completed": [], "active": None}
    atomic_json(state_path, state)
    try:
        for row in rows:
            name = row["name"]
            state.update(active=name)
            atomic_json(state_path, state)
            run_dir = Path(row["run_dir"])
            if "inherited_worker_pid" in row:
                pid = int(row["inherited_worker_pid"])
                deadline = time.monotonic() + 12 * 3600
                while worker_alive(pid):
                    if time.monotonic() > deadline:
                        raise TimeoutError("inherited gravity worker exceeded 12-hour wait")
                    time.sleep(30)
            else:
                if run_dir.exists():
                    raise FileExistsError(f"unstarted child run already exists: {run_dir}")
                wait_gpu_idle(physical)
                config_path = Path(row["config"])
                if sha256(config_path) != row["config_sha256"]:
                    raise ValueError(f"operational config hash changed: {name}")
                command = [sys.executable, "-u", "-B", "-m", "training.v3", "run",
                           "--config", str(config_path)]
                plan_result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                if plan_result.returncode:
                    raise RuntimeError(f"V3 guarded plan failed: {name}: {plan_result.stderr[-2000:]}")
                log = DUAL / "logs" / f"{name}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run([*command, "--execute", "--max-train-positions",
                                             str(CHILD_POSITIONS)], cwd=ROOT,
                                            stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"V3 child exited {result.returncode}: {name}: {log}")
            receipt = verify_complete(row, lane=lane)
            state["completed"].append(receipt)
            state.update(active=None)
            atomic_json(state_path, state)
        state.update(status="complete", active=None)
        atomic_json(state_path, state)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(state_path, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--lane", choices=tuple(LANES))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare and args.lane:
        parser.error("choose prepare or lane")
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
        return 0
    if args.lane:
        plan = read(PLAN)
        if not args.execute:
            print(json.dumps(plan["lanes"][args.lane], indent=2))
            return 0
        run_lane(args.lane)
        return 0
    parser.error("specify --prepare or --lane")


if __name__ == "__main__":
    raise SystemExit(main())
