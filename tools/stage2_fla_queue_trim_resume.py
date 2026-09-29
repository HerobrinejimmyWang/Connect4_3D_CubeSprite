"""Safely take over the live FLA queue after removing slow unstarted variants.

The prior controller must be stopped by PID, leaving its current V3 child to
finish. This script never restarts or resumes an active run and never prunes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2_fla_selfplay_queue import (
    BAL2, NEW, POSITIONS, QUEUE_ROOT, ROOT, SEED, TEMPLATE,
    atomic_json, donor_paths, prepare_config, run_is_complete, sha256,
    validate_donor,
)

SLOW_REMOVED = (
    "column3d_v2_b8", "multiview3d_b8", "winning3d_b8",
)
ORDER = (
    ("gravity_b8", BAL2, "balance_gravity_control"),
    ("column_2d_b8c192", NEW, "column_2d_b8c192"),
    ("raw3d_to2d_b8", BAL2, "balance_raw3d_to2d"),
    ("raw3d_to2d_thin_b8c192", NEW, "raw3d_to2d_thin_b8c192"),
    ("column3d_v2_thin_b8c192", NEW, "column3d_v2_thin_b8c192"),
)
B6 = "raw3d_to2d_thin_b6c128"


def worker_running(run_id: str) -> bool:
    target = f"training.v3 run --config".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except (OSError, PermissionError):
            continue
        if target in args and f"{run_id}.json".encode() in args:
            return True
    return False


def verify_completion(run_dir: Path) -> dict:
    if not run_is_complete(run_dir):
        raise ValueError(f"V3 run has not reached the exact 1M bound: {run_dir}")
    pointer = json.loads((run_dir / "manifests/latest_generation.json").read_text(encoding="utf-8"))
    commit = run_dir / pointer["commit"]
    if sha256(commit) != pointer["commit_sha256"]:
        raise ValueError(f"generation commit hash mismatch: {commit}")
    row = json.loads(commit.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("accepted_model_path", "accepted_model_sha256"),
    ):
        artifact = run_dir / row[path_key]
        if sha256(artifact) != row[hash_key]:
            raise ValueError(f"committed model hash mismatch: {artifact}")
    metrics = (run_dir / "metrics/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    commits = [json.loads(line) for line in metrics if '"stage": "generation_commit"' in line]
    if not commits or commits[-1]["train_positions_consumed"] != POSITIONS:
        raise ValueError(f"learner commit does not reach 1M: {run_dir}")
    if not row["replay_shards"] or any(not (run_dir / shard["path"]).is_file() for shard in row["replay_shards"]):
        raise ValueError(f"committed replay inventory is incomplete: {run_dir}")
    return {
        "terminal_checkpoint": row["checkpoint"],
        "terminal_sha256": row["checkpoint_sha256"],
        "accepted_model": row["accepted_model_path"],
        "accepted_sha256": row["accepted_model_sha256"],
        "generation": pointer["generation"],
        "train_positions_consumed": POSITIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--old-controller-pid", type=int, required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"order": [x[0] for x in ORDER], "removed": SLOW_REMOVED, "b6_pending": B6}))
        return 0
    if Path(f"/proc/{args.old_controller_pid}").exists():
        raise RuntimeError("old queue controller is still alive; refusing concurrent control")
    output = ROOT / QUEUE_ROOT
    state_path = output / "queue_state.json"
    status = json.loads(state_path.read_text(encoding="utf-8"))
    if status["status"] != "running" or status["active"] not in {row[0] for row in ORDER}:
        raise ValueError(f"unexpected queue handoff state: {status['status']} {status['active']}")
    if (output / "queue_state_before_trim.json").exists():
        raise FileExistsError("queue handoff snapshot already exists")
    atomic_json(output / "queue_state_before_trim.json", status)
    status.update(
        plan_revision="cpu_mean_le_3p8_plus_b6_screen_v1",
        removed_unstarted=list(SLOW_REMOVED),
        planned=[row[0] for row in ORDER] + [B6],
        b6_gate="requires_offline_1m_and_cpu_512_mean_le_3p8; faster_than_3p0_allowed",
        controller_pid=os.getpid(),
    )
    atomic_json(state_path, status)
    template = json.loads((ROOT / TEMPLATE).read_text(encoding="utf-8"))
    names = [row[0] for row in ORDER]
    try:
        active = status["active"]
        index = names.index(active)
        completed_names = [row["name"] for row in status["completed"]]
        if completed_names != names[:index]:
            raise ValueError(f"completed queue prefix mismatch: {completed_names}")
        active_run_id = f"fla6m_r1_{active}_warm_seed{SEED}"
        active_dir = output / "runs" / active_run_id
        while not run_is_complete(active_dir):
            if not worker_running(active_run_id):
                raise RuntimeError(f"active worker exited before a verified 1M boundary: {active}")
            time.sleep(10)
        receipt = verify_completion(active_dir)
        status["completed"].append({
            "name": active, "run_dir": str(active_dir),
            "config_sha256": status["config_sha256"],
            "donor_sha256": status["donor_sha256"], **receipt,
        })
        status.update(status="waiting_donors", active=None)
        atomic_json(state_path, status)
        for name, source, variant in ORDER[index + 1:]:
            config_source, donor, report = donor_paths(source, variant)
            model, digest = validate_donor(config_source, donor, report)
            config_path, run_dir, config_digest = prepare_config(template, name, model, donor, digest)
            if run_dir.exists():
                raise RuntimeError(f"existing run directory; refusing implicit resume: {run_dir}")
            status.update(status="preflight", active=name)
            atomic_json(state_path, status)
            plan = subprocess.run(
                [sys.executable, "-B", "-m", "training.v3", "run", "--config", str(config_path)],
                cwd=ROOT, capture_output=True, text=True,
            )
            if plan.returncode:
                raise RuntimeError(f"V3 guarded plan failed for {name}: {plan.stderr[-2000:]}")
            status.update(status="running", config_sha256=config_digest, donor_sha256=digest)
            atomic_json(state_path, status)
            log_path = output / "logs" / f"{name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, "-u", "-B", "-m", "training.v3", "run", "--config", str(config_path),
                     "--execute", "--max-train-positions", str(POSITIONS)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"{name} V3 worker exited {result.returncode}: {log_path}")
            receipt = verify_completion(run_dir)
            status["completed"].append({
                "name": name, "run_dir": str(run_dir),
                "config_sha256": config_digest, "donor_sha256": digest, **receipt,
            })
            status.update(status="waiting_donors", active=None)
            atomic_json(state_path, status)
        status.update(status="awaiting_b6_cpu_gate", active=None, updated_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, status)
        return 0
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(state_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
