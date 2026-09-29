"""Launch four-worker FLA direct matches on physical GPU 0 after 1M."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2_fla_selfplay_queue import QUEUE_ROOT, ROOT, atomic_json

MATCH_ROOT = ROOT / "training/runs/stage2/fla/direct_1m/r2_four_workers"
WATCHER_STATE = MATCH_ROOT / "watcher_state.json"
LOCK = MATCH_ROOT / "watcher.lock"
PHYSICAL_GPU_INDEX = 0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_gpu_busy() -> tuple[bool, str]:
    """Check only the reserved physical card; other GPUs may have other jobs."""
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if gpu.returncode or apps.returncode:
        return True, "nvidia-smi GPU inventory failed"
    uuids = {}
    for line in gpu.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2:
            uuids[int(fields[0])] = fields[1]
    if PHYSICAL_GPU_INDEX not in uuids:
        return True, f"physical GPU {PHYSICAL_GPU_INDEX} is unavailable"
    selected_uuid = uuids[PHYSICAL_GPU_INDEX]
    pids = []
    for line in apps.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2 and fields[1] == selected_uuid:
            pids.append(fields[0])
    if pids:
        return True, f"physical GPU {PHYSICAL_GPU_INDEX} compute PIDs: {','.join(pids)}"
    return False, f"physical GPU {PHYSICAL_GPU_INDEX} idle"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("poll interval must be at least ten seconds")
    command = [sys.executable, "-u", "-B", str(ROOT / "tools/stage2_fla_post1m.py"),
               "--device", "cuda:0"]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(PHYSICAL_GPU_INDEX)}
    plan = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    if plan.returncode:
        raise RuntimeError(f"post-1M plan failed: {plan.stderr[-1000:]}")
    if not args.execute:
        print(plan.stdout)
        return 0
    MATCH_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (MATCH_ROOT / "selection.json").exists():
            raise FileExistsError("selection already exists; inspect prior controller before relaunch")
        atomic_json(WATCHER_STATE, {"status": "waiting_for_1m", "started_at_utc": now(),
                                    "physical_gpu_index": PHYSICAL_GPU_INDEX,
                                    "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                                    "continue_to_3m": False})
        try:
            consecutive_idle = 0
            while True:
                queue = json.loads((ROOT / QUEUE_ROOT / "queue_state.json").read_text(encoding="utf-8"))
                status = queue.get("status")
                if status == "failed":
                    raise RuntimeError("1M self-play queue failed; inspect its log and commit boundary")
                if status == "complete" and len(queue.get("completed", [])) == 6:
                    busy, reason = selected_gpu_busy()
                    consecutive_idle = 0 if busy else consecutive_idle + 1
                    state = "waiting_for_idle" if busy else "confirming_idle"
                    if consecutive_idle >= 2:
                        break
                else:
                    reason = f"self-play queue {status}"
                    consecutive_idle = 0
                    state = "waiting_for_1m"
                atomic_json(WATCHER_STATE, {"status": state, "last_checked_at_utc": now(),
                                            "last_reason": reason})
                time.sleep(args.poll_seconds)
            log_path = MATCH_ROOT / "post1m.log"
            atomic_json(WATCHER_STATE, {"status": "running_direct_matches", "started_at_utc": now(),
                                        "physical_gpu_index": PHYSICAL_GPU_INDEX,
                                        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
                                        "continue_to_3m": False})
            with log_path.open("a", encoding="utf-8") as stream:
                result = subprocess.run([*command, "--execute"], cwd=ROOT, env=environment,
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"post-1M controller exited {result.returncode}; see {log_path}")
            atomic_json(WATCHER_STATE, {"status": "complete_direct_matches",
                                        "completed_at_utc": now(),
                                        "continue_to_3m": False})
            return 0
        except Exception as exc:
            atomic_json(WATCHER_STATE, {"status": "failed", "failed_at_utc": now(),
                                        "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
