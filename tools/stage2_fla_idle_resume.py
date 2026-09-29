"""Wait for an uncontended cloud host, then finish the FLA B6 1M gate.

This one-shot supervisor is deliberately conservative: a live unrelated
training controller or any CUDA compute process postpones the launch. It
never repairs an existing V3 run implicitly.
"""

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

SOURCE = ROOT / "training/runs/stage2/fla/b6_raw_2m"
STATE = SOURCE / "idle_resume_state.json"
LOCK = SOURCE / "idle_resume.lock"
GATE_ARGS = (
    "--cpu-result", str(SOURCE / "cpu_gate/cpu_result.json"),
    "--cpu-summary", str(SOURCE / "cpu_gate/cpu_summary.json"),
    "--reference-summary", str(SOURCE / "cpu_gate/reference_summary.json"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cmdlines() -> list[str]:
    lines = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        if raw:
            lines.append(raw.replace(b"\x00", b" ").decode("utf-8", "replace"))
    return lines


def host_busy() -> tuple[bool, str]:
    for line in cmdlines():
        if "ValSelection/run_method_table.py" in line or "5_DANN_mixup.py" in line:
            return True, "InfantReason training still active"
        if "training.v3 run --config" in line:
            return True, "another V3 run still active"
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode:
        return True, f"nvidia-smi unavailable: {result.stderr[-200:]}"
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if pids:
        return True, f"CUDA compute processes: {','.join(pids)}"
    return False, "idle"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("poll interval must be at least 10 seconds")
    busy, reason = host_busy()
    queue_path = ROOT / QUEUE_ROOT / "queue_state.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("status") != "awaiting_b6_cpu_gate":
        raise ValueError(f"unexpected queue state: {queue.get('status')}")
    gate = [sys.executable, "-B", str(ROOT / "tools/stage2_fla_b6_selfplay_gate.py"), *GATE_ARGS]
    if not args.execute:
        print(json.dumps({"status": "plan_only", "busy": busy, "reason": reason, "gate": gate}))
        return 0
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(STATE, {"status": "waiting_for_idle", "started_at_utc": utc_now(), "last_reason": reason})
        try:
            consecutive_idle = 0
            while consecutive_idle < 2:
                busy, reason = host_busy()
                consecutive_idle = 0 if busy else consecutive_idle + 1
                atomic_json(STATE, {"status": "waiting_for_idle", "last_checked_at_utc": utc_now(),
                                    "last_reason": reason, "consecutive_idle": consecutive_idle})
                if consecutive_idle < 2:
                    time.sleep(args.poll_seconds)
            # Validate all donor/CPU/queue preconditions immediately before launch.
            subprocess.run(gate, cwd=ROOT, check=True)
            atomic_json(STATE, {"status": "running_b6", "started_at_utc": utc_now(), "pid": os.getpid()})
            result = subprocess.run([*gate, "--execute"], cwd=ROOT)
            if result.returncode:
                raise RuntimeError(f"B6 gate exited {result.returncode}")
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            if queue.get("status") != "complete" or len(queue.get("completed", [])) != 6:
                raise RuntimeError("B6 gate exited without six verified 1M completions")
            atomic_json(STATE, {"status": "complete_1m", "completed_at_utc": utc_now(),
                                "queue_sha256": __import__("hashlib").sha256(queue_path.read_bytes()).hexdigest()})
            return 0
        except Exception as exc:
            atomic_json(STATE, {"status": "failed", "failed_at_utc": utc_now(),
                                "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
