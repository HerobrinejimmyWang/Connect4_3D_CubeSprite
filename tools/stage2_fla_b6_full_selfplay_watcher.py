"""Handoff a CPU-passing full B6 to isolated 1M self-play after donor matches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "connect4_gpu_2608"
REMOTE = "/root/autodl-tmp/Connect4_3D_game_refactor"
BASE = ROOT / "training/runs/stage2/fla/b6_raw_full_2m"
CPU = ROOT / "training/runs/stage2/fla/cpu_b6_full_2m/idle10_group60"
NAME = "raw3d_to2d_full_b6c128"
STATE = BASE / "selfplay_watcher_state.json"
R2 = "training/runs/stage2/fla/selfplay/r2_raw_b6_full"
DIRECT = "training/runs/stage2/fla/direct_1m/r3_raw_b6_full"
PEERS = ("raw3d_to2d_thin_b6c128", "raw3d_to2d_b8", "gravity_b8")


def status(**fields: object) -> None:
    value = {"schema": "connect4-stage2-fla-b6-full-selfplay-watcher-v1",
             "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE)


def call(argv: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} exited {result.returncode}: {result.stderr[-1000:]}")
    return result


def ssh(command: str, *, timeout: int = 60) -> str:
    return call(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
                 f"cd {REMOTE} && {command}"], timeout=timeout).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-hours", type=float, default=24.0)
    args = parser.parse_args()
    if STATE.exists():
        raise FileExistsError(f"watcher already started: {STATE}")
    deadline = time.monotonic() + args.wait_hours * 3600
    status(status="waiting_cpu")
    try:
        while True:
            local = json.loads((BASE / "local_watcher_state.json").read_text(encoding="utf-8"))
            if local["status"] == "failed":
                raise RuntimeError(f"local CPU watcher failed: {local.get('error')}")
            if local["status"].startswith("complete_cpu_"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("CPU watcher did not finish")
            time.sleep(60)
        mean = float(local["cpu_mean_s"])
        if not mean < 3.1:
            status(status="skipped_cpu_mean_ge_3p1", cpu_mean_s=mean)
            return 0
        status(status="waiting_donor_matches", cpu_mean_s=mean)
        while True:
            try:
                command = " && ".join(
                    f"test -f {DIRECT}/{NAME}__vs__{peer}.json" for peer in PEERS
                )
                ssh(command, timeout=30)
                break
            except (RuntimeError, subprocess.TimeoutExpired):
                if time.monotonic() > deadline:
                    raise TimeoutError("B6 full donor matches did not complete")
                time.sleep(60)
        # Keep the local CPU machine distinct from the cloud GPU. Transfer
        # exact evidence files and let the gate validate hashes and protocol.
        gate = "training/runs/stage2/fla/b6_raw_full_2m/cpu_gate"
        ssh(f"mkdir -p {gate}")
        files = {
            "cpu_result.json": CPU / f"{NAME}_512.json",
            "cpu_summary.json": CPU / "summary.json",
            "reference_summary.json": ROOT / "training/runs/stage2/fla/cpu_boundary_6m/idle10_group60/summary.json",
        }
        for target, source in files.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            call(["scp", "-q", str(source), f"{HOST}:{REMOTE}/{gate}/{target}"], timeout=120)
        arguments = (f"--cpu-result {gate}/cpu_result.json "
                     f"--cpu-summary {gate}/cpu_summary.json "
                     f"--reference-summary {gate}/reference_summary.json")
        dry_run = ssh(f"CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python -B "
                      f"tools/stage2_fla_b6_full_selfplay_gate.py {arguments}", timeout=180)
        status(status="dry_run_passed_waiting_gpu0", cpu_mean_s=mean,
               dry_run=json.loads(dry_run))
        while True:
            memory = ssh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", timeout=30)
            first = memory.splitlines()[0].split(",")
            if first[0].strip() == "0" and int(first[1].strip().split()[0]) < 500:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("GPU0 did not become idle before self-play")
            time.sleep(60)
        # The strict gate checks the CPU result a second time before creating
        # any V3 run. SSH can remain attached to a remote background shell;
        # a bounded timeout is followed by an actual state/process check.
        launch = (f"nohup env CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python -u -B "
                  f"tools/stage2_fla_b6_full_selfplay_gate.py {arguments} --execute "
                  f"> {R2}/selfplay_controller.log 2>&1 < /dev/null &")
        ssh(f"mkdir -p {R2}")
        try:
            ssh(launch, timeout=20)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(10)
        state_path = f"{R2}/controller_state.json"
        remote_state = ssh(f"test -f {state_path} && cat {state_path}", timeout=30)
        state = json.loads(remote_state)
        if state.get("status") not in ("running", "complete"):
            raise RuntimeError(f"self-play controller did not start cleanly: {state}")
        status(status="selfplay_started", cpu_mean_s=mean,
               remote_state=state_path, run_id=state["run_id"])
        return 0
    except Exception as exc:
        status(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
