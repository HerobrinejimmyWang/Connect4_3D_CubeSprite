"""Handoff the multiview donor through the formal Windows CPU gate to GPU0 V3."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training/runs/stage2/fla/multiview_b8_1m"
NAME = "multiview_resnet_b8c192"
RUN = BASE / "runs" / NAME / "standard_late/seed271828"
CPU = ROOT / "training/runs/stage2/fla/cpu_multiview_b8_1m/idle10_group60"
REFERENCE = ROOT / "training/runs/stage2/fla/cpu_boundary_6m/idle10_group60/summary.json"
STATE = BASE / "local_watcher_state.json"
REMOTE = "/root/autodl-tmp/Connect4_3D_game_refactor"
HOST = "connect4_gpu_2608"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(status: str, **fields: object) -> None:
    payload = {"schema": "connect4-stage2-fla-multiview-b8-local-watcher-v1",
               "status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, STATE)


def run(command: list[str], *, timeout: int = 120) -> str:
    last: Exception | None = None
    for attempt in range(3):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result.stdout
            last = RuntimeError(f"command exited {result.returncode}: {result.stderr[-1000:]}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            last = exc
        if attempt < 2:
            time.sleep(5 * (attempt + 1))
    assert last is not None
    raise last


def remote(command: str, *, timeout: int = 120) -> str:
    return run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
                f"cd {REMOTE} && {command}"], timeout=timeout)


def scp(source: str, destination: str) -> None:
    run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         source, destination], timeout=300)


def main() -> int:
    if STATE.exists():
        raise FileExistsError(f"local watcher already has state: {STATE}")
    design = json.loads((BASE / "design.json").read_text(encoding="utf-8"))
    config = BASE / "configs" / f"{NAME}__standard_late__seed271828.json"
    if sha256(config) != design["config_sha256"]:
        raise ValueError("frozen design config hash mismatch")
    deadline = time.monotonic() + 12 * 3600
    write("waiting_donor")
    try:
        while True:
            try:
                raw = remote("cat training/runs/stage2/fla/multiview_b8_1m/offline_watcher_state.json", timeout=30)
                state = json.loads(raw)
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
                state = {}
            if state.get("status") == "donor_complete":
                break
            if state.get("status") == "failed":
                raise RuntimeError(f"remote offline donor failed: {state.get('error')}")
            if time.monotonic() > deadline:
                raise TimeoutError("remote donor did not finish within 12 hours")
            time.sleep(60)
        write("downloading_donor")
        RUN.mkdir(parents=True, exist_ok=True)
        for member in ("report.json", "model.pt"):
            destination = RUN / member
            if destination.exists():
                raise FileExistsError(f"donor artifact already exists: {destination}")
            temporary = destination.with_name(destination.name + ".download")
            if temporary.exists():
                raise FileExistsError(f"stale partial download: {temporary}")
            scp(f"{HOST}:{REMOTE}/training/runs/stage2/fla/multiview_b8_1m/runs/{NAME}/standard_late/seed271828/{member}",
                str(temporary))
            os.replace(temporary, destination)
        report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
        donor_sha = sha256(RUN / "model.pt")
        if not report["training_execution"]["target_positions_reached"] or donor_sha != report["model_artifact"]["sha256"]:
            raise ValueError("downloaded donor report/model mismatch")
        remote_hashes = remote(
            f"sha256sum training/runs/stage2/fla/multiview_b8_1m/configs/{config.name} "
            f"training/runs/stage2/fla/multiview_b8_1m/runs/{NAME}/standard_late/seed271828/model.pt")
        if sha256(config) not in remote_hashes or donor_sha not in remote_hashes:
            raise ValueError("local/remote donor or config hash mismatch")
        if CPU.exists():
            raise FileExistsError(f"CPU output already exists: {CPU}")
        write("measuring_cpu", donor_sha256=donor_sha)
        log = BASE / "cpu_screen.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-B", "tools/stage2_fla_cpu_screen.py",
                                     "--source", "multiview_b8", "--sims", "512",
                                     "--repeats", "3", "--idle-s", "10",
                                     "--group-idle-s", "60"], cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"CPU screen exited {result.returncode}; see {log}")
        summary = json.loads((CPU / "summary.json").read_text(encoding="utf-8"))
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        row = summary["results"][0]
        if (summary["machine"] != reference["machine"] or summary["simulations"] != 512
                or summary["repeats"] != 3 or summary["idle_s"] != 10
                or summary["group_idle_s"] != 60 or row["count"] != 45
                or row["model_sha256"] != donor_sha or row["config_sha256"] != sha256(config)):
            raise ValueError("CPU protocol, host, corpus or model identity mismatch")
        mean = float(row["mean_s"])
        if mean > 3.8:
            write("cpu_screen_failed", donor_sha256=donor_sha,
                  cpu_mean_s=mean, cpu_p95_s=row["p95_s"])
            return 0
        write("cpu_screen_passed", donor_sha256=donor_sha,
              cpu_mean_s=mean, cpu_p95_s=row["p95_s"])
        remote("mkdir -p training/runs/stage2/fla/multiview_b8_1m/cpu_gate")
        gate = f"{HOST}:{REMOTE}/training/runs/stage2/fla/multiview_b8_1m/cpu_gate/"
        for local, remote_name in ((CPU / f"{NAME}_512.json", f"{NAME}_512.json"),
                                   (CPU / "summary.json", "summary.json"),
                                   (REFERENCE, "reference_summary.json")):
            scp(str(local), gate + remote_name)
        remote("/root/miniconda3/bin/python -B tools/stage2_fla_multiview_b8_postcpu.py")
        remote("setsid -f env CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python -u -B "
               "tools/stage2_fla_multiview_b8_postcpu.py --execute > "
               "training/runs/stage2/fla/multiview_b8_1m/postcpu.log 2>&1 < /dev/null")
        post_state = None
        for _ in range(18):
            try:
                post_state = json.loads(remote(
                    "cat training/runs/stage2/fla/multiview_b8_1m/postcpu_state.json",
                    timeout=30))
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
                post_state = None
            if post_state is not None:
                break
            time.sleep(5)
        if post_state is None or post_state.get("status") != "selfplay_running":
            raise RuntimeError(f"post-CPU controller launch not verified: {post_state}")
        write("postcpu_controller_started", donor_sha256=donor_sha,
              cpu_mean_s=mean, cpu_p95_s=row["p95_s"],
              cpu_result_sha256=sha256(CPU / f"{NAME}_512.json"),
              cpu_summary_sha256=sha256(CPU / "summary.json"))
        return 0
    except Exception as exc:
        write("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
