"""Finish the full B6 donor handoff through the formal local CPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "raw3d_to2d_full_b6c128"
REL = Path("training/runs/stage2/fla/b6_raw_full_2m")
BASE = ROOT / REL
MODEL_DIR = BASE / "runs" / NAME / "standard_late/seed271828"
REMOTE = "/root/autodl-tmp/Connect4_3D_game_refactor"
HOST = "connect4_gpu_2608"
STATE = BASE / "local_watcher_state.json"
CPU = ROOT / "training/runs/stage2/fla/cpu_b6_full_2m/idle10_group60"
REFERENCE = ROOT / "training/runs/stage2/fla/cpu_boundary_6m/idle10_group60/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def status(**fields: object) -> None:
    payload = {"schema": "connect4-stage2-fla-b6-full-local-watcher-v1",
               "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE)


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}: {result.stderr[-1000:]}")
    return result


def remote(command: str, *, timeout: int = 120) -> str:
    return run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
                f"cd {REMOTE} && {command}"], timeout=timeout).stdout


def download(member: str) -> Path:
    target = MODEL_DIR / member
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    if temporary.exists():
        temporary.unlink()
    source = f"{HOST}:{REMOTE}/{REL.as_posix()}/runs/{NAME}/standard_late/seed271828/{member}"
    run(["scp", "-q", source, str(temporary)], timeout=300)
    os.replace(temporary, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-hours", type=float, default=24.0)
    args = parser.parse_args()
    if STATE.exists():
        raise FileExistsError(f"watcher already has state: {STATE}")
    deadline = time.monotonic() + args.wait_hours * 3600
    status(status="waiting_donor")
    try:
        while True:
            try:
                ready = remote(f"test -f {REL.as_posix()}/runs/{NAME}/standard_late/seed271828/report.json && echo ready", timeout=30)
            except (RuntimeError, subprocess.TimeoutExpired):
                ready = ""
            if "ready" in ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("B6 full donor report not available before deadline")
            time.sleep(60)
        status(status="downloading_donor")
        report_path = download("report.json")
        model = download("model.pt")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("training_execution", {}).get("target_positions_reached"):
            raise ValueError("B6 full donor has not reached 1M")
        model_sha = sha256(model)
        if model_sha != report["model_artifact"]["sha256"]:
            raise ValueError("B6 full donor model/report hash mismatch")
        config = BASE / "configs" / f"{NAME}__standard_late__seed271828.json"
        remote_hashes = remote(f"sha256sum {REL.as_posix()}/configs/{config.name} {REL.as_posix()}/runs/{NAME}/standard_late/seed271828/model.pt")
        if sha256(config) not in remote_hashes or model_sha not in remote_hashes:
            raise ValueError("local and remote donor/config hashes differ")
        if CPU.exists():
            raise FileExistsError(f"CPU output already exists; inspect before retry: {CPU}")
        status(status="measuring_cpu", donor_sha256=model_sha)
        command = [sys.executable, "-B", "tools/stage2_fla_cpu_screen.py", "--source", "b6_full",
                   "--sims", "512", "--repeats", "3", "--idle-s", "10", "--group-idle-s", "60"]
        log = BASE / "cpu_screen.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"CPU screen exited {result.returncode}; see {log}")
        summary = json.loads((CPU / "summary.json").read_text(encoding="utf-8"))
        row = summary["results"][0]
        if row["model_sha256"] != model_sha or row["config_sha256"] != sha256(config):
            raise ValueError("CPU result donor/config hash mismatch")
        if summary["simulations"] != 512 or summary["repeats"] != 3 or summary["idle_s"] != 10 or summary["group_idle_s"] != 60:
            raise ValueError("CPU protocol mismatch")
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        if summary["machine"] != reference["machine"] or row["count"] != 45:
            raise ValueError("CPU host or search-state count differs from reference")
        mean = float(row["mean_s"])
        status(status="complete_cpu_gate_passed" if mean < 3.1 else "complete_cpu_gate_failed",
               donor_sha256=model_sha, cpu_mean_s=mean, cpu_p95_s=row["p95_s"],
               cpu_result_sha256=sha256(CPU / f"{NAME}_512.json"),
               cpu_summary_sha256=sha256(CPU / "summary.json"))
        return 0
    except Exception as exc:
        status(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
