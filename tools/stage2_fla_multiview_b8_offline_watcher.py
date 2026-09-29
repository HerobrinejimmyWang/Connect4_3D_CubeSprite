"""Wait for the active FLA child to drain, pause its queue, then train multiview B8."""

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
BASE = ROOT / "training/runs/stage2/fla/multiview_b8_1m"
FORK = ROOT / "training/runs/stage2/fla/selfplay/r3_bal5r1_fork_v2"
RUN = FORK / "runs/fla_r3_gravity_b8_bal5r1_from1m_seed271828"
NAME = "multiview_resnet_b8c192"
CONFIG = BASE / "configs" / f"{NAME}__standard_late__seed271828.json"
OUTPUT = BASE / "runs" / NAME / "standard_late/seed271828"
STATE = BASE / "offline_watcher_state.json"
QUEUED = FORK / "queue_state.json"
PYTHON = Path("/root/miniconda3/bin/python")
WORKER_PID = 906660


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def status(value: str, **fields: object) -> None:
    write(STATE, {"schema": "connect4-stage2-fla-multiview-b8-offline-watcher-v1",
                  "status": value, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields})


def alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    return stat.read_text().split()[2] != "Z"


def train_donor() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"offline donor output already exists: {OUTPUT}")
    log = BASE / "offline_train.log"
    status("training_donor", config_sha256=sha256(CONFIG))
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(PYTHON), "-u", "-B", "-m", "training.v3.stage2",
                                 "train", "--config", str(CONFIG)],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"offline donor exited {result.returncode}; see {log}")
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    model = OUTPUT / "model.pt"
    if not report["training_execution"]["target_positions_reached"]:
        raise ValueError("offline donor did not reach 1M positions")
    if sha256(model) != report["model_artifact"]["sha256"]:
        raise ValueError("offline donor model/report hash mismatch")
    status("donor_complete", model_sha256=sha256(model),
           report_sha256=sha256(OUTPUT / "report.json"),
           config_sha256=sha256(CONFIG))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause-only", action="store_true")
    parser.add_argument("--train-after-pause", action="store_true")
    args = parser.parse_args()
    if args.pause_only and args.train_after_pause:
        parser.error("pause-only and train-after-pause are mutually exclusive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("offline donor must use physical GPU0; GPU1 reserved")
    if not args.train_after_pause and (STATE.exists() or (OUTPUT.exists() and not args.pause_only)):
        raise FileExistsError("watcher or donor output already exists")
    design = json.loads((BASE / "design.json").read_text(encoding="utf-8"))
    if sha256(CONFIG) != design["config_sha256"]:
        raise ValueError("frozen offline config SHA-256 mismatch")
    deadline = time.monotonic() + 6 * 3600
    if args.train_after_pause:
        while True:
            prior = json.loads(STATE.read_text(encoding="utf-8"))
            if prior.get("status") == "queue_paused_safe_boundary":
                break
            if prior.get("status") == "failed":
                raise RuntimeError(f"safe pause failed: {prior.get('error')}")
            if time.monotonic() > deadline:
                raise TimeoutError("safe pause did not complete within six hours")
            time.sleep(20)
        if json.loads(QUEUED.read_text(encoding="utf-8")).get("status") != "paused_by_user":
            raise ValueError("queue is not paused")
        try:
            train_donor()
            return 0
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
    status("waiting_safe_boundary", worker_pid=WORKER_PID)
    try:
        while alive(WORKER_PID):
            if time.monotonic() > deadline:
                raise TimeoutError("V3 worker did not drain within six hours")
            time.sleep(20)
        manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "stopped_at_safe_boundary" or manifest.get("stop_reason") != "drained_after_signal_15":
            raise ValueError(f"V3 worker did not stop at requested boundary: {manifest.get('stop_reason')}")
        pointer = json.loads((RUN / "manifests/latest_generation.json").read_text(encoding="utf-8"))
        commit_path = RUN / pointer["commit"]
        if sha256(commit_path) != pointer["commit_sha256"]:
            raise ValueError("generation pointer checksum mismatch")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        for member, expected in (("checkpoint", "checkpoint_sha256"),
                                 ("accepted_model_path", "accepted_model_sha256")):
            if sha256(RUN / commit[member]) != commit[expected]:
                raise ValueError(f"committed {member} hash mismatch")
        queue = json.loads(QUEUED.read_text(encoding="utf-8"))
        if queue.get("status") != "running" or queue.get("active") != "gravity_b8":
            raise ValueError("queue changed before pause could be recorded")
        queue.update(status="paused_by_user", pause_reason="multiview_resnet B8 screening",
                     paused_generation=pointer["generation"],
                     paused_generation_commit_sha256=pointer["commit_sha256"],
                     paused_train_positions=manifest["formal_loop_state"]["train_positions_consumed"])
        write(QUEUED, queue)
        status("queue_paused_safe_boundary", generation=pointer["generation"],
               train_positions=queue["paused_train_positions"],
               generation_commit_sha256=pointer["commit_sha256"])
        if args.pause_only:
            return 0
        train_donor()
        return 0
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
