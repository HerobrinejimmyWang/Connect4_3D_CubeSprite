"""Start B6 offline training after the retained FLA self-play queue drains."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2_fla_selfplay_queue import QUEUE_ROOT, ROOT, atomic_json, validate_donor

NAME = "raw3d_to2d_thin_b6c128"
BASE = ROOT / "training/runs/stage2/fla/b6_raw_2m"
CONFIG = BASE / "configs" / f"{NAME}__standard_late__seed271828.json"
MODEL = BASE / "runs" / NAME / "standard_late/seed271828/model.pt"
REPORT = MODEL.parent / "report.json"
STATE = BASE / "offline_watcher_state.json"
EXPECTED_COMPLETED = [
    "gravity_b8", "column_2d_b8c192", "raw3d_to2d_b8",
    "raw3d_to2d_thin_b8c192", "column3d_v2_thin_b8c192",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-hours", type=float, default=24)
    args = parser.parse_args()
    if args.wait_hours <= 0:
        parser.error("--wait-hours must be positive")
    if not args.execute:
        print(json.dumps({"config": str(CONFIG), "wait_for": EXPECTED_COMPLETED}))
        return 0
    if STATE.exists() or MODEL.exists():
        raise FileExistsError("B6 offline watcher or donor already exists")
    deadline = time.monotonic() + args.wait_hours * 3600
    state = {"schema": "connect4-stage2-fla-b6-offline-watcher-v1", "status": "waiting_queue"}
    atomic_json(STATE, state)
    try:
        while True:
            queue = json.loads((ROOT / QUEUE_ROOT / "queue_state.json").read_text(encoding="utf-8"))
            if queue["status"] == "failed":
                raise RuntimeError("FLA self-play queue failed before B6 offline work")
            if queue["status"] == "awaiting_b6_cpu_gate":
                if queue.get("plan_revision") != "cpu_mean_le_3p8_plus_b6_screen_v1":
                    raise ValueError("queue plan revision changed")
                if [row["name"] for row in queue["completed"]] != EXPECTED_COMPLETED:
                    raise ValueError("retained queue completion inventory mismatch")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("retained self-play queue did not drain within wait bound")
            time.sleep(60)
        # The original handoff controller was already live when the B6C128
        # requirement was clarified. It has now exited, so publish the
        # corrected pending candidate without touching completed runs.
        queue_path = ROOT / QUEUE_ROOT / "queue_state.json"
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        prior = queue.get("planned", [])[-1]
        if prior not in ("raw3d_to2d_thin_b6c208", "raw3d_to2d_thin_b6c192", NAME):
            raise ValueError(f"unexpected pending B6 candidate: {prior}")
        queue["planned"][-1] = NAME
        queue["b6_design_revision"] = {
            "prior": prior,
            "selected": NAME,
            "reason": "user clarified that B6 trunk means B6C128; formal CPU screen remains pending",
        }
        queue["b6_gate"] = "requires_offline_1m_and_cpu_512_mean_le_3p8; faster_than_3p0_allowed"
        atomic_json(queue_path, queue)
        state.update(status="training", started_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(STATE, state)
        log = BASE / "offline_train.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, "-u", "-B", "-m", "training.v3.stage2", "train", "--config", str(CONFIG)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"B6 offline training exited {result.returncode}; see {log}")
        _model, digest = validate_donor(CONFIG, MODEL, REPORT)
        state.update(status="complete", model_sha256=digest, completed_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(STATE, state)
        return 0
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(STATE, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
