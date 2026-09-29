"""Audit and force-stop only the specified raw B8 V3 process tree."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training/runs/stage2/fla3_pre/raw_b8_pool_10m"
RUN = BASE / "runs/fla3_pre_raw_b8_g31_pool10m_seed271829"
OUT = BASE / "actor36_switch"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage2_fla_raw_force_stop.py FORMAL_PID")
    pid = int(sys.argv[1])
    proc = psutil.Process(pid)
    cmd = proc.cmdline()
    joined = " ".join(cmd)
    if ("training.v3 run" not in joined
            or "fla3_pre_raw_b8_g31_pool10m_seed271829.json" not in joined
            or "--max-train-positions 10000000" not in joined):
        raise ValueError("PID is not the expected bounded V3 process")
    if OUT.exists():
        raise FileExistsError("actor36 switch audit already exists")
    pointer_path = RUN / "manifests/latest_generation.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    commit_path = RUN / pointer["commit"]
    if sha(commit_path) != pointer["commit_sha256"]:
        raise ValueError("latest committed generation SHA differs")
    children = proc.children(recursive=True)
    foreign = [child for child in children if not ("multiprocessing" in " ".join(child.cmdline())
                                                      or "resource_tracker" in " ".join(child.cmdline()))]
    if foreign:
        raise ValueError(f"unexpected V3 child processes: {[p.pid for p in foreign]}")
    OUT.mkdir(parents=True)
    audit = {
        "reason": "user_requested_force_stop_for_36_actor_topology_switch",
        "utc": datetime.now(timezone.utc).isoformat(),
        "formal_pid": pid,
        "formal_cmdline": cmd,
        "children": [{"pid": child.pid, "cmdline": child.cmdline()} for child in children],
        "last_commit_generation": pointer["generation"],
        "last_commit_sha256": pointer["commit_sha256"],
        "last_commit_path": pointer["commit"],
        "status": "kill_requested",
    }
    (OUT / "force_stop.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    proc.kill()
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([proc, *children], timeout=15)
    audit["status"] = "killed" if not alive else "some_processes_alive"
    audit["still_alive_pids"] = [item.pid for item in alive]
    (OUT / "force_stop.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if alive:
        raise RuntimeError(f"V3 descendants still alive: {audit['still_alive_pids']}")


if __name__ == "__main__":
    main()
