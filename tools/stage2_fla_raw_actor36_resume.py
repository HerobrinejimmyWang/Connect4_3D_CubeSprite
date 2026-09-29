"""Recover the user-interrupted raw B8 generation and resume V3 at 36 actors."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.config import load_config
from training.v3.pipeline import lineage_config_hash

BASE = ROOT / "training/runs/stage2/fla3_pre/raw_b8_pool_10m"
RUN = BASE / "runs/fla3_pre_raw_b8_g31_pool10m_seed271829"
ORIGINAL = BASE / "configs/fla3_pre_raw_b8_g31_pool10m_seed271829.json"
NEW = BASE / "configs/fla3_pre_raw_b8_g31_pool10m_seed271829_actors36.json"
OUT = BASE / "actor36_switch"
STATE = OUT / "state.json"
EXPECTED_ORIGINAL = "20c022b8f2b03637b82e2e97682621e59972a2187602d5203991845950ded433"
EXPECTED_LINEAGE = "4b6c16f17ce0ae89084b5de43867c8ab0c79e54653d8dddbeb72eb0adb1bc6f0"
EXPECTED_GEN11 = "c1be04be79da44bc060e043d217a35e0daed03d6cc3fe4e58dddb750dc5d2ebc"
TARGET = 10_000_000


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def latest_commit() -> tuple[int, dict, str]:
    pointer = read(RUN / "manifests/latest_generation.json")
    path = RUN / pointer["commit"]
    digest = sha(path)
    if digest != pointer["commit_sha256"]:
        raise ValueError("latest generation pointer SHA differs")
    return int(pointer["generation"]), read(path), digest


def gpu_idle() -> bool:
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    return not proc.stdout.strip()


def prepare() -> None:
    if STATE.exists() or NEW.exists():
        raise FileExistsError("actor36 recovery already prepared; inspect existing evidence")
    stop = read(OUT / "force_stop.json")
    if (stop["status"] != "killed" or stop["still_alive_pids"]
            or stop["last_commit_generation"] != 11
            or stop["last_commit_sha256"] != EXPECTED_GEN11):
        raise ValueError("force-stop audit differs from the expected generation 11 commit")
    if not gpu_idle():
        raise RuntimeError("GPU jobs remain after force stop")
    if shutil.disk_usage(BASE).free < 25 * 1024**3:
        raise RuntimeError("less than 25 GiB free for the 10M V3 continuation")
    generation, commit, digest = latest_commit()
    if generation != 11 or digest != EXPECTED_GEN11:
        raise ValueError("latest committed generation changed after force stop")
    checkpoint = RUN / commit["checkpoint"]
    accepted = RUN / commit["accepted_model_path"]
    if (sha(checkpoint) != commit["checkpoint_sha256"]
            or sha(accepted) != commit["accepted_model_sha256"]
            or commit["config_hash"] != EXPECTED_LINEAGE
            or not commit["replay_shards"]):
        raise ValueError("generation 11 committed model/replay evidence is incomplete")
    for shard in commit["replay_shards"]:
        if sha(RUN / shard["path"]) != shard["checksum_sha256"]:
            raise ValueError(f"committed replay shard SHA differs: {shard['path']}")
    manifest = read(RUN / "run_manifest.json")
    if manifest["config_hash"] != EXPECTED_LINEAGE:
        raise ValueError("V3 run manifest semantic hash changed")
    lock_path = RUN / "manifests/coordinator.lock"
    draft_path = RUN / "manifests/generation_drafts/g000012.json"
    lock = read(lock_path)
    draft = read(draft_path)
    if (int(lock["pid"]) != stop["formal_pid"]
            or Path(f"/proc/{lock['pid']}").exists()
            or lock["run_id"] != draft["run_id"]
            or draft["generation"] != 12
            or draft["phase"] != "started"
            or draft["artifacts"]
            or draft["config_hash"] != EXPECTED_LINEAGE
            or (RUN / "manifests/generations/g000012.json").exists()):
        raise ValueError("interrupted coordinator lock or empty draft changed")
    if sha(ORIGINAL) != EXPECTED_ORIGINAL:
        raise ValueError("original runtime config changed")
    config = read(ORIGINAL)
    if config["runtime"]["actor_processes"] != 24:
        raise ValueError("original runtime is not 24 actors")
    config["runtime"]["actor_processes"] = 36
    atomic(NEW, config)
    if lineage_config_hash(load_config(NEW)) != lineage_config_hash(load_config(ORIGINAL)):
        raise ValueError("actor36 config changes learning semantics")
    if lineage_config_hash(load_config(NEW)) != EXPECTED_LINEAGE:
        raise ValueError("actor36 config lineage hash differs from the run")
    recovery = OUT / "recovery_originals"
    recovery.mkdir(exist_ok=False)
    lock_sha, draft_sha = sha(lock_path), sha(draft_path)
    lock_path.rename(recovery / "coordinator.lock")
    draft_path.rename(recovery / "g000012.json")
    receipt = {
        "status": "prepared", "generation": 11,
        "generation_commit_sha256": digest,
        "checkpoint_sha256": commit["checkpoint_sha256"],
        "accepted_sha256": commit["accepted_model_sha256"],
        "replay_shards": len(commit["replay_shards"]),
        "original_config_sha256": EXPECTED_ORIGINAL,
        "actor36_config_sha256": sha(NEW),
        "semantic_config_hash": EXPECTED_LINEAGE,
        "interrupted_empty_draft_sha256": draft_sha,
        "interrupted_coordinator_lock_sha256": lock_sha,
        "recovery_originals": str(recovery),
    }
    atomic(OUT / "recovery_receipt.json", receipt)
    atomic(STATE, receipt)


def launch() -> None:
    receipt = read(OUT / "recovery_receipt.json")
    if read(STATE).get("status") != "prepared" or sha(NEW) != receipt["actor36_config_sha256"]:
        raise ValueError("actor36 prepared receipt/config changed")
    if not gpu_idle():
        raise RuntimeError("GPUs are in use before V3 resume")
    generation, _, digest = latest_commit()
    if generation != 11 or digest != receipt["generation_commit_sha256"]:
        raise ValueError("committed boundary changed before V3 resume")
    command = [sys.executable, "-u", "-B", "-m", "training.v3", "run",
               "--config", str(NEW), "--resume", "--execute",
               "--max-train-positions", str(TARGET)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    atomic(STATE, {"status": "running_to10m_actors36", "last_verified_generation": 11,
                   "actor36_config_sha256": receipt["actor36_config_sha256"]})
    try:
        with (OUT / "to10m_actors36.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            first_commit_verified = False
            while process.poll() is None:
                if not first_commit_verified:
                    gen, commit, digest = latest_commit()
                    if gen >= 12:
                        if (commit["config_hash"] != EXPECTED_LINEAGE
                                or sha(RUN / commit["checkpoint"]) != commit["checkpoint_sha256"]
                                or sha(RUN / commit["accepted_model_path"]) != commit["accepted_model_sha256"]):
                            raise ValueError("first actor36 generation commit failed validation")
                        atomic(STATE, {"status": "running_to10m_actors36",
                                       "last_verified_generation": gen,
                                       "first_actor36_commit_sha256": digest,
                                       "actor36_config_sha256": receipt["actor36_config_sha256"]})
                        first_commit_verified = True
                time.sleep(20)
        if process.returncode:
            raise RuntimeError(f"actor36 V3 process exited {process.returncode}")
        manifest = read(RUN / "run_manifest.json")
        positions = int(manifest["formal_loop_state"]["train_positions_consumed"])
        status = "complete" if (positions >= TARGET and manifest.get("stop_reason") == "max_train_positions") else "stopped_before_10m"
        atomic(STATE, {"status": status, "positions": positions,
                       "stop_reason": manifest.get("stop_reason"),
                       "generation_commit_sha256": latest_commit()[2],
                       "actor36_config_sha256": receipt["actor36_config_sha256"]})
    except Exception as exc:
        atomic(STATE, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "launch"}:
        raise SystemExit("usage: stage2_fla_raw_actor36_resume.py prepare|launch")
    if sys.argv[1] == "prepare":
        prepare()
    else:
        launch()
