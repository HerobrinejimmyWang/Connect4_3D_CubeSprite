"""Drain a V3 generation, screen raw B8 actor counts, then resume unchanged."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training/runs/stage2/fla3_pre/raw_b8_pool_10m"
RUN = BASE / "runs/fla3_pre_raw_b8_g31_pool10m_seed271829"
SCREEN = BASE / "topology_screen_24_36_40"
STATE = SCREEN / "watcher_state.json"
TRAIN_STATE = BASE / "watcher_state.json"
PYTHON = Path(sys.executable).resolve()


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
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def running(pid: int) -> bool:
    status = Path(f"/proc/{pid}/stat")
    return status.exists() and status.read_text().split()[2] != "Z"


def gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    return not result.stdout.strip()


def committed_generation() -> tuple[int, dict, str]:
    pointer = read(RUN / "manifests/latest_generation.json")
    path = RUN / pointer["commit"]
    digest = sha(path)
    if digest != pointer["commit_sha256"]:
        raise ValueError("latest generation pointer checksum differs")
    return pointer["generation"], read(path), digest


def wait_for_drain(pid: int, initial_generation: int) -> tuple[int, dict, str, int]:
    os.kill(pid, signal.SIGTERM)
    atomic(STATE, {"status": "draining_current_generation", "formal_pid": pid,
                   "initial_generation": initial_generation})
    deadline = time.monotonic() + 5400
    while running(pid):
        if time.monotonic() > deadline:
            raise TimeoutError("V3 did not drain within 90 minutes; inspect it before continuing")
        time.sleep(10)
    for _ in range(30):
        if read(TRAIN_STATE).get("status") == "failed" and gpu_idle():
            break
        time.sleep(2)
    else:
        raise RuntimeError("V3 stopped, but the original controller or GPUs did not quiesce")
    manifest = read(RUN / "run_manifest.json")
    generation, commit, digest = committed_generation()
    positions = int(manifest["formal_loop_state"]["train_positions_consumed"])
    if (generation != initial_generation + 1
            or manifest.get("status") != "stopped_at_safe_boundary"
            or manifest.get("stop_reason") != "drained_after_signal_15"
            or not commit.get("replay_shards")
            or sha(RUN / commit["checkpoint"]) != commit["checkpoint_sha256"]
            or sha(RUN / commit["accepted_model_path"]) != commit["accepted_model_sha256"]):
        raise ValueError("post-signal generation boundary is incomplete or unexpected")
    return generation, commit, digest, positions


def benchmark(generation: int, commit: dict, commit_sha: str, positions: int) -> None:
    checkpoint = RUN / commit["checkpoint"]
    checkpoint_sha = commit["checkpoint_sha256"]
    inputs = {
        "generation": generation, "drained_positions": positions,
        "generation_commit_sha256": commit_sha,
        "checkpoint_sha256": checkpoint_sha,
        "accepted_sha256": commit["accepted_model_sha256"],
        "config_sha256": sha(RUN / "resolved_config.json"),
        "actors_order": [36, 40, 24], "games_per_point": 128,
        "repeats": 2, "devices": ["cuda:0", "cuda:1"],
        "purpose": "same-checkpoint topology screen; 24 is the matched control",
    }
    atomic(SCREEN / "inputs.json", inputs)
    results: list[dict] = []
    for repeat in (1, 2):
        for actors in (36, 40, 24):
            if not gpu_idle():
                raise RuntimeError("another GPU job started before a topology point")
            label = f"actors{actors}_rep{repeat}"
            out = SCREEN / label
            atomic(STATE, {"status": "measuring", "label": label,
                           "generation": generation, "drained_positions": positions})
            cmd = [str(PYTHON), "-u", "-B", str(ROOT / "tools/benchmark_bal5_topology_point.py"),
                   "--run-dir", str(RUN), "--checkpoint", str(checkpoint),
                   "--expected-checkpoint-sha256", checkpoint_sha,
                   "--generation", str(generation), "--actors", str(actors),
                   "--devices", "cuda:0,cuda:1", "--games", "128",
                   "--lanes", "4", "--inference-batch-size", "32",
                   "--inference-timeout-ms", "1", "--game-id-base", str(900000 + repeat * 128),
                   "--output-root", str(out), "--label", label]
            with (SCREEN / f"{label}.log").open("w", encoding="utf-8") as stream:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f"topology point {label} exited {proc.returncode}")
            point_path = out / "point.json"
            point = read(point_path)
            semantics = point["semantics"]
            if (point["provenance"]["checkpoint_sha256"] != checkpoint_sha
                    or semantics["actor_processes"] != actors
                    or semantics["games"] != 128
                    or semantics["selfplay_devices"] != ["cuda:0", "cuda:1"]
                    or semantics["mcts_lanes_per_actor"] != 4
                    or semantics["full_search_sims"] != 512
                    or semantics["fast_search_sims"] != 32
                    or point["result"]["games"] != 128):
                raise ValueError(f"topology point {label} differs from the fixed workload")
            results.append({"label": label, "actors": actors, "repeat": repeat,
                            "point_sha256": sha(point_path),
                            "wall_seconds": point["result"]["wall_seconds"],
                            "simulations_per_second": point["result"]["simulations_per_second"],
                            "mean_inference_batch": point["result"]["mean_inference_batch"],
                            "cpu_util_percent_of_quota": point["resources"].get("cpu_util_percent_of_quota"),
                            "gpu_util_mean": point["resources"].get("gpu_util_mean")})
            atomic(SCREEN / "progress.json", {"completed_points": results})
    atomic(SCREEN / "complete.json", {"status": "complete", **inputs, "points": results})


def finish_screen(generation: int, commit: dict, digest: str, positions: int) -> None:
    benchmark(generation, commit, digest, positions)
    atomic(BASE / "watcher_state_topology_drain.json", read(TRAIN_STATE))
    atomic(STATE, {"status": "training_resumed_24_actors",
                   "generation": generation, "drained_positions": positions,
                   "complete_sha256": sha(SCREEN / "complete.json")})
    with (SCREEN / "resume_training.log").open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            [str(PYTHON), "-u", str(ROOT / "tools/stage2_fla_raw_10m_continue.py"),
             "resume_after_topology"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
        )
    if proc.returncode:
        raise RuntimeError(f"24-actor V3 resume exited {proc.returncode}")
    atomic(STATE, {"status": "training_complete", "complete_sha256": sha(SCREEN / "complete.json")})


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage2_fla_raw_topology_screen.py FORMAL_PID|resume_screen")
    if sys.argv[1] == "resume_screen":
        if read(STATE).get("status") != "failed" or read(TRAIN_STATE).get("status") != "failed":
            raise ValueError("expected the recorded failed screen and drained training controller")
        drain = read(SCREEN / "drain.json")
        generation, commit, digest = committed_generation()
        manifest = read(RUN / "run_manifest.json")
        positions = int(manifest["formal_loop_state"]["train_positions_consumed"])
        if (generation != drain["generation"] or positions != drain["drained_positions"]
                or digest != drain["generation_commit_sha256"]
                or sha(RUN / commit["checkpoint"]) != drain["checkpoint_sha256"]
                or manifest.get("stop_reason") != "drained_after_signal_15"
                or not gpu_idle()):
            raise ValueError("committed drained boundary changed; inspect before retry")
        try:
            finish_screen(generation, commit, digest, positions)
        except Exception as exc:
            atomic(STATE, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise
        return
    pid = int(sys.argv[1])
    if SCREEN.exists():
        raise FileExistsError("topology screen already exists; inspect before retry")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if ("training.v3 run" not in cmdline
            or "fla3_pre_raw_b8_g31_pool10m_seed271829.json" not in cmdline
            or "--max-train-positions 10000000" not in cmdline):
        raise ValueError("PID is not the expected bounded raw B8 V3 process")
    if read(TRAIN_STATE).get("status") != "running_to10m":
        raise ValueError("original 10M controller is not running")
    if shutil.disk_usage(BASE).free < 25 * 1024**3:
        raise RuntimeError("less than 25 GiB free before topology screen")
    generation, _, _ = committed_generation()
    SCREEN.mkdir(parents=True)
    try:
        generation, commit, digest, positions = wait_for_drain(pid, generation)
        atomic(SCREEN / "drain.json", {"generation": generation,
                                      "drained_positions": positions,
                                      "generation_commit_sha256": digest,
                                      "checkpoint_sha256": commit["checkpoint_sha256"]})
        finish_screen(generation, commit, digest, positions)
    except Exception as exc:
        atomic(STATE, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
