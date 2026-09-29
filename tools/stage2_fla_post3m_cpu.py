"""Fetch verified FLA child-2M terminal snapshots and screen them on local CPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_stage2_cpu_latency import run_one
from stage2_fla_selfplay_queue import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "connect4_gpu_2608"
REMOTE_ROOT = "/root/autodl-tmp/Connect4_3D_game_refactor"
REMOTE_MATCH = f"{REMOTE_ROOT}/training/runs/stage2/fla/direct_3m/r2_gpu1_round_robin"
OUTPUT = ROOT / "training/runs/stage2/fla/cpu_terminal_3m/idle10_group60"
NAMES = ("gravity_b8", "raw3d_to2d_b8",
         "raw3d_to2d_thin_b8c192", "raw3d_to2d_thin_b6c128")
SIMS, REPEATS, IDLE, GROUP_IDLE = 512, 3, 10.0, 60.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote_ready() -> bool:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", REMOTE,
         f"test -f {REMOTE_MATCH}/inputs.json"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(f"SSH readiness check failed: {result.stderr[-1000:]}")


def copy(remote_path: str, local: Path, expected: str | None = None) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        if expected and sha256(local) != expected:
            raise ValueError(f"existing local file SHA-256 differs: {local}")
        return
    temporary = local.with_name(local.name + ".downloading")
    if temporary.exists():
        temporary.unlink()
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["scp", "-o", "ConnectTimeout=15", f"{REMOTE}:{remote_path}", str(temporary)],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode == 0:
                break
            detail = result.stderr[-1000:]
        except subprocess.TimeoutExpired as exc:
            detail = str(exc)
        temporary.unlink(missing_ok=True)
        if attempt == 2:
            raise RuntimeError(f"SCP failed for {remote_path}: {detail}")
        time.sleep(10)
    if expected and sha256(temporary) != expected:
        temporary.unlink()
        raise ValueError(f"downloaded file SHA-256 differs: {remote_path}")
    temporary.replace(local)


def fetch_inputs() -> dict:
    path = OUTPUT / "inputs_remote.json"
    copy(f"{REMOTE_MATCH}/inputs.json", path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema") != "connect4-stage2-fla-terminal-four-v1"
            or set(payload.get("models", {})) != set(NAMES)):
        raise ValueError("unexpected terminal inputs schema or model set")
    for name in NAMES:
        row = payload["models"][name]
        if (row["child_positions"] != 2_000_000
                or row["parent_positions"] != 1_000_000
                or row["cumulative_exposure_positions"] != 3_000_000):
            raise ValueError(f"terminal training boundary differs: {name}")
        copy(row["snapshot"], OUTPUT / "inputs" / f"{name}.pt",
             row["snapshot_sha256"])
        copy(row["config"], OUTPUT / "inputs" / f"{name}.json",
             row["config_sha256"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    if args.poll_seconds < 30:
        parser.error("poll interval must be at least 30 seconds")
    if not args.execute:
        print(json.dumps({"remote_inputs": f"{REMOTE_MATCH}/inputs.json",
                          "output": str(OUTPUT), "names": NAMES,
                          "simulations": SIMS, "repeats": REPEATS,
                          "idle_s": IDLE, "group_idle_s": GROUP_IDLE}, indent=2))
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / "watcher_state.json"
    try:
        deadline = time.monotonic() + 72 * 3600
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("terminal snapshots not ready within 72 hours")
            try:
                ready = remote_ready()
                if ready:
                    payload = fetch_inputs()
                    break
                reason = "remote inputs not yet published"
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if not args.watch:
                raise RuntimeError(reason)
            atomic_json(state_path, {"status": "waiting_remote_inputs",
                                     "last_reason": reason,
                                     "updated_at_utc": now()})
            time.sleep(args.poll_seconds)
        for index, name in enumerate(NAMES):
            row = payload["models"][name]
            target = OUTPUT / f"{name}_{SIMS}.json"
            if target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                meta = existing["metadata"]
                if (meta["artifact_sha256"] != row["snapshot_sha256"]
                        or meta["config_sha256"] != row["config_sha256"]
                        or meta["mcts_sims"] != SIMS or meta["repeats"] != REPEATS
                        or meta["idle_s"] != IDLE
                        or meta["artifact_kind"] != "formal_v3_snapshot"):
                    raise ValueError(f"existing CPU result identity differs: {name}")
                continue
            atomic_json(state_path, {"status": "measuring", "active": name,
                                     "updated_at_utc": now()})
            result = run_one(
                name, OUTPUT / "inputs" / f"{name}.pt",
                OUTPUT / "inputs" / f"{name}.json", SIMS,
                repeats=REPEATS, idle_s=IDLE,
                artifact_kind="formal_v3_snapshot",
            )
            result["metadata"]["group_idle_s"] = GROUP_IDLE
            result["metadata"]["source_checkpoint_sha256"] = row["terminal_sha256"]
            result["metadata"]["cumulative_exposure_positions"] = 3_000_000
            atomic_json(target, result)
            if index + 1 < len(NAMES):
                time.sleep(GROUP_IDLE)
        results = []
        for name in NAMES:
            path = OUTPUT / f"{name}_{SIMS}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            meta = value["metadata"]
            stats = value["summary"]["excluding_shortcuts"]
            if (meta["repeats"] != REPEATS or meta["idle_s"] != IDLE
                    or meta["group_idle_s"] != GROUP_IDLE
                    or stats["count"] != 45):
                raise ValueError(f"CPU protocol mismatch: {name}")
            results.append({"name": name, "mean_s": stats["mean_s"],
                            "median_s": stats["median_s"], "p90_s": stats["p90_s"],
                            "p95_s": stats["p95_s"], "count": stats["count"],
                            "passes_3_3_s": stats["mean_s"] <= 3.3,
                            "model_sha256": meta["artifact_sha256"],
                            "config_sha256": meta["config_sha256"],
                            "terminal_checkpoint_sha256": meta["source_checkpoint_sha256"],
                            "result_sha256": sha256(path)})
        atomic_json(OUTPUT / "summary.json", {
            "schema": "connect4-stage2-fla-cpu-terminal-four-v1",
            "created_at_utc": now(),
            "status": "relative_latency_only_not_flash_qualified",
            "simulations": SIMS, "repeats": REPEATS, "idle_s": IDLE,
            "group_idle_s": GROUP_IDLE, "cpu_gate_mean_s": 3.3,
            "source_inputs_sha256": sha256(OUTPUT / "inputs_remote.json"),
            "machine": {key: json.loads((OUTPUT / f"{NAMES[0]}_{SIMS}.json").read_text())
                        ["metadata"][key] for key in (
                            "runtime", "platform", "processor", "cpu_count",
                            "torch_num_threads", "torch_num_interop_threads",
                            "corpus", "corpus_seed", "forced_tactics")},
            "results": results,
        })
        atomic_json(state_path, {"status": "complete", "updated_at_utc": now(),
                                 "summary_sha256": sha256(OUTPUT / "summary.json")})
    except Exception as exc:
        atomic_json(state_path, {"status": "failed", "updated_at_utc": now(),
                                 "error": f"{type(exc).__name__}: {exc}"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
