"""BAL-5 machine-topology calibration: one 64-game self-play work point.

This is the calibration harness agreed for the 16M-class winning3d_fusion_resnet
checkpoint. It is READ-ONLY with respect to source run directories: the model and
optimizer state are restored into memory and every output is written below an
explicit, timestamped ``--output-root`` that must not already exist.

Semantic contract (fixed, not swept):
  * one checkpoint, identified by SHA-256
  * mcts_lanes_per_actor = 4
  * full/fast simulations, opening/seed, temperature/noise untouched
  * inference_batch_size = 32, inference timeout 1 ms
  * FP32 learner, batch 256

Only the operational topology is swept:
  * ``--actors``   actor process count
  * ``--devices``  self-play GPU assignment; actors are split across the devices
                   by the runtime's own round-robin (``actor_id % len(devices)``),
                   so one invocation can cover both cards of the 2x3080Ti host.

Every point emits a JSON record containing the throughput counters, the resolved
semantic config hash, the source gate/checkpoint identity, and the sampled
resource metrics (GPU utilisation/memory/power, cgroup CPU quota utilisation and
throttling, RSS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from benchmark_v3_selfplay_topology import _ResourceSampler
from training.v3.actor_runtime import run_self_play_actor_pool
from training.v3.checkpoint import load_checkpoint
from training.v3.config import config_hash, load_config
from training.v3.pipeline import lineage_config_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_dirty() -> str:
    try:
        out = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        ).stdout
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _cpu_quota() -> dict[str, Any]:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    quota = read("/sys/fs/cgroup/cpu.max")
    cores = None
    if quota and quota != "max":
        parts = quota.split()
        if len(parts) == 2 and parts[1] != "0":
            cores = int(parts[0]) / int(parts[1])
    return {
        "cpu_max": quota,
        "quota_cores": cores,
        "memory_max": read("/sys/fs/cgroup/memory.max"),
        "cpu_stat": read("/sys/fs/cgroup/cpu.stat"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--actors", type=int, required=True)
    parser.add_argument(
        "--devices",
        required=True,
        help="comma-separated self-play devices, e.g. cuda:0 or cuda:0,cuda:1",
    )
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--inference-timeout-ms", type=float, default=1.0)
    parser.add_argument("--game-id-base", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("calibration requires CUDA")
    if args.lanes != 4:
        raise RuntimeError("calibration fixes mcts_lanes_per_actor=4; lanes=6 is historical only")
    if args.actors < 1 or args.games < 1:
        raise ValueError("actors and games must be positive")

    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    if not devices:
        raise ValueError("at least one self-play device is required")
    for device in devices:
        if re.fullmatch(r"cuda:\d+", device) is None:
            raise ValueError(f"invalid device name: {device}")

    # 1. Checkpoint identity: fail closed before any GPU work.
    actual_sha = _sha256(args.checkpoint)
    if actual_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: expected %s, found %s"
            % (args.expected_checkpoint_sha256, actual_sha)
        )

    # 2. Resolved config / semantic hash, and the runtime topology override.
    config = load_config(args.run_dir / "resolved_config.json")
    semantic_hash = lineage_config_hash(config)
    original_runtime = {
        "actor_processes": config.runtime.actor_processes,
        "mcts_lanes_per_actor": config.runtime.mcts_lanes_per_actor,
        "selfplay_devices": list(config.runtime.selfplay_devices),
        "device": config.runtime.device,
        "inference_batch_size": config.runtime.inference_batch_size,
        "learner_amp": config.runtime.learner_amp,
    }

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    train_positions = int(checkpoint.extra_state["formal_loop_state"]["train_positions_consumed"])

    effective = config.selfplay.for_train_positions(train_positions)
    stage = effective.stage_for_generation(args.generation)
    # Replace the multi-generation schedule with a single fixed point so the
    # work load is identical on both machines.
    effective = replace(
        effective,
        search_schedule=(replace(stage, start_generation=0, games=args.games),),
    )
    runtime = replace(
        config.runtime,
        device=devices[0],
        selfplay_devices=devices,
        actor_processes=args.actors,
        mcts_lanes_per_actor=args.lanes,
        inference_batch_size=args.inference_batch_size,
    )
    config = replace(config, selfplay=effective, runtime=runtime)

    # The checkpoint's own weights are the producer. Using them directly keeps the
    # measurement independent of gate/commit artifacts, so both machines are
    # guaranteed to play from the identical, SHA-256-verified state.
    producer_state = checkpoint.model_state
    producer_id = str(checkpoint.extra_state.get("accepted_model_id") or "calibration-g57")

    if args.output_root.exists():
        raise FileExistsError(f"output root must not already exist: {args.output_root}")
    args.output_root.mkdir(parents=True)

    # 3. Measure.
    sampler = _ResourceSampler(0.5, tuple(dict.fromkeys(devices)))
    sampler.start()
    started = time.perf_counter()
    try:
        result = run_self_play_actor_pool(
            config,
            accepted_model_state=producer_state,
            producer_model_id=producer_id,
            generation=args.generation,
            start_game_id=args.game_id_base,
            inference_batch_timeout_s=args.inference_timeout_ms / 1000.0,
        )
    finally:
        resources = sampler.stop()
    wall_seconds = time.perf_counter() - started

    lengths = [len(game.moves) for game in result.games]
    simulations = sum(game.total_simulations for game in result.games)
    inference_positions = sum(row.positions for row in result.metrics.inference_services)
    inference_batches = sum(row.batches for row in result.metrics.inference_services)

    payload: dict[str, Any] = {
        "schema": "connect4-bal5-topology-calibration-v1",
        "label": args.label,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "cpu_count_visible": os.cpu_count(),
            "cgroup": _cpu_quota(),
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "gpu_total": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_total": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "provenance": {
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "run_dir": str(args.run_dir),
            "resolved_config_sha256": _sha256(args.run_dir / "resolved_config.json"),
            "semantic_config_hash": semantic_hash,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": actual_sha,
            "checkpoint_generation": args.generation,
            "producer_model_id": producer_id,
        },
        "semantics": {
            "actor_processes": args.actors,
            "selfplay_devices": list(devices),
            "actors_per_device": [
                sum(1 for actor in range(min(args.actors, args.games)) if actor % len(devices) == index)
                for index in range(len(devices))
            ],
            "mcts_lanes_per_actor": args.lanes,
            "inference_batch_size": args.inference_batch_size,
            "inference_timeout_ms": args.inference_timeout_ms,
            "full_search_sims": stage.full_search_sims,
            "fast_search_sims": stage.fast_search_sims,
            "full_probability": stage.full_probability,
            "games": args.games,
            "game_id_base": args.game_id_base,
            "original_runtime_topology": original_runtime,
            "learner_amp": config.runtime.learner_amp,
        },
        "result": {
            "games": len(result.games),
            "wall_seconds": wall_seconds,
            "games_per_second": len(result.games) / wall_seconds,
            "simulations": simulations,
            "simulations_per_second": simulations / wall_seconds,
            "raw_positions": sum(lengths),
            "raw_positions_per_second": sum(lengths) / wall_seconds,
            "mean_game_length": statistics.fmean(lengths),
            "min_game_length": min(lengths),
            "max_game_length": max(lengths),
            "inference_positions": inference_positions,
            "inference_batches": inference_batches,
            "mean_inference_batch": inference_positions / max(inference_batches, 1),
            "max_inference_batch": max(
                (row.max_batch for row in result.metrics.inference_services), default=0
            ),
            "inference_services": [
                {
                    "device": row.device,
                    "actor_count": row.actor_count,
                    "requests": row.requests,
                    "positions": row.positions,
                    "batches": row.batches,
                    "mean_batch": row.mean_batch,
                    "max_batch": row.max_batch,
                    "wall_seconds": row.wall_seconds,
                }
                for row in result.metrics.inference_services
            ],
            "actor_pool_wall_seconds": result.metrics.wall_seconds,
        },
        "resources": resources,
    }

    (args.output_root / "point.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "label": args.label,
                "actors": args.actors,
                "devices": list(devices),
                "games_per_second": round(payload["result"]["games_per_second"], 4),
                "simulations_per_second": round(payload["result"]["simulations_per_second"], 1),
                "raw_positions_per_second": round(payload["result"]["raw_positions_per_second"], 3),
                "wall_seconds": round(wall_seconds, 2),
                "gpu_util_mean": resources.get("gpu_util_mean"),
                "cpu_util_percent_of_quota": resources.get("cpu_util_percent_of_quota"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
