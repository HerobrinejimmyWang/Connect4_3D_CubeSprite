"""Short V3 self-play topology scan with cgroup-aware resource metrics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.actor_runtime import run_self_play_actor_pool
from training.v3.config import (
    ExplorationPhaseConfig,
    ModelConfig,
    SearchStageConfig,
    V3Config,
)
from training.v3.model import build_model


def _topologies(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for raw in value.split(","):
        try:
            actors, lanes = (int(item) for item in raw.lower().split("x", 1))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "topologies must look like 8x2,12x2,16x2,8x4"
            ) from exc
        if actors < 1 or lanes < 1:
            raise argparse.ArgumentTypeError("actors and lanes must be positive")
        result.append((actors, lanes))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("topologies must be non-empty and unique")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        default="",
        help="optional comma-separated self-play service devices, e.g. cuda:0,cuda:1",
    )
    parser.add_argument("--topologies", type=_topologies, default=_topologies("8x2,12x2,16x2,8x4"))
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--full-sims", type=int, default=32)
    parser.add_argument("--fast-sims", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--inference-timeout-ms", type=float, default=2.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args()


def _read_key_values(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return values


def _cpu_quota() -> float:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if quota != "max":
            return float(quota) / float(period)
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _cuda_index(device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    if ":" not in str(device):
        return 0
    try:
        return int(str(device).split(":", 1)[1])
    except ValueError:
        return None


def _selfplay_devices(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.devices.strip():
        return ()
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must contain unique non-empty device names")
    return devices


class _ResourceSampler:
    def __init__(self, interval: float, devices: tuple[str, ...]) -> None:
        self.interval = max(0.2, float(interval))
        self.selected_gpu_indices = {
            index for index in (_cuda_index(device) for device in devices) if index is not None
        }
        self.cpu_quota = _cpu_quota()
        self.cpu_start: dict[str, int] = {}
        self.samples: list[list[dict[str, float]]] = []
        self.memory_samples: list[int] = []
        self.started = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.cpu_start = _read_key_values(Path("/sys/fs/cgroup/cpu.stat"))
        self.started = time.perf_counter()
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.interval * 2.0))
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        cpu_end = _read_key_values(Path("/sys/fs/cgroup/cpu.stat"))
        usage = max(0, cpu_end.get("usage_usec", 0) - self.cpu_start.get("usage_usec", 0))
        throttled = max(
            0,
            cpu_end.get("throttled_usec", 0) - self.cpu_start.get("throttled_usec", 0),
        )
        periods = max(0, cpu_end.get("nr_periods", 0) - self.cpu_start.get("nr_periods", 0))
        throttled_periods = max(
            0,
            cpu_end.get("nr_throttled", 0) - self.cpu_start.get("nr_throttled", 0),
        )
        flattened = [gpu for sample in self.samples for gpu in sample]
        selected = [
            gpu
            for sample in self.samples
            for gpu in sample
            if gpu["index"] in self.selected_gpu_indices
        ]
        active = [sample for sample in selected if sample["memory_used_mib"] >= 100.0]

        def summary(name: str, rows: list[dict[str, float]]) -> dict[str, float]:
            if not rows:
                return {f"{name}_mean": 0.0, f"{name}_max": 0.0}
            return {
                f"{name}_mean": statistics.fmean(row[name] for row in rows),
                f"{name}_max": max(row[name] for row in rows),
            }

        return {
            "wall_seconds": elapsed,
            "sample_count": len(selected),
            "active_gpu_samples": len(active),
            "selected_gpu_indices": sorted(self.selected_gpu_indices),
            "cpu_quota_cores": self.cpu_quota,
            "cpu_util_percent_of_quota": 100.0 * (usage / 1e6) / elapsed / self.cpu_quota,
            "cpu_used_core_seconds": usage / 1e6,
            "cpu_throttled_seconds": throttled / 1e6,
            "cpu_throttled_period_ratio": throttled_periods / max(periods, 1),
            "memory_current_mib_max": max(self.memory_samples, default=0) / (1024.0**2),
            **summary("gpu_util", selected),
            "active_gpu_util_mean": (
                statistics.fmean(row["gpu_util"] for row in active) if active else 0.0
            ),
            "active_gpu_util_max": max(
                (row["gpu_util"] for row in active), default=0.0
            ),
            **summary("memory_used_mib", selected),
            **summary("power_w", selected),
            "gpus": [
                {
                    "index": index,
                    **summary(
                        "gpu_util", [gpu for gpu in flattened if gpu["index"] == index]
                    ),
                    **summary(
                        "memory_used_mib",
                        [gpu for gpu in flattened if gpu["index"] == index],
                    ),
                    **summary(
                        "power_w", [gpu for gpu in flattened if gpu["index"] == index]
                    ),
                }
                for index in sorted({int(gpu["index"]) for gpu in flattened})
            ],
        }

    def _loop(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        memory_path = Path("/sys/fs/cgroup/memory.current")
        while not self.stop_event.is_set():
            try:
                completed = subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=5.0
                )
                sample = []
                for line in completed.stdout.splitlines():
                    gpu_index, gpu_util, memory_used, power = (
                        float(value.strip()) for value in line.split(",")
                    )
                    sample.append(
                        {
                            "index": int(gpu_index),
                            "gpu_util": gpu_util,
                            "memory_used_mib": memory_used,
                            "power_w": power,
                        }
                    )
                if sample:
                    self.samples.append(sample)
                self.memory_samples.append(int(memory_path.read_text(encoding="utf-8")))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval)


def _config(args: argparse.Namespace, actors: int, lanes: int) -> V3Config:
    base = V3Config()
    return replace(
        base,
        run=replace(base.run, seed=args.seed),
        model=ModelConfig("gravity_resnet", args.channels, args.blocks),
        selfplay=replace(
            base.selfplay,
            search_schedule=(
                SearchStageConfig(0, args.games, args.full_sims, args.fast_sims, 0.5),
            ),
            exploration_phases=(
                ExplorationPhaseConfig(0, 1.0, 0.24, 0.060),
                ExplorationPhaseConfig(28, 0.5, 0.5, 0.005),
                ExplorationPhaseConfig(50, 0.0, 0.0, 0.0),
            ),
        ),
        runtime=replace(
            base.runtime,
            device=args.device,
            selfplay_devices=_selfplay_devices(args),
            actor_processes=actors,
            mcts_lanes_per_actor=lanes,
            inference_batch_size=args.inference_batch_size,
            deterministic=False,
            learner_amp=True,
        ),
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if min(args.games, args.channels, args.blocks, args.full_sims, args.fast_sims) < 1:
        raise ValueError("games, model dimensions, and simulation counts must be positive")
    torch.manual_seed(args.seed)
    model_config = ModelConfig("gravity_resnet", args.channels, args.blocks)
    accepted_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in build_model(model_config).state_dict().items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "git_revision": _git_revision(),
        "config": {
            "device": args.device,
            "selfplay_devices": list(_selfplay_devices(args)),
            "topologies": args.topologies,
            "games_per_point": args.games,
            "model": f"{args.channels}x{args.blocks}",
            "full_sims": args.full_sims,
            "fast_sims": args.fast_sims,
            "inference_batch_size": args.inference_batch_size,
            "inference_timeout_ms": args.inference_timeout_ms,
            "seed": args.seed,
        },
        "results": [],
    }
    for actors, lanes in args.topologies:
        service_devices = _selfplay_devices(args) or (args.device,)
        sampler = _ResourceSampler(args.sample_interval, service_devices)
        sampler.start()
        try:
            result = run_self_play_actor_pool(
                _config(args, actors, lanes),
                accepted_model_state=accepted_state,
                producer_model_id="v3-topology-fixed-model",
                inference_batch_timeout_s=args.inference_timeout_ms / 1000.0,
            )
        finally:
            resources = sampler.stop()
        lengths = [len(game.moves) for game in result.games]
        services = result.metrics.inference_services
        inference_positions = sum(service.positions for service in services)
        inference_batches = sum(service.batches for service in services)
        point = {
            "actors": actors,
            "lanes": lanes,
            "games": len(result.games),
            "wall_seconds": result.metrics.wall_seconds,
            "games_per_second": len(result.games) / result.metrics.wall_seconds,
            "mean_game_length": statistics.fmean(lengths),
            "game_length_stdev": statistics.pstdev(lengths),
            "p1_wins": sum(game.winner == 1 for game in result.games),
            "p2_wins": sum(game.winner == -1 for game in result.games),
            "draws": sum(game.is_draw for game in result.games),
            "simulations": sum(game.total_simulations for game in result.games),
            "simulations_per_second": (
                sum(game.total_simulations for game in result.games)
                / result.metrics.wall_seconds
            ),
            "inference": {
                "service_count": len(services),
                "positions": inference_positions,
                "batches": inference_batches,
                "mean_batch": inference_positions / max(inference_batches, 1),
                "max_batch": max((service.max_batch for service in services), default=0),
                "services": [service.to_dict() for service in services],
            },
            "resources": resources,
        }
        payload["results"].append(point)
        _atomic_write(args.output, payload)
        print(json.dumps(point, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
