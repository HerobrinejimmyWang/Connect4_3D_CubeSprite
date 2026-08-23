"""Benchmark game-parallel V3 evaluation without changing per-game search semantics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.anchored_elo import (  # noqa: E402
    LegacyCheckpointPredictor,
    evaluator_code_hash,
    load_anchored_config,
    load_v3_artifact_predictor,
    verify_anchor_files,
    verify_opening_suite,
)
from connect4_core.rules import CLASSIC_RULE, DEFAULT_RULE_REGISTRY  # noqa: E402
from training.v3.evaluation import play_paired_openings  # noqa: E402
from training.v3.evaluation_runtime import play_paired_openings_parallel  # noqa: E402


def _topologies(raw: str) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    for item in raw.split(","):
        try:
            games, batch = (int(value) for value in item.lower().split("x", 1))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError("topologies must look like 4x8,8x16,12x32") from exc
        if games < 1 or batch < 1:
            raise argparse.ArgumentTypeError("parallel games and batch size must be positive")
        values.append((games, batch))
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("topologies must be non-empty and unique")
    return tuple(values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "training/v3/configs/anchored_elo_historical_v1.json",
    )
    parser.add_argument("--profile", default="primary_256")
    parser.add_argument("--model-a", default="anchor:v2_2_balance")
    parser.add_argument("--model-b", default="anchor:cubesprite_v3_iter240")
    parser.add_argument("--model-a-id", default=None)
    parser.add_argument("--model-b-id", default=None)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topologies", type=_topologies, default=_topologies("4x8,8x16,12x32"))
    parser.add_argument("--batch-timeout-ms", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cpu_quota() -> float:
    raw = _read_text("/sys/fs/cgroup/cpu.max")
    if raw:
        try:
            quota, period = raw.split()
            if quota != "max":
                return float(quota) / float(period)
        except ValueError:
            pass
    return float(os.cpu_count() or 1)


def _cpu_usage_usec() -> int:
    raw = _read_text("/sys/fs/cgroup/cpu.stat") or ""
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "usage_usec":
            return int(fields[1])
    return 0


class _ResourceProbe:
    def __init__(self, device: str) -> None:
        self.device = device
        self.stop_event = threading.Event()
        self.samples: list[tuple[float, float, float]] = []
        self.cpu_start = 0
        self.started = 0.0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self.cpu_start = _cpu_usage_usec()
        self.started = time.perf_counter()
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        wall = max(time.perf_counter() - self.started, 1e-12)
        used = max(0, _cpu_usage_usec() - self.cpu_start) / 1e6
        return {
            "wall_seconds": wall,
            "cpu_quota_cores": _cpu_quota(),
            "cpu_used_core_seconds": used,
            "cpu_util_percent_of_quota": 100.0 * used / wall / _cpu_quota(),
            "gpu_sample_count": len(self.samples),
            "gpu_util_mean": statistics.fmean(row[0] for row in self.samples)
            if self.samples
            else 0.0,
            "gpu_util_max": max((row[0] for row in self.samples), default=0.0),
            "gpu_memory_used_mib_max": max((row[1] for row in self.samples), default=0.0),
            "gpu_power_w_mean": statistics.fmean(row[2] for row in self.samples)
            if self.samples
            else 0.0,
        }

    def _sample(self) -> None:
        if not self.device.startswith("cuda"):
            return
        index = torch.device(self.device).index or 0
        while not self.stop_event.is_set():
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={index}",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                ).stdout.strip()
                self.samples.append(tuple(float(value.strip()) for value in output.split(",")))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(0.5)


def _hardware(device: str) -> dict[str, Any]:
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    memory_max = _read_text("/sys/fs/cgroup/memory.max")
    gpu: dict[str, Any] | None = None
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "os_cpu_count": os.cpu_count(),
        "cgroup_cpu_max": cpu_max,
        "cgroup_memory_max": memory_max,
        "gpu": gpu,
        "torch": torch.__version__,
    }


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_model(
    spec: str,
    *,
    requested_id: str | None,
    anchors: dict[str, dict[str, Any]],
    device: str,
) -> tuple[Any, dict[str, Any]]:
    kind, separator, value = spec.partition(":")
    if not separator or not value:
        raise ValueError("model spec must be anchor:<id> or v3:<checkpoint>")
    if kind == "anchor":
        if value not in anchors:
            raise ValueError(f"unknown frozen anchor: {value}")
        if requested_id is not None and requested_id != value:
            raise ValueError("a frozen anchor cannot be renamed")
        identity = dict(anchors[value])
        return LegacyCheckpointPredictor(identity["path"], device=device), identity
    if kind == "v3":
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        predictor, identity = load_v3_artifact_predictor(path, device=device)
        if requested_id is not None:
            identity["model_id"] = requested_id
        return predictor, identity
    raise ValueError("model spec must use the anchor or v3 prefix")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pair_start < 0 or args.pair_count < 1 or args.batch_timeout_ms < 0.0:
        raise ValueError("pair range and batch timeout are invalid")
    config = load_anchored_config(args.config)
    anchors = {row["model_id"]: row for row in verify_anchor_files(config, ROOT)}
    profile = config.profile(args.profile)
    all_openings = verify_opening_suite(config, ROOT)
    stop = args.pair_start + args.pair_count
    if stop > min(len(all_openings), profile.max_pairs):
        raise ValueError("pair range exceeds the frozen profile")
    openings = all_openings[args.pair_start:stop]
    predictor_a, identity_a = _load_model(
        args.model_a,
        requested_id=args.model_a_id,
        anchors=anchors,
        device=args.device,
    )
    predictor_b, identity_b = _load_model(
        args.model_b,
        requested_id=args.model_b_id,
        anchors=anchors,
        device=args.device,
    )
    if identity_a["model_id"] == identity_b["model_id"]:
        raise ValueError("benchmark models must have distinct model ids")
    warmup_board = np.zeros((1, 6, 5, 5), dtype=np.int8)
    warmup_role = np.asarray(((1.0, 0.0),), dtype=np.float32)
    warmup_rules = np.asarray(
        (DEFAULT_RULE_REGISTRY.features(CLASSIC_RULE),), dtype=np.float32
    )
    for predictor in (predictor_a, predictor_b):
        predictor.predict_batch(
            warmup_board,
            role_to_play=warmup_role,
            rule_features=warmup_rules,
        )

    reference_probe = _ResourceProbe(args.device)
    reference_probe.start()
    try:
        started = time.perf_counter()
        reference = play_paired_openings(
            openings,
            candidate_predictor=predictor_a,
            incumbent_predictor=predictor_b,
            search_sims=profile.search_sims,
            cpuct=profile.cpuct,
        )
        reference_wall = time.perf_counter() - started
    finally:
        reference_resources = reference_probe.stop()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "git_revision": _git_revision(),
        "evaluator_code_hash": evaluator_code_hash(ROOT),
        "hardware": _hardware(args.device),
        "contract": {
            "profile": asdict(profile),
            "model_a": identity_a,
            "model_b": identity_b,
            "opening_ids": [opening.opening_id for opening in openings],
            "seeds": [opening.seed for opening in openings],
            "mcts_lanes_per_game": 1,
            "batch_timeout_ms": args.batch_timeout_ms,
            "warmup_positions_per_model": 1,
        },
        "serial_reference": {
            "wall_seconds": reference_wall,
            "resources": reference_resources,
            "games": [asdict(row) for row in reference],
        },
        "results": [],
    }
    for parallel_games, batch_size in args.topologies:
        probe = _ResourceProbe(args.device)
        probe.start()
        try:
            evaluated = play_paired_openings_parallel(
                openings,
                candidate_predictor=predictor_a,
                incumbent_predictor=predictor_b,
                search_sims=profile.search_sims,
                cpuct=profile.cpuct,
                parallel_games=parallel_games,
                inference_batch_size=batch_size,
                inference_batch_timeout_s=args.batch_timeout_ms / 1000.0,
            )
        finally:
            resources = probe.stop()
        mismatches = sum(left != right for left, right in zip(reference, evaluated.games))
        row = {
            "parallel_games": parallel_games,
            "inference_batch_size": batch_size,
            "speedup_vs_serial": reference_wall / max(evaluated.metrics.wall_seconds, 1e-12),
            "exact_result_match": mismatches == 0,
            "result_mismatches": mismatches,
            "metrics": evaluated.metrics.to_dict(),
            "resources": resources,
            "games": [asdict(game) for game in evaluated.games],
        }
        payload["results"].append(row)
        _write(args.output, payload)
        print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
