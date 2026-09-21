"""BAL-5 learner semantic calibration: repeated identical 256-step measurements.

The learner contract is FIXED and must not be tuned for a better score:
batch_size=256, FP32 (learner_amp=false), grad clip, loss weights, LR schedule,
replay recipe and token reuse are all inherited from the source run's resolved
config. Only the operational device is supplied by the caller.

Each invocation performs exactly one 256-step measurement and writes one JSON
record, so the caller controls how many repeats happen and can interleave them
with the other machine's measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from benchmark_v3_selfplay_topology import _ResourceSampler
from training.v3.checkpoint import load_checkpoint
from training.v3.config import load_config
from training.v3.layout import RunLayout
from training.v3.model import build_model
from training.v3.pipeline import (
    _build_active_datasets,
    _build_learner,
    _load_replay_from_cursor,
    lineage_config_hash,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=str(ROOT)
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_dirty() -> str:
    try:
        return subprocess.run(
            ["git", "status", "--short"], check=True, capture_output=True, text=True, cwd=str(ROOT)
        ).stdout.strip()
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
    return {"cpu_max": quota, "quota_cores": cores, "memory_max": read("/sys/fs/cgroup/memory.max")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.steps != 256:
        raise RuntimeError("learner calibration fixes 256 steps; do not change the work load")

    actual_sha = _sha256(args.checkpoint)
    if actual_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: expected %s, found %s"
            % (args.expected_checkpoint_sha256, actual_sha)
        )

    config = load_config(args.run_dir / "resolved_config.json")
    if config.learner.batch_size != 256:
        raise RuntimeError("learner calibration requires the source batch_size=256")
    if config.runtime.learner_amp:
        raise RuntimeError("learner calibration requires FP32 (learner_amp=false)")
    semantic_hash = lineage_config_hash(config)

    config = replace(
        config,
        runtime=replace(config.runtime, device=args.device, selfplay_devices=(args.device,)),
    )

    saved = load_checkpoint(args.checkpoint, map_location=args.device)
    layout = RunLayout.from_root(args.run_dir)
    loaded = _load_replay_from_cursor(
        layout,
        saved.replay_cursor,
        config=config,
        expected_hash=saved.config_hash,
    )
    dataset, _validation, selection = _build_active_datasets(loaded, config)
    model = build_model(config.model)
    learner, optimizer = _build_learner(config, model)
    saved.restore(
        model=model,
        optimizer=optimizer,
        scaler=learner.scaler,
        expected_config_hash=saved.config_hash,
    )
    learner.load_state_dict(saved.extra_state["learner_state"])

    if args.output_root.exists():
        raise FileExistsError(f"output root must not already exist: {args.output_root}")
    args.output_root.mkdir(parents=True)

    sampler = _ResourceSampler(0.5, (args.device,))
    sampler.start()
    started = time.perf_counter()
    try:
        metrics = learner.train_steps(dataset, steps=args.steps)
    finally:
        resources = sampler.stop()
    wall_seconds = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema": "connect4-bal5-learner-calibration-v1",
        "label": args.label,
        "repeat_index": args.repeat_index,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "cpu_count_visible": os.cpu_count(),
            "cgroup": _cpu_quota(),
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
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
        },
        "semantics": {
            "batch_size": config.learner.batch_size,
            "learner_amp": config.runtime.learner_amp,
            "device": args.device,
            "steps": args.steps,
            "max_optimizer_steps_per_cycle": config.learner.max_optimizer_steps_per_cycle,
            "grad_clip_norm": config.learner.grad_clip_norm,
            "lr_schedule": [
                {
                    "start_train_positions": stage.start_train_positions,
                    "learning_rate": stage.learning_rate,
                }
                for stage in config.learner.lr_schedule
            ],
        },
        "active_replay": selection,
        "result": {
            "wall_seconds": wall_seconds,
            "steps": metrics.steps,
            "positions": metrics.positions,
            "positions_per_second": metrics.positions / wall_seconds,
            "steps_per_second": metrics.steps / wall_seconds,
            "metrics": metrics.to_dict(),
        },
        "resources": resources,
    }

    (args.output_root / "learner.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "label": args.label,
                "repeat_index": args.repeat_index,
                "steps": metrics.steps,
                "positions": metrics.positions,
                "positions_per_second": round(payload["result"]["positions_per_second"], 2),
                "wall_seconds": round(wall_seconds, 3),
                "gpu_util_mean": resources.get("gpu_util_mean"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
