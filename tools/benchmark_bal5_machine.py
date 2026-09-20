"""Benchmark BAL-5 self-play topology and learner throughput on one machine.

The script is read-only with respect to source run directories. Models and
optimizer state are restored into memory; benchmark outputs go only below the
explicit output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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
from training.v3.config import load_config
from training.v3.layout import RunLayout
from training.v3.model import build_model
from training.v3.pipeline import (
    _build_active_datasets,
    _build_learner,
    _load_replay_from_cursor,
)


def _topologies(raw: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in raw.split(","):
        actors, lanes = (int(value) for value in item.lower().split("x", 1))
        if actors < 1 or lanes not in (4, 6):
            raise argparse.ArgumentTypeError("topologies require positive actors and 4 or 6 lanes")
        result.append((actors, lanes))
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("topologies must be non-empty and unique")
    return tuple(result)


def _case(raw: str) -> tuple[Path, int]:
    path, generation = raw.rsplit(":", 1)
    return Path(path), int(generation)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=_case, required=True)
    parser.add_argument("--adapt-case-index", type=int, default=1)
    parser.add_argument(
        "--topologies", type=_topologies, default=_topologies("16x4,18x4,20x4,16x6,18x6,20x6")
    )
    parser.add_argument("--adapt-games", type=int, default=64)
    parser.add_argument("--case-games", type=int, default=128)
    parser.add_argument("--learner-steps", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--inference-timeout-ms", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _commit(run_dir: Path, generation: int) -> dict[str, Any]:
    path = run_dir / "manifests" / "generations" / f"g{generation:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_state(run_dir: Path, commit: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = run_dir / str(commit["accepted_model_path"])
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return artifact["model_state"], str(commit["accepted_model_id"])


def _selfplay_point(
    run_dir: Path,
    generation: int,
    *,
    actors: int,
    lanes: int,
    games: int,
    device: str,
    batch_size: int,
    timeout_ms: float,
) -> dict[str, Any]:
    config = load_config(run_dir / "resolved_config.json")
    checkpoint = load_checkpoint(run_dir / str(_commit(run_dir, generation)["checkpoint"]), map_location="cpu")
    train_positions = int(checkpoint.extra_state["formal_loop_state"]["train_positions_consumed"])
    effective_selfplay = config.selfplay.for_train_positions(train_positions)
    stage = effective_selfplay.stage_for_generation(generation)
    effective_selfplay = replace(
        effective_selfplay,
        search_schedule=(replace(stage, start_generation=0, games=games),),
    )
    runtime = replace(
        config.runtime,
        device=device,
        selfplay_devices=(device,),
        actor_processes=actors,
        mcts_lanes_per_actor=lanes,
        inference_batch_size=batch_size,
    )
    config = replace(config, selfplay=effective_selfplay, runtime=runtime)
    state, producer_id = _artifact_state(run_dir, _commit(run_dir, generation))
    sampler = _ResourceSampler(0.5, (device,))
    sampler.start()
    try:
        result = run_self_play_actor_pool(
            config,
            accepted_model_state=state,
            producer_model_id=producer_id,
            generation=generation,
            inference_batch_timeout_s=timeout_ms / 1000.0,
        )
    finally:
        resources = sampler.stop()
    lengths = [len(game.moves) for game in result.games]
    simulations = sum(game.total_simulations for game in result.games)
    inference_positions = sum(row.positions for row in result.metrics.inference_services)
    inference_batches = sum(row.batches for row in result.metrics.inference_services)
    return {
        "run": run_dir.name,
        "generation": generation,
        "actors": actors,
        "lanes": lanes,
        "games": len(result.games),
        "wall_seconds": result.metrics.wall_seconds,
        "games_per_second": len(result.games) / result.metrics.wall_seconds,
        "simulations": simulations,
        "simulations_per_second": simulations / result.metrics.wall_seconds,
        "mean_game_length": statistics.fmean(lengths),
        "raw_positions_per_second": sum(lengths) / result.metrics.wall_seconds,
        "inference_positions": inference_positions,
        "inference_batches": inference_batches,
        "mean_inference_batch": inference_positions / max(inference_batches, 1),
        "max_inference_batch": max(
            (row.max_batch for row in result.metrics.inference_services), default=0
        ),
        "search": {
            "full_sims": stage.full_search_sims,
            "fast_sims": stage.fast_search_sims,
            "full_probability": stage.full_probability,
        },
        "resources": resources,
    }


def _learner_point(
    run_dir: Path,
    generation: int,
    *,
    steps: int,
    device: str,
) -> dict[str, Any]:
    config = load_config(run_dir / "resolved_config.json")
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            device=device,
            selfplay_devices=(device,),
        ),
    )
    commit = _commit(run_dir, generation)
    checkpoint_path = run_dir / str(commit["checkpoint"])
    saved = load_checkpoint(checkpoint_path, map_location=device)
    layout = RunLayout.from_root(run_dir)
    replay = _load_replay_from_cursor(
        layout,
        saved.replay_cursor,
        config=config,
        expected_hash=saved.config_hash,
    )
    dataset, _validation, selection = _build_active_datasets(replay, config)
    model = build_model(config.model)
    learner, optimizer = _build_learner(config, model)
    saved.restore(
        model=model,
        optimizer=optimizer,
        scaler=learner.scaler,
        expected_config_hash=saved.config_hash,
    )
    learner.load_state_dict(saved.extra_state["learner_state"])
    sampler = _ResourceSampler(0.5, (device,))
    sampler.start()
    try:
        metrics = learner.train_steps(dataset, steps=steps)
    finally:
        resources = sampler.stop()
    return {
        "run": run_dir.name,
        "generation": generation,
        "checkpoint": str(checkpoint_path),
        "steps_requested": steps,
        "active_replay": selection,
        "metrics": metrics.to_dict(),
        "resources": resources,
    }


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("BAL-5 machine benchmark requires CUDA")
    if not 0 <= args.adapt_case_index < len(args.case):
        raise ValueError("adapt-case-index is out of range")
    if min(args.adapt_games, args.case_games, args.learner_steps) < 1:
        raise ValueError("games and learner steps must be positive")
    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root / "benchmark.json"
    payload: dict[str, Any] = {
        "schema": "connect4-bal5-machine-benchmark-v1",
        "hardware": {
            "gpu": torch.cuda.get_device_name(torch.device(args.device).index or 0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "os_cpu_count": os.cpu_count(),
            "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        },
        "contract": {
            "cases": [[str(path), generation] for path, generation in args.case],
            "adapt_case_index": args.adapt_case_index,
            "topologies": args.topologies,
            "adapt_games": args.adapt_games,
            "case_games": args.case_games,
            "learner_steps": args.learner_steps,
            "inference_batch_size": args.inference_batch_size,
            "inference_timeout_ms": args.inference_timeout_ms,
        },
        "topology_scan": [],
        "selfplay": [],
        "learner": [],
    }
    adapt_run, adapt_generation = args.case[args.adapt_case_index]
    for actors, lanes in args.topologies:
        point = _selfplay_point(
            adapt_run,
            adapt_generation,
            actors=actors,
            lanes=lanes,
            games=args.adapt_games,
            device=args.device,
            batch_size=args.inference_batch_size,
            timeout_ms=args.inference_timeout_ms,
        )
        payload["topology_scan"].append(point)
        _write(output, payload)
        print(json.dumps({"stage": "topology", **point}, sort_keys=True), flush=True)
        torch.cuda.empty_cache()
    peak = max(payload["topology_scan"], key=lambda row: row["simulations_per_second"])
    original_lanes = load_config(adapt_run / "resolved_config.json").runtime.mcts_lanes_per_actor
    compatible = [row for row in payload["topology_scan"] if row["lanes"] == original_lanes]
    if not compatible:
        raise RuntimeError("topology scan omitted the lineage-compatible lane count")
    selected = max(compatible, key=lambda row: row["simulations_per_second"])
    payload["throughput_peak_topology"] = {"actors": peak["actors"], "lanes": peak["lanes"]}
    payload["selected_resume_topology"] = {
        "actors": selected["actors"],
        "lanes": selected["lanes"],
        "reason": "highest simulations_per_second at the checkpointed semantic lane count",
    }
    _write(output, payload)
    for run_dir, generation in args.case:
        point = _selfplay_point(
            run_dir,
            generation,
            actors=selected["actors"],
            lanes=selected["lanes"],
            games=args.case_games,
            device=args.device,
            batch_size=args.inference_batch_size,
            timeout_ms=args.inference_timeout_ms,
        )
        payload["selfplay"].append(point)
        _write(output, payload)
        print(json.dumps({"stage": "selfplay", **point}, sort_keys=True), flush=True)
        torch.cuda.empty_cache()
    for run_dir, generation in args.case:
        point = _learner_point(
            run_dir,
            generation,
            steps=args.learner_steps,
            device=args.device,
        )
        payload["learner"].append(point)
        _write(output, payload)
        print(json.dumps({"stage": "learner", **point}, sort_keys=True), flush=True)
        torch.cuda.empty_cache()
    payload["completed"] = True
    payload["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
