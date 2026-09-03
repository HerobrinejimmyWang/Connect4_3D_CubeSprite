"""Compare frozen-model self-play search/exploration arms without writing replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.actor_runtime import run_self_play_actor_pool
from training.v3.config import (
    ExplorationPhaseConfig,
    SearchStageConfig,
    config_hash,
    load_config,
    model_config_dict,
)
from training.v3.pipeline import _selfplay_health


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--start-game-id", type=int, default=8_000_000)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--actor-processes", type=int, default=40)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--inference-timeout-ms", type=float, default=1.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _phase_rows(phases: tuple[ExplorationPhaseConfig, ...]) -> list[dict[str, Any]]:
    return [asdict(phase) for phase in phases]


def main() -> None:
    args = _args()
    if min(args.games, args.actor_processes, args.lanes, args.inference_batch_size) < 1:
        raise ValueError("games and runtime counts must be positive")
    if args.start_game_id < 0 or args.inference_timeout_ms < 0.0:
        raise ValueError("start-game-id and inference timeout must be non-negative")
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must contain unique non-empty device names")

    config = load_config(args.config)
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    if not isinstance(snapshot, dict) or snapshot.get("format") != "connect4-v3-model":
        raise ValueError("snapshot is not a V3 evaluation artifact")
    if dict(snapshot.get("model_config", {})) != model_config_dict(config.model):
        raise ValueError("snapshot model_config differs from the diagnostic config")
    model_state = snapshot.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("snapshot model_state is missing")
    model_id = str(snapshot.get("metadata", {}).get("model_id") or args.snapshot.stem)

    base_stage = config.selfplay.stage_for_generation(0)
    base_phases = tuple(config.selfplay.exploration_phases)
    no_noise_phases = tuple(
        replace(phase, dirichlet_epsilon=0.0) for phase in base_phases
    )
    arms = (
        {
            "id": "baseline_mixed_128_32",
            "full_sims": base_stage.full_search_sims,
            "fast_sims": base_stage.fast_search_sims,
            "full_probability": base_stage.full_probability,
            "force_full_before_ply": 0,
            "phases": base_phases,
        },
        {
            "id": "opening12_full_then_mixed",
            "full_sims": base_stage.full_search_sims,
            "fast_sims": base_stage.fast_search_sims,
            "full_probability": base_stage.full_probability,
            "force_full_before_ply": 12,
            "phases": base_phases,
        },
        {
            "id": "all_full_128",
            "full_sims": base_stage.full_search_sims,
            "fast_sims": base_stage.fast_search_sims,
            "full_probability": 1.0,
            "force_full_before_ply": 0,
            "phases": base_phases,
        },
        {
            "id": "mixed_128_32_no_root_noise",
            "full_sims": base_stage.full_search_sims,
            "fast_sims": base_stage.fast_search_sims,
            "full_probability": base_stage.full_probability,
            "force_full_before_ply": 0,
            "phases": no_noise_phases,
        },
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_only": True,
        "writes_replay": False,
        "git_revision": _git_revision(),
        "source_sha256": {
            relative: _sha256(ROOT / relative)
            for relative in (
                "tools/diagnose_v3_selfplay_search.py",
                "training/v3/actor_runtime.py",
                "training/v3/selfplay.py",
            )
        },
        "formal_config_hash": config_hash(config),
        "snapshot": {
            "path": str(args.snapshot.resolve()),
            "sha256": _sha256(args.snapshot),
            "model_id": model_id,
            "metadata": snapshot.get("metadata", {}),
        },
        "common": {
            "games": args.games,
            "start_game_id": args.start_game_id,
            "devices": list(devices),
            "actor_processes": args.actor_processes,
            "lanes": args.lanes,
            "inference_batch_size": args.inference_batch_size,
            "inference_timeout_ms": args.inference_timeout_ms,
            "paired_game_seeds": True,
        },
        "arms": [],
    }

    expected_ids = tuple(range(args.start_game_id, args.start_game_id + args.games))
    expected_seeds: tuple[int, ...] | None = None
    for arm in arms:
        stage = SearchStageConfig(
            0,
            args.games,
            int(arm["full_sims"]),
            int(arm["fast_sims"]),
            float(arm["full_probability"]),
        )
        phases = tuple(arm["phases"])
        arm_config = replace(
            config,
            selfplay=replace(
                config.selfplay,
                search_schedule=(stage,),
                exploration_phases=phases,
            ),
            runtime=replace(
                config.runtime,
                device=devices[0],
                selfplay_devices=devices,
                actor_processes=args.actor_processes,
                mcts_lanes_per_actor=args.lanes,
                inference_batch_size=args.inference_batch_size,
                deterministic=False,
            ),
        )
        started = time.perf_counter()
        result = run_self_play_actor_pool(
            arm_config,
            accepted_model_state=model_state,
            producer_model_id=model_id,
            start_game_id=args.start_game_id,
            generation=0,
            inference_batch_timeout_s=args.inference_timeout_ms / 1000.0,
            force_full_search_before_ply=int(arm["force_full_before_ply"]),
        )
        ids = tuple(game.game_id for game in result.games)
        seeds = tuple(game.seed for game in result.games)
        if ids != expected_ids:
            raise RuntimeError("diagnostic arm returned a different game ID set")
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise RuntimeError("diagnostic arms did not use identical paired game seeds")
        health = _selfplay_health(
            result.games,
            expected_search_sims={"full": stage.full_search_sims, "fast": stage.fast_search_sims},
            exploration_phases=_phase_rows(phases),
        )
        row = {
            "arm_id": arm["id"],
            "search": {
                "full_sims": stage.full_search_sims,
                "fast_sims": stage.fast_search_sims,
                "full_probability": stage.full_probability,
                "force_full_search_before_ply": arm["force_full_before_ply"],
                "exploration_phases": _phase_rows(phases),
            },
            "wall_seconds": time.perf_counter() - started,
            "actor_runtime": result.metrics.to_dict(),
            "health": health,
            "full_search_positions": sum(game.full_search_positions for game in result.games),
            "fast_search_positions": sum(game.fast_search_positions for game in result.games),
            "total_simulations": sum(game.total_simulations for game in result.games),
            "games": [
                {
                    "game_id": game.game_id,
                    "seed": game.seed,
                    "turns": len(game.moves),
                    "winner": game.winner,
                    "full_search_positions": game.full_search_positions,
                    "fast_search_positions": game.fast_search_positions,
                    "total_simulations": game.total_simulations,
                }
                for game in result.games
            ],
        }
        payload["arms"].append(row)
        _write(args.output, payload)
        print(
            json.dumps(
                {
                    "arm_id": row["arm_id"],
                    "wall_seconds": row["wall_seconds"],
                    "game_length": health["game_length"],
                    "results": health["results"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
