"""Collect a locked, calibration-only Replay V2 pool without learner promotion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from itertools import chain
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.actor_runtime import run_self_play_actor_pool  # noqa: E402
from training.v3.config import config_hash, load_config  # noqa: E402
from training.v3.model import build_model  # noqa: E402
from training.v3.pipeline import (  # noqa: E402
    _atomic_save_model_artifact,
    _atomic_write_json,
    _git_commit,
    _result_counts,
    _seed_runtime,
    _sha256_file,
    _utc_now,
)
from training.v3.preflight import run_preflight  # noqa: E402
from training.v3.replay import (  # noqa: E402
    ReplayShard,
    replay_manifest_path,
    replay_ready_path,
    write_replay_shard,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V3 P6 calibration replay collector")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=768)
    parser.add_argument("--batch-games", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.games < 1 or args.batch_games < 1:
            raise ValueError("games and batch-games must be positive")
        config = load_config(args.config)
        for device in dict.fromkeys((config.runtime.device, *config.runtime.selfplay_devices)):
            run_preflight(device)
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite P6 collection: {output}")
        replay_dir = output / "replay" / "raw"
        replay_dir.mkdir(parents=True)
        _seed_runtime(config)
        model = build_model(config.model)
        model_state = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
        lineage = config_hash(config)
        producer_id = f"p6-calibration-frozen-random-{lineage[:12]}"
        producer_path = output / "producer" / f"{producer_id}.pt"
        _atomic_save_model_artifact(
            producer_path,
            model=model,
            model_config=asdict(config.model),
            metadata={
                "producer_model_id": producer_id,
                "purpose": "P6 calibration only; never eligible for formal replay or promotion",
                "config_hash": lineage,
            },
        )
        code_commit = _git_commit()
        rows: list[dict[str, object]] = []
        next_game_id = 0
        total_positions = 0
        batch_index = 0
        while next_game_id < args.games:
            count = min(args.batch_games, args.games - next_game_id)
            stage = replace(config.selfplay.stage_for_generation(0), games=count)
            batch_config = replace(
                config,
                selfplay=replace(config.selfplay, search_schedule=(stage,)),
                runtime=replace(config.runtime, actor_processes=min(config.runtime.actor_processes, count)),
            )
            batch = run_self_play_actor_pool(
                batch_config,
                accepted_model_state=model_state,
                producer_model_id=producer_id,
                start_game_id=next_game_id,
                generation=batch_index,
            )
            games = list(batch.games)
            shard = ReplayShard.from_samples(chain.from_iterable(game.samples for game in games))
            shard_path = replay_dir / (
                f"p6_b{batch_index:04d}_games{games[0].game_id:08d}-{games[-1].game_id:08d}.npz"
            )
            manifest = write_replay_shard(
                shard_path,
                shard,
                {
                    "run_id": f"p6_calibration_{lineage[:12]}",
                    "generation": batch_index,
                    "producer_model_id": producer_id,
                    "seed_range": {"start": games[0].seed, "end": games[-1].seed},
                    "results": _result_counts(games),
                    "search_config": {
                        "active_stage": asdict(stage),
                        "exploration_phases": [asdict(row) for row in config.selfplay.exploration_phases],
                        "cpuct": config.selfplay.cpuct,
                        "virtual_loss": config.selfplay.virtual_loss,
                        "mcts_lanes_per_actor": config.runtime.mcts_lanes_per_actor,
                    },
                    "position_range": {
                        "start": total_positions,
                        "end": total_positions + len(shard),
                    },
                    "rule_registry_hash": config.selfplay.rule_registry_hash,
                    "config_hash": lineage,
                    "git_commit": code_commit,
                    "calibration_only": True,
                },
            )
            rows.append(
                {
                    "path": shard_path.relative_to(output).as_posix(),
                    "manifest": replay_manifest_path(shard_path).relative_to(output).as_posix(),
                    "ready": replay_ready_path(shard_path).relative_to(output).as_posix(),
                    "checksum_sha256": manifest["checksum_sha256"],
                    "games": len(games),
                    "positions": len(shard),
                    "actor_runtime": batch.metrics.to_dict(),
                }
            )
            total_positions += len(shard)
            next_game_id += len(games)
            batch_index += 1
        collection = {
            "schema_version": 1,
            "format": "connect4-v3-p6-calibration-pool",
            "created_at": _utc_now(),
            "purpose": "P6 auxiliary-head screening only; excluded from formal model lineage",
            "config": str(args.config.resolve()),
            "config_hash": lineage,
            "git_commit": code_commit,
            "producer_model_id": producer_id,
            "producer_model": producer_path.relative_to(output).as_posix(),
            "producer_sha256": _sha256_file(producer_path),
            "games": args.games,
            "positions": total_positions,
            "minimum_screening_games": 768,
            "minimum_screening_samples": 12000,
            "ready_for_screening": args.games >= 768 and total_positions >= 12000,
            "shards": rows,
        }
        manifest_path = output / "collection_manifest.json"
        _atomic_write_json(manifest_path, collection)
        sys.stdout.write(json.dumps(collection, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 0
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"v3-p6-collector error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
