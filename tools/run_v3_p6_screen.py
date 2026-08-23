"""Run the frozen five-way P6 auxiliary-head screen on one locked replay pool."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.config import load_config  # noqa: E402
from training.v3.local_validation import (  # noqa: E402
    build_p6_ablation_configs,
    validate_p6_ablation_matrix,
)
from training.v3.model import build_model  # noqa: E402
from training.v3.pipeline import (  # noqa: E402
    _atomic_save_model_artifact,
    _atomic_write_json,
    _build_active_datasets,
    _build_learner,
    _evaluate_validation,
    _git_commit,
    _seed_runtime,
    _sha256_file,
    _utc_now,
)
from training.v3.replay import TrainTokenBucket, concatenate_replay, load_replay_shard  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V3 P6 auxiliary ablation screen")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-positions", type=int, default=40000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[314159, 271828])
    return parser


def _load_pool(replay_dir: Path):
    paths = sorted(replay_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError("P6 replay directory has no NPZ shards")
    shards = []
    manifests = []
    for path in paths:
        shard, manifest = load_replay_shard(path)
        shards.append(shard)
        manifests.append(
            {
                "path": str(path.resolve()),
                "checksum_sha256": manifest["checksum_sha256"],
                "sample_count": int(manifest["sample_count"]),
            }
        )
    return concatenate_replay(shards), manifests


def _occupancy_weights(replay) -> tuple[float, float, float]:
    terminal_canonical = (
        replay.terminal_board
        * replay.player_to_move[:, np.newaxis, np.newaxis, np.newaxis]
    )
    labels = np.where(
        terminal_canonical > 0,
        0,
        np.where(terminal_canonical < 0, 1, 2),
    )
    mask = replay.board == 0
    labels = labels[mask]
    counts = np.bincount(labels.astype(np.int64), minlength=3).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("P6 pool must cover all three future-occupancy classes")
    fractions = counts / counts.sum()
    inverse = 1.0 / (3.0 * fractions)
    clipped = np.clip(inverse, 0.25, 5.0)
    return tuple(float(value) for value in clipped)


def _finite_metrics(payload: dict[str, object]) -> None:
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite P6 metric {key}={value}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.train_positions < 1 or not args.seeds or any(seed < 0 for seed in args.seeds):
            raise ValueError("train-positions and seeds must be positive/non-negative")
        base = load_config(args.config)
        replay, sources = _load_pool(args.replay_dir.resolve())
        games = len(set(int(value) for value in replay.game_id))
        if games < 768 or len(replay) < 12000:
            raise RuntimeError(
                f"P6 pool is below the frozen floor: games={games}/768 samples={len(replay)}/12000"
            )
        class_weights = _occupancy_weights(replay)
        base = replace(
            base,
            learner=replace(base.learner, future_occupancy_class_weights=class_weights),
        )
        matrix = validate_p6_ablation_matrix(base, build_p6_ablation_configs(base))
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite P6 screen: {output}")
        output.mkdir(parents=True)
        results: list[dict[str, object]] = []
        for seed in args.seeds:
            seeded_base = replace(base, run=replace(base.run, seed=int(seed)))
            _seed_runtime(seeded_base)
            initial_model = build_model(seeded_base.model)
            initial_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in initial_model.state_dict().items()
            }
            del initial_model
            for name, variant in build_p6_ablation_configs(seeded_base):
                model = build_model(variant.model)
                model.load_state_dict(initial_state, strict=True)
                learner, optimizer = _build_learner(variant, model)
                dataset, validation, selection = _build_active_datasets(replay, variant)
                bucket = TrainTokenBucket(variant.replay.train_tokens_per_raw_position)
                bucket.add(len(replay))
                calls: list[dict[str, object]] = []
                while bucket.total_positions_consumed < args.train_positions:
                    remaining = args.train_positions - bucket.total_positions_consumed
                    metrics = learner.train_steps(
                        dataset,
                        steps=variant.learner.max_optimizer_steps_per_cycle,
                        token_bucket=bucket,
                        position_limit=remaining,
                    ).to_dict()
                    _finite_metrics(metrics)
                    if int(metrics["steps"]) == 0:
                        raise RuntimeError("P6 token bucket exhausted before the fixed budget")
                    calls.append(metrics)
                validation_metrics = _evaluate_validation(model, validation, variant)
                _finite_metrics(validation_metrics)
                model_path = output / "models" / f"seed{seed}_{name}.pt"
                _atomic_save_model_artifact(
                    model_path,
                    model=model,
                    model_config=asdict(variant.model),
                    metadata={
                        "purpose": "P6 auxiliary ablation; never an accepted champion",
                        "variant": name,
                        "seed": seed,
                        "train_positions": bucket.total_positions_consumed,
                    },
                )
                results.append(
                    {
                        "variant": name,
                        "seed": seed,
                        "train_positions": bucket.total_positions_consumed,
                        "optimizer_steps": learner.global_step,
                        "selection": selection,
                        "last_train_metrics": calls[-1],
                        "validation_metrics": validation_metrics,
                        "model": model_path.relative_to(output).as_posix(),
                        "model_sha256": _sha256_file(model_path),
                    }
                )
                del learner, optimizer, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        report = {
            "schema_version": 1,
            "format": "connect4-v3-p6-auxiliary-screen",
            "created_at": _utc_now(),
            "git_commit": _git_commit(),
            "base_config": str(args.config.resolve()),
            "replay_dir": str(args.replay_dir.resolve()),
            "replay_games": games,
            "replay_samples": len(replay),
            "replay_sources": sources,
            "future_occupancy_class_weights": class_weights,
            "class_weight_rule": "balanced inverse frequency clipped to [0.25,5.0]",
            "train_positions_per_seed_variant": args.train_positions,
            "seeds": args.seeds,
            "ablation_matrix": matrix,
            "automatic_winner_selection": False,
            "results": results,
        }
        report_path = output / "report.json"
        _atomic_write_json(report_path, report)
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 0
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"v3-p6-screen error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
