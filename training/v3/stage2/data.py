"""B10 trajectory auditing and immutable four-regime Replay V2 pools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..replay import (
    ReplayShard,
    concatenate_replay,
    load_replay_shard,
    sha256_file,
    stable_split_mask,
    write_replay_shard,
)


METRIC_FIELDS = (
    "anchored_strength",
    "mean_game_length",
    "short_game_rate",
    "policy_entropy",
    "accepted_cadence",
)
STANDARD_REGIMES = ("standard_early", "standard_mid", "standard_late")
REGIMES = (*STANDARD_REGIMES, "mixed_late")


def _read_metrics(path: str | Path, *, minimum_rows: int = 9) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = {"generation", *METRIC_FIELDS}.difference(raw)
        if missing:
            raise ValueError(f"metrics line {line_number} is missing {sorted(missing)}")
        row = {"generation": int(raw["generation"])}
        for field in METRIC_FIELDS:
            value = float(raw[field])
            if not np.isfinite(value):
                raise ValueError(f"metrics line {line_number} has non-finite {field}")
            row[field] = value
        rows.append(row)
    if len(rows) < minimum_rows:
        raise ValueError(f"trajectory metrics need at least {minimum_rows} generations")
    rows.sort(key=lambda item: item["generation"])
    generations = [row["generation"] for row in rows]
    if len(set(generations)) != len(generations):
        raise ValueError("trajectory metrics contain duplicate generations")
    return rows


def _segment_rows(rows: Sequence[Mapping[str, Any]], minimum_fraction: float) -> tuple[int, int]:
    """Find the minimum-SSE contiguous three-segment partition."""

    if not 0.0 < minimum_fraction < 1.0 / 3.0:
        raise ValueError("minimum_fraction must be in (0, 1/3)")
    values = np.asarray([[row[field] for field in METRIC_FIELDS] for row in rows], dtype=np.float64)
    median = np.median(values, axis=0)
    scale = np.median(np.abs(values - median), axis=0)
    scale = np.where(scale > 1e-12, scale, np.std(values, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    values = (values - median) / scale
    count = len(rows)
    minimum = max(2, int(np.ceil(count * minimum_fraction)))
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    prefix_sq = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values * values, axis=0)))

    def cost(start: int, end: int) -> float:
        length = end - start
        total = prefix[end] - prefix[start]
        squares = prefix_sq[end] - prefix_sq[start]
        return float(np.sum(squares - total * total / length))

    best: tuple[float, int, int] | None = None
    for first in range(minimum, count - 2 * minimum + 1):
        for second in range(first + minimum, count - minimum + 1):
            candidate = (cost(0, first) + cost(first, second) + cost(second, count), first, second)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("trajectory is too short for the requested minimum segment size")
    return best[1], best[2]


def _audit_source(
    source_dir: str | Path,
    *,
    lineage_prefix: str,
    data_recipe_id: str,
) -> tuple[Path, list[dict[str, Any]]]:
    source = Path(source_dir).resolve()
    shard_paths = sorted(source.rglob("*.npz"))
    if not shard_paths:
        raise ValueError(f"no Replay V2 shards found under {source}")
    manifests: list[dict[str, Any]] = []
    for path in shard_paths:
        _shard, manifest = load_replay_shard(path, verify_checksum=True)
        run_id = str(manifest["run_id"])
        if lineage_prefix and not run_id.startswith(lineage_prefix):
            raise ValueError(
                f"replay run_id is outside recipe {data_recipe_id!r}: {run_id}"
            )
        manifests.append(
            {
                **manifest,
                "path": str(path),
                "file_sha256": sha256_file(path),
                "data_recipe_id": data_recipe_id,
            }
        )
    return source, manifests


def audit_trajectory(
    source_dir: str | Path,
    metrics_path: str | Path,
    *,
    mixed_source_dir: str | Path,
    mixed_metrics_path: str | Path,
    standard_lineage_prefix: str = "stage1_b10c256_relative_role_guard_coldstart",
    mixed_lineage_prefix: str = "stage1_b10c256_g487_mixed_opening_temp_position_balanced",
    minimum_fraction: float = 0.15,
) -> dict[str, Any]:
    source, standard_manifests = _audit_source(
        source_dir,
        lineage_prefix=standard_lineage_prefix,
        data_recipe_id="b10_standard_v1",
    )
    mixed_source, mixed_manifests = _audit_source(
        mixed_source_dir,
        lineage_prefix=mixed_lineage_prefix,
        data_recipe_id="b10_mixed_opening_position_balanced_v1",
    )
    manifests = [*standard_manifests, *mixed_manifests]
    rule_hashes = {str(row["rule_registry_hash"]) for row in manifests}
    if len(rule_hashes) != 1:
        raise ValueError("B10 source shards use different rule registries")

    rows = _read_metrics(metrics_path)
    first, second = _segment_rows(rows, minimum_fraction)
    slices = (rows[:first], rows[first:second], rows[second:])
    segments = {}
    for name, subset in zip(STANDARD_REGIMES, slices, strict=True):
        start, end = subset[0]["generation"], subset[-1]["generation"]
        positions = sum(
            int(manifest["sample_count"])
            for manifest in standard_manifests
            if start <= int(manifest["generation"]) <= end
        )
        segments[name] = {
            "data_recipe_id": "b10_standard_v1",
            "generation_start": start,
            "generation_end": end,
            "metric_rows": len(subset),
            "available_positions": positions,
            "metric_mean": {
                field: float(np.mean([row[field] for row in subset])) for field in METRIC_FIELDS
            },
        }
    mixed_rows = _read_metrics(mixed_metrics_path, minimum_rows=1)
    mixed_start, mixed_end = mixed_rows[0]["generation"], mixed_rows[-1]["generation"]
    segments["mixed_late"] = {
        "data_recipe_id": "b10_mixed_opening_position_balanced_v1",
        "generation_start": mixed_start,
        "generation_end": mixed_end,
        "metric_rows": len(mixed_rows),
        "available_positions": sum(
            int(manifest["sample_count"])
            for manifest in mixed_manifests
            if mixed_start <= int(manifest["generation"]) <= mixed_end
        ),
        "metric_mean": {
            field: float(np.mean([row[field] for row in mixed_rows])) for field in METRIC_FIELDS
        },
        "role": "promotion_only_engineering_pool",
    }
    return {
        "schema": "connect4-v3-stage2-trajectory-audit-v2",
        "source_dirs": {
            "b10_standard_v1": str(source),
            "b10_mixed_opening_position_balanced_v1": str(mixed_source),
        },
        "lineage_prefixes": {
            "b10_standard_v1": standard_lineage_prefix,
            "b10_mixed_opening_position_balanced_v1": mixed_lineage_prefix,
        },
        "rule_registry_hash": next(iter(rule_hashes)),
        "source_run_ids": {
            recipe: sorted({str(row["run_id"]) for row in manifests if row["data_recipe_id"] == recipe})
            for recipe in ("b10_standard_v1", "b10_mixed_opening_position_balanced_v1")
        },
        "source_config_hashes": {
            recipe: sorted({str(row["config_hash"]) for row in manifests if row["data_recipe_id"] == recipe})
            for recipe in ("b10_standard_v1", "b10_mixed_opening_position_balanced_v1")
        },
        "source_shards": manifests,
        "metric_fields": list(METRIC_FIELDS),
        "segments": segments,
    }


def _stable_sample_order(shard: ReplayShard, indices: np.ndarray, seed: int) -> np.ndarray:
    keys = []
    for index in indices:
        payload = f"{seed}:{int(shard.game_id[index])}:{int(shard.turn_index[index])}".encode()
        keys.append(hashlib.blake2b(payload, digest_size=16, person=b"v3-stage2-pool").digest())
    order = np.argsort(np.asarray(keys, dtype="S16"), kind="stable")
    return indices[order]


def _digest_sample_ids(shard: ReplayShard) -> str:
    digest = hashlib.sha256()
    for game_id, turn in zip(shard.game_id, shard.turn_index, strict=True):
        digest.update(int(game_id).to_bytes(8, "big"))
        digest.update(int(turn).to_bytes(2, "big"))
    return digest.hexdigest()


def freeze_regime_datasets(
    audit: Mapping[str, Any],
    output_dir: str | Path,
    *,
    train_positions: int = 1_000_000,
    validation_positions: int = 50_000,
    seed: int = 271828,
    validation_fraction: float = 0.05,
) -> dict[str, Any]:
    if train_positions < 1 or validation_positions < 1:
        raise ValueError("train and validation positions must be positive")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Stage 2 dataset directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "connect4-v3-stage2-frozen-pools-v2",
        "seed": seed,
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "regimes": {},
    }
    source_rows = list(audit["source_shards"])
    for regime in REGIMES:
        segment = audit["segments"][regime]
        paths = [
            Path(row["path"])
            for row in source_rows
            if row["data_recipe_id"] == segment["data_recipe_id"]
            and int(segment["generation_start"]) <= int(row["generation"]) <= int(segment["generation_end"])
        ]
        if not paths:
            raise ValueError(f"regime {regime} contains no replay shards")
        loaded = [load_replay_shard(path, verify_checksum=True)[0] for path in paths]
        combined = concatenate_replay(loaded)
        train_indices = np.flatnonzero(
            stable_split_mask(
                combined.game_id,
                split="train",
                validation_fraction=validation_fraction,
                split_seed=seed,
            )
        )
        validation_indices = np.flatnonzero(
            stable_split_mask(
                combined.game_id,
                split="validation",
                validation_fraction=validation_fraction,
                split_seed=seed,
            )
        )
        train_indices = _stable_sample_order(combined, train_indices, seed)[:train_positions]
        validation_indices = _stable_sample_order(combined, validation_indices, seed)[:validation_positions]
        if len(train_indices) != train_positions or len(validation_indices) != validation_positions:
            raise ValueError(
                f"regime {regime} has insufficient split positions: "
                f"train={len(train_indices)}, validation={len(validation_indices)}"
            )
        if set(map(int, combined.game_id[train_indices])).intersection(
            map(int, combined.game_id[validation_indices])
        ):
            raise AssertionError("game-level Stage 2 split leaked between train and validation")

        regime_result: dict[str, Any] = {
            "data_recipe_id": segment["data_recipe_id"],
            "role": segment.get("role", "controlled_architecture_screen"),
            "source_shards": [str(path) for path in paths],
        }
        source_digest = hashlib.sha256(
            "\n".join(str(row["file_sha256"]) for row in source_rows if Path(row["path"]) in paths).encode()
        ).hexdigest()
        for split, indices in (("train", train_indices), ("validation", validation_indices)):
            frozen = combined.take(indices)
            target = output / regime / f"{split}.npz"
            metadata = {
                "run_id": f"stage2-b10-{regime}-{split}",
                "generation": int(segment["generation_end"]),
                "producer_model_id": f"stage2-frozen-{segment['data_recipe_id']}",
                "seed_range": {"start": seed, "end": seed},
                "results": {
                    "regime": regime,
                    "split": split,
                    "data_recipe_id": segment["data_recipe_id"],
                },
                "search_config": {"source": "immutable-replay-v2"},
                "rule_registry_hash": str(audit["rule_registry_hash"]),
                "config_hash": source_digest,
                "git_commit": "derived-stage2-pool",
                "stage2": {
                    "source_digest": source_digest,
                    "sample_id_digest": _digest_sample_ids(frozen),
                    "generation_start": int(segment["generation_start"]),
                    "generation_end": int(segment["generation_end"]),
                    "data_recipe_id": segment["data_recipe_id"],
                    "role": segment.get("role", "controlled_architecture_screen"),
                },
            }
            manifest = write_replay_shard(target, frozen, metadata)
            regime_result[split] = {
                "path": str(target),
                "positions": len(frozen),
                "sha256": manifest["checksum_sha256"],
                "sample_id_digest": metadata["stage2"]["sample_id_digest"],
            }
        result["regimes"][regime] = regime_result
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output / "stage2_pools.json").write_text(encoded, encoding="utf-8")
    result["manifest_sha256"] = sha256_file(output / "stage2_pools.json")
    return result
