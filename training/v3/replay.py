"""Immutable replay shards and deterministic sampling utilities for V3.

The raw replay format deliberately contains only compact, model-independent
training facts.  A JSON sidecar records provenance and the SHA-256 digest of
the NPZ payload.  Existing shard paths are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPLAY_SCHEMA_VERSION = 1
BOARD_SHAPE = (6, 5, 5)
COLUMN_COUNT = 25
REQUIRED_MANIFEST_METADATA = frozenset(
    {
        "run_id",
        "generation",
        "producer_model_id",
        "seed_range",
        "results",
        "search_config",
        "config_hash",
        "git_commit",
    }
)

WDL_WIN = 0
WDL_DRAW = 1
WDL_LOSS = 2

SEARCH_FAST = 0
SEARCH_FULL = 1

ARRAY_SCHEMA: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "board": (np.dtype(np.int8), BOARD_SHAPE),
    "visit_counts": (np.dtype(np.uint32), (COLUMN_COUNT,)),
    "wdl": (np.dtype(np.uint8), ()),
    "game_id": (np.dtype(np.uint64), ()),
    "ply": (np.dtype(np.uint16), ()),
    "player": (np.dtype(np.int8), ()),
    "search_kind": (np.dtype(np.uint8), ()),
}


def _field(sample: object, name: str) -> Any:
    if isinstance(sample, Mapping):
        return sample[name]
    return getattr(sample, name)


def _encode_search_kind(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "fast":
            return SEARCH_FAST
        if normalized == "full":
            return SEARCH_FULL
        raise ValueError(f"unknown search_kind: {value!r}")
    encoded = int(value)
    if encoded not in (SEARCH_FAST, SEARCH_FULL):
        raise ValueError(f"search_kind must be 0/1 or fast/full, got {value!r}")
    return encoded


@dataclass(frozen=True)
class ReplayShard:
    """In-memory representation of one or more ordered replay shards."""

    board: np.ndarray
    visit_counts: np.ndarray
    wdl: np.ndarray
    game_id: np.ndarray
    ply: np.ndarray
    player: np.ndarray
    search_kind: np.ndarray

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.board.shape[0])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ARRAY_SCHEMA}

    def validate(self) -> None:
        lengths: set[int] = set()
        for name, (expected_dtype, trailing_shape) in ARRAY_SCHEMA.items():
            array = getattr(self, name)
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray")
            if array.dtype != expected_dtype:
                raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
            if array.ndim != 1 + len(trailing_shape) or tuple(array.shape[1:]) != trailing_shape:
                raise ValueError(
                    f"{name} must have shape [N{''.join(',' + str(v) for v in trailing_shape)}], "
                    f"got {tuple(array.shape)}"
                )
            lengths.add(int(array.shape[0]))
        if len(lengths) != 1:
            raise ValueError("all replay arrays must have the same leading dimension")
        if not np.all(np.isin(self.board, (-1, 0, 1))):
            raise ValueError("board values must be canonical -1/0/1")
        if not np.all(np.isin(self.wdl, (WDL_WIN, WDL_DRAW, WDL_LOSS))):
            raise ValueError("wdl values must be 0=win, 1=draw, or 2=loss")
        if not np.all(np.isin(self.player, (-1, 1))):
            raise ValueError("player values must be -1 or 1")
        if not np.all(np.isin(self.search_kind, (SEARCH_FAST, SEARCH_FULL))):
            raise ValueError("search_kind values must be 0=fast or 1=full")
        if len(self) and np.any(self.visit_counts.sum(axis=1, dtype=np.uint64) == 0):
            raise ValueError("every replay sample must contain at least one visit")

    @classmethod
    def empty(cls) -> "ReplayShard":
        return cls(
            board=np.empty((0, *BOARD_SHAPE), dtype=np.int8),
            visit_counts=np.empty((0, COLUMN_COUNT), dtype=np.uint32),
            wdl=np.empty((0,), dtype=np.uint8),
            game_id=np.empty((0,), dtype=np.uint64),
            ply=np.empty((0,), dtype=np.uint16),
            player=np.empty((0,), dtype=np.int8),
            search_kind=np.empty((0,), dtype=np.uint8),
        )

    @classmethod
    def from_mapping(cls, arrays: Mapping[str, Any]) -> "ReplayShard":
        missing = set(ARRAY_SCHEMA).difference(arrays)
        extra = set(arrays).difference(ARRAY_SCHEMA)
        if missing or extra:
            raise ValueError(f"replay fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        return cls(**{name: np.asarray(arrays[name]) for name in ARRAY_SCHEMA})

    @classmethod
    def from_samples(cls, samples: Iterable[object], *, full_only: bool = False) -> "ReplayShard":
        """Pack dataclass-like or mapping samples into the canonical arrays."""

        rows: list[object] = []
        encoded_kinds: list[int] = []
        for sample in samples:
            kind = _encode_search_kind(_field(sample, "search_kind"))
            if full_only and kind != SEARCH_FULL:
                continue
            rows.append(sample)
            encoded_kinds.append(kind)
        if not rows:
            return cls.empty()

        def checked_integer(name: str, minimum: int, maximum: int) -> list[int]:
            values = [int(_field(row, name)) for row in rows]
            if any(value < minimum or value > maximum for value in values):
                raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
            return values

        board = np.stack([np.asarray(_field(row, "board"), dtype=np.int8) for row in rows])
        visits_raw = [np.asarray(_field(row, "visit_counts")) for row in rows]
        if any(array.shape != (COLUMN_COUNT,) for array in visits_raw):
            raise ValueError("visit_counts must have shape [25]")
        if any(np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0) for array in visits_raw):
            raise ValueError("visit_counts cannot be negative")
        if any(np.any(array > np.iinfo(np.uint32).max) for array in visits_raw):
            raise ValueError("visit_counts exceed uint32")
        visit_counts = np.stack(visits_raw).astype(np.uint32, copy=False)
        return cls(
            board=board,
            visit_counts=visit_counts,
            wdl=np.asarray(checked_integer("wdl", 0, 2), dtype=np.uint8),
            game_id=np.asarray(checked_integer("game_id", 0, np.iinfo(np.uint64).max), dtype=np.uint64),
            ply=np.asarray(checked_integer("ply", 0, np.iinfo(np.uint16).max), dtype=np.uint16),
            player=np.asarray(checked_integer("player", -1, 1), dtype=np.int8),
            search_kind=np.asarray(encoded_kinds, dtype=np.uint8),
        )

    def take(self, indices: Sequence[int] | np.ndarray) -> "ReplayShard":
        index_array = np.asarray(indices)
        if index_array.dtype != np.bool_:
            index_array = index_array.astype(np.int64, copy=False)
        return ReplayShard(**{name: array[index_array] for name, array in self.as_dict().items()})

    def tail(self, count: int) -> "ReplayShard":
        if count < 0:
            raise ValueError("count cannot be negative")
        start = max(0, len(self) - int(count))
        return self.take(np.arange(start, len(self), dtype=np.int64))


def concatenate_replay(shards: Sequence[ReplayShard]) -> ReplayShard:
    if not shards:
        return ReplayShard.empty()
    return ReplayShard(
        **{
            name: np.concatenate([shard.as_dict()[name] for shard in shards], axis=0)
            for name in ARRAY_SCHEMA
        }
    )


def replay_manifest_path(shard_path: str | Path) -> Path:
    path = Path(shard_path)
    return path.with_name(f"{path.stem}.manifest.json")


def replay_ready_path(shard_path: str | Path) -> Path:
    path = Path(shard_path)
    return path.with_name(f"{path.stem}.ready.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_manifest_metadata(metadata: Mapping[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"replay manifest metadata is missing required fields: {sorted(missing)}")
    if not isinstance(metadata["run_id"], str) or not metadata["run_id"]:
        raise ValueError("manifest run_id must be a non-empty string")
    if not isinstance(metadata["generation"], int) or metadata["generation"] < 0:
        raise ValueError("manifest generation must be a non-negative integer")
    if not isinstance(metadata["producer_model_id"], str) or not metadata["producer_model_id"]:
        raise ValueError("manifest producer_model_id must be a non-empty string")
    seed_range = metadata["seed_range"]
    if not isinstance(seed_range, Mapping) or set(seed_range) != {"start", "end"}:
        raise ValueError("manifest seed_range must contain exactly start and end")
    if any(not isinstance(seed_range[key], int) or seed_range[key] < 0 for key in ("start", "end")):
        raise ValueError("manifest seed range values must be non-negative integers")
    if seed_range["end"] < seed_range["start"]:
        raise ValueError("manifest seed_range end cannot precede start")
    if not isinstance(metadata["results"], Mapping):
        raise ValueError("manifest results must be a mapping")
    if not isinstance(metadata["search_config"], Mapping):
        raise ValueError("manifest search_config must be a mapping")
    for key in ("config_hash", "git_commit"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"manifest {key} must be a non-empty string")


def write_replay_shard(
    shard_path: str | Path,
    shard: ReplayShard,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically create an immutable NPZ shard and its JSON sidecar."""

    shard.validate()
    target = Path(shard_path)
    if target.suffix.lower() != ".npz":
        raise ValueError("replay shard path must end in .npz")
    manifest_target = replay_manifest_path(target)
    ready_target = replay_ready_path(target)
    if target.exists() or manifest_target.exists() or ready_target.exists():
        raise FileExistsError(f"replay shard is append-only and already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    npz_temp = _temporary_path(target)
    manifest_temp = _temporary_path(manifest_target)
    ready_temp = _temporary_path(ready_target)
    reserved = {
        "schema_version",
        "shard_file",
        "sample_count",
        "arrays",
        "checksum_sha256",
        "compressed_bytes",
        "bytes_per_position",
    }
    overlap = reserved.intersection(metadata)
    if overlap:
        raise ValueError(f"metadata cannot override reserved fields: {sorted(overlap)}")
    _validate_manifest_metadata(metadata)
    try:
        with npz_temp.open("wb") as handle:
            np.savez_compressed(handle, **shard.as_dict())
            handle.flush()
            os.fsync(handle.fileno())
        checksum = sha256_file(npz_temp)
        compressed_bytes = npz_temp.stat().st_size
        manifest: dict[str, Any] = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "shard_file": target.name,
            "sample_count": len(shard),
            "arrays": {
                name: {"dtype": dtype.name, "shape": ["N", *trailing]}
                for name, (dtype, trailing) in ARRAY_SCHEMA.items()
            },
            "checksum_sha256": checksum,
            "compressed_bytes": compressed_bytes,
            "bytes_per_position": compressed_bytes / max(len(shard), 1),
            **dict(metadata),
        }
        encoded = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        with manifest_temp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        ready = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "shard_file": target.name,
            "manifest_file": manifest_target.name,
            "manifest_sha256": sha256_file(manifest_temp),
        }
        with ready_temp.open("wb") as handle:
            handle.write(
                (json.dumps(ready, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(npz_temp, target)
        os.replace(manifest_temp, manifest_target)
        os.replace(ready_temp, ready_target)
        _fsync_directory(target.parent)
        return manifest
    finally:
        for temporary in (npz_temp, manifest_temp, ready_temp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load_replay_shard(
    shard_path: str | Path,
    *,
    verify_checksum: bool = True,
) -> tuple[ReplayShard, dict[str, Any]]:
    target = Path(shard_path)
    manifest = validate_replay_shard_artifacts(target, verify_checksum=verify_checksum)
    with np.load(target, allow_pickle=False) as payload:
        shard = ReplayShard.from_mapping({name: payload[name] for name in payload.files})
    if len(shard) != int(manifest.get("sample_count", -1)):
        raise ValueError("manifest sample_count does not match replay shard")
    return shard, manifest


def validate_replay_shard_artifacts(
    shard_path: str | Path,
    *,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    """Validate a committed shard triplet without materializing its arrays."""

    target = Path(shard_path)
    manifest_target = replay_manifest_path(target)
    ready_target = replay_ready_path(target)
    if not target.is_file() or not manifest_target.is_file() or not ready_target.is_file():
        raise ValueError("replay shard is missing its NPZ, manifest, or ready marker")
    ready = json.loads(ready_target.read_text(encoding="utf-8"))
    expected_ready = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "shard_file": target.name,
        "manifest_file": manifest_target.name,
        "manifest_sha256": sha256_file(manifest_target),
    }
    if ready != expected_ready:
        raise ValueError("replay shard ready marker does not match its manifest")
    manifest = json.loads(manifest_target.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError(f"unsupported replay schema: {manifest.get('schema_version')!r}")
    if manifest.get("shard_file") != target.name:
        raise ValueError("manifest shard_file does not match the NPZ filename")
    if verify_checksum:
        actual = sha256_file(target)
        if actual != manifest.get("checksum_sha256"):
            raise ValueError("replay shard checksum mismatch")
    if int(manifest.get("compressed_bytes", -1)) != target.stat().st_size:
        raise ValueError("manifest compressed_bytes does not match replay shard")
    if int(manifest.get("sample_count", -1)) < 0:
        raise ValueError("manifest sample_count must be non-negative")
    return manifest


def _stable_u64(namespace: bytes, *values: int) -> int:
    digest = hashlib.blake2b(digest_size=8, person=namespace[:16])
    for value in values:
        digest.update(struct.pack(">Q", int(value) & ((1 << 64) - 1)))
    return int.from_bytes(digest.digest(), "big")


def stable_game_split(
    game_id: int,
    *,
    validation_fraction: float = 0.05,
    split_seed: int = 0,
) -> str:
    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be in [0, 1]")
    threshold = int(validation_fraction * (1 << 64))
    bucket = _stable_u64(b"v3-game-split", split_seed, game_id)
    return "validation" if bucket < threshold else "train"


def stable_split_mask(
    game_ids: np.ndarray,
    *,
    split: str,
    validation_fraction: float = 0.05,
    split_seed: int = 0,
) -> np.ndarray:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    unique_ids = np.unique(np.asarray(game_ids, dtype=np.uint64))
    membership = {
        int(game_id): stable_game_split(
            int(game_id), validation_fraction=validation_fraction, split_seed=split_seed
        )
        == split
        for game_id in unique_ids
    }
    return np.asarray([membership[int(game_id)] for game_id in game_ids], dtype=np.bool_)


def growing_window_size(
    total_positions: int,
    *,
    c: int,
    alpha: float,
    beta: float,
) -> int:
    """Return KataGo's power-law recent-data window size, rounded down."""

    total = int(total_positions)
    if total < 0:
        raise ValueError("total_positions cannot be negative")
    if total == 0:
        return 0
    if c <= 0:
        raise ValueError("c must be positive")
    if alpha == 0.0:
        raise ValueError("alpha cannot be zero")
    if beta < 0.0:
        raise ValueError("beta cannot be negative")
    ratio = total / float(c)
    raw_window = c * (1.0 + beta * (math.pow(ratio, alpha) - 1.0) / alpha)
    return min(total, max(1, int(math.floor(raw_window))))


def select_active_replay(
    replay: ReplayShard,
    *,
    c: int,
    alpha: float,
    beta: float,
    split: str | None = None,
    validation_fraction: float = 0.05,
    split_seed: int = 0,
) -> ReplayShard:
    indices = active_replay_indices(
        replay,
        c=c,
        alpha=alpha,
        beta=beta,
        split=split,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    return replay.take(indices)


def active_replay_indices(
    replay: ReplayShard,
    *,
    c: int,
    alpha: float,
    beta: float,
    split: str | None = None,
    validation_fraction: float = 0.05,
    split_seed: int = 0,
) -> np.ndarray:
    window_size = growing_window_size(len(replay), c=c, alpha=alpha, beta=beta)
    indices = np.arange(len(replay) - window_size, len(replay), dtype=np.int64)
    if split is None:
        return indices
    mask = stable_split_mask(
        replay.game_id[indices],
        split=split,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    return indices[mask]


@dataclass
class TrainTokenBucket:
    tokens_per_position: float = 4.0
    available_tokens: float = 0.0
    total_positions_added: int = 0
    total_positions_consumed: int = 0

    def __post_init__(self) -> None:
        if self.tokens_per_position <= 0:
            raise ValueError("tokens_per_position must be positive")
        if self.available_tokens < 0:
            raise ValueError("available_tokens cannot be negative")

    def add(self, new_positions: int) -> float:
        count = int(new_positions)
        if count < 0:
            raise ValueError("new_positions cannot be negative")
        self.total_positions_added += count
        self.available_tokens += count * self.tokens_per_position
        return self.available_tokens

    def consumable(self, requested: int) -> int:
        if requested < 0:
            raise ValueError("requested cannot be negative")
        return min(int(requested), int(math.floor(self.available_tokens + 1e-12)))

    def consume(self, requested: int) -> int:
        consumed = self.consumable(requested)
        self.available_tokens -= consumed
        self.total_positions_consumed += consumed
        return consumed

    @property
    def train_data_ratio(self) -> float:
        if self.total_positions_added == 0:
            return 0.0
        return self.total_positions_consumed / float(self.total_positions_added)

    def state_dict(self) -> dict[str, int | float]:
        return {
            "tokens_per_position": self.tokens_per_position,
            "available_tokens": self.available_tokens,
            "total_positions_added": self.total_positions_added,
            "total_positions_consumed": self.total_positions_consumed,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TrainTokenBucket":
        return cls(
            tokens_per_position=float(state["tokens_per_position"]),
            available_tokens=float(state["available_tokens"]),
            total_positions_added=int(state["total_positions_added"]),
            total_positions_consumed=int(state["total_positions_consumed"]),
        )


def apply_d4(array: np.ndarray, transform: int) -> np.ndarray:
    """Apply one of eight square symmetries to the final two dimensions."""

    index = int(transform)
    if not 0 <= index < 8:
        raise ValueError("D4 transform must be in [0, 7]")
    source = np.asarray(array)
    if source.ndim < 2 or source.shape[-2:] != (5, 5):
        raise ValueError("D4 input must end in [5, 5]")
    result = np.rot90(source, k=index % 4, axes=(-2, -1))
    if index >= 4:
        result = np.flip(result, axis=-1)
    return np.ascontiguousarray(result)


def inverse_d4_index(transform: int) -> int:
    index = int(transform)
    if not 0 <= index < 8:
        raise ValueError("D4 transform must be in [0, 7]")
    return (-index) % 4 if index < 4 else index


def stable_d4_index(
    *,
    augmentation_seed: int,
    game_id: int,
    ply: int,
    augmentation_token: int,
) -> int:
    return _stable_u64(
        b"v3-d4-augment",
        augmentation_seed,
        game_id,
        ply,
        augmentation_token,
    ) % 8


__all__ = [
    "ARRAY_SCHEMA",
    "BOARD_SHAPE",
    "COLUMN_COUNT",
    "REPLAY_SCHEMA_VERSION",
    "REQUIRED_MANIFEST_METADATA",
    "ReplayShard",
    "SEARCH_FAST",
    "SEARCH_FULL",
    "TrainTokenBucket",
    "WDL_DRAW",
    "WDL_LOSS",
    "WDL_WIN",
    "apply_d4",
    "active_replay_indices",
    "concatenate_replay",
    "growing_window_size",
    "inverse_d4_index",
    "load_replay_shard",
    "replay_manifest_path",
    "replay_ready_path",
    "select_active_replay",
    "sha256_file",
    "stable_d4_index",
    "stable_game_split",
    "stable_split_mask",
    "validate_replay_shard_artifacts",
    "write_replay_shard",
]
