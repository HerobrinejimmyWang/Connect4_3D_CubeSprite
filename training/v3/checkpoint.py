"""Versioned, atomic V3 checkpoint capture and restoration."""

from __future__ import annotations

import copy
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT = "connect4-v3-checkpoint"
CHECKPOINT_FORMAT_VERSION = 1


def _snapshot_to_cpu(value: Any) -> Any:
    """Detach a nested checkpoint value from live state and CUDA storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _snapshot_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if hasattr(torch.backends, "cudnn"):
        state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu_state = torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).detach().cpu()
    torch.set_rng_state(torch_cpu_state)
    torch.use_deterministic_algorithms(bool(state.get("torch_deterministic_algorithms", False)))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(state.get("cudnn_deterministic", False))
        torch.backends.cudnn.benchmark = bool(state.get("cudnn_benchmark", False))
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8).detach().cpu() for item in cuda_state]
        )


@dataclass
class CheckpointV1:
    global_step: int
    generation: int
    replay_cursor: dict[str, Any]
    sample_ids: list[tuple[int, int]]
    accepted_model_id: str | None
    candidate_model_id: str | None
    config_hash: str
    code_version: str
    recent_evaluation: dict[str, Any] | None
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    extra_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.global_step < 0 or self.generation < 0:
            raise ValueError("checkpoint counters cannot be negative")
        if not self.config_hash:
            raise ValueError("checkpoint config_hash cannot be empty")
        if not self.code_version:
            raise ValueError("checkpoint code_version cannot be empty")
        self.sample_ids = [tuple(map(int, sample_id)) for sample_id in self.sample_ids]
        if any(len(sample_id) != 2 for sample_id in self.sample_ids):
            raise ValueError("sample_ids entries must be (game_id, ply)")

    @classmethod
    def capture(
        cls,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        generation: int,
        replay_cursor: Mapping[str, Any],
        sample_ids: Sequence[tuple[int, int]],
        accepted_model_id: str | None,
        candidate_model_id: str | None,
        config_hash: str,
        code_version: str,
        recent_evaluation: Mapping[str, Any] | None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        extra_state: Mapping[str, Any] | None = None,
    ) -> "CheckpointV1":
        return cls(
            global_step=int(global_step),
            generation=int(generation),
            replay_cursor=_snapshot_to_cpu(dict(replay_cursor)),
            sample_ids=list(sample_ids),
            accepted_model_id=accepted_model_id,
            candidate_model_id=candidate_model_id,
            config_hash=str(config_hash),
            code_version=str(code_version),
            recent_evaluation=(
                _snapshot_to_cpu(dict(recent_evaluation))
                if recent_evaluation is not None
                else None
            ),
            # state_dict() may retain live tensor storage.  A checkpoint object
            # must be an immutable point-in-time snapshot even when the learner
            # continues before a background writer serializes it.
            model_state=_snapshot_to_cpu(model.state_dict()),
            optimizer_state=_snapshot_to_cpu(optimizer.state_dict()),
            scheduler_state=(
                _snapshot_to_cpu(scheduler.state_dict()) if scheduler is not None else None
            ),
            scaler_state=(
                _snapshot_to_cpu(scaler.state_dict()) if scaler is not None else None
            ),
            rng_state=_snapshot_to_cpu(capture_rng_state()),
            extra_state=_snapshot_to_cpu(dict(extra_state or {})),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "global_step": self.global_step,
            "generation": self.generation,
            "replay_cursor": self.replay_cursor,
            "sample_ids": self.sample_ids,
            "accepted_model_id": self.accepted_model_id,
            "candidate_model_id": self.candidate_model_id,
            "config_hash": self.config_hash,
            "code_version": self.code_version,
            "recent_evaluation": self.recent_evaluation,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "rng_state": self.rng_state,
            "extra_state": self.extra_state,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CheckpointV1":
        if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise ValueError("not a V3 checkpoint")
        if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version: {payload.get('checkpoint_format_version')!r}"
            )
        fields = (
            "global_step",
            "generation",
            "replay_cursor",
            "sample_ids",
            "accepted_model_id",
            "candidate_model_id",
            "config_hash",
            "code_version",
            "recent_evaluation",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "rng_state",
            "extra_state",
        )
        missing = [name for name in fields if name not in payload]
        if missing:
            raise ValueError(f"checkpoint is missing fields: {', '.join(missing)}")
        return cls(**{name: payload[name] for name in fields})

    def restore(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        restore_rng: bool = True,
        expected_config_hash: str | None = None,
        strict_model: bool = True,
    ) -> None:
        if expected_config_hash is not None and self.config_hash != expected_config_hash:
            raise ValueError(
                "checkpoint config hash mismatch: "
                f"checkpoint={self.config_hash}, configured={expected_config_hash}"
            )
        if (scheduler is None) != (self.scheduler_state is None):
            raise ValueError("scheduler presence differs from the checkpoint")
        if scaler is None and self.scaler_state is not None:
            raise ValueError("checkpoint contains scaler state but no scaler was provided")
        model.load_state_dict(self.model_state, strict=strict_model)
        optimizer.load_state_dict(self.optimizer_state)
        if scheduler is not None:
            scheduler.load_state_dict(self.scheduler_state)
        if scaler is not None and self.scaler_state is not None:
            scaler.load_state_dict(self.scaler_state)
        if restore_rng:
            restore_rng_state(self.rng_state)


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


def save_checkpoint(path: str | Path, checkpoint: CheckpointV1) -> Path:
    """Atomically replace ``path`` while preserving an existing good file on failure."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(target)
    try:
        with temporary.open("wb") as handle:
            torch.save(checkpoint.to_payload(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        return target
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> CheckpointV1:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return CheckpointV1.from_payload(payload)


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointV1",
    "capture_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]
