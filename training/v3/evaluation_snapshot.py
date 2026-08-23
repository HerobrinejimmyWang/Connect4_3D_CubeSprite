from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import load_checkpoint
from .model import build_model


EVALUATION_SNAPSHOT_FORMAT = "connect4-v3-model"
EVALUATION_SNAPSHOT_FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def export_evaluation_snapshot(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Export checkpoint weights as an immutable, evaluation-only V3 artifact.

    Formal checkpoints contain learner and RNG state and are not directly accepted
    by evaluation tools.  This projection intentionally carries no optimizer state
    and does not change the run's accepted-model commit or self-play producer.
    """

    source = Path(checkpoint_path).resolve()
    target = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"V3 checkpoint does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"evaluation snapshot is immutable: {target}")

    checkpoint = load_checkpoint(source, map_location="cpu")
    extra_state = checkpoint.extra_state
    if not isinstance(extra_state, Mapping):
        raise ValueError("V3 checkpoint extra_state must be a mapping")
    model_config = extra_state.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("V3 checkpoint is missing extra_state.model_config")

    # Strictly reconstruct once before writing so malformed or mismatched formal
    # checkpoints cannot become apparently valid evaluation artifacts.
    model = build_model(dict(model_config))
    model.load_state_dict(checkpoint.model_state, strict=True)

    train_positions = extra_state.get("train_positions_consumed")
    resolved_model_id = model_id or (
        f"evaluation-snapshot-g{checkpoint.generation:06d}-"
        f"s{checkpoint.global_step:08d}"
        + (f"-t{int(train_positions):09d}" if train_positions is not None else "")
    )
    if not resolved_model_id.strip():
        raise ValueError("model_id must be non-empty")

    source_sha256 = _sha256(source)
    metadata = {
        "model_id": resolved_model_id,
        "evaluation_only": True,
        "eligible_for_acceptance": False,
        "eligible_for_selfplay": False,
        "source_kind": "formal_v3_checkpoint",
        "source_checkpoint_name": source.name,
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_config_hash": checkpoint.config_hash,
        "source_checkpoint_code_version": checkpoint.code_version,
        "generation": checkpoint.generation,
        "global_step": checkpoint.global_step,
        "train_positions_consumed": train_positions,
        "accepted_model_id_at_checkpoint": checkpoint.accepted_model_id,
    }
    payload = {
        "format": EVALUATION_SNAPSHOT_FORMAT,
        "format_version": EVALUATION_SNAPSHOT_FORMAT_VERSION,
        "model_config": dict(model_config),
        "model_state": {
            name: tensor.detach().cpu() for name, tensor in checkpoint.model_state.items()
        },
        "metadata": metadata,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    return {
        "output_path": str(target),
        "output_sha256": _sha256(target),
        "source_checkpoint_sha256": source_sha256,
        "model_id": resolved_model_id,
        "generation": checkpoint.generation,
        "global_step": checkpoint.global_step,
        "train_positions_consumed": train_positions,
        "evaluation_only": True,
        "eligible_for_acceptance": False,
        "eligible_for_selfplay": False,
    }


__all__ = [
    "EVALUATION_SNAPSHOT_FORMAT",
    "EVALUATION_SNAPSHOT_FORMAT_VERSION",
    "export_evaluation_snapshot",
]
