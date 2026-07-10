from __future__ import annotations

from pathlib import Path

from connect4_core import BOARD_SHAPE
from arena.agent import (
    MCTSAgent,
    TinyPolicyAgent,
    _extract_state_dict_and_metadata,
    _load_checkpoint_payload,
    infer_model_config,
)


def validate_v22_checkpoint(model_path, game, requested_config=None):
    model_path = Path(model_path)
    expected_shape = tuple(game.get_board_size())
    if expected_shape != BOARD_SHAPE:
        raise ValueError(f"The v2.2 runtime requires board shape {BOARD_SHAPE}, got {expected_shape}.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    payload = _load_checkpoint_payload(model_path)
    if _is_tiny_payload(payload):
        config = dict(payload.get("model_config") or {})
        layers = int(config.get("board_layers", BOARD_SHAPE[0]))
        size = int(config.get("board_size", BOARD_SHAPE[1]))
        candidates = int(config.get("candidate_count", BOARD_SHAPE[1] * BOARD_SHAPE[2]))
        if (layers, size, size) != BOARD_SHAPE or candidates != 25:
            raise ValueError(
                f"Tiny checkpoint is incompatible: board={(layers, size, size)}, candidates={candidates}; "
                f"expected board={BOARD_SHAPE}, candidates=25."
            )
        return {**config, "architecture": config.get("architecture", "tiny-candidate-policy-v1"), "action_dim": 150}

    state_dict, metadata = _extract_state_dict_and_metadata(payload)
    embedded = metadata.get("student_model_config") or metadata.get("model_config") or {}
    merged = dict(embedded) if isinstance(embedded, dict) else {}
    merged.update({key: value for key, value in (requested_config or {}).items() if value is not None})
    config = infer_model_config(state_dict, requested_config=merged, game=game)
    shape = (int(config["board_layers"]), int(config["board_size"]), int(config["board_size"]))
    action_dim = int(config.get("action_dim", 0))
    if shape != BOARD_SHAPE or action_dim != 150:
        raise ValueError(
            f"Checkpoint is incompatible: board={shape}, action_dim={action_dim}; "
            f"expected board={BOARD_SHAPE}, action_dim=150."
        )
    return config


def load_v22_agent(
    game,
    model_path,
    name=None,
    device=None,
    model_config=None,
    num_mcts_sims=64,
    cpuct=1.0,
    num_mcts_threads=2,
    virtual_loss=1.0,
    inference_batch_size=16,
    inference_timeout_s=0.003,
):
    """Validate and construct a standard, gravity, or tiny v2.2 agent."""

    config = validate_v22_checkpoint(model_path, game, requested_config=model_config)
    if str(config.get("architecture", "")).startswith("tiny-"):
        return TinyPolicyAgent(game=game, model_path=model_path, name=name, device=device)
    return MCTSAgent(
        game=game,
        model_path=model_path,
        name=name,
        device=device,
        model_config=config,
        num_mcts_sims=num_mcts_sims,
        cpuct=cpuct,
        num_mcts_threads=num_mcts_threads,
        virtual_loss=virtual_loss,
        inference_batch_size=inference_batch_size,
        inference_timeout_s=inference_timeout_s,
    )


def _is_tiny_payload(payload):
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        return False
    config = payload.get("model_config") or {}
    return isinstance(config, dict) and "global_dim" in config and "candidate_dim" in config
