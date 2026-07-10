"""Shared game rules and runtime defaults for Connect4 3D."""

from .game_rules import (
    BOARD_SHAPE,
    BOARD_SIZE,
    CONNECT_N,
    MAX_LAYERS,
    GameRules,
    action_to_coords,
    coords_to_action,
    infer_board_size_from_action_dim,
)

__all__ = [
    "BOARD_SHAPE", "BOARD_SIZE", "CONNECT_N", "MAX_LAYERS", "GameRules",
    "action_to_coords", "coords_to_action", "infer_board_size_from_action_dim",
]
