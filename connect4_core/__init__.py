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
from .rules import (
    CLASSIC_RULE,
    DEFAULT_RULE_REGISTRY,
    FEATURE_DIM,
    P1_LAYER0_IGNORED_MODIFIER,
    P1_LAYER0_IGNORED_RULE,
    P1_VERTICAL_FORBIDDEN_MODIFIER,
    P1_VERTICAL_FORBIDDEN_RULE,
    P1_VERTICAL_IGNORED_MODIFIER,
    P1_VERTICAL_IGNORED_RULE,
    RULE1,
    RULE2,
    RULE3,
    GameOutcome,
    GameState,
    RuleEngine,
    RuleFeatureSchema,
    RuleModifier,
    RuleRegistry,
    RuleSpec,
    TurnAction,
    TurnKind,
    compose_rule,
)

__all__ = [
    "BOARD_SHAPE", "BOARD_SIZE", "CONNECT_N", "MAX_LAYERS", "GameRules",
    "action_to_coords", "coords_to_action", "infer_board_size_from_action_dim",
    "CLASSIC_RULE", "DEFAULT_RULE_REGISTRY", "FEATURE_DIM", "P1_LAYER0_IGNORED_MODIFIER",
    "P1_LAYER0_IGNORED_RULE", "P1_VERTICAL_FORBIDDEN_MODIFIER", "P1_VERTICAL_FORBIDDEN_RULE",
    "P1_VERTICAL_IGNORED_MODIFIER", "P1_VERTICAL_IGNORED_RULE", "RULE1", "RULE2", "RULE3",
    "GameOutcome", "GameState", "RuleEngine", "RuleFeatureSchema", "RuleModifier",
    "RuleRegistry", "RuleSpec", "TurnAction", "TurnKind", "compose_rule",
]
