from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np

from connect4_core.game_rules import (
    BOARD_SHAPE,
    BOARD_SIZE,
    CONNECT_N,
    MAX_LAYERS,
    WIN_DIRECTIONS,
)

from .specs import (
    CLASSIC_RULE,
    DEFAULT_RULE_REGISTRY,
    Layer0WinMode,
    NoLegalPlacementMode,
    RuleRegistry,
    RuleSpec,
    VerticalWinMode,
)


COLUMN_COUNT = BOARD_SIZE * BOARD_SIZE
MAX_PLACEMENTS = MAX_LAYERS * COLUMN_COUNT
MAX_TURNS = MAX_PLACEMENTS * 2
FIRST_PLAYER = 1
SECOND_PLAYER = -1
D4_SYMMETRIES = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "reflect",
    "reflect_rotate90",
    "reflect_rotate180",
    "reflect_rotate270",
)


class TurnKind(str, Enum):
    PLACE = "place"
    FORCED_PASS = "forced_pass"


@dataclass(frozen=True)
class TurnAction:
    kind: TurnKind
    column: int | None = None

    def __post_init__(self) -> None:
        if self.kind == TurnKind.PLACE:
            if isinstance(self.column, bool) or not isinstance(self.column, Integral):
                raise ValueError("A place action requires an integer column.")
            object.__setattr__(self, "column", int(self.column))
            if not 0 <= self.column < COLUMN_COUNT:
                raise ValueError(f"Column must be in [0, {COLUMN_COUNT - 1}].")
        elif self.column is not None:
            raise ValueError("A forced pass must not contain a column.")

    @classmethod
    def place(cls, column: int) -> TurnAction:
        return cls(TurnKind.PLACE, column)

    @classmethod
    def forced_pass(cls) -> TurnAction:
        return cls(TurnKind.FORCED_PASS)


class GameOutcome(str, Enum):
    ONGOING = "ongoing"
    DRAW = "draw"
    FIRST_PLAYER_WIN = "first_player_win"
    SECOND_PLAYER_WIN = "second_player_win"

    @property
    def winner(self) -> int | None:
        if self == GameOutcome.FIRST_PLAYER_WIN:
            return FIRST_PLAYER
        if self == GameOutcome.SECOND_PLAYER_WIN:
            return SECOND_PLAYER
        return None


@dataclass(frozen=True, eq=False)
class GameState:
    board: np.ndarray
    player_to_move: int
    rule_id: str
    turn_index: int = 0
    placement_count: int = 0
    consecutive_passes: int = 0
    last_action: TurnAction | None = None
    outcome: GameOutcome = GameOutcome.ONGOING

    def __post_init__(self) -> None:
        board = np.array(self.board, dtype=np.int8, copy=True)
        if board.shape != BOARD_SHAPE:
            raise ValueError(f"Board shape must be {BOARD_SHAPE}, got {board.shape}.")
        if not np.all(np.isin(board, (-1, 0, 1))):
            raise ValueError("Board cells must contain only -1, 0, or +1.")
        board.setflags(write=False)
        object.__setattr__(self, "board", board)
        if self.player_to_move not in (FIRST_PLAYER, SECOND_PLAYER):
            raise ValueError("player_to_move must be +1 (first) or -1 (second).")
        if not self.rule_id:
            raise ValueError("rule_id must not be empty.")
        if self.turn_index < 0 or self.placement_count < 0:
            raise ValueError("turn_index and placement_count must be non-negative.")
        if self.placement_count > MAX_PLACEMENTS:
            raise ValueError(f"placement_count must not exceed {MAX_PLACEMENTS}.")
        if not 0 <= self.consecutive_passes <= 2:
            raise ValueError("consecutive_passes must be in [0, 2].")
        if self.turn_index < self.placement_count:
            raise ValueError("turn_index cannot be smaller than placement_count.")

    @property
    def terminal(self) -> bool:
        return self.outcome != GameOutcome.ONGOING


class RuleEngine:
    """Authoritative transitions for the fixed 6x5x5 gravity game contract."""

    def __init__(
        self,
        rule: str | int | RuleSpec = CLASSIC_RULE,
        *,
        registry: RuleRegistry = DEFAULT_RULE_REGISTRY,
    ) -> None:
        self.registry = registry
        self.spec = registry.get(rule)
        if not self.spec.d4_symmetry:
            raise ValueError("The V1 engine requires rules that preserve horizontal D4 symmetry.")

    @property
    def valid_symmetries(self) -> tuple[str, ...]:
        return D4_SYMMETRIES

    def initial_state(self, *, player_to_move: int = FIRST_PLAYER) -> GameState:
        return GameState(
            board=np.zeros(BOARD_SHAPE, dtype=np.int8),
            player_to_move=player_to_move,
            rule_id=self.spec.rule_id,
        )

    def state_from_board(
        self,
        board: np.ndarray,
        *,
        player_to_move: int,
        turn_index: int | None = None,
        placement_count: int | None = None,
        consecutive_passes: int = 0,
        last_action: TurnAction | None = None,
        outcome: GameOutcome = GameOutcome.ONGOING,
    ) -> GameState:
        validated = self._validated_gravity_board(board)
        occupied = int(np.count_nonzero(validated))
        placements = occupied if placement_count is None else int(placement_count)
        turns = placements + int(consecutive_passes) if turn_index is None else int(turn_index)
        state = GameState(
            board=validated,
            player_to_move=int(player_to_move),
            rule_id=self.spec.rule_id,
            turn_index=turns,
            placement_count=placements,
            consecutive_passes=int(consecutive_passes),
            last_action=last_action,
            outcome=outcome,
        )
        if state.terminal:
            return state
        return self._resolve_no_placement_terminal(state)

    def legal_column_mask(self, state: GameState) -> np.ndarray:
        self._validate_state(state)
        mask = np.zeros(COLUMN_COUNT, dtype=np.int8)
        if state.terminal:
            return mask
        heights = np.count_nonzero(state.board, axis=0).reshape(-1)
        for column in np.flatnonzero(heights < MAX_LAYERS):
            column = int(column)
            layer = int(heights[column])
            if (
                state.player_to_move == FIRST_PLAYER
                and self.spec.p1_vertical_mode == VerticalWinMode.ILLEGAL
                and self._would_form_vertical_line(state.board, layer, column)
            ):
                continue
            mask[column] = 1
        return mask

    def required_action(self, state: GameState) -> TurnAction | None:
        self._validate_state(state)
        if state.terminal or np.any(self.legal_column_mask(state)):
            return None
        if self.spec.no_legal_placement_mode == NoLegalPlacementMode.FORCED_PASS:
            return TurnAction.forced_pass()
        return None

    def step(self, state: GameState, action: TurnAction) -> GameState:
        self._validate_state(state)
        if state.terminal:
            raise ValueError(f"Cannot act after terminal outcome {state.outcome.value!r}.")
        if not isinstance(action, TurnAction):
            raise TypeError("action must be a TurnAction.")
        if action.kind == TurnKind.FORCED_PASS:
            return self._apply_forced_pass(state, action)
        return self._apply_placement(state, action)

    def legacy_action_for_column(self, state: GameState, column: int) -> int:
        """Map a legal 25-way column to its deterministic 150-way coordinate."""
        self._validate_state(state)
        action = TurnAction.place(column)
        if self.legal_column_mask(state)[action.column] == 0:
            raise ValueError(f"Column {action.column} is not legal under rule {self.spec.rule_id!r}.")
        layer = int(np.count_nonzero(state.board[:, action.column // BOARD_SIZE, action.column % BOARD_SIZE]))
        return layer * COLUMN_COUNT + action.column

    def _apply_forced_pass(self, state: GameState, action: TurnAction) -> GameState:
        if self.spec.no_legal_placement_mode != NoLegalPlacementMode.FORCED_PASS:
            raise ValueError(f"Rule {self.spec.rule_id!r} does not enable forced pass.")
        if np.any(self.legal_column_mask(state)):
            raise ValueError("Forced pass is only legal when no placement is legal.")
        passes = state.consecutive_passes + 1
        outcome = GameOutcome.DRAW if passes >= 2 else GameOutcome.ONGOING
        return GameState(
            board=state.board,
            player_to_move=-state.player_to_move,
            rule_id=state.rule_id,
            turn_index=state.turn_index + 1,
            placement_count=state.placement_count,
            consecutive_passes=passes,
            last_action=action,
            outcome=outcome,
        )

    def _apply_placement(self, state: GameState, action: TurnAction) -> GameState:
        assert action.column is not None
        heights = np.count_nonzero(state.board, axis=0).reshape(-1)
        layer = int(heights[action.column])
        if layer >= MAX_LAYERS:
            raise ValueError(f"Column {action.column} is full.")
        if (
            state.player_to_move == FIRST_PLAYER
            and self.spec.p1_vertical_mode == VerticalWinMode.ILLEGAL
            and self._would_form_vertical_line(state.board, layer, action.column)
        ):
            raise ValueError(
                f"Column {action.column} is illegal because it completes a forbidden vertical line."
            )
        if self.legal_column_mask(state)[action.column] == 0:
            raise ValueError(f"Column {action.column} is not legal under rule {self.spec.rule_id!r}.")

        row, col = divmod(action.column, BOARD_SIZE)
        board = np.array(state.board, copy=True)
        board[layer, row, col] = state.player_to_move
        outcome = self._outcome_after_placement(
            board,
            player=state.player_to_move,
            layer=layer,
            row=row,
            col=col,
        )
        next_state = GameState(
            board=board,
            player_to_move=-state.player_to_move,
            rule_id=state.rule_id,
            turn_index=state.turn_index + 1,
            placement_count=state.placement_count + 1,
            consecutive_passes=0,
            last_action=action,
            outcome=outcome,
        )
        if next_state.terminal:
            return next_state
        return self._resolve_no_placement_terminal(next_state)

    def _resolve_no_placement_terminal(self, state: GameState) -> GameState:
        if np.any(self.legal_column_mask(state)):
            return state
        mode = self.spec.no_legal_placement_mode
        if mode == NoLegalPlacementMode.FORCED_PASS:
            return state
        if mode == NoLegalPlacementMode.DRAW:
            outcome = GameOutcome.DRAW
        else:
            outcome = (
                GameOutcome.SECOND_PLAYER_WIN
                if state.player_to_move == FIRST_PLAYER
                else GameOutcome.FIRST_PLAYER_WIN
            )
        return GameState(
            board=state.board,
            player_to_move=state.player_to_move,
            rule_id=state.rule_id,
            turn_index=state.turn_index,
            placement_count=state.placement_count,
            consecutive_passes=state.consecutive_passes,
            last_action=state.last_action,
            outcome=outcome,
        )

    def _outcome_after_placement(
        self,
        board: np.ndarray,
        *,
        player: int,
        layer: int,
        row: int,
        col: int,
    ) -> GameOutcome:
        scoring_line = False
        for direction, line_length in self._completed_line_directions(
            board, player=player, layer=layer, row=row, col=col
        ):
            dz, _, _ = direction
            if player == FIRST_PLAYER:
                if dz != 0 and direction == (1, 0, 0):
                    if self.spec.p1_vertical_mode == VerticalWinMode.IGNORED:
                        continue
                if dz == 0 and layer == 0:
                    if self.spec.p1_layer0_mode == Layer0WinMode.IGNORED:
                        continue
            if line_length >= CONNECT_N:
                scoring_line = True
                break
        if not scoring_line:
            return GameOutcome.ONGOING
        return (
            GameOutcome.FIRST_PLAYER_WIN
            if player == FIRST_PLAYER
            else GameOutcome.SECOND_PLAYER_WIN
        )

    def _completed_line_directions(
        self,
        board: np.ndarray,
        *,
        player: int,
        layer: int,
        row: int,
        col: int,
    ) -> tuple[tuple[tuple[int, int, int], int], ...]:
        completed: list[tuple[tuple[int, int, int], int]] = []
        for direction in WIN_DIRECTIONS:
            dz, dy, dx = direction
            count = 1
            for sign in (-1, 1):
                for step in range(1, CONNECT_N + 2):
                    nl = layer + sign * step * dz
                    nr = row + sign * step * dy
                    nc = col + sign * step * dx
                    if not (
                        0 <= nl < MAX_LAYERS
                        and 0 <= nr < BOARD_SIZE
                        and 0 <= nc < BOARD_SIZE
                        and board[nl, nr, nc] == player
                    ):
                        break
                    count += 1
            if count >= CONNECT_N:
                completed.append((direction, count))
        return tuple(completed)

    def _would_form_vertical_line(self, board: np.ndarray, layer: int, column: int) -> bool:
        if layer >= MAX_LAYERS:
            return False
        row, col = divmod(column, BOARD_SIZE)
        candidate = np.array(board, copy=True)
        candidate[layer, row, col] = FIRST_PLAYER
        return any(
            direction == (1, 0, 0)
            for direction, _ in self._completed_line_directions(
                candidate,
                player=FIRST_PLAYER,
                layer=layer,
                row=row,
                col=col,
            )
        )

    def _validated_gravity_board(self, board: np.ndarray) -> np.ndarray:
        raw = np.asarray(board)
        if raw.shape != BOARD_SHAPE:
            raise ValueError(f"Board shape must be {BOARD_SHAPE}, got {raw.shape}.")
        if not np.all(np.isin(raw, (-1, 0, 1))):
            raise ValueError("Board cells must contain only -1, 0, or +1.")
        occupied = raw != 0
        floating = occupied[1:] & ~occupied[:-1]
        if np.any(floating):
            raise ValueError("Board violates gravity: an occupied cell has empty support below it.")
        return np.asarray(raw, dtype=np.int8)

    def _validate_state(self, state: GameState) -> None:
        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState.")
        if state.rule_id != self.spec.rule_id:
            raise ValueError(
                f"State rule {state.rule_id!r} does not match engine rule {self.spec.rule_id!r}."
            )
