from __future__ import annotations

import math
from numbers import Integral

import numpy as np


BOARD_SIZE = 5
MAX_LAYERS = 6
CONNECT_N = 4
BOARD_SHAPE = (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
WIN_DIRECTIONS = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
    and next(component for component in (dz, dy, dx) if component != 0) > 0
)


def _strict_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def action_to_coords(action, board_size=BOARD_SIZE, max_layers=MAX_LAYERS):
    board_size = _strict_int(board_size, "board_size")
    max_layers = _strict_int(max_layers, "max_layers")
    action = _strict_int(action, "action")
    action_size = max_layers * board_size * board_size
    if action < 0 or action >= action_size:
        raise ValueError(f"Action {action} out of range [0, {action_size - 1}].")
    layer = action // (board_size * board_size)
    remainder = action % (board_size * board_size)
    return layer, remainder // board_size, remainder % board_size


def coords_to_action(layer, row, col, board_size=BOARD_SIZE, max_layers=MAX_LAYERS):
    board_size = _strict_int(board_size, "board_size")
    max_layers = _strict_int(max_layers, "max_layers")
    layer = _strict_int(layer, "layer")
    row = _strict_int(row, "row")
    col = _strict_int(col, "col")
    if not (0 <= layer < max_layers and 0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Coordinates ({layer}, {row}, {col}) are outside {(max_layers, board_size, board_size)}.")
    return layer * board_size * board_size + row * board_size + col


class GameRules:
    """Rules for a gravity-based 3D Connect Four board using 0/+1/-1 cells."""

    def __init__(self, board_size=BOARD_SIZE, max_layers=MAX_LAYERS, connect_n=CONNECT_N):
        self.board_size = int(board_size)
        self.max_layers = int(max_layers)
        self.connect_n = int(connect_n)
        if self.board_size <= 0 or self.max_layers <= 0 or self.connect_n <= 1:
            raise ValueError("board_size and max_layers must be positive; connect_n must be at least 2.")
        self.board_shape = (self.max_layers, self.board_size, self.board_size)
        self.board = self.get_init_board()
        self.player = 1
        self.last_move = None

    def get_init_board(self):
        return np.zeros(self.board_shape, dtype=np.int8)

    def get_board_size(self):
        return self.board_shape

    def get_action_size(self):
        return self.max_layers * self.board_size * self.board_size

    def action_to_coords(self, action):
        return action_to_coords(action, self.board_size, self.max_layers)

    def coords_to_action(self, layer, row, col):
        return coords_to_action(layer, row, col, self.board_size, self.max_layers)

    def get_next_state(self, board, player, action):
        board = self._validated_board(board)
        player = int(player)
        if player not in (-1, 1):
            raise ValueError(f"Player must be +1 or -1, got {player}.")
        layer, row, col = self.action_to_coords(action)
        self._validate_move(board, layer, row, col, action=int(action))
        new_board = np.array(board, dtype=np.int8, copy=True)
        new_board[layer, row, col] = player
        return new_board, -player

    def get_next_state_fast(self, board, player, action):
        """Apply a trusted legal move without repeating public API validation."""
        action = int(action)
        plane = self.board_size * self.board_size
        layer = action // plane
        remainder = action - layer * plane
        row = remainder // self.board_size
        col = remainder - row * self.board_size
        new_board = np.array(board, dtype=np.int8, copy=True)
        new_board[layer, row, col] = int(player)
        return new_board, -int(player)

    def get_valid_moves(self, board):
        board = self._validated_board(board)
        valid_moves = np.zeros(self.get_action_size(), dtype=np.int8)
        for row in range(self.board_size):
            for col in range(self.board_size):
                for layer in range(self.max_layers):
                    if board[layer, row, col] == 0:
                        if layer == 0 or board[layer - 1, row, col] != 0:
                            valid_moves[self.coords_to_action(layer, row, col)] = 1
                        break
        return valid_moves

    def get_valid_moves_fast(self, board):
        """Return gravity-valid legacy actions for a trusted internal board."""
        board = np.asarray(board)
        plane = self.board_size * self.board_size
        heights = np.count_nonzero(board, axis=0).reshape(-1)
        open_columns = heights < self.max_layers
        valid_moves = np.zeros(self.get_action_size(), dtype=np.int8)
        column_indices = np.flatnonzero(open_columns)
        valid_moves[heights[column_indices] * plane + column_indices] = 1
        return valid_moves

    def get_game_ended(self, board, player):
        board = self._validated_board(board)
        player = int(player)
        if self.check_win(board, -player):
            return -1
        if self.check_win(board, player):
            return 1
        return 1e-4 if not np.any(board == 0) else 0

    def get_game_ended_after_action_fast(self, board, player, action):
        """Check a trusted state by inspecting only lines through its last move."""
        board = np.asarray(board)
        action = int(action)
        plane = self.board_size * self.board_size
        layer = action // plane
        remainder = action - layer * plane
        row = remainder // self.board_size
        col = remainder - row * self.board_size
        stone = int(board[layer, row, col])
        if stone in (-1, 1) and self._check_win_from_coords_fast(board, stone, layer, row, col):
            return 1 if stone == int(player) else -1
        return 1e-4 if not np.any(board == 0) else 0

    def check_win(self, board, player):
        board = self._validated_board(board)
        player = int(player)
        occupied = np.argwhere(board == player)
        if occupied.size == 0:
            return False
        directions = [
            (dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)
        ]
        for layer, row, col in occupied:
            for dz, dy, dx in directions:
                for step in range(1, self.connect_n):
                    nl, nr, nc = layer + step * dz, row + step * dy, col + step * dx
                    if not self._is_inside(nl, nr, nc) or board[nl, nr, nc] != player:
                        break
                else:
                    return True
        return False

    def get_canonical_form(self, board, player):
        return self._validated_board(board) * int(player)

    def get_canonical_form_fast(self, board, player):
        """Canonicalize a trusted internal board without rescanning its contents."""
        return np.asarray(board, dtype=np.int8) * int(player)

    def get_symmetries(self, board, pi):
        board = self._validated_board(board)
        pi_board = np.asarray(pi).reshape(self.board_shape)
        symmetries = []
        for rotation in range(4):
            rotated_board = np.rot90(board, rotation, axes=(1, 2))
            rotated_pi = np.rot90(pi_board, rotation, axes=(1, 2))
            symmetries.append((np.array(rotated_board, copy=True), np.array(rotated_pi, copy=True).flatten()))
            symmetries.append((np.array(np.flip(rotated_board, axis=2), copy=True), np.array(np.flip(rotated_pi, axis=2), copy=True).flatten()))
        return symmetries

    def string_representation(self, board):
        return self._validated_board(board).tobytes()

    def reconstruct_board(self, moves, move_index=None):
        board = self.get_init_board()
        limit = len(moves) if move_index is None else max(0, min(int(move_index), len(moves)))
        for move in moves[:limit]:
            coords = move["coords"]
            layer, row, col = int(coords["layer"]), int(coords["row"]), int(coords["col"])
            self._validate_move(board, layer, row, col)
            player = int(move["player"])
            if player not in (-1, 1):
                raise ValueError(f"Player must be +1 or -1, got {player}.")
            board[layer, row, col] = player
        return board

    def _validated_board(self, board):
        raw_board = np.asarray(board)
        if raw_board.shape != self.board_shape:
            raise ValueError(f"Board shape must be {self.board_shape}, got {raw_board.shape}.")
        if not np.all(np.isin(raw_board, (-1, 0, 1))):
            raise ValueError("Board cells must contain only -1, 0, or +1.")
        return np.asarray(raw_board, dtype=np.int8)

    def _is_inside(self, layer, row, col):
        return 0 <= int(layer) < self.max_layers and 0 <= int(row) < self.board_size and 0 <= int(col) < self.board_size

    def _check_win_from_coords_fast(self, board, player, layer, row, col):
        for dz, dy, dx in WIN_DIRECTIONS:
            count = 1
            for sign in (-1, 1):
                for step in range(1, self.connect_n):
                    nl = layer + sign * step * dz
                    nr = row + sign * step * dy
                    nc = col + sign * step * dx
                    if not self._is_inside(nl, nr, nc) or board[nl, nr, nc] != player:
                        break
                    count += 1
            if count >= self.connect_n:
                return True
        return False

    def _validate_move(self, board, layer, row, col, action=None):
        if not self._is_inside(layer, row, col):
            label = f"Action {action}" if action is not None else "Move"
            raise ValueError(f"{label} maps outside the board to ({layer}, {row}, {col}).")
        if board[layer, row, col] != 0:
            label = f"Action {action}" if action is not None else "Move"
            raise ValueError(f"{label} targets occupied position ({layer}, {row}, {col}).")
        if layer > 0 and board[layer - 1, row, col] == 0:
            label = f"Action {action}" if action is not None else "Move"
            raise ValueError(f"{label} violates gravity at ({layer}, {row}, {col}).")


def infer_board_size_from_action_dim(action_dim, preferred_size=None):
    action_dim = int(action_dim)
    if action_dim <= 0:
        raise ValueError(f"Action dimension must be positive, got {action_dim}.")
    if preferred_size is not None:
        preferred_size = int(preferred_size)
        if preferred_size > 0 and action_dim % (preferred_size * preferred_size) == 0:
            return preferred_size, action_dim // (preferred_size * preferred_size)
    candidates = []
    for board_size in range(2, int(math.sqrt(action_dim)) + 1):
        if action_dim % (board_size * board_size) == 0:
            board_layers = action_dim // (board_size * board_size)
            if 1 <= board_layers <= 16:
                candidates.append((board_size, board_layers))
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate[0] == BOARD_SIZE:
            return candidate
    if candidates:
        return candidates[0]
    raise ValueError(f"Cannot infer board dimensions from action dimension {action_dim}.")
