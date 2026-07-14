from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from connect4_core import GameRules

from .model_runtime import ModelRegistry, ModelRegistryError, ModelUnavailableError
from .search import NumpyMCTS


MCTS_OPTIONS = {32, 64, 128, 256, 512, 1024}
AI_ROLES = ("combat", "hint", "win_rate")
DEFAULT_AI = {"model_id": "v2.2_balance", "mcts_sims": 128, "temperature": 1.0}


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


class CubeSpriteService:
    """Authoritative, thread-safe game and analysis service used by JSONL IPC."""

    def __init__(self, resource_dir: Path):
        self.game = GameRules()
        try:
            self.models = ModelRegistry(resource_dir)
        except ModelRegistryError as exc:
            raise ServiceError("MODEL_REGISTRY_INVALID", str(exc)) from exc
        self._lock = threading.RLock()
        self._revision_counter = 0
        self.session_id = ""
        self.mode = "pvp"
        self.human_player = 1
        self.board = self.game.get_init_board()
        self.current_player = 1
        self.history: list[dict[str, int]] = []
        self.status = "playing"
        self.winner: int | None = None
        self.winning_line: list[dict[str, int]] = []
        self.last_move: dict[str, int] | None = None
        self.settings = {
            "roles": {role: dict(DEFAULT_AI) for role in AI_ROLES},
            "preload_hint": False,
        }
        self._new_session("pvp", 1)

    def handle(self, command: str, params: dict[str, Any] | None):
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ServiceError("INVALID_PARAMS", "Request params must be a JSON object.")
        handlers = {
            "system.ping": lambda _: {"pong": True, "version": "0.1.0"},
            "models.list": lambda _: {"models": self.models.list_models()},
            "settings.get": lambda _: self._settings_snapshot(),
            "settings.update": self._cmd_settings_update,
            "settings.set_preload_hint": self._cmd_set_preload_hint,
            "game.new": self._cmd_new,
            "game.state": lambda _: self.snapshot(),
            "game.move": self._cmd_move,
            "game.ai_move": self._cmd_ai_move,
            "game.undo": self._cmd_undo,
            "game.restart": self._cmd_restart,
            "analysis.hint": self._cmd_hint,
            "analysis.win_rate": self._cmd_win_rate,
        }
        try:
            handler = handlers[str(command)]
        except KeyError as exc:
            raise ServiceError("UNKNOWN_COMMAND", f"Unknown command: {command}") from exc
        return handler(params)

    def initialize(self) -> dict[str, Any]:
        return {
            "backend_version": "0.1.0",
            "protocol_version": 1,
            "board": {"layers": 6, "size": 5, "connect_n": 4},
            "mcts_options": sorted(MCTS_OPTIONS),
            "models": self.models.list_models(),
            "settings": self._settings_snapshot(),
            "state": self.snapshot(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            legal_moves: list[dict[str, int]] = []
            if self.status == "playing":
                for action in np.flatnonzero(self.game.get_valid_moves(self.board) > 0):
                    layer, row, col = self.game.action_to_coords(int(action))
                    legal_moves.append({"action": int(action), "layer": layer, "row": row, "col": col})
            return {
                "session_id": self.session_id,
                "revision": self._revision_counter,
                "mode": self.mode,
                "human_player": self.human_player,
                "board": self.board.astype(int).tolist(),
                "current_player": self.current_player,
                "move_count": len(self.history),
                "status": self.status,
                "winner": self.winner,
                "last_move": deepcopy(self.last_move),
                "winning_line": deepcopy(self.winning_line),
                "legal_moves": legal_moves,
                "can_undo": bool(self.history) if self.mode == "pvp" else any(
                    move["player"] == self.human_player for move in self.history
                ),
            }

    def _settings_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self.settings)

    def _cmd_settings_update(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_revision(params, required=True)
            changed = False
            if "preload_hint" in params:
                preload = params["preload_hint"]
                if not isinstance(preload, bool):
                    raise ServiceError("INVALID_PRELOAD_HINT", "preload_hint must be a boolean.")
                if preload != self.settings["preload_hint"]:
                    self.settings["preload_hint"] = preload
                    changed = True
            roles = params.get("roles", {})
            if not isinstance(roles, dict):
                raise ServiceError("INVALID_AI_SETTINGS", "roles must be a JSON object.")
            unknown = set(roles) - set(AI_ROLES)
            if unknown:
                raise ServiceError("INVALID_AI_ROLE", f"Unknown AI role(s): {', '.join(sorted(unknown))}")
            for role, partial in roles.items():
                if not isinstance(partial, dict):
                    raise ServiceError("INVALID_AI_SETTINGS", f"Settings for {role} must be an object.")
                merged = dict(self.settings["roles"][role])
                merged.update(partial)
                validated = self._validate_ai_config(merged)
                if validated != self.settings["roles"][role]:
                    self.settings["roles"][role] = validated
                    changed = True
            if changed:
                self._bump_revision()
            return {"settings": self._settings_snapshot(), "state": self.snapshot()}

    def _cmd_set_preload_hint(self, params: dict[str, Any]) -> dict[str, Any]:
        if "enabled" not in params:
            raise ServiceError("INVALID_PRELOAD_HINT", "enabled is required.")
        forwarded = dict(params)
        forwarded["preload_hint"] = forwarded.pop("enabled")
        return self._cmd_settings_update(forwarded)

    def _cmd_new(self, params: dict[str, Any]) -> dict[str, Any]:
        mode = str(params.get("mode", "pvp"))
        human_player = self._as_int(params.get("human_player", 1), "human_player")
        if mode not in {"pvp", "pvai"}:
            raise ServiceError("INVALID_MODE", f"Unsupported game mode: {mode}")
        if human_player not in {-1, 1}:
            raise ServiceError("INVALID_PLAYER", "human_player must be +1 or -1")
        with self._lock:
            self._new_session(mode, human_player)
            return self.snapshot()

    def _cmd_move(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_revision(params, required=True)
            if self.status != "playing":
                raise ServiceError("GAME_FINISHED", "The game has already ended.")
            if self.mode == "pvai" and self.current_player != self.human_player:
                raise ServiceError("NOT_HUMAN_TURN", "It is the AI player's turn.")
            try:
                layer = self._as_int(params["layer"], "layer")
                row = self._as_int(params["row"], "row")
                col = self._as_int(params["col"], "col")
                action = self.game.coords_to_action(layer, row, col)
                self._apply_action(action, self.current_player)
            except KeyError as exc:
                raise ServiceError("INVALID_PARAMS", f"Missing move coordinate: {exc.args[0]}") from exc
            except (TypeError, ValueError) as exc:
                raise ServiceError("ILLEGAL_MOVE", str(exc)) from exc
            return self.snapshot()

    def _cmd_ai_move(self, params: dict[str, Any]) -> dict[str, Any]:
        board, player, revision, session_id, ai = self._capture_analysis(params, "combat", require_ai_turn=True)
        result = self._search(board, player, ai)
        with self._lock:
            self._assert_fresh(session_id, revision)
            self._apply_action(result.action, player)
            snapshot = self.snapshot()
            snapshot["analysis"] = {"value": result.value, "policy": result.policy}
            return snapshot

    def _cmd_hint(self, params: dict[str, Any]) -> dict[str, Any]:
        board, player, revision, session_id, ai = self._capture_analysis(params, "hint")
        result = self._search(board, player, ai)
        self._assert_fresh(session_id, revision)
        layer, row, col = self.game.action_to_coords(result.action)
        return {
            "session_id": session_id,
            "for_revision": revision,
            "move": {"action": result.action, "layer": layer, "row": row, "col": col},
            "value": result.value,
        }

    def _cmd_win_rate(self, params: dict[str, Any]) -> dict[str, Any]:
        board, player, revision, session_id, ai = self._capture_analysis(params, "win_rate")
        result = self._search(board, player, ai)
        self._assert_fresh(session_id, revision)
        current_probability = (float(result.value) + 1.0) / 2.0
        red = current_probability if player == 1 else 1.0 - current_probability
        red = float(np.clip(red, 0.0, 1.0))
        return {
            "session_id": session_id,
            "for_revision": revision,
            "red": red,
            "blue": 1.0 - red,
            "estimate": "model_mcts",
        }

    def _cmd_undo(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_revision(params, required=True)
            if not self.history:
                return self.snapshot()
            if self.mode == "pvp":
                self.history.pop()
            else:
                if not any(move["player"] == self.human_player for move in self.history):
                    return self.snapshot()
                # Undo an AI reply and the human move that preceded it. If AI has
                # not replied yet, undo the pending human move only.
                if self.history and self.history[-1]["player"] != self.human_player:
                    self.history.pop()
                if self.history and self.history[-1]["player"] == self.human_player:
                    self.history.pop()
            self._rebuild_from_history()
            self._bump_revision()
            return self.snapshot()

    def _cmd_restart(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_revision(params, required=True)
            self._reset_position()
            self._bump_revision()
            return self.snapshot()

    def _new_session(self, mode: str, human_player: int) -> None:
        self.session_id = str(uuid.uuid4())
        self.mode = mode
        self.human_player = human_player
        self._reset_position()
        self._bump_revision()

    def _reset_position(self) -> None:
        self.board = self.game.get_init_board()
        self.current_player = 1
        self.history = []
        self.status = "playing"
        self.winner = None
        self.winning_line = []
        self.last_move = None

    def _apply_action(self, action: int, player: int) -> None:
        if int(player) != self.current_player:
            raise ServiceError("WRONG_PLAYER", "The move player does not match the authoritative turn.")
        try:
            self.board, next_player = self.game.get_next_state(self.board, player, action)
        except (ValueError, KeyError, TypeError) as exc:
            raise ServiceError("ILLEGAL_MOVE", str(exc)) from exc
        layer, row, col = self.game.action_to_coords(action)
        move = {"action": int(action), "layer": layer, "row": row, "col": col, "player": int(player)}
        self.history.append(move)
        self.last_move = move
        line = find_winning_line(self.board, player, self.game.connect_n)
        if line:
            self.status = "won"
            self.winner = int(player)
            self.winning_line = line
        elif not np.any(self.game.get_valid_moves(self.board)):
            self.status = "draw"
            self.winner = 0
        else:
            self.current_player = int(next_player)
        self._bump_revision()

    def _rebuild_from_history(self) -> None:
        moves = list(self.history)
        self._reset_position()
        for move in moves:
            if self.status != "playing":
                raise ServiceError("CORRUPT_HISTORY", "History contains moves after the game ended.")
            self.board, next_player = self.game.get_next_state(self.board, move["player"], move["action"])
            self.current_player = int(next_player)
            self.history.append(move)
            self.last_move = move
            line = find_winning_line(self.board, move["player"], self.game.connect_n)
            if line:
                self.status = "won"
                self.winner = int(move["player"])
                self.winning_line = line
            elif not np.any(self.game.get_valid_moves(self.board)):
                self.status = "draw"
                self.winner = 0

    def _capture_analysis(self, params: dict[str, Any], role: str, require_ai_turn: bool = False):
        with self._lock:
            self._check_revision(params, required=True)
            if self.status != "playing":
                raise ServiceError("GAME_FINISHED", "The game has already ended.")
            if require_ai_turn and (self.mode != "pvai" or self.current_player == self.human_player):
                raise ServiceError("NOT_AI_TURN", "It is not the AI player's turn.")
            ai = dict(self.settings["roles"][role])
            override = params.get("ai")
            if override is not None:
                if not isinstance(override, dict):
                    raise ServiceError("INVALID_AI_SETTINGS", "ai must be a JSON object.")
                ai.update(override)
            ai = self._validate_ai_config(ai)
            return self.board.copy(), self.current_player, self._revision_counter, self.session_id, ai

    def _search(self, board: np.ndarray, player: int, ai: dict[str, Any]):
        try:
            predictor = self.models.predictor(ai["model_id"])
        except (ModelUnavailableError, ModelRegistryError, OSError) as exc:
            raise ServiceError("MODEL_UNAVAILABLE", str(exc)) from exc
        try:
            return NumpyMCTS(
                self.game,
                predictor,
                simulations=ai["mcts_sims"],
                temperature=ai["temperature"],
            ).run(board, player)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("INFERENCE_FAILED", str(exc)) from exc

    def _validate_ai_config(self, config: dict[str, Any]) -> dict[str, Any]:
        allowed = {"model_id", "mcts_sims", "temperature"}
        unknown = set(config) - allowed
        if unknown:
            raise ServiceError("INVALID_AI_SETTINGS", f"Unknown AI setting(s): {', '.join(sorted(unknown))}")
        model_id = str(config.get("model_id", DEFAULT_AI["model_id"]))
        try:
            spec = self.models.get(model_id)
        except ModelUnavailableError as exc:
            raise ServiceError("MODEL_UNAVAILABLE", str(exc)) from exc
        if getattr(spec, "placeholder", False):
            raise ServiceError("MODEL_UNAVAILABLE", f"Model {spec.display_name} is a future placeholder.")
        simulations = self._as_int(config.get("mcts_sims", 128), "mcts_sims")
        if simulations not in MCTS_OPTIONS:
            raise ServiceError("INVALID_MCTS", f"MCTS simulations must be one of {sorted(MCTS_OPTIONS)}")
        try:
            temperature = float(config.get("temperature", 1.0))
        except (TypeError, ValueError) as exc:
            raise ServiceError("INVALID_TEMPERATURE", "temperature must be a number from 0 to 5.") from exc
        if not np.isfinite(temperature) or not 0.0 <= temperature <= 5.0:
            raise ServiceError("INVALID_TEMPERATURE", "temperature must be a number from 0 to 5.")
        # The UI uses a 0.1 step; normalize floating-point noise at the protocol boundary.
        temperature = round(temperature, 1)
        return {"model_id": model_id, "mcts_sims": simulations, "temperature": temperature}

    def _check_revision(self, params: dict[str, Any], required: bool = False) -> None:
        session_id = params.get("session_id")
        revision = params.get("expected_revision")
        if required and (session_id is None or revision is None):
            raise ServiceError(
                "MISSING_CONCURRENCY_TOKEN",
                "session_id and expected_revision are required for this command.",
            )
        if session_id is not None and str(session_id) != self.session_id:
            raise ServiceError("STALE_SESSION", "The game session has changed.")
        if revision is not None:
            parsed = self._as_int(revision, "expected_revision")
            if parsed != self._revision_counter:
                raise ServiceError("STALE_REVISION", "The game state changed while this request was pending.")

    def _assert_fresh(self, session_id: str, revision: int) -> None:
        with self._lock:
            if session_id != self.session_id:
                raise ServiceError("STALE_SESSION", "The game session changed while this request was pending.")
            if revision != self._revision_counter:
                raise ServiceError("STALE_REVISION", "The game state changed while this request was pending.")

    def _bump_revision(self) -> None:
        self._revision_counter += 1

    @staticmethod
    def _as_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ServiceError("INVALID_PARAMS", f"{name} must be an integer.")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ServiceError("INVALID_PARAMS", f"{name} must be an integer.") from exc
        if isinstance(value, float) and not value.is_integer():
            raise ServiceError("INVALID_PARAMS", f"{name} must be an integer.")
        return parsed


def find_winning_line(board: np.ndarray, player: int, connect_n: int = 4) -> list[dict[str, int]]:
    board = np.asarray(board, dtype=np.int8)
    # Only one direction from each opposite pair is needed. Starting at a cell
    # whose predecessor is occupied is skipped, yielding one stable maximal line.
    directions = [
        (dz, dy, dx)
        for dz in (0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0) and not (dz == 0 and dy < 0) and not (dz == 0 and dy == 0 and dx < 0)
    ]
    shape = board.shape
    for layer, row, col in np.argwhere(board == int(player)):
        layer, row, col = int(layer), int(row), int(col)
        for dz, dy, dx in directions:
            previous = (layer - dz, row - dy, col - dx)
            if _inside(previous, shape) and board[previous] == player:
                continue
            cells: list[dict[str, int]] = []
            nl, nr, nc = layer, row, col
            while _inside((nl, nr, nc), shape) and board[nl, nr, nc] == player:
                cells.append({"layer": nl, "row": nr, "col": nc})
                nl, nr, nc = nl + dz, nr + dy, nc + dx
            if len(cells) >= connect_n:
                return cells[:connect_n]
    return []


def _inside(coords: tuple[int, int, int], shape: tuple[int, ...]) -> bool:
    return all(0 <= coordinate < dimension for coordinate, dimension in zip(coords, shape))
