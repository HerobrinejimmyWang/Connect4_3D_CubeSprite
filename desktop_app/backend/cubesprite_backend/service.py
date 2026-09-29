from __future__ import annotations

import threading
import tempfile
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from connect4_core import GameRules
from . import __version__
from connect4_core.rules import BAL5_R2_RULE_REGISTRY, RuleEngine, TurnAction, RuleFeatureSchema, GameOutcome

from .model_runtime import ModelRegistry, ModelRegistryError, ModelUnavailableError
from .replay_store import (
    ANALYSIS_FORMAT,
    ANALYSIS_PROTOCOL_VERSION,
    ReplayStore,
    ReplayStoreError,
    build_replay_frames,
    find_winning_line,
    replay_summary,
)
from .search import NumpyMCTS, RuleMCTS, find_forced_tactical_action


MCTS_OPTIONS = {16, 32, 64, 128, 256, 512, 1024}
AI_ROLES = ("combat", "hint", "win_rate")
DEFAULT_AI = {"model_id": "cubesprite_v4_flash_preview1", "mcts_sims": 256, "temperature": 0.4}


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


class CubeSpriteService:
    """Authoritative, thread-safe game and analysis service used by JSONL IPC."""

    def __init__(self, resource_dir: Path, data_dir: Path | None = None):
        self.game = GameRules()
        self.rule_id = "classic"
        self.engine = RuleEngine(self.rule_id, registry=BAL5_R2_RULE_REGISTRY)
        self._state = self.engine.initial_state()
        try:
            self.models = ModelRegistry(resource_dir)
        except ModelRegistryError as exc:
            raise ServiceError("MODEL_REGISTRY_INVALID", str(exc)) from exc
        self._temporary_data_dir = None
        if data_dir is None:
            self._temporary_data_dir = tempfile.TemporaryDirectory(prefix="cubesprite-data-")
            data_dir = Path(self._temporary_data_dir.name)
        try:
            self.replays = ReplayStore(data_dir, self.game)
        except OSError as exc:
            raise ServiceError("DATA_DIRECTORY_UNAVAILABLE", f"Cannot prepare the application data directory: {exc}") from exc
        self._lock = threading.RLock()
        self._analysis_generations: dict[str, int] = {}
        self._revision_counter = 0
        self.session_id = ""
        self._session_model_id = DEFAULT_AI["model_id"]
        self._session_model_ids: set[str] = set()
        self.mode = "pvp"
        self.human_player = 1
        self.board = self.game.get_init_board()
        self.current_player = 1
        self.history: list[dict[str, int]] = []
        self.status = "playing"
        self.winner: int | None = None
        self.winning_line: list[dict[str, int]] = []
        self.last_move: dict[str, int] | None = None
        self._session_origin_history: list[dict[str, int]] = []
        self._history_floor = 0
        self.settings = {
            "roles": {role: dict(DEFAULT_AI) for role in AI_ROLES},
            "preload_hint": False,
        }
        self._new_session("pvp", 1)

    def close(self) -> None:
        temporary = self._temporary_data_dir
        self._temporary_data_dir = None
        if temporary is not None:
            temporary.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def handle(self, command: str, params: dict[str, Any] | None):
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ServiceError("INVALID_PARAMS", "Request params must be a JSON object.")
        handlers = {
            "system.ping": lambda _: {"pong": True, "version": __version__},
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
            "analysis.tactical_hint": self._cmd_tactical_hint,
            "analysis.win_rate": self._cmd_win_rate,
            "replay.list": self._cmd_replay_list,
            "replay.save": self._cmd_replay_save,
            "replay.open": self._cmd_replay_open,
            "replay.delete": self._cmd_replay_delete,
            "replay.export": self._cmd_replay_export,
            "replay.import": self._cmd_replay_import,
            "replay.hint": self._cmd_replay_hint,
            "replay.analyze": self._cmd_replay_analyze,
            "replay.continue": self._cmd_replay_continue,
        }
        try:
            handler = handlers[str(command)]
        except KeyError as exc:
            raise ServiceError("UNKNOWN_COMMAND", f"Unknown command: {command}") from exc
        try:
            return handler(params)
        except ReplayStoreError as exc:
            raise ServiceError(exc.code, str(exc), exc.details) from exc

    def initialize(self) -> dict[str, Any]:
        return {
            "backend_version": __version__,
            "protocol_version": 1,
            "board": {"layers": 6, "size": 5, "connect_n": 4},
            "mcts_options": sorted(MCTS_OPTIONS),
            "rule_ids": [spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs],
            "models": self.models.list_models(),
            "settings": self._settings_snapshot(),
            "state": self.snapshot(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            undoable_history = self.history[self._history_floor :]
            legal_moves: list[dict[str, int]] = []
            if self.status == "playing":
                for column in np.flatnonzero(self.engine.legal_column_mask(self._state)):
                    action = self.engine.legacy_action_for_column(self._state, int(column))
                    layer, row, col = self.game.action_to_coords(int(action))
                    legal_moves.append({"action": int(action), "layer": layer, "row": row, "col": col})
            return {
                "session_id": self.session_id,
                "revision": self._revision_counter,
                "mode": self.mode,
                "rule_id": self.rule_id,
                "human_player": self.human_player,
                "board": self.board.astype(int).tolist(),
                "current_player": self.current_player,
                "move_count": len(self.history),
                "status": self.status,
                "winner": self.winner,
                "last_move": deepcopy(self.last_move),
                "winning_line": deepcopy(self.winning_line),
                "legal_moves": legal_moves,
                "can_undo": bool(undoable_history) if self.mode == "pvp" else any(
                    move["player"] == self.human_player for move in undoable_history
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
                if self.rule_id != "classic" and self.rule_id not in self.models.get(validated["model_id"]).supported_rule_ids:
                    raise ServiceError("MODEL_UNSUPPORTED_RULE", "The selected model does not support this rule.")
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
        rule_id = str(params.get("rule_id", "classic"))
        ai = self._combat_config_for_new_game(params)
        human_player = self._as_int(params.get("human_player", 1), "human_player")
        if mode not in {"pvp", "pvai"}:
            raise ServiceError("INVALID_MODE", f"Unsupported game mode: {mode}")
        if human_player not in {-1, 1}:
            raise ServiceError("INVALID_PLAYER", "human_player must be +1 or -1")
        try:
            BAL5_R2_RULE_REGISTRY.get(rule_id)
        except KeyError as exc:
            raise ServiceError("INVALID_RULE", f"Unsupported game rule: {rule_id}") from exc
        with self._lock:
            if mode == "pvai" and rule_id not in self.models.get(ai["model_id"]).supported_rule_ids:
                raise ServiceError("MODEL_UNSUPPORTED_RULE", "The selected model does not support this rule.")
            self.rule_id = rule_id
            self.engine = RuleEngine(rule_id, registry=BAL5_R2_RULE_REGISTRY)
            self._new_session(mode, human_player)
            self._session_model_id = ai["model_id"]
            self._session_model_ids = {ai["model_id"]} if mode == "pvai" else set()
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
        forced_tactics = params.get("forced_tactics", True)
        if not isinstance(forced_tactics, bool):
            raise ServiceError("INVALID_FORCED_TACTICS", "forced_tactics must be a boolean.")
        result = self._search(board, player, ai, forced_tactics=forced_tactics)
        with self._lock:
            self._assert_fresh(session_id, revision)
            self._session_model_id = ai["model_id"]
            self._session_model_ids.add(ai["model_id"])
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

    def _cmd_tactical_hint(self, params: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            self._check_revision(params, required=True)
            if self.status != "playing":
                raise ServiceError("GAME_FINISHED", "The game has already ended.")
            board = self.board.copy()
            player = self.current_player
            revision = self._revision_counter
            session_id = self.session_id
        forced = find_forced_tactical_action(self.game, board, player, engine=self.engine, state=self._state)
        self._assert_fresh(session_id, revision)
        if forced is None:
            return None
        action, kind = forced
        layer, row, col = self.game.action_to_coords(action)
        return {
            "session_id": session_id,
            "for_revision": revision,
            "move": {"action": action, "layer": layer, "row": row, "col": col},
            "value": 1.0 if kind == "win" else 0.0,
            "kind": kind,
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

    def _cmd_replay_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return {"replays": self.replays.list_replays()}

    def _cmd_replay_save(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_revision(params, required=True)
            replay = self.replays.save_history(deepcopy(self.history), name=params.get("name"), rule_id=self.rule_id, participants=self._replay_participants())
            return {"replay": replay_summary(replay)}

    def _cmd_replay_open(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            replay = self._load_expected_replay(params)
            return {
                "replay": replay,
                "frames": build_replay_frames(replay, self.game),
                "analysis": self.replays.load_analysis(replay),
            }

    def _cmd_replay_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            replay = self._load_expected_replay(params)
            self._analysis_generations[replay["id"]] = self._analysis_generations.get(replay["id"], 0) + 1
            return {
                "deleted": self.replays.delete_replay(
                    replay["id"],
                    expected_fingerprint=replay["fingerprint"],
                )
            }

    def _cmd_replay_export(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            replay = self._load_expected_replay(params)
            return self.replays.export_replay(replay)

    def _cmd_replay_import(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if "content" in params:
                replay = self.replays.import_content(params["content"], filename=params.get("filename"))
            elif "path" in params:
                replay = self.replays.import_path(params["path"])
            else:
                raise ServiceError(
                    "INVALID_REPLAY_IMPORT",
                    "replay.import requires content (and optional filename) or a selected file path.",
                )
            return {"replay": replay_summary(replay)}

    def _cmd_replay_hint(self, params: dict[str, Any]) -> dict[str, Any]:
        step = self._as_int(params.get("step", 0), "step")
        with self._lock:
            replay = self._load_expected_replay(params)
            if not 0 <= step <= replay["turn_count"]:
                raise ServiceError("INVALID_REPLAY_STEP", f"Replay step must be between 0 and {replay['turn_count']}.")
            frame = build_replay_frames(replay, self.game)[step]
            if frame["status"] != "playing":
                raise ServiceError("GAME_FINISHED", "Cannot calculate a hint for a terminal replay position.")
            ai = dict(self.settings["roles"]["hint"])
            override = params.get("ai")
            if override is not None:
                if not isinstance(override, dict):
                    raise ServiceError("INVALID_AI_SETTINGS", "ai must be a JSON object.")
                ai.update(override)
            ai = self._validate_ai_config(ai)
            board = np.asarray(frame["board"], dtype=np.int8)
            player = int(frame["current_player"])
            engine = RuleEngine(replay["rule_id"], registry=BAL5_R2_RULE_REGISTRY)
            position = engine.state_from_board(board, player_to_move=player)
            if not np.any(engine.legal_column_mask(position)):
                raise ServiceError("FORCED_PASS_REQUIRED", "This replay position requires a forced pass; there is no move to suggest.")

        result = self._search(board, player, ai, rule_id=replay["rule_id"])
        with self._lock:
            current = self.replays.load_replay(replay["id"])
            if current["fingerprint"] != replay["fingerprint"]:
                raise ServiceError("STALE_REPLAY", "The replay changed while hint analysis was running.")
        layer, row, col = self.game.action_to_coords(result.action)
        return {
            "replay_id": replay["id"],
            "replay_fingerprint": replay["fingerprint"],
            "for_step": step,
            "for_revision": step,
            "move": {"action": result.action, "layer": layer, "row": row, "col": col},
            "value": result.value,
        }

    def _cmd_replay_analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            replay = self._load_expected_replay(params)
            ai = dict(self.settings["roles"]["win_rate"])
            override = params.get("ai")
            if override is not None:
                if not isinstance(override, dict):
                    raise ServiceError("INVALID_AI_SETTINGS", "ai must be a JSON object.")
                ai.update(override)
            ai = self._validate_ai_config(ai)
            spec = self.models.get(ai["model_id"])
            generation = self._analysis_generations.get(replay["id"], 0) + 1
            self._analysis_generations[replay["id"]] = generation
            request_generation = self.replays.begin_analysis(replay)

        started_at = _utc_now()
        started = time.monotonic()
        points: list[dict[str, Any]] = []
        frames = build_replay_frames(replay, self.game)
        for frame in frames:
            self._assert_latest_analysis(replay["id"], generation)
            if frame["status"] != "playing":
                red = 1.0 if frame["winner"] == 1 else 0.0 if frame["winner"] == -1 else 0.5
                estimate = "terminal"
            else:
                board = np.asarray(frame["board"], dtype=np.int8)
                player = int(frame["current_player"])
                engine = RuleEngine(replay["rule_id"], registry=BAL5_R2_RULE_REGISTRY)
                position = engine.state_from_board(board, player_to_move=player)
                if not np.any(engine.legal_column_mask(position)):
                    after_pass = engine.step(position, TurnAction.forced_pass())
                    if after_pass.terminal:
                        red = 1.0 if after_pass.outcome.winner == 1 else 0.0 if after_pass.outcome.winner == -1 else 0.5
                    elif np.any(engine.legal_column_mask(after_pass)):
                        result = self._search(np.asarray(after_pass.board), after_pass.player_to_move, ai, rule_id=replay["rule_id"], consecutive_passes=after_pass.consecutive_passes)
                        probability = (float(result.value) + 1.0) / 2.0
                        red = probability if after_pass.player_to_move == 1 else 1.0 - probability
                    else:
                        red = 0.5  # A second forced pass ends in a draw.
                    estimate = "forced_pass_successor"
                else:
                    result = self._search(board, player, ai, rule_id=replay["rule_id"], consecutive_passes=1 if frame["last_move"] and frame["last_move"].get("kind") == "forced_pass" else 0)
                    self._assert_latest_analysis(replay["id"], generation)
                    current_probability = (float(result.value) + 1.0) / 2.0
                    red = current_probability if player == 1 else 1.0 - current_probability
                    red = float(np.clip(red, 0.0, 1.0))
                    estimate = "model_mcts"
            points.append(
                {
                    "step": int(frame["move_count"]),
                    "red": red,
                    "blue": 1.0 - red,
                    "estimate": estimate,
                }
            )

        analysis = {
            "format": ANALYSIS_FORMAT,
            "protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "replay_id": replay["id"],
            "replay_fingerprint": replay["fingerprint"],
            "model": {
                "id": spec.id,
                "display_name": spec.display_name,
                "architecture": spec.architecture,
                "artifact_sha256": spec.artifact_sha256,
                "source_iteration": spec.source_iteration,
            },
            "config": dict(ai),
            "request_generation": request_generation,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "points": points,
        }
        with self._lock:
            self._assert_latest_analysis(replay["id"], generation)
            current = self.replays.load_replay(replay["id"])
            if current["fingerprint"] != replay["fingerprint"]:
                raise ServiceError("STALE_REPLAY", "The replay changed while win-rate analysis was running.")
            return self.replays.save_analysis(current, analysis)

    def _cmd_replay_continue(self, params: dict[str, Any]) -> dict[str, Any]:
        mode = str(params.get("mode", "pvp"))
        ai = self._combat_config_for_new_game(params)
        human_player = self._as_int(params.get("human_player", 1), "human_player")
        step = self._as_int(params.get("step", 0), "step")
        if mode not in {"pvp", "pvai"}:
            raise ServiceError("INVALID_MODE", f"Unsupported game mode: {mode}")
        if human_player not in {-1, 1}:
            raise ServiceError("INVALID_PLAYER", "human_player must be +1 or -1")
        with self._lock:
            replay = self._load_expected_replay(params)
            if not 0 <= step <= replay["turn_count"]:
                raise ServiceError("INVALID_REPLAY_STEP", f"Replay step must be between 0 and {replay['turn_count']}.")
            frame = build_replay_frames(replay, self.game)[step]
            if frame["status"] != "playing":
                raise ServiceError("GAME_FINISHED", "Cannot continue from a terminal replay position.")
            if mode == "pvai" and replay["rule_id"] not in self.models.get(ai["model_id"]).supported_rule_ids:
                raise ServiceError("MODEL_UNSUPPORTED_RULE", "The selected model does not support this replay rule.")
            self.rule_id = replay["rule_id"]
            self.engine = RuleEngine(self.rule_id, registry=BAL5_R2_RULE_REGISTRY)
            self._new_session_from_history(mode, human_player, replay["turns"][:step])
            self._session_model_id = ai["model_id"]
            self._session_model_ids = {ai["model_id"]} if mode == "pvai" else set()
            return self.snapshot()

    def _cmd_undo(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_revision(params, required=True)
            if len(self.history) <= self._history_floor:
                return self.snapshot()
            if self.mode == "pvp":
                while len(self.history) > self._history_floor and self.history[-1].get("kind") == "forced_pass":
                    self.history.pop()
                self.history.pop()
            else:
                undoable_history = self.history[self._history_floor :]
                if not any(move["player"] == self.human_player and move.get("kind") != "forced_pass" for move in undoable_history):
                    return self.snapshot()
                while len(self.history) > self._history_floor:
                    removed = self.history.pop()
                    if removed["player"] == self.human_player and removed.get("kind") != "forced_pass":
                        break
            self._rebuild_from_history()
            self._bump_revision()
            return self.snapshot()

    def _cmd_restart(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_revision(params, required=True)
            origin = deepcopy(self._session_origin_history)
            self._restore_history(origin)
            self._history_floor = len(self.history)
            self._bump_revision()
            return self.snapshot()

    def _new_session(self, mode: str, human_player: int) -> None:
        self.session_id = str(uuid.uuid4())
        self.mode = mode
        self.human_player = human_player
        self._session_model_id = self.settings["roles"]["combat"]["model_id"]
        self._session_model_ids = {self._session_model_id} if mode == "pvai" else set()
        self._session_origin_history = []
        self._history_floor = 0
        self._reset_position()
        self._bump_revision()

    def _new_session_from_history(
        self,
        mode: str,
        human_player: int,
        source_moves: list[dict[str, Any]],
    ) -> None:
        self.session_id = str(uuid.uuid4())
        self.mode = mode
        self.human_player = human_player
        self._session_model_id = self.settings["roles"]["combat"]["model_id"]
        self._session_model_ids = {self._session_model_id} if mode == "pvai" else set()
        self._session_origin_history = []
        self._history_floor = 0
        self._restore_history(source_moves)
        while self.status == "playing" and self.engine.required_action(self._state) is not None:
            self._apply_turn(TurnAction.forced_pass(), self.current_player, record_revision=False)
        self._session_origin_history = deepcopy(self.history)
        self._history_floor = len(self.history)
        self._bump_revision()

    def _restore_history(self, source_moves: list[dict[str, Any]]) -> None:
        self._reset_position()
        for source in source_moves:
            player = int(source["player"])
            if self.status != "playing":
                raise ServiceError("CORRUPT_HISTORY", "Replay prefix contains moves after the game ended.")
            if player != self.current_player:
                raise ServiceError("CORRUPT_HISTORY", "Replay prefix does not alternate players.")
            try:
                if source.get("kind") == "forced_pass":
                    self._apply_turn(TurnAction.forced_pass(), player, record_revision=False)
                else:
                    action = int(source["action"])
                    layer, row, col = self.game.action_to_coords(action)
                    column = row * 5 + col
                    if action != self.engine.legacy_action_for_column(self._state, column):
                        raise ValueError("History action does not match gravity or game rule.")
                    self._apply_turn(TurnAction.place(column), player, record_revision=False)
            except (ValueError, KeyError, TypeError) as exc:
                raise ServiceError("CORRUPT_HISTORY", str(exc)) from exc

    def _reset_position(self) -> None:
        self._state = self.engine.initial_state()
        self.board = np.array(self._state.board, copy=True)
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
            layer, row, col = self.game.action_to_coords(action)
            column = row * 5 + col
            if action != self.engine.legacy_action_for_column(self._state, column):
                raise ValueError("Action does not match the legal gravity landing cell.")
            self._apply_turn(TurnAction.place(column), player, record_revision=False)
            while self.status == "playing" and self.engine.required_action(self._state) is not None:
                self._apply_turn(TurnAction.forced_pass(), self.current_player, record_revision=False)
        except (ValueError, KeyError, TypeError) as exc:
            raise ServiceError("ILLEGAL_MOVE", str(exc)) from exc
        self._bump_revision()

    def _apply_turn(self, turn: TurnAction, player: int, *, record_revision: bool = True) -> None:
        action = self.engine.legacy_action_for_column(self._state, turn.column) if turn.column is not None else None
        self._state = self.engine.step(self._state, turn)
        self.board = np.array(self._state.board, copy=True)
        self.current_player = self._state.player_to_move
        if action is None:
            move = {"kind": "forced_pass", "player": player}
        else:
            layer, row, col = self.game.action_to_coords(action)
            move = {"kind": "place", "action": action, "layer": layer, "row": row, "col": col, "player": player}
        self.history.append(move)
        self.last_move = move
        self.status = "won" if self._state.outcome in {GameOutcome.FIRST_PLAYER_WIN, GameOutcome.SECOND_PLAYER_WIN} else "draw" if self._state.outcome == GameOutcome.DRAW else "playing"
        self.winner = self._state.outcome.winner if self.status == "won" else 0 if self.status == "draw" else None
        self.winning_line = find_winning_line(self.board, player, self.game.connect_n, rule_id=self.rule_id) if self.status == "won" else []
        if record_revision:
            self._bump_revision()

    def _replay_participants(self) -> list[dict[str, Any]]:
        result = []
        for seat, player in (("FIRST", 1), ("SECOND", -1)):
            is_ai = self.mode == "pvai" and player != self.human_player
            single_model = is_ai and len(self._session_model_ids) == 1
            model_id = next(iter(self._session_model_ids)) if single_model else None
            artifact_sha256 = self.models.get(model_id).artifact_sha256 if model_id else None
            result.append({"seat": seat, "player": player, "controller_type": "model" if single_model else "external" if is_ai else "human", "controller_id": model_id, "display_name": "AI" if single_model else "Mixed AI" if is_ai else "Player", "model_id": model_id, "lineage_hash": None, "artifact_sha256": artifact_sha256})
        return result

    def _combat_config_for_new_game(self, params: dict[str, Any]) -> dict[str, Any]:
        ai = dict(self.settings["roles"]["combat"])
        override = params.get("ai")
        if override is not None:
            if not isinstance(override, dict):
                raise ServiceError("INVALID_AI_SETTINGS", "ai must be a JSON object.")
            ai.update(override)
        return self._validate_ai_config(ai)

    def _rebuild_from_history(self) -> None:
        moves = list(self.history)
        self._restore_history(moves)

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
            ai["_rule_id"] = self.rule_id
            return self.board.copy(), self.current_player, self._revision_counter, self.session_id, ai

    def _search(
        self,
        board: np.ndarray,
        player: int,
        ai: dict[str, Any],
        *,
        forced_tactics: bool = True,
        rule_id: str | None = None,
        consecutive_passes: int = 0,
    ):
        selected_rule = rule_id or ai.get("_rule_id", self.rule_id)
        spec = self.models.get(ai["model_id"])
        if selected_rule not in spec.supported_rule_ids:
            raise ServiceError("MODEL_UNSUPPORTED_RULE", "The selected model does not support this game rule.")
        try:
            predictor = self.models.predictor(ai["model_id"])
        except (ModelUnavailableError, ModelRegistryError, OSError) as exc:
            raise ServiceError("MODEL_UNAVAILABLE", str(exc)) from exc
        try:
            engine = RuleEngine(selected_rule, registry=BAL5_R2_RULE_REGISTRY)
            state = engine.state_from_board(board, player_to_move=player, consecutive_passes=consecutive_passes)
            rule_features = np.asarray(RuleFeatureSchema.encode(engine.spec), dtype=np.float32) if ai["model_id"] == "cubesprite_v4_flash_preview1" else None
            return RuleMCTS(self.game, engine, predictor, simulations=ai["mcts_sims"], temperature=ai["temperature"], forced_tactics=forced_tactics, rule_features=rule_features).run(state)
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
        simulations = self._as_int(config.get("mcts_sims", 256), "mcts_sims")
        if simulations not in MCTS_OPTIONS:
            raise ServiceError("INVALID_MCTS", f"MCTS simulations must be one of {sorted(MCTS_OPTIONS)}")
        try:
            temperature = float(config.get("temperature", 0.4))
        except (TypeError, ValueError) as exc:
            raise ServiceError("INVALID_TEMPERATURE", "temperature must be a number from 0 to 2.") from exc
        if not np.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ServiceError("INVALID_TEMPERATURE", "temperature must be a number from 0 to 2.")
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

    def _assert_latest_analysis(self, replay_id: str, generation: int) -> None:
        with self._lock:
            if self._analysis_generations.get(replay_id) != generation:
                raise ServiceError(
                    "ANALYSIS_SUPERSEDED",
                    "A newer win-rate calculation replaced this replay analysis request.",
                )

    def _bump_revision(self) -> None:
        self._revision_counter += 1

    @staticmethod
    def _replay_id_param(params: dict[str, Any]) -> Any:
        if "id" in params:
            return params["id"]
        if "replay_id" in params:
            return params["replay_id"]
        raise ServiceError("INVALID_REPLAY_ID", "Replay id is required.")

    def _load_expected_replay(self, params: dict[str, Any]) -> dict[str, Any]:
        replay = self.replays.load_replay(self._replay_id_param(params))
        expected = params.get("expected_fingerprint")
        if not isinstance(expected, str) or not expected:
            raise ServiceError(
                "MISSING_REPLAY_FINGERPRINT",
                "expected_fingerprint is required for this replay operation.",
            )
        if expected != replay["fingerprint"]:
            raise ServiceError(
                "STALE_REPLAY",
                "The replay file changed after it was listed or opened.",
            )
        return replay

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
