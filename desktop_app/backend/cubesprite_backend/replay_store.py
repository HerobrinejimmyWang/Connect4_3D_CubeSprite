from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from connect4_core import GameRules

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - exercised by non-Windows development environments.
    import fcntl


REPLAY_FORMAT = "cubesprite.replay"
REPLAY_PROTOCOL_VERSION = 1
RULES_FORMAT = "connect4-3d-gravity"
RULES_VERSION = 1
ANALYSIS_FORMAT = "cubesprite.win-rate-analysis"
ANALYSIS_PROTOCOL_VERSION = 1
ANALYSIS_GENERATION_FORMAT = "cubesprite.analysis-generation"
ANALYSIS_MCTS_OPTIONS = {32, 64, 128, 256, 512, 1024}
MAX_REPLAY_BYTES = 512 * 1024
MAX_REPLAY_NAME_LENGTH = 120
REPLAY_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReplayStoreError(RuntimeError):
    def __init__(self, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


class ReplayStore:
    """Validated, atomic local storage for replay and analysis documents."""

    def __init__(self, data_dir: Path, game: GameRules):
        self.data_dir = Path(data_dir).resolve()
        self.replay_dir = self.data_dir / "replays"
        self.analysis_dir = self.data_dir / "replay_analysis"
        self.lock_path = self.data_dir / ".replay-store.lock"
        self.game = game
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def list_replays(self) -> list[dict[str, Any]]:
        replays: list[tuple[float, dict[str, Any]]] = []
        for path in self.replay_dir.glob("*.c4replay.json"):
            try:
                replay = self._read_replay(path)
                modified = path.stat().st_mtime
            except (OSError, ReplayStoreError):
                continue
            replays.append((modified, replay_summary(replay)))
        replays.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        return [summary for _, summary in replays]

    def save_history(self, history: list[dict[str, Any]], name: Any = None) -> dict[str, Any]:
        replay_id = uuid.uuid4().hex
        replay_name = _normalize_name(name, fallback=_default_replay_name(len(history)))
        saved_at = _utc_now()
        moves = [_normalize_internal_move(move, index, self.game) for index, move in enumerate(history)]
        replay = self._build_document(replay_id, replay_name, saved_at, moves)
        path = self._replay_path(replay_id)
        with self._exclusive_store_lock():
            self._write_json_atomic(path, replay)
        return replay

    def import_content(self, content: Any, filename: Any = None) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ReplayStoreError("INVALID_REPLAY_IMPORT", "Replay content must be a UTF-8 JSON string.")
        try:
            encoded = content.encode("utf-8")
        except UnicodeError as exc:
            raise ReplayStoreError("INVALID_REPLAY_IMPORT", "Replay content is not valid UTF-8 text.") from exc
        if len(encoded) > MAX_REPLAY_BYTES:
            raise ReplayStoreError(
                "REPLAY_TOO_LARGE",
                f"Replay file exceeds the {MAX_REPLAY_BYTES}-byte import limit.",
                {"filename": str(filename or "")},
            )
        replay = self._parse_replay(content, source=str(filename or "import"))
        return self._store_imported(replay)

    def import_path(self, source: Any) -> dict[str, Any]:
        if not isinstance(source, str) or not source.strip():
            raise ReplayStoreError("INVALID_REPLAY_IMPORT", "Replay import path must be a non-empty string.")
        try:
            path = Path(source).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ReplayStoreError("REPLAY_IMPORT_FAILED", f"Cannot resolve replay file: {exc}") from exc
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ReplayStoreError("REPLAY_IMPORT_FAILED", f"Cannot read replay file: {exc}") from exc
        if not path.is_file():
            raise ReplayStoreError("REPLAY_IMPORT_FAILED", "Replay import path is not a file.")
        if size > MAX_REPLAY_BYTES:
            raise ReplayStoreError("REPLAY_TOO_LARGE", f"Replay file exceeds the {MAX_REPLAY_BYTES}-byte import limit.")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ReplayStoreError("REPLAY_IMPORT_FAILED", f"Cannot read replay file: {exc}") from exc
        replay = self._parse_replay(content, source=str(path))
        return self._store_imported(replay)

    def load_replay(self, replay_id: Any) -> dict[str, Any]:
        return self._read_replay(self._replay_path(_normalize_replay_id(replay_id)))

    def open_replay(self, replay_id: Any) -> dict[str, Any]:
        replay = self.load_replay(replay_id)
        return {
            "replay": replay,
            "frames": build_replay_frames(replay, self.game),
            "analysis": self.load_analysis(replay),
        }

    def export_replay(self, replay: dict[str, Any]) -> dict[str, str]:
        """Return a validated, portable protocol document for a save dialog."""

        validated = validate_replay(replay, self.game)
        return {
            "filename": f"CubeSprite-{validated['id'][:12]}.c4replay.json",
            "content": json.dumps(
                validated,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        }

    def delete_replay(self, replay_id: Any, expected_fingerprint: str | None = None) -> bool:
        normalized_id = _normalize_replay_id(replay_id)
        replay_path = self._replay_path(normalized_id)
        analysis_path = self._analysis_path(normalized_id)
        generation_path = self._analysis_generation_path(normalized_id)
        with self._exclusive_store_lock():
            replay = self._read_replay(replay_path)
            if expected_fingerprint is not None and replay["fingerprint"] != expected_fingerprint:
                raise ReplayStoreError("STALE_REPLAY", "The replay changed before it could be deleted.")
            try:
                # The analysis is disposable and can be regenerated. Delete it
                # first so a failure never removes the authoritative replay
                # while reporting the whole operation as unsuccessful.
                if analysis_path.exists():
                    analysis_path.unlink()
                if generation_path.exists():
                    generation_path.unlink()
                replay_path.unlink()
            except OSError as exc:
                raise ReplayStoreError("REPLAY_DELETE_FAILED", f"Cannot delete replay: {exc}") from exc
        return True

    def load_analysis(self, replay: dict[str, Any]) -> dict[str, Any] | None:
        path = self._analysis_path(replay["id"])
        if not path.is_file():
            return None
        try:
            payload = _strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))
            return validate_analysis(payload, replay)
        except (OSError, UnicodeError, ReplayStoreError):
            return None

    def begin_analysis(self, replay: dict[str, Any]) -> int:
        """Reserve a cross-process generation for a new calculation."""

        with self._exclusive_store_lock():
            current = self._read_replay(self._replay_path(replay["id"]))
            if current["fingerprint"] != replay["fingerprint"]:
                raise ReplayStoreError("STALE_REPLAY", "The replay changed before analysis could start.")
            path = self._analysis_generation_path(replay["id"])
            previous = 0
            if path.is_file():
                try:
                    payload = _strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))
                except (OSError, UnicodeError) as exc:
                    raise ReplayStoreError(
                        "ANALYSIS_GENERATION_INVALID",
                        f"Cannot read the analysis generation marker: {exc}",
                    ) from exc
                expected_keys = {"format", "replay_id", "replay_fingerprint", "generation"}
                if not isinstance(payload, dict) or set(payload) != expected_keys:
                    raise ReplayStoreError(
                        "ANALYSIS_GENERATION_INVALID",
                        "The analysis generation marker is invalid.",
                    )
                if (
                    payload["format"] != ANALYSIS_GENERATION_FORMAT
                    or payload["replay_id"] != replay["id"]
                    or payload["replay_fingerprint"] != replay["fingerprint"]
                ):
                    raise ReplayStoreError(
                        "ANALYSIS_GENERATION_INVALID",
                        "The analysis generation marker does not match this replay.",
                    )
                previous = _strict_int(payload["generation"], "generation")
                if previous <= 0:
                    raise ReplayStoreError(
                        "ANALYSIS_GENERATION_INVALID",
                        "The analysis generation marker must be positive.",
                    )
            generation = previous + 1
            self._write_json_atomic(
                path,
                {
                    "format": ANALYSIS_GENERATION_FORMAT,
                    "replay_id": replay["id"],
                    "replay_fingerprint": replay["fingerprint"],
                    "generation": generation,
                },
            )
            return generation

    def save_analysis(self, replay: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
        validated = validate_analysis(analysis, replay)
        with self._exclusive_store_lock():
            current = self._read_replay(self._replay_path(replay["id"]))
            if current["fingerprint"] != replay["fingerprint"]:
                raise ReplayStoreError("STALE_REPLAY", "The replay changed before its analysis could be saved.")
            generation_path = self._analysis_generation_path(replay["id"])
            try:
                marker = _strict_json_loads(
                    generation_path.read_text(encoding="utf-8"),
                    source=str(generation_path),
                )
            except (OSError, UnicodeError) as exc:
                raise ReplayStoreError(
                    "ANALYSIS_SUPERSEDED",
                    "The analysis request is no longer current.",
                ) from exc
            if (
                not isinstance(marker, dict)
                or marker.get("format") != ANALYSIS_GENERATION_FORMAT
                or marker.get("replay_id") != replay["id"]
                or marker.get("replay_fingerprint") != replay["fingerprint"]
                or marker.get("generation") != validated["request_generation"]
            ):
                raise ReplayStoreError(
                    "ANALYSIS_SUPERSEDED",
                    "A newer win-rate calculation replaced this analysis request.",
                )
            self._write_json_atomic(self._analysis_path(replay["id"]), validated)
        return validated

    def _store_imported(self, replay: dict[str, Any]) -> dict[str, Any]:
        path = self._replay_path(replay["id"])
        with self._exclusive_store_lock():
            if path.exists():
                existing = self._read_replay(path)
                if existing["fingerprint"] == replay["fingerprint"]:
                    # The fingerprint deliberately identifies the rules and
                    # move sequence, not mutable display metadata. Return the
                    # document that is actually persisted so the caller never
                    # shows an imported name/timestamp that disappears on the
                    # next refresh.
                    return existing
                raise ReplayStoreError(
                    "REPLAY_ID_CONFLICT",
                    f"Replay id {replay['id']} already exists with different moves.",
                )
            self._write_json_atomic(path, replay)
            return replay

    def _read_replay(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ReplayStoreError("REPLAY_NOT_FOUND", f"Replay {path.name} does not exist.")
        try:
            if path.stat().st_size > MAX_REPLAY_BYTES:
                raise ReplayStoreError("REPLAY_TOO_LARGE", "Stored replay exceeds the supported size limit.")
            content = path.read_text(encoding="utf-8")
        except ReplayStoreError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ReplayStoreError("REPLAY_READ_FAILED", f"Cannot read replay file: {exc}") from exc
        return self._parse_replay(content, source=str(path))

    def _parse_replay(self, content: str, source: str) -> dict[str, Any]:
        payload = _strict_json_loads(content, source=source)
        return validate_replay(payload, self.game)

    def _build_document(
        self,
        replay_id: str,
        name: str,
        saved_at: str,
        moves: list[dict[str, int]],
    ) -> dict[str, Any]:
        replay = {
            "format": REPLAY_FORMAT,
            "protocol_version": REPLAY_PROTOCOL_VERSION,
            "id": replay_id,
            "name": name,
            "saved_at": saved_at,
            "rules": _rules_document(self.game),
            "moves": moves,
        }
        state = _replay_terminal_state(moves, self.game)
        replay.update(state)
        replay["fingerprint"] = replay_fingerprint(replay)
        return validate_replay(replay, self.game)

    def _replay_path(self, replay_id: str) -> Path:
        return self.replay_dir / f"{replay_id}.c4replay.json"

    def _analysis_path(self, replay_id: str) -> Path:
        return self.analysis_dir / f"{replay_id}.winrate.json"

    def _analysis_generation_path(self, replay_id: str) -> Path:
        return self.analysis_dir / f".{replay_id}.generation.json"

    @contextmanager
    def _exclusive_store_lock(self):
        """Serialize mutations across threads and separate app instances."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except (OSError, ValueError) as exc:
            raise ReplayStoreError("REPLAY_WRITE_FAILED", f"Cannot write replay data: {exc}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def validate_replay(payload: Any, game: GameRules) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ReplayStoreError("INVALID_REPLAY", "Replay document must be a JSON object.")
    required = {
        "format", "protocol_version", "id", "name", "saved_at", "rules",
        "moves", "move_count", "status", "winner", "fingerprint",
    }
    _require_exact_keys(payload, required, "replay")
    if payload["format"] != REPLAY_FORMAT or _strict_int(payload["protocol_version"], "protocol_version") != 1:
        raise ReplayStoreError("UNSUPPORTED_REPLAY_PROTOCOL", "Only CubeSprite replay protocol v1 is supported.")
    replay_id = _normalize_replay_id(payload["id"])
    name = _normalize_name(payload["name"])
    saved_at = _normalize_timestamp(payload["saved_at"], "saved_at")
    rules = _validate_rules(payload["rules"], game)
    if not isinstance(payload["moves"], list) or len(payload["moves"]) > game.get_action_size():
        raise ReplayStoreError("INVALID_REPLAY", f"moves must contain at most {game.get_action_size()} entries.")

    board = game.get_init_board()
    current_player = 1
    status = "playing"
    winner: int | None = None
    normalized_moves: list[dict[str, int]] = []
    for index, raw_move in enumerate(payload["moves"]):
        if status != "playing":
            raise ReplayStoreError("INVALID_REPLAY", "Replay contains moves after the game ended.")
        move = _normalize_protocol_move(raw_move, index, game)
        if move["player"] != current_player:
            raise ReplayStoreError("INVALID_REPLAY", f"Move {index + 1} does not alternate players.")
        try:
            board, next_player = game.get_next_state(board, current_player, move["action"])
        except (TypeError, ValueError) as exc:
            raise ReplayStoreError("INVALID_REPLAY_MOVE", f"Move {index + 1} is illegal: {exc}") from exc
        normalized_moves.append(move)
        line = find_winning_line(board, current_player, game.connect_n)
        if line:
            status, winner = "won", current_player
        elif not np.any(game.get_valid_moves(board)):
            status, winner = "draw", 0
        else:
            current_player = int(next_player)

    move_count = _strict_int(payload["move_count"], "move_count")
    declared_winner = payload["winner"]
    if isinstance(declared_winner, bool) or declared_winner not in (None, -1, 0, 1):
        raise ReplayStoreError("INVALID_REPLAY", "winner must be null, -1, 0, or +1.")
    if move_count != len(normalized_moves) or payload["status"] != status or declared_winner != winner:
        raise ReplayStoreError("INVALID_REPLAY", "Replay summary does not match the validated move sequence.")

    normalized = {
        "format": REPLAY_FORMAT,
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "id": replay_id,
        "name": name,
        "saved_at": saved_at,
        "rules": rules,
        "moves": normalized_moves,
        "move_count": len(normalized_moves),
        "status": status,
        "winner": winner,
    }
    fingerprint = replay_fingerprint(normalized)
    if payload["fingerprint"] != fingerprint:
        raise ReplayStoreError("REPLAY_FINGERPRINT_MISMATCH", "Replay fingerprint does not match its move sequence.")
    normalized["fingerprint"] = fingerprint
    return normalized


def validate_analysis(payload: Any, replay: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ReplayStoreError("INVALID_ANALYSIS", "Win-rate analysis must be a JSON object.")
    required = {
        "format", "protocol_version", "replay_id", "replay_fingerprint", "model",
        "config", "request_generation", "started_at", "completed_at", "duration_ms", "points",
    }
    _require_exact_keys(payload, required, "analysis")
    if payload["format"] != ANALYSIS_FORMAT or _strict_int(payload["protocol_version"], "protocol_version") != 1:
        raise ReplayStoreError("UNSUPPORTED_ANALYSIS_PROTOCOL", "Only win-rate analysis protocol v1 is supported.")
    if _normalize_replay_id(payload["replay_id"]) != replay["id"] or payload["replay_fingerprint"] != replay["fingerprint"]:
        raise ReplayStoreError("ANALYSIS_REPLAY_MISMATCH", "Win-rate analysis does not match this replay.")
    model = payload["model"]
    config = payload["config"]
    model_fields = {
        "id",
        "display_name",
        "architecture",
        "artifact_sha256",
        "source_iteration",
    }
    if not isinstance(model, dict) or set(model) != model_fields:
        raise ReplayStoreError("INVALID_ANALYSIS", "analysis.model is invalid.")
    string_model_fields = {"id", "display_name", "architecture"}
    if not all(isinstance(model[key], str) and model[key] for key in string_model_fields):
        raise ReplayStoreError("INVALID_ANALYSIS", "analysis.model values must be non-empty strings.")
    artifact_sha256 = model["artifact_sha256"]
    if not isinstance(artifact_sha256, str) or not SHA256_PATTERN.fullmatch(artifact_sha256):
        raise ReplayStoreError(
            "INVALID_ANALYSIS",
            "analysis.model.artifact_sha256 must be a lowercase SHA-256 digest.",
        )
    source_iteration = model["source_iteration"]
    if source_iteration is not None:
        source_iteration = _strict_int(source_iteration, "source_iteration")
        if source_iteration <= 0:
            raise ReplayStoreError(
                "INVALID_ANALYSIS",
                "analysis.model.source_iteration must be a positive integer or null.",
            )
    if not isinstance(config, dict) or set(config) != {"model_id", "mcts_sims", "temperature"}:
        raise ReplayStoreError("INVALID_ANALYSIS", "analysis.config is invalid.")
    if config["model_id"] != model["id"]:
        raise ReplayStoreError("INVALID_ANALYSIS", "analysis config and model id do not match.")
    request_generation = _strict_int(payload["request_generation"], "request_generation")
    if request_generation <= 0:
        raise ReplayStoreError("INVALID_ANALYSIS", "analysis.request_generation must be positive.")
    mcts_sims = _strict_int(config["mcts_sims"], "mcts_sims")
    temperature = _strict_float(config["temperature"], "temperature")
    if mcts_sims not in ANALYSIS_MCTS_OPTIONS:
        raise ReplayStoreError(
            "INVALID_ANALYSIS",
            f"analysis.config.mcts_sims must be one of {sorted(ANALYSIS_MCTS_OPTIONS)}.",
        )
    if not 0.0 <= temperature <= 5.0:
        raise ReplayStoreError(
            "INVALID_ANALYSIS",
            "analysis.config.temperature must be between 0 and 5.",
        )
    started_at = _normalize_timestamp(payload["started_at"], "started_at")
    completed_at = _normalize_timestamp(payload["completed_at"], "completed_at")
    if _parse_timestamp(completed_at, "completed_at") < _parse_timestamp(started_at, "started_at"):
        raise ReplayStoreError(
            "INVALID_ANALYSIS",
            "analysis.completed_at must not precede analysis.started_at.",
        )
    duration_ms = _strict_int(payload["duration_ms"], "duration_ms")
    if duration_ms < 0:
        raise ReplayStoreError("INVALID_ANALYSIS", "duration_ms must not be negative.")
    points = payload["points"]
    if not isinstance(points, list) or len(points) != replay["move_count"] + 1:
        raise ReplayStoreError("INVALID_ANALYSIS", "Analysis must contain one point for every replay step, including step 0.")
    normalized_points = []
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != {"step", "red", "blue", "estimate"}:
            raise ReplayStoreError("INVALID_ANALYSIS", f"Analysis point {index} is invalid.")
        step = _strict_int(point["step"], "step")
        red = _strict_float(point["red"], "red")
        blue = _strict_float(point["blue"], "blue")
        if step != index or not 0.0 <= red <= 1.0 or not 0.0 <= blue <= 1.0:
            raise ReplayStoreError("INVALID_ANALYSIS", f"Analysis point {index} is outside its valid range.")
        if not np.isclose(red + blue, 1.0, rtol=0.0, atol=1e-6):
            raise ReplayStoreError("INVALID_ANALYSIS", f"Analysis point {index} probabilities do not sum to one.")
        if not isinstance(point["estimate"], str) or not point["estimate"]:
            raise ReplayStoreError("INVALID_ANALYSIS", f"Analysis point {index} estimate is invalid.")
        normalized_points.append({"step": step, "red": red, "blue": blue, "estimate": point["estimate"]})
    return {
        "format": ANALYSIS_FORMAT,
        "protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "replay_id": replay["id"],
        "replay_fingerprint": replay["fingerprint"],
        "model": {
            "id": model["id"],
            "display_name": model["display_name"],
            "architecture": model["architecture"],
            "artifact_sha256": artifact_sha256,
            "source_iteration": source_iteration,
        },
        "config": {"model_id": model["id"], "mcts_sims": mcts_sims, "temperature": temperature},
        "request_generation": request_generation,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "points": normalized_points,
    }


def replay_summary(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": replay["id"],
        "name": replay["name"],
        "saved_at": replay["saved_at"],
        "move_count": replay["move_count"],
        "status": replay["status"],
        "winner": replay["winner"],
        "fingerprint": replay["fingerprint"],
    }


def replay_fingerprint(replay: dict[str, Any]) -> str:
    canonical = {
        "format": REPLAY_FORMAT,
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "rules": replay["rules"],
        "moves": replay["moves"],
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_replay_frames(replay: dict[str, Any], game: GameRules) -> list[dict[str, Any]]:
    board = game.get_init_board()
    current_player = 1
    status = "playing"
    winner: int | None = None
    last_move: dict[str, int] | None = None
    winning_line: list[dict[str, int]] = []
    frames = [
        _frame_snapshot(replay, 0, board, current_player, status, winner, last_move, winning_line, game)
    ]
    for step, move in enumerate(replay["moves"], start=1):
        acting_player = move["player"]
        board, next_player = game.get_next_state(board, acting_player, move["action"])
        last_move = deepcopy(move)
        winning_line = find_winning_line(board, acting_player, game.connect_n)
        if winning_line:
            status, winner, current_player = "won", acting_player, acting_player
        elif not np.any(game.get_valid_moves(board)):
            status, winner, current_player = "draw", 0, acting_player
        else:
            status, winner, current_player = "playing", None, int(next_player)
        frames.append(
            _frame_snapshot(
                replay, step, board, current_player, status, winner, last_move, winning_line, game
            )
        )
    return frames


def find_winning_line(board: np.ndarray, player: int, connect_n: int = 4) -> list[dict[str, int]]:
    board = np.asarray(board, dtype=np.int8)
    directions = [
        (dz, dy, dx)
        for dz in (0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
        and not (dz == 0 and dy < 0)
        and not (dz == 0 and dy == 0 and dx < 0)
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


def _frame_snapshot(
    replay: dict[str, Any],
    step: int,
    board: np.ndarray,
    current_player: int,
    status: str,
    winner: int | None,
    last_move: dict[str, int] | None,
    winning_line: list[dict[str, int]],
    game: GameRules,
) -> dict[str, Any]:
    legal_moves = []
    if status == "playing":
        for action in np.flatnonzero(game.get_valid_moves(board) > 0):
            layer, row, col = game.action_to_coords(int(action))
            legal_moves.append({"action": int(action), "layer": layer, "row": row, "col": col})
    return {
        "session_id": f"replay:{replay['id']}",
        "revision": step,
        "mode": "replay",
        "human_player": 1,
        "board": np.asarray(board, dtype=np.int8).astype(int).tolist(),
        "current_player": int(current_player),
        "move_count": step,
        "status": status,
        "winner": winner,
        "last_move": deepcopy(last_move),
        "winning_line": deepcopy(winning_line),
        "legal_moves": legal_moves,
        "can_undo": False,
        "replay_id": replay["id"],
        "replay_step": step,
        "replay_total_steps": replay["move_count"],
    }


def _replay_terminal_state(moves: list[dict[str, int]], game: GameRules) -> dict[str, Any]:
    board = game.get_init_board()
    current_player = 1
    status = "playing"
    winner: int | None = None
    for index, move in enumerate(moves):
        if move["player"] != current_player:
            raise ReplayStoreError("INVALID_REPLAY", f"Move {index + 1} does not alternate players.")
        if status != "playing":
            raise ReplayStoreError("INVALID_REPLAY", "Replay contains moves after the game ended.")
        try:
            board, next_player = game.get_next_state(board, current_player, move["action"])
        except (TypeError, ValueError) as exc:
            raise ReplayStoreError("INVALID_REPLAY_MOVE", f"Move {index + 1} is illegal: {exc}") from exc
        if find_winning_line(board, current_player, game.connect_n):
            status, winner = "won", current_player
        elif not np.any(game.get_valid_moves(board)):
            status, winner = "draw", 0
        else:
            current_player = int(next_player)
    return {"move_count": len(moves), "status": status, "winner": winner}


def _normalize_internal_move(move: Any, index: int, game: GameRules) -> dict[str, int]:
    if not isinstance(move, dict):
        raise ReplayStoreError("INVALID_REPLAY", f"Move {index + 1} must be an object.")
    action = _strict_int(move.get("action"), "action")
    player = _strict_int(move.get("player"), "player")
    if player not in (-1, 1):
        raise ReplayStoreError("INVALID_REPLAY", f"Move {index + 1} player must be -1 or +1.")
    try:
        layer, row, col = game.action_to_coords(action)
    except ValueError as exc:
        raise ReplayStoreError("INVALID_REPLAY_MOVE", f"Move {index + 1} is invalid: {exc}") from exc
    return {"ply": index + 1, "action": action, "layer": layer, "row": row, "col": col, "player": player}


def _normalize_protocol_move(move: Any, index: int, game: GameRules) -> dict[str, int]:
    if not isinstance(move, dict):
        raise ReplayStoreError("INVALID_REPLAY", f"Move {index + 1} must be an object.")
    _require_exact_keys(move, {"ply", "action", "layer", "row", "col", "player"}, f"move {index + 1}")
    normalized = _normalize_internal_move(move, index, game)
    if _strict_int(move["ply"], "ply") != index + 1:
        raise ReplayStoreError("INVALID_REPLAY", f"Move {index + 1} has an invalid ply number.")
    for coordinate in ("layer", "row", "col"):
        if _strict_int(move[coordinate], coordinate) != normalized[coordinate]:
            raise ReplayStoreError("INVALID_REPLAY", f"Move {index + 1} coordinates do not match its action.")
    return normalized


def _rules_document(game: GameRules) -> dict[str, Any]:
    return {
        "format": RULES_FORMAT,
        "version": RULES_VERSION,
        "board_layers": game.max_layers,
        "board_size": game.board_size,
        "connect_n": game.connect_n,
        "gravity": "layer_ascending",
        "starting_player": 1,
    }


def _validate_rules(payload: Any, game: GameRules) -> dict[str, Any]:
    expected = _rules_document(game)
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ReplayStoreError("UNSUPPORTED_REPLAY_RULES", "Replay rules descriptor is invalid.")
    normalized = dict(payload)
    for key in ("version", "board_layers", "board_size", "connect_n", "starting_player"):
        normalized[key] = _strict_int(payload[key], key)
    if normalized != expected:
        raise ReplayStoreError("UNSUPPORTED_REPLAY_RULES", "Replay uses an unsupported rules version or board shape.")
    return expected


def _strict_json_loads(content: str, source: str) -> Any:
    def reject_constant(value: str):
        raise ValueError(f"Non-finite JSON value {value} is not permitted.")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} is not permitted.")
            result[key] = value
        return result

    try:
        return json.loads(content, parse_constant=reject_constant, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReplayStoreError("INVALID_REPLAY_JSON", f"Cannot parse {source}: {exc}") from exc


def _normalize_replay_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ReplayStoreError("INVALID_REPLAY_ID", "Replay id must be a UUID string.")
    compact = value.strip().lower().replace("-", "")
    if not REPLAY_ID_PATTERN.fullmatch(compact):
        raise ReplayStoreError("INVALID_REPLAY_ID", "Replay id must be a UUID string.")
    return compact


def _normalize_name(value: Any, fallback: str | None = None) -> str:
    if value is None and fallback is not None:
        value = fallback
    if not isinstance(value, str):
        raise ReplayStoreError("INVALID_REPLAY_NAME", "Replay name must be a string.")
    name = value.strip()
    if not name or len(name) > MAX_REPLAY_NAME_LENGTH or any(ord(char) < 32 for char in name):
        raise ReplayStoreError(
            "INVALID_REPLAY_NAME",
            f"Replay name must contain 1 to {MAX_REPLAY_NAME_LENGTH} printable characters.",
        )
    return name


def _normalize_timestamp(value: Any, name: str) -> str:
    _parse_timestamp(value, name)
    return value


def _parse_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ReplayStoreError("INVALID_REPLAY", f"{name} must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReplayStoreError("INVALID_REPLAY", f"{name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ReplayStoreError("INVALID_REPLAY", f"{name} must include a timezone.")
    return parsed


def _require_exact_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ReplayStoreError(
            "INVALID_REPLAY",
            f"{label} fields are invalid.",
            {"missing": missing, "unknown": unknown},
        )


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayStoreError("INVALID_REPLAY", f"{name} must be an integer.")
    return int(value)


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayStoreError("INVALID_ANALYSIS", f"{name} must be a finite number.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ReplayStoreError("INVALID_ANALYSIS", f"{name} must be a finite number.")
    return parsed


def _default_replay_name(move_count: int) -> str:
    return f"Replay {_utc_now().replace(':', '-')} ({move_count} moves)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _inside(coords: tuple[int, int, int], shape: tuple[int, ...]) -> bool:
    return all(0 <= coordinate < dimension for coordinate, dimension in zip(coords, shape))
