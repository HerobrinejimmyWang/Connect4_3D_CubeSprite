"""CubeSprite replay export helpers for V3 self-play audit games.

The portable replay document intentionally contains no training metadata.  The
desktop application accepts an exact protocol-v1 schema, so provenance belongs
in the separate audit index built by :func:`build_audit_index`.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from connect4_core import GameRules

from .selfplay import GameRecord


REPLAY_FORMAT = "cubesprite.replay"
REPLAY_PROTOCOL_VERSION = 1
RULES_FORMAT = "connect4-3d-gravity"
RULES_VERSION = 1
MAX_REPLAY_BYTES = 512 * 1024
AUDIT_INDEX_FORMAT = "connect4-v3-audit-index"
AUDIT_INDEX_VERSION = 1

REPLAY_KEYS = frozenset(
    {
        "format",
        "protocol_version",
        "id",
        "name",
        "saved_at",
        "rules",
        "moves",
        "move_count",
        "status",
        "winner",
        "fingerprint",
    }
)
MOVE_KEYS = frozenset({"ply", "action", "layer", "row", "col", "player"})


@dataclass(frozen=True)
class RepresentativeGame:
    """One deterministically selected game and the reasons it was retained."""

    game: GameRecord
    reasons: tuple[str, ...]

    @property
    def game_id(self) -> int:
        return int(self.game.game_id)


def _rules_document(game: GameRules) -> dict[str, Any]:
    return {
        "format": RULES_FORMAT,
        "version": RULES_VERSION,
        "board_layers": int(game.max_layers),
        "board_size": int(game.board_size),
        "connect_n": int(game.connect_n),
        "gravity": "layer_ascending",
        "starting_player": 1,
    }


def _timestamp(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("saved_at must be an ISO-8601 timestamp") from exc
    else:
        raise TypeError("saved_at must be a datetime, ISO-8601 string, or None")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("saved_at must include a timezone")
    normalized = parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
    return normalized.replace("+00:00", "Z")


def _deterministic_replay_id(run_id: str, game: GameRecord) -> str:
    identity = {
        "run_id": run_id,
        "generation": int(game.generation),
        "game_id": int(game.game_id),
        "producer_model_id": str(game.producer_model_id),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def replay_fingerprint(document: Mapping[str, Any]) -> str:
    """Return the exact fingerprint used by CubeSprite replay protocol v1."""

    canonical = {
        "format": REPLAY_FORMAT,
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "rules": document["rules"],
        "moves": document["moves"],
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("replay name must be a string")
    normalized = name.strip()
    if not normalized or len(normalized) > 120 or any(ord(character) < 32 for character in normalized):
        raise ValueError("replay name must contain 1 to 120 printable characters")
    return normalized


def game_record_to_replay(
    game_record: GameRecord,
    *,
    run_id: str,
    saved_at: datetime | str | None = None,
    name: str | None = None,
    replay_id: str | None = None,
) -> dict[str, Any]:
    """Convert a completed V3 game to the strict CubeSprite protocol-v1 schema."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if int(game_record.game_id) < 0 or int(game_record.generation) < 0 or int(game_record.seed) < 0:
        raise ValueError("game_id, generation, and seed must be non-negative")
    producer_model_id = str(game_record.producer_model_id)
    if not producer_model_id:
        raise ValueError("producer_model_id must be non-empty")
    if int(game_record.winner) not in (-1, 0, 1):
        raise ValueError("game winner must be -1, 0, or 1")
    if bool(game_record.is_draw) != (int(game_record.winner) == 0):
        raise ValueError("game is_draw and winner fields disagree")

    game = GameRules()
    board = game.get_init_board()
    current_player = 1
    status = "playing"
    validated_winner: int | None = None
    moves: list[dict[str, int]] = []

    for index, record in enumerate(game_record.moves):
        if status != "playing":
            raise ValueError("game record contains moves after the terminal position")
        if int(record.ply) != index:
            raise ValueError("V3 move ply must be contiguous and zero-based")
        player = int(record.player)
        action = int(record.legacy_action)
        if player != current_player:
            raise ValueError("game record does not alternate players")
        layer, row, col = game.action_to_coords(action)
        if int(record.column) != row * int(game.board_size) + col:
            raise ValueError("V3 column and legacy action coordinates disagree")
        try:
            board, next_player = game.get_next_state(board, player, action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"illegal V3 replay move at ply {index + 1}: {exc}") from exc
        moves.append(
            {
                "ply": index + 1,
                "action": action,
                "layer": int(layer),
                "row": int(row),
                "col": int(col),
                "player": player,
            }
        )
        if game.check_win(board, player):
            status, validated_winner = "won", player
        elif not np.any(game.get_valid_moves(board)):
            status, validated_winner = "draw", 0
        else:
            current_player = int(next_player)

    if status == "playing":
        raise ValueError("only completed V3 games can be exported for audit")
    if validated_winner != int(game_record.winner):
        raise ValueError("game record winner does not match its move sequence")

    normalized_id = replay_id or _deterministic_replay_id(run_id.strip(), game_record)
    if (
        not isinstance(normalized_id, str)
        or len(normalized_id) != 32
        or any(character not in "0123456789abcdef" for character in normalized_id)
    ):
        raise ValueError("replay_id must contain exactly 32 lowercase hexadecimal characters")
    display_name = name or (
        f"V3 g{int(game_record.generation):06d} game {int(game_record.game_id):08d} "
        f"({producer_model_id[:72]})"
    )
    document: dict[str, Any] = {
        "format": REPLAY_FORMAT,
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "id": normalized_id,
        "name": _normalize_name(display_name),
        "saved_at": _timestamp(saved_at),
        "rules": _rules_document(game),
        "moves": moves,
        "move_count": len(moves),
        "status": status,
        "winner": validated_winner,
    }
    document["fingerprint"] = replay_fingerprint(document)
    return document


def validate_replay_document(document: Mapping[str, Any]) -> None:
    """Validate export-only invariants before a document is written."""

    if not isinstance(document, Mapping) or set(document) != REPLAY_KEYS:
        raise ValueError("replay document fields do not match protocol v1")
    if document["format"] != REPLAY_FORMAT or document["protocol_version"] != 1:
        raise ValueError("unsupported replay protocol")
    _timestamp(document["saved_at"])
    _normalize_name(document["name"])
    moves = document["moves"]
    if not isinstance(moves, list) or len(moves) != int(document["move_count"]):
        raise ValueError("replay move_count does not match moves")
    for index, move in enumerate(moves):
        if not isinstance(move, Mapping) or set(move) != MOVE_KEYS:
            raise ValueError(f"move {index + 1} fields do not match protocol v1")
        if int(move["ply"]) != index + 1:
            raise ValueError(f"move {index + 1} has a non-contiguous ply")
    if document["fingerprint"] != replay_fingerprint(document):
        raise ValueError("replay fingerprint does not match rules and moves")


def serialize_replay_document(document: Mapping[str, Any]) -> str:
    validate_replay_document(document)
    text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > MAX_REPLAY_BYTES:
        raise ValueError("replay document exceeds the CubeSprite import size limit")
    return text


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return path
        raise FileExistsError(f"refusing to replace a different audit artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":  # pragma: no cover - exercised on cloud/Linux hosts.
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return path
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_replay_atomic(path: str | Path, document: Mapping[str, Any]) -> Path:
    target = Path(path)
    if not target.name.endswith(".c4replay.json"):
        raise ValueError("CubeSprite replay filenames must end in .c4replay.json")
    return _atomic_write_bytes(target, serialize_replay_document(document).encode("utf-8"))


def _outcome_label(winner: int) -> str:
    return {1: "p1_win", -1: "p2_win", 0: "draw"}[int(winner)]


def select_representative_games(
    games: Iterable[GameRecord],
    *,
    limit: int = 5,
) -> tuple[RepresentativeGame, ...]:
    """Select length anchors, then fill missing outcomes, independent of input order."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("representative game limit must be a non-negative integer")
    ordered = sorted(
        tuple(games),
        key=lambda game: (
            len(game.moves),
            int(game.game_id),
            int(game.seed),
            str(game.producer_model_id),
        ),
    )
    game_ids = [int(game.game_id) for game in ordered]
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("representative selection requires unique game IDs")
    if not ordered or limit == 0:
        return ()
    target_count = min(limit, len(ordered))
    for game in ordered:
        if int(game.winner) not in (-1, 0, 1):
            raise ValueError("game winner must be -1, 0, or 1")

    selected: dict[int, list[str]] = {}

    def add_nearest(target: float, reason: str, candidates: Sequence[int] | None = None) -> None:
        if len(selected) >= target_count:
            return
        pool = tuple(range(len(ordered))) if candidates is None else tuple(candidates)
        pool = tuple(index for index in pool if index not in selected)
        if not pool:
            return
        choice = min(pool, key=lambda index: (abs(index - target), index, game_ids[index]))
        selected[choice] = [reason]

    last_index = len(ordered) - 1
    lower_anchor = 1 if len(ordered) > 2 else 0
    upper_anchor = last_index - 1 if len(ordered) > 2 else last_index
    length_anchors = (
        (float(lower_anchor), "length:second-shortest"),
        ((lower_anchor + upper_anchor) / 2.0, "length:median-interior"),
        (float(upper_anchor), "length:second-longest"),
    )
    for target, reason in length_anchors:
        add_nearest(target, reason)

    present_outcomes = sorted({int(game.winner) for game in ordered}, reverse=True)
    covered_outcomes = {int(ordered[index].winner) for index in selected}
    median_rank = (lower_anchor + upper_anchor) / 2.0
    for outcome in present_outcomes:
        if outcome in covered_outcomes or len(selected) >= target_count:
            continue
        candidates = [index for index, game in enumerate(ordered) if int(game.winner) == outcome]
        add_nearest(median_rank, f"outcome:{_outcome_label(outcome)}", candidates)
        covered_outcomes.add(outcome)

    fill_quantiles = (0.25, 0.75, 0.125, 0.375, 0.625, 0.875)
    interior_span = upper_anchor - lower_anchor
    for quantile in fill_quantiles:
        add_nearest(
            lower_anchor + quantile * interior_span,
            f"length:interior-q{int(round(quantile * 100))}",
        )
    while len(selected) < target_count:
        add_nearest(median_rank, "length:nearest-unselected")

    return tuple(
        RepresentativeGame(game=ordered[index], reasons=tuple(selected[index]))
        for index in sorted(selected)
    )


def build_audit_index(
    selections: Sequence[RepresentativeGame],
    documents: Mapping[int, Mapping[str, Any]],
    *,
    run_id: str,
    generation: int,
    checkpoint_id: str,
    checkpoint_sha256: str,
    created_at: datetime | str | None = None,
    filenames: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Build the training-only sidecar for portable replay files."""

    if not run_id or not checkpoint_id:
        raise ValueError("run_id and checkpoint_id must be non-empty")
    if (
        len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    if int(generation) < 0:
        raise ValueError("generation cannot be negative")
    rows: list[dict[str, Any]] = []
    seen_game_ids: set[int] = set()
    for selection in selections:
        game = selection.game
        game_id = int(game.game_id)
        if game_id in seen_game_ids:
            raise ValueError("audit index cannot contain duplicate game IDs")
        seen_game_ids.add(game_id)
        try:
            document = documents[game_id]
        except KeyError as exc:
            raise ValueError(f"missing replay document for game {game_id}") from exc
        serialized = serialize_replay_document(document).encode("utf-8")
        if (
            int(document["move_count"]) != len(game.moves)
            or document["winner"] != int(game.winner)
        ):
            raise ValueError(f"replay document does not describe game {game_id}")
        filename = (filenames or {}).get(game_id, f"{document['id']}.c4replay.json")
        if Path(filename).name != filename or not filename.endswith(".c4replay.json"):
            raise ValueError("audit replay filename must be a basename ending in .c4replay.json")
        rows.append(
            {
                "game_id": game_id,
                "seed": int(game.seed),
                "producer_model_id": str(game.producer_model_id),
                "winner": int(game.winner),
                "is_draw": bool(game.is_draw),
                "move_count": len(game.moves),
                "full_search_positions": int(game.full_search_positions),
                "fast_search_positions": int(game.fast_search_positions),
                "total_simulations": int(game.total_simulations),
                "inference_batches": int(game.inference_batches),
                "inference_positions": int(game.inference_positions),
                "max_inference_batch": int(game.max_inference_batch),
                "selection_reasons": list(selection.reasons),
                "replay_id": str(document["id"]),
                "fingerprint": str(document["fingerprint"]),
                "filename": filename,
                "file_sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )
    return {
        "format": AUDIT_INDEX_FORMAT,
        "format_version": AUDIT_INDEX_VERSION,
        "run_id": run_id,
        "generation": int(generation),
        "created_at": _timestamp(created_at),
        "checkpoint": {"id": checkpoint_id, "sha256": checkpoint_sha256},
        "selection": {
            "method": "length-anchors-with-outcome-coverage",
            "actual_games": len(rows),
        },
        "replays": rows,
    }


def write_audit_index_atomic(path: str | Path, index: Mapping[str, Any]) -> Path:
    encoded = json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _atomic_write_bytes(Path(path), encoded.encode("utf-8"))


__all__ = [
    "AUDIT_INDEX_FORMAT",
    "AUDIT_INDEX_VERSION",
    "MOVE_KEYS",
    "REPLAY_FORMAT",
    "REPLAY_KEYS",
    "REPLAY_PROTOCOL_VERSION",
    "RepresentativeGame",
    "build_audit_index",
    "game_record_to_replay",
    "replay_fingerprint",
    "select_representative_games",
    "serialize_replay_document",
    "validate_replay_document",
    "write_audit_index_atomic",
    "write_replay_atomic",
]
