"""Fixed-opening, color-swapped evaluation used by the V3 gate."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from connect4_core import GameRules

from .gate import GateGameResult
from .model import column_to_legacy_action, legal_column_mask
from .search import MCTS, Predictor, policy_from_visits


OPENING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Opening:
    opening_id: str
    seed: int
    columns: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        return payload


def _d4_board_key(board: np.ndarray, player: int) -> tuple[int, bytes]:
    variants: list[bytes] = []
    for transform in range(8):
        transformed = np.rot90(board, k=transform % 4, axes=(-2, -1))
        if transform >= 4:
            transformed = np.flip(transformed, axis=-1)
        variants.append(np.ascontiguousarray(transformed).tobytes())
    return int(player), min(variants)


def _has_immediate_win(game: GameRules, board: np.ndarray, player: int) -> bool:
    for column in np.flatnonzero(legal_column_mask(board).reshape(-1)):
        action = column_to_legacy_action(board, int(column))
        next_board, _next_player = game.get_next_state(board, player, action)
        if game.check_win(next_board, player):
            return True
    return False


def build_openings(
    count: int,
    *,
    run_seed: int,
    prefix_lengths: Iterable[int] = (0, 2, 4, 6),
) -> tuple[Opening, ...]:
    """Build deterministic, D4-deduplicated, non-terminal opening prefixes."""

    if count < 1:
        raise ValueError("opening count must be positive")
    lengths = tuple(int(length) for length in prefix_lengths)
    if not lengths or lengths[0] != 0 or tuple(sorted(set(lengths))) != lengths:
        raise ValueError("prefix_lengths must be strictly increasing and start at zero")
    if lengths[-1] >= 150:
        raise ValueError("opening prefixes must be shorter than board capacity")
    non_empty_lengths = lengths[1:] or (1,)
    openings: list[Opening] = []
    seen_positions: set[tuple[int, bytes]] = set()
    game = GameRules()
    attempt = 0
    max_attempts = max(1000, count * 1000)
    while len(openings) < count and attempt < max_attempts:
        seed = int((run_seed + 1_000_003 + attempt) % (2**63 - 1))
        rng = np.random.default_rng(seed)
        board = game.get_init_board()
        player = 1
        columns: list[int] = []
        prefix_length = 0 if not openings else non_empty_lengths[(attempt - 1) % len(non_empty_lengths)]
        valid_prefix = True
        for _ in range(prefix_length):
            legal = np.flatnonzero(legal_column_mask(board).reshape(-1))
            column = int(rng.choice(legal))
            action = column_to_legacy_action(board, column)
            board, player = game.get_next_state(board, player, action)
            if game.get_game_ended(board, player) != 0:
                valid_prefix = False
                break
            columns.append(column)
        if not valid_prefix:
            attempt += 1
            continue
        if _has_immediate_win(game, board, player):
            attempt += 1
            continue
        key = _d4_board_key(board, player)
        if key not in seen_positions:
            seen_positions.add(key)
            openings.append(Opening(f"opening-{len(openings):04d}", seed, tuple(columns)))
        attempt += 1
    if len(openings) != count:
        raise RuntimeError("could not generate enough unique opening positions")
    return tuple(openings)


def write_opening_manifest(path: str | Path, openings: Iterable[Opening]) -> Path:
    target = Path(path)
    rows = tuple(openings)
    payload = {
        "schema_version": OPENING_SCHEMA_VERSION,
        "openings": [opening.to_dict() for opening in rows],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def load_opening_manifest(path: str | Path) -> tuple[Opening, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != OPENING_SCHEMA_VERSION:
        raise ValueError(f"unsupported opening schema: {payload.get('schema_version')!r}")
    openings = tuple(
        Opening(
            opening_id=str(row["opening_id"]),
            seed=int(row["seed"]),
            columns=tuple(int(column) for column in row["columns"]),
        )
        for row in payload.get("openings", [])
    )
    if not openings or len({opening.opening_id for opening in openings}) != len(openings):
        raise ValueError("opening manifest must contain unique, non-empty openings")
    return openings


def _apply_opening(game: GameRules, opening: Opening) -> tuple[np.ndarray, int]:
    board = game.get_init_board()
    player = 1
    for column in opening.columns:
        action = column_to_legacy_action(board, column)
        board, player = game.get_next_state(board, player, action)
        if game.get_game_ended(board, player) != 0:
            raise ValueError(f"opening {opening.opening_id!r} is terminal")
    return board, player


def _deterministic_rng(seed: int, ply: int, stream: int) -> np.random.Generator:
    seed_value = int(seed)
    return np.random.default_rng(
        np.random.SeedSequence(
            [seed_value & 0xFFFFFFFF, (seed_value >> 32) & 0xFFFFFFFF, int(ply), int(stream)]
        )
    )


def _search_column(
    predictor: Predictor,
    canonical_board: np.ndarray,
    *,
    simulations: int,
    cpuct: float,
    seed: int,
    ply: int,
) -> int:
    result = MCTS(predictor, cpuct=cpuct, virtual_loss=1.0, num_threads=1).search(
        canonical_board,
        simulations,
        rng=_deterministic_rng(seed, ply, 0),
        add_root_noise=False,
    )
    policy = policy_from_visits(
        result.visit_counts,
        temperature=0.0,
        valid_mask=legal_column_mask(canonical_board).reshape(-1),
    )
    return int(np.argmax(policy))


def play_paired_game(
    opening: Opening,
    *,
    candidate_is_first: bool,
    candidate_predictor: Predictor,
    incumbent_predictor: Predictor | None,
    search_sims: int,
    cpuct: float,
) -> GateGameResult:
    """Play one deterministic gate game; ``None`` denotes a random baseline."""

    if search_sims < 1 or cpuct <= 0.0:
        raise ValueError("search_sims and cpuct must be positive")
    game = GameRules()
    board, player = _apply_opening(game, opening)
    start_ply = len(opening.columns)
    winner = 0
    for ply in range(start_ply, 150):
        candidate_turn = (player == 1) == bool(candidate_is_first)
        predictor = candidate_predictor if candidate_turn else incumbent_predictor
        canonical = game.get_canonical_form(board, player)
        if predictor is None:
            legal = np.flatnonzero(legal_column_mask(canonical).reshape(-1))
            column = int(_deterministic_rng(opening.seed, ply, 1).choice(legal))
        else:
            column = _search_column(
                predictor,
                canonical,
                simulations=search_sims,
                cpuct=cpuct,
                seed=opening.seed,
                ply=ply,
            )
        action = column_to_legacy_action(board, column)
        board, player = game.get_next_state(board, player, action)
        result = float(game.get_game_ended(board, player))
        if result != 0.0:
            if not np.isclose(result, 1e-4, rtol=0.0, atol=1e-12):
                winner = 1 if game.check_win(board, 1) else -1
            break
    else:
        raise RuntimeError("paired evaluation did not terminate within 150 plies")
    return GateGameResult.from_player_winner(
        opening_id=opening.opening_id,
        seed=opening.seed,
        candidate_is_first=candidate_is_first,
        winner=winner,
        is_draw=winner == 0,
    )


def play_paired_openings(
    openings: Iterable[Opening],
    *,
    candidate_predictor: Predictor,
    incumbent_predictor: Predictor | None,
    search_sims: int,
    cpuct: float,
) -> tuple[GateGameResult, ...]:
    results: list[GateGameResult] = []
    for opening in openings:
        for candidate_is_first in (True, False):
            results.append(
                play_paired_game(
                    opening,
                    candidate_is_first=candidate_is_first,
                    candidate_predictor=candidate_predictor,
                    incumbent_predictor=incumbent_predictor,
                    search_sims=search_sims,
                    cpuct=cpuct,
                )
            )
    return tuple(results)


__all__ = [
    "OPENING_SCHEMA_VERSION",
    "Opening",
    "build_openings",
    "load_opening_manifest",
    "play_paired_game",
    "play_paired_openings",
    "write_opening_manifest",
]
