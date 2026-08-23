"""Fixed-opening, color-swapped evaluation used by the V3 gate."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from connect4_core.rules import (
    CLASSIC_RULE,
    DEFAULT_RULE_REGISTRY,
    GameOutcome,
    GameState,
    RuleEngine,
    TurnAction,
)

from .gate import GateGameResult
from .search import MCTS, Predictor, policy_from_visits


OPENING_SCHEMA_VERSION = 2
MAX_GATE_TURNS = 300


@dataclass(frozen=True)
class Opening:
    opening_id: str
    seed: int
    columns: tuple[int, ...]
    rule_id: str
    rule_version: int

    def __post_init__(self) -> None:
        if not self.opening_id:
            raise ValueError("opening_id must not be empty")
        if not self.rule_id:
            raise ValueError("rule_id must not be empty")
        if int(self.rule_version) < 1:
            raise ValueError("rule_version must be positive")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "columns", tuple(int(column) for column in self.columns))
        object.__setattr__(self, "rule_version", int(self.rule_version))
        if any(column < 0 or column >= 25 for column in self.columns):
            raise ValueError("opening columns must be in [0, 24]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        return payload


def _d4_board_key(state: GameState) -> tuple[str, int, bytes]:
    variants: list[bytes] = []
    for transform in range(8):
        transformed = np.rot90(state.board, k=transform % 4, axes=(-2, -1))
        if transform >= 4:
            transformed = np.flip(transformed, axis=-1)
        variants.append(np.ascontiguousarray(transformed).tobytes())
    return state.rule_id, int(state.player_to_move), min(variants)


def _has_immediate_win(engine: RuleEngine, state: GameState) -> bool:
    for column in np.flatnonzero(engine.legal_column_mask(state)):
        next_state = engine.step(state, TurnAction.place(int(column)))
        if next_state.outcome.winner == state.player_to_move:
            return True
    return False


def build_openings(
    count: int,
    *,
    run_seed: int,
    rule_id: str = CLASSIC_RULE.rule_id,
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
    seen_positions: set[tuple[str, int, bytes]] = set()
    engine = RuleEngine(rule_id)
    attempt = 0
    max_attempts = max(1000, count * 1000)
    while len(openings) < count and attempt < max_attempts:
        seed = int((run_seed + 1_000_003 + attempt) % (2**63 - 1))
        rng = np.random.default_rng(seed)
        state = engine.initial_state()
        columns: list[int] = []
        prefix_length = 0 if not openings else non_empty_lengths[(attempt - 1) % len(non_empty_lengths)]
        valid_prefix = True
        for _ in range(prefix_length):
            while (required := engine.required_action(state)) is not None:
                state = engine.step(state, required)
                if state.terminal:
                    valid_prefix = False
                    break
            if not valid_prefix:
                break
            legal = np.flatnonzero(engine.legal_column_mask(state))
            if legal.size == 0:
                valid_prefix = False
                break
            column = int(rng.choice(legal))
            state = engine.step(state, TurnAction.place(column))
            if state.terminal:
                valid_prefix = False
                break
            columns.append(column)
        if not valid_prefix:
            attempt += 1
            continue
        if _has_immediate_win(engine, state):
            attempt += 1
            continue
        key = _d4_board_key(state)
        if key not in seen_positions:
            seen_positions.add(key)
            openings.append(
                Opening(
                    opening_id=f"opening-{len(openings):04d}",
                    seed=seed,
                    columns=tuple(columns),
                    rule_id=engine.spec.rule_id,
                    rule_version=engine.spec.rule_version,
                )
            )
        attempt += 1
    if len(openings) != count:
        raise RuntimeError("could not generate enough unique opening positions")
    return tuple(openings)


def write_opening_manifest(path: str | Path, openings: Iterable[Opening]) -> Path:
    target = Path(path)
    rows = tuple(openings)
    if not rows:
        raise ValueError("opening manifest must contain at least one opening")
    rule_contexts = {(opening.rule_id, opening.rule_version) for opening in rows}
    if len(rule_contexts) != 1:
        raise ValueError("an opening manifest must use exactly one rule context")
    rule_id, rule_version = next(iter(rule_contexts))
    payload = {
        "schema_version": OPENING_SCHEMA_VERSION,
        "rule_id": rule_id,
        "rule_version": rule_version,
        "rule_registry_hash": DEFAULT_RULE_REGISTRY.registry_hash,
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
    expected_manifest_keys = {
        "schema_version",
        "rule_id",
        "rule_version",
        "rule_registry_hash",
        "openings",
    }
    if not isinstance(payload, dict) or set(payload) != expected_manifest_keys:
        raise ValueError("opening manifest fields do not match schema V2")
    if payload.get("schema_version") != OPENING_SCHEMA_VERSION:
        raise ValueError(f"unsupported opening schema: {payload.get('schema_version')!r}")
    if payload.get("rule_registry_hash") != DEFAULT_RULE_REGISTRY.registry_hash:
        raise ValueError("opening manifest rule registry hash does not match executable rules")
    rule_id = str(payload.get("rule_id", ""))
    rule_version = int(payload.get("rule_version", 0))
    raw_openings = payload.get("openings", [])
    expected_row_keys = {"opening_id", "seed", "columns", "rule_id", "rule_version"}
    if not isinstance(raw_openings, list) or any(
        not isinstance(row, dict) or set(row) != expected_row_keys
        for row in raw_openings
    ):
        raise ValueError("opening rows do not match schema V2")
    openings = tuple(
        Opening(
            opening_id=str(row["opening_id"]),
            seed=int(row["seed"]),
            columns=tuple(int(column) for column in row["columns"]),
            rule_id=str(row["rule_id"]),
            rule_version=int(row["rule_version"]),
        )
        for row in raw_openings
    )
    if not openings or len({opening.opening_id for opening in openings}) != len(openings):
        raise ValueError("opening manifest must contain unique, non-empty openings")
    if any(
        opening.rule_id != rule_id or opening.rule_version != rule_version
        for opening in openings
    ):
        raise ValueError("opening rows do not match the manifest rule context")
    spec = DEFAULT_RULE_REGISTRY.get(rule_id)
    if spec.rule_version != rule_version:
        raise ValueError(
            f"opening rule version {rule_version} does not match registered version {spec.rule_version}"
        )
    return openings


def _apply_opening(engine: RuleEngine, opening: Opening) -> GameState:
    if opening.rule_id != engine.spec.rule_id or opening.rule_version != engine.spec.rule_version:
        raise ValueError(f"opening {opening.opening_id!r} does not match the rule engine")
    state = engine.initial_state()
    for column in opening.columns:
        while (required := engine.required_action(state)) is not None:
            state = engine.step(state, required)
            if state.terminal:
                raise ValueError(f"opening {opening.opening_id!r} terminates during forced pass")
        state = engine.step(state, TurnAction.place(column))
        if state.terminal:
            raise ValueError(f"opening {opening.opening_id!r} is terminal")
    return state


def _deterministic_rng(seed: int, ply: int, stream: int) -> np.random.Generator:
    seed_value = int(seed)
    return np.random.default_rng(
        np.random.SeedSequence(
            [seed_value & 0xFFFFFFFF, (seed_value >> 32) & 0xFFFFFFFF, int(ply), int(stream)]
        )
    )


def _search_column(
    predictor: Predictor,
    state: GameState,
    *,
    engine: RuleEngine,
    simulations: int,
    cpuct: float,
    seed: int,
    ply: int,
) -> int:
    result = MCTS(
        predictor,
        engine=engine,
        cpuct=cpuct,
        virtual_loss=1.0,
        num_threads=1,
    ).search(
        state,
        simulations,
        rng=_deterministic_rng(seed, ply, 0),
        add_root_noise=False,
    )
    policy = policy_from_visits(
        result.visit_counts,
        temperature=0.0,
        valid_mask=engine.legal_column_mask(state),
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
    engine = RuleEngine(opening.rule_id)
    if engine.spec.rule_version != opening.rule_version:
        raise ValueError(
            f"opening rule version {opening.rule_version} does not match registered version "
            f"{engine.spec.rule_version}"
        )
    state = _apply_opening(engine, opening)
    while not state.terminal and state.turn_index < MAX_GATE_TURNS:
        required = engine.required_action(state)
        if required is not None:
            state = engine.step(state, required)
            continue
        ply = state.turn_index
        candidate_turn = (state.player_to_move == 1) == bool(candidate_is_first)
        predictor = candidate_predictor if candidate_turn else incumbent_predictor
        if predictor is None:
            legal = np.flatnonzero(engine.legal_column_mask(state))
            column = int(_deterministic_rng(opening.seed, ply, 1).choice(legal))
        else:
            column = _search_column(
                predictor,
                state,
                engine=engine,
                simulations=search_sims,
                cpuct=cpuct,
                seed=opening.seed,
                ply=ply,
            )
        state = engine.step(state, TurnAction.place(column))
    if not state.terminal:
        raise RuntimeError(f"paired evaluation did not terminate within {MAX_GATE_TURNS} turns")
    winner = state.outcome.winner or 0
    return GateGameResult.from_player_winner(
        opening_id=opening.opening_id,
        seed=opening.seed,
        candidate_is_first=candidate_is_first,
        winner=winner,
        is_draw=state.outcome == GameOutcome.DRAW,
    )


def play_paired_openings(
    openings: Iterable[Opening],
    *,
    candidate_predictor: Predictor,
    incumbent_predictor: Predictor | None,
    search_sims: int,
    cpuct: float,
) -> tuple[GateGameResult, ...]:
    rows = tuple(openings)
    rule_contexts = {(opening.rule_id, opening.rule_version) for opening in rows}
    if len(rule_contexts) > 1:
        raise ValueError("paired gate openings must use exactly one rule context")
    results: list[GateGameResult] = []
    for opening in rows:
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
    "MAX_GATE_TURNS",
    "OPENING_SCHEMA_VERSION",
    "Opening",
    "build_openings",
    "load_opening_manifest",
    "play_paired_game",
    "play_paired_openings",
    "write_opening_manifest",
]
