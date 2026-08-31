from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .game import (
    BLUE,
    DRAW,
    FULL_MASK,
    ONGOING,
    RED,
    Layer0State,
    canonical_pair,
    can_still_make_four,
    has_four,
    winning_cells,
)


WIN = 1
LOSS = -1

@dataclass(frozen=True, slots=True)
class Score:
    outcome: int
    distance: int

    def advanced(self) -> Score:
        return Score(-self.outcome, self.distance + 1)


@dataclass(frozen=True, slots=True)
class Analysis:
    outcome: int
    distance: int
    optimal_moves: tuple[int, ...]
    principal_move: int | None
    nodes: int
    cache_hits: int
    proven: bool = True
    note: str = "exact"

    @property
    def label(self) -> str:
        return {WIN: "win", DRAW: "draw", LOSS: "loss"}[self.outcome]


class StrongSolver:
    """Exact solver whose random choices are restricted to proven equal-value moves."""

    def __init__(self, *, seed: int | None = None, timeout: float = 180.0) -> None:
        from .native import PersistentNativeSolver

        self._rng = random.Random(seed)
        self.native = PersistentNativeSolver(timeout=timeout)

    def analyze(self, state: Layer0State) -> Analysis:
        terminal = state.outcome(stop_when_dead=True)
        if terminal != ONGOING:
            return Analysis(
                outcome=DRAW if terminal == DRAW else (WIN if terminal == state.to_move else LOSS),
                distance=0,
                optimal_moves=(),
                principal_move=None,
                nodes=0,
                cache_hits=0,
                proven=True,
                note="terminal",
            )
        result = self.native.analyze(state)
        if not result.optimal_moves:
            raise RuntimeError("exact backend returned no optimal move for a non-terminal state")
        return Analysis(
            outcome=result.outcome,
            distance=0,
            optimal_moves=result.optimal_moves,
            principal_move=self._rng.choice(result.optimal_moves),
            nodes=result.nodes,
            cache_hits=result.cache_hits,
            proven=True,
            note=(
                f"persistent native exact ({result.seconds:.3f}s, "
                f"cache {result.cache_size:,}"
                + (", reset" if result.cache_reset else "")
                + ")"
            ),
        )

    def choose_move(self, state: Layer0State) -> int:
        result = self.analyze(state)
        if result.principal_move is None:
            raise ValueError("terminal state has no move")
        return result.principal_move

    def close(self, *, force: bool = False) -> None:
        self.native.close(force=force)

    def __enter__(self) -> StrongSolver:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class HybridSolver:
    """Responsive UI solver: exact tactics/endgames, bounded search in wide openings."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        time_limit: float = 1.0,
        max_depth: int = 5,
        native_timeout: float = 8.0,
        native_empty_limit: int = 16,
    ) -> None:
        from .native import NativeSolver

        self._rng = random.Random(seed)
        self.time_limit = float(time_limit)
        self.max_depth = int(max_depth)
        self.native_empty_limit = int(native_empty_limit)
        self.native = NativeSolver(timeout=native_timeout)
        self._ordering = ExactSolver(seed=0)
        self._nodes = 0
        self._deadline = 0.0

    def analyze(self, state: Layer0State) -> Analysis:
        terminal = state.outcome()
        if terminal != ONGOING:
            return Analysis(
                DRAW if terminal == DRAW else (WIN if terminal == state.to_move else LOSS),
                0,
                (),
                None,
                0,
                0,
                True,
                "terminal",
            )
        occupied = state.occupied
        immediate = winning_cells(state.current_bits, occupied)
        if immediate:
            moves = tuple(move.bit_length() for move in ExactSolver._iter_bits(immediate))
            return self._analysis(WIN, moves, 1, True, "immediate win")
        threats = winning_cells(state.opponent_bits, occupied)
        if threats.bit_count() >= 2:
            legal = state.legal_positions
            return self._analysis(LOSS, legal, 2, True, "unavoidable double threat")

        if self.native.available and state.empty_count <= self.native_empty_limit:
            try:
                native = self.native.analyze(state)
                return Analysis(
                    outcome=native.outcome,
                    distance=0,
                    optimal_moves=native.optimal_moves,
                    principal_move=self._rng.choice(native.optimal_moves),
                    nodes=native.nodes,
                    cache_hits=native.cache_hits,
                    proven=True,
                    note=f"native exhaustive ({native.seconds:.2f}s)",
                )
            except TimeoutError:
                pass
            except Exception as exc:
                native_note = f"native unavailable: {type(exc).__name__}"
            else:
                native_note = "native timeout"
        else:
            native_note = "opening exceeds exhaustive threshold"

        if threats:
            forced = (threats.bit_length(),)
            return self._analysis(DRAW, forced, 0, False, "forced block; continuation unproven")

        self._deadline = time.monotonic() + self.time_limit
        self._nodes = 0
        moves = tuple(ExactSolver._iter_bits(FULL_MASK ^ occupied))
        best_positions: tuple[int, ...] = ()
        best_score = -10**9
        completed_depth = 0
        for depth in range(1, self.max_depth + 1):
            scores: list[tuple[int, int]] = []
            try:
                for move in self._order_for_bounded(state.current_bits, state.opponent_bits, moves):
                    score = -self._bounded(
                        state.opponent_bits,
                        state.current_bits | move,
                        depth - 1,
                        -10**9,
                        10**9,
                    )
                    scores.append((move.bit_length(), score))
            except TimeoutError:
                break
            if scores:
                best_score = max(score for _, score in scores)
                best_positions = tuple(sorted(position for position, score in scores if score == best_score))
                completed_depth = depth
        if not best_positions:
            best_positions = tuple(move.bit_length() for move in moves)
        estimated = WIN if best_score >= 50_000 else LOSS if best_score <= -50_000 else DRAW
        return Analysis(
            estimated,
            completed_depth,
            best_positions,
            self._rng.choice(best_positions),
            self._nodes,
            0,
            False,
            f"bounded depth {completed_depth}; {native_note}",
        )

    def choose_move(self, state: Layer0State) -> int:
        result = self.analyze(state)
        if result.principal_move is None:
            raise ValueError("terminal state has no move")
        return result.principal_move

    def _analysis(
        self,
        outcome: int,
        moves: tuple[int, ...],
        distance: int,
        proven: bool,
        note: str,
    ) -> Analysis:
        ordered = tuple(sorted(moves))
        return Analysis(outcome, distance, ordered, self._rng.choice(ordered), 0, 0, proven, note)

    def _bounded(
        self,
        current: int,
        opponent: int,
        depth: int,
        alpha: int,
        beta: int,
    ) -> int:
        self._nodes += 1
        if self._nodes & 1023 == 0 and time.monotonic() >= self._deadline:
            raise TimeoutError
        occupied = current | opponent
        wins = winning_cells(current, occupied)
        if wins:
            return 100_000 + depth
        threats = winning_cells(opponent, occupied)
        if threats.bit_count() >= 2:
            return -100_000 - depth
        if occupied == FULL_MASK:
            return 0
        if depth <= 0:
            return self._evaluate(current, opponent)
        candidates = (threats,) if threats else tuple(ExactSolver._iter_bits(FULL_MASK ^ occupied))
        value = -10**9
        for move in self._order_for_bounded(current, opponent, candidates):
            score = -self._bounded(opponent, current | move, depth - 1, -beta, -alpha)
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    @staticmethod
    def _evaluate(current: int, opponent: int) -> int:
        from .game import WINNING_LINES

        weights = (0, 2, 15, 240, 100_000)
        score = 0
        for line in WINNING_LINES:
            own = (line & current).bit_count()
            other = (line & opponent).bit_count()
            if other == 0:
                score += weights[own]
            if own == 0:
                score -= weights[other]
        return score

    def _order_for_bounded(
        self, current: int, opponent: int, moves: Iterable[int]
    ) -> tuple[int, ...]:
        allowed = set(moves)
        return tuple(
            move for move in self._ordering._ordered_moves(current, opponent) if move in allowed
        )


class ExactSolver:
    """Strong 5x5 solver using exact negamax, D4 canonicalization and threat pruning."""

    def __init__(self, *, seed: int | None = None, cache_path: str | Path | None = None) -> None:
        self._cache: dict[tuple[int, int], Score] = {}
        self._rng = random.Random(seed)
        self._nodes = 0
        self._cache_hits = 0
        self._lock = threading.RLock()
        self._cache_path = Path(cache_path).resolve() if cache_path else None
        if self._cache_path is not None and self._cache_path.exists():
            self._load_disk_cache()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._nodes = 0
            self._cache_hits = 0

    def analyze(self, state: Layer0State) -> Analysis:
        terminal = state.outcome(stop_when_dead=True)
        if terminal != ONGOING:
            return Analysis(
                outcome=DRAW if terminal == DRAW else (WIN if terminal == state.to_move else LOSS),
                distance=0,
                optimal_moves=(),
                principal_move=None,
                nodes=0,
                cache_hits=0,
            )
        with self._lock:
            start_nodes = self._nodes
            start_hits = self._cache_hits
            moves = self._ordered_moves(state.current_bits, state.opponent_bits)
            scored: list[tuple[int, Score]] = []
            for move in moves:
                child = self._score_after_move(state.current_bits, state.opponent_bits, move)
                scored.append((move.bit_length(), child))
            best = max((score for _, score in scored), key=self._rank)
            optimal = tuple(
                position for position, score in scored if self._rank(score) == self._rank(best)
            )
            principal = self._rng.choice(optimal)
            return Analysis(
                outcome=best.outcome,
                distance=best.distance,
                optimal_moves=tuple(sorted(optimal)),
                principal_move=principal,
                nodes=self._nodes - start_nodes,
                cache_hits=self._cache_hits - start_hits,
            )

    def choose_move(self, state: Layer0State) -> int:
        analysis = self.analyze(state)
        if analysis.principal_move is None:
            raise ValueError("terminal state has no move")
        return analysis.principal_move

    def solve_value(self, state: Layer0State) -> Score:
        terminal = state.outcome(stop_when_dead=True)
        if terminal != ONGOING:
            return Score(DRAW if terminal == DRAW else (WIN if terminal == state.to_move else LOSS), 0)
        with self._lock:
            return self._solve(state.current_bits, state.opponent_bits)

    def save_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._cache_path)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS scores "
                "(current_bits INTEGER NOT NULL, opponent_bits INTEGER NOT NULL, "
                " outcome INTEGER NOT NULL, distance INTEGER NOT NULL, "
                " PRIMARY KEY (current_bits, opponent_bits))"
            )
            connection.executemany(
                "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                ((key[0], key[1], score.outcome, score.distance) for key, score in self._cache.items()),
            )
            connection.commit()
        finally:
            connection.close()

    def _load_disk_cache(self) -> None:
        connection = sqlite3.connect(self._cache_path)
        try:
            rows = connection.execute(
                "SELECT current_bits, opponent_bits, outcome, distance FROM scores"
            )
            self._cache.update(
                ((int(current), int(opponent)), Score(int(outcome), int(distance)))
                for current, opponent, outcome, distance in rows
            )
        except sqlite3.OperationalError:
            pass
        finally:
            connection.close()

    @staticmethod
    def _rank(score: Score) -> tuple[int, int]:
        if score.outcome == WIN:
            return WIN, -score.distance
        if score.outcome == LOSS:
            return LOSS, score.distance
        return DRAW, score.distance

    def _score_after_move(self, current: int, opponent: int, move: int) -> Score:
        played = current | move
        if has_four(played):
            return Score(WIN, 1)
        return self._solve(opponent, played).advanced()

    def _solve(self, current: int, opponent: int) -> Score:
        self._nodes += 1
        key = canonical_pair(current, opponent)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        occupied = current | opponent
        if occupied == FULL_MASK:
            result = Score(DRAW, 0)
        elif not can_still_make_four(current, opponent) and not can_still_make_four(opponent, current):
            result = Score(DRAW, 0)
        else:
            immediate = winning_cells(current, occupied)
            if immediate:
                result = Score(WIN, 1)
            else:
                opponent_wins = winning_cells(opponent, occupied)
                if opponent_wins.bit_count() >= 2:
                    result = Score(LOSS, 2)
                else:
                    moves = (opponent_wins,) if opponent_wins else self._ordered_moves(current, opponent)
                    best: Score | None = None
                    for move in moves:
                        child = self._score_after_move(current, opponent, move)
                        if best is None or self._rank(child) > self._rank(best):
                            best = child
                        if best.outcome == WIN and best.distance <= 3:
                            break
                    assert best is not None
                    result = best
        self._cache[key] = result
        return result

    def _ordered_moves(self, current: int, opponent: int) -> tuple[int, ...]:
        occupied = current | opponent
        empty = FULL_MASK ^ occupied
        own_wins = winning_cells(current, occupied)
        if own_wins:
            return tuple(self._iter_bits(own_wins))
        opponent_wins = winning_cells(opponent, occupied)
        if opponent_wins.bit_count() == 1:
            return (opponent_wins,)

        candidates: list[tuple[tuple[int, int, int, int], int]] = []
        for move in self._iter_bits(empty):
            own_after = current | move
            fork_count = winning_cells(own_after, occupied | move).bit_count()
            blocks = sum(1 for line in self._live_lines(opponent, current) if line & move)
            extends = sum(1 for line in self._live_lines(current, opponent) if line & move)
            cell = move.bit_length() - 1
            row, col = divmod(cell, 5)
            centrality = 4 - abs(row - 2) - abs(col - 2)
            candidates.append(((fork_count, blocks, extends, centrality), move))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return tuple(move for _, move in candidates)

    @staticmethod
    def _live_lines(bits: int, blocker: int) -> Iterable[int]:
        from .game import WINNING_LINES

        return (line for line in WINNING_LINES if line & blocker == 0 and line & bits)

    @staticmethod
    def _iter_bits(bits: int) -> Iterable[int]:
        while bits:
            low = bits & -bits
            yield low
            bits ^= low

    def stats_json(self) -> str:
        return json.dumps(
            {"nodes": self._nodes, "cache_hits": self._cache_hits, "cache_size": len(self._cache)},
            ensure_ascii=False,
            sort_keys=True,
        )
