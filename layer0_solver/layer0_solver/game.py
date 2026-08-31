from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


BOARD_SIZE = 5
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
FULL_MASK = (1 << CELL_COUNT) - 1
CONNECT_N = 4

RED = 1
BLUE = -1
ONGOING = 2
DRAW = 0


def _cell(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def _build_winning_lines() -> tuple[int, ...]:
    masks: set[int] = set()
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            for dr, dc in directions:
                cells = [
                    (row + step * dr, col + step * dc)
                    for step in range(CONNECT_N)
                ]
                if all(0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE for r, c in cells):
                    masks.add(sum(1 << _cell(r, c) for r, c in cells))
    return tuple(sorted(masks))


WINNING_LINES = _build_winning_lines()
LINES_BY_CELL = tuple(
    tuple(line for line in WINNING_LINES if line & (1 << cell))
    for cell in range(CELL_COUNT)
)


def _transform_cell(cell: int, symmetry: int) -> int:
    """Map a cell through one of the eight D4 symmetries."""

    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be in [0, 7]")
    row, col = divmod(cell, BOARD_SIZE)
    if symmetry >= 4:
        col = BOARD_SIZE - 1 - col
        symmetry -= 4
    for _ in range(symmetry):
        row, col = col, BOARD_SIZE - 1 - row
    return _cell(row, col)


SYMMETRY_MAPS = tuple(
    tuple(_transform_cell(cell, symmetry) for cell in range(CELL_COUNT))
    for symmetry in range(8)
)


def transform_bits(bits: int, symmetry: int) -> int:
    transformed = 0
    mapping = SYMMETRY_MAPS[symmetry]
    remaining = bits
    while remaining:
        low = remaining & -remaining
        cell = low.bit_length() - 1
        transformed |= 1 << mapping[cell]
        remaining ^= low
    return transformed


def transform_position(position: int, symmetry: int) -> int:
    if not 1 <= position <= CELL_COUNT:
        raise ValueError("position must be in [1, 25]")
    return SYMMETRY_MAPS[symmetry][position - 1] + 1


def canonical_pair(current_bits: int, opponent_bits: int) -> tuple[int, int]:
    return min(
        (transform_bits(current_bits, symmetry), transform_bits(opponent_bits, symmetry))
        for symmetry in range(8)
    )


def has_four(bits: int) -> bool:
    return any(bits & line == line for line in WINNING_LINES)


def winning_cells(bits: int, occupied: int) -> int:
    wins = 0
    empty = FULL_MASK ^ occupied
    remaining = empty
    while remaining:
        move = remaining & -remaining
        cell = move.bit_length() - 1
        if any((bits | move) & line == line for line in LINES_BY_CELL[cell]):
            wins |= move
        remaining ^= move
    return wins


def can_still_make_four(bits: int, opponent_bits: int) -> bool:
    del bits  # A line stays possible exactly when the opponent has not blocked it.
    return any(line & opponent_bits == 0 for line in WINNING_LINES)


@dataclass(frozen=True, slots=True)
class Layer0State:
    red_bits: int = 0
    blue_bits: int = 0
    to_move: int = RED
    ply: int = 0
    invisible_turns: int = 0
    last_position: int | None = None
    last_was_invisible: bool = False

    def __post_init__(self) -> None:
        if self.to_move not in (RED, BLUE):
            raise ValueError("to_move must be RED (+1) or BLUE (-1)")
        if self.red_bits < 0 or self.blue_bits < 0:
            raise ValueError("bitboards must be non-negative")
        if (self.red_bits | self.blue_bits) & ~FULL_MASK:
            raise ValueError("bitboards contain cells outside the 5x5 board")
        if self.red_bits & self.blue_bits:
            raise ValueError("red and blue bitboards overlap")
        if self.ply < 0 or self.invisible_turns < 0:
            raise ValueError("ply counters must be non-negative")
        if self.last_position is not None and not 1 <= self.last_position <= CELL_COUNT:
            raise ValueError("last_position must be in [1, 25]")

    @property
    def occupied(self) -> int:
        return self.red_bits | self.blue_bits

    @property
    def empty_count(self) -> int:
        return CELL_COUNT - self.occupied.bit_count()

    @property
    def current_bits(self) -> int:
        return self.red_bits if self.to_move == RED else self.blue_bits

    @property
    def opponent_bits(self) -> int:
        return self.blue_bits if self.to_move == RED else self.red_bits

    @property
    def legal_positions(self) -> tuple[int, ...]:
        return tuple(cell + 1 for cell in self.iter_empty_cells())

    def iter_empty_cells(self) -> Iterator[int]:
        remaining = FULL_MASK ^ self.occupied
        while remaining:
            low = remaining & -remaining
            yield low.bit_length() - 1
            remaining ^= low

    def play(self, position: int, *, allow_after_dead_draw: bool = False) -> Layer0State:
        terminal = self.outcome(stop_when_dead=not allow_after_dead_draw)
        if terminal != ONGOING:
            raise ValueError("cannot play after the Layer0 game is terminal")
        if not 1 <= position <= CELL_COUNT:
            raise ValueError("position must be in [1, 25]")
        move = 1 << (position - 1)
        if self.occupied & move:
            raise ValueError(f"Layer0 position {position} is occupied")
        red = self.red_bits | move if self.to_move == RED else self.red_bits
        blue = self.blue_bits | move if self.to_move == BLUE else self.blue_bits
        return Layer0State(
            red_bits=red,
            blue_bits=blue,
            to_move=-self.to_move,
            ply=self.ply + 1,
            invisible_turns=self.invisible_turns,
            last_position=position,
            last_was_invisible=False,
        )

    def pass_invisible(self) -> Layer0State:
        """Record a legal full-game move above Layer0; the visible board is unchanged."""

        if self.outcome(stop_when_dead=True) != ONGOING:
            raise ValueError("cannot pass after the Layer0 game is terminal")
        return Layer0State(
            red_bits=self.red_bits,
            blue_bits=self.blue_bits,
            to_move=-self.to_move,
            ply=self.ply + 1,
            invisible_turns=self.invisible_turns + 1,
            last_position=None,
            last_was_invisible=True,
        )

    def transformed(self, symmetry: int) -> Layer0State:
        last = (
            transform_position(self.last_position, symmetry)
            if self.last_position is not None
            else None
        )
        return Layer0State(
            red_bits=transform_bits(self.red_bits, symmetry),
            blue_bits=transform_bits(self.blue_bits, symmetry),
            to_move=self.to_move,
            ply=self.ply,
            invisible_turns=self.invisible_turns,
            last_position=last,
            last_was_invisible=self.last_was_invisible,
        )

    def outcome(self, *, stop_when_dead: bool = True) -> int:
        if has_four(self.red_bits):
            return RED
        if has_four(self.blue_bits):
            return BLUE
        if self.occupied == FULL_MASK:
            return DRAW
        if stop_when_dead:
            red_live = can_still_make_four(self.red_bits, self.blue_bits)
            blue_live = can_still_make_four(self.blue_bits, self.red_bits)
            if not red_live and not blue_live:
                return DRAW
        return ONGOING

    @classmethod
    def from_moves(cls, moves: Iterable[int | str], *, first: int = RED) -> Layer0State:
        state = cls(to_move=first)
        for raw in moves:
            if isinstance(raw, str) and raw.strip().lower() in {"pass", "p", "skip"}:
                state = state.pass_invisible()
            else:
                state = state.play(int(raw))
        return state

    def rows(self) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = []
        for row in range(BOARD_SIZE):
            values = []
            for col in range(BOARD_SIZE):
                bit = 1 << _cell(row, col)
                values.append(RED if self.red_bits & bit else BLUE if self.blue_bits & bit else 0)
            result.append(tuple(values))
        return tuple(result)
