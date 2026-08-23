"""Generate immutable fixed-opening MCTS targets for search-budget audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from connect4_core.rules import DEFAULT_RULE_REGISTRY, RuleEngine, TurnAction

from .evaluation import Opening
from .model import WDL_DRAW
from .replay import ReplayShard, SEARCH_FULL, TURN_PLACE
from .search import MCTS, Predictor, canonical_board_for_state


@dataclass(frozen=True)
class FixedTargetGeneration:
    replay: ReplayShard
    inference_calls: int
    inference_positions: int
    max_inference_batch: int


def _state_for_opening(opening: Opening):
    engine = RuleEngine(opening.rule_id)
    if engine.spec.rule_version != opening.rule_version:
        raise ValueError("opening rule version does not match the executable registry")
    state = engine.initial_state()
    for column in opening.columns:
        while (required := engine.required_action(state)) is not None:
            state = engine.step(state, required)
        if state.terminal:
            raise ValueError(f"opening {opening.opening_id} reaches a terminal state")
        state = engine.step(state, TurnAction.place(column))
    if state.terminal or engine.required_action(state) is not None:
        raise ValueError(f"opening {opening.opening_id} is not a searchable position")
    return engine, state


def generate_fixed_opening_targets(
    openings: Sequence[Opening],
    *,
    predictor: Predictor,
    search_sims: int,
    audit_seed: int,
    position_start: int = 0,
    cpuct: float = 1.5,
    virtual_loss: float = 1.0,
    mcts_lanes: int = 4,
    root_noise_alpha: float = 0.24,
    root_noise_epsilon: float = 0.06,
) -> FixedTargetGeneration:
    """Search the same immutable openings at one explicit budget.

    Primary and reference runs use the same ``audit_seed`` so they receive the
    same root-noise draw. A repeated-primary run uses a different seed to
    measure stochastic target variability without changing any position.
    """

    rows = tuple(openings)
    if not rows:
        raise ValueError("fixed-opening target generation needs at least one opening")
    if search_sims < 1 or audit_seed < 0 or position_start < 0:
        raise ValueError("search_sims must be positive and seeds/offsets non-negative")
    if cpuct <= 0.0 or virtual_loss < 0.0 or mcts_lanes < 1:
        raise ValueError("invalid MCTS audit settings")
    if root_noise_alpha <= 0.0 or not 0.0 <= root_noise_epsilon <= 1.0:
        raise ValueError("invalid root-noise audit settings")

    boards: list[np.ndarray] = []
    visits: list[np.ndarray] = []
    players: list[int] = []
    turn_indices: list[int] = []
    placement_counts: list[int] = []
    rule_codes: list[int] = []
    terminal_boards: list[np.ndarray] = []
    inference_calls = 0
    inference_positions = 0
    max_inference_batch = 0
    for local_index, opening in enumerate(rows):
        engine, state = _state_for_opening(opening)
        global_index = position_start + local_index
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [audit_seed & 0xFFFFFFFF, audit_seed >> 32, global_index]
            )
        )
        result = MCTS(
            predictor,
            engine=engine,
            cpuct=cpuct,
            virtual_loss=virtual_loss,
            num_threads=mcts_lanes,
        ).search(
            state,
            search_sims,
            rng=rng,
            add_root_noise=root_noise_epsilon > 0.0,
            dirichlet_alpha=root_noise_alpha,
            dirichlet_epsilon=root_noise_epsilon,
        )
        boards.append(canonical_board_for_state(state))
        visits.append(result.visit_counts)
        players.append(state.player_to_move)
        turn_indices.append(state.turn_index)
        placement_counts.append(state.placement_count)
        rule_codes.append(engine.spec.rule_code)
        terminal_boards.append(np.asarray(state.board, dtype=np.int8))
        inference_calls += result.inference_calls
        inference_positions += result.inference_positions
        max_inference_batch = max(max_inference_batch, result.max_inference_batch)

    count = len(rows)
    replay = ReplayShard(
        board=np.stack(boards).astype(np.int8, copy=False),
        visit_counts=np.stack(visits).astype(np.uint32, copy=False),
        policy_weight=np.ones(count, dtype=np.float32),
        wdl=np.full(count, WDL_DRAW, dtype=np.uint8),
        game_id=np.arange(position_start, position_start + count, dtype=np.uint64),
        turn_index=np.asarray(turn_indices, dtype=np.uint16),
        player_to_move=np.asarray(players, dtype=np.int8),
        search_kind=np.full(count, SEARCH_FULL, dtype=np.uint8),
        rule_code=np.asarray(rule_codes, dtype=np.uint16),
        turn_kind=np.full(count, TURN_PLACE, dtype=np.uint8),
        placement_count=np.asarray(placement_counts, dtype=np.uint16),
        opponent_reply_column=np.full(count, -1, dtype=np.int8),
        opponent_reply_mask=np.zeros(count, dtype=np.uint8),
        terminal_board=np.stack(terminal_boards).astype(np.int8, copy=False),
        remaining_turns=np.zeros(count, dtype=np.uint16),
    )
    return FixedTargetGeneration(
        replay=replay,
        inference_calls=inference_calls,
        inference_positions=inference_positions,
        max_inference_batch=max_inference_batch,
    )


__all__ = ["FixedTargetGeneration", "generate_fixed_opening_targets"]
