from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral

import numpy as np

from connect4_core import BOARD_SHAPE
from connect4_core.rules import (
    DEFAULT_RULE_REGISTRY,
    MAX_PLACEMENTS,
    MAX_TURNS,
    GameOutcome,
    RuleEngine,
    TurnAction,
    TurnKind,
)

from .config import SelfPlayConfig, V3Config
from .model import WDL_DRAW, WDL_LOSS, WDL_WIN
from .search import (
    MCTS,
    Predictor,
    RandomPredictor,
    SEARCH_FAST,
    SEARCH_FULL,
    SEARCH_NONE,
    canonical_board_for_state,
    policy_from_visits,
)


MAX_GAME_MOVES = MAX_PLACEMENTS
MAX_GAME_TURNS = MAX_TURNS


@dataclass(frozen=True)
class ParticipantDescriptor:
    """Stable game provenance; only ``player`` reaches neural role encoding."""

    seat: str
    player: int
    controller_type: str
    controller_id: str
    display_name: str
    model_id: str | None = None
    lineage_hash: str | None = None
    artifact_hash: str | None = None

    def __post_init__(self) -> None:
        if self.seat not in {"FIRST", "SECOND"}:
            raise ValueError("participant seat must be FIRST or SECOND.")
        expected_player = 1 if self.seat == "FIRST" else -1
        if self.player != expected_player:
            raise ValueError("participant seat and numeric player disagree.")
        if self.controller_type not in {"model", "human", "random", "external"}:
            raise ValueError("unsupported participant controller_type.")
        if not self.controller_id or not self.display_name:
            raise ValueError("participant controller_id and display_name must be non-empty.")


@dataclass(frozen=True)
class SelfPlaySample:
    """One Replay V2 position captured before its tagged turn."""

    board: np.ndarray
    visit_counts: np.ndarray
    wdl: int
    game_id: int
    ply: int
    player: int
    search_kind: str = SEARCH_FULL
    policy_weight: float | None = None
    rule_code: int = 0
    turn_kind: str = TurnKind.PLACE.value
    placement_count: int = 0
    opponent_reply_column: int = -1
    opponent_reply_mask: int = 0
    terminal_board: np.ndarray | None = None
    remaining_turns: int = 0

    def __post_init__(self) -> None:
        board_raw = np.asarray(self.board)
        visits_raw = np.asarray(self.visit_counts)
        board = np.asarray(board_raw, dtype=np.int8)
        if board.shape != BOARD_SHAPE:
            raise ValueError(f"Self-play board shape must be {BOARD_SHAPE}, got {board.shape}.")
        if not np.all(np.isin(board_raw, (-1, 0, 1))):
            raise ValueError("Self-play board cells must be canonical -1/0/1 values.")
        if visits_raw.shape != (25,):
            raise ValueError(f"Self-play visits shape must be (25,), got {visits_raw.shape}.")
        if not np.issubdtype(visits_raw.dtype, np.integer):
            raise TypeError("Self-play visits must contain integer counts.")
        if np.any(visits_raw < 0) or np.any(visits_raw > np.iinfo(np.uint32).max):
            raise ValueError("Self-play visit counts are outside uint32 range.")
        visits = visits_raw.astype(np.uint32, copy=False)
        if isinstance(self.wdl, (bool, np.bool_)) or not isinstance(self.wdl, Integral):
            raise TypeError("Self-play WDL class must be an integer.")
        if int(self.wdl) not in (WDL_WIN, WDL_DRAW, WDL_LOSS):
            raise ValueError(f"Unknown WDL class: {self.wdl}.")
        integer_fields = (
            self.game_id,
            self.ply,
            self.player,
            self.rule_code,
            self.placement_count,
            self.opponent_reply_column,
            self.opponent_reply_mask,
            self.remaining_turns,
        )
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in integer_fields
        ):
            raise TypeError("Self-play Replay V2 scalar fields must be integers.")
        if int(self.game_id) < 0 or not 0 <= int(self.ply) < MAX_GAME_TURNS:
            raise ValueError("Invalid self-play game_id or turn index.")
        if int(self.player) not in (-1, 1):
            raise ValueError("Self-play player must be +1 or -1.")
        if not 0 <= int(self.rule_code) <= np.iinfo(np.uint16).max:
            raise ValueError("rule_code must fit in uint16.")
        if not 0 <= int(self.placement_count) <= MAX_GAME_MOVES:
            raise ValueError("placement_count is outside the board capacity.")
        if not 0 <= int(self.remaining_turns) <= MAX_GAME_TURNS:
            raise ValueError("remaining_turns is outside the turn limit.")
        if self.search_kind not in (SEARCH_FAST, SEARCH_FULL, SEARCH_NONE):
            raise ValueError(f"Unknown search kind: {self.search_kind!r}.")
        if self.turn_kind not in (TurnKind.PLACE.value, TurnKind.FORCED_PASS.value):
            raise ValueError(f"Unknown turn kind: {self.turn_kind!r}.")
        is_pass = self.turn_kind == TurnKind.FORCED_PASS.value
        if is_pass and (self.search_kind != SEARCH_NONE or np.any(visits)):
            raise ValueError("Forced-pass samples require search_kind=none and zero visits.")
        if not is_pass and (self.search_kind == SEARCH_NONE or not np.any(visits)):
            raise ValueError("Placement samples require fast/full search and positive visits.")
        policy_weight = (
            0.0
            if self.policy_weight is None and self.search_kind != SEARCH_FULL
            else 1.0
            if self.policy_weight is None
            else float(self.policy_weight)
        )
        if not np.isfinite(policy_weight) or policy_weight < 0.0:
            raise ValueError("policy_weight must be finite and non-negative.")
        if self.search_kind != SEARCH_FULL and policy_weight != 0.0:
            raise ValueError("Fast and forced-pass samples must have policy_weight=0.")
        if int(self.opponent_reply_mask) not in (0, 1):
            raise ValueError("opponent_reply_mask must be 0 or 1.")
        if int(self.opponent_reply_mask) == 1:
            if not 0 <= int(self.opponent_reply_column) < 25:
                raise ValueError("Unmasked opponent reply must be a column in [0,24].")
        elif int(self.opponent_reply_column) != -1:
            raise ValueError("Masked opponent reply must use column=-1.")
        terminal = (
            np.zeros(BOARD_SHAPE, dtype=np.int8)
            if self.terminal_board is None
            else np.asarray(self.terminal_board, dtype=np.int8)
        )
        if terminal.shape != BOARD_SHAPE or not np.all(np.isin(terminal, (-1, 0, 1))):
            raise ValueError("terminal_board must be an absolute -1/0/1 board.")
        object.__setattr__(self, "board", np.array(board, copy=True))
        object.__setattr__(self, "visit_counts", np.array(visits, copy=True))
        object.__setattr__(self, "terminal_board", np.array(terminal, copy=True))
        object.__setattr__(self, "policy_weight", policy_weight)

    @property
    def canonical_board(self) -> np.ndarray:
        return self.board

    @property
    def wdl_target(self) -> int:
        return self.wdl

    @property
    def turn_index(self) -> int:
        return self.ply

    @property
    def player_to_move(self) -> int:
        return self.player


@dataclass(frozen=True)
class MoveRecord:
    """Compatibility name for a tagged turn record."""

    ply: int
    player: int
    column: int | None
    legacy_action: int | None
    search_kind: str
    simulations: int
    turn_kind: str = TurnKind.PLACE.value
    placement_count: int = 0

    def __post_init__(self) -> None:
        if self.turn_kind == TurnKind.PLACE.value:
            if self.column is None or self.legacy_action is None:
                raise ValueError("Placement records require column and legacy_action.")
            if self.search_kind not in (SEARCH_FAST, SEARCH_FULL) or self.simulations < 1:
                raise ValueError("Placement records require fast/full search and simulations.")
        elif self.turn_kind == TurnKind.FORCED_PASS.value:
            if self.column is not None or self.legacy_action is not None:
                raise ValueError("Forced-pass records cannot contain placement coordinates.")
            if self.search_kind != SEARCH_NONE or self.simulations != 0:
                raise ValueError("Forced-pass records require search_kind=none and zero simulations.")
        else:
            raise ValueError(f"Unknown turn kind: {self.turn_kind!r}.")

    @property
    def turn_index(self) -> int:
        return self.ply

    @property
    def player_to_move(self) -> int:
        return self.player


@dataclass(frozen=True)
class GameRecord:
    game_id: int
    seed: int
    generation: int
    producer_model_id: str
    winner: int
    is_draw: bool
    moves: tuple[MoveRecord, ...]
    samples: tuple[SelfPlaySample, ...]
    full_search_positions: int
    fast_search_positions: int
    total_simulations: int
    exploration_variant: str = "baseline"
    rule_id: str = "classic"
    rule_code: int = 0
    rule_version: int = 1
    participants: tuple[ParticipantDescriptor, ...] = field(default_factory=tuple)
    forced_pass_positions: int = 0
    inference_batches: int = 0
    inference_positions: int = 0
    max_inference_batch: int = 0

    @property
    def p1_result(self) -> int:
        return WDL_DRAW if self.is_draw else (WDL_WIN if self.winner == 1 else WDL_LOSS)

    @property
    def turn_count(self) -> int:
        return len(self.moves)

    @property
    def placement_count(self) -> int:
        return sum(move.turn_kind == TurnKind.PLACE.value for move in self.moves)


def derive_game_seed(run_seed: int, game_id: int) -> int:
    if run_seed < 0 or game_id < 0:
        raise ValueError("run_seed and game_id must be non-negative.")
    return int((int(run_seed) + int(game_id)) % (2**63 - 1))


def _position_rng(game_seed: int, turn_index: int, stream: int) -> np.random.Generator:
    low = game_seed & 0xFFFFFFFF
    high = (game_seed >> 32) & 0xFFFFFFFF
    return np.random.default_rng(
        np.random.SeedSequence([low, high, int(turn_index), int(stream)])
    )


def _sample_action(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def _forced_full_search_placement(game_seed: int) -> int:
    # No four-in-a-row game can terminate before the seventh placement.
    return int(_position_rng(game_seed, 0, 3).integers(7))


def wdl_target_for_player(winner: int, player: int) -> int:
    if winner not in (-1, 0, 1) or player not in (-1, 1):
        raise ValueError("winner must be -1/0/1 and player must be -1/1.")
    if winner == 0:
        return WDL_DRAW
    return WDL_WIN if player == winner else WDL_LOSS


def _label_samples(
    samples: list[SelfPlaySample],
    moves: list[MoveRecord],
    *,
    winner: int,
    terminal_board: np.ndarray,
) -> tuple[SelfPlaySample, ...]:
    if len(samples) != len(moves):
        raise RuntimeError("Every tagged turn must have exactly one Replay V2 position.")
    total_turns = len(moves)
    labeled: list[SelfPlaySample] = []
    for index, sample in enumerate(samples):
        reply_column = -1
        reply_mask = 0
        if index + 1 < total_turns:
            reply = moves[index + 1]
            if reply.turn_kind == TurnKind.PLACE.value:
                assert reply.column is not None
                reply_column = int(reply.column)
                reply_mask = 1
        labeled.append(
            replace(
                sample,
                wdl=wdl_target_for_player(winner, sample.player),
                opponent_reply_column=reply_column,
                opponent_reply_mask=reply_mask,
                terminal_board=np.asarray(terminal_board, dtype=np.int8),
                remaining_turns=total_turns - sample.ply,
            )
        )
    return tuple(labeled)


def _participants_for_producer(producer_model_id: str) -> tuple[ParticipantDescriptor, ...]:
    controller_type = "random" if producer_model_id == "random" else "model"
    model_id = None if controller_type == "random" else producer_model_id
    return tuple(
        ParticipantDescriptor(
            seat=seat,
            player=player,
            controller_type=controller_type,
            controller_id=producer_model_id,
            display_name=producer_model_id,
            model_id=model_id,
        )
        for seat, player in (("FIRST", 1), ("SECOND", -1))
    )


def run_self_play_game(
    selfplay_config: SelfPlayConfig,
    *,
    run_seed: int,
    game_id: int,
    generation: int,
    predictor: Predictor,
    producer_model_id: str,
    mcts_lanes: int = 1,
    force_full_search_before_ply: int = 0,
) -> GameRecord:
    if mcts_lanes < 1:
        raise ValueError("mcts_lanes must be positive")
    if force_full_search_before_ply < 0:
        raise ValueError("force_full_search_before_ply must be non-negative")
    exploration_variant = selfplay_config.exploration_variant_for_game(game_id)
    effective_selfplay = selfplay_config.for_exploration_variant(exploration_variant)
    force_full_search_before_ply = max(
        int(force_full_search_before_ply),
        int(effective_selfplay.opening_full_search_plies),
    )
    search_stage = effective_selfplay.stage_for_generation(generation)
    rule_spec = DEFAULT_RULE_REGISTRY.get(effective_selfplay.rule_id)
    engine = RuleEngine(rule_spec)
    game_seed = derive_game_seed(run_seed, game_id)
    state = engine.initial_state()
    pending_samples: list[SelfPlaySample] = []
    moves: list[MoveRecord] = []
    full_count = 0
    fast_count = 0
    pass_count = 0
    total_simulations = 0
    inference_batches = 0
    inference_positions = 0
    max_inference_batch = 0
    forced_full_placement = _forced_full_search_placement(game_seed)

    while not state.terminal:
        if state.turn_index >= MAX_GAME_TURNS:
            raise RuntimeError(
                f"Self-play game {game_id} exceeded the {MAX_GAME_TURNS}-turn invariant."
            )
        canonical = canonical_board_for_state(state)
        required = engine.required_action(state)
        if required is not None:
            pending_samples.append(
                SelfPlaySample(
                    board=canonical,
                    visit_counts=np.zeros(25, dtype=np.uint32),
                    wdl=WDL_DRAW,
                    game_id=game_id,
                    ply=state.turn_index,
                    player=state.player_to_move,
                    search_kind=SEARCH_NONE,
                    policy_weight=0.0,
                    rule_code=rule_spec.rule_code,
                    turn_kind=TurnKind.FORCED_PASS.value,
                    placement_count=state.placement_count,
                )
            )
            moves.append(
                MoveRecord(
                    ply=state.turn_index,
                    player=state.player_to_move,
                    column=None,
                    legacy_action=None,
                    search_kind=SEARCH_NONE,
                    simulations=0,
                    turn_kind=TurnKind.FORCED_PASS.value,
                    placement_count=state.placement_count,
                )
            )
            state = engine.step(state, required)
            pass_count += 1
            continue

        budget_rng = _position_rng(game_seed, state.turn_index, 0)
        is_full = (
            state.turn_index < force_full_search_before_ply
            or state.placement_count == forced_full_placement
            or float(budget_rng.random()) < search_stage.full_probability
        )
        search_kind = SEARCH_FULL if is_full else SEARCH_FAST
        simulations = (
            search_stage.full_search_sims if is_full else search_stage.fast_search_sims
        )
        exploration = effective_selfplay.exploration_for_ply(state.turn_index)
        search = MCTS(
            predictor,
            engine=engine,
            cpuct=effective_selfplay.cpuct,
            virtual_loss=effective_selfplay.virtual_loss,
            num_threads=mcts_lanes,
        )
        result = search.search(
            state,
            simulations,
            rng=_position_rng(game_seed, state.turn_index, 1),
            add_root_noise=exploration.dirichlet_epsilon > 0.0,
            dirichlet_alpha=exploration.dirichlet_alpha,
            dirichlet_epsilon=exploration.dirichlet_epsilon,
        )
        total_simulations += simulations
        inference_batches += result.inference_calls
        inference_positions += result.inference_positions
        max_inference_batch = max(max_inference_batch, result.max_inference_batch)
        if is_full:
            full_count += 1
        else:
            fast_count += 1
        pending_samples.append(
            SelfPlaySample(
                board=canonical,
                visit_counts=result.visit_counts,
                wdl=WDL_DRAW,
                game_id=game_id,
                ply=state.turn_index,
                player=state.player_to_move,
                search_kind=search_kind,
                policy_weight=1.0 if is_full else 0.0,
                rule_code=rule_spec.rule_code,
                turn_kind=TurnKind.PLACE.value,
                placement_count=state.placement_count,
            )
        )

        valid_mask = engine.legal_column_mask(state).astype(bool)
        action_policy = policy_from_visits(
            result.visit_counts,
            temperature=exploration.temperature,
            valid_mask=valid_mask,
        )
        column = _sample_action(
            action_policy, _position_rng(game_seed, state.turn_index, 2)
        )
        legacy_action = engine.legacy_action_for_column(state, column)
        moves.append(
            MoveRecord(
                ply=state.turn_index,
                player=state.player_to_move,
                column=column,
                legacy_action=legacy_action,
                search_kind=search_kind,
                simulations=simulations,
                turn_kind=TurnKind.PLACE.value,
                placement_count=state.placement_count,
            )
        )
        state = engine.step(state, TurnAction.place(column))

    if state.outcome == GameOutcome.ONGOING:
        raise RuntimeError("Self-play stopped without a terminal rule outcome.")
    winner = state.outcome.winner or 0
    if full_count < 1 or not pending_samples:
        raise RuntimeError("Every self-play game must contribute at least one full-search sample.")
    return GameRecord(
        game_id=game_id,
        seed=game_seed,
        generation=generation,
        producer_model_id=producer_model_id,
        winner=winner,
        is_draw=winner == 0,
        moves=tuple(moves),
        samples=_label_samples(
            pending_samples,
            moves,
            winner=winner,
            terminal_board=state.board,
        ),
        full_search_positions=full_count,
        fast_search_positions=fast_count,
        forced_pass_positions=pass_count,
        total_simulations=total_simulations,
        exploration_variant=exploration_variant,
        rule_id=rule_spec.rule_id,
        rule_code=rule_spec.rule_code,
        rule_version=rule_spec.rule_version,
        participants=_participants_for_producer(producer_model_id),
        inference_batches=inference_batches,
        inference_positions=inference_positions,
        max_inference_batch=max_inference_batch,
    )


def run_self_play_games(
    config: V3Config,
    *,
    accepted_predictor: Predictor | None = None,
    producer_model_id: str | None = None,
    start_game_id: int = 0,
    generation: int = 0,
) -> list[GameRecord]:
    """Generate games from random bootstrap or one committed accepted model."""

    if not isinstance(config, V3Config):
        raise TypeError("run_self_play_games requires a resolved V3Config.")
    if start_game_id < 0 or generation < 0:
        raise ValueError("start_game_id and generation must be non-negative.")
    if accepted_predictor is None:
        if producer_model_id not in (None, "random"):
            raise ValueError("A non-random producer_model_id requires an accepted predictor.")
        predictor: Predictor = RandomPredictor()
        producer = "random"
    else:
        if not producer_model_id or producer_model_id == "random":
            raise ValueError("An accepted predictor requires its non-random producer_model_id.")
        predictor = accepted_predictor
        producer = str(producer_model_id)

    search_stage = config.selfplay.stage_for_generation(generation)
    return [
        run_self_play_game(
            config.selfplay,
            run_seed=config.run.seed,
            game_id=start_game_id + offset,
            generation=generation,
            predictor=predictor,
            producer_model_id=producer,
            mcts_lanes=config.runtime.mcts_lanes_per_actor,
        )
        for offset in range(search_stage.games)
    ]


__all__ = [
    "GameRecord",
    "MAX_GAME_MOVES",
    "MAX_GAME_TURNS",
    "MoveRecord",
    "ParticipantDescriptor",
    "SelfPlaySample",
    "derive_game_seed",
    "run_self_play_game",
    "run_self_play_games",
    "wdl_target_for_player",
]
