from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from connect4_core import BOARD_SHAPE, GameRules

from .config import SelfPlayConfig, V3Config
from .model import (
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
    column_to_legacy_action,
    legal_column_mask,
)
from .search import MCTS, Predictor, RandomPredictor, SEARCH_FAST, SEARCH_FULL, policy_from_visits


MAX_GAME_MOVES = int(np.prod(BOARD_SHAPE))


@dataclass(frozen=True)
class SelfPlaySample:
    board: np.ndarray
    visit_counts: np.ndarray
    wdl: int
    game_id: int
    ply: int
    player: int
    search_kind: str = SEARCH_FULL

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
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in (self.game_id, self.ply, self.player)):
            raise TypeError("Self-play game_id, ply, and player must be integers.")
        if int(self.game_id) < 0 or int(self.ply) < 0 or int(self.player) not in (-1, 1):
            raise ValueError("Invalid self-play game_id, ply, or player.")
        if self.search_kind not in (SEARCH_FAST, SEARCH_FULL):
            raise ValueError(f"Unknown search kind: {self.search_kind!r}.")
        object.__setattr__(self, "board", np.array(board, copy=True))
        object.__setattr__(self, "visit_counts", np.array(visits, copy=True))

    @property
    def canonical_board(self) -> np.ndarray:
        return self.board

    @property
    def wdl_target(self) -> int:
        return self.wdl


@dataclass(frozen=True)
class MoveRecord:
    ply: int
    player: int
    column: int
    legacy_action: int
    search_kind: str
    simulations: int


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
    inference_batches: int = 0
    inference_positions: int = 0
    max_inference_batch: int = 0

    @property
    def p1_result(self) -> int:
        return WDL_DRAW if self.is_draw else (WDL_WIN if self.winner == 1 else WDL_LOSS)


def derive_game_seed(run_seed: int, game_id: int) -> int:
    if run_seed < 0 or game_id < 0:
        raise ValueError("run_seed and game_id must be non-negative.")
    return int((int(run_seed) + int(game_id)) % (2**63 - 1))


def _position_rng(game_seed: int, ply: int, stream: int) -> np.random.Generator:
    low = game_seed & 0xFFFFFFFF
    high = (game_seed >> 32) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([low, high, int(ply), int(stream)]))


def _sample_action(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def _forced_full_search_ply(game_seed: int, max_moves: int) -> int:
    """Choose one early full-search position without always biasing ply zero.

    A Connect4 win from an empty board cannot occur before the seventh move, so
    every normal game reaches the selected member of ``[0, min(7, max_moves))``.
    The independent RNG stream keeps the choice stable across actor schedules.
    """

    candidate_count = min(7, int(max_moves))
    if candidate_count < 1:
        raise ValueError("max_moves must leave at least one potential position")
    return int(_position_rng(game_seed, 0, 3).integers(candidate_count))


def _winner_from_terminal_board(game: GameRules, board: np.ndarray) -> int:
    if game.check_win(board, 1):
        return 1
    if game.check_win(board, -1):
        return -1
    if not np.any(board == 0):
        return 0
    raise RuntimeError("Terminal result did not contain a winner or a full-board draw.")


def wdl_target_for_player(winner: int, player: int) -> int:
    if winner not in (-1, 0, 1) or player not in (-1, 1):
        raise ValueError("winner must be -1/0/1 and player must be -1/1.")
    if winner == 0:
        return WDL_DRAW
    return WDL_WIN if player == winner else WDL_LOSS


def _label_samples(samples: list[SelfPlaySample], winner: int) -> tuple[SelfPlaySample, ...]:
    labeled: list[SelfPlaySample] = []
    for sample in samples:
        labeled.append(replace(sample, wdl=wdl_target_for_player(winner, sample.player)))
    return tuple(labeled)


def run_self_play_game(
    selfplay_config: SelfPlayConfig,
    *,
    run_seed: int,
    game_id: int,
    generation: int,
    predictor: Predictor,
    producer_model_id: str,
    mcts_lanes: int = 1,
) -> GameRecord:
    if mcts_lanes < 1:
        raise ValueError("mcts_lanes must be positive")
    search_stage = selfplay_config.stage_for_generation(generation)
    game = GameRules()
    game_seed = derive_game_seed(run_seed, game_id)
    board = game.get_init_board()
    player = 1
    pending_samples: list[SelfPlaySample] = []
    moves: list[MoveRecord] = []
    full_count = 0
    fast_count = 0
    total_simulations = 0
    inference_batches = 0
    inference_positions = 0
    max_inference_batch = 0
    winner: int | None = None
    forced_full_ply = _forced_full_search_ply(game_seed, MAX_GAME_MOVES)

    for ply in range(MAX_GAME_MOVES):
        canonical = game.get_canonical_form(board, player)
        budget_rng = _position_rng(game_seed, ply, 0)
        is_full = (
            ply == forced_full_ply
            or float(budget_rng.random()) < search_stage.full_probability
        )
        search_kind = SEARCH_FULL if is_full else SEARCH_FAST
        simulations = search_stage.full_search_sims if is_full else search_stage.fast_search_sims
        exploration = selfplay_config.exploration_for_ply(ply)
        search = MCTS(
            predictor,
            cpuct=selfplay_config.cpuct,
            virtual_loss=selfplay_config.virtual_loss,
            num_threads=mcts_lanes,
            game=game,
        )
        result = search.search(
            canonical,
            simulations,
            rng=_position_rng(game_seed, ply, 1),
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
                board=np.asarray(canonical, dtype=np.int8),
                visit_counts=result.visit_counts,
                wdl=WDL_DRAW,
                game_id=game_id,
                ply=ply,
                player=player,
                search_kind=search_kind,
            )
        )

        action_policy = policy_from_visits(
            result.visit_counts,
            temperature=exploration.temperature,
            valid_mask=legal_column_mask(canonical).reshape(-1),
        )
        column = _sample_action(action_policy, _position_rng(game_seed, ply, 2))
        legacy_action = column_to_legacy_action(board, column)
        moves.append(
            MoveRecord(
                ply=ply,
                player=player,
                column=column,
                legacy_action=legacy_action,
                search_kind=search_kind,
                simulations=simulations,
            )
        )
        board, player = game.get_next_state(board, player, legacy_action)
        terminal_result = float(game.get_game_ended(board, player))
        if terminal_result != 0.0:
            if math.isclose(terminal_result, 1e-4, rel_tol=0.0, abs_tol=1e-9):
                winner = 0
            else:
                winner = _winner_from_terminal_board(game, board)
            break

    if winner is None:
        raise RuntimeError(
            f"Self-play game {game_id} reached the {MAX_GAME_MOVES}-move board capacity "
            "before a terminal result."
        )
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
        samples=_label_samples(pending_samples, winner),
        full_search_positions=full_count,
        fast_search_positions=fast_count,
        total_simulations=total_simulations,
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
    """Generate deterministic games from random bootstrap or one accepted model.

    The predictor is fixed for the whole call and each game constructs its search
    at the game boundary. Candidate models are intentionally not an input to this
    API, preventing rejected candidates from producing replay data.
    """
    if not isinstance(config, V3Config):
        raise TypeError("run_self_play_games requires a resolved V3Config.")
    if start_game_id < 0:
        raise ValueError("start_game_id must be non-negative.")
    if generation < 0:
        raise ValueError("generation must be non-negative.")
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
    "MoveRecord",
    "SelfPlaySample",
    "derive_game_seed",
    "run_self_play_game",
    "run_self_play_games",
    "wdl_target_for_player",
]
