"""Operational game-parallel runtime for deterministic paired evaluation.

Parallelism is across independent games. Each game's search simulations,
cpuct, seed, opening, role swap, and single-lane MCTS semantics are unchanged.
Model requests from those games are combined by one batching thread per model.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from .evaluation import Opening, play_paired_game
from .gate import GateGameResult
from .model import COLUMN_COUNT, ROLE_FEATURE_COUNT, RULE_FEATURE_COUNT
from .search import Predictor


@dataclass
class _InferenceRequest:
    boards: np.ndarray
    roles: np.ndarray
    rules: np.ndarray
    complete: threading.Event = field(default_factory=threading.Event)
    policies: np.ndarray | None = None
    values: np.ndarray | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationInferenceMetrics:
    service_id: str
    requests: int
    positions: int
    batches: int
    max_batch: int
    batch_limit: int
    batch_timeout_s: float
    wall_seconds: float

    @property
    def mean_batch(self) -> float:
        return self.positions / max(self.batches, 1)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "mean_batch": self.mean_batch}


@dataclass(frozen=True)
class PairedEvaluationMetrics:
    parallel_games: int
    games: int
    wall_seconds: float
    inference_services: tuple[EvaluationInferenceMetrics, ...]

    @property
    def games_per_second(self) -> float:
        return self.games / max(self.wall_seconds, 1e-12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parallel_games": self.parallel_games,
            "games": self.games,
            "wall_seconds": self.wall_seconds,
            "games_per_second": self.games_per_second,
            "inference_services": [row.to_dict() for row in self.inference_services],
        }


@dataclass(frozen=True)
class PairedEvaluationResult:
    games: tuple[GateGameResult, ...]
    metrics: PairedEvaluationMetrics


class BatchingPredictor:
    """Thread-safe blocking predictor backed by one cross-game batch worker."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        service_id: str,
        batch_limit: int,
        batch_timeout_s: float,
        response_timeout_s: float = 300.0,
    ) -> None:
        if not hasattr(predictor, "predict_batch"):
            raise TypeError("batched evaluation predictor must implement predict_batch")
        if not service_id or batch_limit < 1 or batch_timeout_s < 0.0:
            raise ValueError("invalid evaluation batching service settings")
        if response_timeout_s <= 0.0:
            raise ValueError("response_timeout_s must be positive")
        self.predictor = predictor
        self.service_id = service_id
        self.batch_limit = int(batch_limit)
        self.batch_timeout_s = float(batch_timeout_s)
        self.response_timeout_s = float(response_timeout_s)
        self._queue: queue.Queue[_InferenceRequest | None] = queue.Queue()
        self._failure_lock = threading.Lock()
        self._failure: str | None = None
        self._closed = False
        self._metrics: EvaluationInferenceMetrics | None = None
        self._thread = threading.Thread(
            target=self._worker,
            name=f"evaluation-inference-{service_id}",
            daemon=True,
        )
        self._thread.start()

    def predict(
        self,
        canonical_board: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        role = np.asarray(role_to_play)
        rules = np.asarray(rule_features)
        policies, values = self.predict_batch(
            np.asarray(canonical_board)[None, ...],
            role_to_play=role[None, ...] if role.ndim == 1 else role,
            rule_features=rules[None, ...] if rules.ndim == 1 else rules,
        )
        return policies[0], values[0]

    def predict_batch(
        self,
        canonical_boards: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        boards = np.asarray(canonical_boards, dtype=np.int8)
        roles = np.asarray(role_to_play, dtype=np.float32)
        rules = np.asarray(rule_features, dtype=np.float32)
        if boards.ndim != 4 or boards.shape[1:] != (6, 5, 5) or len(boards) < 1:
            raise ValueError("evaluation inference boards must have shape [N,6,5,5]")
        if roles.shape != (len(boards), ROLE_FEATURE_COUNT):
            raise ValueError("evaluation inference role shape mismatch")
        if rules.shape != (len(boards), RULE_FEATURE_COUNT):
            raise ValueError("evaluation inference rule shape mismatch")
        if len(boards) > self.batch_limit:
            raise ValueError("one evaluation request exceeds the hard batch limit")
        with self._failure_lock:
            if self._closed:
                raise RuntimeError("evaluation inference service is closed")
            if self._failure is not None:
                raise RuntimeError(f"evaluation inference service failed: {self._failure}")
            request = _InferenceRequest(
                boards=np.array(boards, copy=True),
                roles=np.array(roles, copy=True),
                rules=np.array(rules, copy=True),
            )
            self._queue.put(request)
        if not request.complete.wait(timeout=self.response_timeout_s):
            raise RuntimeError(f"evaluation inference timed out in {self.service_id}")
        if request.error is not None:
            raise RuntimeError(f"evaluation inference failed: {request.error}")
        assert request.policies is not None and request.values is not None
        return request.policies, request.values

    def _worker(self) -> None:
        started = time.perf_counter()
        requests = positions = batches = max_batch = 0
        deferred: _InferenceRequest | None = None
        active: list[_InferenceRequest] = []
        try:
            stopping = False
            while not stopping:
                if deferred is None:
                    message = self._queue.get()
                else:
                    message = deferred
                    deferred = None
                if message is None:
                    break
                active = [message]
                active_positions = len(message.boards)
                deadline = time.perf_counter() + self.batch_timeout_s
                while active_positions < self.batch_limit:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    try:
                        following = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if following is None:
                        stopping = True
                        break
                    following_positions = len(following.boards)
                    if active_positions + following_positions > self.batch_limit:
                        deferred = following
                        break
                    active.append(following)
                    active_positions += following_positions

                boards = np.concatenate([row.boards for row in active], axis=0)
                roles = np.concatenate([row.roles for row in active], axis=0)
                rules = np.concatenate([row.rules for row in active], axis=0)
                policies, values = self.predictor.predict_batch(
                    boards,
                    role_to_play=roles,
                    rule_features=rules,
                )
                policies = np.asarray(policies, dtype=np.float32)
                values = np.asarray(values, dtype=np.float32)
                if policies.shape != (len(boards), COLUMN_COUNT):
                    raise ValueError(
                        "evaluation predictor policy output must have shape "
                        f"[N,{COLUMN_COUNT}], got {policies.shape}"
                    )
                if values.ndim not in (1, 2) or values.shape[0] != len(boards):
                    raise ValueError(
                        "evaluation predictor value output must retain the batch dimension"
                    )
                if not np.all(np.isfinite(policies)) or not np.all(np.isfinite(values)):
                    raise ValueError("evaluation predictor returned non-finite outputs")
                offset = 0
                for row in active:
                    count = len(row.boards)
                    row.policies = policies[offset : offset + count]
                    row.values = values[offset : offset + count]
                    row.complete.set()
                    offset += count
                requests += len(active)
                positions += len(boards)
                batches += 1
                max_batch = max(max_batch, len(boards))
                active = []
        except BaseException:
            error = traceback.format_exc()
            with self._failure_lock:
                self._failure = error
            for row in active:
                row.error = error
                row.complete.set()
            while True:
                try:
                    row = self._queue.get_nowait()
                except queue.Empty:
                    break
                if row is not None:
                    row.error = error
                    row.complete.set()
        finally:
            self._metrics = EvaluationInferenceMetrics(
                service_id=self.service_id,
                requests=requests,
                positions=positions,
                batches=batches,
                max_batch=max_batch,
                batch_limit=self.batch_limit,
                batch_timeout_s=self.batch_timeout_s,
                wall_seconds=time.perf_counter() - started,
            )

    def close(self) -> EvaluationInferenceMetrics:
        with self._failure_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join(timeout=self.response_timeout_s)
        if self._thread.is_alive():
            raise RuntimeError(f"evaluation inference service {self.service_id} did not drain")
        assert self._metrics is not None
        return self._metrics


def play_paired_openings_parallel(
    openings: Iterable[Opening],
    *,
    candidate_predictor: Predictor,
    incumbent_predictor: Predictor | None,
    search_sims: int,
    cpuct: float,
    parallel_games: int,
    inference_batch_size: int,
    inference_batch_timeout_s: float = 0.001,
    inference_response_timeout_s: float = 300.0,
) -> PairedEvaluationResult:
    """Run exact paired games concurrently and return topology evidence."""

    rows = tuple(openings)
    if not rows:
        raise ValueError("paired evaluation needs at least one opening")
    if parallel_games < 1 or inference_batch_size < 1:
        raise ValueError("evaluation parallelism and batch size must be positive")
    rule_contexts = {(opening.rule_id, opening.rule_version) for opening in rows}
    if len(rule_contexts) != 1:
        raise ValueError("paired evaluation openings must use exactly one rule context")
    candidate_service = BatchingPredictor(
        candidate_predictor,
        service_id="candidate",
        batch_limit=inference_batch_size,
        batch_timeout_s=inference_batch_timeout_s,
        response_timeout_s=inference_response_timeout_s,
    )
    incumbent_service = (
        None
        if incumbent_predictor is None
        else BatchingPredictor(
            incumbent_predictor,
            service_id="incumbent",
            batch_limit=inference_batch_size,
            batch_timeout_s=inference_batch_timeout_s,
            response_timeout_s=inference_response_timeout_s,
        )
    )
    started = time.perf_counter()
    tasks = tuple((opening, role) for opening in rows for role in (True, False))
    metrics: list[EvaluationInferenceMetrics] = []
    try:
        with ThreadPoolExecutor(
            max_workers=min(parallel_games, len(tasks)),
            thread_name_prefix="paired-evaluation-game",
        ) as executor:
            futures = [
                executor.submit(
                    play_paired_game,
                    opening,
                    candidate_is_first=candidate_is_first,
                    candidate_predictor=candidate_service,
                    incumbent_predictor=incumbent_service,
                    search_sims=search_sims,
                    cpuct=cpuct,
                )
                for opening, candidate_is_first in tasks
            ]
            games = tuple(future.result() for future in futures)
    finally:
        metrics.append(candidate_service.close())
        if incumbent_service is not None:
            metrics.append(incumbent_service.close())
    return PairedEvaluationResult(
        games=games,
        metrics=PairedEvaluationMetrics(
            parallel_games=min(parallel_games, len(tasks)),
            games=len(games),
            wall_seconds=time.perf_counter() - started,
            inference_services=tuple(metrics),
        ),
    )


__all__ = [
    "BatchingPredictor",
    "EvaluationInferenceMetrics",
    "PairedEvaluationMetrics",
    "PairedEvaluationResult",
    "play_paired_openings_parallel",
]
