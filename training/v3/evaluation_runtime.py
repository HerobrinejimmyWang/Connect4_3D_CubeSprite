"""Operational game-parallel runtime for deterministic paired evaluation.

Parallelism is across independent games. Each game's search simulations,
cpuct, seed, opening, role swap, and single-lane MCTS semantics are unchanged.
Model requests from those games are combined by one batching thread per model.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from .actor_runtime import RemotePredictor
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
    worker_processes: int
    start_method: str
    games: int
    wall_seconds: float
    inference_services: tuple[EvaluationInferenceMetrics, ...]
    replicated_workers: tuple["ReplicatedWorkerMetrics", ...] = ()

    @property
    def games_per_second(self) -> float:
        return self.games / max(self.wall_seconds, 1e-12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parallel_games": self.parallel_games,
            "worker_processes": self.worker_processes,
            "start_method": self.start_method,
            "games": self.games,
            "wall_seconds": self.wall_seconds,
            "games_per_second": self.games_per_second,
            "inference_services": [row.to_dict() for row in self.inference_services],
            "replicated_workers": [row.to_dict() for row in self.replicated_workers],
        }


@dataclass(frozen=True)
class PairedEvaluationResult:
    games: tuple[GateGameResult, ...]
    metrics: PairedEvaluationMetrics


@dataclass(frozen=True)
class EvaluationTask:
    task_index: int
    opening: Opening
    candidate_is_first: bool


@dataclass(frozen=True)
class EvaluationModelSource:
    kind: str
    path: str
    model_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"v3_artifact", "legacy_artifact", "random"}:
            raise ValueError("evaluation model source kind is unsupported")
        if (self.kind != "random" and not self.path) or not self.model_id:
            raise ValueError("evaluation model source needs path and model_id")

    @classmethod
    def from_identity(cls, identity: dict[str, Any]) -> "EvaluationModelSource":
        lineage = str(identity.get("lineage", ""))
        kind = "legacy_artifact" if lineage == "external_legacy_anchor" else "v3_artifact"
        return cls(kind, str(identity.get("path", "")), str(identity.get("model_id", "")))


@dataclass(frozen=True)
class ReplicatedWorkerMetrics:
    worker_id: int
    device: str
    opening_pairs: int
    games: int
    model_load_seconds: float
    play_seconds: float
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


class _MultiprocessBatchingService:
    """Parent-owned model service for independently spawned game workers."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        service_id: str,
        actor_ids: tuple[int, ...],
        request_queue: Any,
        response_queues: dict[int, Any],
        batch_limit: int,
        batch_timeout_s: float,
        response_timeout_s: float,
    ) -> None:
        self.predictor = predictor
        self.service_id = service_id
        self.actor_ids = actor_ids
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.batch_limit = batch_limit
        self.batch_timeout_s = batch_timeout_s
        self.response_timeout_s = response_timeout_s
        self.failure: str | None = None
        self.metrics: EvaluationInferenceMetrics | None = None
        self.thread = threading.Thread(
            target=self._worker,
            name=f"evaluation-mp-inference-{service_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _worker(self) -> None:
        started = time.perf_counter()
        requests = positions = batches = max_batch = 0
        active: list[tuple[Any, ...]] = []
        deferred: tuple[Any, ...] | None = None
        try:
            stopping = False
            while not stopping:
                if deferred is None:
                    message = self.request_queue.get()
                else:
                    message = deferred
                    deferred = None
                if message is None:
                    break
                active = [message]
                active_positions = len(message[2])
                if active_positions > self.batch_limit:
                    raise ValueError("one evaluation request exceeds the hard batch limit")
                deadline = time.perf_counter() + self.batch_timeout_s
                while active_positions < self.batch_limit:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    try:
                        following = self.request_queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if following is None:
                        stopping = True
                        break
                    following_positions = len(following[2])
                    if active_positions + following_positions > self.batch_limit:
                        deferred = following
                        break
                    active.append(following)
                    active_positions += following_positions

                boards = np.concatenate([row[2] for row in active], axis=0)
                roles = np.concatenate([row[3] for row in active], axis=0)
                rules = np.concatenate([row[4] for row in active], axis=0)
                policies, values = self.predictor.predict_batch(
                    boards,
                    role_to_play=roles,
                    rule_features=rules,
                )
                policies = np.asarray(policies, dtype=np.float32)
                values = np.asarray(values, dtype=np.float32)
                if policies.shape != (len(boards), COLUMN_COUNT):
                    raise ValueError("evaluation predictor returned an invalid policy batch")
                if values.ndim not in (1, 2) or values.shape[0] != len(boards):
                    raise ValueError("evaluation predictor returned an invalid value batch")
                if not np.all(np.isfinite(policies)) or not np.all(np.isfinite(values)):
                    raise ValueError("evaluation predictor returned non-finite outputs")
                offset = 0
                for actor_id, request_id, request_boards, _request_roles, _request_rules in active:
                    count = len(request_boards)
                    self.response_queues[int(actor_id)].put(
                        (
                            int(request_id),
                            policies[offset : offset + count],
                            values[offset : offset + count],
                            None,
                        )
                    )
                    offset += count
                requests += len(active)
                positions += len(boards)
                batches += 1
                max_batch = max(max_batch, len(boards))
                active = []
        except BaseException:
            self.failure = traceback.format_exc()
            for actor_id in self.actor_ids:
                try:
                    self.response_queues[actor_id].put(
                        (-1, None, None, self.failure), timeout=0.1
                    )
                except (queue.Full, OSError):
                    pass
        finally:
            self.metrics = EvaluationInferenceMetrics(
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
        if self.thread.is_alive():
            self.request_queue.put(None)
        self.thread.join(timeout=self.response_timeout_s)
        if self.thread.is_alive():
            raise RuntimeError(f"evaluation inference service {self.service_id} did not drain")
        if self.failure is not None:
            raise RuntimeError(f"evaluation inference service failed: {self.failure}")
        assert self.metrics is not None
        return self.metrics


def _evaluation_actor_main(
    actor_id: int,
    task_queue: Any,
    result_queue: Any,
    candidate_request_queue: Any,
    candidate_response_queue: Any,
    incumbent_request_queue: Any | None,
    incumbent_response_queue: Any | None,
    search_sims: int,
    cpuct: float,
    response_timeout_s: float,
) -> None:
    try:
        candidate = RemotePredictor(
            actor_id,
            candidate_request_queue,
            candidate_response_queue,
            response_timeout_s=response_timeout_s,
        )
        incumbent = (
            None
            if incumbent_request_queue is None
            else RemotePredictor(
                actor_id,
                incumbent_request_queue,
                incumbent_response_queue,
                response_timeout_s=response_timeout_s,
            )
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            if not isinstance(task, EvaluationTask):
                raise TypeError("evaluation actor received an invalid task")
            game = play_paired_game(
                task.opening,
                candidate_is_first=task.candidate_is_first,
                candidate_predictor=candidate,
                incumbent_predictor=incumbent,
                search_sims=search_sims,
                cpuct=cpuct,
            )
            result_queue.put(("game", actor_id, task.task_index, game))
    except BaseException:
        result_queue.put(("error", actor_id, None, traceback.format_exc()))


def _terminate_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


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
    start_method: str = "spawn",
) -> PairedEvaluationResult:
    """Run exact paired games concurrently and return topology evidence."""

    rows = tuple(openings)
    if not rows:
        raise ValueError("paired evaluation needs at least one opening")
    if parallel_games < 1 or inference_batch_size < 1:
        raise ValueError("evaluation parallelism and batch size must be positive")
    if inference_batch_timeout_s < 0.0 or inference_response_timeout_s <= 0.0:
        raise ValueError("evaluation inference timeouts are invalid")
    rule_contexts = {(opening.rule_id, opening.rule_version) for opening in rows}
    if len(rule_contexts) != 1:
        raise ValueError("paired evaluation openings must use exactly one rule context")
    game_specs = tuple(
        (opening, candidate_is_first)
        for opening in rows
        for candidate_is_first in (True, False)
    )
    tasks = tuple(
        EvaluationTask(index, opening, candidate_is_first)
        for index, (opening, candidate_is_first) in enumerate(game_specs)
    )
    actor_count = min(parallel_games, len(tasks))
    context = mp.get_context(start_method)
    queue_capacity = max(1, 2 * actor_count)
    task_queue = context.Queue(maxsize=queue_capacity)
    result_queue = context.Queue(maxsize=queue_capacity)
    candidate_request_queue = context.Queue(maxsize=queue_capacity)
    candidate_response_queues = {
        actor_id: context.Queue(maxsize=2) for actor_id in range(actor_count)
    }
    incumbent_request_queue = (
        None if incumbent_predictor is None else context.Queue(maxsize=queue_capacity)
    )
    incumbent_response_queues = (
        {}
        if incumbent_predictor is None
        else {actor_id: context.Queue(maxsize=2) for actor_id in range(actor_count)}
    )
    actor_ids = tuple(range(actor_count))
    services = [
        _MultiprocessBatchingService(
            candidate_predictor,
            service_id="candidate",
            actor_ids=actor_ids,
            request_queue=candidate_request_queue,
            response_queues=candidate_response_queues,
            batch_limit=inference_batch_size,
            batch_timeout_s=inference_batch_timeout_s,
            response_timeout_s=inference_response_timeout_s,
        )
    ]
    if incumbent_predictor is not None:
        services.append(
            _MultiprocessBatchingService(
                incumbent_predictor,
                service_id="incumbent",
                actor_ids=actor_ids,
                request_queue=incumbent_request_queue,
                response_queues=incumbent_response_queues,
                batch_limit=inference_batch_size,
                batch_timeout_s=inference_batch_timeout_s,
                response_timeout_s=inference_response_timeout_s,
            )
        )
    actors: list[mp.Process] = []
    started = time.perf_counter()
    service_metrics: list[EvaluationInferenceMetrics] = []
    try:
        for service in services:
            service.start()
        for actor_id in actor_ids:
            actor = context.Process(
                target=_evaluation_actor_main,
                args=(
                    actor_id,
                    task_queue,
                    result_queue,
                    candidate_request_queue,
                    candidate_response_queues[actor_id],
                    incumbent_request_queue,
                    incumbent_response_queues.get(actor_id),
                    search_sims,
                    cpuct,
                    inference_response_timeout_s,
                ),
                name=f"v3-evaluation-game-{actor_id}",
            )
            actor.start()
            actors.append(actor)

        next_task = 0
        for _ in range(min(len(tasks), queue_capacity)):
            task_queue.put(tasks[next_task])
            next_task += 1
        games_by_index: dict[int, GateGameResult] = {}
        while len(games_by_index) < len(tasks):
            try:
                kind, actor_id, task_index, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                failed_actors = [actor for actor in actors if actor.exitcode not in (None, 0)]
                failed_services = [service for service in services if service.failure is not None]
                if failed_actors or failed_services:
                    raise RuntimeError("V3 evaluation worker or inference service failed")
                continue
            if kind == "error":
                raise RuntimeError(f"V3 evaluation actor {actor_id} failed:\n{payload}")
            if kind != "game" or not isinstance(payload, GateGameResult):
                raise RuntimeError("V3 evaluation actor returned an invalid result")
            index = int(task_index)
            if index in games_by_index:
                raise RuntimeError(f"duplicate V3 evaluation result for task {index}")
            games_by_index[index] = payload
            if next_task < len(tasks):
                task_queue.put(tasks[next_task])
                next_task += 1

        for _ in actors:
            task_queue.put(None)
        for actor in actors:
            actor.join(timeout=30.0)
            if actor.exitcode != 0:
                raise RuntimeError(f"V3 evaluation actor {actor.name} exited with {actor.exitcode}")
        for service in services:
            service_metrics.append(service.close())
        games = tuple(games_by_index[index] for index in range(len(tasks)))
        return PairedEvaluationResult(
            games=games,
            metrics=PairedEvaluationMetrics(
                parallel_games=actor_count,
                worker_processes=actor_count,
                start_method=start_method,
                games=len(games),
                wall_seconds=time.perf_counter() - started,
                inference_services=tuple(service_metrics),
            ),
        )
    finally:
        _terminate_processes(actors)
        for service in services:
            if service.thread.is_alive():
                try:
                    service.request_queue.put(None)
                except (OSError, ValueError):
                    pass
                service.thread.join(timeout=5.0)
        managed_queues = [
            task_queue,
            result_queue,
            candidate_request_queue,
            *candidate_response_queues.values(),
            *incumbent_response_queues.values(),
        ]
        if incumbent_request_queue is not None:
            managed_queues.append(incumbent_request_queue)
        for managed_queue in managed_queues:
            try:
                managed_queue.close()
                managed_queue.join_thread()
            except (AttributeError, OSError, ValueError):
                pass


def _load_replicated_predictor(source: EvaluationModelSource, device: str) -> Predictor:
    # Local import avoids making the anchored compatibility layer a dependency
    # of the ordinary serial evaluation module.
    from .anchored_elo import LegacyCheckpointPredictor, load_v3_artifact_predictor
    from .search import RandomPredictor

    if source.kind == "random":
        return RandomPredictor()
    if source.kind == "legacy_artifact":
        return LegacyCheckpointPredictor(source.path, device=device)
    predictor, identity = load_v3_artifact_predictor(source.path, device=device)
    if identity["model_id"] != source.model_id:
        # A benchmark label may rename a V3 target, but the artifact itself must
        # still be valid; identity is recorded by the caller's immutable batch.
        identity["model_id"] = source.model_id
    return predictor


def _replicated_worker_main(
    worker_id: int,
    device: str,
    task_queue: Any,
    result_queue: Any,
    candidate_source: EvaluationModelSource,
    incumbent_source: EvaluationModelSource | None,
    search_sims: int,
    cpuct: float,
) -> None:
    started = time.perf_counter()
    pairs = games = 0
    try:
        import torch

        torch.set_num_threads(1)
        load_started = time.perf_counter()
        candidate = _load_replicated_predictor(candidate_source, device)
        incumbent = (
            None
            if incumbent_source is None
            else _load_replicated_predictor(incumbent_source, device)
        )
        load_seconds = time.perf_counter() - load_started
        play_started = time.perf_counter()
        while True:
            task = task_queue.get()
            if task is None:
                break
            opening_index, opening = task
            first = play_paired_game(
                opening,
                candidate_is_first=True,
                candidate_predictor=candidate,
                incumbent_predictor=incumbent,
                search_sims=search_sims,
                cpuct=cpuct,
            )
            second = play_paired_game(
                opening,
                candidate_is_first=False,
                candidate_predictor=candidate,
                incumbent_predictor=incumbent,
                search_sims=search_sims,
                cpuct=cpuct,
            )
            pairs += 1
            games += 2
            result_queue.put(("pair", worker_id, int(opening_index), (first, second)))
        metrics = ReplicatedWorkerMetrics(
            worker_id=worker_id,
            device=device,
            opening_pairs=pairs,
            games=games,
            model_load_seconds=load_seconds,
            play_seconds=time.perf_counter() - play_started,
            wall_seconds=time.perf_counter() - started,
        )
        result_queue.put(("stopped", worker_id, None, metrics))
    except BaseException:
        result_queue.put(("error", worker_id, None, traceback.format_exc()))


def play_paired_openings_replicated(
    openings: Iterable[Opening],
    *,
    candidate_source: EvaluationModelSource,
    incumbent_source: EvaluationModelSource | None,
    search_sims: int,
    cpuct: float,
    worker_devices: Iterable[str],
    start_method: str = "spawn",
) -> PairedEvaluationResult:
    """Evaluate opening pairs in device-local serial worker replicas.

    Each worker owns its model instances and runs ordinary single-lane games.
    There is no per-move IPC or cross-game inference batching, so search and
    floating-point execution match the serial evaluator on that device.
    """

    rows = tuple(openings)
    devices = tuple(str(device) for device in worker_devices)
    if not rows or not devices:
        raise ValueError("replicated evaluation needs openings and worker devices")
    if search_sims < 1 or cpuct <= 0.0:
        raise ValueError("replicated evaluation search settings are invalid")
    rule_contexts = {(opening.rule_id, opening.rule_version) for opening in rows}
    if len(rule_contexts) != 1:
        raise ValueError("replicated evaluation openings need one rule context")
    worker_count = min(len(devices), len(rows))
    devices = devices[:worker_count]
    context = mp.get_context(start_method)
    task_queue = context.Queue(maxsize=len(rows) + worker_count)
    result_queue = context.Queue(maxsize=max(1, 2 * worker_count))
    workers: list[mp.Process] = []
    started = time.perf_counter()
    try:
        for worker_id, device in enumerate(devices):
            worker = context.Process(
                target=_replicated_worker_main,
                args=(
                    worker_id,
                    device,
                    task_queue,
                    result_queue,
                    candidate_source,
                    incumbent_source,
                    search_sims,
                    cpuct,
                ),
                name=f"v3-evaluation-replica-{worker_id}-{device}",
            )
            worker.start()
            workers.append(worker)
        for opening_index, opening in enumerate(rows):
            task_queue.put((opening_index, opening))
        for _ in workers:
            task_queue.put(None)

        pairs_by_index: dict[int, tuple[GateGameResult, GateGameResult]] = {}
        worker_metrics: dict[int, ReplicatedWorkerMetrics] = {}
        while len(pairs_by_index) < len(rows) or len(worker_metrics) < worker_count:
            try:
                kind, worker_id, opening_index, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError("replicated evaluation worker exited unexpectedly")
                continue
            if kind == "error":
                raise RuntimeError(f"replicated evaluation worker {worker_id} failed:\n{payload}")
            if kind == "pair":
                index = int(opening_index)
                if index in pairs_by_index:
                    raise RuntimeError(f"duplicate replicated pair result {index}")
                if not isinstance(payload, tuple) or len(payload) != 2:
                    raise RuntimeError("replicated evaluation returned an invalid pair")
                pairs_by_index[index] = payload
                continue
            if kind == "stopped" and isinstance(payload, ReplicatedWorkerMetrics):
                worker_metrics[int(worker_id)] = payload
                continue
            raise RuntimeError("replicated evaluation returned an invalid message")

        for worker in workers:
            worker.join(timeout=30.0)
            if worker.exitcode != 0:
                raise RuntimeError(f"replicated worker {worker.name} exited with {worker.exitcode}")
        games = tuple(
            game
            for opening_index in range(len(rows))
            for game in pairs_by_index[opening_index]
        )
        elapsed = time.perf_counter() - started
        return PairedEvaluationResult(
            games=games,
            metrics=PairedEvaluationMetrics(
                parallel_games=worker_count,
                worker_processes=worker_count,
                start_method=start_method,
                games=len(games),
                wall_seconds=elapsed,
                inference_services=(),
                replicated_workers=tuple(
                    worker_metrics[index] for index in range(worker_count)
                ),
            ),
        )
    finally:
        _terminate_processes(workers)
        for managed_queue in (task_queue, result_queue):
            try:
                managed_queue.close()
                managed_queue.join_thread()
            except (AttributeError, OSError, ValueError):
                pass


__all__ = [
    "BatchingPredictor",
    "EvaluationModelSource",
    "EvaluationInferenceMetrics",
    "PairedEvaluationMetrics",
    "PairedEvaluationResult",
    "ReplicatedWorkerMetrics",
    "play_paired_openings_parallel",
    "play_paired_openings_replicated",
]
