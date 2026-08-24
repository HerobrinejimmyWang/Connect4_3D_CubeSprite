"""Bounded deterministic self-play actor runtime for V3 training.

The coordinator owns game IDs and commits results in sorted game order. Actors
may finish in any order without changing per-game seeds or replay identity.
Accepted-model inference is represented by one shared service per configured
self-play device; random bootstrap needs no inference service.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .config import ModelConfig, SelfPlayConfig, V3Config
from .model import ROLE_FEATURE_COUNT, RULE_FEATURE_COUNT, TorchPredictor, build_model
from .search import RandomPredictor
from .selfplay import GameRecord, run_self_play_game


@dataclass(frozen=True)
class ActorTask:
    game_id: int
    generation: int


@dataclass(frozen=True)
class InferenceServiceMetrics:
    device: str
    requests: int
    positions: int
    batches: int
    max_batch: int
    batch_limit: int
    actor_count: int
    queue_capacity: int
    wall_seconds: float

    @property
    def mean_batch(self) -> float:
        return self.positions / max(self.batches, 1)

    def to_dict(self) -> dict[str, int | float | str]:
        return {**asdict(self), "mean_batch": self.mean_batch}


@dataclass(frozen=True)
class ActorPoolMetrics:
    actor_processes: int
    task_queue_capacity: int
    result_queue_capacity: int
    games: int
    wall_seconds: float
    inference_services: tuple[InferenceServiceMetrics, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "inference_services": [service.to_dict() for service in self.inference_services],
        }


@dataclass(frozen=True)
class ActorPoolResult:
    games: tuple[GameRecord, ...]
    metrics: ActorPoolMetrics


class RemotePredictor:
    """Blocking actor-side adapter for the shared inference service."""

    def __init__(
        self,
        actor_id: int,
        request_queue: Any,
        response_queue: Any,
        *,
        response_timeout_s: float,
    ) -> None:
        self.actor_id = int(actor_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.response_timeout_s = float(response_timeout_s)
        self.request_id = 0

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
        if boards.ndim != 4 or boards.shape[0] < 1 or boards.shape[1:] != (6, 5, 5):
            raise ValueError(f"remote inference boards must have shape [N,6,5,5], got {boards.shape}")
        roles = np.asarray(role_to_play, dtype=np.float32)
        rules = np.asarray(rule_features, dtype=np.float32)
        expected_roles = (boards.shape[0], ROLE_FEATURE_COUNT)
        expected_rules = (boards.shape[0], RULE_FEATURE_COUNT)
        if roles.shape != expected_roles:
            raise ValueError(
                f"remote role_to_play must have shape {expected_roles}, got {roles.shape}"
            )
        if rules.shape != expected_rules:
            raise ValueError(
                f"remote rule_features must have shape {expected_rules}, got {rules.shape}"
            )
        if not np.all(np.isfinite(roles)) or not np.all(
            np.logical_or(roles == 0.0, roles == 1.0)
        ) or not np.all(roles.sum(axis=1) == 1.0):
            raise ValueError("remote role_to_play rows must be finite FIRST/SECOND one-hot vectors")
        if not np.all(np.isfinite(rules)):
            raise ValueError("remote rule_features must contain finite values")
        request_id = self.request_id
        self.request_id += 1
        self.request_queue.put(
            (
                self.actor_id,
                request_id,
                np.array(boards, copy=True),
                np.array(roles, copy=True),
                np.array(rules, copy=True),
            )
        )
        try:
            response_id, policies, values, error = self.response_queue.get(
                timeout=self.response_timeout_s
            )
        except queue.Empty as exc:
            raise RuntimeError(
                f"actor {self.actor_id} timed out waiting for inference request {request_id}"
            ) from exc
        if error is not None:
            raise RuntimeError(f"shared inference failed: {error}")
        if int(response_id) != request_id:
            raise RuntimeError(
                f"actor {self.actor_id} received inference response {response_id}, expected {request_id}"
            )
        return np.asarray(policies, dtype=np.float32), np.asarray(values, dtype=np.float32)


def _load_service_predictor(
    model_config: ModelConfig,
    model_state: Mapping[str, Any],
    device: str,
) -> TorchPredictor:
    model = build_model(model_config)
    model.load_state_dict(model_state, strict=True)
    return TorchPredictor(model, device)


def _inference_service_main(
    device: str,
    actor_ids: tuple[int, ...],
    request_queue: Any,
    response_queues: Mapping[int, Any],
    status_queue: Any,
    model_config: ModelConfig,
    model_state: Mapping[str, Any],
    batch_limit: int,
    batch_timeout_s: float,
) -> None:
    started = time.perf_counter()
    requests = positions = batches = max_batch = 0
    try:
        torch.set_num_threads(1)
        predictor = _load_service_predictor(model_config, model_state, device)
        status_queue.put(("ready", device, None))
        stopping = False
        deferred_message = None
        while not stopping:
            if deferred_message is None:
                message = request_queue.get()
            else:
                message = deferred_message
                deferred_message = None
            if message is None:
                break
            pending = [message]
            pending_positions = len(message[2])
            if pending_positions > batch_limit:
                raise ValueError(
                    f"inference request of {pending_positions} positions exceeds hard batch "
                    f"limit {batch_limit}"
                )
            deadline = time.perf_counter() + max(0.0, float(batch_timeout_s))
            while pending_positions < batch_limit:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    next_message = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if next_message is None:
                    stopping = True
                    break
                next_positions = len(next_message[2])
                if pending_positions + next_positions > batch_limit:
                    deferred_message = next_message
                    break
                pending.append(next_message)
                pending_positions += next_positions

            boards = np.concatenate([item[2] for item in pending], axis=0)
            roles = np.concatenate([item[3] for item in pending], axis=0)
            rules = np.concatenate([item[4] for item in pending], axis=0)
            policies, values = predictor.predict_batch(
                boards,
                role_to_play=roles,
                rule_features=rules,
            )
            offset = 0
            for actor_id, request_id, request_boards, _request_roles, _request_rules in pending:
                count = len(request_boards)
                response_queues[int(actor_id)].put(
                    (
                        int(request_id),
                        policies[offset : offset + count],
                        values[offset : offset + count],
                        None,
                    )
                )
                offset += count
            requests += len(pending)
            positions += len(boards)
            batches += 1
            max_batch = max(max_batch, len(boards))
        metrics = InferenceServiceMetrics(
            device=device,
            requests=requests,
            positions=positions,
            batches=batches,
            max_batch=max_batch,
            batch_limit=int(batch_limit),
            actor_count=len(actor_ids),
            queue_capacity=max(1, 2 * len(actor_ids)),
            wall_seconds=time.perf_counter() - started,
        )
        status_queue.put(("stopped", device, metrics))
    except BaseException:
        error = traceback.format_exc()
        for actor_id in actor_ids:
            try:
                response_queues[int(actor_id)].put((-1, None, None, error), timeout=0.1)
            except (queue.Full, OSError):
                pass
        status_queue.put(("error", device, error))
        raise


def _actor_main(
    actor_id: int,
    task_queue: Any,
    result_queue: Any,
    selfplay_config: SelfPlayConfig,
    run_seed: int,
    producer_model_id: str,
    mcts_lanes: int,
    inference_request_queue: Any | None,
    inference_response_queue: Any | None,
    inference_response_timeout_s: float,
    force_full_search_before_ply: int,
) -> None:
    try:
        predictor = (
            RandomPredictor()
            if inference_request_queue is None
            else RemotePredictor(
                actor_id,
                inference_request_queue,
                inference_response_queue,
                response_timeout_s=inference_response_timeout_s,
            )
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            if not isinstance(task, ActorTask):
                raise TypeError(f"actor received unsupported task: {type(task).__name__}")
            game = run_self_play_game(
                selfplay_config,
                run_seed=run_seed,
                game_id=task.game_id,
                generation=task.generation,
                predictor=predictor,
                producer_model_id=producer_model_id,
                mcts_lanes=mcts_lanes,
                force_full_search_before_ply=force_full_search_before_ply,
            )
            result_queue.put(("game", actor_id, task.game_id, game))
    except BaseException:
        result_queue.put(("error", actor_id, None, traceback.format_exc()))


def _terminate_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def _service_devices(config: V3Config) -> tuple[str, ...]:
    if config.runtime.device == "cpu":
        return ("cpu",)
    return config.runtime.selfplay_devices or (config.runtime.device,)


def run_self_play_actor_pool(
    config: V3Config,
    *,
    accepted_model_state: Mapping[str, Any] | None = None,
    producer_model_id: str | None = None,
    start_game_id: int = 0,
    generation: int = 0,
    start_method: str = "spawn",
    inference_response_timeout_s: float = 120.0,
    inference_batch_timeout_s: float = 0.001,
    force_full_search_before_ply: int = 0,
) -> ActorPoolResult:
    """Generate one deterministic V3 self-play batch with bounded processes.

    ``accepted_model_state=None`` is the random bootstrap producer. A supplied
    state requires a stable non-random producer ID and is served centrally.
    """

    if not isinstance(config, V3Config):
        raise TypeError("run_self_play_actor_pool requires a resolved V3Config")
    if start_game_id < 0 or generation < 0:
        raise ValueError("start_game_id and generation must be non-negative")
    if force_full_search_before_ply < 0:
        raise ValueError("force_full_search_before_ply must be non-negative")
    if accepted_model_state is None:
        if producer_model_id not in (None, "random"):
            raise ValueError("random bootstrap cannot use a non-random producer_model_id")
        producer = "random"
    else:
        if not producer_model_id or producer_model_id == "random":
            raise ValueError("accepted model state requires a non-random producer_model_id")
        if config.runtime.inference_batch_size < config.runtime.mcts_lanes_per_actor:
            raise ValueError(
                "shared inference batch size must cover one complete MCTS lane request"
            )
        producer = str(producer_model_id)

    search_stage = config.selfplay.stage_for_generation(generation)
    game_count = int(search_stage.games)
    actor_count = min(int(config.runtime.actor_processes), game_count)
    if actor_count < 1:
        raise ValueError("actor pool requires at least one actor and one game")
    task_capacity = max(1, 2 * actor_count)
    result_capacity = max(1, 2 * actor_count)
    context = mp.get_context(start_method)
    task_queue = context.Queue(maxsize=task_capacity)
    result_queue = context.Queue(maxsize=result_capacity)
    status_queue = context.Queue()
    actors: list[mp.Process] = []
    services: list[mp.Process] = []
    request_queues: list[Any] = []
    response_queues: dict[int, Any] = {
        actor_id: context.Queue(maxsize=2) for actor_id in range(actor_count)
    }
    actor_service: dict[int, int] = {}
    service_metrics: list[InferenceServiceMetrics] = []
    started = time.perf_counter()

    try:
        if accepted_model_state is not None:
            devices = _service_devices(config)
            service_actor_ids = [
                tuple(actor_id for actor_id in range(actor_count) if actor_id % len(devices) == index)
                for index in range(len(devices))
            ]
            for service_index, (device, actor_ids) in enumerate(
                zip(devices, service_actor_ids, strict=True)
            ):
                if not actor_ids:
                    continue
                request_queue = context.Queue(maxsize=max(1, 2 * len(actor_ids)))
                request_queues.append(request_queue)
                for actor_id in actor_ids:
                    actor_service[actor_id] = len(request_queues) - 1
                service = context.Process(
                    target=_inference_service_main,
                    args=(
                        device,
                        actor_ids,
                        request_queue,
                        {actor_id: response_queues[actor_id] for actor_id in actor_ids},
                        status_queue,
                        config.model,
                        dict(accepted_model_state),
                        int(config.runtime.inference_batch_size),
                        float(inference_batch_timeout_s),
                    ),
                    name=f"v3-inference-{service_index}-{device}",
                )
                service.start()
                services.append(service)

            ready = 0
            while ready < len(services):
                try:
                    status, device, payload = status_queue.get(timeout=inference_response_timeout_s)
                except queue.Empty as exc:
                    raise RuntimeError("timed out starting V3 inference services") from exc
                if status == "error":
                    raise RuntimeError(f"V3 inference service {device} failed to start:\n{payload}")
                if status != "ready":
                    raise RuntimeError(f"unexpected inference service status during startup: {status}")
                ready += 1

        for actor_id in range(actor_count):
            request_queue = (
                None
                if accepted_model_state is None
                else request_queues[actor_service[actor_id]]
            )
            actor = context.Process(
                target=_actor_main,
                args=(
                    actor_id,
                    task_queue,
                    result_queue,
                    config.selfplay,
                    int(config.run.seed),
                    producer,
                    int(config.runtime.mcts_lanes_per_actor),
                    request_queue,
                    None if request_queue is None else response_queues[actor_id],
                    float(inference_response_timeout_s),
                    int(force_full_search_before_ply),
                ),
                name=f"v3-actor-{actor_id}",
            )
            actor.start()
            actors.append(actor)

        next_offset = 0
        initial_tasks = min(game_count, task_capacity)
        for _ in range(initial_tasks):
            task_queue.put(ActorTask(start_game_id + next_offset, generation))
            next_offset += 1

        games_by_id: dict[int, GameRecord] = {}
        while len(games_by_id) < game_count:
            try:
                kind, actor_id, game_id, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                failed_actors = [process for process in actors if process.exitcode not in (None, 0)]
                failed_services = [process for process in services if process.exitcode not in (None, 0)]
                if failed_actors or failed_services:
                    failures = [
                        f"{process.name}={process.exitcode}"
                        for process in (*failed_actors, *failed_services)
                    ]
                    raise RuntimeError(f"V3 self-play process failed: {', '.join(failures)}")
                continue
            if kind == "error":
                raise RuntimeError(f"V3 actor {actor_id} failed:\n{payload}")
            if kind != "game" or not isinstance(payload, GameRecord):
                raise RuntimeError(f"V3 actor {actor_id} returned an invalid result")
            game_id = int(game_id)
            if game_id in games_by_id:
                raise RuntimeError(f"duplicate V3 self-play result for game {game_id}")
            games_by_id[game_id] = payload
            if next_offset < game_count:
                task_queue.put(ActorTask(start_game_id + next_offset, generation))
                next_offset += 1

        for _ in actors:
            task_queue.put(None)
        for actor in actors:
            actor.join(timeout=30.0)
            if actor.exitcode != 0:
                raise RuntimeError(f"V3 actor {actor.name} exited with code {actor.exitcode}")

        for request_queue in request_queues:
            request_queue.put(None)
        stopped = 0
        while stopped < len(services):
            try:
                status, device, payload = status_queue.get(timeout=inference_response_timeout_s)
            except queue.Empty as exc:
                raise RuntimeError("timed out stopping V3 inference services") from exc
            if status == "error":
                raise RuntimeError(f"V3 inference service {device} failed:\n{payload}")
            if status != "stopped" or not isinstance(payload, InferenceServiceMetrics):
                raise RuntimeError(f"unexpected inference service shutdown status: {status}")
            service_metrics.append(payload)
            stopped += 1
        for service in services:
            service.join(timeout=30.0)
            if service.exitcode != 0:
                raise RuntimeError(f"V3 inference service {service.name} exited with code {service.exitcode}")

        ordered_games = tuple(games_by_id[game_id] for game_id in sorted(games_by_id))
        expected_ids = tuple(range(start_game_id, start_game_id + game_count))
        if tuple(game.game_id for game in ordered_games) != expected_ids:
            raise RuntimeError("V3 actor pool did not return the exact requested game ID range")
        return ActorPoolResult(
            games=ordered_games,
            metrics=ActorPoolMetrics(
                actor_processes=actor_count,
                task_queue_capacity=task_capacity,
                result_queue_capacity=result_capacity,
                games=game_count,
                wall_seconds=time.perf_counter() - started,
                inference_services=tuple(sorted(service_metrics, key=lambda item: item.device)),
            ),
        )
    finally:
        _terminate_processes(actors)
        _terminate_processes(services)
        for managed_queue in (
            task_queue,
            result_queue,
            status_queue,
            *request_queues,
            *response_queues.values(),
        ):
            try:
                managed_queue.close()
                managed_queue.join_thread()
            except (AttributeError, OSError, ValueError):
                pass


__all__ = [
    "ActorPoolMetrics",
    "ActorPoolResult",
    "ActorTask",
    "InferenceServiceMetrics",
    "RemotePredictor",
    "run_self_play_actor_pool",
]
