"""Pure static hardware planning for the V3 training scheduler.

The planner deliberately does not import PyTorch or probe the host.  Callers
provide a fake or real CUDA inventory and persist the returned plan in the run
manifest. One GPU is time-sliced between self-play and learning; with two or
more GPUs the explicitly configured learner is kept fixed and the remaining
active devices each own one self-play inference service. DDP is intentionally
not part of the V3.1 small-model plan.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence


_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class HardwarePlanWarning:
    """A stable warning code plus a human-readable explanation."""

    code: str
    message: str


@dataclass(frozen=True)
class InferenceServicePlan:
    """One CUDA-owning inference service and its statically assigned actors."""

    device: str
    actor_ids: tuple[int, ...]
    effective_batch_limit: int
    request_queue_capacity: int

    @property
    def actor_count(self) -> int:
        return len(self.actor_ids)


@dataclass(frozen=True)
class HardwarePlan:
    """Resolved resources consumed by a future long-running V3 scheduler."""

    mode: str
    cuda_devices: tuple[str, ...]
    learner_device: str
    selfplay_devices: tuple[str, ...]
    unused_cuda_devices: tuple[str, ...]
    stages_may_overlap: bool
    ddp_enabled: bool
    actor_count: int
    mcts_lanes: int
    cpu_cores: int
    requested_inference_batch_limit: int
    effective_inference_batch_limit: int
    game_queue_capacity: int
    completed_game_queue_capacity: int
    checkpoint_queue_capacity: int
    inference_services: tuple[InferenceServicePlan, ...]
    warnings: tuple[HardwarePlanWarning, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _parse_cuda_devices(
    devices: Sequence[str],
    *,
    cuda_inventory_count: int | None,
) -> tuple[str, ...]:
    if isinstance(devices, (str, bytes)) or not isinstance(devices, Sequence):
        raise TypeError("cuda_devices must be a sequence of 'cuda:N' strings")
    if not devices:
        raise ValueError("cuda_devices must contain at least one CUDA device")
    if cuda_inventory_count is not None:
        if type(cuda_inventory_count) is not int or cuda_inventory_count < 0:
            raise ValueError("cuda_inventory_count must be a non-negative integer")

    parsed: list[tuple[int, str]] = []
    seen: set[int] = set()
    for raw in devices:
        if type(raw) is not str:
            raise TypeError("every cuda_devices entry must be a string")
        match = _CUDA_DEVICE.fullmatch(raw)
        if match is None:
            raise ValueError(f"invalid CUDA device {raw!r}; expected the exact form 'cuda:N'")
        index = int(match.group(1))
        if index in seen:
            raise ValueError(f"duplicate CUDA device: cuda:{index}")
        if cuda_inventory_count is not None and index >= cuda_inventory_count:
            raise ValueError(
                f"CUDA device cuda:{index} is outside inventory count {cuda_inventory_count}"
            )
        seen.add(index)
        parsed.append((index, f"cuda:{index}"))

    parsed.sort(key=lambda item: item[0])
    return tuple(device for _index, device in parsed)


def plan_hardware(
    cuda_devices: Sequence[str],
    *,
    learner_device: str | None = None,
    actors: int,
    mcts_lanes: int,
    inference_batch_limit: int,
    cpu_cores: int,
    cuda_inventory_count: int | None = None,
) -> HardwarePlan:
    """Return a deterministic, bounded, no-DDP hardware plan.

    ``cuda_inventory_count`` is injectable so preflight and tests can validate
    indices without this module importing or querying PyTorch. Device order in
    the input is irrelevant. If ``learner_device`` is omitted the lowest device
    is used for backward compatibility; production callers should pass it
    explicitly. In the multi-GPU plan actors are assigned round-robin to one
    service per active self-play device.
    """

    actor_count = _positive_int(actors, "actors")
    lane_count = _positive_int(mcts_lanes, "mcts_lanes")
    requested_batch = _positive_int(inference_batch_limit, "inference_batch_limit")
    available_cores = _positive_int(cpu_cores, "cpu_cores")
    devices = _parse_cuda_devices(
        cuda_devices,
        cuda_inventory_count=cuda_inventory_count,
    )

    if learner_device is None:
        resolved_learner = devices[0]
    else:
        parsed_learner = _parse_cuda_devices(
            (learner_device,),
            cuda_inventory_count=cuda_inventory_count,
        )[0]
        if parsed_learner not in devices:
            raise ValueError("learner_device must be included in cuda_devices")
        resolved_learner = parsed_learner
    if len(devices) == 1:
        mode = "single_gpu_staged"
        service_devices = devices
        unused_devices: tuple[str, ...] = ()
        stages_may_overlap = False
    else:
        mode = "multi_gpu_role_split"
        available_selfplay_devices = tuple(
            device for device in devices if device != resolved_learner
        )
        active_service_count = min(actor_count, len(available_selfplay_devices))
        service_devices = available_selfplay_devices[:active_service_count]
        unused_devices = available_selfplay_devices[active_service_count:]
        stages_may_overlap = True

    actor_assignments: list[list[int]] = [[] for _device in service_devices]
    for actor_id in range(actor_count):
        actor_assignments[actor_id % len(service_devices)].append(actor_id)

    services: list[InferenceServicePlan] = []
    for device, actor_ids in zip(service_devices, actor_assignments, strict=True):
        outstanding_requests = len(actor_ids) * lane_count
        effective_batch = min(requested_batch, outstanding_requests)
        services.append(
            InferenceServicePlan(
                device=device,
                actor_ids=tuple(actor_ids),
                effective_batch_limit=effective_batch,
                request_queue_capacity=max(1, 2 * outstanding_requests),
            )
        )

    warnings: list[HardwarePlanWarning] = []
    if actor_count >= available_cores:
        warnings.append(
            HardwarePlanWarning(
                code="cpu_actor_oversubscription",
                message=(
                    f"{actor_count} actor processes leave no dedicated CPU core for the "
                    f"coordinator/learner on a {available_cores}-core allocation."
                ),
            )
        )
    potential_search_threads = actor_count * lane_count
    if potential_search_threads > available_cores:
        warnings.append(
            HardwarePlanWarning(
                code="cpu_lane_oversubscription",
                message=(
                    f"actors*mcts_lanes={potential_search_threads} exceeds "
                    f"cpu_cores={available_cores}; actor search threads may contend."
                ),
            )
        )
    if any(service.effective_batch_limit < requested_batch for service in services):
        warnings.append(
            HardwarePlanWarning(
                code="inference_batch_concurrency_limited",
                message=(
                    f"Requested inference batch {requested_batch} cannot be filled by every "
                    "service's assigned actors and MCTS lanes."
                ),
            )
        )
    if unused_devices:
        warnings.append(
            HardwarePlanWarning(
                code="idle_selfplay_devices",
                message=(
                    "Some CUDA devices are left idle because there are fewer actors than "
                    f"self-play devices: {', '.join(unused_devices)}."
                ),
            )
        )

    effective_batch_limit = max(service.effective_batch_limit for service in services)
    return HardwarePlan(
        mode=mode,
        cuda_devices=devices,
        learner_device=resolved_learner,
        selfplay_devices=tuple(service.device for service in services),
        unused_cuda_devices=unused_devices,
        stages_may_overlap=stages_may_overlap,
        ddp_enabled=False,
        actor_count=actor_count,
        mcts_lanes=lane_count,
        cpu_cores=available_cores,
        requested_inference_batch_limit=requested_batch,
        effective_inference_batch_limit=effective_batch_limit,
        game_queue_capacity=2 * actor_count,
        completed_game_queue_capacity=2 * actor_count,
        checkpoint_queue_capacity=1,
        inference_services=tuple(services),
        warnings=tuple(warnings),
    )


__all__ = [
    "HardwarePlan",
    "HardwarePlanWarning",
    "InferenceServicePlan",
    "plan_hardware",
]
