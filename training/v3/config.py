from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints

from connect4_core.rules import DEFAULT_RULE_REGISTRY


@dataclass(frozen=True)
class RunConfig:
    run_id: str = "v3_1_smoke"
    seed: int = 20260810
    run_dir: str = ""
    resume: bool = False
    warm_start_checkpoint: str = ""
    warm_start_checkpoint_sha256: str = ""
    warm_start_mode: str = ""

    def __post_init__(self) -> None:
        if not self.run_id or any(part in self.run_id for part in ("/", "\\", "..")):
            raise ValueError("run.run_id must be a non-empty path-safe name.")
        if self.seed < 0:
            raise ValueError("run.seed must be non-negative.")
        warm_values = (
            self.warm_start_checkpoint,
            self.warm_start_checkpoint_sha256,
            self.warm_start_mode,
        )
        if any(warm_values) and not all(warm_values):
            raise ValueError("run warm-start fields must be supplied together.")
        if self.warm_start_mode and self.warm_start_mode != "optimizer_fresh_replay_v1":
            raise ValueError("run.warm_start_mode is unsupported.")
        if self.warm_start_checkpoint_sha256 and (
            len(self.warm_start_checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.warm_start_checkpoint_sha256)
        ):
            raise ValueError("run.warm_start_checkpoint_sha256 must be lowercase SHA-256.")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "gravity_resnet"
    channels: int = 16
    blocks: int = 1
    global_input_schema: str = "role_rule_v1"
    output_schema: str = "policy_wdl_aux_v1"
    rule_feature_dim: int = 32
    moves_left_classes: int = 301

    def __post_init__(self) -> None:
        if self.architecture != "gravity_resnet":
            raise ValueError("model.architecture must be 'gravity_resnet'.")
        if self.channels < 4:
            raise ValueError("model.channels must be at least 4.")
        if self.blocks < 1:
            raise ValueError("model.blocks must be at least 1.")
        if self.global_input_schema != "role_rule_v1":
            raise ValueError("model.global_input_schema must be 'role_rule_v1'.")
        if self.output_schema != "policy_wdl_aux_v1":
            raise ValueError("model.output_schema must be 'policy_wdl_aux_v1'.")
        if self.rule_feature_dim != 32:
            raise ValueError("model.rule_feature_dim must be 32 for role_rule_v1.")
        if self.moves_left_classes != 301:
            raise ValueError("model.moves_left_classes must be 301 for policy_wdl_aux_v1.")


@dataclass(frozen=True)
class SearchStageConfig:
    start_generation: int
    games: int
    full_search_sims: int
    fast_search_sims: int
    full_probability: float

    def __post_init__(self) -> None:
        if self.start_generation < 0 or self.games < 1:
            raise ValueError("selfplay search stages need a non-negative generation and positive games.")
        if self.full_search_sims < 1 or self.fast_search_sims < 1:
            raise ValueError("selfplay search simulation counts must be positive.")
        if self.full_search_sims < self.fast_search_sims:
            raise ValueError("full_search_sims must be at least fast_search_sims.")
        if not 0.0 <= self.full_probability <= 1.0:
            raise ValueError("selfplay full_probability must be in [0, 1].")


@dataclass(frozen=True)
class ExplorationPhaseConfig:
    start_ply: int
    temperature: float
    dirichlet_alpha: float
    dirichlet_epsilon: float

    def __post_init__(self) -> None:
        if self.start_ply < 0 or self.temperature < 0.0:
            raise ValueError("exploration phase ply and temperature must be non-negative.")
        if self.dirichlet_alpha < 0.0:
            raise ValueError("exploration phase dirichlet_alpha must be non-negative.")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("exploration phase dirichlet_epsilon must be in [0, 1].")
        if self.dirichlet_epsilon > 0.0 and self.dirichlet_alpha <= 0.0:
            raise ValueError("positive Dirichlet noise requires a positive alpha.")


@dataclass(frozen=True)
class SelfPlayConfig:
    search_schedule: tuple[SearchStageConfig, ...] = field(
        default_factory=lambda: (SearchStageConfig(0, 4, 8, 2, 0.5),)
    )
    exploration_phases: tuple[ExplorationPhaseConfig, ...] = field(
        default_factory=lambda: (
            ExplorationPhaseConfig(0, 1.0, 0.3, 0.25),
            ExplorationPhaseConfig(12, 0.0, 0.0, 0.0),
        )
    )
    cpuct: float = 1.5
    virtual_loss: float = 1.0
    rule_id: str = "classic"
    rule_registry_hash: str = DEFAULT_RULE_REGISTRY.registry_hash
    opening_full_search_plies: int = 0

    def __post_init__(self) -> None:
        _validate_starts(
            self.search_schedule,
            lambda stage: stage.start_generation,
            "selfplay.search_schedule",
        )
        _validate_starts(
            self.exploration_phases,
            lambda phase: phase.start_ply,
            "selfplay.exploration_phases",
        )
        if self.cpuct <= 0.0 or self.virtual_loss < 0.0:
            raise ValueError("selfplay.cpuct must be positive and virtual_loss non-negative.")
        if self.opening_full_search_plies < 0:
            raise ValueError("selfplay.opening_full_search_plies must be non-negative.")
        try:
            DEFAULT_RULE_REGISTRY.get(self.rule_id)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"selfplay.rule_id is unknown: {self.rule_id!r}.") from exc
        if self.rule_registry_hash != DEFAULT_RULE_REGISTRY.registry_hash:
            raise ValueError(
                "selfplay.rule_registry_hash does not match the executable V1 registry."
            )

    def stage_for_generation(self, generation: int) -> SearchStageConfig:
        if generation < 0:
            raise ValueError("generation must be non-negative.")
        selected = self.search_schedule[0]
        for stage in self.search_schedule:
            if stage.start_generation > generation:
                break
            selected = stage
        return selected

    def exploration_for_ply(self, ply: int) -> ExplorationPhaseConfig:
        if ply < 0:
            raise ValueError("ply must be non-negative.")
        selected = self.exploration_phases[0]
        for phase in self.exploration_phases:
            if phase.start_ply > ply:
                break
            selected = phase
        return selected


@dataclass(frozen=True)
class ReplayConfig:
    train_fraction: float = 0.95
    window_c: int = 1
    window_alpha: float = 0.75
    window_beta: float = 0.4
    train_tokens_per_raw_position: float = 4.0
    shard_games: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("replay.train_fraction must be in (0, 1).")
        if self.window_c < 1:
            raise ValueError("replay.window_c must be a positive position count.")
        if not 0.0 < self.window_alpha <= 1.0:
            raise ValueError("replay.window_alpha must be in (0, 1].")
        if not 0.0 <= self.window_beta <= 1.0:
            raise ValueError("replay.window_beta must be in [0, 1].")
        if self.train_tokens_per_raw_position <= 0.0 or self.shard_games < 1:
            raise ValueError("replay token ratio and shard_games must be positive.")


@dataclass(frozen=True)
class LearningRateStageConfig:
    start_train_positions: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.start_train_positions < 0 or self.learning_rate <= 0.0:
            raise ValueError("learner LR stages need non-negative positions and a positive rate.")


@dataclass(frozen=True)
class LearnerConfig:
    max_optimizer_steps_per_cycle: int = 2
    batch_size: int = 8
    lr_schedule: tuple[LearningRateStageConfig, ...] = field(
        default_factory=lambda: (LearningRateStageConfig(0, 1e-3),)
    )
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    policy_loss_weight: float = 1.0
    wdl_loss_weight: float = 1.0
    opponent_reply_loss_weight: float = 0.15
    future_occupancy_loss_weight: float = 0.15
    moves_left_loss_weight: float = 0.05
    future_occupancy_class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.max_optimizer_steps_per_cycle < 0 or self.batch_size < 1:
            raise ValueError("learner step cap must be non-negative and batch_size positive.")
        _validate_starts(
            self.lr_schedule,
            lambda stage: stage.start_train_positions,
            "learner.lr_schedule",
        )
        if self.weight_decay < 0.0 or self.grad_clip_norm <= 0.0:
            raise ValueError("learner weight_decay or grad_clip_norm is invalid.")
        loss_weights = (
            self.policy_loss_weight,
            self.wdl_loss_weight,
            self.opponent_reply_loss_weight,
            self.future_occupancy_loss_weight,
            self.moves_left_loss_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in loss_weights):
            raise ValueError("learner loss weights must be finite and non-negative.")
        if self.policy_loss_weight <= 0.0 or self.wdl_loss_weight <= 0.0:
            raise ValueError("learner policy and WDL loss weights must be positive.")
        if len(self.future_occupancy_class_weights) != 3 or any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in self.future_occupancy_class_weights
        ):
            raise ValueError(
                "learner.future_occupancy_class_weights must contain three positive values."
            )

    def learning_rate_for_positions(self, train_positions: int) -> float:
        if train_positions < 0:
            raise ValueError("train_positions must be non-negative.")
        selected = self.lr_schedule[0]
        for stage in self.lr_schedule:
            if stage.start_train_positions > train_positions:
                break
            selected = stage
        return selected.learning_rate


@dataclass(frozen=True)
class GateSearchStageConfig:
    start_generation: int
    search_sims: int

    def __post_init__(self) -> None:
        if self.start_generation < 0 or self.search_sims < 1:
            raise ValueError("gate search stages need a non-negative generation and positive sims.")


@dataclass(frozen=True)
class GateConfig:
    bootstrap_candidate_train_positions: int = 8
    candidate_train_positions: int = 16
    initial_opening_pairs: int = 2
    pair_increment: int = 2
    max_opening_pairs: int = 2
    opening_depths: tuple[int, ...] = (0, 2, 4, 6)
    search_schedule: tuple[GateSearchStageConfig, ...] = field(
        default_factory=lambda: (GateSearchStageConfig(0, 4),)
    )
    cpuct: float = 1.5
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    role_floor: float = 0.45
    accept_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.bootstrap_candidate_train_positions < 1 or self.candidate_train_positions < 1:
            raise ValueError("gate bootstrap and regular candidate intervals must be positive.")
        if self.initial_opening_pairs < 1 or self.pair_increment < 1:
            raise ValueError("gate initial and incremental pair counts must be positive.")
        if self.max_opening_pairs < self.initial_opening_pairs:
            raise ValueError("gate.max_opening_pairs must cover the initial pair count.")
        if (self.max_opening_pairs - self.initial_opening_pairs) % self.pair_increment != 0:
            raise ValueError("gate pair range must be exactly divisible by pair_increment.")
        if not self.opening_depths or self.opening_depths[0] != 0:
            raise ValueError("gate.opening_depths must be non-empty and start at zero.")
        if tuple(sorted(set(self.opening_depths))) != self.opening_depths:
            raise ValueError("gate.opening_depths must be strictly increasing.")
        if self.opening_depths[-1] >= 150:
            raise ValueError("gate opening depths must be below the board capacity.")
        _validate_starts(
            self.search_schedule,
            lambda stage: stage.start_generation,
            "gate.search_schedule",
        )
        if self.cpuct <= 0.0 or self.bootstrap_samples < 1:
            raise ValueError("gate cpuct and bootstrap_samples must be positive.")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("gate.confidence must be in (0.5, 1).")
        if not 0.0 <= self.role_floor <= 1.0:
            raise ValueError("gate.role_floor must be in [0, 1].")
        if not 0.0 <= self.accept_threshold <= 1.0:
            raise ValueError("gate.accept_threshold must be in [0, 1].")

    def search_sims_for_generation(self, generation: int) -> int:
        if generation < 0:
            raise ValueError("generation must be non-negative.")
        selected = self.search_schedule[0]
        for stage in self.search_schedule:
            if stage.start_generation > generation:
                break
            selected = stage
        return selected.search_sims

    def sequential_looks(self) -> int:
        return 1 + (self.max_opening_pairs - self.initial_opening_pairs) // self.pair_increment

    def decision_confidence(self) -> float:
        """Bonferroni confidence for each predeclared sequential gate look."""

        return 1.0 - (1.0 - self.confidence) / self.sequential_looks()


@dataclass(frozen=True)
class StoragePolicyConfig:
    mode: str = "keep_all"
    soft_used_fraction: float = 0.80
    hard_free_gib: float = 20.0
    active_window_margin: float = 1.25
    bundle_target_gib: float = 8.0
    keep_checkpoints: int = 3
    keep_accepted: int = 2
    keep_rejected: int = 1
    representative_games: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {"keep_all", "archive_ack_prune"}:
            raise ValueError("runtime.storage.mode must be keep_all or archive_ack_prune.")
        if not 0.0 < self.soft_used_fraction < 1.0:
            raise ValueError("runtime.storage.soft_used_fraction must be in (0, 1).")
        if self.hard_free_gib < 0.0 or self.bundle_target_gib <= 0.0:
            raise ValueError("runtime.storage free-space and bundle limits are invalid.")
        if self.active_window_margin < 1.0:
            raise ValueError("runtime.storage.active_window_margin must be at least 1.")
        if min(
            self.keep_checkpoints,
            self.keep_accepted,
            self.keep_rejected,
            self.representative_games,
        ) < 0:
            raise ValueError("runtime.storage retention counts must be non-negative.")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    selfplay_devices: tuple[str, ...] = ()
    actor_processes: int = 1
    mcts_lanes_per_actor: int = 1
    inference_batch_size: int = 1
    evaluation_parallel_games: int = 1
    evaluation_inference_batch_size: int = 1
    evaluation_inference_batch_timeout_ms: float = 1.0
    evaluation_devices: tuple[str, ...] = ()
    evaluation_replicas_per_device: int = 1
    num_workers: int = 0
    torch_threads: int = 1
    deterministic: bool = True
    learner_amp: bool = False
    storage: StoragePolicyConfig = field(default_factory=StoragePolicyConfig)

    def __post_init__(self) -> None:
        cuda_pattern = re.compile(r"cuda:(0|[1-9][0-9]*)\Z")
        if self.device != "cpu" and cuda_pattern.fullmatch(self.device) is None:
            raise ValueError("runtime.device must be 'cpu' or an exact CUDA device such as 'cuda:0'.")
        if any(cuda_pattern.fullmatch(device) is None for device in self.selfplay_devices):
            raise ValueError("runtime.selfplay_devices must contain exact CUDA names such as 'cuda:1'.")
        if self.device == "cpu" and self.selfplay_devices:
            raise ValueError("runtime.selfplay_devices must be empty for CPU execution.")
        if len(set(self.selfplay_devices)) != len(self.selfplay_devices):
            raise ValueError("runtime.selfplay_devices cannot contain duplicates.")
        if self.actor_processes < 1 or self.mcts_lanes_per_actor < 1:
            raise ValueError("runtime actor processes and MCTS lanes must be positive.")
        if self.inference_batch_size < 1:
            raise ValueError("runtime inference batch size must be positive.")
        if self.evaluation_parallel_games < 1 or self.evaluation_inference_batch_size < 1:
            raise ValueError("runtime evaluation parallel and batch sizes must be positive.")
        if self.evaluation_inference_batch_timeout_ms < 0.0:
            raise ValueError("runtime evaluation batch timeout must be non-negative.")
        if any(
            device != "cpu" and cuda_pattern.fullmatch(device) is None
            for device in self.evaluation_devices
        ):
            raise ValueError("runtime.evaluation_devices contains an invalid device name.")
        if len(set(self.evaluation_devices)) != len(self.evaluation_devices):
            raise ValueError("runtime.evaluation_devices cannot contain duplicates.")
        if self.evaluation_replicas_per_device < 1:
            raise ValueError("runtime evaluation replicas per device must be positive.")
        if self.evaluation_devices and (
            self.evaluation_parallel_games != 1
            or self.evaluation_inference_batch_size != 1
        ):
            raise ValueError(
                "runtime replicated and central-batched evaluation modes cannot be combined."
            )
        if self.num_workers < 0 or self.torch_threads < 1:
            raise ValueError("runtime.num_workers must be non-negative and torch_threads positive.")
        if self.device == "cpu" and self.learner_amp:
            raise ValueError("runtime.learner_amp must be false for CPU execution.")


@dataclass(frozen=True)
class V3Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "V3Config":
        if not isinstance(raw, Mapping):
            raise TypeError("V3 config must be a JSON object.")
        section_types = {
            "run": RunConfig,
            "model": ModelConfig,
            "selfplay": SelfPlayConfig,
            "replay": ReplayConfig,
            "learner": LearnerConfig,
            "gate": GateConfig,
            "runtime": RuntimeConfig,
        }
        unknown = sorted(set(raw) - set(section_types))
        if unknown:
            raise ValueError(f"Unknown V3 config section(s): {', '.join(unknown)}")
        decoded = {
            name: _decode_dataclass(section_type, raw.get(name, {}), name)
            for name, section_type in section_types.items()
        }
        return cls(**decoded)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True) + "\n"


_T = TypeVar("_T")


def _validate_starts(rows: tuple[Any, ...], key: Any, label: str) -> None:
    if not rows:
        raise ValueError(f"{label} must not be empty.")
    starts = tuple(int(key(row)) for row in rows)
    if starts[0] != 0 or tuple(sorted(set(starts))) != starts:
        raise ValueError(f"{label} must start at zero and have strictly increasing boundaries.")


def _decode_dataclass(section_type: type[_T], raw: Any, section_name: str) -> _T:
    if not isinstance(raw, Mapping):
        raise TypeError(f"Config section {section_name!r} must be an object.")
    known = {item.name for item in fields(section_type)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown field(s) in {section_name}: {', '.join(unknown)}")
    hints = get_type_hints(section_type)
    values = {
        key: _decode_value(value, hints[key], f"{section_name}.{key}")
        for key, value in raw.items()
    }
    return section_type(**values)


def _decode_value(value: Any, expected_type: Any, label: str) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{label} must be a JSON array, got {value!r}.")
        if len(args) == 2 and args[1] is Ellipsis:
            item_types = (args[0],) * len(value)
        else:
            if len(value) != len(args):
                raise TypeError(f"{label} must contain exactly {len(args)} items.")
            item_types = args
        return tuple(
            _decode_value(item, item_type, f"{label}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, item_types, strict=True))
        )
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return _decode_dataclass(expected_type, value, label)
    return _strict_scalar(value, expected_type, label)


def _strict_scalar(value: Any, expected_type: type[Any], label: str) -> Any:
    if expected_type is bool:
        if type(value) is not bool:
            raise TypeError(f"{label} must be a boolean, got {value!r}.")
        return value
    if expected_type is int:
        if type(value) is not int:
            raise TypeError(f"{label} must be an integer, got {value!r}.")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be a number, got {value!r}.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite, got {value!r}.")
        return number
    if expected_type is str:
        if type(value) is not str:
            raise TypeError(f"{label} must be a string, got {value!r}.")
        return value
    raise TypeError(f"Unsupported config field type for {label}: {expected_type!r}")


def load_config(path: str | Path) -> V3Config:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"V3 config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in V3 config {config_path}: {exc}") from exc
    return V3Config.from_dict(raw)


def resolve_config(
    config: V3Config | str | Path,
    *,
    run_dir: str | Path | None = None,
    resume: bool | None = None,
    device: str | None = None,
) -> V3Config:
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    if not isinstance(resolved, V3Config):
        raise TypeError("config must be a V3Config or a JSON path.")
    run = resolved.run
    runtime = resolved.runtime
    if run_dir is not None:
        run = replace(run, run_dir=str(Path(run_dir)))
    if resume is not None:
        if type(resume) is not bool:
            raise TypeError("resume override must be a boolean.")
        run = replace(run, resume=resume)
    if device is not None:
        selected_device = str(device)
        runtime = replace(
            runtime,
            device=selected_device,
            selfplay_devices=() if selected_device == "cpu" else runtime.selfplay_devices,
            learner_amp=False if selected_device == "cpu" else runtime.learner_amp,
        )
    return replace(resolved, run=run, runtime=runtime)


def canonical_config_json(config: V3Config) -> str:
    return json.dumps(config.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def config_hash(config: V3Config) -> str:
    """Hash learning semantics, excluding relocatable operational controls.

    Runtime topology, storage retention, output location, and the act of resuming
    may change without starting a new model lineage. They remain recorded in the
    resolved config and run manifest, but are intentionally absent here.
    """

    replay_semantics = asdict(config.replay)
    replay_semantics.pop("shard_games")
    run_semantics: dict[str, Any] = {"run_id": config.run.run_id, "seed": config.run.seed}
    if config.run.warm_start_mode:
        run_semantics["warm_start_mode"] = config.run.warm_start_mode
        run_semantics["warm_start_checkpoint_sha256"] = config.run.warm_start_checkpoint_sha256
    selfplay_semantics = asdict(config.selfplay)
    # Preserve hashes of pre-extension lineages while making every non-zero
    # opening override an explicit semantic fork.
    if selfplay_semantics["opening_full_search_plies"] == 0:
        selfplay_semantics.pop("opening_full_search_plies")
    semantic = {
        "run": run_semantics,
        "model": asdict(config.model),
        "selfplay": selfplay_semantics,
        "replay": replay_semantics,
        "learner": asdict(config.learner),
        "gate": asdict(config.gate),
        "search_runtime": {
            "mcts_lanes_per_actor": config.runtime.mcts_lanes_per_actor,
        },
    }
    payload = json.dumps(semantic, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ExplorationPhaseConfig",
    "GateConfig",
    "GateSearchStageConfig",
    "LearnerConfig",
    "LearningRateStageConfig",
    "ModelConfig",
    "ReplayConfig",
    "RunConfig",
    "RuntimeConfig",
    "SearchStageConfig",
    "SelfPlayConfig",
    "StoragePolicyConfig",
    "V3Config",
    "canonical_config_json",
    "config_hash",
    "load_config",
    "resolve_config",
]
