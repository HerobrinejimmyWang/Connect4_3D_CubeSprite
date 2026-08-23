"""Deterministic online augmentation and the minimal V3 policy/WDL learner."""

from __future__ import annotations

import hashlib
import inspect
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from connect4_core.rules import DEFAULT_RULE_REGISTRY, RuleEngine, RuleRegistry

from .replay import ReplayShard, TrainTokenBucket, apply_d4, stable_d4_index


def _cursor_index(seed: int, cursor: int, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot sample an empty replay dataset")
    digest = hashlib.blake2b(digest_size=8, person=b"v3-sample-order")
    digest.update(struct.pack(">Q", int(seed) & ((1 << 64) - 1)))
    digest.update(struct.pack(">Q", int(cursor) & ((1 << 64) - 1)))
    return int.from_bytes(digest.digest(), "big") % size


class OnlineD4Dataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Replay dataset whose augmentation is stable for a sampling cursor.

    A key may be either an integer index, or ``(index, augmentation_token)``.
    The latter lets the learner reproduce the exact online transform after a
    checkpoint without saving prefetched DataLoader state.
    """

    def __init__(
        self,
        replay: ReplayShard,
        *,
        augmentation_seed: int,
        source_positions: Sequence[int] | np.ndarray | None = None,
        rule_registry: RuleRegistry = DEFAULT_RULE_REGISTRY,
    ) -> None:
        self.replay = replay
        self.augmentation_seed = int(augmentation_seed)
        self.rule_registry = rule_registry
        self._rule_engines = {
            spec.rule_code: RuleEngine(spec, registry=rule_registry)
            for spec in rule_registry.specs
        }
        if source_positions is None:
            self.source_positions = np.arange(len(replay), dtype=np.int64)
        else:
            self.source_positions = np.asarray(source_positions, dtype=np.int64)
            if self.source_positions.shape != (len(replay),):
                raise ValueError("source_positions must have one entry per replay sample")
        self._epoch = 0
        self._newest_source_position = (
            int(self.source_positions.max()) if len(self.source_positions) else -1
        )

    def __len__(self) -> int:
        return len(self.replay)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self._epoch = int(epoch)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(key, tuple):
            index, augmentation_token = int(key[0]), int(key[1])
        else:
            index = int(key)
            augmentation_token = self._epoch * max(1, len(self)) + index
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        game_id = int(self.replay.game_id[index])
        ply = int(self.replay.ply[index])
        transform = stable_d4_index(
            augmentation_seed=self.augmentation_seed,
            game_id=game_id,
            ply=ply,
            augmentation_token=augmentation_token,
        )
        board = apply_d4(self.replay.board[index], transform)
        visits = apply_d4(self.replay.visit_counts[index].reshape(5, 5), transform).reshape(25)
        player_to_move = int(self.replay.player_to_move[index])
        rule_code = int(self.replay.rule_code[index])
        absolute_board = np.asarray(self.replay.board[index]) * player_to_move
        try:
            engine = self._rule_engines[rule_code]
        except KeyError as exc:
            raise KeyError(f"replay references unknown rule_code {rule_code}") from exc
        state = engine.state_from_board(
            absolute_board,
            player_to_move=player_to_move,
            turn_index=int(self.replay.turn_index[index]),
            placement_count=int(self.replay.placement_count[index]),
        )
        legal_flat = apply_d4(
            engine.legal_column_mask(state).reshape(5, 5), transform
        ).reshape(25).astype(np.bool_, copy=False)
        if np.any(visits[~legal_flat] != 0):
            raise ValueError(f"sample {(game_id, ply)} has visits on illegal columns")
        visit_total = int(visits.sum(dtype=np.uint64))
        policy = (
            visits.astype(np.float32) / float(visit_total)
            if visit_total > 0
            else np.zeros((25,), dtype=np.float32)
        )
        role_to_play = np.asarray(
            (1.0, 0.0) if player_to_move == 1 else (0.0, 1.0), dtype=np.float32
        )
        rule_features = np.asarray(self.rule_registry.features(rule_code), dtype=np.float32)

        reply_mask = int(self.replay.opponent_reply_mask[index])
        reply_column = int(self.replay.opponent_reply_column[index])
        if reply_mask:
            reply_one_hot = np.zeros((5, 5), dtype=np.uint8)
            reply_one_hot.reshape(25)[reply_column] = 1
            opponent_reply = int(np.argmax(apply_d4(reply_one_hot, transform).reshape(25)))
        else:
            opponent_reply = 0

        terminal_absolute = apply_d4(self.replay.terminal_board[index], transform)
        terminal_canonical = terminal_absolute * player_to_move
        occupancy_target = np.where(
            terminal_canonical > 0,
            0,
            np.where(terminal_canonical < 0, 1, 2),
        ).astype(np.int64, copy=False)
        occupancy_mask = board == 0
        source_position = int(self.source_positions[index])
        return {
            "board": torch.from_numpy(board.astype(np.float32, copy=False)),
            "policy": torch.from_numpy(policy),
            "policy_weight": torch.tensor(float(self.replay.policy_weight[index])),
            "wdl": torch.tensor(int(self.replay.wdl[index]), dtype=torch.long),
            "legal_mask": torch.from_numpy(legal_flat),
            "role_to_play": torch.from_numpy(role_to_play),
            "rule_features": torch.from_numpy(rule_features),
            "opponent_reply": torch.tensor(opponent_reply, dtype=torch.long),
            "opponent_reply_mask": torch.tensor(reply_mask, dtype=torch.float32),
            "future_occupancy": torch.from_numpy(occupancy_target),
            "future_occupancy_mask": torch.from_numpy(occupancy_mask),
            "moves_left": torch.tensor(
                int(self.replay.remaining_turns[index]), dtype=torch.long
            ),
            "sample_id": (game_id, ply),
            "sample_age": max(0, self._newest_source_position - source_position),
            "d4": transform,
        }


class DeterministicKeyBatchSampler(
    torch.utils.data.Sampler[list[tuple[int, int]]]
):
    """Yield reproducible ``(dataset index, sample cursor)`` batches."""

    def __init__(
        self,
        *,
        dataset_size: int,
        sample_seed: int,
        start_cursor: int,
        batch_sizes: Sequence[int],
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if start_cursor < 0:
            raise ValueError("start_cursor cannot be negative")
        if any(int(size) <= 0 for size in batch_sizes):
            raise ValueError("batch sizes must be positive")
        self.dataset_size = int(dataset_size)
        self.sample_seed = int(sample_seed)
        self.start_cursor = int(start_cursor)
        self.batch_sizes = tuple(int(size) for size in batch_sizes)

    def __len__(self) -> int:
        return len(self.batch_sizes)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        cursor = self.start_cursor
        for batch_size in self.batch_sizes:
            batch: list[tuple[int, int]] = []
            for _ in range(batch_size):
                batch.append(
                    (_cursor_index(self.sample_seed, cursor, self.dataset_size), cursor)
                )
                cursor += 1
            yield batch


def _collate_replay_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Picklable collator used by both single- and multi-worker DataLoaders."""

    if not items:
        raise ValueError("cannot collate an empty replay batch")
    return {
        **{
            name: torch.stack([item[name] for item in items])
            for name in (
                "board",
                "policy",
                "policy_weight",
                "wdl",
                "legal_mask",
                "role_to_play",
                "rule_features",
                "opponent_reply",
                "opponent_reply_mask",
                "future_occupancy",
                "future_occupancy_mask",
                "moves_left",
            )
        },
        "sample_ids": [tuple(map(int, item["sample_id"])) for item in items],
        "average_sample_age": sum(float(item["sample_age"]) for item in items) / len(items),
    }


@dataclass(frozen=True)
class LearnerMetrics:
    steps: int
    positions: int
    policy_positions: int
    value_positions: int
    opponent_reply_positions: int
    future_occupancy_cells: int
    moves_left_positions: int
    policy_loss: float
    wdl_loss: float
    opponent_reply_loss: float
    future_occupancy_loss: float
    moves_left_loss: float
    total_loss: float
    opponent_reply_accuracy: float
    future_occupancy_accuracy: float
    moves_left_accuracy: float
    brier_score: float
    calibration_error: float
    grad_norm: float
    average_sample_age: float
    train_data_ratio: float
    positions_per_second: float
    learning_rate: float

    def to_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["future_occupancy_accuracy_unweighted"] = (
            self.future_occupancy_accuracy
        )
        return payload

    @property
    def future_occupancy_accuracy_unweighted(self) -> float:
        return self.future_occupancy_accuracy


class _CalibrationAccumulator:
    def __init__(self, bins: int = 10) -> None:
        self.count = np.zeros(bins, dtype=np.int64)
        self.confidence_sum = np.zeros(bins, dtype=np.float64)
        self.correct_sum = np.zeros(bins, dtype=np.float64)

    def add(self, probabilities: torch.Tensor, labels: torch.Tensor) -> None:
        confidence, prediction = probabilities.max(dim=1)
        correct = prediction.eq(labels).to(dtype=torch.float32)
        confidence_np = confidence.detach().cpu().numpy()
        correct_np = correct.detach().cpu().numpy()
        indices = np.minimum((confidence_np * len(self.count)).astype(np.int64), len(self.count) - 1)
        for bin_index in range(len(self.count)):
            mask = indices == bin_index
            if np.any(mask):
                self.count[bin_index] += int(mask.sum())
                self.confidence_sum[bin_index] += float(confidence_np[mask].sum())
                self.correct_sum[bin_index] += float(correct_np[mask].sum())

    def value(self) -> float:
        total = int(self.count.sum())
        if total == 0:
            return 0.0
        error = 0.0
        for count, confidence_sum, correct_sum in zip(
            self.count, self.confidence_sum, self.correct_sum
        ):
            if count:
                error += (count / total) * abs(confidence_sum / count - correct_sum / count)
        return float(error)


def build_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid AdamW learning rate or weight decay")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def _model_outputs(
    output: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if isinstance(output, Mapping):
        try:
            policy_logits = output["policy_logits"]
            wdl_logits = output["wdl_logits"]
        except KeyError as exc:
            raise ValueError("model mapping output needs policy_logits and wdl_logits") from exc
        opponent_reply_logits = output.get("opponent_reply_logits")
        future_occupancy_logits = output.get("future_occupancy_logits")
        moves_left_logits = output.get("moves_left_logits")
    elif hasattr(output, "policy_logits") and hasattr(output, "wdl_logits"):
        policy_logits = output.policy_logits
        wdl_logits = output.wdl_logits
        opponent_reply_logits = getattr(output, "opponent_reply_logits", None)
        future_occupancy_logits = getattr(output, "future_occupancy_logits", None)
        moves_left_logits = getattr(output, "moves_left_logits", None)
    elif isinstance(output, (tuple, list)) and len(output) == 2:
        policy_logits, wdl_logits = output
        opponent_reply_logits = None
        future_occupancy_logits = None
        moves_left_logits = None
    elif isinstance(output, (tuple, list)) and len(output) == 5:
        (
            policy_logits,
            wdl_logits,
            opponent_reply_logits,
            future_occupancy_logits,
            moves_left_logits,
        ) = output
    else:
        raise TypeError("model must return the named V3 outputs or a compatible tuple")
    if not torch.is_tensor(policy_logits) or not torch.is_tensor(wdl_logits):
        raise TypeError("model outputs must be tensors")
    if policy_logits.ndim != 2 or policy_logits.shape[1] != 25:
        raise ValueError(f"policy_logits must have shape [N,25], got {tuple(policy_logits.shape)}")
    if wdl_logits.ndim != 2 or wdl_logits.shape[1] != 3:
        raise ValueError(f"wdl_logits must have shape [N,3], got {tuple(wdl_logits.shape)}")
    if policy_logits.shape[0] != wdl_logits.shape[0]:
        raise ValueError("policy and WDL batch dimensions differ")
    optional_outputs = (
        ("opponent_reply_logits", opponent_reply_logits, (25,)),
        ("future_occupancy_logits", future_occupancy_logits, (3, 6, 5, 5)),
        ("moves_left_logits", moves_left_logits, (301,)),
    )
    for name, tensor, trailing_shape in optional_outputs:
        if tensor is None:
            continue
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(tensor.shape) != (policy_logits.shape[0], *trailing_shape):
            raise ValueError(
                f"{name} must have shape [N,{','.join(map(str, trailing_shape))}], "
                f"got {tuple(tensor.shape)}"
            )
    return (
        policy_logits,
        wdl_logits,
        opponent_reply_logits,
        future_occupancy_logits,
        moves_left_logits,
    )


class V3Learner:
    """Small deterministic learner used by smoke and future synchronous runs."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: str | torch.device,
        batch_size: int,
        grad_clip_norm: float,
        sample_seed: int,
        scheduler: Any | None = None,
        amp: bool = False,
        scaler: Any | None = None,
        ece_bins: int = 10,
        num_workers: int = 0,
        learning_rate_schedule: Sequence[tuple[int, float]] | None = None,
        policy_loss_weight: float = 1.0,
        wdl_loss_weight: float = 1.0,
        opponent_reply_loss_weight: float = 0.0,
        future_occupancy_loss_weight: float = 0.0,
        moves_left_loss_weight: float = 0.0,
        future_occupancy_class_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        if batch_size < 1 or grad_clip_norm <= 0:
            raise ValueError("batch_size and grad_clip_norm must be positive")
        if ece_bins < 2:
            raise ValueError("ece_bins must be at least 2")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        main_weights = (float(policy_loss_weight), float(wdl_loss_weight))
        if any(not np.isfinite(weight) or weight <= 0.0 for weight in main_weights):
            raise ValueError("policy and WDL loss weights must be finite and positive")
        auxiliary_weights = (
            float(opponent_reply_loss_weight),
            float(future_occupancy_loss_weight),
            float(moves_left_loss_weight),
        )
        if any(not np.isfinite(weight) or weight < 0.0 for weight in auxiliary_weights):
            raise ValueError("auxiliary loss weights must be finite and non-negative")
        occupancy_weights = tuple(float(weight) for weight in future_occupancy_class_weights)
        if (
            len(occupancy_weights) != 3
            or any(not np.isfinite(weight) or weight <= 0.0 for weight in occupancy_weights)
        ):
            raise ValueError("future occupancy class weights must contain three positive values")
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.grad_clip_norm = float(grad_clip_norm)
        self.sample_seed = int(sample_seed)
        self.policy_loss_weight = main_weights[0]
        self.wdl_loss_weight = main_weights[1]
        self.opponent_reply_loss_weight = auxiliary_weights[0]
        self.future_occupancy_loss_weight = auxiliary_weights[1]
        self.moves_left_loss_weight = auxiliary_weights[2]
        self.future_occupancy_class_weights = torch.tensor(
            occupancy_weights, dtype=torch.float32, device=self.device
        )
        forward_parameters = inspect.signature(self.model.forward).parameters
        self._model_accepts_global_inputs = {
            "role_to_play",
            "rule_features",
        }.issubset(forward_parameters) or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in forward_parameters.values()
        )
        self.scheduler = scheduler
        if scheduler is not None and learning_rate_schedule is not None:
            raise ValueError("use either scheduler or learning_rate_schedule, not both")
        self.learning_rate_schedule = (
            tuple((int(start), float(rate)) for start, rate in learning_rate_schedule)
            if learning_rate_schedule is not None
            else None
        )
        if self.learning_rate_schedule is not None:
            starts = tuple(start for start, _rate in self.learning_rate_schedule)
            if (
                not starts
                or starts[0] != 0
                or tuple(sorted(set(starts))) != starts
                or any(rate <= 0.0 for _start, rate in self.learning_rate_schedule)
            ):
                raise ValueError("learning_rate_schedule must start at zero and be strictly increasing")
        self.amp_enabled = bool(amp and self.device.type == "cuda")
        self.scaler = scaler or torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.ece_bins = int(ece_bins)
        self.num_workers = int(num_workers)
        self.pin_memory = self.device.type == "cuda"
        self.non_blocking = self.device.type == "cuda"
        self.prefetch_factor = (
            2 if self.device.type == "cuda" and self.num_workers > 0 else None
        )
        self.global_step = 0
        self.sample_cursor = 0
        self.last_sample_ids: list[tuple[int, int]] = []
        self.model.to(self.device)

    def _apply_learning_rate_schedule(self) -> None:
        if self.learning_rate_schedule is None:
            return
        selected_rate = self.learning_rate_schedule[0][1]
        for start, rate in self.learning_rate_schedule:
            if start > self.sample_cursor:
                break
            selected_rate = rate
        for group in self.optimizer.param_groups:
            group["lr"] = selected_rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "sample_cursor": self.sample_cursor,
            "sample_seed": self.sample_seed,
            "last_sample_ids": list(self.last_sample_ids),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_seed = int(state["sample_seed"])
        if saved_seed != self.sample_seed:
            raise ValueError(
                f"learner sample seed mismatch: checkpoint={saved_seed}, configured={self.sample_seed}"
            )
        self.global_step = int(state["global_step"])
        self.sample_cursor = int(state["sample_cursor"])
        self.last_sample_ids = [tuple(map(int, item)) for item in state.get("last_sample_ids", [])]

    def _data_loader(
        self,
        dataset: OnlineD4Dataset,
        batch_sizes: Sequence[int],
    ) -> torch.utils.data.DataLoader[dict[str, Any]]:
        sampler = DeterministicKeyBatchSampler(
            dataset_size=len(dataset),
            sample_seed=self.sample_seed,
            start_cursor=self.sample_cursor,
            batch_sizes=batch_sizes,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(
            (self.sample_seed ^ self.sample_cursor ^ 0x5A17D4) & ((1 << 63) - 1)
        )
        loader_options: dict[str, Any] = {}
        if self.pin_memory:
            loader_options["pin_memory"] = True
        if self.prefetch_factor is not None:
            loader_options["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=_collate_replay_items,
            generator=loader_generator,
            persistent_workers=self.num_workers > 0,
            **({"multiprocessing_context": "spawn"} if self.num_workers > 0 else {}),
            **loader_options,
        )

    def train_steps(
        self,
        dataset: OnlineD4Dataset,
        *,
        steps: int,
        token_bucket: TrainTokenBucket | None = None,
        position_limit: int | None = None,
    ) -> LearnerMetrics:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        if steps and len(dataset) == 0:
            raise ValueError("cannot train from an empty replay dataset")

        started = time.perf_counter()
        positions = 0
        policy_positions = 0
        value_positions = 0
        opponent_reply_positions = 0
        future_occupancy_cells = 0
        moves_left_positions = 0
        completed_steps = 0
        weighted_policy_loss = 0.0
        policy_weight_total = 0.0
        weighted_wdl_loss = 0.0
        weighted_opponent_reply_loss = 0.0
        weighted_future_occupancy_loss = 0.0
        future_occupancy_weight_total = 0.0
        weighted_moves_left_loss = 0.0
        weighted_total_loss = 0.0
        correct_opponent_replies = 0
        correct_future_occupancy = 0
        correct_moves_left = 0
        weighted_brier = 0.0
        weighted_age = 0.0
        last_grad_norm = 0.0
        calibration = _CalibrationAccumulator(self.ece_bins)
        self.last_sample_ids = []
        self.model.train()

        permitted_positions = int(steps) * self.batch_size
        if token_bucket is not None:
            permitted_positions = token_bucket.consumable(permitted_positions)
        if position_limit is not None:
            if isinstance(position_limit, bool) or int(position_limit) < 0:
                raise ValueError("position_limit must be a non-negative integer")
            permitted_positions = min(permitted_positions, int(position_limit))
        batch_sizes: list[int] = []
        while permitted_positions > 0 and len(batch_sizes) < int(steps):
            batch_count = min(self.batch_size, permitted_positions)
            batch_sizes.append(batch_count)
            permitted_positions -= batch_count
        loader = self._data_loader(dataset, batch_sizes) if batch_sizes else ()

        for batch in loader:
            batch_count = int(batch["board"].shape[0])
            sample_ids = batch.pop("sample_ids")
            average_age = float(batch.pop("average_sample_age"))
            batch = {
                name: tensor.to(self.device, non_blocking=self.non_blocking)
                for name, tensor in batch.items()
            }
            self._apply_learning_rate_schedule()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.amp_enabled,
            ):
                if self._model_accepts_global_inputs:
                    raw_output = self.model(
                        batch["board"],
                        role_to_play=batch["role_to_play"],
                        rule_features=batch["rule_features"],
                    )
                else:
                    raw_output = self.model(batch["board"])
                (
                    policy_logits,
                    wdl_logits,
                    opponent_reply_logits,
                    future_occupancy_logits,
                    moves_left_logits,
                ) = _model_outputs(raw_output)
                if policy_logits.shape[0] != batch_count:
                    raise ValueError("model output batch size does not match learner batch")
                policy_logits = policy_logits.float()
                wdl_logits = wdl_logits.float()
                # Forced-pass rows have no legal column and a zero policy target.
                # Use an unmasked softmax for numerical stability; policy_weight=0
                # keeps the row out of the policy objective.
                softmax_legal_mask = batch["legal_mask"] | ~batch["legal_mask"].any(
                    dim=1, keepdim=True
                )
                masked_policy_logits = policy_logits.masked_fill(
                    ~softmax_legal_mask, -torch.inf
                )
                policy_log_probabilities = F.log_softmax(masked_policy_logits, dim=1)
                safe_policy_log_probabilities = torch.where(
                    softmax_legal_mask,
                    policy_log_probabilities,
                    torch.zeros_like(policy_log_probabilities),
                )
                policy_loss_per_position = -(
                    batch["policy"].float() * safe_policy_log_probabilities
                ).sum(dim=1)
                policy_weights = batch["policy_weight"].float()
                policy_weight_sum = policy_weights.sum()
                policy_loss_sum = (policy_loss_per_position * policy_weights).sum()
                # Keeping the zero-weight expression in the graph makes an
                # all-fast batch a differentiable zero for the policy head.
                policy_loss = policy_loss_sum / policy_weight_sum.clamp_min(1.0)
                wdl_loss = F.cross_entropy(wdl_logits, batch["wdl"])

                if opponent_reply_logits is None:
                    if self.opponent_reply_loss_weight > 0.0:
                        raise ValueError("model is missing opponent_reply_logits")
                    opponent_reply_loss_sum = policy_logits.sum() * 0.0
                    opponent_reply_loss = opponent_reply_loss_sum
                else:
                    opponent_reply_logits = opponent_reply_logits.float()
                    reply_loss_per_position = F.cross_entropy(
                        opponent_reply_logits,
                        batch["opponent_reply"],
                        reduction="none",
                    )
                    reply_mask = batch["opponent_reply_mask"].float()
                    opponent_reply_loss_sum = (reply_loss_per_position * reply_mask).sum()
                    opponent_reply_loss = opponent_reply_loss_sum / reply_mask.sum().clamp_min(1.0)

                if future_occupancy_logits is None:
                    if self.future_occupancy_loss_weight > 0.0:
                        raise ValueError("model is missing future_occupancy_logits")
                    future_occupancy_loss_sum = policy_logits.sum() * 0.0
                    future_occupancy_loss = future_occupancy_loss_sum
                    future_occupancy_weight_sum = policy_logits.new_zeros(())
                else:
                    future_occupancy_logits = future_occupancy_logits.float()
                    occupancy_loss_per_cell = F.cross_entropy(
                        future_occupancy_logits,
                        batch["future_occupancy"],
                        weight=self.future_occupancy_class_weights,
                        reduction="none",
                    )
                    occupancy_mask = batch["future_occupancy_mask"].float()
                    future_occupancy_loss_sum = (
                        occupancy_loss_per_cell * occupancy_mask
                    ).sum()
                    future_occupancy_weight_sum = (
                        self.future_occupancy_class_weights[
                            batch["future_occupancy"]
                        ]
                        * occupancy_mask
                    ).sum()
                    future_occupancy_loss = (
                        future_occupancy_loss_sum
                        / future_occupancy_weight_sum.clamp_min(1.0)
                    )

                if moves_left_logits is None:
                    if self.moves_left_loss_weight > 0.0:
                        raise ValueError("model is missing moves_left_logits")
                    moves_left_loss_sum = policy_logits.sum() * 0.0
                    moves_left_loss = moves_left_loss_sum
                else:
                    moves_left_logits = moves_left_logits.float()
                    moves_left_loss = F.cross_entropy(
                        moves_left_logits, batch["moves_left"]
                    )
                    moves_left_loss_sum = moves_left_loss * batch_count

                total_loss = (
                    self.policy_loss_weight * policy_loss
                    + self.wdl_loss_weight * wdl_loss
                    + self.opponent_reply_loss_weight * opponent_reply_loss
                    + self.future_occupancy_loss_weight * future_occupancy_loss
                    + self.moves_left_loss_weight * moves_left_loss
                )

            if self.amp_enabled:
                scale_before = float(self.scaler.get_scale())
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_step_applied = float(self.scaler.get_scale()) >= scale_before
            else:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.optimizer.step()
                optimizer_step_applied = True
            last_grad_norm = float(torch.as_tensor(grad_norm).detach().cpu())
            if not optimizer_step_applied:
                # GradScaler found non-finite gradients and intentionally skipped
                # optimizer.step(). Do not consume replay budget or advance any
                # resumable cursor; the next call retries the same deterministic
                # sample IDs with the reduced scale.
                break
            if self.scheduler is not None:
                self.scheduler.step()

            with torch.no_grad():
                wdl_probabilities = F.softmax(wdl_logits.float(), dim=1)
                one_hot = F.one_hot(batch["wdl"], num_classes=3).to(dtype=torch.float32)
                brier = ((wdl_probabilities - one_hot) ** 2).sum(dim=1).mean()
                calibration.add(wdl_probabilities, batch["wdl"])
                if opponent_reply_logits is not None:
                    reply_valid = batch["opponent_reply_mask"].bool()
                    batch_reply_positions = int(reply_valid.sum().detach().cpu())
                    batch_correct_replies = int(
                        (
                            opponent_reply_logits.argmax(dim=1).eq(batch["opponent_reply"])
                            & reply_valid
                        )
                        .sum()
                        .detach()
                        .cpu()
                    )
                else:
                    batch_reply_positions = 0
                    batch_correct_replies = 0
                if future_occupancy_logits is not None:
                    occupancy_valid = batch["future_occupancy_mask"].bool()
                    batch_occupancy_cells = int(occupancy_valid.sum().detach().cpu())
                    batch_correct_occupancy = int(
                        (
                            future_occupancy_logits.argmax(dim=1).eq(
                                batch["future_occupancy"]
                            )
                            & occupancy_valid
                        )
                        .sum()
                        .detach()
                        .cpu()
                    )
                else:
                    batch_occupancy_cells = 0
                    batch_correct_occupancy = 0
                if moves_left_logits is not None:
                    batch_moves_left_positions = batch_count
                    batch_correct_moves_left = int(
                        moves_left_logits.argmax(dim=1)
                        .eq(batch["moves_left"])
                        .sum()
                        .detach()
                        .cpu()
                    )
                else:
                    batch_moves_left_positions = 0
                    batch_correct_moves_left = 0

            if token_bucket is not None:
                consumed = token_bucket.consume(batch_count)
                if consumed != batch_count:
                    raise RuntimeError("token bucket changed during a learner step")
            self.sample_cursor += batch_count
            self.global_step += 1
            self.last_sample_ids.extend(sample_ids)
            positions += batch_count
            batch_policy_positions = int((policy_weights > 0).sum().detach().cpu())
            policy_positions += batch_policy_positions
            value_positions += batch_count
            opponent_reply_positions += batch_reply_positions
            future_occupancy_cells += batch_occupancy_cells
            moves_left_positions += batch_moves_left_positions
            completed_steps += 1
            weighted_policy_loss += float(policy_loss_sum.detach().cpu())
            policy_weight_total += float(policy_weight_sum.detach().cpu())
            weighted_wdl_loss += float(wdl_loss.detach().cpu()) * batch_count
            weighted_opponent_reply_loss += float(
                opponent_reply_loss_sum.detach().cpu()
            )
            weighted_future_occupancy_loss += float(
                future_occupancy_loss_sum.detach().cpu()
            )
            future_occupancy_weight_total += float(
                future_occupancy_weight_sum.detach().cpu()
            )
            weighted_moves_left_loss += float(moves_left_loss_sum.detach().cpu())
            weighted_total_loss += float(total_loss.detach().cpu()) * batch_count
            correct_opponent_replies += batch_correct_replies
            correct_future_occupancy += batch_correct_occupancy
            correct_moves_left += batch_correct_moves_left
            weighted_brier += float(brier.detach().cpu()) * batch_count
            weighted_age += average_age * batch_count

        elapsed = max(time.perf_counter() - started, 1e-12)
        denominator = max(positions, 1)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        return LearnerMetrics(
            steps=completed_steps,
            positions=positions,
            policy_positions=policy_positions,
            value_positions=value_positions,
            opponent_reply_positions=opponent_reply_positions,
            future_occupancy_cells=future_occupancy_cells,
            moves_left_positions=moves_left_positions,
            policy_loss=weighted_policy_loss / max(policy_weight_total, 1.0),
            wdl_loss=weighted_wdl_loss / max(value_positions, 1),
            opponent_reply_loss=(
                weighted_opponent_reply_loss / max(opponent_reply_positions, 1)
            ),
            future_occupancy_loss=(
                weighted_future_occupancy_loss / max(future_occupancy_weight_total, 1.0)
            ),
            moves_left_loss=weighted_moves_left_loss / max(moves_left_positions, 1),
            total_loss=weighted_total_loss / denominator,
            opponent_reply_accuracy=(
                correct_opponent_replies / max(opponent_reply_positions, 1)
            ),
            future_occupancy_accuracy=(
                correct_future_occupancy / max(future_occupancy_cells, 1)
            ),
            moves_left_accuracy=correct_moves_left / max(moves_left_positions, 1),
            brier_score=weighted_brier / denominator,
            calibration_error=calibration.value(),
            grad_norm=last_grad_norm,
            average_sample_age=weighted_age / denominator,
            train_data_ratio=(token_bucket.train_data_ratio if token_bucket is not None else 0.0),
            positions_per_second=positions / elapsed,
            learning_rate=learning_rate,
        )


__all__ = [
    "DeterministicKeyBatchSampler",
    "LearnerMetrics",
    "OnlineD4Dataset",
    "V3Learner",
    "build_adamw",
]
