"""Deterministic online augmentation and the minimal V3 policy/WDL learner."""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .replay import SEARCH_FULL, ReplayShard, TrainTokenBucket, apply_d4, stable_d4_index


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
    ) -> None:
        self.replay = replay
        self.augmentation_seed = int(augmentation_seed)
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
        legal_mask = (board != 0).sum(axis=0) < board.shape[0]
        legal_flat = legal_mask.reshape(25)
        if np.any(visits[~legal_flat] != 0):
            raise ValueError(f"sample {(game_id, ply)} has visits on illegal columns")
        visit_total = int(visits.sum(dtype=np.uint64))
        if visit_total <= 0:
            raise ValueError(f"sample {(game_id, ply)} has no policy visits")
        policy = visits.astype(np.float32) / float(visit_total)
        source_position = int(self.source_positions[index])
        return {
            "board": torch.from_numpy(board.astype(np.float32, copy=False)),
            "policy": torch.from_numpy(policy),
            "policy_weight": torch.tensor(
                1.0 if int(self.replay.search_kind[index]) == SEARCH_FULL else 0.0,
                dtype=torch.float32,
            ),
            "wdl": torch.tensor(int(self.replay.wdl[index]), dtype=torch.long),
            "legal_mask": torch.from_numpy(legal_flat),
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
            for name in ("board", "policy", "policy_weight", "wdl", "legal_mask")
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
    policy_loss: float
    wdl_loss: float
    total_loss: float
    brier_score: float
    calibration_error: float
    grad_norm: float
    average_sample_age: float
    train_data_ratio: float
    positions_per_second: float
    learning_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


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


def _model_outputs(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, Mapping):
        try:
            policy_logits = output["policy_logits"]
            wdl_logits = output["wdl_logits"]
        except KeyError as exc:
            raise ValueError("model mapping output needs policy_logits and wdl_logits") from exc
    elif hasattr(output, "policy_logits") and hasattr(output, "wdl_logits"):
        policy_logits = output.policy_logits
        wdl_logits = output.wdl_logits
    elif isinstance(output, (tuple, list)) and len(output) == 2:
        policy_logits, wdl_logits = output
    else:
        raise TypeError("model must return (policy_logits, wdl_logits) or an equivalent mapping")
    if not torch.is_tensor(policy_logits) or not torch.is_tensor(wdl_logits):
        raise TypeError("model outputs must be tensors")
    if policy_logits.ndim != 2 or policy_logits.shape[1] != 25:
        raise ValueError(f"policy_logits must have shape [N,25], got {tuple(policy_logits.shape)}")
    if wdl_logits.ndim != 2 or wdl_logits.shape[1] != 3:
        raise ValueError(f"wdl_logits must have shape [N,3], got {tuple(wdl_logits.shape)}")
    if policy_logits.shape[0] != wdl_logits.shape[0]:
        raise ValueError("policy and WDL batch dimensions differ")
    return policy_logits, wdl_logits


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
    ) -> None:
        if batch_size < 1 or grad_clip_norm <= 0:
            raise ValueError("batch_size and grad_clip_norm must be positive")
        if ece_bins < 2:
            raise ValueError("ece_bins must be at least 2")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.grad_clip_norm = float(grad_clip_norm)
        self.sample_seed = int(sample_seed)
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
            **loader_options,
        )

    def train_steps(
        self,
        dataset: OnlineD4Dataset,
        *,
        steps: int,
        token_bucket: TrainTokenBucket | None = None,
    ) -> LearnerMetrics:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        if steps and len(dataset) == 0:
            raise ValueError("cannot train from an empty replay dataset")

        started = time.perf_counter()
        positions = 0
        policy_positions = 0
        value_positions = 0
        completed_steps = 0
        weighted_policy_loss = 0.0
        weighted_wdl_loss = 0.0
        weighted_total_loss = 0.0
        weighted_brier = 0.0
        weighted_age = 0.0
        last_grad_norm = 0.0
        calibration = _CalibrationAccumulator(self.ece_bins)
        self.last_sample_ids = []
        self.model.train()

        permitted_positions = int(steps) * self.batch_size
        if token_bucket is not None:
            permitted_positions = token_bucket.consumable(permitted_positions)
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
                policy_logits, wdl_logits = _model_outputs(self.model(batch["board"]))
                if policy_logits.shape[0] != batch_count:
                    raise ValueError("model output batch size does not match learner batch")
                if not batch["legal_mask"].any(dim=1).all():
                    raise ValueError("terminal board found in training batch")
                policy_logits = policy_logits.float()
                wdl_logits = wdl_logits.float()
                masked_policy_logits = policy_logits.masked_fill(
                    ~batch["legal_mask"], -torch.inf
                )
                policy_log_probabilities = F.log_softmax(masked_policy_logits, dim=1)
                safe_policy_log_probabilities = torch.where(
                    batch["legal_mask"],
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
                total_loss = policy_loss + wdl_loss

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

            if token_bucket is not None:
                consumed = token_bucket.consume(batch_count)
                if consumed != batch_count:
                    raise RuntimeError("token bucket changed during a learner step")
            self.sample_cursor += batch_count
            self.global_step += 1
            self.last_sample_ids.extend(sample_ids)
            positions += batch_count
            batch_policy_positions = int(policy_weight_sum.detach().cpu().item())
            policy_positions += batch_policy_positions
            value_positions += batch_count
            completed_steps += 1
            weighted_policy_loss += float(policy_loss_sum.detach().cpu())
            weighted_wdl_loss += float(wdl_loss.detach().cpu()) * batch_count
            weighted_total_loss += float(total_loss.detach().cpu()) * batch_count
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
            policy_loss=weighted_policy_loss / max(policy_positions, 1),
            wdl_loss=weighted_wdl_loss / max(value_positions, 1),
            total_loss=weighted_total_loss / denominator,
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
