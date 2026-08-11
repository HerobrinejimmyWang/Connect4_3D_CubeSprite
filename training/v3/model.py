from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4_core import BOARD_SIZE, MAX_LAYERS

from .config import ModelConfig


COLUMN_COUNT = BOARD_SIZE * BOARD_SIZE
LEGACY_ACTION_COUNT = MAX_LAYERS * COLUMN_COUNT
WDL_WIN = 0
WDL_DRAW = 1
WDL_LOSS = 2


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if groups <= channels and channels % groups == 0:
            return groups
    return 1


def canonical_board_to_planes(board: torch.Tensor) -> torch.Tensor:
    """Encode canonical [N,6,5,5] boards as 14 gravity-aware 2D planes."""
    if board.ndim == 3:
        board = board.unsqueeze(0)
    if board.ndim != 4 or tuple(board.shape[1:]) != (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"Expected canonical board [N,{MAX_LAYERS},{BOARD_SIZE},{BOARD_SIZE}], "
            f"got {tuple(board.shape)}."
        )
    board = board.to(dtype=torch.float32)
    current = (board > 0).to(dtype=board.dtype)
    opponent = (board < 0).to(dtype=board.dtype)
    occupancy = (board != 0).sum(dim=1, keepdim=True).to(dtype=board.dtype)
    normalized_height = occupancy / float(MAX_LAYERS)
    legal_column = (occupancy < float(MAX_LAYERS)).to(dtype=board.dtype)
    return torch.cat((current, opponent, normalized_height, legal_column), dim=1)


def legal_column_mask(board: np.ndarray) -> np.ndarray:
    raw = np.asarray(board)
    if raw.shape[-3:] != (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Board must end in {(MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)}, got {raw.shape}.")
    return np.count_nonzero(raw, axis=-3) < MAX_LAYERS


class _PreActBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


class GravityPolicyValueNetV3(nn.Module):
    """Small gravity-aware policy/WDL network with no hidden configuration."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(model_config, ModelConfig):
            raise TypeError("GravityPolicyValueNetV3 requires an explicit ModelConfig.")
        self.model_config = model_config
        channels = model_config.channels
        self.stem = nn.Conv2d(2 * MAX_LAYERS + 2, channels, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList(_PreActBlock(channels) for _ in range(model_config.blocks))
        self.final_norm = nn.GroupNorm(_group_count(channels), channels)
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)
        value_hidden = max(16, channels)
        self.wdl_head = nn.Sequential(
            nn.Linear(channels * 2, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, 3),
        )

    def forward(self, canonical_board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(canonical_board_to_planes(canonical_board))
        for block in self.blocks:
            x = block(x)
        x = F.silu(self.final_norm(x))
        policy_logits = self.policy_head(x).flatten(1)
        pooled = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        wdl_logits = self.wdl_head(pooled)
        return policy_logits, wdl_logits

    def export_config(self) -> dict[str, Any]:
        return asdict(self.model_config)


def _model_config_from_mapping(raw: Mapping[str, Any]) -> ModelConfig:
    allowed = {"architecture", "channels", "blocks"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown model config field(s): {', '.join(unknown)}")
    return ModelConfig(**dict(raw))


def build_model(model_config: ModelConfig | Mapping[str, Any]) -> GravityPolicyValueNetV3:
    if isinstance(model_config, Mapping):
        model_config = _model_config_from_mapping(model_config)
    if not isinstance(model_config, ModelConfig):
        raise TypeError("build_model requires an explicit ModelConfig or model config mapping.")
    return GravityPolicyValueNetV3(model_config)


def wdl_expected_value(probabilities: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return P(win)-P(loss), preserving NumPy/Torch and leading dimensions."""
    if probabilities.shape[-1] != 3:
        raise ValueError(f"WDL values must have a final dimension of 3, got {probabilities.shape}.")
    return probabilities[..., WDL_WIN] - probabilities[..., WDL_LOSS]


def wdl_logits_expected_value(logits: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(logits, torch.Tensor):
        return wdl_expected_value(torch.softmax(logits, dim=-1))
    raw = np.asarray(logits)
    shifted = raw - np.max(raw, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return wdl_expected_value(probabilities)


def column_to_legacy_action(board: np.ndarray, column: int) -> int:
    raw = np.asarray(board)
    if raw.shape != (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Board shape must be {(MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)}, got {raw.shape}.")
    column = int(column)
    if not 0 <= column < COLUMN_COUNT:
        raise ValueError(f"Column {column} is outside [0, {COLUMN_COUNT - 1}].")
    row, col = divmod(column, BOARD_SIZE)
    empty = np.flatnonzero(raw[:, row, col] == 0)
    if empty.size == 0:
        raise ValueError(f"Column {column} is full.")
    layer = int(empty[0])
    if layer > 0 and raw[layer - 1, row, col] == 0:
        raise ValueError(f"Column {column} contains a gravity-invalid gap.")
    return layer * COLUMN_COUNT + column


def legacy_action_to_column(action: int, board: np.ndarray | None = None) -> int:
    action = int(action)
    if not 0 <= action < LEGACY_ACTION_COUNT:
        raise ValueError(f"Legacy action {action} is outside [0, {LEGACY_ACTION_COUNT - 1}].")
    column = action % COLUMN_COUNT
    if board is not None and column_to_legacy_action(board, column) != action:
        raise ValueError(f"Legacy action {action} is not the current gravity-legal action for its column.")
    return column


def column_policy_to_legacy(
    column_policy: np.ndarray,
    board: np.ndarray,
    *,
    illegal_value: float = 0.0,
) -> np.ndarray:
    policy = np.asarray(column_policy)
    boards = np.asarray(board)
    single = policy.ndim == 1
    if single:
        policy = policy[None, :]
        boards = boards[None, ...]
    if policy.ndim != 2 or policy.shape[1] != COLUMN_COUNT:
        raise ValueError(f"Column policy must be [N,{COLUMN_COUNT}] or [{COLUMN_COUNT}], got {policy.shape}.")
    if boards.shape != (policy.shape[0], MAX_LAYERS, BOARD_SIZE, BOARD_SIZE):
        raise ValueError("Board batch does not match the column policy batch.")
    legacy = np.full((policy.shape[0], LEGACY_ACTION_COUNT), illegal_value, dtype=policy.dtype)
    for batch_index in range(policy.shape[0]):
        for column in range(COLUMN_COUNT):
            try:
                action = column_to_legacy_action(boards[batch_index], column)
            except ValueError as exc:
                if "is full" not in str(exc):
                    raise
                continue
            legacy[batch_index, action] = policy[batch_index, column]
    return legacy[0] if single else legacy


def legacy_policy_to_columns(legacy_policy: np.ndarray, board: np.ndarray | None = None) -> np.ndarray:
    policy = np.asarray(legacy_policy)
    single = policy.ndim == 1
    if single:
        policy = policy[None, :]
    if policy.ndim != 2 or policy.shape[1] != LEGACY_ACTION_COUNT:
        raise ValueError(
            f"Legacy policy must be [N,{LEGACY_ACTION_COUNT}] or [{LEGACY_ACTION_COUNT}], got {policy.shape}."
        )
    if board is None:
        columns = policy.reshape(policy.shape[0], MAX_LAYERS, COLUMN_COUNT).sum(axis=1)
    else:
        boards = np.asarray(board)
        if boards.ndim == 3:
            boards = boards[None, ...]
        if boards.shape != (policy.shape[0], MAX_LAYERS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError("Board batch does not match the legacy policy batch.")
        columns = np.zeros((policy.shape[0], COLUMN_COUNT), dtype=policy.dtype)
        for batch_index in range(policy.shape[0]):
            for column in range(COLUMN_COUNT):
                try:
                    action = column_to_legacy_action(boards[batch_index], column)
                except ValueError as exc:
                    if "is full" not in str(exc):
                        raise
                    continue
                columns[batch_index, column] = policy[batch_index, action]
    return columns[0] if single else columns


class TorchPredictor:
    """Inference adapter exposing 25 policy probabilities and WDL probabilities."""

    def __init__(self, model: GravityPolicyValueNetV3, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, canonical_board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policies, wdl = self.predict_batch(np.asarray(canonical_board)[None, ...])
        return policies[0], wdl[0]

    def predict_batch(self, canonical_boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run one device forward pass for a batch of MCTS leaves."""

        raw = np.asarray(canonical_boards)
        if raw.ndim != 4 or raw.shape[1:] != (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                "canonical_boards must have shape [N,6,5,5], "
                f"got {raw.shape}."
            )
        if raw.shape[0] < 1:
            raise ValueError("canonical_boards must contain at least one board.")
        board = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            policy_logits, wdl_logits = self.model(board)
            policy = torch.softmax(policy_logits.float(), dim=-1)
            wdl = torch.softmax(wdl_logits.float(), dim=-1)
        return (
            policy.cpu().numpy().astype(np.float32, copy=False),
            wdl.cpu().numpy().astype(np.float32, copy=False),
        )


class LegacyPolicyValueAdapter:
    """Expose a V3 predictor through the legacy 150-policy/scalar-value contract."""

    def __init__(self, predictor: Any) -> None:
        if not hasattr(predictor, "predict"):
            raise TypeError("LegacyPolicyValueAdapter requires an object with predict(board).")
        self.predictor = predictor

    def predict(self, canonical_board: np.ndarray) -> tuple[np.ndarray, float]:
        policy, wdl = self.predictor.predict(canonical_board)
        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        if policy.shape != (COLUMN_COUNT,) or np.any(policy < 0.0) or not np.all(np.isfinite(policy)):
            raise ValueError("V3 predictor returned an invalid column policy.")
        valid = legal_column_mask(canonical_board).reshape(-1)
        policy = np.where(valid, policy, 0.0)
        total = float(policy.sum())
        if total <= 0.0:
            policy = valid.astype(np.float64) / int(valid.sum())
        else:
            policy /= total
        legacy = column_policy_to_legacy(policy, canonical_board)
        return legacy.astype(np.float32, copy=False), float(wdl_expected_value(np.asarray(wdl)))


__all__ = [
    "COLUMN_COUNT",
    "LEGACY_ACTION_COUNT",
    "WDL_DRAW",
    "WDL_LOSS",
    "WDL_WIN",
    "GravityPolicyValueNetV3",
    "LegacyPolicyValueAdapter",
    "TorchPredictor",
    "build_model",
    "canonical_board_to_planes",
    "column_policy_to_legacy",
    "column_to_legacy_action",
    "legacy_action_to_column",
    "legacy_policy_to_columns",
    "legal_column_mask",
    "wdl_expected_value",
    "wdl_logits_expected_value",
]
