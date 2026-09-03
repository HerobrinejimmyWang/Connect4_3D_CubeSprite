from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4_core import BOARD_SIZE, MAX_LAYERS

from .config import ModelConfig, model_config_dict


COLUMN_COUNT = BOARD_SIZE * BOARD_SIZE
LEGACY_ACTION_COUNT = MAX_LAYERS * COLUMN_COUNT
WDL_WIN = 0
WDL_DRAW = 1
WDL_LOSS = 2
ROLE_FEATURE_COUNT = 2
RULE_FEATURE_COUNT = 32
MOVES_LEFT_CLASSES = 301
GLOBAL_INPUT_SCHEMA = "role_rule_v1"
OUTPUT_SCHEMA = "policy_wdl_aux_v1"
ROLE_FEATURE_NAMES = ("to_play_first", "to_play_second")
RULE_FEATURE_NAMES = (
    "first_vertical_normal",
    "first_vertical_ignored",
    "first_vertical_illegal",
    "first_layer0_normal",
    "first_layer0_ignored",
    "no_legal_placement_draw",
    "no_legal_placement_loss",
    "no_legal_placement_forced_pass",
) + tuple(f"reserved_{index}" for index in range(8, RULE_FEATURE_COUNT))


@dataclass(frozen=True)
class SearchOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.policy_logits
        yield self.wdl_logits


@dataclass(frozen=True)
class ModelOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    opponent_reply_logits: torch.Tensor
    future_occupancy_logits: torch.Tensor
    moves_left_logits: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Preserve historical two-value unpacking during contract migration."""

        yield self.policy_logits
        yield self.wdl_logits


def classic_rule_features(
    batch_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the frozen V1 feature vector for the classic rule."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    features = torch.zeros((batch_size, RULE_FEATURE_COUNT), dtype=dtype, device=device)
    features[:, (0, 3, 5)] = 1.0
    return features


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if groups <= channels and channels % groups == 0:
            return groups
    return 1


def _default_attention_heads(channels: int) -> int:
    for heads in (8, 4, 2):
        if channels % heads == 0:
            return heads
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


def canonical_board_to_voxels(board: torch.Tensor) -> torch.Tensor:
    """Encode canonical boards as current/opponent voxel channels [N,2,6,5,5]."""

    planes = canonical_board_to_planes(board)
    return torch.stack((planes[:, :MAX_LAYERS], planes[:, MAX_LAYERS : 2 * MAX_LAYERS]), dim=1)


def canonical_board_to_column_features(board: torch.Tensor) -> torch.Tensor:
    """Return the 14 gravity-ordered features for every board column."""

    planes = canonical_board_to_planes(board)
    return planes.permute(0, 2, 3, 1).contiguous()


def _validate_global_inputs(
    role_to_play: torch.Tensor,
    rule_features: torch.Tensor,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(role_to_play):
        raise TypeError("role_to_play must be a tensor with shape [N,2].")
    if not torch.is_tensor(rule_features):
        raise TypeError("rule_features must be a tensor with shape [N,32].")
    if role_to_play.ndim == 1 and batch_size == 1:
        role_to_play = role_to_play.unsqueeze(0)
    if rule_features.ndim == 1 and batch_size == 1:
        rule_features = rule_features.unsqueeze(0)
    if tuple(role_to_play.shape) != (batch_size, ROLE_FEATURE_COUNT):
        raise ValueError(
            f"role_to_play must have shape [{batch_size},{ROLE_FEATURE_COUNT}], "
            f"got {tuple(role_to_play.shape)}."
        )
    if tuple(rule_features.shape) != (batch_size, RULE_FEATURE_COUNT):
        raise ValueError(
            f"rule_features must have shape [{batch_size},{RULE_FEATURE_COUNT}], "
            f"got {tuple(rule_features.shape)}."
        )
    role = role_to_play.to(device=device, dtype=dtype)
    rules = rule_features.to(device=device, dtype=dtype)
    if not torch.isfinite(role).all():
        raise ValueError("role_to_play must contain finite values.")
    if not torch.all((role == 0.0) | (role == 1.0)) or not torch.all(role.sum(dim=1) == 1.0):
        raise ValueError("role_to_play rows must be FIRST/SECOND one-hot vectors.")
    if not torch.isfinite(rules).all():
        raise ValueError("rule_features must contain finite values.")
    return role, rules


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


class _PreAct3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


class _ColumnEncoder(nn.Module):
    def __init__(self, output_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * MAX_LAYERS + 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, output_channels),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        features = self.network(canonical_board_to_column_features(board))
        return features.permute(0, 3, 1, 2).contiguous()


class _Raw3DEncoder(nn.Module):
    def __init__(self, output_channels: int, branch_channels: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Conv3d(2, branch_channels, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList(_PreAct3DBlock(branch_channels) for _ in range(blocks))
        self.final_norm = nn.GroupNorm(_group_count(branch_channels), branch_channels)
        self.height_projection = nn.Conv3d(
            branch_channels,
            output_channels,
            kernel_size=(MAX_LAYERS, 1, 1),
            bias=False,
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = self.stem(canonical_board_to_voxels(board))
        for block in self.blocks:
            x = block(x)
        x = F.silu(self.final_norm(x))
        return self.height_projection(x).squeeze(2)


class _MultiViewEncoder(nn.Module):
    """Encode XY layers and the two families of vertical sections."""

    def __init__(self, output_channels: int, branch_channels: int) -> None:
        super().__init__()
        self.xy = nn.Conv2d(2 * MAX_LAYERS + 2, branch_channels, kernel_size=3, padding=1, bias=False)
        self.xz = nn.Conv2d(2, branch_channels, kernel_size=3, padding=1, bias=False)
        self.yz = nn.Conv2d(2, branch_channels, kernel_size=3, padding=1, bias=False)
        self.fusion = nn.Conv2d(3 * branch_channels, output_channels, kernel_size=1, bias=False)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        planes = canonical_board_to_planes(board)
        voxels = canonical_board_to_voxels(board)
        batch = voxels.shape[0]
        xy = self.xy(planes)

        # One XZ section for every y coordinate; collapse learned section features over z.
        xz_in = voxels.permute(0, 4, 1, 2, 3).reshape(
            batch * BOARD_SIZE, 2, MAX_LAYERS, BOARD_SIZE
        )
        xz = self.xz(xz_in).mean(dim=2)
        xz = xz.reshape(batch, BOARD_SIZE, -1, BOARD_SIZE).permute(0, 2, 3, 1)

        # One YZ section for every x coordinate, reconstructed into the same XY grid.
        yz_in = voxels.permute(0, 3, 1, 2, 4).reshape(
            batch * BOARD_SIZE, 2, MAX_LAYERS, BOARD_SIZE
        )
        yz = self.yz(yz_in).mean(dim=2)
        yz = yz.reshape(batch, BOARD_SIZE, -1, BOARD_SIZE).permute(0, 2, 1, 3)
        return self.fusion(torch.cat((xy, xz, yz), dim=1))


class _TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * mlp_ratio)))
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _weights = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class GravityPolicyValueNetV3(nn.Module):
    """Gravity-aware network with explicit role/rule conditioning and auxiliary heads."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(model_config, ModelConfig):
            raise TypeError("GravityPolicyValueNetV3 requires an explicit ModelConfig.")
        self.model_config = model_config
        channels = model_config.channels
        self.stem = nn.Conv2d(2 * MAX_LAYERS + 2, channels, kernel_size=3, padding=1, bias=False)
        global_hidden = max(16, channels)
        self.global_encoder = nn.Sequential(
            nn.Linear(ROLE_FEATURE_COUNT + RULE_FEATURE_COUNT, global_hidden),
            nn.SiLU(),
            nn.Linear(global_hidden, 2 * channels),
        )
        nn.init.zeros_(self.global_encoder[-1].weight)
        nn.init.zeros_(self.global_encoder[-1].bias)
        self.blocks = nn.ModuleList(_PreActBlock(channels) for _ in range(model_config.blocks))
        self.final_norm = nn.GroupNorm(_group_count(channels), channels)
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)
        value_hidden = max(16, channels)
        self.wdl_head = nn.Sequential(
            nn.Linear(channels * 2, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, 3),
        )
        self.opponent_reply_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.future_occupancy_head = nn.Conv2d(channels, 3 * MAX_LAYERS, kernel_size=1)
        self.moves_left_head = nn.Sequential(
            nn.Linear(channels * 2, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, MOVES_LEFT_CLASSES),
        )

    def _trunk_and_pooled(
        self,
        canonical_board: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        planes = canonical_board_to_planes(canonical_board)
        role, rules = _validate_global_inputs(
            role_to_play,
            rule_features,
            batch_size=planes.shape[0],
            dtype=planes.dtype,
            device=planes.device,
        )
        x = self.stem(planes)
        scale, bias = self.global_encoder(torch.cat((role, rules), dim=1)).chunk(2, dim=1)
        x = x * (1.0 + torch.tanh(scale).unsqueeze(-1).unsqueeze(-1))
        x = x + bias.unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            x = block(x)
        x = F.silu(self.final_norm(x))
        pooled = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        return x, pooled

    def forward_search(
        self,
        canonical_board: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> SearchOutput:
        """Compute only the policy and WDL heads used by MCTS."""

        x, pooled = self._trunk_and_pooled(
            canonical_board,
            role_to_play=role_to_play,
            rule_features=rule_features,
        )
        policy_logits = self.policy_head(x).flatten(1)
        wdl_logits = self.wdl_head(pooled)
        return SearchOutput(policy_logits=policy_logits, wdl_logits=wdl_logits)

    def forward(
        self,
        canonical_board: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> ModelOutput:
        x, pooled = self._trunk_and_pooled(
            canonical_board,
            role_to_play=role_to_play,
            rule_features=rule_features,
        )
        batch_size = x.shape[0]
        return ModelOutput(
            policy_logits=self.policy_head(x).flatten(1),
            wdl_logits=self.wdl_head(pooled),
            opponent_reply_logits=self.opponent_reply_head(x).flatten(1),
            future_occupancy_logits=self.future_occupancy_head(x).reshape(
                batch_size,
                3,
                MAX_LAYERS,
                BOARD_SIZE,
                BOARD_SIZE,
            ),
            moves_left_logits=self.moves_left_head(pooled),
        )

    def export_config(self) -> dict[str, Any]:
        return model_config_dict(self.model_config)


class ArchitecturePolicyValueNetV3(nn.Module):
    """Stage 2 architecture family with the frozen V3 input/output contract."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        if model_config.architecture == "gravity_resnet":
            raise ValueError("Use GravityPolicyValueNetV3 for gravity_resnet.")
        self.model_config = model_config
        channels = model_config.channels
        encoder_channels = model_config.encoder_channels or channels
        branch_channels = model_config.branch_channels or max(4, channels // 2)
        architecture = model_config.architecture
        self.architecture = architecture

        self.column_encoder: nn.Module | None = None
        self.multiview_encoder: nn.Module | None = None
        self.raw3d_encoder: nn.Module | None = None
        self.plane_encoder: nn.Module | None = None
        self.fusion: nn.Module | None = None

        if architecture in {"column_resnet", "column_transformer"}:
            self.column_encoder = _ColumnEncoder(channels, encoder_channels)
        elif architecture in {"multiview_resnet", "multiview_transformer"}:
            self.multiview_encoder = _MultiViewEncoder(channels, branch_channels)
        elif architecture == "raw3d_resnet":
            self.raw3d_encoder = _Raw3DEncoder(channels, branch_channels, model_config.blocks)
        elif architecture == "plane3d_fusion_resnet":
            self.plane_encoder = nn.Conv2d(
                2 * MAX_LAYERS + 2, branch_channels, kernel_size=3, padding=1, bias=False
            )
            self.raw3d_encoder = _Raw3DEncoder(
                branch_channels, branch_channels, max(1, model_config.blocks // 2)
            )
            self.fusion = nn.Conv2d(2 * branch_channels, channels, kernel_size=1, bias=False)
        elif architecture == "column3d_fusion_resnet":
            self.column_encoder = _ColumnEncoder(branch_channels, encoder_channels)
            self.raw3d_encoder = _Raw3DEncoder(
                branch_channels, branch_channels, max(1, model_config.blocks // 2)
            )
            self.fusion = nn.Conv2d(2 * branch_channels, channels, kernel_size=1, bias=False)
        else:  # ModelConfig performs the exhaustive name validation.
            raise ValueError(f"Unsupported Stage 2 architecture: {architecture}")

        global_hidden = max(16, channels)
        self.global_encoder = nn.Sequential(
            nn.Linear(ROLE_FEATURE_COUNT + RULE_FEATURE_COUNT, global_hidden),
            nn.SiLU(),
            nn.Linear(global_hidden, 2 * channels),
        )
        nn.init.zeros_(self.global_encoder[-1].weight)
        nn.init.zeros_(self.global_encoder[-1].bias)

        self.transformer_blocks: nn.ModuleList | None = None
        self.position_embedding: nn.Parameter | None = None
        if architecture.endswith("_transformer"):
            heads = model_config.attention_heads or _default_attention_heads(channels)
            ratio = model_config.transformer_mlp_ratio or 2.0
            self.position_embedding = nn.Parameter(torch.zeros(1, COLUMN_COUNT, channels))
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
            self.transformer_blocks = nn.ModuleList(
                _TransformerBlock(channels, heads, ratio) for _ in range(model_config.blocks)
            )
            self.blocks = nn.ModuleList()
        elif architecture == "raw3d_resnet":
            self.blocks = nn.ModuleList()
        else:
            self.blocks = nn.ModuleList(_PreActBlock(channels) for _ in range(model_config.blocks))

        self.final_norm = nn.GroupNorm(_group_count(channels), channels)
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)
        value_hidden = max(16, channels)
        self.wdl_head = nn.Sequential(
            nn.Linear(channels * 2, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, 3),
        )
        self.opponent_reply_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.future_occupancy_head = nn.Conv2d(channels, 3 * MAX_LAYERS, kernel_size=1)
        self.moves_left_head = nn.Sequential(
            nn.Linear(channels * 2, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, MOVES_LEFT_CLASSES),
        )

    def _encode(self, board: torch.Tensor) -> torch.Tensor:
        if self.architecture.startswith("column") and self.architecture != "column3d_fusion_resnet":
            assert self.column_encoder is not None
            return self.column_encoder(board)
        if self.architecture.startswith("multiview"):
            assert self.multiview_encoder is not None
            return self.multiview_encoder(board)
        if self.architecture == "raw3d_resnet":
            assert self.raw3d_encoder is not None
            return self.raw3d_encoder(board)
        if self.architecture == "plane3d_fusion_resnet":
            assert self.plane_encoder is not None and self.raw3d_encoder is not None
            assert self.fusion is not None
            return self.fusion(
                torch.cat(
                    (self.plane_encoder(canonical_board_to_planes(board)), self.raw3d_encoder(board)),
                    dim=1,
                )
            )
        assert self.architecture == "column3d_fusion_resnet"
        assert self.column_encoder is not None and self.raw3d_encoder is not None
        assert self.fusion is not None
        return self.fusion(torch.cat((self.column_encoder(board), self.raw3d_encoder(board)), dim=1))

    def _trunk_and_pooled(
        self,
        canonical_board: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(canonical_board)
        role, rules = _validate_global_inputs(
            role_to_play,
            rule_features,
            batch_size=x.shape[0],
            dtype=x.dtype,
            device=x.device,
        )
        scale, bias = self.global_encoder(torch.cat((role, rules), dim=1)).chunk(2, dim=1)
        x = x * (1.0 + torch.tanh(scale).unsqueeze(-1).unsqueeze(-1))
        x = x + bias.unsqueeze(-1).unsqueeze(-1)
        if self.transformer_blocks is not None:
            tokens = x.flatten(2).transpose(1, 2) + self.position_embedding
            for block in self.transformer_blocks:
                tokens = block(tokens)
            x = tokens.transpose(1, 2).reshape(-1, self.model_config.channels, BOARD_SIZE, BOARD_SIZE)
        else:
            for block in self.blocks:
                x = block(x)
        x = F.silu(self.final_norm(x))
        pooled = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        return x, pooled

    def forward_search(
        self,
        canonical_board: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> SearchOutput:
        x, pooled = self._trunk_and_pooled(
            canonical_board, role_to_play=role_to_play, rule_features=rule_features
        )
        return SearchOutput(self.policy_head(x).flatten(1), self.wdl_head(pooled))

    def forward(
        self,
        canonical_board: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> ModelOutput:
        x, pooled = self._trunk_and_pooled(
            canonical_board, role_to_play=role_to_play, rule_features=rule_features
        )
        batch_size = x.shape[0]
        return ModelOutput(
            policy_logits=self.policy_head(x).flatten(1),
            wdl_logits=self.wdl_head(pooled),
            opponent_reply_logits=self.opponent_reply_head(x).flatten(1),
            future_occupancy_logits=self.future_occupancy_head(x).reshape(
                batch_size, 3, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE
            ),
            moves_left_logits=self.moves_left_head(pooled),
        )

    def export_config(self) -> dict[str, Any]:
        return model_config_dict(self.model_config)


def _model_config_from_mapping(raw: Mapping[str, Any]) -> ModelConfig:
    allowed = {
        "architecture",
        "channels",
        "blocks",
        "encoder_channels",
        "branch_channels",
        "attention_heads",
        "transformer_mlp_ratio",
        "global_input_schema",
        "output_schema",
        "rule_feature_dim",
        "moves_left_classes",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown model config field(s): {', '.join(unknown)}")
    return ModelConfig(**dict(raw))


def build_model(
    model_config: ModelConfig | Mapping[str, Any],
) -> GravityPolicyValueNetV3 | ArchitecturePolicyValueNetV3:
    if isinstance(model_config, Mapping):
        model_config = _model_config_from_mapping(model_config)
    if not isinstance(model_config, ModelConfig):
        raise TypeError("build_model requires an explicit ModelConfig or model config mapping.")
    if model_config.architecture == "gravity_resnet":
        return GravityPolicyValueNetV3(model_config)
    return ArchitecturePolicyValueNetV3(model_config)


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
    """Search inference adapter with mandatory absolute-role and rule context."""

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        canonical_board: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        policies, wdl = self.predict_batch(
            np.asarray(canonical_board)[None, ...],
            role_to_play=np.asarray(role_to_play)[None, ...]
            if np.asarray(role_to_play).ndim == 1
            else np.asarray(role_to_play),
            rule_features=np.asarray(rule_features)[None, ...]
            if np.asarray(rule_features).ndim == 1
            else np.asarray(rule_features),
        )
        return policies[0], wdl[0]

    def predict_batch(
        self,
        canonical_boards: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        role = torch.as_tensor(role_to_play, dtype=torch.float32, device=self.device)
        rules = torch.as_tensor(rule_features, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            output = self.model.forward_search(
                board,
                role_to_play=role,
                rule_features=rules,
            )
            policy_logits, wdl_logits = output
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
    "GLOBAL_INPUT_SCHEMA",
    "LEGACY_ACTION_COUNT",
    "MOVES_LEFT_CLASSES",
    "ModelOutput",
    "OUTPUT_SCHEMA",
    "ROLE_FEATURE_COUNT",
    "ROLE_FEATURE_NAMES",
    "RULE_FEATURE_COUNT",
    "RULE_FEATURE_NAMES",
    "SearchOutput",
    "WDL_DRAW",
    "WDL_LOSS",
    "WDL_WIN",
    "GravityPolicyValueNetV3",
    "ArchitecturePolicyValueNetV3",
    "LegacyPolicyValueAdapter",
    "TorchPredictor",
    "build_model",
    "canonical_board_to_planes",
    "canonical_board_to_column_features",
    "canonical_board_to_voxels",
    "classic_rule_features",
    "column_policy_to_legacy",
    "column_to_legacy_action",
    "legacy_action_to_column",
    "legacy_policy_to_columns",
    "legal_column_mask",
    "wdl_expected_value",
    "wdl_logits_expected_value",
]
