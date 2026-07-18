import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from game_rules import BOARD_SIZE, MAX_LAYERS


ACTION_SIZE = MAX_LAYERS * BOARD_SIZE * BOARD_SIZE
COLUMN_COUNT = BOARD_SIZE * BOARD_SIZE


def _safe_illegal_logit(tensor, requested=-1e9):
    if not tensor.dtype.is_floating_point:
        return float(requested)
    return max(float(requested), float(torch.finfo(tensor.dtype).min))


def _group_count(channels):
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def board_channels_to_gravity_2d(board_channels):
    """Convert [N,2,6,5,5] piece planes into 14 gravity-aware 2D planes."""
    if board_channels.ndim != 5 or tuple(board_channels.shape[1:]) != (2, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected [N,2,{MAX_LAYERS},{BOARD_SIZE},{BOARD_SIZE}], got {tuple(board_channels.shape)}")
    batch = board_channels.shape[0]
    layer_planes = board_channels.reshape(batch, 2 * MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
    occupancy = board_channels.sum(dim=1).sum(dim=1, keepdim=True)
    normalized_height = occupancy / float(MAX_LAYERS)
    legal_column = (occupancy < float(MAX_LAYERS)).to(dtype=board_channels.dtype)
    return torch.cat((layer_planes, normalized_height, legal_column), dim=1)


def column_valid_mask(board_channels):
    occupancy = board_channels.sum(dim=1).sum(dim=1)
    return occupancy < float(MAX_LAYERS)


def expand_column_log_policy(column_log_policy, board_channels, illegal_logit=-1e9):
    """Scatter 25 legal-column log probabilities into the legacy 150-action space."""
    batch = board_channels.shape[0]
    occupancy = board_channels.sum(dim=1).sum(dim=1).to(dtype=torch.long)
    valid = occupancy < MAX_LAYERS
    safe_height = occupancy.clamp(min=0, max=MAX_LAYERS - 1)
    column_index = torch.arange(COLUMN_COUNT, device=board_channels.device).view(1, BOARD_SIZE, BOARD_SIZE)
    action_index = safe_height * COLUMN_COUNT + column_index
    safe_illegal_logit = _safe_illegal_logit(column_log_policy, illegal_logit)
    expanded = column_log_policy.new_full((batch, ACTION_SIZE), safe_illegal_logit)
    expanded.scatter_(1, action_index.reshape(batch, -1), column_log_policy.reshape(batch, -1))
    expanded_valid = torch.zeros_like(expanded, dtype=torch.bool)
    expanded_valid.scatter_(1, action_index.reshape(batch, -1), valid.reshape(batch, -1))
    return torch.where(expanded_valid, expanded, expanded.new_full((), safe_illegal_logit))


class PreAct2DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


class Factorized3DBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.horizontal = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.vertical = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        nn.init.zeros_(self.vertical.weight)

    def forward(self, x):
        residual = x
        x = self.horizontal(F.silu(self.norm1(x)))
        x = self.vertical(F.silu(self.norm2(x)))
        return residual + x


class ColumnAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        heads = max(1, min(8, channels // 32))
        while channels % heads:
            heads -= 1
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature_map):
        batch, channels, rows, cols = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, rows, cols)


class GravityPolicyValueNet(nn.Module):
    """Experimental gravity-aware network preserving the legacy 150-action output."""

    def __init__(
        self,
        num_channels=128,
        num_res_blocks=6,
        backbone_type="layer2d",
        global_context_blocks=0,
    ):
        super().__init__()
        self.board_layers = MAX_LAYERS
        self.board_size = BOARD_SIZE
        self.num_channels = int(num_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.backbone_type = str(backbone_type)
        self.global_context_blocks = int(global_context_blocks)

        if self.backbone_type == "layer2d":
            self.stem = nn.Conv2d(2 * MAX_LAYERS + 2, self.num_channels, 3, padding=1, bias=False)
            self.blocks = nn.ModuleList([PreAct2DBlock(self.num_channels) for _ in range(self.num_res_blocks)])
        elif self.backbone_type == "factorized3d":
            self.stem = nn.Conv3d(2, self.num_channels, 3, padding=1, bias=False)
            self.blocks = nn.ModuleList([Factorized3DBlock(self.num_channels) for _ in range(self.num_res_blocks)])
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

        groups = _group_count(self.num_channels)
        self.final_norm = nn.GroupNorm(groups, self.num_channels)
        self.global_blocks = nn.ModuleList(
            [ColumnAttentionBlock(self.num_channels) for _ in range(self.global_context_blocks)]
        )
        self.policy_head = nn.Conv2d(self.num_channels, 1, 1)
        value_hidden = max(32, self.num_channels // 2)
        self.value_head = nn.Sequential(
            nn.Linear(self.num_channels * 2, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def _features(self, board_channels):
        if self.backbone_type == "layer2d":
            x = self.stem(board_channels_to_gravity_2d(board_channels))
            for block in self.blocks:
                x = block(x)
        else:
            x = self.stem(board_channels)
            for block in self.blocks:
                x = block(x)
            x = x.mean(dim=2)
        x = F.silu(self.final_norm(x))
        for block in self.global_blocks:
            x = block(x)
        return x

    def forward_columns(self, board_channels):
        feature_map = self._features(board_channels)
        logits = self.policy_head(feature_map).flatten(1)
        valid = column_valid_mask(board_channels).flatten(1)
        any_valid = valid.any(dim=1, keepdim=True)
        # Boolean-valued Where is not implemented by the CPU ONNX Runtime
        # provider. This expression is exactly equivalent: preserve `valid`
        # when at least one column is legal, otherwise make every column safe.
        safe_valid = valid | ~any_valid
        masked_logits = logits.masked_fill(~safe_valid, _safe_illegal_logit(logits))
        column_log_policy = F.log_softmax(masked_logits, dim=1)
        pooled = torch.cat((feature_map.mean(dim=(2, 3)), feature_map.amax(dim=(2, 3))), dim=1)
        value = self.value_head(pooled)
        return column_log_policy, value

    def forward(self, board_channels):
        column_log_policy, value = self.forward_columns(board_channels)
        return expand_column_log_policy(column_log_policy, board_channels), value

    def model_config(self):
        return {
            "architecture": "gravity_resnet_v1",
            "policy_space": "columns25",
            "backbone_type": self.backbone_type,
            "board_layers": MAX_LAYERS,
            "board_size": BOARD_SIZE,
            "input_channels": 2,
            "num_channels": self.num_channels,
            "num_res_blocks": self.num_res_blocks,
            "global_context_blocks": self.global_context_blocks,
            "normalization": "group_norm",
        }


EXPERIMENTAL_MODEL_PRESETS = {
    "mini": {"num_channels": 64, "num_res_blocks": 4, "backbone_type": "layer2d", "global_context_blocks": 0},
    "balanced": {"num_channels": 128, "num_res_blocks": 6, "backbone_type": "layer2d", "global_context_blocks": 1},
    "flagship": {"num_channels": 192, "num_res_blocks": 8, "backbone_type": "layer2d", "global_context_blocks": 2},
    "flagship_wide": {"num_channels": 224, "num_res_blocks": 8, "backbone_type": "layer2d", "global_context_blocks": 2},
    "factorized3d": {"num_channels": 128, "num_res_blocks": 6, "backbone_type": "factorized3d", "global_context_blocks": 1},
}


def build_experimental_model(name):
    try:
        config = EXPERIMENTAL_MODEL_PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown experimental model preset: {name}") from exc
    return GravityPolicyValueNet(**config)
