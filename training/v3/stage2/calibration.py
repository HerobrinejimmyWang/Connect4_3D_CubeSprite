"""Parameter matching and lightweight inference profiling for Stage 2."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from ..config import ModelConfig, model_config_dict
from ..model import build_model, classic_rule_features


ARCHITECTURES = (
    "gravity_resnet",
    "column_resnet",
    "multiview_resnet",
    "raw3d_resnet",
    "plane3d_fusion_resnet",
    "column3d_fusion_resnet",
    "column_transformer",
    "multiview_transformer",
)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_search_macs(model: torch.nn.Module) -> int:
    """Estimate a batch-1 search forward in multiply-accumulate operations."""

    total = 0
    hooks: list[Any] = []

    def convolution(module: Any, _inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        kernel = 1
        for size in module.kernel_size:
            kernel *= int(size)
        total += int(output.numel()) * (module.in_channels // module.groups) * kernel

    def linear(module: torch.nn.Linear, _inputs: Any, output: torch.Tensor) -> None:
        nonlocal total
        total += int(output.numel()) * module.in_features

    def attention(module: torch.nn.MultiheadAttention, inputs: Any, _output: Any) -> None:
        nonlocal total
        batch, tokens, channels = inputs[0].shape
        total += 4 * batch * tokens * channels * channels
        total += 2 * batch * tokens * tokens * channels

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            hooks.append(module.register_forward_hook(convolution))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear))
        elif isinstance(module, torch.nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(attention))
    device = next(model.parameters()).device
    board = torch.zeros((1, 6, 5, 5), dtype=torch.float32, device=device)
    role = torch.tensor(((1.0, 0.0),), dtype=torch.float32, device=device)
    rules = classic_rule_features(1, device=device)
    try:
        with torch.inference_mode():
            model.forward_search(board, role_to_play=role, rule_features=rules)
    finally:
        for hook in hooks:
            hook.remove()
    return total


def _profile_forward(model: torch.nn.Module, *, repeats: int = 8) -> dict[str, float]:
    model.eval()
    board = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    role = torch.tensor(((1.0, 0.0),), dtype=torch.float32)
    rules = classic_rule_features(1)
    with torch.inference_mode():
        for _ in range(2):
            model.forward_search(board, role_to_play=role, rule_features=rules)
        started = time.perf_counter()
        for _ in range(repeats):
            model.forward_search(board, role_to_play=role, rule_features=rules)
    return {"cpu_batch1_ms": 1000.0 * (time.perf_counter() - started) / repeats}


def calibrate_architecture_matrix(
    *,
    anchor_channels: int = 128,
    anchor_blocks: int = 6,
    tolerance: float = 0.05,
    minimum_channels: int = 8,
    maximum_channels: int = 256,
    profile_latency: bool = False,
) -> dict[str, Any]:
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be in (0, 1)")
    anchor_config = ModelConfig(channels=anchor_channels, blocks=anchor_blocks)
    anchor_parameters = parameter_count(build_model(anchor_config))
    rows = []
    for architecture in ARCHITECTURES:
        if architecture == "gravity_resnet":
            candidates = [(anchor_config, anchor_parameters)]
        else:
            candidates = []
            for channels in range(minimum_channels, maximum_channels + 1, 8):
                config = ModelConfig(architecture=architecture, channels=channels, blocks=anchor_blocks)
                candidates.append((config, parameter_count(build_model(config))))
        within = [item for item in candidates if abs(item[1] - anchor_parameters) / anchor_parameters <= tolerance]
        if within:
            selected = min(within, key=lambda item: (abs(item[1] - anchor_parameters), item[1]))
            status = "within_tolerance"
        else:
            lower = [item for item in candidates if item[1] <= anchor_parameters]
            selected = max(lower, key=lambda item: item[1]) if lower else min(candidates, key=lambda item: item[1])
            status = "nearest_not_above" if lower else "minimum_exceeds_anchor"
        config, parameters = selected
        model = build_model(config)
        row: dict[str, Any] = {
            "architecture": architecture,
            "model": model_config_dict(config),
            "parameters": parameters,
            "search_macs_estimate": estimate_search_macs(model),
            "parameter_ratio": parameters / anchor_parameters,
            "match_status": status,
            "resolved": {
                "encoder_channels": config.encoder_channels or config.channels,
                "branch_channels": config.branch_channels or max(4, config.channels // 2),
                "attention_heads": config.attention_heads or next(
                    heads for heads in (8, 4, 2, 1) if config.channels % heads == 0
                ),
                "transformer_mlp_ratio": config.transformer_mlp_ratio or 2.0,
            },
        }
        if profile_latency:
            row["profile"] = _profile_forward(model)
        row["search_flops_estimate"] = 2 * row["search_macs_estimate"]
        rows.append(row)
    return {
        "schema": "connect4-v3-stage2-architecture-matrix-v1",
        "anchor": {
            "model": model_config_dict(anchor_config),
            "parameters": anchor_parameters,
            "tolerance": tolerance,
        },
        "architectures": rows,
        "notes": {
            "flops": "Hook-based estimate uses two FLOPs per MAC; confirm finalists with the target profiler.",
            "latency": "Optional local CPU batch-1 forward latency is evidence, not a Stage 2A gate.",
        },
    }


def write_architecture_matrix(path: str | Path, matrix: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
