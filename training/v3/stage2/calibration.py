"""Parameter matching and lightweight inference profiling for Stage 2."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    "multiview_winning_resnet",
    "multiview_winning_transformer",
)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def component_parameter_breakdown(model: torch.nn.Module) -> dict[str, int]:
    """Partition parameters by experimental role without changing model state keys."""

    components = {
        "representation_2d": 0,
        "volume_3d": 0,
        "fusion": 0,
        "trunk": 0,
        "conditioning": 0,
        "heads": 0,
        "other": 0,
    }
    representation_roots = {"column_encoder", "multiview_encoder", "plane_encoder", "stem"}
    volume_roots = {"raw3d_encoder", "volume_encoder"}
    head_roots = {
        "final_norm",
        "policy_head",
        "wdl_head",
        "opponent_reply_head",
        "future_occupancy_head",
        "moves_left_head",
    }
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        if root in representation_roots:
            bucket = "representation_2d"
        elif root in volume_roots:
            bucket = "volume_3d"
        elif root == "fusion":
            bucket = "fusion"
        elif root in {"blocks", "transformer_blocks", "position_embedding"}:
            bucket = "trunk"
        elif root == "global_encoder":
            bucket = "conditioning"
        elif root in head_roots:
            bucket = "heads"
        else:
            bucket = "other"
        components[bucket] += int(parameter.numel())
    components["total"] = sum(components.values())
    return components


def calibrate_model_grid(
    *,
    variant_id: str,
    architecture: str,
    anchor_channels: int,
    anchor_blocks: int,
    tolerance: float = 0.05,
    minimum_channels: int = 8,
    maximum_channels: int = 384,
    branch_fractions: Sequence[float] = (0.25, 0.5),
    volume_blocks: Sequence[int] = (1, 2, 3),
    collapse_modes: Sequence[str] = ("learned",),
    fusion_modes: Sequence[str] = ("concat",),
    fixed_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Match a multi-axis Stage 2 candidate against a gravity parameter anchor.

    Unlike the original Stage 2 calibrator, this searches branch width and 3D
    depth as first-class variables.  The result is descriptive only; it never
    launches training.
    """

    if not variant_id:
        raise ValueError("variant_id must be non-empty")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be in (0, 1)")
    if minimum_channels < 4 or maximum_channels < minimum_channels:
        raise ValueError("invalid channel search range")
    if any(not 0.0 < value <= 1.0 for value in branch_fractions):
        raise ValueError("branch fractions must be in (0, 1]")
    if any(value < 1 for value in volume_blocks):
        raise ValueError("volume block candidates must be positive")

    anchor_config = ModelConfig(channels=anchor_channels, blocks=anchor_blocks)
    anchor_parameters = parameter_count(build_model(anchor_config))
    fixed = dict(fixed_fields or {})
    candidates: list[tuple[ModelConfig, int]] = []
    uses_v2_volume = architecture in {
        "raw3d_to2d_resnet",
        "factorized3d_resnet",
        "plane3d_fusion_v2",
        "column3d_fusion_v2",
        "multiview3d_fusion_resnet",
        "winning3d_fusion_resnet",
    }
    uses_fusion = architecture in {
        "plane3d_fusion_v2",
        "column3d_fusion_v2",
        "multiview3d_fusion_resnet",
        "winning3d_fusion_resnet",
    }
    for channels in range(minimum_channels, maximum_channels + 1, 8):
        fractions = branch_fractions if uses_v2_volume else (0.5,)
        depths = volume_blocks if uses_v2_volume else (0,)
        collapses = collapse_modes if uses_v2_volume else ("",)
        fusions = fusion_modes if uses_fusion else ("",)
        for fraction in fractions:
            branch_channels = max(4, int(round(channels * fraction / 4.0)) * 4)
            for depth in depths:
                for collapse in collapses:
                    for fusion in fusions:
                        raw: dict[str, Any] = {
                            "architecture": architecture,
                            "channels": channels,
                            "blocks": anchor_blocks,
                            **fixed,
                        }
                        if uses_v2_volume:
                            raw.update(
                                branch_channels=branch_channels,
                                volume_blocks=depth,
                                collapse_mode=collapse,
                            )
                        if uses_fusion:
                            raw["fusion_mode"] = fusion
                        config = ModelConfig(**raw)
                        candidates.append((config, parameter_count(build_model(config))))

    within = [
        item
        for item in candidates
        if abs(item[1] - anchor_parameters) / anchor_parameters <= tolerance
    ]
    if within:
        selected = min(within, key=lambda item: (abs(item[1] - anchor_parameters), item[1]))
        status = "within_tolerance"
    else:
        lower = [item for item in candidates if item[1] <= anchor_parameters]
        selected = (
            max(lower, key=lambda item: item[1])
            if lower
            else min(candidates, key=lambda item: item[1])
        )
        status = "nearest_not_above" if lower else "minimum_exceeds_anchor"
    config, parameters = selected
    model = build_model(config)
    breakdown = component_parameter_breakdown(model)
    return {
        "variant_id": variant_id,
        "architecture": architecture,
        "model": model_config_dict(config),
        "parameters": parameters,
        "parameter_ratio": parameters / anchor_parameters,
        "match_status": status,
        "component_parameters": breakdown,
        "component_parameter_fractions": {
            key: value / parameters
            for key, value in breakdown.items()
            if key != "total"
        },
        "search_macs_estimate": estimate_search_macs(model),
        "search_flops_estimate": 2 * estimate_search_macs(model),
        "anchor": {
            "model": model_config_dict(anchor_config),
            "parameters": anchor_parameters,
            "tolerance": tolerance,
        },
    }


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
