"""Additive Stage 2A-R3 deployment-tier experiment design.

This module is deliberately declarative.  It prepares the next experiment
matrix without selecting finalists, creating run directories, or starting
training while the current Stage 2B evidence is still being completed.
"""

from __future__ import annotations

from typing import Any, Iterable


ROUND3_DEPENDENCY_ID = "R2_PRIMARY_SEED_COMPLETE"
ROUND3_PRIMARY_SEED = 271828
ROUND3_TARGET_POSITIONS = 5_000_000
ROUND3_PRIMARY_ARCHITECTURES = (
    "gravity_resnet",
    "plane3d_fusion_resnet",
    "column3d_fusion_resnet",
)


def _factorial_cells(
    parameter_anchors: Iterable[str], consumed_positions: Iterable[int]
) -> list[dict[str, Any]]:
    return [
        {
            "parameter_anchor": anchor,
            "consumed_positions": positions,
            "checkpoint_rule": (
                "the 3M cell must exactly resume the matching 1M checkpoint"
                if positions == 3_000_000
                else "random initialization with the frozen sample order"
            ),
        }
        for anchor in parameter_anchors
        for positions in consumed_positions
    ]


def build_round3_design() -> dict[str, Any]:
    """Return the deterministic, plan-only Stage 2A-R3 experiment manifest."""

    flash_candidates = [
        {
            "variant_id": "flash_gravity_resnet",
            "architecture": "gravity_resnet",
            "purpose": "unchanged deployment baseline",
        },
        {
            "variant_id": "flash_column_resnet",
            "architecture": "column_resnet",
            "purpose": "gravity-column representation control",
        },
        {
            "variant_id": "flash_column_bottleneck_resnet",
            "architecture": "column_bottleneck_resnet",
            "purpose": "wider representation with a compact convolutional trunk",
        },
        {
            "variant_id": "flash_column_conv_attention",
            "architecture": "column_conv_attention",
            "purpose": "compact local convolution plus sparse global attention",
        },
    ]
    balance_candidates = [
        {
            "variant_id": "balance_gravity_control",
            "architecture": "gravity_resnet",
            "uses_explicit_3d_information": False,
            "purpose": "pure 2D efficiency and attribution control",
        },
        {
            "variant_id": "balance_raw3d_control",
            "architecture": "raw3d_resnet",
            "uses_explicit_3d_information": True,
            "purpose": "dense voxel 3D baseline",
        },
        {
            "variant_id": "balance_raw3d_to2d",
            "architecture": "raw3d_to2d_resnet",
            "uses_explicit_3d_information": True,
            "purpose": "shallow dense 3D representation followed by a 2D trunk",
        },
        {
            "variant_id": "balance_factorized3d",
            "architecture": "factorized3d_resnet",
            "uses_explicit_3d_information": True,
            "purpose": "factorized spatial and vertical 3D convolution",
        },
        {
            "variant_id": "balance_plane3d_fusion_v2",
            "architecture": "plane3d_fusion_v2",
            "uses_explicit_3d_information": True,
            "purpose": "14-plane branch with independently budgeted raw-3D branch",
        },
        {
            "variant_id": "balance_column3d_fusion_v2",
            "architecture": "column3d_fusion_v2",
            "uses_explicit_3d_information": True,
            "purpose": "vertical-column prior fused with raw voxels",
        },
        {
            "variant_id": "balance_multiview3d_fusion",
            "architecture": "multiview3d_fusion_resnet",
            "uses_explicit_3d_information": True,
            "purpose": "strengthened section encoders retained beside a raw-3D branch",
        },
        {
            "variant_id": "balance_winning3d_fusion",
            "architecture": "winning3d_fusion_resnet",
            "uses_explicit_3d_information": True,
            "purpose": "winning-section views fused with raw-3D geometry",
        },
    ]
    pro_candidates = [
        {
            "variant_id": "pro_a0_column_absolute",
            "family": "pro_transformer",
            "tokenization": "column_mlp_v1",
            "position_encoding": "absolute_2d_v1",
            "residual_routing": "vanilla",
        },
        {
            "variant_id": "pro_a1_column_relative_d4",
            "family": "pro_transformer",
            "tokenization": "column_mlp_v1",
            "position_encoding": "relative_d4_2d_v1",
            "residual_routing": "vanilla",
        },
        {
            "variant_id": "pro_t1_layer_aware_columns",
            "family": "pro_transformer",
            "tokenization": "layer_aware_columns_v1",
            "position_encoding": "absolute_2d_v1",
            "residual_routing": "vanilla",
        },
        {
            "variant_id": "pro_t2_columns_plus_layers",
            "family": "pro_transformer",
            "tokenization": "columns_plus_layer_summaries_v1",
            "position_encoding": "typed_absolute_v1",
            "residual_routing": "vanilla",
        },
        {
            "variant_id": "pro_t3_cells3d",
            "family": "pro_transformer",
            "tokenization": "cells3d_v1",
            "position_encoding": "absolute_3d_v1",
            "residual_routing": "vanilla",
        },
    ]

    return {
        "schema": "connect4-v3-stage2-round3-design-v1",
        "stage": "Stage 2A-R3",
        "mode": "preparation_only",
        "execution_authorized": False,
        "dependency": {
            "id": ROUND3_DEPENDENCY_ID,
            "evidence_schema": "connect4-v3-stage2-r2-primary-evidence-v1",
            "readiness_receipt_schema": "connect4-v3-stage2-round3-readiness-v1",
            "primary_seed": ROUND3_PRIMARY_SEED,
            "target_positions": ROUND3_TARGET_POSITIONS,
            "required_lines": [
                {
                    "architecture": architecture,
                    "initializations": ["cold", "warm"],
                }
                for architecture in ROUND3_PRIMARY_ARCHITECTURES
            ],
            "terminal_rule": "5M complete_at_bound or explicit safe_guard_stop",
            "required_evidence": [
                "checkpoint hash",
                "offline/final report hash",
                "fixed-opening color-swapped anchored Elo hash",
                "queue-state hash",
                "archive index and verified receipt hashes",
            ],
        },
        "deployment_tiers": {
            "flash": {
                "target": "ordinary CPU, 512 simulations near or below 3 seconds",
                "candidates": flash_candidates,
                "screen": {
                    "parameter_anchor": "b8",
                    "consumed_positions": 1_000_000,
                    "seed": ROUND3_PRIMARY_SEED,
                },
                "promoted_scaling": {
                    "selector": "all candidates not stably dominated at the 1M screen",
                    "factorial_cells": _factorial_cells(
                        ("b8", "b10"), (1_000_000, 3_000_000)
                    ),
                },
            },
            "balance": {
                "target": "CPU usable near a 10-second 512-simulation design point",
                "priority": "explicit 3D information and 3D/2D fusion",
                "candidates": balance_candidates,
                "diagnostic": {
                    "parameter_anchor": "b6",
                    "consumed_positions": 250_000,
                    "elimination_rule": "numerical/interface failure or stable domination only",
                },
                "scaling": {
                    "selector": "baseline and every non-dominated explicit-3D candidate",
                    "factorial_cells": _factorial_cells(
                        ("b6", "b8"), (1_000_000, 3_000_000)
                    ),
                    "branch_allocation_control": [
                        "approximately 25 percent of parameters in the 3D branch",
                        "approximately 50 percent of parameters in the 3D branch",
                    ],
                },
                "large_capacity_confirmation": {
                    "selector": "gravity control plus the best two positively scaling 3D families",
                    "parameter_anchor": "b10",
                    "consumed_positions": [3_000_000, 5_000_000],
                    "five_million_rule": "run only while the 3M trend remains unresolved or rising",
                },
                "frozen_3d_followup": {
                    "base_model": {
                        "architecture": "column3d_fusion_v2",
                        "channels": 248,
                        "blocks": 10,
                        "encoder_channels": 248,
                        "branch_channels": 64,
                        "volume_channels": 96,
                        "volume_blocks": 5,
                        "collapse_mode": "learned",
                        "fusion_mode": "concat",
                    },
                    "depth_reserve": {
                        "volume_blocks": 7,
                        "rule": "run only with spare compute or focused B5/B7 confirmation",
                    },
                    "representation_control": {
                        "architecture": "winning3d_fusion_resnet",
                        "volume_channels": 96,
                        "volume_blocks": 5,
                        "fusion_mode": "concat",
                    },
                    "post_trunk_screen": [
                        {"id": "T0", "post_trunk_mode": "none"},
                        {
                            "id": "T1",
                            "post_trunk_mode": "serial_attention",
                            "post_attention_blocks": 2,
                            "post_attention_heads": 8,
                            "post_attention_mlp_ratio": 2.0,
                        },
                        {
                            "id": "T2",
                            "post_trunk_mode": "parallel_attention",
                            "post_attention_blocks": 2,
                            "post_attention_heads": 8,
                            "post_attention_mlp_ratio": 2.0,
                        },
                    ],
                    "encoder_width_screen": [64, 96, 128],
                    "new_training_cells": ["T1", "T2", "E96", "E128"],
                    "positions": [1_000_000, 3_000_000],
                    "combination_rule": (
                        "combine only independently positive tail and encoder-width factors"
                    ),
                    "optional_trunk_depth": {
                        "blocks": 12,
                        "after": "post-trunk and encoder-width factors are frozen",
                    },
                },
            },
            "pro": {
                "target": "GPU recommended; no CPU latency elimination gate",
                "candidates": pro_candidates,
                "tokenization_screen": {
                    "parameter_anchor": "b10",
                    "consumed_positions": 1_000_000,
                    "seed": ROUND3_PRIMARY_SEED,
                    "rule": "change token organization or position encoding one factor at a time",
                },
                "routing_screen": {
                    "selector": "best tokenization only",
                    "depth_anchor": "b12",
                    "residual_routing": ["vanilla", "gated_input_skip", "attnres"],
                    "rule": "AttnRes must beat both controls before b16",
                },
                "promoted_scaling": {
                    "selector": "best tokenization and routing combination",
                    "parameter_anchors": ["approximately_12m", "approximately_25_to_35m"],
                    "factorial_cells": _factorial_cells(
                        ("approximately_12m", "approximately_25_to_35m"),
                        (1_000_000, 3_000_000),
                    ),
                    "optional_anchor": "approximately_60m only after two positive scaling points",
                },
            },
        },
        "implementation_scope": {
            "ready_for_smoke": [
                "gravity_resnet",
                "column_resnet",
                "raw3d_resnet",
                "raw3d_to2d_resnet",
                "factorized3d_resnet",
                "plane3d_fusion_v2",
                "column3d_fusion_v2",
                "multiview3d_fusion_resnet",
                "winning3d_fusion_resnet",
            ],
            "planned_after_primary_evidence": [
                "column_bottleneck_resnet",
                "column_conv_attention",
                "pro_transformer tokenization and routing variants",
            ],
            "rule": (
                "R2 readiness does not waive architecture implementation, focused tests, "
                "or dry-run preflight for a selected Round 3 cell."
            ),
        },
        "scaling_contract": {
            "required_consumed_positions": [1_000_000, 3_000_000],
            "rule": (
                "Every promoted parameter anchor must contain both the 1M and 3M cells; "
                "the 3M cell resumes the matching 1M checkpoint, and diagonal-only comparisons "
                "cannot support a scaling claim."
            ),
            "seeds": {
                "primary": ROUND3_PRIMARY_SEED,
                "confirmation": 314159,
                "confirmation_rule": "finalists, close results, and anomalous stability only",
            },
        },
        "shared_evaluation": {
            "offline": [
                "four-regime policy CE/JSD/top-1 agreement",
                "WDL CE/Brier/ECE/accuracy",
                "auxiliary losses and accuracy",
                "geometry-stratified tactical metrics with bootstrap confidence intervals",
            ],
            "strength": [
                "fixed-opening color-swapped anchored Elo",
                "fixed-simulation matches",
                "fixed-time matches",
            ],
            "efficiency": [
                "parameters and component parameter shares",
                "search MACs/FLOPs",
                "training throughput and GPU-hours",
                "CPU/GPU latency distributions and peak memory",
            ],
        },
        "execution_rule": (
            "Design generation, model implementation, tests, and dry-runs may proceed now. "
            "No Round 3 training may start without a verified readiness receipt whose dependency "
            f"id is {ROUND3_DEPENDENCY_ID}."
        ),
    }


__all__ = [
    "ROUND3_DEPENDENCY_ID",
    "ROUND3_PRIMARY_ARCHITECTURES",
    "ROUND3_PRIMARY_SEED",
    "ROUND3_TARGET_POSITIONS",
    "build_round3_design",
]
