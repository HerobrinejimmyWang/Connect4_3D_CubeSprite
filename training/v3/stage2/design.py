"""Freeze the Stage 2 architecture x parameter x data experiment design."""

from __future__ import annotations

from typing import Any

from .calibration import ARCHITECTURES, calibrate_architecture_matrix


def build_experiment_design(*, tolerance: float = 0.05) -> dict[str, Any]:
    anchors = {
        "b6": (128, 6),
        "b8": (192, 8),
        "b10": (256, 10),
    }
    scaling = {
        name: calibrate_architecture_matrix(
            anchor_channels=channels,
            anchor_blocks=blocks,
            tolerance=tolerance,
            maximum_channels=max(256, channels * 2),
        )
        for name, (channels, blocks) in anchors.items()
    }
    return {
        "schema": "connect4-v3-stage2-experiment-design-v2",
        "data_regimes": {
            "screen": ["standard_early", "standard_mid", "standard_late"],
            "evaluation": [
                "standard_early",
                "standard_mid",
                "standard_late",
                "mixed_late",
            ],
            "promotion": "mixed_late",
            "promotion_rule": (
                "mixed_late may train only a promoted architecture from an exact standard_late "
                "model/scale checkpoint; both the standard continuation control and mixed branch "
                "reset optimizer, scheduler, augmentation stream, and replay cursor"
            ),
        },
        "parameter_anchors": scaling,
        "elo_protocol": {
            "path": "training/v3/configs/stage2_elo_protocol_v1.json",
            "registry_hash": "806753498c10ce585a9b7586276eaa9037637be2b050072d0b684b9461773b79",
            "primary_profile": "primary_256",
            "final_profile": "final_512",
            "pressure_profile": "pressure_256v512",
            "rule": "fresh Stage 2 matches may use only the matching frozen v3 scale",
        },
        "phases": [
            {
                "phase": "A0",
                "purpose": "matched-parameter architecture screen",
                "architectures": list(ARCHITECTURES),
                "parameter_anchor": "b6",
                "train_regimes": ["standard_early", "standard_mid", "standard_late"],
                "unique_pool_positions": 1_000_000,
                "consumed_positions": 250_000,
                "seeds": [271828],
            },
            {
                "phase": "A1",
                "purpose": "two-seed standard-regime confirmation",
                "architecture_selector": "round1 promotion rule, at most six including baseline",
                "parameter_anchor": "b6",
                "train_regimes": ["standard_early", "standard_mid", "standard_late"],
                "unique_pool_positions": 1_000_000,
                "consumed_positions": 1_000_000,
                "seeds": [271828, 314159],
            },
            {
                "phase": "A2",
                "purpose": "separate parameter hunger from data hunger",
                "architecture_selector": "baseline plus top three A1 architectures",
                "parameter_anchors": ["b6", "b8"],
                "train_regimes": ["standard_late"],
                "unique_pool_positions": [1_000_000, 3_000_000],
                "consumed_positions": [1_000_000, 3_000_000],
                "design": "full 2x2 parameter-anchor by data-budget factorial",
                "seeds": [271828, 314159],
            },
            {
                "phase": "A3",
                "purpose": "paired plateau-crossing test of the engineering mixed pool",
                "architecture_selector": "architectures passing A2",
                "initial_checkpoint": "matching A2 standard_late checkpoint",
                "branches": [
                    "restart_from_parent_weights_on_standard_late",
                    "switch_to_mixed_late_with_fresh_optimizer_and_cursor",
                ],
                "paired_control": (
                    "both branches inherit the identical model weights only and consume equal "
                    "additional positions"
                ),
                "additional_consumed_positions": 1_000_000,
                "seeds": [271828, 314159],
            },
            {
                "phase": "A4",
                "purpose": "large-capacity confirmation only when B6-to-B8 scaling remains positive",
                "architecture_selector": "baseline plus top two A2/A3 candidates",
                "parameter_anchor": "b10",
                "train_regimes": ["standard_late", "mixed_late_promotion"],
                "unique_pool_positions": 3_000_000,
                "consumed_positions": 3_000_000,
                "seeds": [271828, 314159],
            },
        ],
        "report_metrics": {
            "identity": [
                "architecture",
                "model_config",
                "data_recipe_id",
                "train_regime",
                "seed",
                "dataset/checkpoint/config hashes",
            ],
            "resources": [
                "parameters",
                "search MACs/FLOPs",
                "unique/consumed positions",
                "training positions/s",
                "CPU/GPU latency and throughput",
                "peak memory",
            ],
            "offline": [
                "policy CE/JSD/top1 agreement",
                "WDL CE/Brier/ECE/accuracy",
                "weighted total validation loss",
                "auxiliary losses/accuracies",
                "learning-curve slope and threshold positions",
                "cross-regime and mixed-late gaps",
                "seed mean/std",
            ],
            "strength": [
                "anchored Elo with 95% CI and pair count",
                "direct paired W/D/L",
                "first/second-player score",
                "search profile and saturation status",
            ],
            "closed_loop": [
                "Elo per million self-play positions and per GPU-hour",
                "accepted cadence and gate outcomes",
                "game length/short-game rate/policy entropy",
                "first-player advantage/collapse pauses/seed variance",
            ],
        },
        "claim_rule": (
            "Report matched-parameter efficiency, data scaling, parameter scaling, and best "
            "attainable strength separately; an enlarged model cannot establish an architecture win "
            "against smaller controls."
        ),
    }
