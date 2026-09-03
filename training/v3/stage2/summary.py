"""Round-1 ranking and deterministic Stage 2A advancement."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


TRAIN_REGIMES = {"standard_early", "standard_mid", "standard_late"}
EVAL_REGIMES = {*TRAIN_REGIMES, "mixed_late"}
METRICS = (
    "total_loss",
    "policy_loss",
    "wdl_loss",
    "policy_jsd",
    "policy_top1_agreement",
    "brier_score",
    "calibration_error",
    "wdl_accuracy",
    "opponent_reply_loss",
    "future_occupancy_loss",
    "moves_left_loss",
    "opponent_reply_accuracy",
    "future_occupancy_accuracy_unweighted",
    "moves_left_accuracy",
)


def summarize_reports(paths: Iterable[str | Path]) -> dict[str, Any]:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not reports:
        raise ValueError("at least one Stage 2 report is required")
    cells: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    secondary: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    efficiencies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        architecture = str(report["model"]["architecture"])
        train_regime = str(report["train_regime"])
        efficiencies[architecture].append(dict(report.get("efficiency", {})))
        for eval_regime, metrics in report["cross_regime"].items():
            if metrics.get("status") != "complete":
                raise ValueError(f"incomplete report for {architecture}/{train_regime}/{eval_regime}")
            cells[architecture][train_regime][eval_regime].append(float(metrics["total_loss"]))
            for metric in METRICS[1:]:
                secondary[architecture][metric].append(float(metrics[metric]))
    architectures = sorted(cells)
    expected = TRAIN_REGIMES
    for architecture in architectures:
        if set(cells[architecture]) != expected:
            raise ValueError(f"architecture {architecture} lacks one or more training regimes")
        if any(set(cells[architecture][train]) != EVAL_REGIMES for train in expected):
            raise ValueError(f"architecture {architecture} lacks cross-regime evaluation cells")

    rows = []
    for architecture in architectures:
        losses = [
            mean(cells[architecture][train][evaluation])
            for train in sorted(expected)
            for evaluation in sorted(EVAL_REGIMES)
        ]
        mixed_gap = mean(
            mean(cells[architecture][train]["mixed_late"])
            - mean(cells[architecture][train]["standard_late"])
            for train in sorted(expected)
        )
        efficiency = efficiencies[architecture][0]
        rows.append(
            {
                "architecture": architecture,
                "macro_total_loss": mean(losses),
                "cross_regime_std": pstdev(losses),
                "policy_loss": mean(secondary[architecture]["policy_loss"]),
                "wdl_loss": mean(secondary[architecture]["wdl_loss"]),
                "policy_jsd": mean(secondary[architecture]["policy_jsd"]),
                "policy_top1_agreement": mean(
                    secondary[architecture]["policy_top1_agreement"]
                ),
                "brier_score": mean(secondary[architecture]["brier_score"]),
                "calibration_error": mean(secondary[architecture]["calibration_error"]),
                "wdl_accuracy": mean(secondary[architecture]["wdl_accuracy"]),
                "opponent_reply_loss": mean(
                    secondary[architecture]["opponent_reply_loss"]
                ),
                "future_occupancy_loss": mean(
                    secondary[architecture]["future_occupancy_loss"]
                ),
                "moves_left_loss": mean(secondary[architecture]["moves_left_loss"]),
                "opponent_reply_accuracy": mean(
                    secondary[architecture]["opponent_reply_accuracy"]
                ),
                "future_occupancy_accuracy_unweighted": mean(
                    secondary[architecture]["future_occupancy_accuracy_unweighted"]
                ),
                "moves_left_accuracy": mean(
                    secondary[architecture]["moves_left_accuracy"]
                ),
                "mixed_late_generalization_gap": mixed_gap,
                "seed_count": len({int(report["seed"]) for report in reports if report["model"]["architecture"] == architecture}),
                "parameters": efficiency.get("parameters"),
                "search_macs_estimate": efficiency.get("search_macs_estimate"),
                "in_regime": {
                    regime: mean(cells[architecture][regime][regime]) for regime in sorted(expected)
                },
            }
        )
    rows.sort(
        key=lambda row: (
            row["macro_total_loss"],
            row["policy_loss"],
            row["wdl_loss"],
            row["cross_regime_std"],
            row["architecture"],
        )
    )
    promoted = {row["architecture"] for row in rows[:4]}
    promoted.add("gravity_resnet")
    for regime in sorted(expected):
        in_regime = sorted(rows, key=lambda row: (row["in_regime"][regime], row["architecture"]))
        promoted.update(row["architecture"] for row in in_regime[:2])
    if len(promoted) > 6:
        order = {row["architecture"]: index for index, row in enumerate(rows)}
        nonbaseline = sorted(promoted - {"gravity_resnet"}, key=order.__getitem__)[:5]
        promoted = {"gravity_resnet", *nonbaseline}
    return {
        "schema": "connect4-v3-stage2-round1-summary-v2",
        "ranking": rows,
        "promoted_architectures": sorted(promoted),
        "rule": "baseline + macro top4 + any in-regime top2; cap six using macro ordering",
    }
