"""Convert formal Stage 1 event metrics into the Stage 2 trajectory schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


STAGE2_ELO_REGISTRY_HASH = "806753498c10ce585a9b7586276eaa9037637be2b050072d0b684b9461773b79"


def prepare_trajectory_metrics(
    event_metrics_path: str | Path,
    strength_points_path: str | Path,
    output_path: str | Path,
    *,
    cadence_window: int = 16,
) -> dict[str, Any]:
    if cadence_window < 1:
        raise ValueError("cadence_window must be positive")
    selfplay: dict[int, dict[str, Any]] = {}
    commits: dict[int, dict[str, Any]] = {}
    for line in Path(event_metrics_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        generation = row.get("generation")
        if generation is None:
            continue
        generation = int(generation)
        if row.get("stage") == "selfplay" and isinstance(row.get("health"), dict):
            selfplay[generation] = row
        elif row.get("stage") == "generation_commit":
            commits[generation] = row
    generations = sorted(set(selfplay).intersection(commits))
    if len(generations) < 9 or generations != list(range(generations[0], generations[-1] + 1)):
        raise ValueError("Stage 1 event metrics need at least nine contiguous complete generations")

    strength_document = json.loads(Path(strength_points_path).read_text(encoding="utf-8"))
    if strength_document.get("schema") != "connect4-v3-stage2-strength-points-v1":
        raise ValueError("unsupported Stage 2 strength-points schema")
    if (
        strength_document.get("registry_hash") != STAGE2_ELO_REGISTRY_HASH
        or strength_document.get("profile_id") != "primary_256"
    ):
        raise ValueError("strength points must use the frozen Stage 2 primary_256 v3 ruler")
    points = sorted(
        (
            int(row["generation"]),
            float(row["anchored_strength"]),
            str(row["report_sha256"]),
        )
        for row in strength_document.get("points", ())
    )
    if len(points) < 3 or len({row[0] for row in points}) != len(points):
        raise ValueError("strength points need at least three unique generations")
    hexadecimal = set("0123456789abcdef")
    if any(len(row[2]) != 64 or not set(row[2]).issubset(hexadecimal) for row in points):
        raise ValueError("strength point report_sha256 values must be lowercase SHA-256")
    if points[0][0] > generations[0] or points[-1][0] < generations[-1]:
        raise ValueError("strength points must bracket the complete trajectory")
    if not all(np.isfinite(row[1]) for row in points):
        raise ValueError("strength points contain a non-finite rating")

    point_generations = np.asarray([row[0] for row in points], dtype=np.float64)
    point_ratings = np.asarray([row[1] for row in points], dtype=np.float64)
    accepted_changes = []
    previous_model: str | None = None
    rows = []
    for generation in generations:
        commit = commits[generation]
        accepted = commit.get("accepted_model_id")
        changed = int(accepted is not None and accepted != previous_model)
        accepted_changes.append(changed)
        if accepted is not None:
            previous_model = str(accepted)
        health = selfplay[generation]["health"]
        game_length = health.get("game_length", {})
        entropy = health.get("mean_policy_entropy", {})
        start = max(0, len(accepted_changes) - cadence_window)
        rows.append(
            {
                "generation": generation,
                "anchored_strength": float(
                    np.interp(generation, point_generations, point_ratings)
                ),
                "mean_game_length": float(game_length["mean"]),
                "short_game_rate": float(game_length["short_le_12_rate"]),
                "policy_entropy": float(entropy["full"]),
                "accepted_cadence": float(np.mean(accepted_changes[start:])),
            }
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "schema": "connect4-v3-stage2-trajectory-metrics-preparation-v1",
        "output": str(target.resolve()),
        "generation_start": generations[0],
        "generation_end": generations[-1],
        "generation_count": len(generations),
        "strength_point_count": len(points),
        "strength_registry_hash": strength_document.get("registry_hash"),
        "strength_profile_id": strength_document.get("profile_id"),
        "strength_report_sha256s": [row[2] for row in points],
        "strength_interpolation": "piecewise_linear_bracketed_v1",
        "accepted_cadence": f"trailing_{cadence_window}_generation_acceptance_fraction",
    }


__all__ = ["STAGE2_ELO_REGISTRY_HASH", "prepare_trajectory_metrics"]
