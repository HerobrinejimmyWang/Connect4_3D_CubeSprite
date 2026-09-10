"""Measure fixed-state CPU response latency for native Stage 2 models.

The benchmark shares the deterministic corpus used by the desktop benchmark.
Stage 2 models also receive their mandatory absolute FIRST/SECOND role at every
MCTS leaf, inferred from classic-rule canonical-board occupancy parity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "desktop_app" / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "desktop_app" / "backend"))

from connect4_core import GameRules  # noqa: E402
from cubesprite_backend.search import NumpyMCTS, find_forced_tactical_action  # noqa: E402
from training.v3.config import ModelConfig, model_config_dict  # noqa: E402
from training.v3.model import (  # noqa: E402
    LegacyPolicyValueAdapter,
    TorchPredictor,
    build_model,
    classic_rule_features,
)

EXCLUDED_STATE_INDICES = (9, 21, 23)
CORPUS_SEED = 20260908


def build_states() -> list[tuple[np.ndarray, int]]:
    """Build exactly the same 25 deterministic states as the desktop benchmark."""

    game = GameRules()
    rng = random.Random(CORPUS_SEED)
    board = game.get_init_board()
    player = 1
    states = [(board.copy(), player)]
    while len(states) < 25:
        actions = [int(action) for action in np.flatnonzero(game.get_valid_moves(board))]
        rng.shuffle(actions)
        for action in actions:
            candidate, next_player = game.get_next_state(board, player, action)
            if game.get_game_ended(candidate, next_player) == 0:
                board, player = candidate, int(next_player)
                states.append((board.copy(), player))
                break
        else:
            raise RuntimeError("could not extend the deterministic non-terminal corpus")
    return states


def role_features_for_canonical(canonical_board: np.ndarray) -> np.ndarray:
    """Infer the absolute role of the side to move under classic no-pass rules."""

    board = np.asarray(canonical_board)
    if board.shape != (6, 5, 5):
        raise ValueError(f"canonical board must have shape (6,5,5), got {board.shape}")
    if not np.all(np.isin(board, (-1, 0, 1))):
        raise ValueError("canonical board contains values outside {-1,0,1}")
    plus = int(np.count_nonzero(board == 1))
    minus = int(np.count_nonzero(board == -1))
    if plus == minus:
        return np.asarray((1.0, 0.0), dtype=np.float32)
    if minus == plus + 1:
        return np.asarray((0.0, 1.0), dtype=np.float32)
    raise ValueError(
        "canonical board occupancy cannot identify a legal classic-rule role: "
        f"plus={plus}, minus={minus}"
    )


class ClassicContextPredictor:
    """Supply V3 role and rule context independently at every MCTS leaf."""

    def __init__(self, predictor: TorchPredictor) -> None:
        self.predictor = predictor
        self.rules = classic_rule_features(1).cpu().numpy()[0]

    def predict(self, canonical_board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predictor.predict(
            canonical_board,
            role_to_play=role_features_for_canonical(canonical_board),
            rule_features=self.rules,
        )


def load_model(
    config_path: Path, model_path: Path
) -> tuple[LegacyPolicyValueAdapter, dict[str, Any], dict[str, str]]:
    config_bytes = config_path.read_bytes()
    config_raw = json.loads(config_bytes.decode("utf-8"))
    if not isinstance(config_raw, dict) or not isinstance(config_raw.get("model"), dict):
        raise ValueError(f"Stage 2 config is missing a model mapping: {config_path}")
    model_config = ModelConfig(**config_raw["model"])
    expected_model_config = model_config_dict(model_config)

    artifact_bytes = model_path.read_bytes()
    payload = torch.load(io.BytesIO(artifact_bytes), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stage 2 artifact must contain a mapping: {model_path}")
    if payload.get("format") != "connect4-v3-model" or payload.get("format_version") != 1:
        raise ValueError(f"unsupported V3 model artifact: {model_path}")
    artifact_model_config = payload.get("model_config")
    if artifact_model_config != expected_model_config:
        raise ValueError(
            "artifact model_config does not match benchmark config: "
            f"artifact={artifact_model_config!r}, config={expected_model_config!r}"
        )
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError(f"Stage 2 artifact is missing model_state: {model_path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("lineage") != "v3_stage2_offline":
        raise ValueError("latency benchmark accepts only V3 Stage 2 offline artifacts")
    identity_fields = {
        "train_regime": "train_regime",
        "seed": "seed",
        "target_positions": "train_positions",
    }
    for config_field, artifact_field in identity_fields.items():
        if config_field not in config_raw:
            raise ValueError(f"Stage 2 benchmark config is missing {config_field}")
        if metadata.get(artifact_field) != config_raw[config_field]:
            raise ValueError(
                f"artifact {artifact_field} does not match config {config_field}: "
                f"artifact={metadata.get(artifact_field)!r}, config={config_raw[config_field]!r}"
            )

    model = build_model(model_config)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    context_predictor = ClassicContextPredictor(TorchPredictor(model, device="cpu"))
    evidence = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    return LegacyPolicyValueAdapter(context_predictor), config_raw, evidence


def latency_statistics(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency statistics require at least one measurement")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_s": fmean(values),
        "median_s": median(values),
        "p90_s": float(np.percentile(array, 90)),
        "p95_s": float(np.percentile(array, 95)),
    }


def run_one(
    model_id: str,
    model_path: Path,
    config_path: Path,
    simulations: int,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    predictor, config, evidence = load_model(config_path, model_path)
    game = GameRules()
    predictor.predict(game.get_canonical_form(game.get_init_board(), 1))
    records: list[dict[str, Any]] = []
    states = build_states()
    for repeat_index in range(repeats):
        for index, (board, player) in enumerate(states):
            if index in EXCLUDED_STATE_INDICES:
                continue
            shortcut = find_forced_tactical_action(game, board, player)
            started = time.perf_counter()
            result = NumpyMCTS(
                game,
                predictor,
                simulations=simulations,
                temperature=0.4,
                forced_tactics=True,
                seed=CORPUS_SEED + index,
            ).run(board, player)
            elapsed = time.perf_counter() - started
            canonical = game.get_canonical_form(board, player)
            role = role_features_for_canonical(canonical)
            records.append(
                {
                    "repeat_index": repeat_index,
                    "state_index": index,
                    "player": int(player),
                    "absolute_role": "FIRST" if tuple(role) == (1.0, 0.0) else "SECOND",
                    "shortcut_triggered": shortcut is not None,
                    "shortcut_action": None if shortcut is None else int(shortcut[0]),
                    "latency_s": elapsed,
                    "returned_action": int(result.action),
                }
            )
    searched = [row["latency_s"] for row in records if not row["shortcut_triggered"]]
    all_latencies = [row["latency_s"] for row in records]
    shortcut_states = {
        int(row["state_index"]) for row in records if row["shortcut_triggered"]
    }
    return {
        "schema_version": 2,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "native-stage2-pytorch-numpymcts-cpu",
            "model_id": model_id,
            "architecture": config["model"]["architecture"],
            "train_regime": config.get("train_regime"),
            "config_sha256": evidence["config_sha256"],
            "artifact_sha256": evidence["artifact_sha256"],
            "mcts_sims": simulations,
            "repeats": repeats,
            "temperature": 0.4,
            "forced_tactics": True,
            "idle_s": 0.0,
            "excluded_state_indices": list(EXCLUDED_STATE_INDICES),
            "corpus": "deterministic-nonterminal-v1",
            "corpus_seed": CORPUS_SEED,
            "state_rule": "States match the desktop corpus; returned actions are not applied.",
            "percentile_method": "numpy-linear",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "records": records,
        "summary": {
            "state_count": len(states) - len(EXCLUDED_STATE_INDICES),
            "measurement_count": len(records),
            "shortcut_state_count": len(shortcut_states),
            "searched_measurement_count": len(searched),
            "including_shortcuts": latency_statistics(all_latencies),
            "excluding_shortcuts": latency_statistics(searched),
        },
    }


def build_summary_markdown(
    results: Mapping[str, Mapping[str, Any]], simulations: Sequence[int]
) -> str:
    sims = list(simulations)
    lines = [
        "# Stage 2 CPU response latency",
        "",
        "- Deterministic fixed states shared with the desktop benchmark; returned actions are not applied.",
        "- Temperature 0.4; forced tactical shortcuts enabled; table excludes shortcut measurements.",
        "- Cells report median / p90 / p95 latency across all repeats and searched states.",
        "",
        "| Architecture | " + " | ".join(f"{value} sims" for value in sims) + " |",
        "|---|" + "---:|" * len(sims),
    ]
    models = sorted({key.rsplit("@", 1)[0] for key in results})
    for model in models:
        cells = []
        for simulation_count in sims:
            stats = results[f"{model}@{simulation_count}"]["summary"]["excluding_shortcuts"]
            cells.append(
                f"{stats['median_s']:.4f}s / {stats['p90_s']:.4f}s / {stats['p95_s']:.4f}s"
            )
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sims", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.sims):
        raise ValueError("repeats and MCTS simulation counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(args.model_dir.glob("*.model.pt"))
    if not model_paths:
        raise ValueError(f"no *.model.pt artifacts found in {args.model_dir}")
    results: dict[str, dict[str, Any]] = {}
    for model_path in model_paths:
        model_id = model_path.name.removesuffix(".model.pt")
        config_path = args.model_dir / f"{model_id}.config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing config for {model_path.name}: {config_path}")
        for simulations in args.sims:
            key = f"{model_id}@{simulations}"
            payload = run_one(
                model_id,
                model_path,
                config_path,
                simulations,
                repeats=args.repeats,
            )
            results[key] = payload
            (args.output_dir / f"{model_id}_{simulations}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            stats = payload["summary"]["excluding_shortcuts"]
            print(
                f"{model_id} sims={simulations} median={stats['median_s']:.4f}s "
                f"p95={stats['p95_s']:.4f}s",
                flush=True,
            )
    summary = {"schema_version": 2, "results": results}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        build_summary_markdown(results, args.sims), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
