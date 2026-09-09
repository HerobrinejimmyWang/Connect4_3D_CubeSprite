"""Measure CPU fixed-state response latency for native Stage 2 models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "desktop_app" / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "desktop_app" / "backend"))

from connect4_core import GameRules  # noqa: E402
from cubesprite_backend.search import NumpyMCTS, find_forced_tactical_action  # noqa: E402
from training.v3.config import ModelConfig  # noqa: E402
from training.v3.model import (  # noqa: E402
    LegacyPolicyValueAdapter,
    TorchPredictor,
    build_model,
    classic_rule_features,
)

EXCLUDED_STATE_INDICES = (9, 21, 23)
CORPUS_SEED = 20260908


def build_states() -> list[tuple[np.ndarray, int]]:
    game = GameRules()
    rng = np.random.default_rng(CORPUS_SEED)
    board = game.get_init_board()
    player = 1
    states = [(board.copy(), player)]
    while len(states) < 25:
        actions = np.flatnonzero(game.get_valid_moves(board)).astype(int).tolist()
        rng.shuffle(actions)
        for action in actions:
            candidate, next_player = game.get_next_state(board, player, action)
            if game.get_game_ended(candidate, next_player) == 0:
                board, player = candidate, int(next_player)
                states.append((board.copy(), player))
                break
        else:
            raise RuntimeError("could not extend deterministic state corpus")
    return states


def load_model(config_path: Path, model_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = ModelConfig(**config["model"])
    model = build_model(model_config)
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    predictor = TorchPredictor(model, device="cpu")
    role = np.asarray((1.0, 0.0), dtype=np.float32)
    rules = classic_rule_features(1).cpu().numpy()[0]

    class FixedContextPredictor:
        def predict(self, canonical_board):
            policy, wdl = predictor.predict(
                canonical_board,
                role_to_play=role,
                rule_features=rules,
            )
            return policy, wdl

    return LegacyPolicyValueAdapter(FixedContextPredictor()), config


def run_one(model_id: str, model_path: Path, config_path: Path, simulations: int) -> dict:
    predictor, config = load_model(config_path, model_path)
    game = GameRules()
    predictor.predict(game.get_canonical_form(game.get_init_board(), 1))
    records = []
    for index, (board, player) in enumerate(build_states()):
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
        records.append(
            {
                "state_index": index,
                "player": int(player),
                "shortcut_triggered": shortcut is not None,
                "shortcut_action": None if shortcut is None else int(shortcut[0]),
                "latency_s": elapsed,
                "returned_action": int(result.action),
            }
        )
    searched = [row["latency_s"] for row in records if not row["shortcut_triggered"]]
    payload = {
        "schema_version": 1,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "native-stage2-pytorch-numpymcts-cpu",
            "model_id": model_id,
            "architecture": config["model"]["architecture"],
            "train_regime": config["train_regime"],
            "mcts_sims": simulations,
            "temperature": 0.4,
            "forced_tactics": True,
            "idle_s": 0.0,
            "excluded_state_indices": list(EXCLUDED_STATE_INDICES),
            "corpus": "deterministic-nonterminal-v1",
            "corpus_seed": CORPUS_SEED,
            "state_rule": "States are reconstructed independently; no returned action is applied.",
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "records": records,
        "summary": {
            "state_count": len(records),
            "shortcut_count": len(records) - len(searched),
            "mean_including_shortcuts_s": fmean(row["latency_s"] for row in records),
            "mean_excluding_shortcuts_s": fmean(searched),
            "searched_state_count": len(searched),
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sims", nargs="+", type=int, default=[16, 64, 256])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(args.model_dir.glob("*.model.pt"))
    results = {}
    for model_path in model_paths:
        model_id = model_path.name.removesuffix(".model.pt")
        config_path = args.model_dir / f"{model_id}.config.json"
        for simulations in args.sims:
            key = f"{model_id}@{simulations}"
            payload = run_one(model_id, model_path, config_path, simulations)
            results[key] = payload
            (args.output_dir / f"{model_id}_{simulations}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"{model_id} sims={simulations} "
                f"mean={payload['summary']['mean_excluding_shortcuts_s']:.4f}s",
                flush=True,
            )
    summary = {"schema_version": 1, "results": results}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage 2A round1 standard_late CPU response latency",
        "",
        "- 22 deterministic fixed states; states are independent and returned actions are not applied.",
        "- Temperature 0.4; forced tactical shortcuts enabled; latency excludes any idle interval.",
        "",
        "| Architecture | 16 sims | 64 sims | 256 sims |",
        "|---|---:|---:|---:|",
    ]
    models = sorted({key.split("@", 1)[0] for key in results})
    for model in models:
        cells = []
        for sims in args.sims:
            cells.append(f"{results[f'{model}@{sims}']['summary']['mean_excluding_shortcuts_s']:.4f}s")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
