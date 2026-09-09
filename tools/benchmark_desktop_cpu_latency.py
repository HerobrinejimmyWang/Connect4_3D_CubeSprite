"""Reproduce the desktop fixed-state CPU latency measurement for bundled models.

The historical v0.1.0 replay is no longer locally available.  This runner uses
a deterministic, non-terminal 25-state corpus and retains the original method:
22 selected states, isolated MCTS per state, temperature 0.4, forced tactical
shortcuts, and a 10-second excluded idle period after each non-final response.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "desktop_app" / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from connect4_core import GameRules  # noqa: E402
from cubesprite_backend.model_runtime import ModelRegistry  # noqa: E402
from cubesprite_backend.search import NumpyMCTS, find_forced_tactical_action  # noqa: E402


EXCLUDED_STATE_INDICES = (9, 21, 23)
CORPUS_SEED = 20260908


def build_states() -> list[tuple[np.ndarray, int]]:
    """Build 25 deterministic non-terminal positions without applying MCTS moves."""

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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    meta, summary = payload["metadata"], payload["summary"]
    lines = [
        "# Targeted desktop fixed-state latency",
        "",
        f"- Model: `{meta['model_id']}`; MCTS: `{meta['mcts_sims']}`; temperature: `{meta['temperature']}`.",
        f"- 22 deterministic non-terminal states; `{meta['idle_s']:.0f}s` idle interval after each non-final response (excluded from latency).",
        f"- Mean including shortcuts: **{summary['mean_including_shortcuts_s']:.3f}s**.",
        f"- Shortcut states: **{summary['shortcut_count']}**; mean excluding shortcuts: **{summary['mean_excluding_shortcuts_s']:.3f}s** across {summary['searched_state_count']} states.",
        "",
        "| State | Side to move | Shortcut | Latency | Returned action |",
        "|---:|---:|:---:|---:|---:|",
    ]
    for row in payload["records"]:
        lines.append(
            f"| {row['state_index']} | {row['player']} | {'yes' if row['shortcut_triggered'] else 'no'} | "
            f"{row['latency_s']:.3f}s | {row['returned_action']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(model_id: str, simulations: int, output_dir: Path, idle_s: float) -> dict:
    resources = ROOT / "desktop_app" / "src-tauri" / "resources"
    registry, game = ModelRegistry(resources), GameRules()
    predictor = registry.predictor(model_id)
    # Match desktop use: warm the ONNX session before timing the first state.
    predictor.predict(game.get_init_board())
    records = []
    for index, (board, player) in enumerate(build_states()):
        if index in EXCLUDED_STATE_INDICES:
            continue
        shortcut = find_forced_tactical_action(game, board, player)
        started = time.perf_counter()
        result = NumpyMCTS(
            game, predictor, simulations=simulations, temperature=0.4, forced_tactics=True, seed=CORPUS_SEED + index
        ).run(board, player)
        elapsed = time.perf_counter() - started
        records.append({
            "state_index": index, "player": int(player), "shortcut_triggered": shortcut is not None,
            "shortcut_action": None if shortcut is None else int(shortcut[0]),
            "latency_s": elapsed, "returned_action": int(result.action),
        })
        if elapsed > 60.0:
            break
        if idle_s > 0:
            time.sleep(idle_s)
    all_latencies = [row["latency_s"] for row in records]
    searched = [row["latency_s"] for row in records if not row["shortcut_triggered"]]
    payload = {
        "schema_version": 1,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "desktop-app-onnx-numpymcts", "model_id": model_id,
            "mcts_sims": simulations, "temperature": 0.4, "idle_s": idle_s,
            "excluded_state_indices": list(EXCLUDED_STATE_INDICES), "corpus": "deterministic-nonterminal-v1",
            "corpus_seed": CORPUS_SEED,
            "state_rule": "States are reconstructed independently; no returned action is applied.",
            "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        },
        "records": records,
        "summary": {
            "state_count": len(records), "shortcut_count": len(records) - len(searched),
            "mean_including_shortcuts_s": fmean(all_latencies),
            "mean_excluding_shortcuts_s": fmean(searched) if searched else None,
            "searched_state_count": len(searched),
        },
        "completed": len(records) == 22,
        "stop_reason": None if len(records) == 22 else "response_exceeded_60_seconds",
    }
    stem = f"{model_id}_{simulations}"
    write_json(output_dir / f"{stem}.json", payload)
    write_markdown(output_dir / f"{stem}.md", payload)
    return payload


def write_summary_markdown(path: Path, results: dict[str, dict]) -> None:
    models = ["v3_b6c128", "v3_b8c192", "v3_b10c256"]
    simulations = [32, 128, 256, 512, 1024]
    lines = [
        "# Desktop CPU response-latency matrix",
        "",
        "- Method: 22 deterministic non-terminal fixed states, independently reconstructed; returned actions are not applied.",
        "- Temperature: 0.4; forced tactical shortcuts enabled; 10-second idle after each non-final response and 60 seconds between groups. Idle time is excluded.",
        "- The v0.1.0 source replay is unavailable locally, so this corpus is deterministic but not directly position-for-position comparable to v0.1.0.",
        "- A blank cell means a response exceeded 60 seconds and that group was stopped.",
        "",
        "| Model \\ MCTS simulations | 32 | 128 | 256 | 512 | 1024 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        cells = []
        for sims in simulations:
            result = results.get(f"{model}@{sims}")
            cells.append(
                f"{result['summary']['mean_excluding_shortcuts_s']:.3f}s"
                if result and result["completed"] else ""
            )
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["v3_b6c128", "v3_b8c192", "v3_b10c256"])
    parser.add_argument("--sims", nargs="+", type=int, default=[32, 128, 256, 512, 1024])
    parser.add_argument("--idle-s", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release" / "v0.1.1" / "test_results" / "cpu_response_latency")
    args = parser.parse_args()
    if args.idle_s < 0 or any(value not in {32, 64, 128, 256, 512, 1024} for value in args.sims):
        raise ValueError("invalid idle interval or MCTS setting")
    summary_path = args.output_dir / "summary.json"
    results = {}
    if summary_path.exists():
        results = json.loads(summary_path.read_text(encoding="utf-8")).get("results", {})
    groups = [(model, sims) for model in args.models for sims in args.sims]
    for group_index, (model, sims) in enumerate(groups):
            results[f"{model}@{sims}"] = run_one(model, sims, args.output_dir, args.idle_s)
            if group_index + 1 < len(groups):
                time.sleep(60.0)
    write_json(args.output_dir / "summary.json", {"schema_version": 1, "results": results})
    write_summary_markdown(args.output_dir / "summary.md", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
