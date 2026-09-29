"""Untrained architecture-only CPU probe; never use as formal FLA latency."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "desktop_app/backend"))

from connect4_core import GameRules  # noqa: E402
from cubesprite_backend.search import NumpyMCTS, find_forced_tactical_action  # noqa: E402
from training.v3.config import ModelConfig  # noqa: E402
from training.v3.model import LegacyPolicyValueAdapter, TorchPredictor, build_model  # noqa: E402
from benchmark_stage2_cpu_latency import (  # noqa: E402
    CORPUS_SEED, EXCLUDED_STATE_INDICES, ClassicContextPredictor, build_states,
)


def main() -> int:
    game = GameRules()
    results = []
    for name, blocks, channels in (
        ("b8c192_control", 8, 192),
        ("b6c208_probe", 6, 208),
        ("b6c192_probe", 6, 192),
    ):
        torch.manual_seed(271828)
        config = ModelConfig(
            architecture="raw3d_to2d_resnet", blocks=blocks, channels=channels,
            branch_channels=48, volume_blocks=1, collapse_mode="learned",
        )
        model = build_model(config).eval()
        predictor = LegacyPolicyValueAdapter(ClassicContextPredictor(TorchPredictor(model, device="cpu")))
        predictor.predict(game.get_canonical_form(game.get_init_board(), 1))
        durations = []
        for index, (board, player) in enumerate(build_states()):
            if index in EXCLUDED_STATE_INDICES:
                continue
            shortcut = find_forced_tactical_action(game, board, player)
            started = time.perf_counter()
            NumpyMCTS(
                game, predictor, simulations=512, temperature=0.4,
                forced_tactics=True, seed=CORPUS_SEED + index,
            ).run(board, player)
            duration = time.perf_counter() - started
            if shortcut is None:
                durations.append(duration)
        results.append({"name": name, "searched_count": len(durations), "searched_mean_s": mean(durations)})
        print(f"{name}: {mean(durations):.3f} s", flush=True)
    output = ROOT / "training/runs/stage2/fla/b6_raw_5m/cpu_untrained_probe.json"
    output.write_text(json.dumps({
        "status": "untrained_architecture_probe_not_formal_cpu_evidence",
        "simulations": 512, "repeats": 1, "idle_s": 0,
        "torch_threads": torch.get_num_threads(),
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
