from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent


@dataclass(frozen=True)
class HumanEvalConfig:
    model_path: Path
    model_name: str = "candidate_model"
    agent_type: str = "auto"  # auto | mcts | tiny
    human_name: str = "Human"
    human_plays_first: bool = True

    num_mcts_sims: int = 160
    virtual_loss: float = 1.0
    cpuct: float = 1.0
    temperature: float = 0.0
    num_mcts_threads: int = 8
    inference_batch_size: int = 32
    inference_timeout_s: float = 0.003

    # Use None to auto-select: cuda if available else cpu.
    device: str | None = None

    # Keep aligned with the training rules by default: 5x5x6, connect 4.
    board_size: int = 5
    board_layers: int = 6
    connect_n: int = 4

    # Optional explicit model metadata. Keep None for automatic inference.
    model_config: dict[str, Any] | None = None

    history_dir: Path = CURRENT_DIR / "history"
    seed: int = 42


EVAL_CONFIG = HumanEvalConfig(
    model_path=(
        WORKSPACE_ROOT
        / "training"
        / "runs"
        / "local_archive_validation"
        / "materialized"
        / "accepted"
        / "candidate-g000057-s00007713-d00415640.pt"
    ),
    model_name="v3_b8c192_g57",
    agent_type="auto",
    human_name="Developer",
    human_plays_first=True,
    num_mcts_sims=256,
    virtual_loss=0.8,
    temperature=0.1,
)
# V3 B8 (192ch x 8 blocks) on CPU is slow at 256 sims; lower num_mcts_sims
# (e.g. 64) for a responsive UI without a GPU, or point model_path at a
# smaller accepted artifact (B6/B4) which loads through the same adapter.
