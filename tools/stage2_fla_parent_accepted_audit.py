"""Inspect parent accepted model kinds before a post-1M semantic fork."""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "training/runs/stage2/fla/selfplay/r3_bal5r1_fork/fork_manifest.json").read_text())
for row in manifest["rows"]:
    run_dir = Path(row["parent_run_dir"])
    pointer = json.loads((run_dir / "manifests/latest_generation.json").read_text())
    commit = json.loads((run_dir / pointer["commit"]).read_text())
    path = run_dir / commit["accepted_model_path"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    print(json.dumps({"name": row["name"], "accepted": str(path),
                      "metadata": payload.get("metadata", {})}, ensure_ascii=False))
