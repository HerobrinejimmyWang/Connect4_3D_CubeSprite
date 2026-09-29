"""Freeze a full-branch raw3d-to2d B6C128 donor beside the thin B6 run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.config import ModelConfig, model_config_dict  # noqa: E402
from training.v3.model import build_model, classic_rule_features  # noqa: E402
from training.v3.stage2.calibration import estimate_search_macs, parameter_count  # noqa: E402

NAME = "raw3d_to2d_full_b6c128"
OUTPUT = ROOT / "training/runs/stage2/fla/b6_raw_full_2m"
SOURCE = ROOT / "training/runs/stage2/fla/efficiency_6m/configs/raw3d_to2d_thin_b8c192__standard_late__seed271828.json"


def main() -> int:
    model_config = ModelConfig(
        architecture="raw3d_to2d_resnet", blocks=6, channels=128,
        branch_channels=64, volume_blocks=2, collapse_mode="learned",
    )
    model = build_model(model_config).eval()
    with torch.inference_mode():
        output = model.forward_search(
            torch.zeros((1, 6, 5, 5)),
            role_to_play=torch.tensor([[1.0, 0.0]]),
            rule_features=classic_rule_features(1),
        )
    if output.policy_logits.shape != (1, 25) or output.wdl_logits.shape != (1, 3):
        raise ValueError("B6 full-branch search output has unexpected dimensions")
    if not torch.isfinite(output.policy_logits).all() or not torch.isfinite(output.wdl_logits).all():
        raise ValueError("B6 full-branch search output is non-finite")

    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    raw.update({
        "model": model_config_dict(model_config),
        "device": "cuda:0",
        "output_dir": f"training/runs/stage2/fla/b6_raw_full_2m/runs/{NAME}/standard_late/seed271828",
        "resume": False,
        "target_positions": 1_000_000,
    })
    config = OUTPUT / "configs" / f"{NAME}__standard_late__seed271828.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    if config.exists():
        raise FileExistsError(f"frozen B6 full-branch config exists: {config}")
    encoded = (json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    config.write_bytes(encoded)
    design = {
        "schema": "connect4-stage2-fla-b6-full-design-v1",
        "variant": NAME,
        "status": "design_only_requires_cpu_and_strength_evidence",
        "architecture": model_config.architecture,
        "parameters": parameter_count(model),
        "search_macs_estimate": estimate_search_macs(model),
        "config": config.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_config_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "target_positions": 1_000_000,
        "seed": 271828,
        "rule": "classic",
        "cpu_screen_upper_mean_s": 3.8,
    }
    (OUTPUT / "design.json").write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(design, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
