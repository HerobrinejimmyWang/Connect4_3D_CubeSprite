"""Freeze the 2D multiview B8C192 FLA donor against the same offline pool."""

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
from training.v3.stage2.calibration import parameter_count  # noqa: E402

NAME = "multiview_resnet_b8c192"
BASE = ROOT / "training/runs/stage2/fla/multiview_b8_1m"
SOURCE = ROOT / "training/runs/stage2/fla/efficiency_6m/configs/column_2d_b8c192__standard_late__seed271828.json"


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_frozen(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"frozen design changed: {path}")
    else:
        path.write_bytes(data)


def main() -> int:
    config = ModelConfig(architecture="multiview_resnet", blocks=8, channels=192)
    model = build_model(config).eval()
    anchor = parameter_count(build_model(ModelConfig(architecture="gravity_resnet", blocks=8, channels=192)))
    params = parameter_count(model)
    if abs(params / anchor - 1) > 0.03:
        raise ValueError("multiview B8 is outside the 3% capacity window")
    with torch.inference_mode():
        output = model.forward_search(
            torch.zeros((1, 6, 5, 5)),
            role_to_play=torch.tensor([[1.0, 0.0]]),
            rule_features=classic_rule_features(1),
        )
    if output.policy_logits.shape != (1, 25) or output.wdl_logits.shape != (1, 3):
        raise ValueError("invalid multiview search shapes")
    if not torch.isfinite(output.policy_logits).all() or not torch.isfinite(output.wdl_logits).all():
        raise ValueError("non-finite multiview search output")

    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    raw.update({
        "model": model_config_dict(config), "device": "cuda:0",
        "output_dir": f"training/runs/stage2/fla/multiview_b8_1m/runs/{NAME}/standard_late/seed271828",
        "resume": False, "target_positions": 1_000_000,
    })
    path = BASE / "configs" / f"{NAME}__standard_late__seed271828.json"
    config_bytes = encode(raw)
    write_frozen(path, config_bytes)
    design = {
        "schema": "connect4-stage2-fla-multiview-b8-design-v1",
        "variant": NAME, "architecture": config.architecture,
        "blocks": 8, "channels": 192, "parameters": params,
        "gravity_b8_parameters": anchor, "parameter_ratio": params / anchor,
        "source_config": SOURCE.relative_to(ROOT).as_posix(),
        "source_config_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "config": path.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "train_regime": "standard_late", "seed": 271828,
        "rule": "classic", "target_positions": 1_000_000,
        "cpu_screen_upper_mean_s": 3.8,
        "status": "design_only_not_flash_qualified",
    }
    write_frozen(BASE / "design.json", encode(design))
    print(json.dumps(design, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
