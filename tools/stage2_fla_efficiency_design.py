"""Freeze parameter-matched FLA efficiency candidates using existing V3 models.

The output is a design manifest and offline config templates, not trained
weights or a Flash qualification. Source BAL-2 configurations remain intact.
"""

from __future__ import annotations

import argparse
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
from training.v3.stage2.calibration import (  # noqa: E402
    component_parameter_breakdown,
    estimate_search_macs,
    parameter_count,
)


SOURCE = ROOT / "training/runs/stage2/archive/experiments/stage2r3/round3_balance/bal2/configs"
OUTPUT = ROOT / "training/runs/stage2/fla/efficiency_6m"
REFERENCE = SOURCE / "balance_gravity_control__standard_late__seed271828.json"
SPECS = {
    "column_2d_b8c192": ModelConfig(architecture="column_resnet", blocks=8, channels=192),
    "raw3d_to2d_thin_b8c192": ModelConfig(
        architecture="raw3d_to2d_resnet", blocks=8, channels=192,
        branch_channels=48, volume_blocks=1, collapse_mode="learned",
    ),
    "column3d_v2_thin_b8c192": ModelConfig(
        architecture="column3d_fusion_v2", blocks=8, channels=192,
        branch_channels=48, volume_blocks=1, collapse_mode="learned", fusion_mode="concat",
    ),
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_design(output_dir: Path) -> dict[str, object]:
    source_bytes = REFERENCE.read_bytes()
    base = json.loads(source_bytes)
    anchor = parameter_count(build_model(ModelConfig(**base["model"])))
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant, model_config in SPECS.items():
        model = build_model(model_config).eval()
        parameters = parameter_count(model)
        if abs(parameters / anchor - 1) > 0.03:
            raise ValueError(f"{variant} is more than 3% from the BAL-2 parameter anchor")
        board = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
        role = torch.tensor([[1.0, 0.0]])
        with torch.inference_mode():
            prediction = model.forward_search(
                board, role_to_play=role, rule_features=classic_rule_features(1)
            )
        if prediction.policy_logits.shape != (1, 25) or prediction.wdl_logits.shape != (1, 3):
            raise ValueError(f"{variant} has an invalid search interface")
        if not torch.isfinite(prediction.policy_logits).all() or not torch.isfinite(prediction.wdl_logits).all():
            raise ValueError(f"{variant} has non-finite initial predictions")
        config = dict(base)
        config.update({
            "model": model_config_dict(model_config),
            "device": "cuda:0" if variant != "column3d_v2_thin_b8c192" else "cuda:1",
            "output_dir": f"training/runs/stage2/fla/efficiency_6m/runs/{variant}/standard_late/seed271828",
            "resume": False,
            "target_positions": 1_000_000,
        })
        config_path = output_dir / "configs" / f"{variant}__standard_late__seed271828.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_bytes = _json_bytes(config)
        config_path.write_bytes(config_bytes)
        rows.append({
            "variant": variant,
            "architecture": model_config.architecture,
            "parameters": parameters,
            "parameter_ratio": parameters / anchor,
            "component_parameters": component_parameter_breakdown(model),
            "search_macs_estimate": estimate_search_macs(model),
            "config": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "training_status": "not_started",
            "latency_status": "not_measured",
        })
    manifest = {
        "schema": "connect4-stage2-fla-efficiency-design-v1",
        "status": "design_only",
        "source_config": REFERENCE.relative_to(ROOT).as_posix(),
        "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "anchor_parameters": anchor,
        "match_tolerance": 0.03,
        "rule": "classic",
        "train_regime": "standard_late",
        "seed": 271828,
        "target_positions": 1_000_000,
        "candidates": rows,
    }
    (output_dir / "design.json").write_bytes(_json_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = build_design(args.output_dir.resolve())
    for row in manifest["candidates"]:
        print(row["variant"], row["parameters"], row["search_macs_estimate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
