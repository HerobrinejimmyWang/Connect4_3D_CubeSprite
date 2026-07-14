"""Export the two checked-in-reference checkpoints to deterministic ONNX resources.

The source files under ``tmp_built_app`` are read-only inputs. Outputs are first
written to a temporary file, validated with ONNX Runtime, and atomically moved
into ``src-tauri/resources/models`` so the registry never observes a partial
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from model_compat import load_compatible_model  # noqa: E402


EXPORTS = {
    "v22": {
        "source": "v2.2_balance.pth",
        "output": "v2.2_balance.onnx",
        "architecture": "modern",
        "layers": 6,
        "channels": 2,
        "actions": 150,
    },
    "v21": {
        "source": "v2.1_high.pth.tar",
        "output": "v2.1_high.onnx",
        "architecture": "legacy-v21",
        "layers": 8,
        "channels": 1,
        "actions": 200,
    },
}


def export_one(checkpoint: Path, output: Path, expected: dict, opset: int = 17) -> dict:
    checkpoint = checkpoint.resolve(strict=True)
    output = output.resolve()
    model, config, _metadata = load_compatible_model(str(checkpoint), device="cpu")
    actual = {
        "architecture": config["architecture"],
        "layers": int(config["board_layers"]),
        "channels": int(config["input_channels"]),
        "actions": int(config["board_layers"]) * int(config["board_size"]) ** 2,
    }
    wanted = {key: expected[key] for key in actual}
    if actual != wanted or int(config["board_size"]) != 5:
        raise ValueError(f"Checkpoint {checkpoint.name} has {actual}, expected {wanted} on a 5x5 board.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    dummy = torch.zeros((1, expected["channels"], expected["layers"], 5, 5), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        reference_policy, reference_value = model(dummy)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(temporary),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["board"],
            output_names=["policy", "value"],
            dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
            dynamo=False,
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        actual_policy, actual_value = session.run(["policy", "value"], {"board": dummy.numpy()})
        np.testing.assert_allclose(actual_policy, reference_policy.numpy(), rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(actual_value, reference_value.numpy(), rtol=2e-4, atol=2e-5)
        if actual_policy.shape != (1, expected["actions"]):
            raise ValueError(f"Exported policy shape {actual_policy.shape} is incorrect.")
        if actual_value.shape != (1, 1):
            raise ValueError(f"Exported value shape {actual_value.shape} is incorrect.")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        **actual,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export CubeSprite PyTorch checkpoints to validated ONNX files.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "tmp_built_app",
        help="Read-only directory containing v2.2_balance.pth and v2.1_high.pth.tar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "desktop_app" / "src-tauri" / "resources" / "models",
    )
    parser.add_argument("--models", choices=("all", "v22", "v21"), default="all")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    selected = EXPORTS if args.models == "all" else {args.models: EXPORTS[args.models]}
    for model_id, expected in selected.items():
        result = export_one(args.source_dir / expected["source"], args.output_dir / expected["output"], expected, args.opset)
        print(
            f"{model_id}: {result['output']} ({result['bytes']} bytes, sha256={result['sha256']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
