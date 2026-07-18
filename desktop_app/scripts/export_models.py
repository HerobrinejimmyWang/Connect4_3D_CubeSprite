"""Export checked-in-reference checkpoints to deterministic ONNX resources.

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
    "cubesprite_v3": {
        "source": Path("iter_0192") / "best_state.pth.tar",
        "source_sha256": "5faff6811e9d87e0e81a74098bac1e4704cc5c9694147471a9d6e57d02f009e0",
        "output": "cubesprite_v3.onnx",
        "architecture": "gravity_resnet_v1",
        "layers": 6,
        "channels": 2,
        "actions": 150,
    },
    "cubesprite_v3_mini": {
        "source": Path("iter_0208") / "best_recent.pth.tar",
        "source_sha256": "494250df1a71a7a49d5a2d8e163d0a5d783193d3cf71a3678e0e324b2b053142",
        "output": "cubesprite_v3_mini.onnx",
        "architecture": "gravity_resnet_v1",
        "layers": 6,
        "channels": 2,
        "actions": 150,
    },
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
    if expected.get("source_sha256"):
        verify_source_sha256(checkpoint, expected["source_sha256"])
    output = output.resolve()
    model, config, _metadata = load_compatible_model(str(checkpoint), device="cpu")
    actual = {
        "architecture": config["architecture"],
        "layers": int(config["board_layers"]),
        "channels": int(config["input_channels"]),
        "actions": int(
            config.get("action_dim", int(config["board_layers"]) * int(config["board_size"]) ** 2)
        ),
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
        _set_static_output_dimension(graph, "policy", axis=1, value=expected["actions"])
        onnx.checker.check_model(graph)
        onnx.save(graph, str(temporary))
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        policy_metadata = next(item for item in session.get_outputs() if item.name == "policy")
        if policy_metadata.shape[-1] != expected["actions"]:
            raise ValueError(f"Exported policy metadata shape {policy_metadata.shape} is incorrect.")
        validation_batch = _validation_batch(expected)
        with torch.inference_mode():
            reference_policy, reference_value = model(validation_batch)
        actual_policy, actual_value = session.run(["policy", "value"], {"board": validation_batch.numpy()})
        np.testing.assert_allclose(actual_policy, reference_policy.numpy(), rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(actual_value, reference_value.numpy(), rtol=2e-4, atol=2e-5)
        if actual_policy.shape != (validation_batch.shape[0], expected["actions"]):
            raise ValueError(f"Exported policy shape {actual_policy.shape} is incorrect.")
        if actual_value.shape != (validation_batch.shape[0], 1):
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


def _validation_batch(expected: dict) -> torch.Tensor:
    """Cover empty, mixed-height, and full-column positions during export."""

    channels = int(expected["channels"])
    layers = int(expected["layers"])
    batch = torch.zeros((3, channels, layers, 5, 5), dtype=torch.float32)
    placements = (
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (0, 0, 2, 3),
        (1, 1, 2, 3),
        (0, 2, 2, 3),
        (1, 0, 4, 4),
    )
    for player_channel, layer, row, column in placements:
        if channels == 1:
            batch[1, 0, layer, row, column] = 1.0 if player_channel == 0 else -1.0
        else:
            batch[1, player_channel, layer, row, column] = 1.0
    for layer in range(min(layers, 6)):
        if channels == 1:
            batch[2, 0, layer, 2, 2] = 1.0 if layer % 2 == 0 else -1.0
        else:
            batch[2, layer % 2, layer, 2, 2] = 1.0
    return batch


def _set_static_output_dimension(graph, output_name: str, axis: int, value: int) -> None:
    output = next((item for item in graph.graph.output if item.name == output_name), None)
    if output is None:
        raise ValueError(f"Exported graph does not contain the {output_name!r} output.")
    dimensions = output.type.tensor_type.shape.dim
    if not 0 <= axis < len(dimensions):
        raise ValueError(f"Output {output_name!r} does not contain axis {axis}.")
    dimension = dimensions[axis]
    dimension.ClearField("dim_param")
    dimension.dim_value = int(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_sha256(path: Path, expected_sha256: str) -> str:
    expected = str(expected_sha256).strip().lower()
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Source checkpoint hash mismatch for {path.name}: expected {expected}, got {actual}."
        )
    return actual


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export CubeSprite PyTorch checkpoints to validated ONNX files.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "tmp_built_app",
        help="Read-only directory containing the reference checkpoints and iteration bundles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "desktop_app" / "src-tauri" / "resources" / "models",
    )
    parser.add_argument("--models", choices=("all", *EXPORTS), default="all")
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
