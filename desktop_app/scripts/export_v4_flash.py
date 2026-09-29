"""Export the V3-native terminal snapshot for the desktop role/rule contract.

The checkpoint is a read-only V3 evaluation artifact. This exporter never loads
historical training weights into a V3 model. The ONNX resource has explicit
canonical board, absolute role, and rule feature inputs.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from connect4_core.rules import BAL5_R2_RULE_REGISTRY  # noqa: E402
from training.v3.config import ModelConfig  # noqa: E402
from training.v3.model import build_model  # noqa: E402


SOURCE = REPO_ROOT / "terminal_snapshot.pt"
DESTINATION = REPO_ROOT / "desktop_app/src-tauri/resources/models/cubesprite_v4_flash_preview1.onnx"
SOURCE_SHA256 = "2524860a99c85ae2e2aa2eb018368e906c7bbcfb1191051532a2d75f42cf6f2f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DesktopRoleRuleAdapter(torch.nn.Module):
    """Map V3 25-column policy and WDL to the desktop 150-action contract."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, board: torch.Tensor, role_to_play: torch.Tensor, rule_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        column_logits, wdl_logits = self.model.forward_search(
            board, role_to_play=role_to_play, rule_features=rule_features
        )
        policy = column_logits.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 150)
        wdl = torch.softmax(wdl_logits, dim=1)
        value = (wdl[:, 0] - wdl[:, 2]).unsqueeze(1)
        return policy, value


def export(source: Path = SOURCE, destination: Path = DESTINATION) -> dict[str, str | int]:
    source = source.resolve(strict=True)
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Terminal snapshot SHA-256 differs from the reviewed source.")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format") != "connect4-v3-model":
        raise ValueError("Terminal snapshot is not a V3 model artifact.")
    config = ModelConfig(**dict(payload["model_config"]))
    if config.global_input_schema != "role_rule_v1" or config.rule_feature_dim != 32:
        raise ValueError("Terminal snapshot lacks the required role/rule input protocol.")
    model = build_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    adapter = DesktopRoleRuleAdapter(model).eval()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    board = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    role = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    rules = torch.tensor([BAL5_R2_RULE_REGISTRY.features("classic")], dtype=torch.float32)
    try:
        torch.onnx.export(
            adapter,
            (board, role, rules),
            str(temporary),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["board", "role_to_play", "rule_features"],
            output_names=["policy", "value"],
            dynamo=False,
            dynamic_axes={
                "board": {0: "batch"},
                "role_to_play": {0: "batch"},
                "rule_features": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            },
        )
        onnx.checker.check_model(onnx.load(str(temporary)))
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        boards = np.zeros((10, 6, 5, 5), dtype=np.float32)
        # Two nontrivial, gravity-valid positions, each evaluated under five rules.
        for index in range(5):
            boards[index, 0, 0, 0] = 1
            boards[index, 0, 1, 0] = -1
            boards[index, 1, 0, 0] = 1
            boards[index + 5, 0, 4, 4] = -1
            boards[index + 5, 0, 2, 2] = 1
            boards[index + 5, 1, 2, 2] = -1
        roles = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5, dtype=np.float32)
        all_rules = np.array(
            [BAL5_R2_RULE_REGISTRY.features(spec) for spec in BAL5_R2_RULE_REGISTRY.specs] * 2,
            dtype=np.float32,
        )
        with torch.inference_mode():
            expected = adapter(torch.from_numpy(boards), torch.from_numpy(roles), torch.from_numpy(all_rules))
        actual = session.run(None, {"board": boards, "role_to_play": roles, "rule_features": all_rules})
        for observed, reference in zip(actual, expected):
            np.testing.assert_allclose(observed, reference.numpy(), rtol=2e-4, atol=2e-5)
        if actual[0].shape != (10, 150) or actual[1].shape != (10, 1):
            raise ValueError("Exported ONNX output shapes are wrong.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source_sha256": sha256(source), "artifact_sha256": sha256(destination), "bytes": destination.stat().st_size}


if __name__ == "__main__":
    print(export())
