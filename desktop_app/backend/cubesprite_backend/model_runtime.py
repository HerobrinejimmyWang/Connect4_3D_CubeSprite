from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


PRODUCT_BOARD_LAYERS = 6
PRODUCT_BOARD_SIZE = 5
PRODUCT_ACTION_DIM = PRODUCT_BOARD_LAYERS * PRODUCT_BOARD_SIZE * PRODUCT_BOARD_SIZE
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelRegistryError(ValueError):
    """Raised when the bundled model manifest is invalid."""


class ModelUnavailableError(RuntimeError):
    """Raised when a manifest model cannot be used for inference."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    model_path: str | None
    architecture: str
    board_layers: int
    board_size: int
    input_channels: int
    action_dim: int
    artifact_sha256: str
    source_iteration: int | None
    defaults: dict[str, Any]
    description: dict[str, str]
    placeholder: bool = False

    @property
    def default_mcts_sims(self) -> int:
        return int(self.defaults["mcts_sims"])

    @property
    def default_temperature(self) -> float:
        return float(self.defaults["temperature"])

    @classmethod
    def from_manifest(cls, entry: Any) -> "ModelSpec":
        if not isinstance(entry, dict):
            raise ModelRegistryError("Every model manifest entry must be an object.")
        try:
            spec = cls(**entry)
        except (TypeError, ValueError) as exc:
            raise ModelRegistryError(f"Invalid model manifest entry: {exc}") from exc
        if not spec.id or not spec.display_name:
            raise ModelRegistryError("Model id and display_name must not be empty.")
        if not isinstance(spec.artifact_sha256, str) or _SHA256_PATTERN.fullmatch(spec.artifact_sha256) is None:
            raise ModelRegistryError(
                f"Model {spec.id} artifact_sha256 must be a canonical lowercase SHA-256 digest."
            )
        if spec.source_iteration is not None and (
            isinstance(spec.source_iteration, bool)
            or not isinstance(spec.source_iteration, int)
            or spec.source_iteration <= 0
        ):
            raise ModelRegistryError(f"Model {spec.id} source_iteration must be a positive integer or null.")
        if spec.board_size != PRODUCT_BOARD_SIZE:
            raise ModelRegistryError(f"Model {spec.id} has unsupported board_size={spec.board_size}.")
        if spec.architecture in {"modern-v22", "gravity_resnet_v1"}:
            expected = (6, 2, 150)
        elif spec.architecture == "legacy-v21-adapted-6-layer":
            expected = (8, 1, 200)
        elif spec.placeholder:
            expected = (spec.board_layers, spec.input_channels, spec.action_dim)
        else:
            raise ModelRegistryError(f"Model {spec.id} has unknown architecture {spec.architecture!r}.")
        actual = (spec.board_layers, spec.input_channels, spec.action_dim)
        if not spec.placeholder and actual != expected:
            raise ModelRegistryError(f"Model {spec.id} dimensions {actual} do not match {expected}.")
        if not isinstance(spec.description, dict) or not {"zh", "en"}.issubset(spec.description):
            raise ModelRegistryError(f"Model {spec.id} requires zh/en descriptions.")
        if not isinstance(spec.defaults, dict) or set(spec.defaults) != {"mcts_sims", "temperature"}:
            raise ModelRegistryError(f"Model {spec.id} requires mcts_sims and temperature defaults.")
        if spec.default_mcts_sims not in {32, 64, 128, 256, 512, 1024}:
            raise ModelRegistryError(f"Model {spec.id} has an unsupported default MCTS count.")
        if not 0.0 <= float(spec.default_temperature) <= 5.0:
            raise ModelRegistryError(f"Model {spec.id} has an invalid default temperature.")
        return spec


class ModelRegistry:
    """Manifest-only registry; model directories are deliberately never scanned."""

    def __init__(self, resource_dir: Path):
        self.resource_dir = Path(resource_dir).resolve()
        manifest_path = self.resource_dir / "model_registry.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelRegistryError(f"Cannot read model registry {manifest_path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ModelRegistryError("model_registry.json must contain a models array.")
        specs = [ModelSpec.from_manifest(entry) for entry in payload["models"]]
        if len({spec.id for spec in specs}) != len(specs):
            raise ModelRegistryError("Model ids in model_registry.json must be unique.")
        self._specs = {spec.id: spec for spec in specs}
        self._sessions: dict[str, OnnxPredictor] = {}
        self._load_errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def list_models(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for spec in self._specs.values():
            path = self._resolve_model_path(spec)
            if spec.placeholder:
                reason = "model_placeholder"
            elif path is None or not path.is_file():
                reason = "model_file_missing"
            else:
                reason = self._load_errors.get(spec.id)
            result.append(
                {
                    "id": spec.id,
                    "display_name": spec.display_name,
                    "model_path": spec.model_path,
                    "architecture": spec.architecture,
                    "board_layers": spec.board_layers,
                    "board_size": spec.board_size,
                    "input_channels": spec.input_channels,
                    "action_dim": spec.action_dim,
                    "artifact_sha256": spec.artifact_sha256,
                    "source_iteration": spec.source_iteration,
                    "defaults": dict(spec.defaults),
                    "default_mcts_sims": spec.default_mcts_sims,
                    "default_temperature": spec.default_temperature,
                    "description": dict(spec.description),
                    "placeholder": spec.placeholder,
                    "available": reason is None,
                    "unavailable_reason": reason,
                }
            )
        return result

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[str(model_id)]
        except KeyError as exc:
            raise ModelUnavailableError(f"Unknown model id: {model_id}") from exc

    def predictor(self, model_id: str) -> "OnnxPredictor":
        spec = self.get(model_id)
        path = self._resolve_model_path(spec)
        if spec.placeholder:
            raise ModelUnavailableError(f"Model {spec.display_name} is a future placeholder.")
        if path is None or not path.is_file():
            raise ModelUnavailableError(f"Model file for {spec.display_name} is not installed.")
        with self._lock:
            predictor = self._sessions.get(spec.id)
            if predictor is not None:
                return predictor
            try:
                predictor = OnnxPredictor(spec, path)
            except Exception as exc:
                reason = f"model_load_failed: {exc}"
                self._load_errors[spec.id] = reason
                raise ModelUnavailableError(f"Cannot load {spec.display_name}: {exc}") from exc
            self._sessions[spec.id] = predictor
            self._load_errors.pop(spec.id, None)
            return predictor

    def is_available(self, model_id: str) -> bool:
        spec = self.get(model_id)
        path = self._resolve_model_path(spec)
        return not spec.placeholder and path is not None and path.is_file() and spec.id not in self._load_errors

    def _resolve_model_path(self, spec: ModelSpec) -> Path | None:
        if not spec.model_path:
            return None
        candidate = (self.resource_dir / spec.model_path).resolve()
        try:
            candidate.relative_to(self.resource_dir)
        except ValueError as exc:
            raise ModelRegistryError(f"Model path for {spec.id} escapes the resource directory.") from exc
        return candidate


class OnnxPredictor:
    def __init__(self, spec: ModelSpec, path: Path):
        self.spec = spec
        self._verify_artifact_sha256(path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 2:
            raise ValueError("Expected exactly one ONNX input and two outputs (policy, value).")
        self.input_name = inputs[0].name
        self._validate_static_shape(inputs[0].shape)
        policy_output = next((item for item in outputs if self._last_static_dim(item.shape) == spec.action_dim), None)
        value_output = next((item for item in outputs if self._last_static_dim(item.shape) == 1), None)
        if policy_output is None or value_output is None or policy_output.name == value_output.name:
            raise ValueError(f"ONNX outputs must expose policy[{spec.action_dim}] and value[1].")
        self.policy_output_name = policy_output.name
        self.value_output_name = value_output.name
        self._run_lock = threading.Lock()

    def _verify_artifact_sha256(self, path: Path) -> None:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValueError(f"Cannot read ONNX artifact for {self.spec.id}: {exc}") from exc
        actual = digest.hexdigest()
        if actual != self.spec.artifact_sha256:
            raise ValueError(
                f"ONNX artifact SHA-256 mismatch for {self.spec.id}: "
                f"expected {self.spec.artifact_sha256}, found {actual}."
            )

    def predict(self, canonical_board: np.ndarray) -> tuple[np.ndarray, float]:
        encoded = self._encode(canonical_board)
        with self._run_lock:
            policy_output, value_output = self.session.run(
                [self.policy_output_name, self.value_output_name], {self.input_name: encoded}
            )
        policy_raw = np.asarray(policy_output, dtype=np.float64).reshape(-1)
        if policy_raw.size != self.spec.action_dim:
            raise RuntimeError(f"Model policy has {policy_raw.size} actions; expected {self.spec.action_dim}.")
        if not np.all(np.isfinite(policy_raw)):
            raise RuntimeError("Model policy output contains a non-finite value.")
        # v2.1 is an explicit compatibility adapter: only the first six layer
        # planes are legal in the fixed 6x5x5 product and are then renormalized.
        policy_raw = policy_raw[:PRODUCT_ACTION_DIM]
        shifted = policy_raw - np.max(policy_raw)
        policy = np.exp(np.clip(shifted, -80.0, 0.0))
        total = float(policy.sum())
        if not np.isfinite(total) or total <= 0:
            policy = np.full(PRODUCT_ACTION_DIM, 1.0 / PRODUCT_ACTION_DIM, dtype=np.float64)
        else:
            policy /= total
        value = float(np.asarray(value_output, dtype=np.float32).reshape(-1)[0])
        if not np.isfinite(value):
            raise RuntimeError("Model value output is not finite.")
        return policy, float(np.clip(value, -1.0, 1.0))

    def _encode(self, board: np.ndarray) -> np.ndarray:
        board = np.asarray(board, dtype=np.int8)
        expected = (PRODUCT_BOARD_LAYERS, PRODUCT_BOARD_SIZE, PRODUCT_BOARD_SIZE)
        if tuple(board.shape) != expected:
            raise ValueError(f"Expected board shape {expected}, got {board.shape}.")
        if not np.all(np.isin(board, (-1, 0, 1))):
            raise ValueError("Board cells must contain only -1, 0, or +1.")
        if self.spec.architecture == "legacy-v21-adapted-6-layer":
            padded = np.zeros((self.spec.board_layers, PRODUCT_BOARD_SIZE, PRODUCT_BOARD_SIZE), dtype=np.float32)
            padded[:PRODUCT_BOARD_LAYERS] = board
            return padded[np.newaxis, np.newaxis, ...]
        channels = np.stack((board > 0, board < 0), axis=0).astype(np.float32)
        return channels[np.newaxis, ...]

    def _validate_static_shape(self, shape: list[Any]) -> None:
        expected = [None, self.spec.input_channels, self.spec.board_layers, self.spec.board_size, self.spec.board_size]
        if len(shape) != len(expected):
            raise ValueError(f"ONNX input rank {len(shape)} does not match expected rank 5.")
        for actual, wanted in zip(shape, expected):
            if wanted is not None and isinstance(actual, int) and actual != wanted:
                raise ValueError(f"ONNX input shape {shape} does not match expected {expected}.")

    @staticmethod
    def _last_static_dim(shape: list[Any]) -> int | None:
        return int(shape[-1]) if shape and isinstance(shape[-1], int) else None
