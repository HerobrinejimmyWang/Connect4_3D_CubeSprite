"""Generate explicit cold- and model-only warm-start configs for Stage 2B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from ..config import V3Config, load_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_stage2b_configs(
    *,
    base_config_path: str | Path,
    architecture_matrix_path: str | Path,
    finalists_path: str | Path,
    output_dir: str | Path,
    warm_starts_path: str | Path | None = None,
    seeds: Iterable[int] = (271828, 314159),
) -> dict[str, Any]:
    base = load_config(base_config_path)
    matrix = json.loads(Path(architecture_matrix_path).read_text(encoding="utf-8"))
    finalist_document = json.loads(Path(finalists_path).read_text(encoding="utf-8"))
    finalists = tuple(str(name) for name in finalist_document.get("finalists", ()))
    if len(finalists) != 3 or len(set(finalists)) != 3 or "gravity_resnet" not in finalists:
        raise ValueError("finalists must contain exactly gravity_resnet and two distinct candidates")
    models = {
        str(row["architecture"]): dict(row["model"]) for row in matrix["architectures"]
    }
    missing = set(finalists).difference(models)
    if missing:
        raise ValueError(f"finalists are absent from architecture matrix: {sorted(missing)}")
    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) != 2 or len(set(seed_values)) != 2 or any(seed < 0 for seed in seed_values):
        raise ValueError("Stage 2B requires exactly two distinct non-negative seeds")

    warm_starts: dict[str, dict[str, Any]] = {}
    if warm_starts_path is not None:
        warm_document = json.loads(Path(warm_starts_path).read_text(encoding="utf-8"))
        if warm_document.get("schema") != "connect4-v3-stage2b-warm-starts-v1":
            raise ValueError("unsupported Stage 2B warm-start manifest")
        for row in warm_document.get("warm_starts", ()):
            architecture = str(row["architecture"])
            if architecture in warm_starts:
                raise ValueError(f"duplicate warm start for architecture: {architecture}")
            if architecture not in finalists:
                raise ValueError(f"warm start is not a finalist: {architecture}")
            source = Path(row["checkpoint"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"warm-start model artifact does not exist: {source}")
            digest = _sha256_file(source)
            if digest != row.get("checkpoint_sha256"):
                raise ValueError(f"warm-start SHA-256 mismatch: {architecture}")
            payload = torch.load(source, map_location="cpu", weights_only=True)
            if payload.get("format") != "connect4-v3-model" or payload.get("format_version") != 1:
                raise ValueError("warm start must be a V3 model artifact")
            if payload.get("model_config") != models[architecture]:
                raise ValueError(f"warm-start model config mismatch: {architecture}")
            metadata = payload.get("metadata", {})
            if metadata.get("lineage") != "v3_stage2_offline":
                raise ValueError("warm start must have v3_stage2_offline lineage")
            if metadata.get("train_regime") != "standard_late":
                raise ValueError("warm start must be trained on standard_late")
            warm_starts[architecture] = {
                "checkpoint": str(source),
                "checkpoint_sha256": digest,
                "source_train_positions": metadata.get("train_positions"),
            }
        missing_warm = set(finalists).difference(warm_starts)
        if missing_warm:
            raise ValueError(f"warm starts missing finalists: {sorted(missing_warm)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    modes = ("cold", "warm") if warm_starts else ("cold",)
    for architecture in finalists:
        for seed in seed_values:
            for initialization in modes:
                run_id = f"stage2b_{architecture}_{initialization}_seed{seed}"
                raw = base.to_dict()
                warm = warm_starts.get(architecture) if initialization == "warm" else None
                raw["run"] = {
                    "run_id": run_id,
                    "seed": seed,
                    "run_dir": f"training/runs/stage2/selfplay/{run_id}",
                    "resume": False,
                    "warm_start_checkpoint": warm["checkpoint"] if warm else "",
                    "warm_start_checkpoint_sha256": warm["checkpoint_sha256"] if warm else "",
                    "warm_start_mode": "model_only_fresh_optimizer_replay_v1" if warm else "",
                }
                raw["model"] = models[architecture]
                config = V3Config.from_dict(raw)
                target = output / f"{run_id}.json"
                target.write_text(config.to_json(), encoding="utf-8")
                generated.append(
                    {
                        "architecture": architecture,
                        "initialization": initialization,
                        "seed": seed,
                        "config": str(target),
                        "run_id": run_id,
                        "canary_max_train_positions": 1_000_000,
                        "extension_max_train_positions": 5_000_000,
                        "optional_confirmation_max_train_positions": 10_000_000,
                        "warm_start": warm,
                    }
                )
    manifest = {
        "schema": "connect4-v3-stage2b-configs-v2",
        "base_config": str(Path(base_config_path)),
        "architecture_matrix": str(Path(architecture_matrix_path)),
        "finalists": list(finalists),
        "warm_starts": str(Path(warm_starts_path)) if warm_starts_path is not None else "",
        "runs": generated,
        "execution": "run each config with --execute and the recorded absolute position bound",
    }
    (output / "stage2b_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
