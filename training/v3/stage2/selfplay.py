"""Generate explicit cold-start V3 configs for Stage 2B finalists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..config import V3Config, load_config


def generate_stage2b_configs(
    *,
    base_config_path: str | Path,
    architecture_matrix_path: str | Path,
    finalists_path: str | Path,
    output_dir: str | Path,
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

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for architecture in finalists:
        for seed in seed_values:
            run_id = f"stage2b_{architecture}_seed{seed}"
            raw = base.to_dict()
            raw["run"] = {
                "run_id": run_id,
                "seed": seed,
                "run_dir": f"training/runs/stage2/selfplay/{run_id}",
                "resume": False,
                "warm_start_checkpoint": "",
                "warm_start_checkpoint_sha256": "",
                "warm_start_mode": "",
            }
            raw["model"] = models[architecture]
            config = V3Config.from_dict(raw)
            target = output / f"{run_id}.json"
            target.write_text(config.to_json(), encoding="utf-8")
            generated.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "config": str(target),
                    "run_id": run_id,
                    "canary_max_train_positions": 1_000_000,
                    "extension_max_train_positions": 5_000_000,
                }
            )
    manifest = {
        "schema": "connect4-v3-stage2b-coldstart-configs-v1",
        "base_config": str(Path(base_config_path)),
        "architecture_matrix": str(Path(architecture_matrix_path)),
        "finalists": list(finalists),
        "runs": generated,
        "execution": "run each config with --execute and the recorded absolute position bound",
    }
    (output / "stage2b_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
