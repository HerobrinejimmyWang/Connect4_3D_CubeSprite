"""Compare the new full B6 donor against three archived FLA donor models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.evaluation import load_opening_manifest  # noqa: E402
from training.v3.evaluation_runtime import EvaluationModelSource, play_paired_openings_replicated  # noqa: E402
from training.v3.gate import summarize_paired_results  # noqa: E402

OLD = ROOT / "training/runs/stage2/fla/direct_1m/r2_four_workers"
BASE = ROOT / "training/runs/stage2/fla/direct_1m/r3_raw_b6_full"
NAME = "raw3d_to2d_full_b6c128"
NEW = ROOT / "training/runs/stage2/fla/b6_raw_full_2m"
MODEL = NEW / "runs" / NAME / "standard_late/seed271828/model.pt"
CONFIG = NEW / "configs" / f"{NAME}__standard_late__seed271828.json"
PEERS = ("raw3d_to2d_thin_b6c128", "raw3d_to2d_b8", "gravity_b8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = json.loads((MODEL.parent / "report.json").read_text(encoding="utf-8"))
    if not report["training_execution"]["target_positions_reached"] or sha256(MODEL) != report["model_artifact"]["sha256"]:
        raise ValueError("new donor report/model mismatch or incomplete")
    design = json.loads((NEW / "design.json").read_text(encoding="utf-8"))
    if sha256(CONFIG) != design["config_sha256"]:
        raise ValueError("new donor config differs from frozen design")
    old = json.loads((OLD / "inputs.json").read_text(encoding="utf-8"))["model_identity"]
    openings_path = OLD / "classic_openings_32.json"
    openings = load_opening_manifest(openings_path)
    if len(openings) != 32 or any(item.rule_id != "classic" for item in openings):
        raise ValueError("old direct opening set is not 32 Classic positions")
    inputs = {"name": NAME, "model": str(MODEL), "model_sha256": sha256(MODEL),
              "config": str(CONFIG), "config_sha256": sha256(CONFIG),
              "opening_manifest": str(openings_path), "opening_sha256": sha256(openings_path),
              "peers": {name: {"path": old[name]["donor"], "sha256": old[name]["donor_sha256"]}
                        for name in PEERS},
              "search_sims": 256, "cpuct": 1.5, "opening_pairs": 32,
              "worker_processes": 4, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if not args.execute:
        print(json.dumps(inputs, indent=2))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("this direct match is reserved for physical GPU0")
    BASE.mkdir(parents=True, exist_ok=True)
    inputs_path = BASE / "inputs.json"
    if inputs_path.exists() and json.loads(inputs_path.read_text(encoding="utf-8")) != inputs:
        raise ValueError("direct match identity changed")
    inputs_path.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    for name in PEERS:
        peer = inputs["peers"][name]
        peer_path = Path(peer["path"])
        if sha256(peer_path) != peer["sha256"]:
            raise ValueError(f"peer donor hash mismatch: {name}")
        target = BASE / f"{NAME}__vs__{name}.json"
        if target.exists():
            result = json.loads(target.read_text(encoding="utf-8"))
            if result["model_sha256"] != {"a": inputs["model_sha256"], "b": peer["sha256"]}:
                raise ValueError(f"existing result model hash mismatch: {target}")
            continue
        evaluated = play_paired_openings_replicated(
            openings,
            candidate_source=EvaluationModelSource("v3_artifact", str(MODEL), NAME),
            incumbent_source=EvaluationModelSource("v3_artifact", str(peer_path), name),
            search_sims=256, cpuct=1.5, worker_devices=("cuda:0",) * 4,
        )
        summary = summarize_paired_results(evaluated.games, bootstrap_samples=2000,
                                           bootstrap_seed=271828)
        target.write_text(json.dumps({
            "schema": "connect4-stage2-fla-b6-full-direct-v1", "model_a": NAME,
            "model_b": name, "model_sha256": {"a": inputs["model_sha256"], "b": peer["sha256"]},
            "opening_sha256": inputs["opening_sha256"], "search_sims": 256,
            "cpuct": 1.5, "runtime": evaluated.metrics.to_dict(),
            "games": [asdict(game) for game in evaluated.games], "summary": asdict(summary),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {summary.overall.wins}/{summary.overall.draws}/{summary.overall.losses}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
