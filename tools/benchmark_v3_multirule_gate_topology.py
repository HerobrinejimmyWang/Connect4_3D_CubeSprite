"""Compare one-card and two-card V3 gate evaluation on identical opening pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_core.rules import BAL5_R2_RULE_REGISTRY
from training.v3.config import load_config
from training.v3.evaluation import load_opening_manifest
from training.v3.evaluation_runtime import (
    EvaluationModelSource,
    play_paired_openings_replicated,
)
from training.v3.multirule_gate import BAL5_R2_RULE_IDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardware_snapshot() -> dict:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    gpus = []
    for line in query.stdout.splitlines():
        index, name, utilization, used, total = (part.strip() for part in line.split(",", 4))
        gpus.append({
            "index": int(index), "name": name, "utilization_percent": int(utilization),
            "memory_used_mib": int(used), "memory_total_mib": int(total),
        })
    cgroup = {}
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            cgroup[name] = path.read_text(encoding="utf-8").strip()
    return {"gpus": gpus, "cgroup": cgroup, "logical_cpus": os.cpu_count()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--rule", choices=BAL5_R2_RULE_IDS, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    gate_path = args.gate.resolve()
    if not gate_path.is_relative_to(run_dir / "metrics") or args.pairs < 1:
        parser.error("gate must be inside run-dir/metrics and pairs must be positive")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    config = load_config(run_dir / "resolved_config.json")
    index_path = run_dir / gate["opening_manifest"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    opening_ref = index["rules"][args.rule]
    opening_path = run_dir / opening_ref["path"]
    if _sha256(opening_path) != opening_ref["sha256"]:
        raise ValueError("opening manifest checksum mismatch")
    openings = load_opening_manifest(opening_path, registry=BAL5_R2_RULE_REGISTRY)
    if args.pairs > min(len(openings), gate["opening_pairs_per_rule"]):
        parser.error("pairs exceed committed gate evidence")
    candidate_id = gate["candidate_model_id"]
    candidate_paths = [run_dir / part / f"{candidate_id}.pt"
                       for part in ("accepted", "rejected")]
    candidate_path = next((path for path in candidate_paths if path.is_file()), None)
    if candidate_path is None:
        raise FileNotFoundError(f"candidate artifact missing: {candidate_id}")
    incumbent_id = gate["incumbent_model_id"]
    incumbent_path = run_dir / "accepted" / f"{incumbent_id}.pt"
    if not incumbent_path.is_file():
        raise FileNotFoundError(f"incumbent artifact missing: {incumbent_path}")
    plan = {
        "schema": "v3-multirule-gate-topology-comparison-v1",
        "gate_sha256": _sha256(gate_path),
        "opening_manifest_sha256": _sha256(opening_path),
        "candidate_sha256": _sha256(candidate_path),
        "incumbent_sha256": _sha256(incumbent_path),
        "rule": args.rule,
        "pairs": args.pairs,
        "search_sims": gate["search_sims"],
        "cpuct": config.gate.cpuct,
        "topologies": [["cuda:0"] * 4, ["cuda:0"] * 4 + ["cuda:1"] * 4],
    }
    if not args.execute:
        print(json.dumps({"status": "plan_only", **plan}, indent=2))
        return

    results = []
    for devices in plan["topologies"]:
        before = _hardware_snapshot()
        if any(gpu["utilization_percent"] > 5 for gpu in before["gpus"]
               if gpu["index"] in (0, 1)):
            raise RuntimeError("GPU0 and GPU1 must be idle before topology calibration")
        samples = []
        stop_sampling = threading.Event()

        def sample_hardware() -> None:
            while not stop_sampling.wait(2.0):
                samples.append(_hardware_snapshot())

        sampler = threading.Thread(target=sample_hardware, daemon=True)
        sampler.start()
        try:
            result = play_paired_openings_replicated(
                openings[:args.pairs],
                candidate_source=EvaluationModelSource(
                    "v3_artifact", str(candidate_path), candidate_id
                ),
                incumbent_source=EvaluationModelSource(
                    "v3_artifact", str(incumbent_path), incumbent_id
                ),
                search_sims=gate["search_sims"],
                cpuct=config.gate.cpuct,
                worker_devices=devices,
            )
        finally:
            stop_sampling.set()
            sampler.join(timeout=5.0)
        results.append({
            "devices": devices,
            "games": [asdict(game) for game in result.games],
            "metrics": result.metrics.to_dict(),
            "hardware_before": before,
            "hardware_after": _hardware_snapshot(),
            "hardware_samples": samples,
        })
    same_games = results[0]["games"] == results[1]["games"]
    single_seconds = results[0]["metrics"]["wall_seconds"]
    dual_seconds = results[1]["metrics"]["wall_seconds"]
    report = {
        **plan,
        "status": "complete",
        "same_game_results": same_games,
        "speedup": single_seconds / dual_seconds,
        "recommended_topology": "dual_gpu" if same_games and dual_seconds < single_seconds else "single_gpu",
        "results": results,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
