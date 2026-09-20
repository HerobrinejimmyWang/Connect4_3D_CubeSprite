"""Create a single-GPU operational resume config without changing V3 lineage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.config import config_hash, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.base_config.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    topology = benchmark["selected_resume_topology"]
    original = load_config(args.base_config)
    original_lanes = int(original.runtime.mcts_lanes_per_actor)
    if int(topology["lanes"]) != original_lanes:
        raise RuntimeError("refusing to change semantic MCTS lane count during resume")

    runtime = raw["runtime"]
    runtime["device"] = "cuda:0"
    runtime["selfplay_devices"] = ["cuda:0"]
    runtime["actor_processes"] = int(topology["actors"])
    runtime["mcts_lanes_per_actor"] = original_lanes
    runtime["inference_batch_size"] = 32
    runtime["evaluation_devices"] = []
    runtime["evaluation_parallel_games"] = 8
    runtime["evaluation_inference_batch_size"] = 32
    runtime["evaluation_inference_batch_timeout_ms"] = 1.0
    runtime["evaluation_replicas_per_device"] = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    adapted = load_config(temporary)
    if config_hash(adapted) != config_hash(original):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("single-GPU operational adaptation changed the semantic config hash")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "base_config": str(args.base_config),
                "output": str(args.output),
                "config_hash": config_hash(adapted),
                "actors": adapted.runtime.actor_processes,
                "lanes": adapted.runtime.mcts_lanes_per_actor,
                "selfplay_devices": list(adapted.runtime.selfplay_devices),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
