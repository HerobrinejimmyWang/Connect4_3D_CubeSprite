"""Produce a resume config that changes ONLY operational topology, with a receipt.

Reads the base config, applies the selected operational topology, refuses to write
unless the SEMANTIC config hash is byte-identical before and after, and emits an
adaptation receipt next to the config.

Usage:
  prepare_operational_resume_config.py \
      --base-config <path> --output <path> --receipt <path> \
      --actors N --devices cuda:0[,cuda:1] \
      --selection-reason "..." --benchmark-summary <path> \
      --previous-topology "<text>" --latency-prediction "<text>"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from training.v3.config import config_hash, load_config
from training.v3.pipeline import lineage_config_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--actors", type=int, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--selection-reason", required=True)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--previous-topology", required=True)
    parser.add_argument("--latency-prediction", default="")
    args = parser.parse_args()

    raw = json.loads(args.base_config.read_text(encoding="utf-8"))
    before = load_config(args.base_config)
    before_semantic = lineage_config_hash(before)

    if int(before.runtime.mcts_lanes_per_actor) != 4:
        raise RuntimeError("refusing to touch a config whose lanes are not 4")

    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    if not devices:
        raise ValueError("at least one device required")

    runtime = raw["runtime"]
    runtime["device"] = devices[0]
    runtime["selfplay_devices"] = devices
    runtime["actor_processes"] = int(args.actors)
    # lanes deliberately NOT touched; operational capacity only
    runtime.setdefault("inference_batch_size", 32)

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    after = load_config(temporary)
    after_semantic = lineage_config_hash(after)

    if after_semantic != before_semantic:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "operational adaptation changed the semantic config hash: %s != %s"
            % (after_semantic, before_semantic)
        )
    # Checkpoint compatibility: the run manifest hash the run has been carrying.
    run_manifest = json.loads((args.base_config.parents[2] / "runs"
                               / args.base_config.stem / "run_manifest.json").read_text()) \
        if (args.base_config.parents[2] / "runs" / args.base_config.stem / "run_manifest.json").is_file() \
        else {}
    manifest_hash = run_manifest.get("config_hash")

    os.replace(temporary, args.output)

    receipt = {
        "schema": "connect4-bal5-operational-adaptation-receipt-v1",
        "base_config": str(args.base_config),
        "output_config": str(args.output),
        "previous_topology": args.previous_topology,
        "new_topology": {
            "actor_processes": int(args.actors),
            "selfplay_devices": devices,
            "device": devices[0],
            "mcts_lanes_per_actor": after.runtime.mcts_lanes_per_actor,
            "inference_batch_size": after.runtime.inference_batch_size,
        },
        "semantic_fields_untouched": {
            "mcts_lanes_per_actor": after.runtime.mcts_lanes_per_actor,
            "learner_batch_size": after.learner.batch_size,
            "learner_amp": after.runtime.learner_amp,
            "full_search_sims_check": "inherited from base config",
        },
        "config_hash": {
            "before": before_semantic,
            "after": after_semantic,
            "equal": after_semantic == before_semantic,
            "run_manifest_hash": manifest_hash,
            "run_manifest_matches": (manifest_hash == after_semantic) if manifest_hash else None,
        },
        "selection": {
            "reason": args.selection_reason,
            "benchmark_summary": str(args.benchmark_summary),
        },
        "latency_prediction": args.latency_prediction,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(args.output),
        "receipt": str(args.receipt),
        "semantic_hash_unchanged": after_semantic == before_semantic,
        "actors": int(args.actors),
        "devices": devices,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
