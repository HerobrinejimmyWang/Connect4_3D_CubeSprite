"""Export immutable FLA parent-1M terminal snapshots for CPU search diagnosis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from stage2_fla_bal5r1_fork4_2m import NAMES, latest_commit
from stage2_fla_queue_trim_resume import verify_completion
from stage2_fla_selfplay_queue import QUEUE_ROOT, ROOT, atomic_json, sha256
from training.v3.config import V3Config, model_config_dict
from training.v3.evaluation_snapshot import export_evaluation_snapshot
from training.v3.pipeline import lineage_config_hash


OUTPUT = ROOT / "training/runs/stage2/fla/latency_1m_vs_2m/parent_1m"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export() -> dict:
    queue = read(ROOT / QUEUE_ROOT / "queue_state.json")
    completed = {item["name"]: item for item in queue["completed"]}
    if queue["status"] != "complete" or not set(NAMES).issubset(completed):
        raise ValueError("four parent 1M runs have not all completed")
    rows = {}
    for name in NAMES:
        source = completed[name]
        run_dir = Path(source["run_dir"])
        receipt = verify_completion(run_dir)
        pointer, commit = latest_commit(run_dir)
        checkpoint = run_dir / commit["checkpoint"]
        config = ROOT / QUEUE_ROOT / "configs" / f"{run_dir.name}.json"
        config_raw = read(config)
        config_hash = lineage_config_hash(V3Config.from_dict(config_raw))
        if (receipt["train_positions_consumed"] != 1_000_000
                or receipt["terminal_sha256"] != commit["checkpoint_sha256"]
                or sha256(config) != source["config_sha256"]
                or commit["config_hash"] != config_hash):
            raise ValueError(f"parent 1M commit/config identity differs: {name}")
        snapshot = OUTPUT / f"{name}.pt"
        snapshot_receipt = OUTPUT / f"{name}.receipt.json"
        if snapshot.exists() != snapshot_receipt.exists():
            raise ValueError(f"incomplete existing parent snapshot: {name}")
        if snapshot.exists():
            exported = read(snapshot_receipt)
            if (sha256(snapshot) != exported["output_sha256"]
                    or exported["source_checkpoint_sha256"] != sha256(checkpoint)):
                raise ValueError(f"existing parent snapshot drift: {name}")
        else:
            exported = export_evaluation_snapshot(
                checkpoint, snapshot, model_id=f"fla-parent1m-terminal-{name}")
            atomic_json(snapshot_receipt, exported)
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if (metadata["train_positions_consumed"] != 1_000_000
                or metadata["source_checkpoint_config_hash"] != config_hash
                or payload["model_config"] != model_config_dict(
                    V3Config.from_dict(config_raw).model)):
            raise ValueError(f"parent snapshot metadata/model differs: {name}")
        rows[name] = {
            "run_dir": str(run_dir), "generation": pointer["generation"],
            "config": str(config), "config_sha256": sha256(config),
            "lineage_hash": config_hash,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": commit["checkpoint_sha256"],
            "accepted_sha256": commit["accepted_model_sha256"],
            "snapshot": str(snapshot), "snapshot_sha256": sha256(snapshot),
            "snapshot_receipt_sha256": sha256(snapshot_receipt),
            "train_positions_consumed": 1_000_000,
        }
    result = {
        "schema": "connect4-stage2-fla-parent1m-diagnostic-snapshots-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_queue_sha256": sha256(ROOT / QUEUE_ROOT / "queue_state.json"),
        "models": rows,
    }
    target = OUTPUT / "manifest.json"
    if target.exists():
        previous = read(target)
        previous.pop("created_at_utc")
        result.pop("created_at_utc")
        if previous != result:
            raise ValueError("existing parent snapshot manifest differs")
    else:
        atomic_json(target, result)
    return read(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        result = export()
        print(json.dumps({name: row["snapshot_sha256"] for name, row
                          in result["models"].items()}, indent=2))
    else:
        print(json.dumps({"source": str(ROOT / QUEUE_ROOT / "queue_state.json"),
                          "output": str(OUTPUT), "names": NAMES}, indent=2))


if __name__ == "__main__":
    main()
