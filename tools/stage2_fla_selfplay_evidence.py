"""Write a hash-verified summary of the retained FLA 1M self-play runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stage2_fla_queue_trim_resume import verify_completion
from stage2_fla_selfplay_queue import QUEUE_ROOT, ROOT, sha256

EXPECTED_5 = [
    "gravity_b8", "column_2d_b8c192", "raw3d_to2d_b8",
    "raw3d_to2d_thin_b8c192", "column3d_v2_thin_b8c192",
]
EXPECTED_6 = [*EXPECTED_5, "raw3d_to2d_thin_b6c128"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue_path = ROOT / QUEUE_ROOT / "queue_state.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    completed_names = [row["name"] for row in queue["completed"]]
    if completed_names == EXPECTED_5:
        if queue.get("status") != "awaiting_b6_cpu_gate":
            raise ValueError("five-run receipt requires the pending B6 CPU gate")
    elif completed_names == EXPECTED_6:
        if queue.get("status") != "complete":
            raise ValueError("six-run receipt requires a completed serial queue")
    else:
        raise ValueError("retained 1M queue is incomplete or reordered")
    rows = []
    for row in queue["completed"]:
        run_dir = Path(row["run_dir"])
        receipt = verify_completion(run_dir)
        pointer = json.loads((run_dir / "manifests/latest_generation.json").read_text(encoding="utf-8"))
        commit = json.loads((run_dir / pointer["commit"]).read_text(encoding="utf-8"))
        config = ROOT / QUEUE_ROOT / "configs" / f"{run_dir.name}.json"
        if sha256(config) != row["config_sha256"]:
            raise ValueError(f"run config hash mismatch: {config}")
        if "terminal_sha256" in row and row["terminal_sha256"] != receipt["terminal_sha256"]:
            raise ValueError(f"queue terminal hash mismatch: {row['name']}")
        if "accepted_sha256" in row and row["accepted_sha256"] != receipt["accepted_sha256"]:
            raise ValueError(f"queue accepted hash mismatch: {row['name']}")
        rows.append({
            "name": row["name"],
            "run_dir": row["run_dir"],
            "config_sha256": row["config_sha256"],
            "donor_sha256": row["donor_sha256"],
            "generation_commit": pointer["commit"],
            "generation_commit_sha256": pointer["commit_sha256"],
            "replay_shard_count": len(commit["replay_shards"]),
            "gate_verdict": commit["gate_verdict"],
            **receipt,
        })
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "connect4-stage2-fla-selfplay-verified-1m-v1",
        "source_queue": str(queue_path),
        "source_queue_sha256": sha256(queue_path),
        "status": "remote_artifacts_verified",
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
