"""Run the additional B6C128 self-play group only after its CPU screen passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stage2_fla_queue_trim_resume import verify_completion
from stage2_fla_selfplay_queue import (
    POSITIONS, QUEUE_ROOT, ROOT, SEED, TEMPLATE, atomic_json, sha256, validate_donor,
)
from training.v3.config import V3Config

NAME = "raw3d_to2d_thin_b6c128"
SOURCE = ROOT / "training/runs/stage2/fla/b6_raw_2m"
CONFIG = SOURCE / "configs" / f"{NAME}__standard_late__seed{SEED}.json"
MODEL = SOURCE / "runs" / NAME / "standard_late" / f"seed{SEED}" / "model.pt"
REPORT = MODEL.parent / "report.json"
EXPECTED_COMPLETED = [
    "gravity_b8", "column_2d_b8c192", "raw3d_to2d_b8",
    "raw3d_to2d_thin_b8c192", "column3d_v2_thin_b8c192",
]


def validate_cpu(result_path: Path, summary_path: Path, reference_summary_path: Path,
                 *, config_path: Path = CONFIG, model_path: Path = MODEL,
                 variant: str = NAME) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    metadata = result["metadata"]
    for field, expected in (("mcts_sims", 512), ("repeats", 3), ("idle_s", 10.0)):
        if metadata[field] != expected:
            raise ValueError(f"B6 CPU protocol mismatch: {field}")
    for field, expected in (("simulations", 512), ("repeats", 3), ("idle_s", 10.0), ("group_idle_s", 60.0)):
        if summary[field] != expected:
            raise ValueError(f"B6 CPU group protocol mismatch: {field}")
    if summary["machine"] != reference["machine"]:
        raise ValueError("B6 CPU hardware/runtime differs from B8 screen")
    if result["summary"]["searched_measurement_count"] != 45:
        raise ValueError("B6 CPU search-state count is incomplete")
    if metadata["config_sha256"] != sha256(config_path) or metadata["artifact_sha256"] != sha256(model_path):
        raise ValueError("B6 CPU evidence does not match donor config/model")
    rows = summary["results"]
    if len(rows) != 1 or rows[0]["variant"] != variant or rows[0]["model_sha256"] != sha256(model_path):
        raise ValueError("B6 CPU summary identity mismatch")
    mean = result["summary"]["excluding_shortcuts"]["mean_s"]
    if mean > 3.8:
        raise ValueError(f"B6 CPU mean {mean:.3f}s exceeds 3.8s screen upper bound")
    return {"mean_s": mean, "result_sha256": sha256(result_path), "summary_sha256": sha256(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-result", type=Path, required=True)
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    queue_path = ROOT / QUEUE_ROOT / "queue_state.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue["status"] != "awaiting_b6_cpu_gate" or [x["name"] for x in queue["completed"]] != EXPECTED_COMPLETED:
        raise ValueError("retained serial queue has not completed its verified prefix")
    watcher = json.loads((SOURCE / "offline_watcher_state.json").read_text(encoding="utf-8"))
    model_config, donor_sha = validate_donor(CONFIG, MODEL, REPORT)
    if watcher.get("status") != "complete" or watcher.get("model_sha256") != donor_sha:
        raise ValueError("B6 offline watcher receipt is incomplete or mismatched")
    cpu = validate_cpu(args.cpu_result, args.cpu_summary, args.reference_summary)
    run_id = f"fla2m_r1_{NAME}_warm_seed{SEED}"
    run_dir = ROOT / QUEUE_ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"B6 self-play run already exists: {run_dir}")
    template = json.loads((ROOT / TEMPLATE).read_text(encoding="utf-8"))
    if template["selfplay"]["rule_id"] != "classic":
        raise ValueError("B6 self-play requires Classic rule")
    template["run"] = {
        "run_id": run_id, "seed": SEED, "run_dir": str(run_dir), "resume": False,
        "warm_start_checkpoint": str(MODEL),
        "warm_start_checkpoint_sha256": donor_sha,
        "warm_start_mode": "model_only_fresh_optimizer_replay_v1",
    }
    template["model"] = model_config
    config = V3Config.from_dict(template)
    config_path = ROOT / QUEUE_ROOT / "configs" / f"{run_id}.json"
    if not args.execute:
        print(json.dumps({"status": "plan_only", "run_id": run_id, "cpu": cpu, "donor_sha256": donor_sha}))
        return 0
    config_path.write_text(config.to_json(), encoding="utf-8")
    config_sha = sha256(config_path)
    plan = subprocess.run(
        [sys.executable, "-B", "-m", "training.v3", "run", "--config", str(config_path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if plan.returncode:
        raise RuntimeError(f"V3 guarded plan failed: {plan.stderr[-2000:]}")
    queue.update(status="running_b6", active=NAME, config_sha256=config_sha,
                 donor_sha256=donor_sha, b6_cpu_evidence=cpu)
    atomic_json(queue_path, queue)
    log = ROOT / QUEUE_ROOT / "logs" / f"{NAME}.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, "-u", "-B", "-m", "training.v3", "run", "--config", str(config_path),
             "--execute", "--max-train-positions", str(POSITIONS)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        queue.update(status="failed", error=f"B6 self-play exited {result.returncode}; see {log}")
        atomic_json(queue_path, queue)
        raise RuntimeError(queue["error"])
    receipt = verify_completion(run_dir)
    queue["completed"].append({"name": NAME, "run_dir": str(run_dir),
                               "config_sha256": config_sha, "donor_sha256": donor_sha, **receipt})
    queue.update(status="complete", active=None)
    atomic_json(queue_path, queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
