"""Resume the user-paused four-run FLA fork after the multiview CPU gate failed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_fla_bal5r1_fork4_2m import (  # noqa: E402
    BASE, CHILD_POSITIONS, NAMES, STATE, gpu0_idle, latest_commit, prepare, read,
)
from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402

MULTIVIEW = ROOT / "training/runs/stage2/fla/multiview_b8_1m"
CPU = MULTIVIEW / "cpu_gate"
CPU_RESULT = CPU / "multiview_resnet_b8c192_512.json"
CPU_SUMMARY = CPU / "summary.json"
REFERENCE = CPU / "reference_summary.json"
DONOR = MULTIVIEW / "runs/multiview_resnet_b8c192/standard_late/seed271828/model.pt"
CONFIG = MULTIVIEW / "configs/multiview_resnet_b8c192__standard_late__seed271828.json"


def cpu_failed_receipt() -> dict:
    result = read(CPU_RESULT)
    summary = read(CPU_SUMMARY)
    reference = read(REFERENCE)
    metadata = result["metadata"]
    if summary["machine"] != reference["machine"]:
        raise ValueError("multi-view CPU machine differs from original screen")
    if any(summary[k] != value for k, value in (
        ("simulations", 512), ("repeats", 3), ("idle_s", 10.0), ("group_idle_s", 60.0)
    )):
        raise ValueError("multi-view CPU group protocol mismatch")
    if any(metadata[k] != value for k, value in (
        ("mcts_sims", 512), ("repeats", 3), ("idle_s", 10.0)
    )):
        raise ValueError("multi-view CPU response protocol mismatch")
    if result["summary"]["searched_measurement_count"] != 45:
        raise ValueError("multi-view CPU search corpus incomplete")
    if metadata["artifact_sha256"] != sha256(DONOR) or metadata["config_sha256"] != sha256(CONFIG):
        raise ValueError("multi-view CPU result donor/config hash mismatch")
    row = summary["results"]
    if len(row) != 1 or row[0]["model_sha256"] != sha256(DONOR):
        raise ValueError("multi-view CPU summary identity mismatch")
    mean = float(result["summary"]["excluding_shortcuts"]["mean_s"])
    if mean <= 3.8:
        raise ValueError("multi-view CPU passed: do not resume old queue under failed-gate handoff")
    return {"variant": "multiview_resnet_b8c192", "mean_s": mean,
            "p95_s": result["summary"]["excluding_shortcuts"]["p95_s"],
            "result_sha256": sha256(CPU_RESULT), "summary_sha256": sha256(CPU_SUMMARY),
            "donor_sha256": sha256(DONOR), "config_sha256": sha256(CONFIG),
            "decision": "excluded_cpu_mean_gt_3p8_no_selfplay_or_matches"}


def verify_paused(queue: dict, row: dict) -> None:
    if queue.get("status") != "paused_by_user" or queue.get("active") != "gravity_b8":
        raise ValueError("original fork is not paused at gravity B8")
    if queue.get("completed") or queue.get("planned") != list(NAMES):
        raise ValueError("original fork queue order/completion drifted")
    if queue.get("active_config_sha256") != row["child_config_sha256"]:
        raise ValueError("paused active config hash mismatch")
    run_dir = Path(row["child_run_dir"])
    manifest = read(run_dir / "run_manifest.json")
    positions = manifest["formal_loop_state"]["train_positions_consumed"]
    if (manifest.get("status") != "stopped_at_safe_boundary"
            or manifest.get("stop_reason") != "drained_after_signal_15"
            or positions != queue["paused_train_positions"]
            or positions >= CHILD_POSITIONS):
        raise ValueError("gravity B8 did not pause at the expected committed boundary")
    pointer, commit = latest_commit(run_dir)
    if (pointer["generation"] != queue["paused_generation"]
            or pointer["commit_sha256"] != queue["paused_generation_commit_sha256"]):
        raise ValueError("paused commit pointer changed")
    if not commit["replay_shards"] or any(
        not (run_dir / shard["path"]).is_file() for shard in commit["replay_shards"]
    ):
        raise ValueError("paused replay inventory incomplete")
    metrics = (run_dir / "metrics/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    committed = [json.loads(line) for line in metrics if '"stage": "generation_commit"' in line]
    if not committed or committed[-1]["train_positions_consumed"] != positions:
        raise ValueError("paused learner metric does not match commit")


def wait_gpu0() -> None:
    deadline = time.monotonic() + 30 * 60
    while not gpu0_idle():
        if time.monotonic() > deadline:
            raise TimeoutError("GPU0 did not become idle within 30 minutes")
        time.sleep(30)


def run_one(row: dict, *, resume: bool) -> dict:
    name = row["name"]
    config = Path(row["child_config"])
    run_dir = Path(row["child_run_dir"])
    if sha256(config) != row["child_config_sha256"]:
        raise ValueError(f"fork config hash changed: {name}")
    if resume and not run_dir.exists():
        raise FileNotFoundError(f"paused child missing: {run_dir}")
    if not resume and run_dir.exists():
        raise FileExistsError(f"unstarted child already exists: {run_dir}")
    command = [sys.executable, "-u", "-B", "-m", "training.v3", "run",
               "--config", str(config)]
    if resume:
        command.append("--resume")
    plan = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if plan.returncode:
        raise RuntimeError(f"V3 guarded plan failed: {name}: {plan.stderr[-2000:]}")
    log = BASE / "logs" / f"{name}_resume_or_start.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([*command, "--execute", "--max-train-positions",
                                 str(CHILD_POSITIONS)], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"V3 child exited {result.returncode}: {name}: {log}")
    manifest = read(run_dir / "run_manifest.json")
    positions = int(manifest["formal_loop_state"]["train_positions_consumed"])
    if manifest.get("stop_reason") != "max_train_positions" or positions != CHILD_POSITIONS:
        raise ValueError(f"child did not reach exact 2M boundary: {name} {positions}")
    pointer, commit = latest_commit(run_dir)
    metrics = (run_dir / "metrics/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    commits = [json.loads(line) for line in metrics if '"stage": "generation_commit"' in line]
    if not commits or commits[-1]["train_positions_consumed"] != CHILD_POSITIONS:
        raise ValueError(f"learner metric missing 2M commit: {name}")
    if not commit["replay_shards"] or any(
        not (run_dir / shard["path"]).is_file() for shard in commit["replay_shards"]
    ):
        raise ValueError(f"replay inventory incomplete: {name}")
    receipt = {"name": name, "child_positions": positions,
               "cumulative_exposure_positions": 3_000_000,
               "generation": pointer["generation"],
               "generation_commit_sha256": pointer["commit_sha256"],
               "terminal_sha256": commit["checkpoint_sha256"],
               "accepted_sha256": commit["accepted_model_sha256"],
               "child_config_sha256": sha256(config)}
    atomic_json(BASE / "receipts" / f"{name}_child2m.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = prepare()
    cpu = cpu_failed_receipt()
    queue = read(STATE)
    verify_paused(queue, rows[0])
    if not args.execute:
        print(json.dumps({"status": "plan_only", "cpu_gate": cpu,
                          "paused_generation": queue["paused_generation"],
                          "paused_train_positions": queue["paused_train_positions"],
                          "remaining": list(NAMES)}, indent=2))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("physical GPU0 only; GPU1 reserved")
    atomic_json(MULTIVIEW / "cpu_gate_decision.json", cpu)
    wait_gpu0()
    queue.update(status="resuming_after_multiview_cpu_screen", cpu_gate_decision=cpu)
    atomic_json(STATE, queue)
    try:
        for index, row in enumerate(rows):
            name = row["name"]
            if index:
                wait_gpu0()
            queue.update(status="running", active=name,
                         active_config_sha256=row["child_config_sha256"])
            atomic_json(STATE, queue)
            receipt = run_one(row, resume=index == 0)
            queue["completed"].append(receipt)
            queue.update(active=None)
            atomic_json(STATE, queue)
        queue.update(status="complete", active=None)
        atomic_json(STATE, queue)
        return 0
    except Exception as exc:
        queue.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(STATE, queue)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
