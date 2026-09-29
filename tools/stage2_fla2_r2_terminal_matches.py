"""Evaluate FLA2-R2 terminal models on 100 swapped openings per rule.

Comparison A: independent 2M raw B8 versus independent 2M thin B8.
Comparison B: independent 2M raw B8 versus no-mixture 1M child from raw g16.
The latter is a directional comparison because optimizer/replay histories differ.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_core.rules import BAL5_R2_RULE_REGISTRY  # noqa: E402
from stage2_fla2_r2_no_mixture_fork import (  # noqa: E402
    CONFIG as CONTROL_CONFIG,
    RUN_DIR as CONTROL_RUN,
    STATE as CONTROL_STATE,
    verify_boundary as verify_control_boundary,
)
from stage2_fla2_r2_watch import (  # noqa: E402
    BASE, PLAN_PATH, TARGET_POSITIONS, verify_boundary,
)
from stage2_fla_bal5r1_fork4_2m import latest_commit, read  # noqa: E402
from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402
from training.v3.evaluation import (  # noqa: E402
    build_openings, load_opening_manifest, write_opening_manifest,
)
from training.v3.evaluation_runtime import (  # noqa: E402
    EvaluationModelSource, play_paired_openings_replicated,
)
from training.v3.evaluation_snapshot import export_evaluation_snapshot  # noqa: E402
from training.v3.gate import summarize_paired_results  # noqa: E402

OUT = BASE / "direct_terminal_5rule_100pairs"
STATE = OUT / "watcher_state.json"
RULES = tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs)
PAIRS_PER_RULE = 100
SIMS = 256
CPUCT = 1.5
PHYSICAL_GPU = 0
WORKERS = 4
SEED = 271828
COMPARISONS = {
    "architecture": ("raw3d_to2d_b8", "raw3d_to2d_thin_b8c192"),
    "pool": ("raw3d_to2d_b8", "raw3d_to2d_b8_no_mixture_g16"),
}
GPU_BY_COMPARISON = {"architecture": 0, "pool": 1}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_idle(index: int = PHYSICAL_GPU) -> bool:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    selected = next(line.split(",", 1)[1].strip() for line in gpu.stdout.splitlines()
                    if line.split(",", 1)[0].strip() == str(index))
    return not any(line.split(",", 1)[-1].strip() == selected
                   for line in apps.stdout.splitlines() if "," in line)


def verified_inputs(*, require_thin: bool = True) -> dict[str, dict]:
    plan = read(PLAN_PATH)
    if (plan["rule_registry_hash"] != BAL5_R2_RULE_REGISTRY.registry_hash
            or tuple(plan["rule_ids"]) != RULES):
        raise ValueError("five-rule plan/registry differs")
    rows: dict[str, dict] = {}
    for index, row in enumerate(plan["rows"]):
        if index == 1 and not require_thin:
            continue
        state = read(BASE / f"queue_row{index}.json")
        if state.get("status") != "complete" or state.get("name") != row["name"]:
            raise RuntimeError(f"training not complete: {row['name']}: {state}")
        receipt_path = BASE / "receipts" / f"{row['name']}_to2m.json"
        actual = verify_boundary(row, TARGET_POSITIONS, canary=False)
        if read(receipt_path) != actual:
            raise ValueError(f"2M receipt drift: {row['name']}")
        _, commit = latest_commit(Path(row["run_dir"]))
        rows[row["name"]] = {
            "config": row["config"], "config_sha256": row["config_sha256"],
            "run_dir": row["run_dir"], "positions": TARGET_POSITIONS,
            "generation_commit_sha256": actual["generation_commit_sha256"],
            "checkpoint": str(Path(row["run_dir"]) / commit["checkpoint"]),
            "checkpoint_sha256": actual["checkpoint_sha256"],
            "accepted_sha256": actual["accepted_sha256"],
            "receipt_sha256": sha256(receipt_path),
        }
    expected = ({"raw3d_to2d_b8", "raw3d_to2d_thin_b8c192"}
                if require_thin else {"raw3d_to2d_b8"})
    if set(rows) != expected:
        raise ValueError("unexpected FLA2-R2 architecture rows")
    if read(CONTROL_STATE).get("status") != "complete":
        raise RuntimeError("no-mixture control not complete")
    receipt_path = CONTROL_RUN.parents[1] / "receipts/to1m.json"
    actual = verify_control_boundary(1_000_000, canary=False)
    receipt = read(receipt_path)
    actual.pop("verified_at_utc", None)
    receipt.pop("verified_at_utc", None)
    if receipt != actual:
        raise ValueError("no-mixture receipt drift")
    _, commit = latest_commit(CONTROL_RUN)
    rows["raw3d_to2d_b8_no_mixture_g16"] = {
        "config": str(CONTROL_CONFIG), "config_sha256": sha256(CONTROL_CONFIG),
        "run_dir": str(CONTROL_RUN), "positions": 1_000_000,
        "generation_commit_sha256": actual["generation_commit_sha256"],
        "checkpoint": str(CONTROL_RUN / commit["checkpoint"]),
        "checkpoint_sha256": actual["checkpoint_sha256"],
        "accepted_sha256": actual["accepted_model_sha256"],
        "receipt_sha256": sha256(receipt_path),
        "source_parent_positions": 1_025_352,
        "lineage_note": "g16 accepted warm start, fresh optimizer and replay",
    }
    raw_config = read(Path(rows["raw3d_to2d_b8"]["config"]))
    control_config = read(CONTROL_CONFIG)
    if raw_config["model"] != control_config["model"]:
        raise ValueError("pool comparison changed model architecture")
    if actual["config_sha256"] != rows["raw3d_to2d_b8_no_mixture_g16"]["config_sha256"]:
        raise ValueError("no-mixture config differs from boundary receipt")
    return rows


def snapshot_rows(rows: dict[str, dict]) -> None:
    snapshot_dir = OUT / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with (OUT / "snapshots.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, row in rows.items():
            target = snapshot_dir / f"{name}.pt"
            receipt_path = snapshot_dir / f"{name}.json"
            if target.exists() != receipt_path.exists():
                raise ValueError(f"incomplete terminal snapshot: {name}")
            if not target.exists():
                receipt = export_evaluation_snapshot(
                    row["checkpoint"], target, model_id=f"fla2-r2-terminal-{name}")
                atomic_json(receipt_path, receipt)
            receipt = read(receipt_path)
            if (sha256(target) != receipt["output_sha256"]
                    or receipt["source_checkpoint_sha256"] != row["checkpoint_sha256"]
                    or receipt["train_positions_consumed"] != row["positions"]):
                raise ValueError(f"terminal snapshot identity drift: {name}")
            row["snapshot"] = str(target)
            row["snapshot_sha256"] = receipt["output_sha256"]


def freeze_inputs(rows: dict[str, dict]) -> dict:
    snapshot_rows(rows)
    payload = {
        "schema": "connect4-stage2-fla2-r2-terminal-five-rule-inputs-v1",
        "models": rows,
        "comparisons": {name: list(pair) for name, pair in COMPARISONS.items()},
        "rules": list(RULES),
        "rule_registry_hash": BAL5_R2_RULE_REGISTRY.registry_hash,
        "opening_pairs_per_rule_per_comparison": PAIRS_PER_RULE,
        "search_sims": SIMS, "cpuct": CPUCT,
        "gpu_by_comparison": GPU_BY_COMPARISON, "workers": WORKERS,
        "endpoint": "terminal_checkpoint_weights",
        "pool_caveat": "separate V3 lineages; different optimizer and replay histories",
    }
    path = OUT / "inputs.json"
    if path.exists():
        if read(path) != payload:
            raise ValueError("frozen match inputs changed")
    else:
        atomic_json(path, payload)
    return payload


def rule_openings(rule: str, rule_index: int) -> tuple[Path, tuple]:
    path = OUT / "openings" / f"{rule}.json"
    if not path.exists():
        rows = build_openings(
            PAIRS_PER_RULE, run_seed=20260928 + rule_index * 1009,
            rule_id=rule, registry=BAL5_R2_RULE_REGISTRY,
            opening_id_prefix=f"fla2r2-{rule}",
        )
        write_opening_manifest(path, rows, registry=BAL5_R2_RULE_REGISTRY)
    rows = load_opening_manifest(path, registry=BAL5_R2_RULE_REGISTRY)
    if len(rows) != PAIRS_PER_RULE or any(item.rule_id != rule for item in rows):
        raise ValueError(f"opening manifest changed: {rule}")
    return path, rows


def run_match(inputs: dict, comparison: str, rule: str, openings_path: Path,
              openings: tuple, *, physical_gpu: int = PHYSICAL_GPU) -> dict:
    name_a, name_b = inputs["comparisons"][comparison]
    a, b = inputs["models"][name_a], inputs["models"][name_b]
    output = OUT / "matches" / comparison / f"{rule}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_hashes = {"a": a["snapshot_sha256"], "b": b["snapshot_sha256"]}
    if output.exists():
        result = read(output)
        if (result["model_a"] != name_a or result["model_b"] != name_b
                or result["comparison"] != comparison or result["rule_id"] != rule
                or result["model_sha256"] != expected_hashes
                or result["opening_sha256"] != sha256(openings_path)
                or result["search_sims"] != SIMS
                or result["cpuct"] != CPUCT
                or result["physical_gpu"] != physical_gpu
                or result["cuda_visible_devices"] != str(physical_gpu)
                or len(result["games"]) != 2 * PAIRS_PER_RULE):
            raise ValueError(f"existing match identity/protocol differs: {output}")
        summarize_paired_results(result["games"], bootstrap_samples=2000,
                                 bootstrap_seed=SEED)
        return result
    evaluated = play_paired_openings_replicated(
        openings,
        candidate_source=EvaluationModelSource("v3_artifact", a["snapshot"], name_a),
        incumbent_source=EvaluationModelSource("v3_artifact", b["snapshot"], name_b),
        search_sims=SIMS, cpuct=CPUCT,
        worker_devices=("cuda:0",) * WORKERS,
    )
    summary = summarize_paired_results(
        evaluated.games, bootstrap_samples=4000, bootstrap_seed=SEED + len(rule))
    result = {
        "schema": "connect4-stage2-fla2-r2-terminal-rule-match-v1",
        "created_at_utc": now(), "comparison": comparison, "rule_id": rule,
        "model_a": name_a, "model_b": name_b,
        "model_sha256": expected_hashes,
        "opening_sha256": sha256(openings_path),
        "search_sims": SIMS, "cpuct": CPUCT,
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime": evaluated.metrics.to_dict(),
        "games": [asdict(game) for game in evaluated.games],
        "summary": summary.to_dict(),
    }
    atomic_json(output, result)
    return result


def summarize(inputs: dict) -> None:
    report = {"schema": "connect4-stage2-fla2-r2-terminal-five-rule-report-v1",
              "created_at_utc": now(), "inputs_sha256": sha256(OUT / "inputs.json"),
              "comparisons": {}}
    for comparison, (name_a, name_b) in COMPARISONS.items():
        games = []
        rules = {}
        for rule in RULES:
            path = OUT / "matches" / comparison / f"{rule}.json"
            row = read(path)
            if (row["model_a"] != name_a or row["model_b"] != name_b
                    or row["model_sha256"] != {
                        "a": inputs["models"][name_a]["snapshot_sha256"],
                        "b": inputs["models"][name_b]["snapshot_sha256"],
                    }
                    or row["opening_sha256"] != sha256(OUT / "openings" / f"{rule}.json")
                    or row["search_sims"] != SIMS or row["cpuct"] != CPUCT
                    or row["physical_gpu"] != GPU_BY_COMPARISON[comparison]
                    or row["cuda_visible_devices"] != str(GPU_BY_COMPARISON[comparison])
                    or len(row["games"]) != 2 * PAIRS_PER_RULE):
                raise ValueError(f"incomplete rule match: {path}")
            checked = summarize_paired_results(
                row["games"], bootstrap_samples=4000, bootstrap_seed=SEED + len(rule))
            if checked.to_dict() != row["summary"]:
                raise ValueError(f"rule match summary drift: {path}")
            games.extend(row["games"])
            rules[rule] = {
                "match_sha256": sha256(path),
                "opening_sha256": row["opening_sha256"],
                "w_d_l": [row["summary"]["overall"][key]
                          for key in ("wins", "draws", "losses")],
                "score": row["summary"]["overall"]["point_score"],
                "ci95": [row["summary"]["ci_lower"], row["summary"]["ci_upper"]],
                "a_first": row["summary"]["candidate_as_first"],
                "a_second": row["summary"]["candidate_as_second"],
            }
        combined = summarize_paired_results(
            games, bootstrap_samples=10000, bootstrap_seed=SEED)
        report["comparisons"][comparison] = {
            "model_a": name_a, "model_b": name_b,
            "pairs": len(combined.pairs), "games": combined.overall.games,
            "w_d_l": [getattr(combined.overall, key)
                      for key in ("wins", "draws", "losses")],
            "macro_score": sum(row["score"] for row in rules.values()) / len(RULES),
            "pooled_ci95": [combined.ci_lower, combined.ci_upper],
            "a_first": combined.candidate_as_first.to_dict(),
            "a_second": combined.candidate_as_second.to_dict(),
            "rules": rules,
        }
    atomic_json(OUT / "report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("poll interval must be at least 30 seconds")
    if not args.execute:
        print(json.dumps({"comparisons": COMPARISONS, "rules": RULES,
                          "pairs_per_rule_per_comparison": PAIRS_PER_RULE,
                          "physical_gpu": PHYSICAL_GPU, "search_sims": SIMS}, indent=2))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU):
        raise RuntimeError("terminal matches require physical GPU0")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                statuses = [read(BASE / f"queue_row{i}.json")["status"]
                            for i in (0, 1)]
                if "failed" in statuses:
                    raise RuntimeError(f"FLA2-R2 training failed: {statuses}")
                if all(value == "complete" for value in statuses):
                    break
                if not args.watch:
                    raise RuntimeError(f"FLA2-R2 training incomplete: {statuses}")
                atomic_json(STATE, {"status": "waiting_training", "queues": statuses,
                                    "updated_at_utc": now()})
                time.sleep(args.poll_seconds)
            rows = verified_inputs()
            inputs = freeze_inputs(rows)
            for rule_index, rule in enumerate(RULES):
                openings_path, openings = rule_openings(rule, rule_index)
                for comparison in ("architecture",):
                    output = OUT / "matches" / comparison / f"{rule}.json"
                    if not output.exists():
                        while not gpu_idle():
                            if not args.watch:
                                raise RuntimeError("physical GPU0 occupied")
                            atomic_json(STATE, {"status": "waiting_gpu0",
                                                "next_comparison": comparison,
                                                "next_rule": rule,
                                                "updated_at_utc": now()})
                            time.sleep(args.poll_seconds)
                    atomic_json(STATE, {"status": "matching", "comparison": comparison,
                                        "rule": rule, "updated_at_utc": now()})
                    run_match(inputs, comparison, rule, openings_path, openings)
            pool_state_path = OUT / "pool_gpu1/watcher_state.json"
            while not pool_state_path.exists() or read(pool_state_path)["status"] != "complete":
                if pool_state_path.exists() and read(pool_state_path)["status"] == "failed":
                    raise RuntimeError("GPU1 pool comparison failed")
                if not args.watch:
                    raise RuntimeError("GPU1 pool comparison incomplete")
                atomic_json(STATE, {"status": "waiting_gpu1_pool_match",
                                    "updated_at_utc": now()})
                time.sleep(args.poll_seconds)
            summarize(inputs)
            atomic_json(STATE, {"status": "complete",
                                "report_sha256": sha256(OUT / "report.json"),
                                "updated_at_utc": now()})
        except Exception as exc:
            atomic_json(STATE, {"status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "updated_at_utc": now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
