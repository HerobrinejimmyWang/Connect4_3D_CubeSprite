"""Remeasure four FLA terminals between identical archived-gravity controls.

The old control must return near its original same-protocol 512-simulation
latency both before and after the four models. Otherwise publish no new
selection evidence. Earlier measurements remain immutable.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_stage2_cpu_latency import run_one
from stage2_fla_cpu_screen import SOURCE as ARCHIVED
from stage2_fla_post3m_cpu import NAMES, OUTPUT as ORIGINAL, SIMS, REPEATS, IDLE, GROUP_IDLE
from stage2_fla_selfplay_queue import ROOT, atomic_json, sha256


OUT = ROOT / "training/runs/stage2/fla/cpu_terminal_3m/recheck_bracket_20260927"
CONTROL_NAME = "balance_gravity_control"
CONTROL_CONFIG = ARCHIVED / "configs" / f"{CONTROL_NAME}__standard_late__seed271828.json"
CONTROL_MODEL = ARCHIVED / "runs" / CONTROL_NAME / "standard_late/seed271828/model.pt"
CONTROL_OLD = ROOT / ("training/runs/stage2/fla/cpu_boundary_6m/idle10_group60/"
                      f"{CONTROL_NAME}_512.json")
MAX_CONTROL_RATIO = 1.15
MAX_BRACKET_RATIO = 1.10


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_group(output: Path, name: str, model: Path, config: Path,
              artifact_kind: str, filename: str) -> dict:
    target = output / filename
    if target.exists():
        raise FileExistsError(f"existing bracket group must be inspected: {target}")
    atomic_json(output / "watcher_state.json", {
        "status": "measuring", "active": name, "updated_at_utc": now()})
    result = run_one(name, model, config, SIMS, repeats=REPEATS,
                     idle_s=IDLE, artifact_kind=artifact_kind)
    result["metadata"]["group_idle_s"] = GROUP_IDLE
    atomic_json(target, result)
    stats = result["summary"]["excluding_shortcuts"]
    if stats["count"] != 45:
        raise ValueError(f"group has {stats['count']} searched measurements: {name}")
    print(f"{name}: {stats['mean_s']:.3f}s searched mean", flush=True)
    return {"path": str(target), "sha256": sha256(target),
            "mean_s": stats["mean_s"], "median_s": stats["median_s"],
            "p90_s": stats["p90_s"], "p95_s": stats["p95_s"],
            "count": stats["count"],
            "artifact_sha256": result["metadata"]["artifact_sha256"],
            "config_sha256": result["metadata"]["config_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--variants", nargs="+", choices=NAMES, default=NAMES)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    variants = tuple(args.variants)
    if len(set(variants)) != len(variants):
        parser.error("duplicate variants are not allowed")
    if not args.execute:
        print(json.dumps({"output": str(output), "control": CONTROL_NAME,
                          "control_max_ratio": MAX_CONTROL_RATIO,
                          "bracket_max_ratio": MAX_BRACKET_RATIO,
                          "models": variants, "simulations": SIMS,
                          "repeats": REPEATS, "idle_s": IDLE,
                          "group_idle_s": GROUP_IDLE}, indent=2))
        return
    if (output / "watcher_state.json").exists():
        raise FileExistsError("existing bracket run state must be inspected")
    output.mkdir(parents=True, exist_ok=True)
    old = read(CONTROL_OLD)
    expected_control = old["summary"]["excluding_shortcuts"]["mean_s"]
    if (old["metadata"]["artifact_sha256"] != sha256(CONTROL_MODEL)
            or old["metadata"]["config_sha256"] != sha256(CONTROL_CONFIG)
            or old["metadata"]["mcts_sims"] != SIMS
            or old["metadata"]["repeats"] != REPEATS
            or old["metadata"]["idle_s"] != IDLE):
        raise ValueError("archived gravity control identity/protocol differs")
    inputs = read(ORIGINAL / "inputs_remote.json")
    if inputs["schema"] != "connect4-stage2-fla-terminal-four-v1":
        raise ValueError("terminal input manifest differs")
    try:
        before = run_group(output, CONTROL_NAME, CONTROL_MODEL, CONTROL_CONFIG,
                           "offline", "control_before_512.json")
        atomic_json(output / "control_before_state.json", before)
        if not before["mean_s"] <= MAX_CONTROL_RATIO * expected_control:
            atomic_json(output / "watcher_state.json", {
                "status": "blocked_unstable_cpu_baseline",
                "old_control_mean_s": expected_control,
                "new_control_mean_s": before["mean_s"],
                "updated_at_utc": now()})
            return
        results = {}
        for name in variants:
            time.sleep(GROUP_IDLE)
            row = inputs["models"][name]
            model = ORIGINAL / "inputs" / f"{name}.pt"
            config = ORIGINAL / "inputs" / f"{name}.json"
            if (sha256(model) != row["snapshot_sha256"]
                    or sha256(config) != row["config_sha256"]):
                raise ValueError(f"terminal input hashes differ: {name}")
            results[name] = run_group(output, name, model, config,
                                      "formal_v3_snapshot", f"{name}_512.json")
            if (results[name]["artifact_sha256"] != row["snapshot_sha256"]
                    or results[name]["config_sha256"] != row["config_sha256"]):
                raise ValueError(f"measured terminal identity differs: {name}")
        time.sleep(GROUP_IDLE)
        after = run_group(output, CONTROL_NAME, CONTROL_MODEL, CONTROL_CONFIG,
                          "offline", "control_after_512.json")
        controls_stable = (
            after["mean_s"] <= MAX_CONTROL_RATIO * expected_control
            and max(before["mean_s"], after["mean_s"]) /
            min(before["mean_s"], after["mean_s"]) <= MAX_BRACKET_RATIO
        )
        summary = {
            "schema": "connect4-stage2-fla-terminal-cpu-bracket-v1",
            "created_at_utc": now(),
            "status": "stable_control_pass" if controls_stable else "unstable_control_no_selection",
            "protocol": {"simulations": SIMS, "repeats": REPEATS,
                         "idle_s": IDLE, "group_idle_s": GROUP_IDLE,
                         "searched_measurements_per_model": 45},
            "original_control_sha256": sha256(CONTROL_OLD),
            "old_control_mean_s": expected_control,
            "max_control_ratio": MAX_CONTROL_RATIO,
            "max_bracket_ratio": MAX_BRACKET_RATIO,
            "terminal_inputs_sha256": sha256(ORIGINAL / "inputs_remote.json"),
            "variants": variants,
            "control_before": before, "results": results,
            "control_after": after,
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "watcher_state.json", {
            "status": summary["status"],
            "summary_sha256": sha256(output / "summary.json"),
            "updated_at_utc": now()})
    except Exception as exc:
        atomic_json(output / "watcher_state.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "updated_at_utc": now()})
        raise


if __name__ == "__main__":
    main()
