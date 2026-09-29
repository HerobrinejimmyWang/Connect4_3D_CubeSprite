"""Run the ready same-architecture five-rule pool comparison on physical GPU1.

The independent GPU0 architecture comparison remains gated on thin B8's 2M
commit. This runner writes only the pool edge and reuses its frozen openings.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import stage2_fla2_r2_terminal_matches as match
from stage2_fla_selfplay_queue import atomic_json, sha256
from stage2_fla_bal5r1_fork4_2m import read
from training.v3.gate import summarize_paired_results

OUT = match.OUT / "pool_gpu1"
STATE = OUT / "watcher_state.json"


def freeze_inputs(rows: dict[str, dict]) -> dict:
    match.snapshot_rows(rows)
    pair = list(match.COMPARISONS["pool"])
    payload = {
        "schema": "connect4-stage2-fla2-r2-pool-gpu1-inputs-v1",
        "models": rows, "comparisons": {"pool": pair},
        "rules": list(match.RULES),
        "rule_registry_hash": match.BAL5_R2_RULE_REGISTRY.registry_hash,
        "pairs_per_rule": match.PAIRS_PER_RULE,
        "search_sims": match.SIMS, "cpuct": match.CPUCT,
        "physical_gpu": 1, "workers": match.WORKERS,
        "endpoint": "terminal_checkpoint_weights",
        "caveat": "same architecture, separate V3 optimizer and replay histories",
    }
    path = OUT / "inputs.json"
    if path.exists():
        if read(path) != payload:
            raise ValueError("frozen GPU1 pool inputs changed")
    else:
        atomic_json(path, payload)
    return payload


def report(inputs: dict) -> None:
    games = []
    per_rule = {}
    name_a, name_b = inputs["comparisons"]["pool"]
    for rule in match.RULES:
        path = match.OUT / "matches/pool" / f"{rule}.json"
        row = read(path)
        opening_path = match.OUT / "openings" / f"{rule}.json"
        if (row["model_a"] != name_a or row["model_b"] != name_b
                or row["model_sha256"] != {
                    "a": inputs["models"][name_a]["snapshot_sha256"],
                    "b": inputs["models"][name_b]["snapshot_sha256"],
                }
                or row["opening_sha256"] != sha256(opening_path)
                or row["physical_gpu"] != 1
                or row["cuda_visible_devices"] != "1"
                or len(row["games"]) != 2 * match.PAIRS_PER_RULE):
            raise ValueError(f"pool match identity differs: {path}")
        checked = summarize_paired_results(
            row["games"], bootstrap_samples=4000,
            bootstrap_seed=match.SEED + len(rule))
        if checked.to_dict() != row["summary"]:
            raise ValueError(f"pool match summary drift: {path}")
        games.extend(row["games"])
        per_rule[rule] = {"match_sha256": sha256(path),
                          "score": checked.overall.point_score,
                          "ci95": [checked.ci_lower, checked.ci_upper],
                          "w_d_l": [checked.overall.wins, checked.overall.draws,
                                    checked.overall.losses]}
    combined = summarize_paired_results(
        games, bootstrap_samples=10000, bootstrap_seed=match.SEED)
    atomic_json(OUT / "report.json", {
        "schema": "connect4-stage2-fla2-r2-pool-gpu1-report-v1",
        "created_at_utc": match.now(), "inputs_sha256": sha256(OUT / "inputs.json"),
        "physical_gpu": 1, "pairs": len(combined.pairs),
        "games": combined.overall.games,
        "w_d_l": [combined.overall.wins, combined.overall.draws,
                  combined.overall.losses],
        "macro_score": sum(row["score"] for row in per_rule.values()) / len(per_rule),
        "pooled_ci95": [combined.ci_lower, combined.ci_upper],
        "per_rule": per_rule,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("poll interval must be at least 30 seconds")
    if not args.execute:
        print(json.dumps({"comparison": "pool", "physical_gpu": 1,
                          "rules": match.RULES,
                          "pairs_per_rule": match.PAIRS_PER_RULE,
                          "search_sims": match.SIMS}, indent=2))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("pool comparison requires physical GPU1")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rows = match.verified_inputs(require_thin=False)
            inputs = freeze_inputs(rows)
            for index, rule in enumerate(match.RULES):
                opening_path, openings = match.rule_openings(rule, index)
                output = match.OUT / "matches/pool" / f"{rule}.json"
                if not output.exists():
                    while not match.gpu_idle(1):
                        if not args.watch:
                            raise RuntimeError("physical GPU1 occupied")
                        atomic_json(STATE, {"status": "waiting_gpu1",
                                            "next_rule": rule,
                                            "updated_at_utc": match.now()})
                        time.sleep(args.poll_seconds)
                atomic_json(STATE, {"status": "matching", "rule": rule,
                                    "updated_at_utc": match.now()})
                match.run_match(inputs, "pool", rule, opening_path, openings,
                                physical_gpu=1)
            report(inputs)
            atomic_json(STATE, {"status": "complete",
                                "report_sha256": sha256(OUT / "report.json"),
                                "updated_at_utc": match.now()})
        except Exception as exc:
            atomic_json(STATE, {"status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "updated_at_utc": match.now()})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
