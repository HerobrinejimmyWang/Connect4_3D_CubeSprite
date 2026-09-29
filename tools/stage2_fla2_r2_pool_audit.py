"""Audit FLA2-R2 replay coverage after the five-rule terminal matches.

The Stage 2 frozen pools are classic-only context. Learner sampling by rule is
not logged for these V3 runs, so this audit reports available window shares
and explicitly marks actual rule-wise learner exposure as unavailable.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_core.rules import BAL5_R2_RULE_REGISTRY  # noqa: E402
from stage2_fla2_r2_no_mixture_fork import RUN_DIR as CONTROL_RUN  # noqa: E402
from stage2_fla2_r2_terminal_matches import OUT as MATCH_OUT, RULES  # noqa: E402
from stage2_fla2_r2_watch import BASE, PLAN_PATH  # noqa: E402
from stage2_fla_bal5r1_fork4_2m import latest_commit, read  # noqa: E402
from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402
from training.v3.replay import stable_split_mask  # noqa: E402

OUT = BASE / "pool_quality_audit"
STATE = OUT / "watcher_state.json"
FROZEN = ROOT / "training/runs/stage2/pools/stage2_pools.json"
CODE_TO_RULE = {int(spec.rule_code): spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generation_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics/metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selfplay = [row for row in rows if row.get("stage") == "selfplay"]
    learner = [row for row in rows if row.get("stage") == "learner"]
    commits = [row for row in rows if row.get("stage") == "generation_commit"]
    if not selfplay or len(selfplay) != len(learner) or len(learner) != len(commits):
        raise ValueError(f"incomplete generation metrics: {run_dir}")
    by_rule = {rule: {"games": 0, "raw_positions": 0} for rule in RULES}
    route_games = Counter()
    route_raw = Counter()
    for row in selfplay:
        per_rule = row["actor_runtime"]["per_rule"]
        if set(per_rule) != set(RULES):
            raise ValueError("five-rule generation is incomplete")
        for rule, item in per_rule.items():
            by_rule[rule]["games"] += int(item["games"])
            by_rule[rule]["raw_positions"] += int(item["raw_positions"])
        mixture = row.get("health", {}).get("opening_temperature_mixture")
        if mixture:
            for route, item in mixture["variants"].items():
                route_games[route] += int(item["games"])
                route_raw[route] += int(item["raw_positions"])
        else:
            route_games["baseline"] += int(row["games"])
            route_raw["baseline"] += int(row["new_raw_positions"])
    for rule, item in by_rule.items():
        item["mean_raw_positions_per_game"] = item["raw_positions"] / item["games"]
    total_raw = sum(item["raw_positions"] for item in by_rule.values())
    if total_raw != int(commits[-1]["cumulative_raw_positions"]):
        raise ValueError("cumulative rule positions differ from last commit metric")
    route_sampled = Counter()
    for row in learner:
        route_sampled.update(row.get("sampling_group_positions") or {})
    return {
        "generations": len(commits), "games": sum(item["games"] for item in by_rule.values()),
        "raw_positions": total_raw, "train_positions_consumed": int(commits[-1]["train_positions_consumed"]),
        "train_draws_per_raw_position": int(commits[-1]["train_positions_consumed"]) / total_raw,
        "per_rule": by_rule,
        "temperature_routes": {key: {"games": route_games[key],
                                     "raw_positions": route_raw[key],
                                     "learner_sampled_positions": route_sampled[key]}
                               for key in sorted(route_games | route_sampled)},
        "learner_rule_sample_counts": None,
        "learner_rule_sample_note": "not recorded in historical learner metrics",
        "metrics_sha256": sha256(path),
    }


def active_window(run_dir: Path, mixture_start_game_id: int | None) -> dict:
    selection_path = sorted((run_dir / "replay/shuffle").glob("selection_g*.json"))[-1]
    selection = read(selection_path)
    start, stop = int(selection["window_start"]), int(selection["window_end"])
    validation_fraction = float(selection["validation_fraction"])
    split_seed = int(selection["split_seed"])
    counts = defaultdict(Counter)
    games = defaultdict(lambda: defaultdict(set))
    unique = defaultdict(set)
    observed = 0
    for shard in selection["input_shards"]:
        lo, hi = int(shard["position_start"]), int(shard["position_end"])
        if hi <= start or lo >= stop:
            continue
        path = run_dir / shard["path"]
        if sha256(path) != shard["checksum_sha256"]:
            raise ValueError(f"replay checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            if len(data["rule_code"]) != hi - lo:
                raise ValueError(f"replay length mismatch: {path}")
            sl = slice(max(start, lo) - lo, min(stop, hi) - lo)
            boards = data["board"][sl]
            codes = data["rule_code"][sl]
            players = data["player_to_move"][sl]
            game_ids = data["game_id"][sl]
            train = stable_split_mask(game_ids, split="train",
                                      validation_fraction=validation_fraction,
                                      split_seed=split_seed)
            observed += len(codes)
            for board, code, player, game_id, is_train in zip(
                    boards, codes, players, game_ids, train, strict=True):
                rule = CODE_TO_RULE[int(code)]
                route = ("lowered_opening_temperature" if mixture_start_game_id is not None
                         and int(game_id) >= mixture_start_game_id and int(game_id) % 2
                         else "baseline")
                split = "train" if is_train else "validation"
                counts[rule]["window"] += 1
                counts[rule][split] += 1
                counts[rule][route] += 1
                games[rule][split].add(int(game_id))
                key = hashlib.blake2b(
                    board.tobytes() + int(code).to_bytes(2, "little")
                    + int(player).to_bytes(1, "little", signed=True),
                    digest_size=16, person=b"fla2-pool-state",
                ).digest()
                unique[rule].add(key)
    if observed != int(selection["window_positions"]):
        raise ValueError("reconstructed active window size differs from selection")
    per_rule = {}
    for rule in RULES:
        n = counts[rule]["window"]
        per_rule[rule] = {
            "positions": n,
            "window_share": n / observed if observed else 0.0,
            "train_positions_available": counts[rule]["train"],
            "validation_positions": counts[rule]["validation"],
            "train_games": len(games[rule]["train"]),
            "validation_games": len(games[rule]["validation"]),
            "baseline_positions": counts[rule]["baseline"],
            "lowered_opening_temperature_positions": counts[rule]["lowered_opening_temperature"],
            "unique_board_rule_player_states": len(unique[rule]),
            "duplicate_position_fraction": 1.0 - len(unique[rule]) / n if n else 0.0,
        }
    if sum(item["train_positions_available"] for item in per_rule.values()) != int(selection["train_positions"]):
        raise ValueError("train split does not reconstruct selection manifest")
    if sum(item["validation_positions"] for item in per_rule.values()) != int(selection["validation_positions"]):
        raise ValueError("validation split does not reconstruct selection manifest")
    return {
        "selection_sha256": sha256(selection_path),
        "window_positions": observed,
        "train_positions_available": int(selection["train_positions"]),
        "validation_positions": int(selection["validation_positions"]),
        "train_sample_id_digest": selection["train_sample_id_digest"],
        "validation_sample_id_digest": selection["validation_sample_id_digest"],
        "position_balanced_sampling": selection.get("position_balanced_sampling"),
        "per_rule": per_rule,
    }


def frozen_context() -> dict:
    pool = read(FROZEN)
    if pool.get("schema") != "connect4-v3-stage2-frozen-pools-v2":
        raise ValueError("Stage 2 frozen pool schema differs")
    regimes = {}
    for name in ("standard_late", "mixed_late"):
        row = pool["regimes"][name]
        generations = sorted({int(match.group(1))
                              for path in row["source_shards"]
                              if (match := re.search(r"/g(\d+)_", path))})
        regimes[name] = {
            "recipe": row["data_recipe_id"],
            "source_shards": len(row["source_shards"]),
            "source_generation_count": len(generations),
            "source_generation_min": min(generations),
            "source_generation_max": max(generations),
            "frozen_train_positions": row["train"]["positions"],
            "frozen_validation_positions": row["validation"]["positions"],
            "frozen_train_sha256": row["train"]["sha256"],
        }
    return {"manifest_sha256": sha256(FROZEN), "regimes": regimes,
            "interpretation": "classic-only frozen samples from long source trajectories; not 15M distinct positions"}


def audit() -> None:
    plan = read(PLAN_PATH)
    run_dirs = {row["name"]: Path(row["run_dir"]) for row in plan["rows"]}
    run_dirs["raw3d_to2d_b8_no_mixture_g16"] = CONTROL_RUN
    runs = {}
    for name, run_dir in run_dirs.items():
        manifest_path = run_dir / "run_manifest.json"
        manifest = read(manifest_path)
        expected = 1_000_000 if name.endswith("no_mixture_g16") else 2_000_000
        if (manifest.get("stop_reason") != "max_train_positions"
                or int(manifest["formal_loop_state"]["train_positions_consumed"]) != expected):
            raise ValueError(f"run not at verified endpoint: {name}")
        pointer, commit = latest_commit(run_dir)
        start_id = manifest["formal_loop_state"].get("opening_temperature_mixture_start_game_id")
        runs[name] = {
            "run_manifest_sha256": sha256(manifest_path),
            "generation_commit_sha256": pointer["commit_sha256"],
            "checkpoint_sha256": commit["checkpoint_sha256"],
            "accepted_sha256": commit["accepted_model_sha256"],
            "mixture_start_game_id": start_id,
            "generated": generation_metrics(run_dir),
            "active_window": active_window(run_dir, start_id),
        }
    atomic_json(OUT / "report.json", {
        "schema": "connect4-stage2-fla2-r2-pool-quality-audit-v1",
        "created_at_utc": now(),
        "match_report_sha256": sha256(MATCH_OUT / "report.json"),
        "frozen_stage2_context": frozen_context(),
        "runs": runs,
        "limitations": [
            "Actual learner samples by rule were not logged; active train availability is not sampled exposure.",
            "The frozen Stage 2 pools are classic-only and are context, not a five-rule quality baseline.",
            "Two million optimizer positions may revisit a much smaller active replay window.",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("poll interval must be at least 30 seconds")
    if not args.execute:
        print(json.dumps({"wait_for": str(MATCH_OUT / "report.json"),
                          "runs": 3, "requires_gpu": False}, indent=2))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                match_state = read(MATCH_OUT / "watcher_state.json")
                if match_state["status"] == "failed":
                    raise RuntimeError("terminal match watcher failed")
                if match_state["status"] == "complete":
                    if sha256(MATCH_OUT / "report.json") != match_state["report_sha256"]:
                        raise ValueError("terminal report hash differs")
                    break
                if not args.watch:
                    raise RuntimeError("terminal matches are incomplete")
                atomic_json(STATE, {"status": "waiting_terminal_matches",
                                    "updated_at_utc": now()})
                time.sleep(args.poll_seconds)
            atomic_json(STATE, {"status": "auditing", "updated_at_utc": now()})
            audit()
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
