"""Compare 256 and 512 V3 searches on the same accepted five-rule model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_core.rules import BAL5_R2_RULE_REGISTRY, RuleEngine, TurnAction
from training.v3.evaluation import build_openings, load_opening_manifest
from training.v3.evaluation_runtime import (
    EvaluationModelSource, _load_replicated_predictor,
    play_paired_openings_replicated,
)
from training.v3.gate import GateGameResult, summarize_paired_results
from training.v3.search import MCTS


BASE = ROOT / "training/runs/stage2/fla2_r2"
RUN = BASE / "runs/fla2_r2_raw3d_to2d_b8_five_rule_seed271828"
COMMIT = RUN / "manifests/generations/g000032.json"
EXPECTED_ACCEPTED_SHA = "b969a7c53f0f7197c99f2c27e81d7830f39e93426ab913d33baecb807117d305"
OUT = BASE / "search_budget_256_vs_512_accepted_g31"
RULES = tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"frozen result differs: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def accepted_input() -> tuple[Path, dict]:
    commit = json.loads(COMMIT.read_text(encoding="utf-8"))
    accepted = RUN / commit["accepted_model_path"]
    if commit["accepted_model_sha256"] != EXPECTED_ACCEPTED_SHA or sha256(accepted) != EXPECTED_ACCEPTED_SHA:
        raise ValueError("accepted model SHA-256 differs from committed g32 identity")
    return accepted, commit


def state_for_opening(engine: RuleEngine, columns: tuple[int, ...]):
    state = engine.initial_state()
    for column in columns:
        while (required := engine.required_action(state)) is not None:
            state = engine.step(state, required)
        state = engine.step(state, TurnAction.place(column))
    while (required := engine.required_action(state)) is not None:
        state = engine.step(state, required)
    if state.terminal:
        raise ValueError("diagnostic opening ended the game")
    return state


def positions(accepted: Path, commit: dict, count: int) -> None:
    predictor = _load_replicated_predictor(
        EvaluationModelSource("v3_artifact", str(accepted), commit["accepted_model_id"]), "cuda:0"
    )
    rows = []
    for rule_index, rule in enumerate(RULES):
        engine = RuleEngine(BAL5_R2_RULE_REGISTRY.get(rule), registry=BAL5_R2_RULE_REGISTRY)
        openings = build_openings(
            count, run_seed=20260929 + 1013 * rule_index, rule_id=rule,
            prefix_lengths=(0, 8, 16, 24), registry=BAL5_R2_RULE_REGISTRY,
            opening_id_prefix=f"budget-{rule}",
        )
        for opening in openings:
            state = state_for_opening(engine, opening.columns)
            results = {}
            for sims in (256, 512):
                search = MCTS(predictor, engine=engine, cpuct=1.5, num_threads=1)
                result = search.search(state, sims, rng=np.random.default_rng(opening.seed), add_root_noise=False)
                results[str(sims)] = {
                    "top_action": int(np.argmax(result.visit_counts)),
                    "top_visits": int(result.visit_counts.max()),
                    "root_value": result.root_value,
                    "inference_calls": result.inference_calls,
                    "visits": result.visit_counts.tolist(),
                }
            rows.append({"opening": opening.to_dict(), "placement_count": state.placement_count,
                         "same_top_action": results["256"]["top_action"] == results["512"]["top_action"],
                         "results": results})
        write_once(OUT / "positions" / f"{rule}.json", {
            "schema": "stage2-fla-accepted-search-budget-positions-v1",
            "accepted_sha256": EXPECTED_ACCEPTED_SHA,
            "rule_id": rule,
            "positions": [row for row in rows if row["opening"]["rule_id"] == rule],
        })
    write_once(OUT / "positions_summary.json", {
        "schema": "stage2-fla-accepted-search-budget-summary-v1",
        "accepted_sha256": EXPECTED_ACCEPTED_SHA,
        "count": len(rows),
        "same_top_action": sum(row["same_top_action"] for row in rows),
        "by_rule": {rule: {"count": sum(row["opening"]["rule_id"] == rule for row in rows),
                            "same_top_action": sum(row["same_top_action"] for row in rows if row["opening"]["rule_id"] == rule)}
                    for rule in RULES},
    })


def matches(accepted: Path, commit: dict, pairs: int) -> None:
    source = EvaluationModelSource("v3_artifact", str(accepted), commit["accepted_model_id"])
    all_games = []
    for rule in RULES:
        opening_path = BASE / "direct_terminal_5rule_100pairs/openings" / f"{rule}.json"
        openings = load_opening_manifest(opening_path, registry=BAL5_R2_RULE_REGISTRY)[:pairs]
        if len(openings) != pairs:
            raise ValueError(f"insufficient frozen openings for {rule}")
        result_path = OUT / "matches" / f"{rule}_{pairs}pairs.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            evaluated = play_paired_openings_replicated(
                openings, candidate_source=source, incumbent_source=source,
                search_sims=256, candidate_search_sims=512, incumbent_search_sims=256,
                cpuct=1.5, worker_devices=("cuda:0",) * 4 + ("cuda:1",) * 4,
            )
            summary = summarize_paired_results(evaluated.games, bootstrap_samples=4000,
                                               bootstrap_seed=20260929 + len(rule))
            payload = {
                "schema": "stage2-fla-accepted-search-budget-match-v1",
                "rule_id": rule, "accepted_sha256": EXPECTED_ACCEPTED_SHA,
                "opening_sha256": sha256(opening_path), "pairs": pairs,
                "candidate_sims": 512, "incumbent_sims": 256,
                "runtime": evaluated.metrics.to_dict(),
                "games": [asdict(game) for game in evaluated.games],
                "summary": summary.to_dict(),
            }
            write_once(result_path, payload)
        all_games.extend(GateGameResult(**game) for game in payload["games"])
    summary = summarize_paired_results(all_games, bootstrap_samples=10000,
                                       bootstrap_seed=20260929)
    write_once(OUT / f"match_summary_{pairs}pairs.json", {
        "schema": "stage2-fla-accepted-search-budget-match-summary-v1",
        "accepted_sha256": EXPECTED_ACCEPTED_SHA,
        "pairs_per_rule": pairs, "games": len(all_games),
        "candidate_sims": 512, "incumbent_sims": 256,
        "summary": summary.to_dict(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("positions", "matches"))
    parser.add_argument("--positions-per-rule", type=int, default=10)
    parser.add_argument("--pairs-per-rule", type=int, default=50)
    args = parser.parse_args()
    accepted_path, accepted_commit = accepted_input()
    if args.phase == "positions":
        positions(accepted_path, accepted_commit, args.positions_per_rule)
    else:
        matches(accepted_path, accepted_commit, args.pairs_per_rule)
