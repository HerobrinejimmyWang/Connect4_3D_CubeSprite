"""Launch a guarded two-GPU V3 raw B8 pool fork after search-budget evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_core.rules import BAL5_R2_RULE_REGISTRY
from training.v3.config import V3Config, config_hash

BASE = ROOT / "training/runs/stage2/fla2_r2"
PARENT = BASE / "runs/fla2_r2_raw3d_to2d_b8_five_rule_seed271828"
PARENT_CONFIG = BASE / "configs/fla2_r2_raw3d_to2d_b8_five_rule_seed271828.json"
PARENT_RECEIPT = BASE / "receipts/raw3d_to2d_b8_to2m.json"
MATCH = BASE / "search_budget_256_vs_512_accepted_g31/match_summary_50pairs.json"
CHILD_BASE = ROOT / "training/runs/stage2/fla3_pre/raw_b8_pool_10m"
RUN_ID = "fla3_pre_raw_b8_g31_pool10m_seed271829"
RUN = CHILD_BASE / "runs" / RUN_ID
CONFIG = CHILD_BASE / "configs" / f"{RUN_ID}.json"
PLAN = CHILD_BASE / "plan.json"
STATE = CHILD_BASE / "watcher_state.json"
TARGET = 10_000_000
CANARY = 80_000
FIRST_GATE = 320_000
EXPECTED_ACCEPTED = "b969a7c53f0f7197c99f2c27e81d7830f39e93426ab913d33baecb807117d305"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    return not result.stdout.strip()


def prepare() -> dict:
    if PLAN.exists() or CONFIG.exists() or RUN.exists():
        raise FileExistsError("new pool fork already prepared; inspect before retry")
    parent_receipt = read(PARENT_RECEIPT)
    parent_commit_path = PARENT / "manifests/generations/g000032.json"
    parent_commit = read(parent_commit_path)
    accepted = PARENT / parent_commit["accepted_model_path"]
    if (parent_receipt["positions"] != 2_000_000
            or parent_receipt["accepted_sha256"] != EXPECTED_ACCEPTED
            or parent_commit["accepted_model_sha256"] != EXPECTED_ACCEPTED
            or sha(accepted) != EXPECTED_ACCEPTED
            or sha(parent_commit_path) != parent_receipt["generation_commit_sha256"]
            or sha(PARENT_CONFIG) != parent_receipt["config_sha256"]):
        raise ValueError("parent 2M accepted/commit/config receipt verification failed")
    result = read(MATCH)
    if (result["accepted_sha256"] != EXPECTED_ACCEPTED
            or result["candidate_sims"] != 512
            or result["incumbent_sims"] != 256
            or result["pairs_per_rule"] != 50
            or result["games"] != 500):
        raise ValueError("search-budget match is incomplete or uses another model")
    # Bootstrap opening pairs within each rule, then average the five rule means.
    # A pooled bootstrap would vary the rule mix despite the fixed 1:1:1:1:1 design.
    rng = np.random.default_rng(20260929)
    rule_samples = []
    per_rule = {}
    for spec in BAL5_R2_RULE_REGISTRY.specs:
        rule = spec.rule_id
        rule_result = read(MATCH.parent / "matches" / f"{rule}_50pairs.json")
        if (rule_result["accepted_sha256"] != EXPECTED_ACCEPTED
                or rule_result["rule_id"] != rule
                or rule_result["pairs"] != 50
                or len(rule_result["games"]) != 100):
            raise ValueError(f"incomplete search-budget rule evidence: {rule}")
        scores = np.asarray([row["pair_score"] for row in rule_result["summary"]["pairs"]], dtype=float)
        if len(scores) != 50:
            raise ValueError(f"incomplete paired opening scores: {rule}")
        samples = scores[rng.integers(0, len(scores), size=(10000, len(scores)))].mean(axis=1)
        per_rule[rule] = {
            "point_score": float(scores.mean()),
            "ci_lower": float(np.quantile(samples, 0.025)),
            "ci_upper": float(np.quantile(samples, 0.975)),
            "match_sha256": sha(MATCH.parent / "matches" / f"{rule}_50pairs.json"),
        }
        rule_samples.append(samples)
    score = float(np.mean([item["point_score"] for item in per_rule.values()]))
    stratified = np.mean(np.stack(rule_samples), axis=0)
    ci_lower, ci_upper = map(float, np.quantile(stratified, (0.025, 0.975)))
    if abs(score - result["summary"]["overall"]["point_score"]) > 1e-12:
        raise ValueError("aggregate match score differs from rule-stratified score")
    high_sims = 512 if (score > 0.5 and ci_lower > 0.5
                        and all(item["ci_upper"] >= 0.5 for item in per_rule.values())) else 256
    raw = read(PARENT_CONFIG)
    raw["run"].update({
        "run_id": RUN_ID, "run_dir": str(RUN), "resume": False,
        "seed": 271829, "warm_start_checkpoint": str(accepted),
        "warm_start_checkpoint_sha256": EXPECTED_ACCEPTED,
        "warm_start_mode": "accepted_artifact_fresh_optimizer_replay_v1",
    })
    raw["gate"]["multirule_evaluation_mode"] = "incumbent_first"
    raw["runtime"].update({
        "actor_processes": 24,
        "selfplay_devices": ["cuda:0", "cuda:1"],
        "evaluation_devices": ["cuda:0", "cuda:1"],
        "evaluation_replicas_per_device": 4,
    })
    raw["selfplay"]["search_schedule"][0]["full_search_sims"] = high_sims
    raw["selfplay"]["opening_temperature_mixture"]["start_train_positions"] = 0
    # Warm-start at the parent's low learning rate, then decay for the long pool run.
    raw["learner"]["lr_schedule"] = [
        {"start_train_positions": 0, "learning_rate": 5e-5},
        {"start_train_positions": 5_000_000, "learning_rate": 2.5e-5},
    ]
    config = V3Config.from_dict(raw)
    if (config_hash(config) == parent_receipt["lineage_hash"]
            or config.gate.multirule_evaluation_mode != "incumbent_first"
            or tuple(config.selfplay.multi_rule_ids) != tuple(
                spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs
            )
            or config.selfplay.search_schedule[0].games != 800
            or config.runtime.mcts_lanes_per_actor != 4):
        raise ValueError("new semantic fork does not satisfy five-rule V3 contract")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(config.to_json(), encoding="utf-8")
    plan = {
        "schema": "stage2-fla-raw-b8-pool-10m-fork-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run": str(PARENT), "parent_positions": 2_000_000,
        "parent_receipt_sha256": sha(PARENT_RECEIPT),
        "parent_commit_sha256": sha(parent_commit_path),
        "warm_start_accepted": str(accepted), "warm_start_sha256": EXPECTED_ACCEPTED,
        "child_run": str(RUN), "child_config": str(CONFIG),
        "child_config_sha256": sha(CONFIG), "child_lineage_hash": config_hash(config),
        "match_summary_sha256": sha(MATCH), "match_point_score_512": score,
        "match_ci_lower_512": ci_lower, "match_ci_upper_512": ci_upper,
        "match_bootstrap": "10000 resamples within each rule then equal-rule average",
        "match_by_rule": per_rule, "high_search_sims": high_sims,
        "fast_search_sims": 32, "gate_search_sims": 256,
        "actor_processes": 24, "mcts_lanes_per_actor": 4,
        "selfplay_devices": ["cuda:0", "cuda:1"],
        "canary_positions": CANARY, "first_gate_positions": FIRST_GATE,
        "target_child_positions": TARGET,
        "lineage_note": "new semantic V3 fork with fresh optimizer and replay; parent 2M is separate",
    }
    atomic(PLAN, plan)
    return plan


def verify_boundary(bound: int, *, require_gate: bool, allow_drained: bool = False) -> dict:
    plan = read(PLAN)
    if sha(CONFIG) != plan["child_config_sha256"]:
        raise ValueError("child config SHA changed")
    manifest = read(RUN / "run_manifest.json")
    stop_reason = manifest["stop_reason"]
    valid_stop = (stop_reason == "max_train_positions"
                  or (allow_drained and str(stop_reason).startswith("drained_after_signal_")))
    if (manifest["formal_loop_state"]["train_positions_consumed"] != bound
            or manifest["config_hash"] != plan["child_lineage_hash"]
            or not valid_stop):
        raise ValueError("child run did not stop at the requested position boundary")
    pointer = read(RUN / "manifests/latest_generation.json")
    commit_path = RUN / pointer["commit"]
    commit = read(commit_path)
    if sha(commit_path) != pointer["commit_sha256"]:
        raise ValueError("generation commit SHA differs from latest pointer")
    checkpoint = RUN / commit["checkpoint"]
    accepted = RUN / commit["accepted_model_path"]
    if (sha(checkpoint) != commit["checkpoint_sha256"]
            or sha(accepted) != commit["accepted_model_sha256"]
            or not commit["replay_shards"]):
        raise ValueError("checkpoint, accepted, or replay commit evidence incomplete")
    observed_rule_codes: set[int] = set()
    for shard in commit["replay_shards"]:
        shard_path = RUN / shard["path"]
        if sha(shard_path) != shard["checksum_sha256"]:
            raise ValueError(f"replay shard SHA differs: {shard_path}")
        if bound == CANARY:
            with np.load(shard_path, allow_pickle=False) as data:
                observed_rule_codes.update(map(int, np.unique(data["rule_code"])))
    if bound == CANARY and len(observed_rule_codes) != 5:
        raise ValueError("canary replay does not cover all five rules")
    metrics = [json.loads(line) for line in (RUN / "metrics/metrics.jsonl").read_text().splitlines()]
    selfplays = [row for row in metrics if row.get("stage") == "selfplay"]
    learners = [row for row in metrics if row.get("stage") == "learner"]
    if not selfplays or not learners:
        raise ValueError("missing self-play or learner metrics")
    for row in selfplays:
        per_rule = row["actor_runtime"]["per_rule"]
        if (row["games"] != 800 or set(per_rule) != set(
            spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs
        ) or any(item["games"] != 160 for item in per_rule.values())):
            raise ValueError("five-rule 160 games/rule generation contract differs")
    gate_evidence = None
    if require_gate:
        first_gate_commit = read(RUN / "manifests/generations/g000004.json")
        gate_path = RUN / first_gate_commit["gate_path"]
        gate = read(gate_path)
        if (first_gate_commit["gate_verdict"] != "accept"
                or first_gate_commit["gate_sha256"] != sha(gate_path)
                or first_gate_commit["candidate_model_id"] != first_gate_commit["accepted_model_id"]
                or sha(RUN / first_gate_commit["accepted_model_path"]) != first_gate_commit["accepted_model_sha256"]
                or gate["verdict"] != "accept"
                or gate.get("multirule_evaluation_mode") != "incumbent_first"
                or gate.get("peak_evidence_status") != "complete"):
            raise ValueError("first optimized gate is missing committed acceptance evidence")
        rules = {spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs}
        pairs = gate["opening_pairs_per_rule"]
        if (set(gate["games_by_rule"]) != rules
                or set(gate["peak_games_by_rule"]) != rules
                or pairs < 50
                or any(len(gate["games_by_rule"][rule]) != 2 * pairs
                       or len(gate["peak_games_by_rule"][rule]) != 2 * pairs
                       for rule in rules)):
            raise ValueError("first optimized gate lacks complete five-rule paired evidence")
        gate_evidence = {"generation": 4, "verdict": "accept", "pairs_per_rule": pairs,
                         "gate_sha256": sha(gate_path),
                         "accepted_sha256": first_gate_commit["accepted_model_sha256"]}
    gates = list((RUN / "metrics").glob("gate_g*.json"))
    receipt = {
        "positions": bound, "generation_commit_sha256": sha(commit_path),
        "checkpoint_sha256": sha(checkpoint), "accepted_sha256": sha(accepted),
        "selfplay_generations": len(selfplays), "learner_records": len(learners),
        "gate_records": len(gates), "first_gate": gate_evidence,
        "replay_shards": len(commit["replay_shards"]),
        "config_sha256": sha(CONFIG), "lineage_hash": plan["child_lineage_hash"],
    }
    atomic(CHILD_BASE / "receipts" / f"to{bound}.json", receipt)
    return receipt


def run_phases(phases: tuple[tuple[str, int, bool, bool], ...]) -> None:
    cmd = [sys.executable, "-u", "-B", "-m", "training.v3", "run", "--config", str(CONFIG)]
    plan = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    if plan.returncode:
        raise RuntimeError(f"V3 guarded plan failed: {plan.stderr[-1500:]}")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        for phase, bound, resume, gate in phases:
            atomic(STATE, {"status": f"running_{phase}", "bound": bound})
            log = CHILD_BASE / "logs" / f"{phase}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                finished = subprocess.run(
                    [*cmd, *(["--resume"] if resume else []), "--execute",
                     "--max-train-positions", str(bound)],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                )
            if finished.returncode:
                raise RuntimeError(f"V3 {phase} exited {finished.returncode}: {log}")
            verify_boundary(bound, require_gate=gate)
        atomic(STATE, {"status": "complete", "bound": TARGET})
    except Exception as exc:
        atomic(STATE, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def launch() -> None:
    if STATE.exists():
        raise FileExistsError("watcher already started; inspect before retry")
    if not gpu_idle():
        raise RuntimeError("one or both GPUs are in use")
    free_gib = shutil.disk_usage(CHILD_BASE.parent).free / 1024**3
    if free_gib < 25:
        raise RuntimeError(f"need at least 25 GiB free before new pool fork, have {free_gib:.1f}")
    run_phases((("canary", CANARY, False, False),
                ("first_gate", FIRST_GATE, True, True),
                ("to10m", TARGET, True, True)))


def resume_after_160k() -> None:
    if read(STATE).get("status") != "failed":
        raise ValueError("expected failed launcher state before recovery")
    if not gpu_idle():
        raise RuntimeError("one or both GPUs are in use")
    manifest = read(RUN / "run_manifest.json")
    if manifest["formal_loop_state"]["train_positions_consumed"] != 160_000:
        raise ValueError("recovery is valid only at the committed 160k boundary")
    if read(RUN / "manifests/generations/g000002.json")["gate_verdict"] != "not_run":
        raise ValueError("160k gate state changed; inspect before recovery")
    verify_boundary(160_000, require_gate=False)
    run_phases((("first_gate", FIRST_GATE, True, True),
                ("to10m", TARGET, True, True)))


def resume_after_320k() -> None:
    if read(STATE).get("status") != "failed":
        raise ValueError("expected failed launcher state before recovery")
    if not gpu_idle():
        raise RuntimeError("one or both GPUs are in use")
    manifest = read(RUN / "run_manifest.json")
    if manifest["formal_loop_state"]["train_positions_consumed"] != FIRST_GATE:
        raise ValueError("recovery is valid only at the committed 320k boundary")
    verify_boundary(FIRST_GATE, require_gate=True)
    run_phases((("to10m", TARGET, True, True),))


def resume_after_topology() -> None:
    """Resume unchanged 24-actor training after a verified topology screen."""
    if read(STATE).get("status") != "failed":
        raise ValueError("expected original launcher to stop after the signal drain")
    screen = read(CHILD_BASE / "topology_screen_24_36_40" / "complete.json")
    if screen.get("status") != "complete":
        raise ValueError("topology screen did not complete")
    if not gpu_idle():
        raise RuntimeError("one or both GPUs are in use")
    manifest = read(RUN / "run_manifest.json")
    bound = manifest["formal_loop_state"]["train_positions_consumed"]
    if (bound != screen["drained_positions"]
            or not str(manifest.get("stop_reason", "")).startswith("drained_after_signal_")):
        raise ValueError("training boundary differs from topology screen")
    receipt = verify_boundary(bound, require_gate=True, allow_drained=True)
    if receipt["generation_commit_sha256"] != screen["generation_commit_sha256"]:
        raise ValueError("drained generation commit changed")
    run_phases((("to10m", TARGET, True, True),))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "launch", "resume_after_160k", "resume_after_320k", "resume_after_topology"}:
        raise SystemExit("usage: stage2_fla_raw_10m_continue.py prepare|launch|resume_after_160k|resume_after_320k|resume_after_topology")
    if sys.argv[1] == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    elif sys.argv[1] == "resume_after_160k":
        resume_after_160k()
    elif sys.argv[1] == "resume_after_320k":
        resume_after_320k()
    elif sys.argv[1] == "resume_after_topology":
        resume_after_topology()
    else:
        launch()
