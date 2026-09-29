"""Compare six FLA designs at donor and 1M endpoints on one GPU.

The direct matches use Classic, fixed 256 simulations, 32 paired openings,
four same-card replicas, and a ten-edge wheel graph. Every design faces at
least three distinct peers.
CPU latency remains a separate Windows measurement, never a GPU proxy.
3M self-play requires a separate explicit --continue-to-3m request.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from stage2_fla_queue_trim_resume import verify_completion
from stage2_fla_selfplay_queue import QUEUE_ROOT, ROOT, atomic_json, sha256
from training.v3.anchored_elo import fit_anchor_scale
from training.v3.evaluation import build_openings, write_opening_manifest
from training.v3.evaluation_runtime import EvaluationModelSource, play_paired_openings_replicated
from training.v3.evaluation_snapshot import export_evaluation_snapshot
from training.v3.gate import summarize_paired_results

NAMES = (
    "gravity_b8", "column_2d_b8c192", "raw3d_to2d_b8",
    "raw3d_to2d_thin_b8c192", "column3d_v2_thin_b8c192",
    "raw3d_to2d_thin_b6c128",
)
MATCH_ROOT = ROOT / "training/runs/stage2/fla/direct_1m/r2_four_workers"
STATE = MATCH_ROOT / "controller_state.json"
CPU_TABLE = MATCH_ROOT / "cpu_evidence_table.json"
POSITIONS_3M = 3_000_000
OPENINGS = 32
SIMS = 256
WORKERS = 4
CPUCT = 1.5
SEED = 271828
# A gravity spoke plus a ring on the other five designs: 10 unique edges.
EDGES = tuple((NAMES[0], name) for name in NAMES[1:]) + tuple(
    (NAMES[i], NAMES[1 + (i % 5)]) for i in range(1, 6)
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(queue: dict) -> dict[str, dict]:
    completed = queue.get("completed", [])
    if queue.get("status") != "complete" or [row["name"] for row in completed] != list(NAMES):
        raise ValueError("six serial 1M runs are not complete in the expected order")
    result = {}
    for row in completed:
        name = row["name"]
        run_dir = Path(row["run_dir"])
        receipt = verify_completion(run_dir)
        for key in ("terminal_sha256", "accepted_sha256", "generation"):
            if key in row and row[key] != receipt[key]:
                raise ValueError(f"1M receipt drift for {name}: {key}")
        config_path = ROOT / QUEUE_ROOT / "configs" / f"{run_dir.name}.json"
        if sha256(config_path) != row["config_sha256"]:
            raise ValueError(f"self-play config hash drift: {name}")
        config = read_json(config_path)
        donor = Path(config["run"]["warm_start_checkpoint"])
        if sha256(donor) != row["donor_sha256"]:
            raise ValueError(f"offline donor hash drift: {name}")
        result[name] = {
            "run_dir": str(run_dir), "config": str(config_path),
            "config_sha256": row["config_sha256"], "model_config": config["model"],
            "donor": str(donor), "donor_sha256": row["donor_sha256"],
            "terminal": str(run_dir / receipt["terminal_checkpoint"]),
            "terminal_sha256": receipt["terminal_sha256"],
            "terminal_generation": receipt["generation"],
            "accepted": str(run_dir / receipt["accepted_model"]),
            "accepted_sha256": receipt["accepted_sha256"],
        }
    return result


def cpu_means() -> dict[str, float]:
    payload = read_json(CPU_TABLE)
    if payload.get("cpu_protocol") != "512 simulations, 10 s between responses, 60 s between groups, 3 repeats":
        raise ValueError("CPU table protocol changed")
    rows = {row["name"]: row for row in payload["rows"]}
    result = {}
    for name in NAMES:
        row = rows[name]
        mean = float(row["cpu_512_mean_s"])
        model_sha = row.get("cpu_512_model_sha256", row.get("offline_1m_model_sha256", ""))
        if not (0 < mean <= 3.8) or len(model_sha) != 64:
            raise ValueError(f"missing qualifying CPU evidence: {name}")
        result[name] = mean
    return result


def prepare_evaluation_snapshots(rows: dict[str, dict]) -> None:
    """Project terminal weights without changing the original V3 checkpoint."""
    target_dir = MATCH_ROOT / "evaluation_snapshots"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, row in rows.items():
        target = target_dir / f"{name}.pt"
        receipt_path = target_dir / f"{name}.json"
        model_id = f"fla1m-{name}"
        if target.exists() != receipt_path.exists():
            raise ValueError(f"incomplete terminal evaluation snapshot: {name}")
        if target.exists():
            receipt = read_json(receipt_path)
            if (receipt.get("output_sha256") != sha256(target)
                    or receipt.get("source_checkpoint_sha256") != row["terminal_sha256"]
                    or receipt.get("model_id") != model_id):
                raise ValueError(f"terminal evaluation snapshot drift: {name}")
        else:
            receipt = export_evaluation_snapshot(row["terminal"], target, model_id=model_id)
            if receipt["source_checkpoint_sha256"] != row["terminal_sha256"]:
                raise ValueError(f"terminal source hash drift: {name}")
            atomic_json(receipt_path, receipt)
        payload = torch.load(target, map_location="cpu", weights_only=True)
        if payload.get("model_config") != row["model_config"]:
            raise ValueError(f"terminal evaluation model config drift: {name}")
        row["terminal_snapshot"] = str(target)
        row["terminal_snapshot_sha256"] = receipt["output_sha256"]


def match_all(rows: dict[str, dict], *, device: str) -> None:
    MATCH_ROOT.mkdir(parents=True, exist_ok=True)
    opening_path = MATCH_ROOT / "classic_openings_32.json"
    if not opening_path.exists():
        write_opening_manifest(opening_path, build_openings(OPENINGS, run_seed=20260925,
                                                            opening_id_prefix="fla-r1"))
    from training.v3.evaluation import load_opening_manifest
    openings = load_opening_manifest(opening_path)
    if len(openings) != OPENINGS or any(row.rule_id != "classic" for row in openings):
        raise ValueError("paired opening manifest is not 32 Classic positions")
    opening_sha = sha256(opening_path)
    for phase in ("donor", "selfplay_1m_terminal"):
        phase_dir = MATCH_ROOT / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        for name_a, name_b in EDGES:
            target = phase_dir / f"{name_a}__vs__{name_b}.json"
            model_a = rows[name_a]
            model_b = rows[name_b]
            path_key = "donor" if phase == "donor" else "terminal"
            hash_key = f"{path_key}_sha256"
            expected = {"a": model_a[hash_key], "b": model_b[hash_key]}
            if target.exists():
                existing = read_json(target)
                if (existing.get("model_sha256") != expected or
                        existing.get("opening_sha256") != opening_sha or
                        existing.get("search_sims") != SIMS or
                        existing.get("cpuct") != CPUCT or
                        existing.get("runtime", {}).get("worker_processes") != WORKERS or
                        len(existing.get("games", [])) != 2 * OPENINGS):
                    raise ValueError(f"existing match differs from frozen protocol: {target}")
                continue
            source_a = model_a["donor"] if phase == "donor" else model_a["terminal_snapshot"]
            source_b = model_b["donor"] if phase == "donor" else model_b["terminal_snapshot"]
            evaluated = play_paired_openings_replicated(
                openings,
                candidate_source=EvaluationModelSource("v3_artifact", source_a, name_a),
                incumbent_source=EvaluationModelSource("v3_artifact", source_b, name_b),
                search_sims=SIMS, cpuct=CPUCT,
                worker_devices=(device,) * WORKERS,
            )
            games = evaluated.games
            summary = summarize_paired_results(games, bootstrap_samples=2000,
                                               bootstrap_seed=SEED)
            atomic_json(target, {
                "schema": "connect4-stage2-fla-direct-match-v1", "created_at_utc": now(),
                "phase": phase, "model_a": name_a, "model_b": name_b,
                "model_sha256": expected, "opening_sha256": opening_sha,
                "search_sims": SIMS, "cpuct": CPUCT, "device": device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "evaluation_artifact_sha256": {
                    "a": sha256(Path(source_a)), "b": sha256(Path(source_b)),
                },
                "runtime": evaluated.metrics.to_dict(),
                "games": [asdict(game) for game in games], "summary": asdict(summary),
            })
            atomic_json(STATE, {"status": "matching", "phase": phase,
                                "last_completed": str(target), "updated_at_utc": now()})


def rating_report() -> dict:
    report = {"schema": "connect4-stage2-fla-direct-rating-v1", "created_at_utc": now(),
              "reference": NAMES[0], "search_sims": SIMS, "opening_pairs_per_edge": OPENINGS,
              "edge_count_per_phase": len(EDGES), "phases": {}}
    for phase in ("donor", "selfplay_1m_terminal"):
        matches = [read_json(MATCH_ROOT / phase / f"{a}__vs__{b}.json") for a, b in EDGES]
        observations = [(row["model_a"], row["model_b"], game["candidate_score"])
                        for row in matches for game in row["games"]]
        ratings = fit_anchor_scale(observations, model_ids=NAMES,
                                   reference_model_id=NAMES[0], prior_sigma=400.0)
        # Bootstrap whole opening pairs, keeping color-swapped games together.
        rng = np.random.default_rng(SEED)
        draws = {name: [] for name in NAMES}
        for _ in range(300):
            indexes = rng.integers(0, OPENINGS, size=OPENINGS)
            sampled = []
            for row in matches:
                for index in indexes:
                    for game in row["games"][2 * index:2 * index + 2]:
                        sampled.append((row["model_a"], row["model_b"], game["candidate_score"]))
            fit = fit_anchor_scale(sampled, model_ids=NAMES,
                                   reference_model_id=NAMES[0], prior_sigma=400.0)
            for name in NAMES:
                draws[name].append(fit[name])
        report["phases"][phase] = {
            "ratings_vs_gravity": {name: {"elo": ratings[name],
                                          "ci95": list(map(float, np.quantile(draws[name], [0.025, 0.975])))}
                                   for name in NAMES},
            "matches": [{"a": row["model_a"], "b": row["model_b"],
                         "w_d_l": [row["summary"]["overall"][key] for key in ("wins", "draws", "losses")],
                         "a_first": row["summary"]["candidate_as_first"],
                         "a_second": row["summary"]["candidate_as_second"],
                         "point_score": row["summary"]["overall"]["point_score"],
                         "sha256": sha256(MATCH_ROOT / phase / f"{row['model_a']}__vs__{row['model_b']}.json")}
                        for row in matches],
        }
    atomic_json(MATCH_ROOT / "ratings.json", report)
    return report


def select_four(report: dict, cpu: dict[str, float]) -> dict:
    # Preserve the gravity control and B6 latency frontier. Rank the other four
    # by CPU first, then the two separately measured strength endpoints.
    donor = report["phases"]["donor"]["ratings_vs_gravity"]
    terminal = report["phases"]["selfplay_1m_terminal"]["ratings_vs_gravity"]
    others = [name for name in NAMES if name not in (NAMES[0], NAMES[-1])]
    def ranks(values: dict[str, float], reverse: bool = False) -> dict[str, int]:
        return {name: index + 1 for index, name in enumerate(sorted(others,
                                                                    key=lambda name: values[name],
                                                                    reverse=reverse))}
    speed_rank = ranks(cpu)
    donor_rank = ranks({name: donor[name]["elo"] for name in others}, reverse=True)
    terminal_rank = ranks({name: terminal[name]["elo"] for name in others}, reverse=True)
    combined = {name: 0.50 * speed_rank[name] + 0.35 * terminal_rank[name]
                + 0.15 * donor_rank[name] for name in others}
    selected_other = sorted(others, key=lambda name: (combined[name], cpu[name], name))[:2]
    selected = [name for name in NAMES if name in {NAMES[0], NAMES[-1], *selected_other}]
    decision = {"schema": "connect4-stage2-fla-3m-selection-v1", "created_at_utc": now(),
                "status": "provisional_screen_for_3m_not_flash_qualification",
                "rule": "gravity and B6 control/frontier retained; other four: 50% CPU rank, 35% terminal Elo rank, 15% donor Elo rank; lower is better",
                "cpu_table_sha256": sha256(CPU_TABLE), "ratings_sha256": sha256(MATCH_ROOT / "ratings.json"),
                "ranking": {name: {"cpu_mean_s": cpu[name], "donor_elo": donor[name],
                                   "terminal_elo": terminal[name], "rank_score": combined.get(name)}
                            for name in NAMES},
                "selected": selected, "eliminated": [name for name in NAMES if name not in selected]}
    atomic_json(MATCH_ROOT / "selection.json", decision)
    return decision


def resume_3m(selection: dict, rows: dict[str, dict]) -> None:
    for name in selection["selected"]:
        config_path = Path(rows[name]["config"])
        run_dir = Path(rows[name]["run_dir"])
        log_path = MATCH_ROOT / "resume_logs" / f"{name}_to_3m.log"
        receipt_path = MATCH_ROOT / "resume_receipts" / f"{name}_3m.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["config_sha256"] != sha256(config_path):
                raise ValueError(f"3M receipt config drift: {name}")
            continue
        manifest = read_json(run_dir / "run_manifest.json")
        consumed = int((manifest.get("formal_loop_state") or {}).get("train_positions_consumed", -1))
        if consumed != 1_000_000:
            raise ValueError(f"cannot safely resume {name}; expected exact 1M, got {consumed}")
        command = [sys.executable, "-u", "-B", "-m", "training.v3", "run",
                   "--config", str(config_path), "--resume", "--execute",
                   "--max-train-positions", str(POSITIONS_3M)]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(STATE, {"status": "resuming_3m", "active": name,
                            "started_at_utc": now(), "command": command})
        with log_path.open("a", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"3M resume exited {result.returncode}: {log_path}")
        manifest = read_json(run_dir / "run_manifest.json")
        consumed = int((manifest.get("formal_loop_state") or {}).get("train_positions_consumed", -1))
        if manifest.get("stop_reason") != "max_train_positions" or consumed != POSITIONS_3M:
            raise RuntimeError(f"3M boundary not committed for {name}: {consumed}")
        pointer = read_json(run_dir / "manifests/latest_generation.json")
        commit_path = run_dir / pointer["commit"]
        if sha256(commit_path) != pointer["commit_sha256"]:
            raise ValueError(f"3M generation commit hash drift: {name}")
        commit = read_json(commit_path)
        for path_key, hash_key in (("checkpoint", "checkpoint_sha256"),
                                   ("accepted_model_path", "accepted_model_sha256")):
            if sha256(run_dir / commit[path_key]) != commit[hash_key]:
                raise ValueError(f"3M model hash drift: {name} {path_key}")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt_path, {"name": name, "train_positions_consumed": consumed,
                                  "generation": pointer["generation"], "config_sha256": sha256(config_path),
                                  "generation_commit_sha256": pointer["commit_sha256"],
                                  "terminal_sha256": commit["checkpoint_sha256"],
                                  "accepted_sha256": commit["accepted_model_sha256"],
                                  "completed_at_utc": now()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--continue-to-3m", action="store_true",
                        help="explicitly resume the selected four after direct matches")
    args = parser.parse_args()
    if args.continue_to_3m and not args.execute:
        parser.error("--continue-to-3m requires --execute")
    cpu = cpu_means()
    if not args.execute:
        print(json.dumps({"status": "plan_only", "names": NAMES, "edges": EDGES,
                          "phases": ["donor", "selfplay_1m_terminal"],
                          "openings_per_edge": OPENINGS, "search_sims": SIMS,
                          "replicated_workers_on_device": WORKERS,
                          "cpu_means_s": cpu, "device": args.device,
                          "continue_to_3m": False}))
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for FLA direct matches")
    queue = read_json(ROOT / QUEUE_ROOT / "queue_state.json")
    rows = identity(queue)
    MATCH_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_evaluation_snapshots(rows)
    atomic_json(MATCH_ROOT / "inputs.json", {"model_identity": rows,
                                            "cpu_table_sha256": sha256(CPU_TABLE),
                                            "queue_sha256": sha256(ROOT / QUEUE_ROOT / "queue_state.json"),
                                            "device": args.device,
                                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                            "opening_pairs_per_edge": OPENINGS,
                                            "search_sims": SIMS,
                                            "replicated_workers_on_device": WORKERS,
                                            "continue_to_3m": args.continue_to_3m})
    try:
        match_all(rows, device=args.device)
        report = rating_report()
        selection = select_four(report, cpu)
        if args.continue_to_3m:
            resume_3m(selection, rows)
        atomic_json(STATE, {"status": "complete_3m" if args.continue_to_3m else "complete_direct_matches",
                            "completed_at_utc": now(),
                            "selection_sha256": sha256(MATCH_ROOT / "selection.json")})
        return 0
    except Exception as exc:
        atomic_json(STATE, {"status": "failed", "failed_at_utc": now(),
                            "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
