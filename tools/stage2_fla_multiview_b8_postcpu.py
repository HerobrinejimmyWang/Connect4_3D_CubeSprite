"""CPU-gated multiview B8 1M V3 self-play and paired Classic group matches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_fla_b6_selfplay_gate import validate_cpu  # noqa: E402
from stage2_fla_queue_trim_resume import verify_completion  # noqa: E402
from stage2_fla_selfplay_queue import TEMPLATE, atomic_json, sha256, validate_donor  # noqa: E402
from training.v3.config import V3Config  # noqa: E402
from training.v3.evaluation import load_opening_manifest  # noqa: E402
from training.v3.evaluation_runtime import EvaluationModelSource, play_paired_openings_replicated  # noqa: E402
from training.v3.evaluation_snapshot import export_evaluation_snapshot  # noqa: E402
from training.v3.gate import summarize_paired_results  # noqa: E402

NAME = "multiview_resnet_b8c192"
BASE = ROOT / "training/runs/stage2/fla/multiview_b8_1m"
CONFIG = BASE / "configs" / f"{NAME}__standard_late__seed271828.json"
MODEL = BASE / "runs" / NAME / "standard_late/seed271828/model.pt"
REPORT = MODEL.parent / "report.json"
CPU_DIR = BASE / "cpu_gate"
CPU_RESULT = CPU_DIR / f"{NAME}_512.json"
CPU_SUMMARY = CPU_DIR / "summary.json"
REFERENCE = CPU_DIR / "reference_summary.json"
SELFPLAY = BASE / "selfplay"
RUN_ID = "fla6m_multiview_resnet_b8c192_warm_seed271828"
RUN_DIR = SELFPLAY / "runs" / RUN_ID
RUN_CONFIG = SELFPLAY / "configs" / f"{RUN_ID}.json"
STATE = BASE / "postcpu_state.json"
OLD = ROOT / "training/runs/stage2/fla/direct_1m/r2_four_workers"
PEERS = ("gravity_b8", "raw3d_to2d_b8", "raw3d_to2d_thin_b8c192")
POSITIONS = 1_000_000


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def initial_config() -> tuple[V3Config, str, dict]:
    design = read(BASE / "design.json")
    if sha256(CONFIG) != design["config_sha256"]:
        raise ValueError("frozen multiview donor config hash mismatch")
    model_config, donor_sha = validate_donor(CONFIG, MODEL, REPORT)
    cpu = validate_cpu(CPU_RESULT, CPU_SUMMARY, REFERENCE,
                       config_path=CONFIG, model_path=MODEL, variant=NAME)
    template = read(ROOT / TEMPLATE)
    if template["selfplay"]["rule_id"] != "classic" or template["selfplay"].get("multi_rule_ids", []):
        raise ValueError("Classic-only self-play required")
    template["run"] = {
        "run_id": RUN_ID, "seed": 271828, "run_dir": str(RUN_DIR), "resume": False,
        "warm_start_checkpoint": str(MODEL),
        "warm_start_checkpoint_sha256": donor_sha,
        "warm_start_mode": "model_only_fresh_optimizer_replay_v1",
    }
    template["model"] = model_config
    template["runtime"].update({
        "device": "cuda:0", "selfplay_devices": ["cuda:0"],
        "evaluation_devices": ["cuda:0"], "actor_processes": 24,
        "mcts_lanes_per_actor": 4, "inference_batch_size": 32,
    })
    return V3Config.from_dict(template), donor_sha, cpu


def run_selfplay(config: V3Config, donor_sha: str, cpu: dict) -> dict:
    if RUN_CONFIG.exists() or RUN_DIR.exists():
        raise FileExistsError("multiview 1M run already exists; inspect before retry")
    RUN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    RUN_CONFIG.write_text(config.to_json(), encoding="utf-8")
    plan = subprocess.run([sys.executable, "-B", "-m", "training.v3", "run",
                           "--config", str(RUN_CONFIG)], cwd=ROOT,
                          capture_output=True, text=True)
    if plan.returncode:
        raise RuntimeError(f"V3 guarded plan failed: {plan.stderr[-2000:]}")
    atomic_json(STATE, {"status": "selfplay_running", "run_id": RUN_ID,
                        "config_sha256": sha256(RUN_CONFIG), "donor_sha256": donor_sha,
                        "cpu_evidence": cpu})
    log = SELFPLAY / "selfplay.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-u", "-B", "-m", "training.v3", "run",
                                 "--config", str(RUN_CONFIG), "--execute",
                                 "--max-train-positions", str(POSITIONS)],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"V3 self-play exited {result.returncode}; see {log}")
    receipt = verify_completion(RUN_DIR)
    atomic_json(SELFPLAY / "receipt.json", {"run_id": RUN_ID,
                                             "config_sha256": sha256(RUN_CONFIG),
                                             "donor_sha256": donor_sha, **receipt})
    atomic_json(STATE, {"status": "selfplay_complete", "run_id": RUN_ID,
                        "config_sha256": sha256(RUN_CONFIG), "donor_sha256": donor_sha,
                        "cpu_evidence": cpu, **receipt})
    return receipt


def run_matches(receipt: dict, donor_sha: str, cpu: dict) -> None:
    old = read(OLD / "inputs.json")["model_identity"]
    opening_path = OLD / "classic_openings_32.json"
    openings = load_opening_manifest(opening_path)
    if len(openings) != 32 or any(item.rule_id != "classic" for item in openings):
        raise ValueError("expected 32 Classic paired openings")
    terminal = RUN_DIR / receipt["terminal_checkpoint"]
    snapshot = BASE / "direct_1m/evaluation_snapshots" / f"{NAME}.pt"
    if snapshot.exists():
        existing = read(snapshot.with_suffix(".json"))
        if existing["output_sha256"] != sha256(snapshot) or existing["source_checkpoint_sha256"] != receipt["terminal_sha256"]:
            raise ValueError("existing terminal evaluation snapshot drift")
    else:
        record = export_evaluation_snapshot(terminal, snapshot, model_id=f"fla1m-{NAME}")
        atomic_json(snapshot.with_suffix(".json"), record)
    inputs = {"schema": "connect4-stage2-fla-multiview-direct-inputs-v1",
              "model_config_sha256": sha256(RUN_CONFIG), "donor_sha256": donor_sha,
              "terminal_checkpoint_sha256": receipt["terminal_sha256"],
              "terminal_snapshot_sha256": sha256(snapshot),
              "accepted_sha256": receipt["accepted_sha256"],
              "opening_sha256": sha256(opening_path), "cpu_evidence": cpu,
              "search_sims": 256, "cpuct": 1.5, "opening_pairs": 32,
              "worker_processes": 4, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    base = BASE / "direct_1m"
    input_path = base / "inputs.json"
    if input_path.exists() and read(input_path) != inputs:
        raise ValueError("direct match identity changed")
    atomic_json(input_path, inputs)
    for phase, source in (("donor", MODEL), ("selfplay_1m_terminal", snapshot)):
        for peer in PEERS:
            peer_row = old[peer]
            key = "donor" if phase == "donor" else "terminal_snapshot"
            peer_path = Path(peer_row[key])
            peer_sha = peer_row[f"{key}_sha256"]
            if sha256(peer_path) != peer_sha:
                raise ValueError(f"peer artifact hash changed: {peer} {phase}")
            target = base / phase / f"{NAME}__vs__{peer}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = {"a": sha256(source), "b": peer_sha}
            if target.exists():
                row = read(target)
                if row.get("model_sha256") != expected or len(row.get("games", [])) != 64:
                    raise ValueError(f"existing direct match drift: {target}")
                continue
            evaluated = play_paired_openings_replicated(
                openings,
                candidate_source=EvaluationModelSource("v3_artifact", str(source), NAME),
                incumbent_source=EvaluationModelSource("v3_artifact", str(peer_path), peer),
                search_sims=256, cpuct=1.5, worker_devices=("cuda:0",) * 4,
            )
            summary = summarize_paired_results(evaluated.games, bootstrap_samples=2000,
                                               bootstrap_seed=271828)
            atomic_json(target, {
                "schema": "connect4-stage2-fla-multiview-direct-v1",
                "phase": phase, "model_a": NAME, "model_b": peer,
                "model_sha256": expected, "opening_sha256": inputs["opening_sha256"],
                "search_sims": 256, "cpuct": 1.5,
                "runtime": evaluated.metrics.to_dict(),
                "games": [asdict(game) for game in evaluated.games],
                "summary": asdict(summary),
            })
            atomic_json(STATE, {"status": "matching", "phase": phase,
                                "last_completed": str(target), "run_id": RUN_ID})
    atomic_json(STATE, {"status": "complete", "run_id": RUN_ID,
                        "config_sha256": sha256(RUN_CONFIG), "donor_sha256": donor_sha,
                        "cpu_evidence": cpu, **receipt})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, donor_sha, cpu = initial_config()
    if not args.execute:
        print(json.dumps({"status": "plan_only", "run_id": RUN_ID,
                          "donor_sha256": donor_sha, "cpu": cpu,
                          "peers": PEERS, "phases": ["donor", "selfplay_1m_terminal"]}))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("physical GPU0 only; GPU1 reserved")
    if STATE.exists():
        raise FileExistsError(f"post-CPU controller already started: {STATE}")
    try:
        receipt = run_selfplay(config, donor_sha, cpu)
        run_matches(receipt, donor_sha, cpu)
        return 0
    except Exception as exc:
        atomic_json(STATE, {"status": "failed", "run_id": RUN_ID,
                            "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
