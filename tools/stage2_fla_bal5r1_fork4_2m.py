"""Fork four 1M FLA accepted models into BAL-5 R1 style 2M V3 runs.

The parent 1M plus child 2M gives 3M cumulative exposure. This is a semantic
fork: optimizer and replay start fresh, and the child has a new V3 run ID.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.config import V3Config  # noqa: E402
from training.v3.pipeline import lineage_config_hash  # noqa: E402
from stage2_fla_queue_trim_resume import verify_completion  # noqa: E402
from stage2_fla_selfplay_queue import QUEUE_ROOT, atomic_json, sha256  # noqa: E402

NAMES = ("gravity_b8", "raw3d_to2d_b8", "raw3d_to2d_thin_b8c192",
         "raw3d_to2d_thin_b6c128")
BASE = ROOT / "training/runs/stage2/fla/selfplay/r3_bal5r1_fork_v2"
STATE = BASE / "queue_state.json"
REFERENCE = ROOT / "training/runs/stage2/bal5/r1/configs/bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828.json"
DIRECT = ROOT / "training/runs/stage2/fla/direct_1m/r3_raw_b6_full"
CHILD_POSITIONS = 2_000_000


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_commit(run_dir: Path) -> tuple[dict, dict]:
    pointer = read(run_dir / "manifests/latest_generation.json")
    path = run_dir / pointer["commit"]
    if sha256(path) != pointer["commit_sha256"]:
        raise ValueError(f"commit pointer SHA-256 mismatch: {run_dir}")
    commit = read(path)
    for item, digest in (("checkpoint", "checkpoint_sha256"),
                         ("accepted_model_path", "accepted_model_sha256")):
        if sha256(run_dir / commit[item]) != commit[digest]:
            raise ValueError(f"committed artifact SHA-256 mismatch: {run_dir} {item}")
    return pointer, commit


def prepare() -> list[dict]:
    queue = read(ROOT / QUEUE_ROOT / "queue_state.json")
    completed = {row["name"]: row for row in queue["completed"]}
    if queue["status"] != "complete" or len(completed) != 6:
        raise ValueError("six 1M parent runs are not complete")
    bal5 = read(REFERENCE)
    mix = bal5["selfplay"]["opening_temperature_mixture"].copy()
    if (bal5["selfplay"]["search_schedule"][0]["games"] != 800
            or bal5["gate"]["candidate_train_positions"] != 300_000
            or bal5["gate"]["bootstrap_candidate_train_positions"] != 150_000
            or mix["lowered_temperature_plies"] != 8
            or mix["lowered_temperature_multiplier"] != 0.5):
        raise ValueError("BAL-5 R1 reference contract changed")
    # Child position 0 is cumulative position 1M in the parent plus child
    # evidence. Applying the mixture immediately reproduces its post-1M phase.
    mix["start_train_positions"] = 0
    rows = []
    for name in NAMES:
        parent = completed[name]
        parent_dir = Path(parent["run_dir"])
        receipt = verify_completion(parent_dir)
        if any(parent[key] != receipt[key] for key in
               ("terminal_sha256", "accepted_sha256", "generation") if key in parent):
            raise ValueError(f"parent 1M receipt changed: {name}")
        pointer, commit = latest_commit(parent_dir)
        accepted = parent_dir / commit["accepted_model_path"]
        source_config = ROOT / QUEUE_ROOT / "configs" / f"{parent_dir.name}.json"
        if sha256(source_config) != parent["config_sha256"]:
            raise ValueError(f"parent config SHA-256 changed: {name}")
        raw = read(source_config)
        if raw["selfplay"]["rule_id"] != "classic" or raw["selfplay"]["multi_rule_ids"]:
            raise ValueError(f"single Classic rule required: {name}")
        accepted_payload = torch.load(accepted, map_location="cpu", weights_only=True)
        if accepted_payload.get("format") != "connect4-v3-model":
            raise ValueError(f"parent accepted artifact format mismatch: {name}")
        accepted_metadata = accepted_payload.get("metadata", {})
        if accepted_metadata.get("config_hash") != read(parent_dir / "run_manifest.json")["config_hash"]:
            raise ValueError(f"parent accepted lineage hash mismatch: {name}")
        if isinstance(accepted_metadata.get("candidate_model_id"), str):
            warm_source = accepted
            warm_mode = "accepted_artifact_fresh_optimizer_replay_v1"
        else:
            donor = Path(raw["run"]["warm_start_checkpoint"])
            if sha256(donor) != raw["run"]["warm_start_checkpoint_sha256"]:
                raise ValueError(f"parent donor SHA-256 mismatch: {name}")
            donor_payload = torch.load(donor, map_location="cpu", weights_only=True)
            if donor_payload.get("model_config") != accepted_payload.get("model_config"):
                raise ValueError(f"parent accepted/donor model config mismatch: {name}")
            donor_state = donor_payload["model_state"]
            accepted_state = accepted_payload["model_state"]
            if donor_state.keys() != accepted_state.keys() or any(
                not torch.equal(donor_state[key], accepted_state[key]) for key in donor_state
            ):
                raise ValueError(f"parent accepted root differs from donor weights: {name}")
            warm_source = donor
            warm_mode = "model_only_fresh_optimizer_replay_v1"
        run_id = f"fla_r3_{name}_bal5r1_from1m_seed271828"
        run_dir = BASE / "runs" / run_id
        raw["run"] = {
            "run_id": run_id, "run_dir": str(run_dir), "seed": 271828,
            "resume": False, "warm_start_checkpoint": str(warm_source),
            "warm_start_checkpoint_sha256": sha256(warm_source),
            "warm_start_mode": warm_mode,
        }
        raw["selfplay"]["search_schedule"] = [
            {**raw["selfplay"]["search_schedule"][0], "games": 800,
             "start_generation": 0}
        ]
        raw["selfplay"]["opening_temperature_mixture"] = mix
        raw["gate"]["candidate_train_positions"] = 300_000
        raw["gate"]["bootstrap_candidate_train_positions"] = 150_000
        raw["runtime"].update({
            "device": "cuda:0", "selfplay_devices": ["cuda:0"],
            "evaluation_devices": ["cuda:0"], "actor_processes": 24,
            "mcts_lanes_per_actor": 4, "inference_batch_size": 32,
        })
        config = V3Config.from_dict(raw)
        parent_hash = read(parent_dir / "run_manifest.json")["config_hash"]
        child_hash = lineage_config_hash(config)
        if child_hash == parent_hash:
            raise ValueError(f"expected semantic fork did not change lineage hash: {name}")
        config_path = BASE / "configs" / f"{run_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = config.to_json()
        if config_path.exists() and config_path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"existing fork config differs: {config_path}")
        if not config_path.exists():
            config_path.write_text(encoded, encoding="utf-8")
        rows.append({
            "name": name, "parent_run_dir": str(parent_dir),
            "parent_config_sha256": parent["config_sha256"],
            "parent_lineage_hash": parent_hash,
            "parent_generation_commit_sha256": pointer["commit_sha256"],
            "parent_terminal_sha256": commit["checkpoint_sha256"],
            "parent_accepted_sha256": sha256(accepted),
            "warm_start_checkpoint": str(warm_source),
            "warm_start_sha256": sha256(warm_source),
            "warm_start_mode": warm_mode,
            "child_run_dir": str(run_dir), "child_config": str(config_path),
            "child_config_sha256": sha256(config_path), "child_lineage_hash": child_hash,
            "parent_positions": 1_000_000, "child_target_positions": CHILD_POSITIONS,
            "cumulative_exposure_positions": 3_000_000,
        })
    atomic_json(BASE / "fork_manifest.json", {
        "schema": "connect4-stage2-fla-bal5r1-fork4-v1",
        "reference_config": str(REFERENCE), "reference_config_sha256": sha256(REFERENCE),
        "reason": "post-1M BAL-5 R1 games/gate/temperature settings change V3 semantics",
        "child_uses_fresh_optimizer_and_replay": True,
        "pool": {"physical_gpu": 0, "actor_processes": 24,
                 "mcts_lanes_per_actor": 4, "inference_batch_size": 32},
        "rows": rows,
    })
    return rows


def gpu0_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr[-500:]}")
    first = result.stdout.splitlines()[0].split(",")
    return first[0].strip() == "0" and int(first[1].strip().split()[0]) < 500


def execute(rows: list[dict], wait_hours: float) -> None:
    if STATE.exists():
        raise FileExistsError(f"fork controller already has state: {STATE}")
    state = {"schema": "connect4-stage2-fla-bal5r1-fork4-queue-v1",
             "status": "waiting_gpu0", "planned": list(NAMES), "completed": [],
             "fork_manifest_sha256": sha256(BASE / "fork_manifest.json")}
    atomic_json(STATE, state)
    deadline = time.monotonic() + wait_hours * 3600
    try:
        required = tuple(DIRECT / f"raw3d_to2d_full_b6c128__vs__{peer}.json"
                         for peer in ("raw3d_to2d_thin_b6c128", "raw3d_to2d_b8", "gravity_b8"))
        while not all(path.is_file() for path in required) or not gpu0_idle():
            if time.monotonic() > deadline:
                raise TimeoutError("GPU0/direct-match handoff exceeded wait bound")
            time.sleep(60)
        for row in rows:
            name = row["name"]
            config = Path(row["child_config"])
            run_dir = Path(row["child_run_dir"])
            if run_dir.exists():
                raise FileExistsError(f"child run already exists; inspect before retry: {run_dir}")
            if sha256(config) != row["child_config_sha256"]:
                raise ValueError(f"child config hash drift: {name}")
            plan = subprocess.run(
                [sys.executable, "-B", "-m", "training.v3", "run", "--config", str(config)],
                cwd=ROOT, capture_output=True, text=True,
            )
            if plan.returncode:
                raise RuntimeError(f"guarded plan failed: {name}: {plan.stderr[-2000:]}")
            state.update(status="running", active=name, active_config_sha256=sha256(config))
            atomic_json(STATE, state)
            log = BASE / "logs" / f"{name}_child2m.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    [sys.executable, "-u", "-B", "-m", "training.v3", "run",
                     "--config", str(config), "--execute",
                     "--max-train-positions", str(CHILD_POSITIONS)],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"V3 child run failed: {name}: {log}")
            manifest = read(run_dir / "run_manifest.json")
            positions = int((manifest.get("formal_loop_state") or {}).get("train_positions_consumed", -1))
            if manifest.get("stop_reason") != "max_train_positions" or positions != CHILD_POSITIONS:
                raise ValueError(f"child 2M commit missing: {name}: {positions}")
            pointer, commit = latest_commit(run_dir)
            receipt = {"name": name, "child_positions": positions,
                       "cumulative_exposure_positions": 3_000_000,
                       "generation": pointer["generation"],
                       "generation_commit_sha256": pointer["commit_sha256"],
                       "terminal_sha256": commit["checkpoint_sha256"],
                       "accepted_sha256": commit["accepted_model_sha256"],
                       "child_config_sha256": sha256(config)}
            atomic_json(BASE / "receipts" / f"{name}_child2m.json", receipt)
            state["completed"].append(receipt)
            state.update(active=None)
            atomic_json(STATE, state)
        state.update(status="complete", active=None)
        atomic_json(STATE, state)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(STATE, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-hours", type=float, default=24.0)
    args = parser.parse_args()
    rows = prepare()
    if not args.execute:
        print(json.dumps({"status": "prepared_semantic_fork", "rows": rows}, indent=2))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("physical GPU0 only; GPU1 is reserved by another experiment")
    execute(rows, args.wait_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
