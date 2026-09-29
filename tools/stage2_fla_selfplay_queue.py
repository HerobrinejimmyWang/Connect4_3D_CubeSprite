"""Run a fail-closed, serial 1M Classic self-play screen for FLA candidates.

All warm starts are Stage 2 offline standard_late/seed271828/1M artifacts.
The queue waits for newly trained donors, then validates each one before use.
It never resumes, prunes, or continues a run past the 1M bound automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.config import V3Config  # noqa: E402


QUEUE_ROOT = Path("training/runs/stage2/fla/selfplay/r1")
TEMPLATE = Path(
    "training/runs/stage2/bal5/r1/configs/"
    "bal5_r1_column_no_tail_cold_fp32lr1e4_seed271828.json"
)
BAL2 = Path("training/runs/stage2/round3/balance/bal2")
NEW = Path("training/runs/stage2/fla/efficiency_6m")
POSITIONS = 1_000_000
SEED = 271828
ROWS = (
    ("gravity_b8", BAL2, "balance_gravity_control"),
    ("column_2d_b8c192", NEW, "column_2d_b8c192"),
    ("raw3d_to2d_b8", BAL2, "balance_raw3d_to2d"),
    ("raw3d_to2d_thin_b8c192", NEW, "raw3d_to2d_thin_b8c192"),
    ("column3d_v2_b8", BAL2, "balance_column3d_fusion_v2"),
    ("column3d_v2_thin_b8c192", NEW, "column3d_v2_thin_b8c192"),
    ("multiview3d_b8", BAL2, "balance_multiview3d_fusion"),
    ("winning3d_b8", BAL2, "balance_winning3d_fusion"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def donor_paths(source: Path, variant: str) -> tuple[Path, Path, Path]:
    config = ROOT / source / "configs" / f"{variant}__standard_late__seed{SEED}.json"
    run = ROOT / source / "runs" / variant / "standard_late" / f"seed{SEED}"
    return config, run / "model.pt", run / "report.json"


def validate_donor(config: Path, model: Path, report: Path) -> tuple[dict, str]:
    if not all(path.is_file() for path in (config, model, report)):
        raise FileNotFoundError(f"donor files incomplete: {config}, {model}, {report}")
    raw = json.loads(config.read_text(encoding="utf-8"))
    result = json.loads(report.read_text(encoding="utf-8"))
    if int(raw["target_positions"]) != POSITIONS or int(raw["seed"]) != SEED:
        raise ValueError(f"donor does not match 1M/seed{SEED}: {config}")
    if not result.get("training_execution", {}).get("target_positions_reached"):
        raise ValueError(f"donor report does not confirm 1M completion: {report}")
    digest = sha256(model)
    payload = torch.load(model, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if payload.get("format") != "connect4-v3-model" or payload.get("format_version") != 1:
        raise ValueError(f"invalid V3 donor model: {model}")
    if payload.get("model_config") != raw["model"]:
        raise ValueError(f"donor model config mismatch: {model}")
    if (
        metadata.get("lineage") != "v3_stage2_offline"
        or metadata.get("train_regime") != "standard_late"
        or metadata.get("train_positions") != POSITIONS
        or metadata.get("seed") != SEED
    ):
        raise ValueError(f"donor lineage mismatch: {model}")
    if result.get("model_artifact", {}).get("sha256") != digest:
        raise ValueError(f"donor report hash mismatch: {model}")
    return raw["model"], digest


def prepare_config(template: dict, name: str, model: dict, donor: Path, digest: str) -> tuple[Path, Path, str]:
    run_id = f"fla6m_r1_{name}_warm_seed{SEED}"
    run_dir = ROOT / QUEUE_ROOT / "runs" / run_id
    raw = json.loads(json.dumps(template))
    raw["run"] = {
        "run_id": run_id,
        "seed": SEED,
        "run_dir": str(run_dir),
        "resume": False,
        "warm_start_checkpoint": str(donor),
        "warm_start_checkpoint_sha256": digest,
        "warm_start_mode": "model_only_fresh_optimizer_replay_v1",
    }
    raw["model"] = model
    config = V3Config.from_dict(raw)
    path = ROOT / QUEUE_ROOT / "configs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_json(), encoding="utf-8")
    return path, run_dir, sha256(path)


def run_is_complete(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = manifest.get("formal_loop_state") or {}
    return (
        manifest.get("stop_reason") == "max_train_positions"
        and int(manifest.get("max_train_positions", -1)) == POSITIONS
        and int(state.get("train_positions_consumed", -1)) == POSITIONS
    )


def offline_workers_running() -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except (OSError, PermissionError):
            continue
        if b"training.v3.stage2 train" in args and b"efficiency_6m/configs/" in args:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-minutes", type=int, default=120)
    args = parser.parse_args()
    if args.wait_minutes < 0:
        parser.error("--wait-minutes must be non-negative")
    output = ROOT / QUEUE_ROOT
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "queue_state.json"
    if state_path.exists():
        raise RuntimeError(f"queue state already exists; refusing a second controller: {state_path}")
    template = json.loads((ROOT / TEMPLATE).read_text(encoding="utf-8"))
    if template["selfplay"]["rule_id"] != "classic":
        raise ValueError("FLA queue requires Classic self-play")
    if template["runtime"]["storage"]["mode"] != "archive_ack_prune":
        raise ValueError("FLA queue requires receipt-gated archival storage")
    if template["runtime"]["storage"]["hard_free_gib"] < 10:
        raise ValueError("FLA queue requires at least 10 GiB reserve")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "rows": [row[0] for row in ROWS]}, indent=2))
        return 0
    status = {"schema": "connect4-stage2-fla-selfplay-queue-v1", "status": "waiting_donors", "completed": [], "active": None}
    atomic_json(state_path, status)
    deadline = time.monotonic() + args.wait_minutes * 60
    try:
        # The queue uses both GPUs. Do not overlap it with the new offline
        # donors, even though archived BAL-2 donors are already available.
        new_rows = [(name, source, variant) for name, source, variant in ROWS if source == NEW]
        while True:
            missing = [
                name for name, source, variant in new_rows
                if not all(path.is_file() for path in donor_paths(source, variant))
            ]
            if not missing and not offline_workers_running():
                for name, source, variant in new_rows:
                    validate_donor(*donor_paths(source, variant))
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"offline prerequisites not ready within wait bound: missing={missing}, "
                    f"workers_running={offline_workers_running()}"
                )
            time.sleep(60)
        for name, source, variant in ROWS:
            config_source, donor, report = donor_paths(source, variant)
            while not all(path.is_file() for path in (config_source, donor, report)):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"donor not ready within wait bound: {name}")
                time.sleep(60)
            model, digest = validate_donor(config_source, donor, report)
            config_path, run_dir, config_digest = prepare_config(template, name, model, donor, digest)
            if run_dir.exists():
                raise RuntimeError(f"run directory already exists; refusing implicit resume: {run_dir}")
            status.update(status="preflight", active=name)
            atomic_json(state_path, status)
            plan = subprocess.run(
                [sys.executable, "-B", "-m", "training.v3", "run", "--config", str(config_path)],
                cwd=ROOT, capture_output=True, text=True,
            )
            if plan.returncode:
                raise RuntimeError(f"V3 guarded plan failed for {name}: {plan.stderr[-2000:]}")
            status.update(status="running", active=name, config_sha256=config_digest, donor_sha256=digest)
            atomic_json(state_path, status)
            log_path = output / "logs" / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, "-u", "-B", "-m", "training.v3", "run", "--config", str(config_path),
                     "--execute", "--max-train-positions", str(POSITIONS)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            if result.returncode or not run_is_complete(run_dir):
                raise RuntimeError(f"{name} stopped without verified 1M completion (exit {result.returncode}); see {log_path}")
            status["completed"].append({"name": name, "run_dir": str(run_dir), "config_sha256": config_digest, "donor_sha256": digest})
            status.update(status="waiting_donors", active=None)
            atomic_json(state_path, status)
        status.update(status="complete", active=None, completed_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, status)
        return 0
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(state_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
