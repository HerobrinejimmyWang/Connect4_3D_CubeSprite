"""Run isolated full-branch B6C128 self-play after its formal CPU gate passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stage2_fla_b6_selfplay_gate import validate_cpu
from stage2_fla_queue_trim_resume import verify_completion
from stage2_fla_selfplay_queue import ROOT, SEED, TEMPLATE, atomic_json, sha256, validate_donor
from training.v3.config import V3Config

NAME = "raw3d_to2d_full_b6c128"
SOURCE = ROOT / "training/runs/stage2/fla/b6_raw_full_2m"
CONFIG = SOURCE / "configs" / f"{NAME}__standard_late__seed{SEED}.json"
MODEL = SOURCE / "runs" / NAME / "standard_late" / f"seed{SEED}" / "model.pt"
REPORT = MODEL.parent / "report.json"
RUN_ID = f"fla2m_r2_{NAME}_warm_seed{SEED}"
RUN_DIR = ROOT / "training/runs/stage2/fla/selfplay/r2_raw_b6_full" / RUN_ID
RUN_CONFIG = RUN_DIR.parent / "configs" / f"{RUN_ID}.json"
STATE = RUN_DIR.parent / "controller_state.json"
POSITIONS = 1_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-result", type=Path, required=True)
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model_config, donor_sha = validate_donor(CONFIG, MODEL, REPORT)
    cpu = validate_cpu(args.cpu_result, args.cpu_summary, args.reference_summary,
                       config_path=CONFIG, model_path=MODEL, variant=NAME)
    if not cpu["mean_s"] < 3.1:
        raise ValueError(f"full B6 mean {cpu['mean_s']:.3f}s does not meet the strict 3.1s user screen")
    template = json.loads((ROOT / TEMPLATE).read_text(encoding="utf-8"))
    if template["selfplay"]["rule_id"] != "classic":
        raise ValueError("Classic rule required")
    template["run"] = {
        "run_id": RUN_ID, "seed": SEED, "run_dir": str(RUN_DIR), "resume": False,
        "warm_start_checkpoint": str(MODEL),
        "warm_start_checkpoint_sha256": donor_sha,
        "warm_start_mode": "model_only_fresh_optimizer_replay_v1",
    }
    template["model"] = model_config
    config = V3Config.from_dict(template)
    if STATE.exists() or RUN_CONFIG.exists() or RUN_DIR.exists():
        raise FileExistsError("B6 full-branch self-play already started; inspect state before retry")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "run_id": RUN_ID, "donor_sha256": donor_sha,
                          "cpu": cpu, "positions": POSITIONS}))
        return 0
    RUN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    RUN_CONFIG.write_text(config.to_json(), encoding="utf-8")
    plan = subprocess.run(
        [sys.executable, "-B", "-m", "training.v3", "run", "--config", str(RUN_CONFIG)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if plan.returncode:
        raise RuntimeError(f"V3 guarded plan failed: {plan.stderr[-2000:]}")
    atomic_json(STATE, {"status": "running", "run_id": RUN_ID,
                        "config_sha256": sha256(RUN_CONFIG), "donor_sha256": donor_sha,
                        "cpu_evidence": cpu, "positions": POSITIONS})
    log = RUN_DIR.parent / "selfplay.log"
    try:
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [sys.executable, "-u", "-B", "-m", "training.v3", "run",
                 "--config", str(RUN_CONFIG), "--execute", "--max-train-positions", str(POSITIONS)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"V3 self-play exited {result.returncode}; see {log}")
        receipt = verify_completion(RUN_DIR)
        atomic_json(STATE, {"status": "complete", "run_id": RUN_ID,
                            "config_sha256": sha256(RUN_CONFIG), "donor_sha256": donor_sha,
                            "cpu_evidence": cpu, "positions": POSITIONS, **receipt})
        return 0
    except Exception as exc:
        atomic_json(STATE, {"status": "failed", "run_id": RUN_ID,
                            "error": f"{type(exc).__name__}: {exc}",
                            "config_sha256": sha256(RUN_CONFIG), "donor_sha256": donor_sha})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
