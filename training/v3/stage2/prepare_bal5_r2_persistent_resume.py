"""Prepare a hash-preserving BAL-5 R2 runtime resume after a clean drain."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from training.v3.config import load_config
from training.v3.layout import RunLayout
from training.v3.pipeline import _load_latest_generation_commit, lineage_config_hash

from .prepare_bal5_r2_canary import R2_RUN_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    config_dir = root / "training/runs/stage2/bal5/r2_pre/configs"
    source = config_dir / f"{R2_RUN_ID}.json"
    output = config_dir / f"{R2_RUN_ID}_resume_persistent.json"
    if output.exists():
        raise FileExistsError("optimized resume config already exists")
    old = load_config(source)
    if old.run.run_id != R2_RUN_ID or old.run.resume:
        raise ValueError("source must be the original, non-resume BAL-5 R2 canary config")
    run_dir = Path(old.run.run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_hash = lineage_config_hash(old)
    if (
        manifest.get("config_hash") != old_hash
        or manifest.get("status") != "stopped_at_safe_boundary"
        or manifest.get("stop_reason") != "drained_after_signal_15"
    ):
        raise ValueError("R2 canary is not cleanly drained at its original lineage hash")
    latest = _load_latest_generation_commit(
        RunLayout.from_root(run_dir), expected_hash=old_hash, allow_missing=False
    )
    if latest is None or int(latest[0]["generation"]) < 1:
        raise ValueError("optimized resume requires two committed canary generations")
    new = replace(
        old,
        run=replace(old.run, resume=True),
        runtime=replace(old.runtime, multi_rule_actor_pool_mode="persistent"),
    )
    if lineage_config_hash(new) != old_hash:
        raise ValueError("persistent pool changed the semantic lineage hash")
    output.write_text(new.to_json(), encoding="utf-8")
    print(f"resume_config={output}")
    print(f"lineage_config_hash={old_hash}")
    print(f"resume_after_generation={latest[0]['generation']}")
    print("multi_rule_actor_pool_mode=persistent")


if __name__ == "__main__":
    main()
