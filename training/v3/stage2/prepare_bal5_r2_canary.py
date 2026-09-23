"""Freeze the BAL-5 R2 Classic-routing canary from the verified R1 donor."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

from connect4_core.rules import BAL5_R2_RULE_REGISTRY

from training.v3.config import V3Config, load_config
from training.v3.pipeline import lineage_config_hash


R1_RUN_ID = "bal5_r1_column_no_tail_warm_fp32lr1e4_seed271828"
R2_RUN_ID = "bal5_r2_column_no_tail_classic_canary_seed271828"
DONOR_ID = "candidate-g000062-s00015218-d00971827"
DONOR_SHA256 = "db015181297a2fde4dea93853276d73eec1385e2032f53fc752cea583b88600e"


def prepare_config(source: V3Config, *, donor: Path, run_dir: Path) -> V3Config:
    if source.run.run_id != R1_RUN_ID:
        raise ValueError("R2 canary requires the frozen R1 column_no_tail warm config")
    if donor.name != f"{DONOR_ID}.pt" or donor.parent.name != "accepted":
        raise ValueError("R2 donor must be the R1 last-accepted artifact")
    if not donor.is_file() or hashlib.sha256(donor.read_bytes()).hexdigest() != DONOR_SHA256:
        raise ValueError("R2 donor artifact is absent or has a different SHA-256")
    storage = source.runtime.storage
    if (
        storage.mode != "archive_ack_prune"
        or storage.hard_free_gib < 10
        or storage.soft_used_fraction != 0.9
        or storage.checkpoint_thinning_interval_generations != 2
    ):
        raise ValueError("R1 storage guardrails do not match the frozen R2 canary")
    ids = tuple(spec.rule_id for spec in BAL5_R2_RULE_REGISTRY.specs)
    config = replace(
        source,
        run=replace(
            source.run,
            run_id=R2_RUN_ID,
            run_dir=str(run_dir),
            resume=False,
            warm_start_mode="accepted_artifact_fresh_optimizer_replay_v1",
            warm_start_checkpoint=str(donor.resolve()),
            warm_start_checkpoint_sha256=DONOR_SHA256,
        ),
        selfplay=replace(
            source.selfplay,
            multi_rule_ids=ids,
            rule_registry_hash=BAL5_R2_RULE_REGISTRY.registry_hash,
        ),
    )
    return V3Config.from_dict(config.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    r1 = root / "training/runs/stage2/bal5/r1"
    source_path = r1 / "configs" / f"{R1_RUN_ID}.json"
    donor = r1 / "runs" / R1_RUN_ID / "accepted" / f"{DONOR_ID}.pt"
    output = root / "training/runs/stage2/bal5/r2_pre/configs" / f"{R2_RUN_ID}.json"
    run_dir = root / "training/runs/stage2/bal5/r2_pre/runs" / R2_RUN_ID
    if output.exists() or run_dir.exists():
        raise FileExistsError("R2 canary config/run already exists; refusing to overwrite")
    config = prepare_config(load_config(source_path), donor=donor, run_dir=run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(config.to_json() + "\n", encoding="utf-8")
    print(f"config={output}")
    print(f"lineage_config_hash={lineage_config_hash(config)}")
    print(f"donor_sha256={DONOR_SHA256}")
    print("gate_hard_regression_tolerance=0.05")
    print("bounded_train_positions=2000000 (set via --max-train-positions)")


if __name__ == "__main__":
    main()
