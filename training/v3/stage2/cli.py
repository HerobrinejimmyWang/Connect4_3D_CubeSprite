from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .calibration import calibrate_architecture_matrix, write_architecture_matrix
from .data import audit_trajectory, freeze_regime_datasets
from .design import build_experiment_design
from .elo import verify_stage2_elo_protocol
from .offline import evaluate_checkpoint, train_offline
from .selfplay import generate_stage2b_configs
from .summary import summarize_reports


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m training.v3.stage2")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data")
    audit.add_argument("--source-dir", required=True, type=Path)
    audit.add_argument("--metrics", required=True, type=Path)
    audit.add_argument("--mixed-source-dir", required=True, type=Path)
    audit.add_argument("--mixed-metrics", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument(
        "--standard-lineage-prefix",
        default="stage1_b10c256_relative_role_guard_coldstart",
    )
    audit.add_argument(
        "--mixed-lineage-prefix",
        default="stage1_b10c256_g487_mixed_opening_temp_position_balanced",
    )
    audit.add_argument("--minimum-fraction", type=float, default=0.15)

    freeze = commands.add_parser("freeze-data")
    freeze.add_argument("--audit", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--train-positions", type=int, default=1_000_000)
    freeze.add_argument("--validation-positions", type=int, default=50_000)
    freeze.add_argument("--seed", type=int, default=271828)

    calibrate = commands.add_parser("calibrate-models")
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--anchor-channels", type=int, default=128)
    calibrate.add_argument("--anchor-blocks", type=int, default=6)
    calibrate.add_argument("--tolerance", type=float, default=0.05)
    calibrate.add_argument("--profile-latency", action="store_true")

    design = commands.add_parser("design-matrix")
    design.add_argument("--output", required=True, type=Path)
    design.add_argument("--tolerance", type=float, default=0.05)

    elo = commands.add_parser("verify-elo")
    elo.add_argument("--protocol", required=True, type=Path)
    elo.add_argument("--repo-root", type=Path, default=Path.cwd())

    train = commands.add_parser("train")
    train.add_argument("--config", required=True, type=Path)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--reports", required=True, nargs="+", type=Path)
    summarize.add_argument("--output", required=True, type=Path)

    stage2b = commands.add_parser("generate-stage2b")
    stage2b.add_argument("--base-config", required=True, type=Path)
    stage2b.add_argument("--architecture-matrix", required=True, type=Path)
    stage2b.add_argument("--finalists", required=True, type=Path)
    stage2b.add_argument("--output-dir", required=True, type=Path)
    stage2b.add_argument("--seeds", nargs=2, type=int, default=(271828, 314159))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit-data":
        result = audit_trajectory(
            args.source_dir,
            args.metrics,
            mixed_source_dir=args.mixed_source_dir,
            mixed_metrics_path=args.mixed_metrics,
            standard_lineage_prefix=args.standard_lineage_prefix,
            mixed_lineage_prefix=args.mixed_lineage_prefix,
            minimum_fraction=args.minimum_fraction,
        )
        _write_json(args.output, result)
    elif args.command == "freeze-data":
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        result = freeze_regime_datasets(
            audit,
            args.output_dir,
            train_positions=args.train_positions,
            validation_positions=args.validation_positions,
            seed=args.seed,
        )
    elif args.command == "calibrate-models":
        result = calibrate_architecture_matrix(
            anchor_channels=args.anchor_channels,
            anchor_blocks=args.anchor_blocks,
            tolerance=args.tolerance,
            profile_latency=args.profile_latency,
        )
        write_architecture_matrix(args.output, result)
    elif args.command == "design-matrix":
        result = build_experiment_design(tolerance=args.tolerance)
        _write_json(args.output, result)
    elif args.command == "verify-elo":
        result = verify_stage2_elo_protocol(args.protocol, repo_root=args.repo_root)
    elif args.command == "train":
        result = train_offline(args.config)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.config, checkpoint_path=args.checkpoint)
        _write_json(args.output, result)
    elif args.command == "summarize":
        result = summarize_reports(args.reports)
        _write_json(args.output, result)
    else:
        result = generate_stage2b_configs(
            base_config_path=args.base_config,
            architecture_matrix_path=args.architecture_matrix,
            finalists_path=args.finalists,
            output_dir=args.output_dir,
            seeds=args.seeds,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
