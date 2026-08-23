"""Command-line interface for the V3.1 training foundation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import resolve_config
from .preflight import PreflightError, run_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.v3",
        description="V3.1 isolated training foundation (formal training remains disabled).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("print-config", "validate and print the fully resolved strict JSON config"),
        ("smoke", "run the CPU end-to-end smoke workflow"),
        ("run", "validate the guarded formal-run entry point without training"),
        (
            "validate-local",
            "validate bounded local P6/P7 contracts without starting training",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--run-dir", type=Path, default=None)
        command.add_argument(
            "--resume",
            action="store_const",
            const=True,
            default=None,
            help="resume the checkpoint in the resolved run directory",
        )
        command.add_argument("--device", default=None)
        if name == "validate-local":
            command.add_argument(
                "--replay-dir",
                type=Path,
                default=None,
                help="optional locked Replay V2 dataset directory to checksum and inspect",
            )
            command.add_argument(
                "--minimum-replay-games",
                type=int,
                default=None,
                help="explicit P6 screening floor; requires --minimum-replay-samples",
            )
            command.add_argument(
                "--minimum-replay-samples",
                type=int,
                default=None,
                help="explicit P6 screening floor; requires --minimum-replay-games",
            )
            command.add_argument(
                "--output",
                type=Path,
                default=None,
                help="optional JSON report path; stdout is always emitted",
            )
            command.add_argument(
                "--write-ablation-configs",
                type=Path,
                default=None,
                help="optionally materialize the five explicit P6 configs",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(
            args.config,
            run_dir=args.run_dir,
            resume=args.resume,
            device=args.device,
        )
        # Config inspection and the guarded ``run`` entry do not allocate the
        # configured accelerator. Smoke validates the actual target device.
        preflight_device = config.runtime.device if args.command == "smoke" else "cpu"
        preflight = run_preflight(preflight_device)
        if args.command == "print-config":
            sys.stdout.write(config.to_json())
            return 0

        if args.command == "validate-local":
            from .local_validation import (
                build_local_validation_report,
                write_local_validation_report,
                write_p6_ablation_configs,
            )

            if args.write_ablation_configs is not None:
                write_p6_ablation_configs(config, args.write_ablation_configs)
            result = build_local_validation_report(
                config,
                preflight,
                replay_dir=args.replay_dir,
                minimum_replay_games=args.minimum_replay_games,
                minimum_replay_samples=args.minimum_replay_samples,
            )
            if args.output is not None:
                write_local_validation_report(args.output, result)
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            return 0

        # Importing the pipeline imports PyTorch; keep it behind preflight so a
        # missing dependency produces the concise message above.
        from .pipeline import formal_run_status, run_smoke

        result = run_smoke(config) if args.command == "smoke" else formal_run_status(config)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return 0
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, RuntimeError, PreflightError) as exc:
        sys.stderr.write(f"V3 error: {exc}\n")
        return 1


__all__ = ["main"]
