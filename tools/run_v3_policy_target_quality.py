"""Summarize or compare immutable V3 replay policy-target artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.policy_target_quality import (  # noqa: E402
    compare_replay_policy_targets,
    summarize_replay_policy_targets,
)
from training.v3.replay import load_replay_shard  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V3 256-vs-reference policy target audit")
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--replay", type=Path, required=True)
    summary.add_argument("--expected-sims", type=int, required=True)
    summary.add_argument("--output", type=Path, default=None)
    compare = commands.add_parser("compare")
    compare.add_argument("--primary", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--primary-sims", type=int, default=256)
    compare.add_argument("--reference-sims", type=int, default=512)
    compare.add_argument("--output", type=Path, default=None)
    return parser


def _emit(payload: dict, output: Path | None) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite policy-target audit: {output}")
    output.write_text(encoded, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        replay, manifest = load_replay_shard(args.replay)
        payload = {
            "format": "v3_policy_target_quality_v1",
            "mode": "summary",
            "replay": str(args.replay.resolve()),
            "replay_checksum_sha256": manifest["checksum_sha256"],
            "search_sims": args.expected_sims,
            "metrics": summarize_replay_policy_targets(
                replay, expected_simulations=args.expected_sims
            ),
            "automatic_budget_change": False,
        }
        _emit(payload, args.output)
        return 0

    primary, primary_manifest = load_replay_shard(args.primary)
    reference, reference_manifest = load_replay_shard(args.reference)
    payload = {
        "format": "v3_policy_target_quality_v1",
        "mode": "paired_comparison",
        "primary": {
            "replay": str(args.primary.resolve()),
            "checksum_sha256": primary_manifest["checksum_sha256"],
            "search_sims": args.primary_sims,
            "summary": summarize_replay_policy_targets(
                primary, expected_simulations=args.primary_sims
            ),
        },
        "reference": {
            "replay": str(args.reference.resolve()),
            "checksum_sha256": reference_manifest["checksum_sha256"],
            "search_sims": args.reference_sims,
            "summary": summarize_replay_policy_targets(
                reference, expected_simulations=args.reference_sims
            ),
        },
        "paired_metrics": compare_replay_policy_targets(primary, reference),
        "automatic_budget_change": False,
        "interpretation": "Compare against repeated-primary variability and heldout policy/paired-strength evidence before changing the search budget.",
    }
    _emit(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
