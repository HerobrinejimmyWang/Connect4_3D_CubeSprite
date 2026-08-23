"""Summarize or compare immutable V3 replay policy-target artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.policy_target_quality import (  # noqa: E402
    compare_replay_policy_targets,
    summarize_replay_policy_targets,
)
from training.v3.anchored_elo import load_v3_artifact_predictor  # noqa: E402
from training.v3.evaluation import load_opening_manifest  # noqa: E402
from training.v3.policy_target_audit import generate_fixed_opening_targets  # noqa: E402
from training.v3.replay import (  # noqa: E402
    load_replay_shard,
    sha256_file,
    write_replay_shard,
)
from connect4_core.rules import DEFAULT_RULE_REGISTRY  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V3 256-vs-reference policy target audit")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--openings", type=Path, required=True)
    generate.add_argument("--position-start", type=int, default=0)
    generate.add_argument("--position-count", type=int, required=True)
    generate.add_argument("--search-sims", type=int, required=True)
    generate.add_argument("--audit-seed", type=int, required=True)
    generate.add_argument("--cpuct", type=float, default=1.5)
    generate.add_argument("--virtual-loss", type=float, default=1.0)
    generate.add_argument("--mcts-lanes", type=int, default=4)
    generate.add_argument("--root-noise-alpha", type=float, default=0.24)
    generate.add_argument("--root-noise-epsilon", type=float, default=0.06)
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--report", type=Path, default=None)
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


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        if args.position_start < 0 or args.position_count < 1:
            raise SystemExit("position-start must be non-negative and position-count positive")
        openings = load_opening_manifest(args.openings)
        end = args.position_start + args.position_count
        selected = openings[args.position_start:end]
        if len(selected) != args.position_count:
            raise SystemExit("requested fixed-position range exceeds the opening manifest")
        predictor, model_identity = load_v3_artifact_predictor(
            args.checkpoint, device=args.device
        )
        generated = generate_fixed_opening_targets(
            selected,
            predictor=predictor,
            search_sims=args.search_sims,
            audit_seed=args.audit_seed,
            position_start=args.position_start,
            cpuct=args.cpuct,
            virtual_loss=args.virtual_loss,
            mcts_lanes=args.mcts_lanes,
            root_noise_alpha=args.root_noise_alpha,
            root_noise_epsilon=args.root_noise_epsilon,
        )
        contract = {
            "checkpoint_sha256": model_identity["checksum_sha256"],
            "opening_manifest_sha256": sha256_file(args.openings),
            "position_start": args.position_start,
            "position_count": args.position_count,
            "search_sims": args.search_sims,
            "audit_seed": args.audit_seed,
            "cpuct": args.cpuct,
            "virtual_loss": args.virtual_loss,
            "mcts_lanes": args.mcts_lanes,
            "root_noise_alpha": args.root_noise_alpha,
            "root_noise_epsilon": args.root_noise_epsilon,
        }
        contract_hash = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest = write_replay_shard(
            args.output,
            generated.replay,
            {
                "run_id": args.run_id,
                "generation": 0,
                "producer_model_id": str(model_identity["model_id"]),
                "seed_range": {
                    "start": args.audit_seed,
                    "end": args.audit_seed + args.position_count - 1,
                },
                "results": {
                    "diagnostic_only": True,
                    "positions": args.position_count,
                    "inference_calls": generated.inference_calls,
                    "inference_positions": generated.inference_positions,
                    "max_inference_batch": generated.max_inference_batch,
                },
                "search_config": contract,
                "rule_registry_hash": DEFAULT_RULE_REGISTRY.registry_hash,
                "config_hash": contract_hash,
                "git_commit": _git_commit(),
                "diagnostic_only": True,
                "opening_manifest": str(args.openings.resolve()),
            },
        )
        payload = {
            "format": "v3_fixed_policy_target_generation_v1",
            "output": str(args.output.resolve()),
            "manifest": manifest,
            "model": model_identity,
            "metrics": summarize_replay_policy_targets(
                generated.replay, expected_simulations=args.search_sims
            ),
            "automatic_budget_change": False,
        }
        _emit(payload, args.report)
        return 0
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
