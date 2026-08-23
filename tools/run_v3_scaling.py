"""Plan and validate the two V3 scaling tracks without enabling formal runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.anchored_elo import load_v3_artifact_predictor  # noqa: E402
from training.v3.cross_scale import (  # noqa: E402
    CrossScaleSamplePlanner,
    MixedReplayPlanner,
    TransferSchedule,
    TransferStage,
    build_cross_scale_bundle,
    load_bundle_source_spec,
    load_donor_qualification,
    validate_cross_scale_bundle,
    write_donor_qualification,
)
from training.v3.evaluation import load_opening_manifest  # noqa: E402
from training.v3.scaling_experiment import (  # noqa: E402
    build_scaling_experiment_plan,
    load_scaling_experiment,
)
from training.v3.scale_screen import load_scale_screen  # noqa: E402


DEFAULT_SPEC = ROOT / "training" / "v3" / "configs" / "dual_track_scaling_v1.json"
DEFAULT_SCREEN_SPEC = ROOT / "training" / "v3" / "configs" / "stage1_scale_screen_v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V3 independent-scaling and replay-transfer foundation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print both complete plan-only scaling flows")
    plan.add_argument("--spec", type=Path, default=DEFAULT_SPEC)

    screen = commands.add_parser(
        "screen-plan", help="validate and print the staged B4/B6/B8 scale-screen contract"
    )
    screen.add_argument("--spec", type=Path, default=DEFAULT_SCREEN_SPEC)

    build = commands.add_parser(
        "build-bundle", help="build one immutable authenticated replay-transfer bundle"
    )
    build.add_argument("--sources", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-bundle", help="verify all bundle hashes and replay triplets")
    verify.add_argument("--bundle", type=Path, required=True)

    sample = commands.add_parser(
        "sample-plan", help="inspect deterministic donor/own selection at one online position"
    )
    sample.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    sample.add_argument("--bundle", type=Path, default=None)
    sample.add_argument("--donor-size", type=int, default=None)
    sample.add_argument("--own-size", type=int, required=True)
    sample.add_argument("--own-positions-generated", type=int, required=True)
    sample.add_argument("--start-cursor", type=int, default=0)
    sample.add_argument("--count", type=int, default=10000)
    sample.add_argument("--seed", type=int, default=314159)

    qualify = commands.add_parser(
        "qualify", help="run one immutable prefix qualification against the frozen donor"
    )
    qualify.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    qualify.add_argument("--bundle", type=Path, required=True)
    qualify.add_argument("--candidate", type=Path, required=True)
    qualify.add_argument("--donor", type=Path, required=True)
    qualify.add_argument("--pair-count", type=int, required=True)
    qualify.add_argument("--device", default="cpu")
    qualify.add_argument("--parallel-games", type=int, default=1)
    qualify.add_argument("--inference-batch-size", type=int, default=1)
    qualify.add_argument("--inference-batch-timeout-ms", type=float, default=1.0)
    qualify.add_argument("--output", type=Path, required=True)

    verify_qualification = commands.add_parser(
        "verify-qualification", help="verify an immutable donor qualification artifact"
    )
    verify_qualification.add_argument("--artifact", type=Path, required=True)
    return parser


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _schedule(spec: Mapping[str, Any]) -> TransferSchedule:
    return TransferSchedule(
        tuple(
            TransferStage(
                start_own_positions=int(row["start_own_positions"]),
                donor_fraction=float(row["donor_fraction"]),
            )
            for row in spec["online_transfer_schedule_resolved"]
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            spec = load_scaling_experiment(args.spec)
            sys.stdout.write(_json(build_scaling_experiment_plan(spec, root=ROOT)))
            return 0
        if args.command == "screen-plan":
            sys.stdout.write(_json(load_scale_screen(args.spec, root=ROOT)))
            return 0
        if args.command == "build-bundle":
            donor_run_id, donor_model_id, rule_id, strata = load_bundle_source_spec(args.sources)
            output = build_cross_scale_bundle(
                args.output,
                donor_run_id=donor_run_id,
                qualification_donor_model_id=donor_model_id,
                rule_id=rule_id,
                strata=strata,
            )
            manifest = validate_cross_scale_bundle(output)
            sys.stdout.write(
                _json(
                    {
                        "status": "complete",
                        "bundle": str(output),
                        "bundle_content_sha256": manifest["bundle_content_sha256"],
                        "totals": manifest["totals"],
                    }
                )
            )
            return 0
        if args.command == "verify-bundle":
            manifest = validate_cross_scale_bundle(args.bundle)
            sys.stdout.write(
                _json(
                    {
                        "verified": True,
                        "bundle_content_sha256": manifest["bundle_content_sha256"],
                        "donor_run_id": manifest["donor_run_id"],
                        "qualification_donor_model_id": manifest[
                            "qualification_donor_model_id"
                        ],
                        "strata": manifest["strata"],
                        "totals": manifest["totals"],
                    }
                )
            )
            return 0
        if args.command == "sample-plan":
            spec = load_scaling_experiment(args.spec)
            if (args.bundle is None) == (args.donor_size is None):
                raise ValueError("sample-plan requires exactly one of --bundle or --donor-size")
            bundle = validate_cross_scale_bundle(args.bundle) if args.bundle is not None else None
            if bundle is None:
                planner: Any = MixedReplayPlanner(
                    donor_size=args.donor_size,
                    own_size=args.own_size,
                    schedule=_schedule(spec),
                    seed=args.seed,
                )
            else:
                planner = CrossScaleSamplePlanner(
                    donor_stratum_sizes={
                        name: int(bundle["strata"][name]["positions"])
                        for name in bundle["strata"]
                    },
                    donor_stratum_weights=dict(
                        zip(
                            spec["production_track"]["bundle_strata"],
                            spec["production_track"]["bundle_sampling_weights"],
                            strict=True,
                        )
                    ),
                    own_size=args.own_size,
                    schedule=_schedule(spec),
                    seed=args.seed,
                )
            keys = planner.batch(
                start_cursor=args.start_cursor,
                count=args.count,
                own_positions_generated=args.own_positions_generated,
            )
            donor_count = sum(key.origin == "donor" for key in keys)
            configured_fraction = _schedule(spec).donor_fraction_for(
                args.own_positions_generated
            )
            stratum_counts = {
                name: sum(getattr(key, "stratum", None) == name for key in keys)
                for name in ("early", "middle", "late", "strong")
            }
            sys.stdout.write(
                _json(
                    {
                        "count": len(keys),
                        "own_positions_generated": args.own_positions_generated,
                        "configured_donor_fraction": configured_fraction,
                        "selected_donor": donor_count,
                        "selected_own": len(keys) - donor_count,
                        "observed_donor_fraction": donor_count / max(len(keys), 1),
                        "selected_donor_strata": stratum_counts if bundle is not None else None,
                        "sample_cursor_end": args.start_cursor + len(keys),
                    }
                )
            )
            return 0
        if args.command == "qualify":
            if (
                args.parallel_games < 1
                or args.inference_batch_size < 1
                or args.inference_batch_timeout_ms < 0.0
            ):
                raise ValueError("evaluation parallel/batch settings are invalid")
            spec = load_scaling_experiment(args.spec)
            qualification = spec["production_track"]["qualification"]
            valid_counts = range(
                qualification["initial_pairs"],
                qualification["max_pairs"] + 1,
                qualification["pair_increment"],
            )
            if args.pair_count not in valid_counts:
                raise ValueError("pair-count must follow the frozen initial/increment/max schedule")
            opening_path = (ROOT / qualification["opening_manifest"]).resolve()
            openings = load_opening_manifest(opening_path)
            if args.pair_count > len(openings):
                raise ValueError("pair-count exceeds the frozen opening manifest")
            candidate_predictor, candidate_identity = load_v3_artifact_predictor(
                args.candidate, device=args.device
            )
            donor_predictor, donor_identity = load_v3_artifact_predictor(
                args.donor, device=args.device
            )
            write_donor_qualification(
                args.output,
                bundle_dir=args.bundle,
                opening_manifest_path=opening_path,
                openings=openings[: args.pair_count],
                candidate_identity=candidate_identity,
                donor_identity=donor_identity,
                candidate_predictor=candidate_predictor,
                donor_predictor=donor_predictor,
                search_sims=qualification["search_sims"],
                cpuct=qualification["cpuct"],
                confidence=qualification["confidence"],
                bootstrap_samples=qualification["bootstrap_samples"],
                role_floor=qualification["role_floor"],
                accept_threshold=qualification["accept_threshold"],
                parallel_games=args.parallel_games,
                inference_batch_size=args.inference_batch_size,
                inference_batch_timeout_s=args.inference_batch_timeout_ms / 1000.0,
            )
            evidence = load_donor_qualification(args.output)
            sys.stdout.write(
                _json(
                    {
                        "status": "complete",
                        "output": str(args.output),
                        "qualification_passed": evidence["qualification_passed"],
                        "decision": evidence["decision"],
                        "automatic_promotion": False,
                    }
                )
            )
            return 0
        if args.command == "verify-qualification":
            evidence = load_donor_qualification(args.artifact)
            sys.stdout.write(
                _json(
                    {
                        "verified": True,
                        "content_sha256": evidence["content_sha256"],
                        "qualification_passed": evidence["qualification_passed"],
                        "automatic_promotion": evidence["automatic_promotion"],
                    }
                )
            )
            return 0
        raise AssertionError(args.command)
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        sys.stderr.write(f"v3-scaling error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
