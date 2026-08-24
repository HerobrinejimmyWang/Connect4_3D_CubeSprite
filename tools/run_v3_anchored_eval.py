"""Plan and execute V3 anchored evaluation against frozen external opponents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.anchored_elo import (  # noqa: E402
    LegacyCheckpointPredictor,
    anchored_evaluation_plan,
    build_anchored_report,
    build_pressure_report,
    calibrate_anchor_scale,
    canonical_anchored_config_hash,
    load_anchored_config,
    load_match_batches,
    load_v3_artifact_predictor,
    verify_anchor_files,
    verify_opening_suite,
    write_anchor_scale,
    write_match_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen historical-anchor evaluation on the common V3 MCTS kernel"
    )
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("plan", help="print the bounded evaluation plan")
    commands.add_parser("verify", help="verify anchor and opening hashes")

    match = commands.add_parser("match", help="run one immutable disjoint match batch")
    match.add_argument("--profile", required=True)
    match.add_argument(
        "--milestone", required=True, help="calibration, early, middle, or final"
    )
    match.add_argument("--model-a", required=True, help="anchor:<id> or v3:<checkpoint>")
    match.add_argument("--model-b", required=True, help="anchor:<id> or v3:<checkpoint>")
    match.add_argument("--model-a-id", default=None)
    match.add_argument("--model-b-id", default=None)
    match.add_argument("--pair-start", type=int, required=True)
    match.add_argument("--pair-count", type=int, required=True)
    match.add_argument("--device", default="cpu")
    match.add_argument("--parallel-games", type=int, default=1)
    match.add_argument("--inference-batch-size", type=int, default=1)
    match.add_argument("--inference-batch-timeout-ms", type=float, default=1.0)
    match.add_argument(
        "--devices",
        default="",
        help="comma-separated replicated evaluator devices, e.g. cuda:0,cuda:1",
    )
    match.add_argument("--replicas-per-device", type=int, default=1)
    match.add_argument("--output", type=Path, required=True)

    calibrate = commands.add_parser(
        "calibrate", help="freeze one historical anchor scale from anchor-only batches"
    )
    calibrate.add_argument("--profile", required=True)
    calibrate.add_argument("--matches-dir", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)

    report = commands.add_parser("report", help="report one target on a frozen anchor scale")
    report.add_argument("--profile", required=True)
    report.add_argument("--target-id", required=True)
    report.add_argument("--matches-dir", type=Path, required=True)
    report.add_argument("--anchor-scale", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    pressure = commands.add_parser(
        "pressure-report", help="report one asymmetric target-vs-anchor pressure ruler"
    )
    pressure.add_argument("--profile", required=True)
    pressure.add_argument("--target-id", required=True)
    pressure.add_argument("--matches-dir", type=Path, required=True)
    pressure.add_argument("--output", type=Path, required=True)
    return parser


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _identity_for_anchor(config: Any, anchor_id: str) -> dict[str, Any]:
    anchor = config.anchor(anchor_id)
    path = (ROOT / anchor.path).resolve()
    return {
        "model_id": anchor.anchor_id,
        "label": anchor.label,
        "path": str(path),
        "checksum_sha256": anchor.checksum_sha256,
        "lineage": "external_legacy_anchor",
    }


def _resolve_model(
    spec: str, *, model_id: str | None, config: Any, device: str
) -> tuple[Any, dict[str, Any]]:
    kind, separator, value = spec.partition(":")
    if not separator or not value:
        raise ValueError("model spec must be anchor:<id> or v3:<checkpoint>")
    if kind == "anchor":
        if model_id is not None and model_id != value:
            raise ValueError("--model-*-id cannot rename a frozen anchor")
        identity = _identity_for_anchor(config, value)
        predictor = LegacyCheckpointPredictor(identity["path"], device=device)
        return predictor, identity
    if kind == "v3":
        predictor, identity = load_v3_artifact_predictor(value, device=device)
        if model_id is not None:
            identity["model_id"] = model_id
        if identity["model_id"] in {anchor.anchor_id for anchor in config.anchors}:
            raise ValueError("V3 target model_id collides with a frozen anchor")
        return predictor, identity
    raise ValueError("model spec must use the anchor or v3 prefix")


def _batch_paths(directory: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.rglob("*.match.json")))
    if not paths:
        raise ValueError(f"no *.match.json batches found under {directory}")
    return paths


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_anchored_config(args.config)
        config_hash = canonical_anchored_config_hash(config)
        if args.command == "plan":
            sys.stdout.write(_json(anchored_evaluation_plan(config)))
            return 0
        if args.command == "verify":
            anchors = verify_anchor_files(config, ROOT)
            openings = verify_opening_suite(config, ROOT)
            sys.stdout.write(
                _json(
                    {
                        "anchored_config_hash": config_hash,
                        "anchors": anchors,
                        "opening_manifest": {
                            "path": str((ROOT / config.openings.manifest_path).resolve()),
                            "checksum_sha256": config.openings.checksum_sha256,
                            "count": len(openings),
                        },
                        "verified": True,
                    }
                )
            )
            return 0
        if args.command == "match":
            verify_anchor_files(config, ROOT)
            openings = verify_opening_suite(config, ROOT)
            profile = config.profile(args.profile)
            if args.pair_start < 0 or args.pair_count < 1:
                raise ValueError("pair-start must be non-negative and pair-count positive")
            if (
                args.parallel_games < 1
                or args.inference_batch_size < 1
                or args.inference_batch_timeout_ms < 0.0
                or args.replicas_per_device < 1
            ):
                raise ValueError("evaluation parallel/batch settings are invalid")
            devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
            if args.devices and not devices:
                raise ValueError("replicated evaluation devices are invalid")
            replica_devices = tuple(
                device for _ in range(args.replicas_per_device) for device in devices
            )
            if replica_devices and (
                args.parallel_games != 1 or args.inference_batch_size != 1
            ):
                raise ValueError("replicated and central-batched modes cannot be combined")
            stop = args.pair_start + args.pair_count
            if stop > profile.max_pairs or stop > len(openings):
                raise ValueError("requested pair range exceeds the profile or opening suite")
            load_device = "cpu" if replica_devices else args.device
            predictor_a, identity_a = _resolve_model(
                args.model_a,
                model_id=args.model_a_id,
                config=config,
                device=load_device,
            )
            predictor_b, identity_b = _resolve_model(
                args.model_b,
                model_id=args.model_b_id,
                config=config,
                device=load_device,
            )
            if identity_a["model_id"] == identity_b["model_id"]:
                raise ValueError("a model cannot play itself in anchored evaluation")
            write_match_batch(
                args.output,
                config=config,
                profile=profile,
                openings=openings[args.pair_start:stop],
                opening_manifest_path=ROOT / config.openings.manifest_path,
                model_a=identity_a,
                model_b=identity_b,
                predictor_a=predictor_a,
                predictor_b=predictor_b,
                milestone=args.milestone,
                runtime={
                    "device": args.device,
                    "replica_devices": list(replica_devices),
                },
                parallel_games=args.parallel_games,
                inference_batch_size=args.inference_batch_size,
                inference_batch_timeout_s=args.inference_batch_timeout_ms / 1000.0,
                replica_devices=replica_devices,
            )
            sys.stdout.write(_json({"status": "complete", "output": str(args.output)}))
            return 0
        if args.command == "calibrate":
            batches = load_match_batches(
                _batch_paths(args.matches_dir), expected_config_hash=config_hash
            )
            ratings, batch_ids = calibrate_anchor_scale(
                config, batches, profile_id=args.profile
            )
            write_anchor_scale(
                args.output,
                config=config,
                profile_id=args.profile,
                ratings=ratings,
                source_batch_ids=batch_ids,
            )
            sys.stdout.write(_json({"ratings": ratings, "output": str(args.output)}))
            return 0
        if args.command == "report":
            batches = load_match_batches(
                _batch_paths(args.matches_dir), expected_config_hash=config_hash
            )
            scale = json.loads(args.anchor_scale.read_text(encoding="utf-8"))
            report = build_anchored_report(
                config,
                batches,
                profile_id=args.profile,
                target_model_id=args.target_id,
                anchor_scale=scale,
            )
            _write_report(args.output, report)
            sys.stdout.write(_json(report))
            return 0
        if args.command == "pressure-report":
            batches = load_match_batches(
                _batch_paths(args.matches_dir), expected_config_hash=config_hash
            )
            report = build_pressure_report(
                config,
                batches,
                profile_id=args.profile,
                target_model_id=args.target_id,
            )
            _write_report(args.output, report)
            sys.stdout.write(_json(report))
            return 0
        raise AssertionError(args.command)
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"anchored-eval error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
