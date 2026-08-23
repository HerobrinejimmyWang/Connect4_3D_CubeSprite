"""Create, verify, ingest, and prune receipt-gated V3 archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.archive import (  # noqa: E402
    create_archive_bundle,
    execute_prune,
    ingest_archive_receipt,
    plan_prune,
    verify_archive_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verified V3 run archive management")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--run-dir", type=Path, required=True)
    create.add_argument("--bundle-target-gib", type=float, default=4.0)

    verify = commands.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--extract-to", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)

    ingest = commands.add_parser("ingest-receipt")
    ingest.add_argument("--run-dir", type=Path, required=True)
    ingest.add_argument("--receipt", type=Path, required=True)

    prune = commands.add_parser("prune")
    prune.add_argument("--run-dir", type=Path, required=True)
    prune.add_argument("--execute", action="store_true")
    return parser


def _emit(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            if args.bundle_target_gib <= 0:
                raise ValueError("bundle-target-gib must be positive")
            _emit(
                create_archive_bundle(
                    args.run_dir,
                    bundle_target_bytes=int(args.bundle_target_gib * 1024**3),
                )
            )
            return 0
        if args.command == "verify":
            receipt = verify_archive_bundle(
                args.archive,
                args.manifest,
                extract_to=args.extract_to,
                receipt_path=args.receipt,
            )
            _emit(
                {
                    "status": "verified",
                    "receipt": str(args.receipt.resolve()),
                    "receipt_id": receipt["receipt_id"],
                    "entries": len(receipt["entries"]),
                }
            )
            return 0
        if args.command == "ingest-receipt":
            target = ingest_archive_receipt(args.run_dir, args.receipt)
            _emit({"status": "ingested", "receipt": str(target)})
            return 0
        if args.command == "prune":
            if args.execute:
                _emit(execute_prune(args.run_dir))
            else:
                _emit(plan_prune(args.run_dir))
            return 0
        raise AssertionError(args.command)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"v3-archive error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
