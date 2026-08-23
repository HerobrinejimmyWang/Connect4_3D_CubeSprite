"""Pull verified V3 archive increments from a cloud host and acknowledge them."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.archive import verify_archive_bundle  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verified cloud-to-local V3 run sync")
    parser.add_argument("--remote", default="connect4_gpu_2608")
    parser.add_argument("--remote-repo", default="/root/autodl-tmp/Connect4_3D_game_refactor")
    parser.add_argument("--remote-python", default="/root/miniconda3/bin/python")
    parser.add_argument("--run-dir", required=True, help="run path relative to remote repo")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--bundle-target-gib", type=float, default=4.0)
    parser.add_argument("--max-bundles", type=int, default=1)
    parser.add_argument("--prune", action="store_true")
    return parser


def _run(command: list[str], *, capture: bool = True) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE if capture else None,
    )
    return "" if result.stdout is None else result.stdout


def _remote(args: argparse.Namespace, command: list[str]) -> str:
    body = "cd " + shlex.quote(args.remote_repo) + " && " + " ".join(
        shlex.quote(part) for part in command
    )
    return _run(["ssh", args.remote, body])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bundle_target_gib <= 0 or args.max_bundles < 1:
        raise SystemExit("bundle-target-gib and max-bundles must be positive")
    local_root = args.local_root.resolve()
    bundles_dir = local_root / "bundles"
    materialized = local_root / "materialized"
    receipts_dir = local_root / "receipts"
    for directory in (bundles_dir, materialized, receipts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _remote(args, ["mkdir", "-p", f"{args.run_dir}/archive_receipts"])

    completed: list[dict[str, object]] = []
    for _ in range(args.max_bundles):
        raw = _remote(
            args,
            [
                args.remote_python,
                "tools/manage_v3_archive.py",
                "create",
                "--run-dir",
                args.run_dir,
                "--bundle-target-gib",
                str(args.bundle_target_gib),
            ],
        )
        created = json.loads(raw)
        if created["status"] == "nothing_to_archive":
            break
        archive_remote = str(created["archive"])
        manifest_remote = str(created["manifest"])
        archive_local = bundles_dir / Path(archive_remote).name
        manifest_local = bundles_dir / Path(manifest_remote).name
        _run(["scp", f"{args.remote}:{archive_remote}", str(archive_local)], capture=False)
        _run(["scp", f"{args.remote}:{manifest_remote}", str(manifest_local)], capture=False)
        receipt_local = receipts_dir / f"{created['bundle_id']}.receipt.json"
        receipt = verify_archive_bundle(
            archive_local,
            manifest_local,
            extract_to=materialized,
            receipt_path=receipt_local,
        )
        incoming = f"{args.run_dir}/archive_receipts/incoming-{receipt_local.name}"
        remote_incoming = f"{args.remote_repo.rstrip('/')}/{incoming}"
        _run(["scp", str(receipt_local), f"{args.remote}:{remote_incoming}"], capture=False)
        _remote(
            args,
            [
                args.remote_python,
                "tools/manage_v3_archive.py",
                "ingest-receipt",
                "--run-dir",
                args.run_dir,
                "--receipt",
                incoming,
            ],
        )
        completed.append(
            {
                "bundle_id": created["bundle_id"],
                "archive": str(archive_local),
                "manifest": str(manifest_local),
                "receipt": str(receipt_local),
                "entries": len(receipt["entries"]),
            }
        )
        if int(created["remaining_unarchived_files"]) == 0:
            break
    prune_result = None
    if args.prune:
        prune_result = json.loads(
            _remote(
                args,
                [
                    args.remote_python,
                    "tools/manage_v3_archive.py",
                    "prune",
                    "--run-dir",
                    args.run_dir,
                    "--execute",
                ],
            )
        )
    sys.stdout.write(
        json.dumps(
            {
                "status": "complete",
                "local_root": str(local_root),
                "bundles": completed,
                "prune": prune_result,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
