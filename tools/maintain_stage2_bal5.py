"""Archive and resume a BAL-5 queue stopped at an archive boundary.

This helper is intentionally local-side: bundles are copied to the local archive,
verified, and acknowledged before the remote prune command can remove anything.
It refuses to act on stability, gate, or generic operator stops.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_REPO = "/root/autodl-tmp/Connect4_3D_game_refactor"
DEFAULT_QUEUE_ROOT = "training/runs/stage2/bal5/r1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintain a receipt-gated BAL-5 archive boundary"
    )
    parser.add_argument("--remote", default="connect4_gpu_2608")
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--queue-root", default=DEFAULT_QUEUE_ROOT)
    parser.add_argument(
        "--local-archive-root",
        type=Path,
        default=ROOT / "training/runs/stage2/archive",
    )
    parser.add_argument("--bundle-target-gib", type=float, default=4.0)
    parser.add_argument("--max-bundles", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    return parser


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    return result.stdout


def _remote(args: argparse.Namespace, command: list[str]) -> str:
    body = "cd " + shlex.quote(args.remote_repo) + " && " + " ".join(
        shlex.quote(part) for part in command
    )
    return _run(["ssh", args.remote, body])


def _remote_json(args: argparse.Namespace, relative: str) -> dict:
    raw = _remote(args, ["cat", relative])
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected from {relative}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bundle_target_gib <= 0 or args.max_bundles < 1:
        raise SystemExit("bundle-target-gib and max-bundles must be positive")
    state_path = f"{args.queue_root}/controller_state.json"
    manifest_path = f"{args.queue_root}/manifest.json"
    state = _remote_json(args, state_path)
    if state.get("status") != "stopped_for_operator":
        print(json.dumps({"status": "no_action", "controller": state}, indent=2))
        return 0
    if state.get("stop_reason") != "archive_required":
        raise RuntimeError(
            "BAL-5 is stopped for a non-archive reason; refusing automatic recovery: "
            f"{state.get('stop_reason')}"
        )
    active_job = str(state.get("active_job", ""))
    manifest = _remote_json(args, manifest_path)
    jobs = [row for row in manifest.get("jobs", []) if row.get("run_id") == active_job]
    if len(jobs) != 1:
        raise RuntimeError(f"cannot resolve exactly one active BAL-5 job: {active_job}")
    job = jobs[0]
    run_dir = str(job["run_dir"])
    phase = str(job.get("initialization", "unknown"))
    local_root = (
        args.local_archive_root.resolve() / "bal5" / "r1" / phase / active_job
    )
    plan = {
        "status": "archive_required",
        "active_job": active_job,
        "run_dir": run_dir,
        "local_root": str(local_root),
        "execute": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    sync_command = [
        sys.executable,
        str(ROOT / "tools/sync_v3_run.py"),
        "--remote",
        args.remote,
        "--remote-repo",
        args.remote_repo,
        "--run-dir",
        run_dir,
        "--local-root",
        str(local_root),
        "--bundle-target-gib",
        str(args.bundle_target_gib),
        "--max-bundles",
        str(args.max_bundles),
        "--prune",
        "--remove-verified-bundles",
    ]
    archive = json.loads(_run(sync_command))
    if archive.get("status") != "complete":
        raise RuntimeError("archive sync did not complete")
    launch = (
        f"nohup bash training/runs/stage2/bootstrap/run_bal5_master.sh "
        f"> training/runs/stage2/bal5_master.log 2>&1 < /dev/null &"
    )
    _run(
        [
            "ssh",
            args.remote,
            "cd " + shlex.quote(args.remote_repo) + " && " + launch,
        ]
    )
    time.sleep(3.0)
    resumed = _remote_json(args, state_path)
    if resumed.get("status") != "running" or resumed.get("active_job") != active_job:
        raise RuntimeError("BAL-5 controller did not resume the archived job")
    print(
        json.dumps(
            {"status": "resumed", "archive": archive, "controller": resumed},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
