"""Run frozen Stage 2B configs serially with fail-closed resume semantics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPLETE_STATUS = "stopped_at_safe_boundary"
COMPLETE_REASON = "max_train_positions"
STABILITY_PAUSE_REASON = "stability_pause"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_terminal_result(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return None
    try:
        content = log_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    try:
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    # Worker cleanup warnings can precede the CLI's final JSON document. Only
    # accept a decoded object when everything after it is whitespace, so a
    # partial/nested JSON fragment can never be mistaken for terminal state.
    decoder = json.JSONDecoder()
    for offset, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, end = decoder.raw_decode(content, offset)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and not content[end:].strip():
            return payload
    return None


def _completed_at_bound(result: dict[str, Any] | None, bound: int) -> bool:
    if not result:
        return False
    if result.get("status") != COMPLETE_STATUS or result.get("stop_reason") != COMPLETE_REASON:
        return False
    loop_state = result.get("formal_loop_state")
    consumed = loop_state.get("train_positions_consumed") if isinstance(loop_state, dict) else None
    if not isinstance(consumed, int):
        generations = result.get("results")
        if not isinstance(generations, list) or not generations:
            return False
        consumed = generations[-1].get("train_positions_consumed")
    return isinstance(consumed, int) and consumed >= bound


def _stability_paused(result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    if result.get("status") != COMPLETE_STATUS:
        return False
    if result.get("stop_reason") != STABILITY_PAUSE_REASON:
        return False
    loop_state = result.get("formal_loop_state")
    consumed = loop_state.get("train_positions_consumed") if isinstance(loop_state, dict) else None
    if not isinstance(consumed, int):
        generations = result.get("results")
        if not isinstance(generations, list) or not generations:
            return False
        consumed = generations[-1].get("train_positions_consumed")
    return isinstance(consumed, int) and consumed > 0


def _matching_processes(config_path: Path) -> list[int]:
    if os.name != "posix" or not Path("/proc").is_dir():
        return []
    needles = (str(config_path), config_path.as_posix(), config_path.name)
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if (
            "training.v3" in command
            and " run " in f" {command} "
            and any(needle in command for needle in needles)
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def _run_dir_from_config(config_path: Path, repo_root: Path) -> Path:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(raw["run"]["run_dir"])
    return run_dir if run_dir.is_absolute() else repo_root / run_dir


def _wait_for_adopted_run(
    *, config_path: Path, log_path: Path, bound: int, poll_seconds: int
) -> dict[str, Any]:
    while True:
        pids = _matching_processes(config_path)
        if not pids:
            break
        print(
            f"[{_utc_now()}] waiting for existing run {config_path.stem}: pids={pids}",
            flush=True,
        )
        time.sleep(poll_seconds)
    result = _load_terminal_result(log_path)
    if not _completed_at_bound(result, bound):
        raise RuntimeError(
            f"adopted run ended without a valid {bound}-position terminal result: {config_path}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("canary", "extension", "confirmation"), default="canary")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")

    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "connect4-v3-stage2b-configs-v2":
        raise ValueError("unsupported Stage 2B manifest")
    bound_key = {
        "canary": "canary_max_train_positions",
        "extension": "extension_max_train_positions",
        "confirmation": "optional_confirmation_max_train_positions",
    }[args.phase]
    log_dir = (args.log_dir or repo_root / "training/runs/stage2/stage2b_logs").resolve()
    state_path = (args.state or log_dir / f"queue_{args.phase}_state.json").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "schema": "connect4-v3-stage2b-queue-state-v1",
        "phase": args.phase,
        "manifest": str(manifest_path),
        "started_at": _utc_now(),
        "status": "running",
        "runs": [],
    }
    _atomic_json(state_path, state)

    for index, row in enumerate(manifest["runs"]):
        config_path = Path(row["config"])
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config_path = config_path.resolve()
        bound = int(row[bound_key])
        run_id = str(row["run_id"])
        log_path = log_dir / f"{run_id}.{args.phase}.log"
        run_dir = _run_dir_from_config(config_path, repo_root)
        record = {
            "index": index,
            "run_id": run_id,
            "config": str(config_path),
            "bound": bound,
            "log": str(log_path),
            "status": "pending",
        }
        state["runs"].append(record)
        state["active_index"] = index
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)

        result = _load_terminal_result(log_path)
        if _completed_at_bound(result, bound):
            record["status"] = "already_complete"
            record["completed_at"] = _utc_now()
            _atomic_json(state_path, state)
            print(f"[{_utc_now()}] validated existing completion: {run_id}", flush=True)
            continue
        if _stability_paused(result):
            record["status"] = "guard_triggered"
            record["interpretation"] = "elo_calibration_required"
            record["completed_at"] = _utc_now()
            record["train_positions_consumed"] = result["formal_loop_state"][
                "train_positions_consumed"
            ]
            _atomic_json(state_path, state)
            print(
                f"[{_utc_now()}] recorded stability pause and continuing: {run_id}",
                flush=True,
            )
            continue

        pids = _matching_processes(config_path)
        if pids:
            record["status"] = "adopted_running"
            record["pids"] = pids
            _atomic_json(state_path, state)
            result = _wait_for_adopted_run(
                config_path=config_path,
                log_path=log_path,
                bound=bound,
                poll_seconds=args.poll_seconds,
            )
        else:
            resume_existing = run_dir.exists() and args.phase != "canary"
            if run_dir.exists() and not resume_existing:
                raise RuntimeError(
                    f"refusing to overwrite or implicitly resume incomplete run directory: {run_dir}"
                )
            record["status"] = "running"
            record["launched_at"] = _utc_now()
            _atomic_json(state_path, state)
            command = [
                sys.executable,
                "-m",
                "training.v3",
                "run",
                "--config",
                str(config_path),
            ]
            if resume_existing:
                command.append("--resume")
            command.extend(
                [
                    "--execute",
                    "--max-train-positions",
                    str(bound),
                ]
            )
            if resume_existing and row["initialization"] == "cold":
                command.extend(
                    [
                        "--ack-stability-pause-through-train-positions",
                        str(bound),
                    ]
                )
            print(f"[{_utc_now()}] launching: {run_id}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["exit_code"] = completed.returncode
            result = _load_terminal_result(log_path)
            if completed.returncode != 0:
                raise RuntimeError(f"run failed with exit code {completed.returncode}: {run_id}")

        if _stability_paused(result):
            record["status"] = "guard_triggered"
            record["interpretation"] = "elo_calibration_required"
            record["completed_at"] = _utc_now()
            record["train_positions_consumed"] = result["formal_loop_state"][
                "train_positions_consumed"
            ]
            _atomic_json(state_path, state)
            print(
                f"[{_utc_now()}] recorded stability pause and continuing: {run_id}",
                flush=True,
            )
            continue
        if not _completed_at_bound(result, bound):
            status = result.get("status") if result else "unparseable"
            reason = result.get("stop_reason") if result else "unparseable"
            raise RuntimeError(
                f"run stopped before requested bound: {run_id}; status={status}; reason={reason}"
            )
        record["status"] = "complete"
        record["completed_at"] = _utc_now()
        record["terminal_status"] = result["status"]
        record["stop_reason"] = result["stop_reason"]
        _atomic_json(state_path, state)
        print(f"[{_utc_now()}] completed and validated: {run_id}", flush=True)

    state["status"] = "complete"
    state["completed_at"] = _utc_now()
    state.pop("active_index", None)
    _atomic_json(state_path, state)
    print(f"[{_utc_now()}] Stage 2B {args.phase} queue complete", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[{_utc_now()}] queue stopped: {error}", file=sys.stderr, flush=True)
        raise
