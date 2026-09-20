"""Fail closed unless a V3 run stopped at the requested position bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_stage2b_queue import (
    _completed_at_bound,
    _load_terminal_result,
    _stability_paused,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--bound", type=int, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    result = _load_terminal_result(args.log)
    if _completed_at_bound(result, args.bound):
        status = "complete"
        exit_code = 0
    elif _stability_paused(result):
        status = "stability_pause"
        exit_code = 3
    else:
        status = "unexpected_terminal_state"
        exit_code = 4
    args.status.write_text(
        json.dumps({"status": status, "bound": args.bound, "result": result}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(status)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
