"""Summarize the final gate verdicts for FLA raw3d-to2d runs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training/runs/stage2/fla/selfplay/r1/runs"


def main() -> int:
    for directory in sorted(RUNS.glob("*raw3d_to2d*")):
        rows = []
        for path in sorted((directory / "metrics").glob("gate_g*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            summary = data["summary"]
            rows.append({
                "generation": int(path.stem.split("g")[-1]),
                "verdict": data["verdict"],
                "reason": data["reason"],
                "games": summary["overall"]["games"],
                "wins": summary["overall"]["wins"],
                "draws": summary["overall"]["draws"],
                "losses": summary["overall"]["losses"],
                "point_score": summary["overall"]["point_score"],
                "ci95": [summary["ci_lower"], summary["ci_upper"]],
            })
        print(json.dumps({"run": directory.name, "gates": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
