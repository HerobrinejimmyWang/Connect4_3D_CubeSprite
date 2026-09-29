"""Select two FLA architectures from verified terminal CPU and round-robin evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from stage2_fla_post3m_cpu import NAMES, OUTPUT, REMOTE, REMOTE_MATCH, REMOTE_ROOT, copy
from stage2_fla_selfplay_queue import ROOT, atomic_json, sha256

OUT = ROOT / "training/runs/stage2/fla/selection_3m/r1"
LATENCY_GATE = 3.3
HANDOFF = f"{REMOTE_ROOT}/training/runs/stage2/fla2_r2/handoff"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote_ratings_ready() -> bool:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", REMOTE,
         f"test -f {REMOTE_MATCH}/ratings.json"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(f"remote rating readiness failed: {result.stderr[-1000:]}")


def load_evidence() -> tuple[dict, dict, dict]:
    cpu_path = OUTPUT / "summary.json"
    if not cpu_path.exists():
        raise FileNotFoundError(cpu_path)
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    if (cpu["schema"] != "connect4-stage2-fla-cpu-terminal-four-v1"
            or cpu["simulations"] != 512 or cpu["repeats"] != 3
            or cpu["idle_s"] != 10 or cpu["group_idle_s"] != 60
            or cpu["cpu_gate_mean_s"] != LATENCY_GATE):
        raise ValueError("CPU evidence protocol or gate differs")
    cpu_rows = {row["name"]: row for row in cpu["results"]}
    if set(cpu_rows) != set(NAMES):
        raise ValueError("CPU evidence does not cover four architectures")
    inputs = json.loads((OUTPUT / "inputs_remote.json").read_text(encoding="utf-8"))
    if sha256(OUTPUT / "inputs_remote.json") != cpu["source_inputs_sha256"]:
        raise ValueError("CPU input manifest hash changed")
    for name in NAMES:
        row = inputs["models"][name]
        measured = cpu_rows[name]
        if (measured["model_sha256"] != row["snapshot_sha256"]
                or measured["config_sha256"] != row["config_sha256"]
                or measured["terminal_checkpoint_sha256"] != row["terminal_sha256"]
                or measured["passes_3_3_s"] != (measured["mean_s"] <= LATENCY_GATE)
                or measured["count"] != 45
                or sha256(OUTPUT / f"{name}_512.json") != measured["result_sha256"]):
            raise ValueError(f"CPU model identity/result drift: {name}")
    ratings_path = OUT / "ratings_remote.json"
    copy(f"{REMOTE_MATCH}/ratings.json", ratings_path)
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    if (ratings["schema"] != "connect4-stage2-fla-3m-round-robin-ratings-v1"
            or ratings["search_sims"] != 256
            or ratings["opening_pairs_per_edge"] != 64
            or set(ratings["ratings"]) != set(NAMES)
            or len(ratings["edges"]) != 6):
        raise ValueError("round-robin evidence protocol or model set differs")
    observed_edges = {frozenset((edge["a"], edge["b"])) for edge in ratings["edges"]}
    expected_edges = {frozenset((NAMES[i], NAMES[j]))
                      for i in range(4) for j in range(i + 1, 4)}
    if observed_edges != expected_edges:
        raise ValueError("round-robin is not a complete single cycle")
    for edge in ratings["edges"]:
        if sum(edge["w_d_l"]) != 128:
            raise ValueError(f"edge lacks 64 color-swapped opening pairs: {edge}")
    return cpu_rows, ratings, inputs


def decide(cpu: dict, ratings: dict, inputs: dict) -> dict:
    ranked = sorted(NAMES, key=lambda n: (-ratings["ratings"][n]["elo"], n))
    passing = [name for name in ranked if cpu[name]["passes_3_3_s"]]
    dominant = [
        name for name in NAMES
        if ratings["ratings"][name]["ci95"][0] >
        max(ratings["ratings"][peer]["ci95"][1] for peer in NAMES if peer != name)
    ]
    exception = next((name for name in dominant if not cpu[name]["passes_3_3_s"]), None)
    if exception is not None and passing:
        selected = [passing[0], exception]
        rule = "unique slow model has 95% Elo lower bound above every peer upper bound; replace ordinary second"
    else:
        selected = passing[:2]
        rule = "top two round-robin Elo among models with terminal CPU mean <=3.3 s"
    if len(selected) != 2:
        status = "insufficient_qualifiers_requires_review"
    else:
        status = "provisional_two_architectures_not_flash_qualified"
    return {
        "schema": "connect4-stage2-fla-3m-two-selection-v1",
        "created_at_utc": now(), "status": status, "rule": rule,
        "cpu_gate_mean_s": LATENCY_GATE,
        "exception_model": exception,
        "strict_ci_nonoverlap": "candidate lower95 > every other candidate upper95",
        "ratings_sha256": sha256(OUT / "ratings_remote.json"),
        "cpu_summary_sha256": sha256(OUTPUT / "summary.json"),
        "input_manifest_sha256": sha256(OUTPUT / "inputs_remote.json"),
        "ranking": [{
            "name": name, "elo": ratings["ratings"][name]["elo"],
            "ci95": ratings["ratings"][name]["ci95"],
            "cpu_mean_s": cpu[name]["mean_s"],
            "passes_3_3_s": cpu[name]["passes_3_3_s"],
            "terminal_checkpoint_sha256": inputs["models"][name]["terminal_sha256"],
            "terminal_snapshot_sha256": inputs["models"][name]["snapshot_sha256"],
        } for name in ranked],
        "selected": selected,
        "eliminated": [name for name in NAMES if name not in selected],
    }


def remote_command(command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", REMOTE, command],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"remote handoff command failed: {result.stderr[-1000:]}")
    return result.stdout.strip()


def publish_handoff(selection: Path) -> None:
    """Publish verified CPU/selection evidence; write the ready marker last."""

    if len(json.loads(selection.read_text(encoding="utf-8"))["selected"]) != 2:
        return
    remote_command(f"mkdir -p {HANDOFF}")
    artifacts = [
        (selection, f"{HANDOFF}/selection.json"),
        (OUTPUT / "summary.json", f"{HANDOFF}/cpu_summary.json"),
    ]
    bracket = OUTPUT / "bracket_summary.json"
    if bracket.exists():
        artifacts.append((bracket, f"{HANDOFF}/bracket_summary.json"))
    for local, remote in artifacts:
        expected = sha256(local)
        existing = remote_command(
            f"if test -f {remote}; then sha256sum {remote}; fi")
        if existing:
            if existing.split()[0] != expected:
                raise ValueError(f"remote handoff artifact differs: {remote}")
            continue
        temporary = remote + ".uploading"
        result = subprocess.run(
            ["scp", "-o", "ConnectTimeout=15", str(local), f"{REMOTE}:{temporary}"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode:
            raise RuntimeError(f"handoff upload failed: {result.stderr[-1000:]}")
        observed = remote_command(f"sha256sum {temporary}").split()[0]
        if observed != expected:
            raise ValueError(f"remote handoff SHA-256 mismatch: {remote}")
        remote_command(f"mv {temporary} {remote}")
    marker = OUT / "handoff_ready.json"
    payload = {
        "schema": "connect4-stage2-fla2-r2-handoff-v1",
        "selection_sha256": sha256(selection),
        "cpu_summary_sha256": sha256(OUTPUT / "summary.json"),
        "ratings_sha256": sha256(OUT / "ratings_remote.json"),
        "selected": json.loads(selection.read_text(encoding="utf-8"))["selected"],
    }
    if bracket.exists():
        payload["bracket_summary_sha256"] = sha256(bracket)
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != payload:
            raise ValueError("existing local handoff marker differs")
    else:
        atomic_json(marker, payload)
    remote_marker = f"{HANDOFF}/ready.json"
    existing = remote_command(
        f"if test -f {remote_marker}; then sha256sum {remote_marker}; fi")
    if existing:
        if existing.split()[0] != sha256(marker):
            raise ValueError("remote handoff marker differs")
    else:
        temporary = remote_marker + ".uploading"
        result = subprocess.run(
            ["scp", "-o", "ConnectTimeout=15", str(marker), f"{REMOTE}:{temporary}"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode:
            raise RuntimeError(f"handoff marker upload failed: {result.stderr[-1000:]}")
        if remote_command(f"sha256sum {temporary}").split()[0] != sha256(marker):
            raise ValueError("remote handoff marker SHA-256 mismatch")
        remote_command(f"mv {temporary} {remote_marker}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    if args.poll_seconds < 30:
        parser.error("poll interval must be at least 30 seconds")
    if not args.execute:
        print(json.dumps({"cpu_gate_mean_s": LATENCY_GATE, "names": NAMES,
                          "exception": "slow candidate lower95 exceeds every other upper95",
                          "selected_count": 2}, indent=2))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    state = OUT / "watcher_state.json"
    try:
        deadline = time.monotonic() + 96 * 3600
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU summary or round-robin ratings unavailable after 96 hours")
            try:
                if (OUTPUT / "summary.json").exists() and remote_ratings_ready():
                    cpu, ratings, inputs = load_evidence()
                    break
                reason = "CPU summary or round-robin ratings not yet published"
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if not args.watch:
                raise RuntimeError(reason)
            atomic_json(state, {"status": "waiting_evidence",
                                "last_reason": reason, "updated_at_utc": now()})
            time.sleep(args.poll_seconds)
        decision = decide(cpu, ratings, inputs)
        target = OUT / "selection.json"
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            previous.pop("created_at_utc", None)
            decision.pop("created_at_utc", None)
            if previous != decision:
                raise ValueError("existing selection differs; review before replacement")
        else:
            atomic_json(target, decision)
        if len(decision["selected"]) == 2:
            while True:
                try:
                    publish_handoff(target)
                    break
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    if not args.watch or time.monotonic() >= deadline:
                        raise
                    atomic_json(state, {"status": "waiting_remote_handoff",
                                        "last_reason": f"{type(exc).__name__}: {exc}",
                                        "updated_at_utc": now()})
                    time.sleep(args.poll_seconds)
        atomic_json(state, {"status": decision["status"], "updated_at_utc": now(),
                            "selection_sha256": sha256(target),
                            "handoff_ready": len(decision["selected"]) == 2})
        print(json.dumps(decision, indent=2))
    except Exception as exc:
        atomic_json(state, {"status": "failed", "updated_at_utc": now(),
                            "error": f"{type(exc).__name__}: {exc}"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
