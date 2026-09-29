"""Select two FLA finalists from the stable top-three CPU control bracket."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import stage2_fla_post3m_selection as base
from stage2_fla_post3m_cpu import NAMES, OUTPUT as ORIGINAL, REMOTE_MATCH
from stage2_fla_selfplay_queue import ROOT, atomic_json, sha256


BRACKET = ROOT / "training/runs/stage2/fla/cpu_terminal_3m/recheck_top3_20260927"
HANDOFF_CPU = BRACKET / "handoff"
OUT = ROOT / "training/runs/stage2/fla/selection_3m/r2_stable_top3"
FIRST_SELECTION = ROOT / "training/runs/stage2/fla/selection_3m/r1/selection.json"
EXPECTED_TOP3 = ("raw3d_to2d_b8", "raw3d_to2d_thin_b8c192",
                 "raw3d_to2d_thin_b6c128")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        existing = read(path)
        if {k: v for k, v in existing.items() if k != "created_at_utc"} != {
            k: v for k, v in value.items() if k != "created_at_utc"
        }:
            raise ValueError(f"existing selection evidence differs: {path}")
    else:
        atomic_json(path, value)


def prepare() -> dict:
    state = read(BRACKET / "watcher_state.json")
    bracket_path = BRACKET / "summary.json"
    bracket = read(bracket_path)
    if (state["status"] != "stable_control_pass"
            or state["summary_sha256"] != sha256(bracket_path)
            or bracket["status"] != "stable_control_pass"
            or bracket["schema"] != "connect4-stage2-fla-terminal-cpu-bracket-v1"
            or bracket["max_control_ratio"] != 1.15
            or bracket["max_bracket_ratio"] != 1.10
            or tuple(bracket["variants"]) != EXPECTED_TOP3
            or bracket["protocol"] != {
                "simulations": 512, "repeats": 3, "idle_s": 10.0,
                "group_idle_s": 60.0, "searched_measurements_per_model": 45,
            }):
        raise ValueError("top-three CPU bracket was not stably completed")
    old = bracket["old_control_mean_s"]
    before, after = (bracket[key]["mean_s"] for key in
                     ("control_before", "control_after"))
    if (before > 1.15 * old or after > 1.15 * old
            or max(before, after) / min(before, after) > 1.10):
        raise ValueError("CPU control drift exceeds frozen bracket limits")
    inputs_path = ORIGINAL / "inputs_remote.json"
    inputs = read(inputs_path)
    if (inputs["schema"] != "connect4-stage2-fla-terminal-four-v1"
            or inputs["physical_gpu"] != 1
            or bracket["terminal_inputs_sha256"] != sha256(inputs_path)):
        raise ValueError("official GPU1 terminal input manifest differs")
    cpu_rows = {}
    for name in EXPECTED_TOP3:
        result = bracket["results"][name]
        source = inputs["models"][name]
        if (result["count"] != 45
                or result["sha256"] != sha256(BRACKET / f"{name}_512.json")
                or result["artifact_sha256"] != source["snapshot_sha256"]
                or result["config_sha256"] != source["config_sha256"]):
            raise ValueError(f"top-three CPU model/protocol identity differs: {name}")
        cpu_rows[name] = {
            "name": name, "mean_s": result["mean_s"],
            "median_s": result["median_s"], "p90_s": result["p90_s"],
            "p95_s": result["p95_s"], "count": result["count"],
            "passes_3_3_s": result["mean_s"] <= 3.3,
            "model_sha256": result["artifact_sha256"],
            "config_sha256": result["config_sha256"],
            "terminal_checkpoint_sha256": source["terminal_sha256"],
            "result_sha256": result["sha256"],
        }
    first = read(FIRST_SELECTION)
    if (first["status"] != "insufficient_qualifiers_requires_review"
            or first["selected"]):
        raise ValueError("original zero-finalist record changed")
    OUT.mkdir(parents=True, exist_ok=True)
    HANDOFF_CPU.mkdir(parents=True, exist_ok=True)
    bracket_copy = HANDOFF_CPU / "bracket_summary.json"
    if bracket_copy.exists():
        if sha256(bracket_copy) != sha256(bracket_path):
            raise ValueError("existing bracket handoff copy differs")
    else:
        shutil.copyfile(bracket_path, bracket_copy)
    inputs_copy = HANDOFF_CPU / "inputs_remote.json"
    if inputs_copy.exists():
        if sha256(inputs_copy) != sha256(inputs_path):
            raise ValueError("existing terminal input handoff copy differs")
    else:
        shutil.copyfile(inputs_path, inputs_copy)
    cpu_summary = {
        "schema": "connect4-stage2-fla-cpu-terminal-top3-recheck-v1",
        "status": "stable_control_pass",
        "simulations": 512, "repeats": 3, "idle_s": 10.0,
        "group_idle_s": 60.0, "cpu_gate_mean_s": 3.3,
        "source_inputs_sha256": sha256(inputs_path),
        "bracket_summary_sha256": sha256(bracket_path),
        "results": [cpu_rows[name] for name in EXPECTED_TOP3],
    }
    write_once(HANDOFF_CPU / "summary.json", cpu_summary)
    ratings_path = OUT / "ratings_remote.json"
    source_ratings = FIRST_SELECTION.parent / "ratings_remote.json"
    if sha256(source_ratings) != first["ratings_sha256"]:
        raise ValueError("original GPU1 ratings copy hash changed")
    remote_sha = base.remote_command(f"sha256sum {REMOTE_MATCH}/ratings.json").split()[0]
    if remote_sha != sha256(source_ratings):
        raise ValueError("official remote GPU1 ratings hash changed")
    if ratings_path.exists():
        if sha256(ratings_path) != remote_sha:
            raise ValueError("existing r2 ratings copy differs")
    else:
        shutil.copyfile(source_ratings, ratings_path)
    ratings = read(ratings_path)
    if (ratings["schema"] != "connect4-stage2-fla-3m-round-robin-ratings-v1"
            or ratings["search_sims"] != 256
            or ratings["opening_pairs_per_edge"] != 64
            or len(ratings["edges"]) != 6):
        raise ValueError("official GPU1 match protocol differs")
    selection_cpu = {**cpu_rows, "gravity_b8": {
        "mean_s": None, "passes_3_3_s": False,
    }}
    base.OUT = OUT
    base.OUTPUT = HANDOFF_CPU
    decision = base.decide(selection_cpu, ratings, inputs)
    decision.update({
        "selection_evidence": "stable_top3_cpu_bracket_v1",
        "bracket_summary_sha256": sha256(bracket_path),
        "prior_zero_finalist_selection_sha256": sha256(FIRST_SELECTION),
        "gravity_cpu_status": "not_rechecked_in_stable_top3_bracket; last in GPU1 Elo",
    })
    if (decision["status"] != "provisional_two_architectures_not_flash_qualified"
            or decision["selected"] != list(EXPECTED_TOP3[:2])):
        raise ValueError(f"unexpected stable-bracket finalists: {decision['selected']}")
    write_once(OUT / "selection.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"bracket": str(BRACKET / "summary.json"),
                          "selection": str(OUT / "selection.json"),
                          "expected": EXPECTED_TOP3[:2]}, indent=2))
        return
    decision = prepare()
    base.publish_handoff(OUT / "selection.json")
    atomic_json(OUT / "watcher_state.json", {
        "status": "handoff_ready", "selected": decision["selected"],
        "selection_sha256": sha256(OUT / "selection.json"),
        "remote_ready_marker_sha256": sha256(OUT / "handoff_ready.json"),
    })
    print(json.dumps(decision["selected"], indent=2))


if __name__ == "__main__":
    main()
