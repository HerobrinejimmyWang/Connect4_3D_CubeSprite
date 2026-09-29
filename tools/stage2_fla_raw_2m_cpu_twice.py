"""Remeasure the verified independent FLA2-R2 raw B8 2M terminal twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_stage2_cpu_latency import latency_statistics, run_one

OUT = ROOT / "training/runs/stage2/fla2_r2/cpu_terminal_2m_twice"
INPUTS = OUT / "inputs"
CHECKPOINT = INPUTS / "terminal_checkpoint.pt"
SNAPSHOT = INPUTS / "terminal_snapshot.pt"
CONFIG = INPUTS / "config.json"
RECEIPT = INPUTS / "snapshot_receipt.json"
SIMS = 512
IDLE_S = 10.0
GROUP_IDLE_S = 60.0
EXPECTED = {
    "checkpoint": "8681f8c6e8219969829ccc0ef597a0fea5d6c85fcb9b567b45788ea2c3551f29",
    "snapshot": "2524860a99c85ae2e2aa2eb018368e906c7bbcfb1191051532a2d75f42cf6f2f",
    "config": "725b4a01abc87252b45900c82ada3f33b33dc96189522a559e2644e3fb855d92",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"existing evidence differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def verify_inputs() -> None:
    for label, path in (("checkpoint", CHECKPOINT), ("snapshot", SNAPSHOT), ("config", CONFIG)):
        if sha256(path) != EXPECTED[label]:
            raise ValueError(f"2M {label} SHA-256 differs")
    receipt = read(RECEIPT)
    if (receipt["source_checkpoint_sha256"] != EXPECTED["checkpoint"]
            or receipt["output_sha256"] != EXPECTED["snapshot"]
            or receipt["train_positions_consumed"] != 2_000_000
            or receipt["evaluation_only"] is not True):
        raise ValueError("terminal snapshot receipt differs from 2M checkpoint")


def verify_group(path: Path) -> dict:
    result = read(path)
    meta = result["metadata"]
    if (meta["artifact_kind"] != "formal_v3_snapshot"
            or meta["artifact_sha256"] != EXPECTED["snapshot"]
            or meta["config_sha256"] != EXPECTED["config"]
            or meta["mcts_sims"] != SIMS or meta["repeats"] != 1
            or meta["idle_s"] != IDLE_S
            or result["summary"]["excluding_shortcuts"]["count"] != 15):
        raise ValueError(f"CPU group identity or search count differs: {path}")
    return result


def execute() -> None:
    verify_inputs()
    groups = []
    for index in range(2):
        target = OUT / f"repeat{index + 1}_512.json"
        if target.exists():
            group = verify_group(target)
        else:
            if index:
                time.sleep(GROUP_IDLE_S)
            group = run_one("fla2_r2_raw3d_to2d_b8_terminal_2m", SNAPSHOT, CONFIG,
                            SIMS, repeats=1, idle_s=IDLE_S,
                            artifact_kind="formal_v3_snapshot")
            group["metadata"]["group_idle_s"] = GROUP_IDLE_S
            group["metadata"]["source_checkpoint_sha256"] = EXPECTED["checkpoint"]
            write_once(target, group)
        groups.append(group)
    first, second = (group["metadata"] for group in groups)
    machine_keys = ("runtime", "platform", "processor", "cpu_count",
                    "torch_num_threads", "torch_num_interop_threads", "corpus",
                    "corpus_seed", "forced_tactics", "excluded_state_indices")
    if any(first[key] != second[key] for key in machine_keys):
        raise ValueError("CPU host, thread, or corpus metadata drifted between repeats")
    records = []
    for index, group in enumerate(groups):
        for row in group["records"]:
            records.append({**row, "repeat_index": index})
    if len(records) != 44:
        raise ValueError("two repeats must each contain 22 non-excluded response states")
    searched = [row["latency_s"] for row in records if not row["shortcut_triggered"]]
    if len(searched) != 30:
        raise ValueError("two repeats must contain 30 searched positions")
    summary = {
        "schema": "stage2-fla2-r2-raw-b8-terminal-2m-cpu-twice-v1",
        "status": "relative_latency_only_not_flash_qualified",
        "lineage": "independent_five_rule_2m_terminal_checkpoint",
        "snapshot_evaluation_only": True,
        "simulations": SIMS, "repeats": 2,
        "idle_s": IDLE_S, "group_idle_s": GROUP_IDLE_S,
        "source_checkpoint_sha256": EXPECTED["checkpoint"],
        "snapshot_sha256": EXPECTED["snapshot"],
        "config_sha256": EXPECTED["config"],
        "machine": {key: first[key] for key in machine_keys},
        "searched": latency_statistics(searched),
        "including_shortcuts": latency_statistics([row["latency_s"] for row in records]),
        "repeat_results": [
            {"path": f"repeat{index + 1}_512.json", "sha256": sha256(OUT / f"repeat{index + 1}_512.json"),
             "searched": group["summary"]["excluding_shortcuts"]}
            for index, group in enumerate(groups)
        ],
        "records": records,
    }
    write_once(OUT / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print(json.dumps({"input_snapshot": str(SNAPSHOT), "simulations": SIMS,
                          "repeats": 2, "idle_s": IDLE_S,
                          "group_idle_s": GROUP_IDLE_S}, indent=2))
