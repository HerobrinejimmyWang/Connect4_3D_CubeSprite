"""Measure archived BAL-2 3M models as the FLA CPU capacity boundary.

This is a latency screen, not Flash qualification. It reads the original
configs and model artifacts without copying or modifying their lineage.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_stage2_cpu_latency import run_one


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training/runs/stage2/archive/experiments/stage2r3/round3_balance/bal2_3m"
NEW_SOURCE = ROOT / "training/runs/stage2/fla/efficiency_6m"
B6_SOURCE = ROOT / "training/runs/stage2/fla/b6_raw_2m"
B6_FULL_SOURCE = ROOT / "training/runs/stage2/fla/b6_raw_full_2m"
MULTIVIEW_B8_SOURCE = ROOT / "training/runs/stage2/fla/multiview_b8_1m"
DEFAULT_OUTPUT = ROOT / "training/runs/stage2/fla/cpu_boundary_6m/idle10_group60"
NEW_OUTPUT = ROOT / "training/runs/stage2/fla/cpu_new_6m/idle10_group60"
B6_OUTPUT = ROOT / "training/runs/stage2/fla/cpu_b6_2m/idle10_group60"
B6_FULL_OUTPUT = ROOT / "training/runs/stage2/fla/cpu_b6_full_2m/idle10_group60"
MULTIVIEW_B8_OUTPUT = ROOT / "training/runs/stage2/fla/cpu_multiview_b8_1m/idle10_group60"
VARIANTS = (
    "balance_gravity_control",
    "balance_raw3d_to2d",
    "balance_column3d_fusion_v2",
    "balance_multiview3d_fusion",
    "balance_winning3d_fusion",
)
NEW_VARIANTS = (
    "column_2d_b8c192",
    "raw3d_to2d_thin_b8c192",
    "column3d_v2_thin_b8c192",
)
B6_VARIANTS = ("raw3d_to2d_thin_b6c128",)
B6_FULL_VARIANTS = ("raw3d_to2d_full_b6c128",)
MULTIVIEW_B8_VARIANTS = ("multiview_resnet_b8c192",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("archived", "new", "b6", "b6_full", "multiview_b8"), default="archived")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sims", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--idle-s", type=float, default=10.0)
    parser.add_argument("--group-idle-s", type=float, default=60.0)
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    if args.sims < 1 or args.repeats < 1 or args.idle_s < 0 or args.group_idle_s < 0:
        parser.error("simulations/repeats must be positive and idle intervals non-negative")
    sources = {"archived": SOURCE, "new": NEW_SOURCE, "b6": B6_SOURCE, "b6_full": B6_FULL_SOURCE,
               "multiview_b8": MULTIVIEW_B8_SOURCE}
    options = {"archived": VARIANTS, "new": NEW_VARIANTS, "b6": B6_VARIANTS,
               "b6_full": B6_FULL_VARIANTS, "multiview_b8": MULTIVIEW_B8_VARIANTS}
    defaults = {"archived": DEFAULT_OUTPUT, "new": NEW_OUTPUT, "b6": B6_OUTPUT,
                "b6_full": B6_FULL_OUTPUT, "multiview_b8": MULTIVIEW_B8_OUTPUT}
    source = sources[args.source]
    available = options[args.source]
    variants = args.variants or available
    if any(variant not in available for variant in variants):
        parser.error(f"invalid {args.source} variants: {variants}")
    if args.output_dir is None:
        args.output_dir = defaults[args.source]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, variant in enumerate(variants):
        config = source / "configs" / f"{variant}__standard_late__seed271828.json"
        model = source / "runs" / variant / "standard_late/seed271828/model.pt"
        if args.source in ("new", "b6", "b6_full", "multiview_b8"):
            report = json.loads((model.parent / "report.json").read_text(encoding="utf-8"))
            import hashlib
            if not report.get("training_execution", {}).get("target_positions_reached"):
                raise ValueError(f"1M donor not complete: {variant}")
            if hashlib.sha256(model.read_bytes()).hexdigest() != report["model_artifact"]["sha256"]:
                raise ValueError(f"donor/report SHA-256 mismatch: {variant}")
        payload = run_one(
            variant, model, config, args.sims,
            repeats=args.repeats, idle_s=args.idle_s,
        )
        payload["metadata"]["group_idle_s"] = args.group_idle_s
        payload["metadata"]["fla_screen_status"] = "relative_latency_only_not_flash_qualified"
        destination = args.output_dir / f"{variant}_{args.sims}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        stats = payload["summary"]["excluding_shortcuts"]
        rows.append({
            "variant": variant,
            "architecture": payload["metadata"]["architecture"],
            "source_config": config.relative_to(ROOT).as_posix(),
            "source_model": model.relative_to(ROOT).as_posix(),
            "config_sha256": payload["metadata"]["config_sha256"],
            "model_sha256": payload["metadata"]["artifact_sha256"],
            "result": destination.relative_to(ROOT).as_posix(),
            "searched_states": stats["count"],
            **stats,
        })
        print(f"{variant}: mean={stats['mean_s']:.3f}s p95={stats['p95_s']:.3f}s", flush=True)
        if index + 1 < len(variants):
            time.sleep(args.group_idle_s)
    summary = {
        "schema": "connect4-stage2-fla-cpu-boundary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "relative_latency_only_not_flash_qualified",
        "source": source.relative_to(ROOT).as_posix(),
        "simulations": args.sims,
        "repeats": args.repeats,
        "idle_s": args.idle_s,
        "group_idle_s": args.group_idle_s,
        "machine": {
            key: payload["metadata"][key]
            for key in (
                "runtime", "platform", "processor", "cpu_count",
                "torch_num_threads", "torch_num_interop_threads",
                "corpus", "corpus_seed", "forced_tactics",
            )
        },
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
