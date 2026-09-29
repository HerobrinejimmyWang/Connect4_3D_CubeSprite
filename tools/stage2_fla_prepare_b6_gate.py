"""Prepare local, immutable B6C128 CPU gate inputs for a later remote launch."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLA = ROOT / "training/runs/stage2/fla"
NAME = "raw3d_to2d_thin_b6c128"
B6 = FLA / "b6_raw_2m"
PACKAGE = B6 / "cpu_gate_package"
SOURCES = {
    "cpu_result.json": FLA / "cpu_b6_2m/idle10_group60" / f"{NAME}_512.json",
    "cpu_summary.json": FLA / "cpu_b6_2m/idle10_group60/summary.json",
    "reference_summary.json": FLA / "cpu_boundary_6m/idle10_group60/summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    result = json.loads(SOURCES["cpu_result.json"].read_text(encoding="utf-8"))
    summary = json.loads(SOURCES["cpu_summary.json"].read_text(encoding="utf-8"))
    reference = json.loads(SOURCES["reference_summary.json"].read_text(encoding="utf-8"))
    metadata = result["metadata"]
    if (metadata["mcts_sims"], metadata["repeats"], metadata["idle_s"]) != (512, 3, 10.0):
        raise ValueError("B6 per-model CPU protocol mismatch")
    if (summary["simulations"], summary["repeats"], summary["idle_s"], summary["group_idle_s"]) != (512, 3, 10.0, 60.0):
        raise ValueError("B6 CPU group protocol mismatch")
    if summary["machine"] != reference["machine"]:
        raise ValueError("B6 CPU machine differs from reference")
    if result["summary"]["searched_measurement_count"] != 45:
        raise ValueError("B6 searched position count differs from protocol")
    config = B6 / "configs" / f"{NAME}__standard_late__seed271828.json"
    model = B6 / "runs" / NAME / "standard_late/seed271828/model.pt"
    report_path = model.parent / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["training_execution"]["target_positions_reached"]:
        raise ValueError("B6 donor has not reached 1M")
    if metadata["config_sha256"] != sha256(config):
        raise ValueError("B6 CPU config hash mismatch")
    if metadata["artifact_sha256"] != sha256(model) or report["model_artifact"]["sha256"] != sha256(model):
        raise ValueError("B6 donor/CPU model hash mismatch")
    mean = result["summary"]["excluding_shortcuts"]["mean_s"]
    if mean > 3.8:
        raise ValueError("B6 mean exceeds the self-play gate")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source in SOURCES.items():
        destination = PACKAGE / name
        if destination.exists() and sha256(destination) != sha256(source):
            raise FileExistsError(f"existing gate package member differs: {destination}")
        shutil.copy2(source, destination)
        files[name] = {"source": source.relative_to(ROOT).as_posix(), "sha256": sha256(destination)}
    manifest = {
        "schema": "connect4-stage2-fla-b6-cpu-gate-package-v1",
        "status": "local_inputs_ready_remote_execution_waits_for_user_signal",
        "variant": NAME,
        "cpu_mean_s": mean,
        "cpu_p95_s": result["summary"]["excluding_shortcuts"]["p95_s"],
        "cpu_mean_upper_gate_s": 3.8,
        "donor_config_sha256": sha256(config),
        "donor_model_sha256": sha256(model),
        "files": files,
        "remote_gate_directory": "training/runs/stage2/fla/b6_raw_2m/cpu_gate",
        "remote_dry_run": (
            "/root/miniconda3/bin/python -B tools/stage2_fla_b6_selfplay_gate.py "
            "--cpu-result training/runs/stage2/fla/b6_raw_2m/cpu_gate/cpu_result.json "
            "--cpu-summary training/runs/stage2/fla/b6_raw_2m/cpu_gate/cpu_summary.json "
            "--reference-summary training/runs/stage2/fla/b6_raw_2m/cpu_gate/reference_summary.json"
        ),
    }
    target = PACKAGE / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
