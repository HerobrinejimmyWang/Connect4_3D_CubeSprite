"""Join FLA design, archived strength, direct matches, and measured CPU evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "training/runs/stage2/archive/experiments/stage2r3/round3_balance"
FLA = ROOT / "training/runs/stage2/fla"
OLD = (
    ("gravity_b8", "balance_gravity_control"),
    ("raw3d_to2d_b8", "balance_raw3d_to2d"),
    ("column3d_v2_b8", "balance_column3d_fusion_v2"),
    ("multiview3d_b8", "balance_multiview3d_fusion"),
    ("winning3d_b8", "balance_winning3d_fusion"),
)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_table(cpu_dir: Path, new_cpu_dir: Path, b6_cpu_dir: Path,
                b6_full_cpu_dir: Path) -> dict:
    for directory in (cpu_dir, new_cpu_dir, b6_cpu_dir, b6_full_cpu_dir):
        screen = read_json(directory / "summary.json")
        if screen and (
            screen["simulations"] != 512
            or screen["repeats"] != 3
            or screen["idle_s"] != 10.0
            or screen["group_idle_s"] != 60.0
        ):
            raise ValueError(f"CPU group protocol mismatch: {directory / 'summary.json'}")
    bal2 = read_json(ARCHIVE / "bal2/results_matrix.json")
    elo = read_json(ARCHIVE / "bal2_3m_roundrobin/summary.json")
    design = read_json(FLA / "efficiency_6m/design.json")
    if bal2 is None or elo is None or design is None:
        raise FileNotFoundError("BAL-2 archive or FLA design manifest is missing")
    by_variant = {row["variant_id"]: row for row in bal2["ranking"]}
    by_architecture = {row["architecture"]: row for row in elo["ranking"]}
    rows = []
    cpu_machine = None
    for label, variant in OLD:
        source = by_variant[variant]
        result_path = cpu_dir / f"{variant}_512.json"
        measured = read_json(result_path)
        if measured:
            metadata = measured["metadata"]
            if (
                metadata["mcts_sims"] != 512
                or metadata["idle_s"] != 10.0
                or metadata["repeats"] != 3
                or measured["summary"]["searched_measurement_count"] != 45
            ):
                raise ValueError(f"CPU protocol mismatch: {result_path}")
            machine = {key: metadata[key] for key in (
                "platform", "processor", "torch_num_threads", "torch_num_interop_threads",
                "corpus", "corpus_seed", "forced_tactics",
            )}
            if cpu_machine is None:
                cpu_machine = machine
            elif machine != cpu_machine:
                raise ValueError(f"CPU machine/runtime mismatch: {result_path}")
            source_config = ARCHIVE / "bal2_3m/configs" / f"{variant}__standard_late__seed271828.json"
            source_model = ARCHIVE / "bal2_3m/runs" / variant / "standard_late/seed271828/model.pt"
            if sha256(source_config) != metadata["config_sha256"]:
                raise ValueError(f"CPU source config checksum mismatch: {source_config}")
            if sha256(source_model) != metadata["artifact_sha256"]:
                raise ValueError(f"CPU source model checksum mismatch: {source_model}")
        stats = measured["summary"]["excluding_shortcuts"] if measured else None
        rows.append({
            "name": label,
            "origin": "BAL-2 archived 1M/3M; FLA latency screen is separate",
            "parameters": source["parameters"],
            "search_macs_estimate": source["search_macs_estimate"],
            "archived_3m_relative_elo_vs_gravity": by_architecture[source["architecture"]]["relative_elo_vs_gravity"],
            "archived_3m_elo_ci95": by_architecture[source["architecture"]]["ci95"],
            "cpu_512_result": result_path.relative_to(ROOT).as_posix() if measured else None,
            "cpu_512_config_sha256": measured["metadata"]["config_sha256"] if measured else None,
            "cpu_512_model_sha256": measured["metadata"]["artifact_sha256"] if measured else None,
            "cpu_512_mean_s": stats["mean_s"] if stats else None,
            "cpu_512_p95_s": stats["p95_s"] if stats else None,
            "selfplay_1m_status": "excluded_cpu_mean_gt_3p8" if stats and stats["mean_s"] > 3.8 else "pending",
            "qualification": "relative_latency_only" if measured else "latency_pending",
        })
    for source in design["candidates"]:
        variant = source["variant"]
        result_path = new_cpu_dir / f"{variant}_512.json"
        measured = read_json(result_path)
        donor_config = FLA / "efficiency_6m/configs" / f"{variant}__standard_late__seed271828.json"
        donor_model = FLA / "efficiency_6m/runs" / variant / "standard_late/seed271828/model.pt"
        donor_report = FLA / "efficiency_6m/runs" / variant / "standard_late/seed271828/report.json"
        report = read_json(donor_report)
        if report:
            if not report.get("training_execution", {}).get("target_positions_reached"):
                raise ValueError(f"offline donor did not complete 1M: {donor_report}")
            if sha256(donor_model) != report["model_artifact"]["sha256"]:
                raise ValueError(f"offline donor hash mismatch: {donor_model}")
        if measured:
            metadata = measured["metadata"]
            if (
                metadata["mcts_sims"] != 512
                or metadata["idle_s"] != 10.0
                or metadata["repeats"] != 3
                or measured["summary"]["searched_measurement_count"] != 45
            ):
                raise ValueError(f"CPU protocol mismatch: {result_path}")
            machine = {key: metadata[key] for key in cpu_machine}
            if machine != cpu_machine:
                raise ValueError(f"CPU machine/runtime mismatch: {result_path}")
            if sha256(donor_config) != metadata["config_sha256"]:
                raise ValueError(f"CPU source config checksum mismatch: {donor_config}")
            if sha256(donor_model) != metadata["artifact_sha256"]:
                raise ValueError(f"CPU source model checksum mismatch: {donor_model}")
        stats = measured["summary"]["excluding_shortcuts"] if measured else None
        rows.append({
            "name": variant,
            "origin": "new V3 FLA efficiency design; 1M offline donor and self-play are separate",
            "parameters": source["parameters"],
            "search_macs_estimate": source["search_macs_estimate"],
            "design_config_sha256": source["config_sha256"],
            "offline_1m_model_sha256": report["model_artifact"]["sha256"] if report else None,
            "offline_1m_report": donor_report.relative_to(ROOT).as_posix() if report else None,
            "archived_3m_relative_elo_vs_gravity": None,
            "cpu_512_result": result_path.relative_to(ROOT).as_posix() if measured else None,
            "cpu_512_config_sha256": measured["metadata"]["config_sha256"] if measured else None,
            "cpu_512_model_sha256": measured["metadata"]["artifact_sha256"] if measured else None,
            "cpu_512_mean_s": stats["mean_s"] if stats else None,
            "cpu_512_p95_s": stats["p95_s"] if stats else None,
            "qualification": "relative_latency_only" if measured else "latency_pending",
        })
    b6 = read_json(FLA / "b6_raw_2m/design.json")
    if b6:
        variant = b6["variant"]
        config = ROOT / b6["config"]
        if sha256(config) != b6["config_sha256"]:
            raise ValueError("B6 design config checksum mismatch")
        model = FLA / "b6_raw_2m/runs" / variant / "standard_late/seed271828/model.pt"
        report_path = model.parent / "report.json"
        report = read_json(report_path)
        if report and (
            not report.get("training_execution", {}).get("target_positions_reached")
            or sha256(model) != report["model_artifact"]["sha256"]
        ):
            raise ValueError("B6 offline donor is incomplete or mismatched")
        result_path = b6_cpu_dir / f"{variant}_512.json"
        measured = read_json(result_path)
        if measured:
            metadata = measured["metadata"]
            if (
                metadata["mcts_sims"] != 512 or metadata["repeats"] != 3
                or metadata["idle_s"] != 10.0
                or measured["summary"]["searched_measurement_count"] != 45
                or {key: metadata[key] for key in cpu_machine} != cpu_machine
                or sha256(config) != metadata["config_sha256"]
                or sha256(model) != metadata["artifact_sha256"]
            ):
                raise ValueError("B6 CPU protocol, machine, or source hash mismatch")
        stats = measured["summary"]["excluding_shortcuts"] if measured else None
        rows.append({
            "name": variant,
            "origin": "new lower-capacity B6 V3 design; direct matches use the same per-phase protocol",
            "parameters": b6["parameters"],
            "search_macs_estimate": b6["search_macs_estimate"],
            "design_config_sha256": b6["config_sha256"],
            "offline_1m_model_sha256": report["model_artifact"]["sha256"] if report else None,
            "archived_3m_relative_elo_vs_gravity": None,
            "cpu_512_result": result_path.relative_to(ROOT).as_posix() if measured else None,
            "cpu_512_mean_s": stats["mean_s"] if stats else None,
            "cpu_512_p95_s": stats["p95_s"] if stats else None,
            "qualification": "relative_latency_only" if measured else "latency_pending",
        })
    b6_full = read_json(FLA / "b6_raw_full_2m/design.json")
    if b6_full:
        variant = b6_full["variant"]
        config = ROOT / b6_full["config"]
        model = FLA / "b6_raw_full_2m/runs" / variant / "standard_late/seed271828/model.pt"
        report_path = model.parent / "report.json"
        report = read_json(report_path)
        result_path = b6_full_cpu_dir / f"{variant}_512.json"
        measured = read_json(result_path)
        if sha256(config) != b6_full["config_sha256"]:
            raise ValueError("B6 full design config checksum mismatch")
        if report and (not report.get("training_execution", {}).get("target_positions_reached")
                       or sha256(model) != report["model_artifact"]["sha256"]):
            raise ValueError("B6 full donor report/model mismatch")
        if measured:
            metadata = measured["metadata"]
            if (metadata["mcts_sims"] != 512 or metadata["repeats"] != 3
                    or metadata["idle_s"] != 10.0
                    or measured["summary"]["searched_measurement_count"] != 45
                    or {key: metadata[key] for key in cpu_machine} != cpu_machine
                    or sha256(config) != metadata["config_sha256"]
                    or sha256(model) != metadata["artifact_sha256"]):
                raise ValueError("B6 full CPU protocol, machine, or source hash mismatch")
        stats = measured["summary"]["excluding_shortcuts"] if measured else None
        rows.append({
            "name": variant,
            "origin": "new full-branch B6C128 V3 design; 1M offline donor only",
            "parameters": b6_full["parameters"],
            "search_macs_estimate": b6_full["search_macs_estimate"],
            "design_config_sha256": b6_full["config_sha256"],
            "offline_1m_model_sha256": report["model_artifact"]["sha256"] if report else None,
            "offline_1m_report": report_path.relative_to(ROOT).as_posix() if report else None,
            "archived_3m_relative_elo_vs_gravity": None,
            "cpu_512_result": result_path.relative_to(ROOT).as_posix() if measured else None,
            "cpu_512_config_sha256": measured["metadata"]["config_sha256"] if measured else None,
            "cpu_512_model_sha256": measured["metadata"]["artifact_sha256"] if measured else None,
            "cpu_512_mean_s": stats["mean_s"] if stats else None,
            "cpu_512_p95_s": stats["p95_s"] if stats else None,
            "selfplay_1m_status": "excluded_by_user_cpu_mean_ge_3p1" if stats and stats["mean_s"] >= 3.1 else "pending",
            "qualification": "relative_latency_only" if measured else "latency_pending",
        })
    six_path = FLA / "evidence_table/selfplay_verified_1m_six.json"
    selfplay_path = six_path if six_path.is_file() else FLA / "evidence_table/selfplay_verified_1m.json"
    selfplay = read_json(selfplay_path)
    if selfplay:
        if selfplay.get("schema") != "connect4-stage2-fla-selfplay-verified-1m-v1" or selfplay.get("status") != "remote_artifacts_verified":
            raise ValueError("FLA self-play receipt schema/status mismatch")
        by_name = {item["name"]: item for item in selfplay["rows"]}
        if len(by_name) != len(selfplay["rows"]):
            raise ValueError("duplicate FLA self-play receipt name")
        for row in rows:
            evidence = by_name.get(row["name"])
            if evidence:
                if evidence["train_positions_consumed"] != 1_000_000 or evidence["generation"] < 1:
                    raise ValueError(f"incomplete self-play receipt: {row['name']}")
                row.update({
                    "selfplay_1m_status": "remote_verified_terminal_and_accepted",
                    "selfplay_1m_run_dir": evidence["run_dir"],
                    "selfplay_1m_config_sha256": evidence["config_sha256"],
                    "selfplay_1m_generation_commit_sha256": evidence["generation_commit_sha256"],
                    "selfplay_1m_terminal_sha256": evidence["terminal_sha256"],
                    "selfplay_1m_accepted_sha256": evidence["accepted_sha256"],
                    "selfplay_1m_gate_verdict": evidence["gate_verdict"],
                })
    direct_root = FLA / "direct_1m/r2_four_workers"
    direct_path = direct_root / "ratings.json"
    selection_path = direct_root / "selection.json"
    direct = read_json(direct_path)
    selection = read_json(selection_path)
    if direct:
        if direct.get("schema") != "connect4-stage2-fla-direct-rating-v1" or selection is None:
            raise ValueError("FLA direct-match evidence is incomplete")
        if selection.get("ratings_sha256") != sha256(direct_path):
            raise ValueError("FLA direct-match rating checksum mismatch")
        for phase in ("donor", "selfplay_1m_terminal"):
            phase_data = direct["phases"][phase]
            if len(phase_data["matches"]) != 10:
                raise ValueError(f"FLA {phase} direct-match graph is incomplete")
            for match in phase_data["matches"]:
                path = direct_root / phase / f"{match['a']}__vs__{match['b']}.json"
                if sha256(path) != match["sha256"]:
                    raise ValueError(f"FLA direct-match checksum mismatch: {path}")
            for row in rows:
                rating = phase_data["ratings_vs_gravity"].get(row["name"])
                if rating:
                    row[f"direct_{phase}_elo_vs_gravity"] = rating["elo"]
                    row[f"direct_{phase}_elo_ci95"] = rating["ci95"]
                    row["direct_match_protocol"] = "classic_256_sims_32_pairs_4_workers_gpu0"
    return {
        "schema": "connect4-stage2-fla-evidence-table-v1",
        "qualification": "none_flash_qualified",
        "cpu_protocol": "512 simulations, 10 s between responses, 60 s between groups, 3 repeats",
        "cpu_machine": cpu_machine,
        "source_bal2_results": (ARCHIVE / "bal2/results_matrix.json").relative_to(ROOT).as_posix(),
        "source_bal2_elo": (ARCHIVE / "bal2_3m_roundrobin/summary.json").relative_to(ROOT).as_posix(),
        "selfplay_1m_verified_receipt": selfplay_path.relative_to(ROOT).as_posix() if selfplay else None,
        "selfplay_1m_verified_receipt_sha256": sha256(selfplay_path) if selfplay else None,
        "direct_match_ratings": direct_path.relative_to(ROOT).as_posix() if direct else None,
        "direct_match_ratings_sha256": sha256(direct_path) if direct else None,
        "direct_match_selection": selection_path.relative_to(ROOT).as_posix() if selection else None,
        "direct_match_selection_sha256": sha256(selection_path) if selection else None,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, default=FLA / "cpu_boundary_6m/idle10_group60")
    parser.add_argument("--new-cpu-dir", type=Path, default=FLA / "cpu_new_6m/idle10_group60")
    parser.add_argument("--b6-cpu-dir", type=Path, default=FLA / "cpu_b6_2m/idle10_group60")
    parser.add_argument("--b6-full-cpu-dir", type=Path, default=FLA / "cpu_b6_full_2m/idle10_group60")
    parser.add_argument("--output-dir", type=Path, default=FLA / "evidence_table")
    args = parser.parse_args()
    result = build_table(args.cpu_dir.resolve(), args.new_cpu_dir.resolve(),
                         args.b6_cpu_dir.resolve(), args.b6_full_cpu_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "table.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage 2 FLA evidence table",
        "",
        "CPU values are relative measurements on the recorded machine; no model is Flash qualified.",
        "New designs have no archived strength estimate. Offline loss is not used as Elo.",
        "Direct donor and terminal Elo use Classic, 256 simulations, 32 color-swapped openings per edge, and four GPU0 workers.",
        "",
        "| Candidate | Parameters | 512 mean s | 512 p95 s | Archived 3M Elo vs gravity [95% CI] | Donor direct Elo [95% CI] | 1M terminal direct Elo [95% CI] | 1M self-play | CPU status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in result["rows"]:
        mean = "pending" if row["cpu_512_mean_s"] is None else f"{row['cpu_512_mean_s']:.3f}"
        p95 = "pending" if row["cpu_512_p95_s"] is None else f"{row['cpu_512_p95_s']:.3f}"
        elo = row["archived_3m_relative_elo_vs_gravity"]
        ci = row.get("archived_3m_elo_ci95")
        strength = "n/a" if elo is None else f"{elo:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]"
        donor = row.get("direct_donor_elo_vs_gravity")
        donor_ci = row.get("direct_donor_elo_ci95")
        donor_strength = "n/a" if donor is None else f"{donor:+.1f} [{donor_ci[0]:+.1f}, {donor_ci[1]:+.1f}]"
        terminal = row.get("direct_selfplay_1m_terminal_elo_vs_gravity")
        terminal_ci = row.get("direct_selfplay_1m_terminal_elo_ci95")
        terminal_strength = "n/a" if terminal is None else f"{terminal:+.1f} [{terminal_ci[0]:+.1f}, {terminal_ci[1]:+.1f}]"
        lines.append(
            f"| {row['name']} | {row['parameters']:,} | {mean} | {p95} | {strength} | "
            f"{donor_strength} | {terminal_strength} | "
            f"{row.get('selfplay_1m_status', 'pending')} | {row['qualification']} |"
        )
    (args.output_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
