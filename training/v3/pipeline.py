"""Synchronous V3.1 smoke orchestrator.

This module deliberately implements only a small, inspectable CPU pipeline.  It
establishes the persistence and correctness contracts required before a formal
or asynchronous training loop is enabled.
"""

from __future__ import annotations

import json
import hashlib
import os
import random
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from training.runtime_resources import available_cpu_count

from . import __version__
from .actor_runtime import run_self_play_actor_pool
from .checkpoint import CheckpointV1, load_checkpoint, save_checkpoint
from .config import V3Config, config_hash, model_config_dict
from .evaluation import build_openings, play_paired_openings, write_opening_manifest
from .evaluation_runtime import (
    EvaluationModelSource,
    play_paired_openings_parallel,
    play_paired_openings_replicated,
)
from .formal_state import FormalLoopState, PendingCandidateState
from .gate import evaluate_gate, reject_terminal_inconclusive
from .hardware_plan import plan_hardware
from .layout import RunLayout
from .learner import OnlineD4Dataset, V3Learner, build_adamw
from .model import TorchPredictor, build_model
from .policy_target_quality import summarize_visit_targets
from .replay import (
    ReplayShard,
    TrainTokenBucket,
    active_replay_indices,
    concatenate_replay,
    load_replay_shard,
    replay_manifest_path,
    validate_replay_shard_artifacts,
    write_replay_shard,
)
from .replay_export import (
    AUDIT_INDEX_FORMAT,
    AUDIT_INDEX_VERSION,
    build_audit_index,
    game_record_to_replay,
    select_representative_games,
    validate_replay_document,
    write_audit_index_atomic,
    write_replay_atomic,
)
from .retention import GIB, RetentionPolicy
from .selfplay import GameRecord


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_run_root(config: V3Config) -> Path:
    configured = Path(config.run.run_dir) if config.run.run_dir else Path("training/runs") / config.run.run_id
    if configured.is_absolute():
        return configured.resolve()
    return (repository_root() / configured).resolve()


def lineage_config_hash(config: V3Config) -> str:
    """Hash learning semantics while excluding path and resume control state."""

    return config_hash(config)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _append_metric(layout: RunLayout, payload: Mapping[str, Any]) -> None:
    path = layout.metrics / "metrics.jsonl"
    row = {"time": _utc_now(), **dict(payload)}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _seed_runtime(config: V3Config) -> None:
    seed = int(config.run.seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config.runtime.torch_threads)
    torch.use_deterministic_algorithms(config.runtime.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = config.runtime.deterministic
        torch.backends.cudnn.benchmark = not config.runtime.deterministic


def _atomic_save_model_artifact(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = {
        "format": "connect4-v3-model",
        "format_version": 1,
        "model_config": dict(model_config),
        "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "metadata": dict(metadata),
    }
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _load_model_artifact(path: Path, config: V3Config) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "connect4-v3-model" or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported V3 model artifact: {path}")
    if payload.get("model_config") != model_config_dict(config.model):
        raise ValueError(f"Accepted model config does not match this run: {path}")
    model = build_model(config.model)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def _run_artifact_path(layout: RunLayout, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("generation artifact path must be a non-empty string")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("generation artifact path must stay inside the run directory")
    resolved = (layout.root / raw).resolve()
    if not resolved.is_relative_to(layout.root):
        raise ValueError("generation artifact path escaped the run directory")
    return resolved


def _validate_generation_commit(
    layout: RunLayout,
    commit_path: Path,
    *,
    expected_hash: str,
) -> dict[str, Any]:
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if commit.get("schema_version") != 1:
        raise ValueError("unsupported generation commit schema")
    if commit.get("config_hash") != expected_hash:
        raise ValueError("generation commit config hash mismatch")
    generation = int(commit.get("generation", -1))
    if generation < 0 or commit_path.name != f"g{generation:06d}.json":
        raise ValueError("generation commit filename/generation mismatch")
    for path_key, checksum_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("audit_index", "audit_index_sha256"),
    ):
        artifact = _run_artifact_path(layout, commit.get(path_key))
        if not artifact.is_file() or _sha256_file(artifact) != commit.get(checksum_key):
            raise ValueError(f"generation commit {path_key} is missing or corrupt")
    candidate_id = commit.get("candidate_model_id")
    candidate_path = commit.get("candidate_path")
    candidate_hash = commit.get("candidate_sha256")
    if candidate_id is None:
        if candidate_path is not None or candidate_hash is not None:
            raise ValueError("generation commit has a partial null candidate model")
    else:
        candidate_artifact = _run_artifact_path(layout, candidate_path)
        if (
            candidate_artifact.stem != candidate_id
            or not candidate_artifact.is_file()
            or _sha256_file(candidate_artifact) != candidate_hash
        ):
            raise ValueError("generation commit candidate model is missing or corrupt")

    checkpoint_path = _run_artifact_path(layout, commit.get("checkpoint"))
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.generation != generation or checkpoint.config_hash != expected_hash:
        raise ValueError("generation commit checkpoint lineage mismatch")
    if checkpoint.accepted_model_id != commit.get("accepted_model_id"):
        raise ValueError("generation commit accepted model differs from checkpoint")
    if checkpoint.candidate_model_id != commit.get("candidate_model_id"):
        raise ValueError("generation commit candidate model differs from checkpoint")
    run_id = commit.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("generation commit run_id must be a non-empty string")
    replay_cursor = checkpoint.replay_cursor
    if not isinstance(replay_cursor, Mapping):
        raise ValueError("generation checkpoint replay cursor must be a mapping")
    committed_shards = commit.get("replay_shards")
    cursor_shards = replay_cursor.get("shards")
    if not isinstance(committed_shards, list) or committed_shards != cursor_shards:
        raise ValueError("generation commit replay shards differ from checkpoint cursor")
    replay_positions = 0
    ranged_cursor: int | None = None
    ranged_entries = 0
    for entry in committed_shards:
        allowed_entry_schemas = (
            {"path", "checksum_sha256"},
            {"path", "checksum_sha256", "position_start", "position_end"},
        )
        if not isinstance(entry, Mapping) or set(entry) not in allowed_entry_schemas:
            raise ValueError("generation commit replay shard entry has an invalid schema")
        if "position_start" in entry and (
            int(entry["position_start"]) < 0
            or int(entry["position_end"]) <= int(entry["position_start"])
        ):
            raise ValueError("generation commit replay shard position range is invalid")
        replay_path = _run_artifact_path(layout, entry["path"])
        manifest = validate_replay_shard_artifacts(replay_path)
        if manifest.get("checksum_sha256") != entry["checksum_sha256"]:
            raise ValueError("generation commit replay shard checksum differs from manifest")
        if manifest.get("config_hash") != expected_hash:
            raise ValueError("generation commit replay shard config hash mismatch")
        if manifest.get("run_id") != run_id:
            raise ValueError("generation commit replay shard run_id mismatch")
        sample_count = int(manifest["sample_count"])
        if "position_start" in entry:
            start = int(entry["position_start"])
            end = int(entry["position_end"])
            if end - start != sample_count:
                raise ValueError("generation replay shard position range differs from sample count")
            if ranged_cursor is not None and start != ranged_cursor:
                raise ValueError("generation replay shard position ranges are not contiguous")
            ranged_cursor = end
            ranged_entries += 1
        replay_positions += sample_count
    if ranged_entries not in (0, len(committed_shards)):
        raise ValueError("generation replay cursor mixes ranged and legacy shard entries")
    if replay_positions != int(replay_cursor.get("raw_positions", -1)):
        raise ValueError("generation commit replay position count differs from checkpoint cursor")
    if replay_positions != int(commit.get("replay_raw_positions", -1)):
        raise ValueError("generation commit replay position count mismatch")
    cumulative_positions = int(replay_cursor.get("cumulative_raw_positions", replay_positions))
    if "replay_cumulative_positions" in commit and int(
        commit["replay_cumulative_positions"]
    ) != cumulative_positions:
        raise ValueError("generation commit cumulative replay position count mismatch")
    if int(commit.get("next_game_id", -1)) != int(replay_cursor.get("next_game_id", -2)):
        raise ValueError("generation commit next_game_id differs from checkpoint cursor")

    audit_path = _run_artifact_path(layout, commit.get("audit_index"))
    audit_index = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit_index.get("format") != AUDIT_INDEX_FORMAT
        or audit_index.get("format_version") != AUDIT_INDEX_VERSION
        or int(audit_index.get("generation", -1)) != generation
        or audit_index.get("run_id") != run_id
    ):
        raise ValueError("generation commit audit index lineage mismatch")
    audit_checkpoint = audit_index.get("checkpoint")
    if audit_checkpoint != {
        "id": checkpoint_path.name,
        "sha256": commit.get("checkpoint_sha256"),
    }:
        raise ValueError("generation audit index checkpoint reference mismatch")
    audit_rows = audit_index.get("replays")
    if not isinstance(audit_rows, list):
        raise ValueError("generation audit index replay list is invalid")
    for row in audit_rows:
        if not isinstance(row, Mapping):
            raise ValueError("generation audit index replay row is invalid")
        filename = row.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("generation audit replay filename is invalid")
        replay_path = (audit_path.parent / filename).resolve()
        if not replay_path.is_relative_to(audit_path.parent.resolve()):
            raise ValueError("generation audit replay escaped its sample directory")
        if not replay_path.is_file() or _sha256_file(replay_path) != row.get("file_sha256"):
            raise ValueError("generation audit replay is missing or corrupt")
        replay_document = json.loads(replay_path.read_text(encoding="utf-8"))
        validate_replay_document(replay_document)
        if (
            replay_document.get("id") != row.get("replay_id")
            or replay_document.get("fingerprint") != row.get("fingerprint")
        ):
            raise ValueError("generation audit replay identity differs from its index")
    model_id = commit.get("accepted_model_id")
    model_path = commit.get("accepted_model_path")
    model_hash = commit.get("accepted_model_sha256")
    if model_id is None:
        if model_path is not None or model_hash is not None:
            raise ValueError("generation commit has a partial null accepted model")
    else:
        artifact = _run_artifact_path(layout, model_path)
        if artifact.stem != model_id or not artifact.is_file():
            raise ValueError("generation commit accepted model reference is invalid")
        if _sha256_file(artifact) != model_hash:
            raise ValueError("generation commit accepted model checksum mismatch")
    return commit


def _load_latest_generation_commit(
    layout: RunLayout,
    *,
    expected_hash: str,
    allow_missing: bool,
) -> tuple[dict[str, Any], Path] | None:
    pointer_path = layout.manifests / "latest_generation.json"
    pointer_generation: int | None = None
    if pointer_path.exists():
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if set(pointer) != {"schema_version", "generation", "commit", "commit_sha256"}:
                raise ValueError("latest generation pointer has an unsupported schema")
            parsed_generation = int(pointer["generation"])
            if parsed_generation < 0:
                raise ValueError("latest generation pointer cannot be negative")
            pointer_generation = parsed_generation
            commit_path = _run_artifact_path(layout, pointer["commit"])
            if _sha256_file(commit_path) != pointer["commit_sha256"]:
                raise ValueError("latest generation commit checksum mismatch")
            commit = _validate_generation_commit(
                layout,
                commit_path,
                expected_hash=expected_hash,
            )
            if int(commit["generation"]) != pointer_generation:
                raise ValueError("latest generation pointer generation mismatch")
            return commit, commit_path
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # A torn pointer or referenced commit must not make an older complete
            # generation unrecoverable. Scan only at or before its declared age.
            pass
    candidates = sorted(layout.generation_commits.glob("g*.json"), reverse=True)
    for commit_path in candidates:
        try:
            generation = int(commit_path.stem[1:])
            if pointer_generation is not None and generation > pointer_generation:
                continue
            commit = _validate_generation_commit(
                layout,
                commit_path,
                expected_hash=expected_hash,
            )
            return commit, commit_path
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if allow_missing:
        return None
    raise ValueError("no complete committed V3 generation is available")


def _accepted_predictor(
    layout: RunLayout,
    config: V3Config,
) -> tuple[TorchPredictor | None, str | None]:
    latest = _load_latest_generation_commit(
        layout,
        expected_hash=lineage_config_hash(config),
        allow_missing=True,
    )
    if latest is None:
        return None, None
    commit, _commit_path = latest
    model_id = commit.get("accepted_model_id")
    if model_id is None:
        return None, None
    artifact = _run_artifact_path(layout, commit["accepted_model_path"])
    return TorchPredictor(_load_model_artifact(artifact, config), config.runtime.device), str(model_id)


def _result_counts(games: Iterable[GameRecord]) -> dict[str, int]:
    rows = tuple(games)
    return {
        "games": len(rows),
        "p1_wins": sum(game.winner == 1 for game in rows),
        "p2_wins": sum(game.winner == -1 for game in rows),
        "draws": sum(game.winner == 0 for game in rows),
    }


def _run_sequential_gate(
    config: V3Config,
    *,
    generation: int,
    openings: list[Any],
    candidate_predictor: Any,
    incumbent_predictor: Any,
    existing_results: Iterable[Any] = (),
    existing_control_results: Iterable[Any] = (),
    runtime_records: list[dict[str, Any]] | None = None,
    candidate_source: EvaluationModelSource | None = None,
    incumbent_source: EvaluationModelSource | None = None,
    allow_random_bootstrap_control: bool = False,
) -> tuple[list[Any], list[Any], Any, list[dict[str, Any]]]:
    """Append only new opening pairs until the gate resolves or reaches its cap."""

    results = list(existing_results)
    control_results = list(existing_control_results)
    if len(results) % 2 != 0:
        raise ValueError("paired gate history must contain an even number of games")
    completed_pairs = len(results) // 2
    if completed_pairs > config.gate.max_opening_pairs:
        raise ValueError("paired gate history exceeds gate.max_opening_pairs")
    target_pairs = max(config.gate.initial_opening_pairs, completed_pairs)
    if len(openings) < config.gate.max_opening_pairs:
        raise ValueError("opening suite does not cover gate.max_opening_pairs")
    relative_role_guard = config.gate.role_guard_mode == "relative_noninferiority"
    random_bootstrap_control = (
        relative_role_guard
        and incumbent_predictor is None
        and incumbent_source is None
    )
    if relative_role_guard:
        if random_bootstrap_control and not allow_random_bootstrap_control:
            raise ValueError(
                "relative role non-inferiority requires a committed accepted champion"
            )
        if len(control_results) != len(results):
            raise ValueError(
                "relative role control history must cover the same opening pairs as the gate"
            )
    elif control_results:
        raise ValueError("absolute role-floor gates cannot carry role control history")

    gate_search_sims = config.gate.search_sims_for_generation(generation)
    looks: list[dict[str, Any]] = []

    def evaluate_batch(
        batch_openings: list[Any],
        *,
        batch_candidate_predictor: Any,
        batch_incumbent_predictor: Any,
        batch_candidate_source: EvaluationModelSource | None,
        batch_incumbent_source: EvaluationModelSource | None,
    ) -> tuple[list[Any], dict[str, Any]]:
        if config.runtime.evaluation_devices and batch_candidate_source is not None:
            replica_devices = tuple(
                device
                for _ in range(config.runtime.evaluation_replicas_per_device)
                for device in config.runtime.evaluation_devices
            )
            evaluated = play_paired_openings_replicated(
                batch_openings,
                candidate_source=batch_candidate_source,
                incumbent_source=batch_incumbent_source,
                search_sims=gate_search_sims,
                cpuct=config.gate.cpuct,
                worker_devices=replica_devices,
            )
            return list(evaluated.games), evaluated.metrics.to_dict()
        if (
            config.runtime.evaluation_parallel_games == 1
            and config.runtime.evaluation_inference_batch_size == 1
        ):
            started = time.perf_counter()
            games = play_paired_openings(
                batch_openings,
                candidate_predictor=batch_candidate_predictor,
                incumbent_predictor=batch_incumbent_predictor,
                search_sims=gate_search_sims,
                cpuct=config.gate.cpuct,
            )
            wall_seconds = time.perf_counter() - started
            return list(games), {
                "parallel_games": 1,
                "worker_processes": 0,
                "start_method": "serial",
                "games": len(games),
                "wall_seconds": wall_seconds,
                "games_per_second": len(games) / max(wall_seconds, 1e-12),
                "inference_services": [],
            }
        evaluated = play_paired_openings_parallel(
            batch_openings,
            candidate_predictor=batch_candidate_predictor,
            incumbent_predictor=batch_incumbent_predictor,
            search_sims=gate_search_sims,
            cpuct=config.gate.cpuct,
            parallel_games=config.runtime.evaluation_parallel_games,
            inference_batch_size=config.runtime.evaluation_inference_batch_size,
            inference_batch_timeout_s=(
                config.runtime.evaluation_inference_batch_timeout_ms / 1000.0
            ),
        )
        return list(evaluated.games), evaluated.metrics.to_dict()
    while True:
        if completed_pairs < target_pairs:
            pair_start = completed_pairs
            new_openings = openings[completed_pairs:target_pairs]
            new_results, evaluation_runtime = evaluate_batch(
                new_openings,
                batch_candidate_predictor=candidate_predictor,
                batch_incumbent_predictor=incumbent_predictor,
                batch_candidate_source=candidate_source,
                batch_incumbent_source=incumbent_source,
            )
            if runtime_records is not None:
                runtime_records.append(
                    {
                        "evidence": "candidate_vs_incumbent",
                        "pair_start": pair_start,
                        "pair_stop": target_pairs,
                        **evaluation_runtime,
                    }
                )
            expected_games = 2 * (target_pairs - completed_pairs)
            if len(new_results) != expected_games:
                raise RuntimeError(
                    f"paired gate returned {len(new_results)} games, expected {expected_games}"
                )
            results.extend(new_results)
            if relative_role_guard:
                control_batch, control_runtime = evaluate_batch(
                    new_openings,
                    batch_candidate_predictor=incumbent_predictor,
                    batch_incumbent_predictor=incumbent_predictor,
                    batch_candidate_source=incumbent_source,
                    batch_incumbent_source=incumbent_source,
                )
                if len(control_batch) != expected_games:
                    raise RuntimeError(
                        "accepted-control gate returned "
                        f"{len(control_batch)} games, expected {expected_games}"
                    )
                control_results.extend(control_batch)
                if runtime_records is not None:
                    runtime_records.append(
                        {
                            "evidence": (
                                "random_bootstrap_control"
                                if random_bootstrap_control
                                else "accepted_champion_control"
                            ),
                            "pair_start": pair_start,
                            "pair_stop": target_pairs,
                            **control_runtime,
                        }
                    )
            completed_pairs = target_pairs

        decision = evaluate_gate(
            results,
            bootstrap_samples=config.gate.bootstrap_samples,
            confidence=config.gate.decision_confidence(),
            bootstrap_seed=config.run.seed + 2701,
            accept_threshold=config.gate.accept_threshold,
            role_floor=config.gate.role_floor,
            role_hard_reject_floor=config.gate.role_hard_reject_floor,
            allow_role_extension=(
                config.gate.extend_role_floor_to_max_pairs
                and completed_pairs < config.gate.max_opening_pairs
            ),
            role_guard_mode=config.gate.role_guard_mode,
            role_control_results=(control_results if relative_role_guard else None),
            role_noninferiority_margin=config.gate.role_noninferiority_margin,
        )
        at_max_pairs = completed_pairs >= config.gate.max_opening_pairs
        if at_max_pairs:
            decision = reject_terminal_inconclusive(decision)
        look = {
            "pairs": completed_pairs,
            "games": len(results),
            "verdict": decision.verdict,
            "overall_point_score": decision.summary.overall.point_score,
            "ci_lower": decision.summary.ci_lower,
            "ci_upper": decision.summary.ci_upper,
            "role_guard_mode": getattr(decision, "role_guard_mode", "absolute_floor"),
        }
        role_noninferiority = getattr(decision, "role_noninferiority", None)
        if role_noninferiority is not None:
            look["role_noninferiority"] = role_noninferiority.to_dict()
        looks.append(look)
        if decision.verdict != "inconclusive" or at_max_pairs:
            return results, control_results, decision, looks
        target_pairs = min(
            config.gate.max_opening_pairs,
            completed_pairs + config.gate.pair_increment,
        )


def _d4_column_sequence(columns: tuple[int, ...]) -> tuple[int, ...]:
    variants: list[tuple[int, ...]] = []
    for transform in range(8):
        transformed: list[int] = []
        for column in columns:
            row, col = divmod(int(column), 5)
            if transform >= 4:
                col = 4 - col
            for _ in range(transform % 4):
                row, col = col, 4 - row
            transformed.append(row * 5 + col)
        variants.append(tuple(transformed))
    return min(variants)


def _selfplay_health(
    games: Iterable[GameRecord],
    *,
    expected_search_sims: Mapping[str, int] | None = None,
    exploration_phases: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = tuple(games)
    if not rows:
        raise ValueError("self-play health requires at least one game")
    lengths = np.asarray([len(game.moves) for game in rows], dtype=np.float64)
    entropies: dict[str, list[float]] = {"full": [], "fast": []}
    visit_targets: dict[str, list[np.ndarray]] = {"full": [], "fast": []}
    phase_rows = tuple(dict(phase) for phase in (exploration_phases or ()))
    phase_accumulators = [
        {
            "position_count": 0,
            "entropy_sum": 0.0,
            "selected_top1": 0,
            "selected_probability_sum": 0.0,
        }
        for _phase in phase_rows
    ]
    wdl_counts = {"win": 0, "draw": 0, "loss": 0}
    for game in rows:
        moves_by_ply = {move.ply: move for move in game.moves}
        for sample in game.samples:
            visits = sample.visit_counts.astype(np.float64)
            probabilities = visits / max(float(visits.sum()), 1.0)
            positive = probabilities > 0.0
            entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
            entropies[sample.search_kind].append(entropy)
            visit_targets[sample.search_kind].append(sample.visit_counts)
            if phase_rows and float(visits.sum()) > 0.0:
                phase_index = 0
                for index, phase in enumerate(phase_rows):
                    if int(phase["start_ply"]) > sample.ply:
                        break
                    phase_index = index
                move = moves_by_ply.get(sample.ply)
                if move is not None and move.column is not None:
                    selected = int(move.column)
                    accumulator = phase_accumulators[phase_index]
                    accumulator["position_count"] += 1
                    accumulator["entropy_sum"] += entropy
                    accumulator["selected_top1"] += int(
                        visits[selected] == visits.max()
                    )
                    accumulator["selected_probability_sum"] += float(
                        probabilities[selected]
                    )
            label = {0: "win", 1: "draw", 2: "loss"}[int(sample.wdl)]
            wdl_counts[label] += 1
    diversity: dict[str, dict[str, int | float]] = {}
    for depth in (4, 8, 12):
        eligible = [game for game in rows if len(game.moves) >= depth]
        signatures = {
            _d4_column_sequence(tuple(move.column for move in game.moves[:depth]))
            for game in eligible
        }
        diversity[f"ply_{depth}"] = {
            "eligible_games": len(eligible),
            "unique_d4_prefixes": len(signatures),
            "unique_fraction": len(signatures) / max(len(eligible), 1),
        }
    mean_length = float(lengths.mean())
    variance = float(lengths.var())
    short_rate = float(np.mean(lengths <= 12))
    warnings: list[str] = []
    if mean_length < 18.0:
        warnings.append("mean_game_length_below_runbook_watch_level")
    if variance < 50.0:
        warnings.append("game_length_variance_below_runbook_watch_level")
    if short_rate > 0.10:
        warnings.append("short_game_rate_above_runbook_watch_level")
    exploration: list[dict[str, Any]] = []
    for index, (phase, accumulator) in enumerate(zip(phase_rows, phase_accumulators)):
        start_ply = int(phase["start_ply"])
        end_ply_exclusive = (
            int(phase_rows[index + 1]["start_ply"])
            if index + 1 < len(phase_rows)
            else None
        )
        count = int(accumulator["position_count"])
        games_reaching = sum(len(game.moves) > start_ply for game in rows)
        exploration.append(
            {
                "start_ply": start_ply,
                "end_ply_exclusive": end_ply_exclusive,
                "temperature": float(phase["temperature"]),
                "dirichlet_alpha": float(phase["dirichlet_alpha"]),
                "dirichlet_epsilon": float(phase["dirichlet_epsilon"]),
                "games_reaching_phase": games_reaching,
                "position_count": count,
                "mean_positions_per_reaching_game": count / max(games_reaching, 1),
                "mean_visit_entropy": float(accumulator["entropy_sum"]) / max(count, 1),
                "selected_top1_rate": int(accumulator["selected_top1"]) / max(count, 1),
                "mean_selected_visit_probability": (
                    float(accumulator["selected_probability_sum"]) / max(count, 1)
                ),
            }
        )
    return {
        "results": _result_counts(rows),
        "game_length": {
            "mean": mean_length,
            "variance": variance,
            "min": int(lengths.min()),
            "p25": float(np.quantile(lengths, 0.25)),
            "median": float(np.median(lengths)),
            "p75": float(np.quantile(lengths, 0.75)),
            "max": int(lengths.max()),
            "short_le_12": int(np.sum(lengths <= 12)),
            "short_le_12_rate": short_rate,
        },
        "opening_diversity": diversity,
        "mean_policy_entropy": {
            kind: (float(np.mean(values)) if values else 0.0)
            for kind, values in entropies.items()
        },
        "policy_target_quality": {
            kind: (
                summarize_visit_targets(
                    np.stack(values),
                    expected_simulations=(
                        None if expected_search_sims is None else expected_search_sims.get(kind)
                    ),
                )
                if values
                else None
            )
            for kind, values in visit_targets.items()
        },
        "exploration_by_phase": exploration,
        "wdl_labels": wdl_counts,
        "watch_warnings": warnings,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _game_batches(games: list[GameRecord], shard_games: int) -> Iterable[list[GameRecord]]:
    if shard_games < 1:
        raise ValueError("shard_games must be positive")
    for start in range(0, len(games), shard_games):
        yield games[start : start + shard_games]


def _sample_id_digest(replay: ReplayShard, indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for index in np.asarray(indices, dtype=np.int64):
        digest.update(int(replay.game_id[index]).to_bytes(8, "big", signed=False))
        digest.update(int(replay.ply[index]).to_bytes(2, "big", signed=False))
    return digest.hexdigest()


def _build_active_datasets(
    replay: ReplayShard,
    config: V3Config,
) -> tuple[OnlineD4Dataset, OnlineD4Dataset | None, dict[str, Any]]:
    validation_fraction = 1.0 - config.replay.train_fraction
    common = {
        "c": config.replay.window_c,
        "alpha": config.replay.window_alpha,
        "beta": config.replay.window_beta,
        "validation_fraction": validation_fraction,
        "split_seed": config.run.seed,
    }
    window_indices = active_replay_indices(replay, split=None, **common)
    train_indices = active_replay_indices(replay, split="train", **common)
    validation_indices = active_replay_indices(replay, split="validation", **common)
    if len(train_indices) == 0:
        raise RuntimeError("The active replay window has no training games; adjust the smoke seed/window.")
    train_replay = replay.take(train_indices)
    train_games = sorted({int(game_id) for game_id in train_replay.game_id})
    validation_games = sorted({int(replay.game_id[index]) for index in validation_indices})
    if set(train_games).intersection(validation_games):
        raise RuntimeError("Game-level train/validation split leaked a game across splits.")
    train_sampling_groups = None
    if config.selfplay.opening_temperature_mixture.enabled:
        train_sampling_groups = np.asarray(
            [
                config.selfplay.exploration_variant_index_for_game(int(game_id))
                for game_id in train_replay.game_id
            ],
            dtype=np.int64,
        )
    dataset = OnlineD4Dataset(
        train_replay,
        augmentation_seed=config.run.seed + 701,
        source_positions=train_indices,
        sampling_groups=train_sampling_groups,
    )
    validation_dataset = None
    if len(validation_indices):
        validation_dataset = OnlineD4Dataset(
            replay.take(validation_indices),
            augmentation_seed=config.run.seed + 702,
            source_positions=validation_indices,
        )
    window_start = int(window_indices[0]) if len(window_indices) else len(replay)
    selection = {
        "schema_version": 1,
        "raw_positions": len(replay),
        "window_start": window_start,
        "window_end": len(replay),
        "window_positions": len(window_indices),
        "train_positions": len(train_indices),
        "validation_positions": len(validation_indices),
        "split_seed": config.run.seed,
        "validation_fraction": validation_fraction,
        "train_games": len(train_games),
        "validation_games": len(validation_games),
        "train_sample_id_digest": _sample_id_digest(replay, train_indices),
        "validation_sample_id_digest": _sample_id_digest(replay, validation_indices),
    }
    if train_sampling_groups is not None:
        selection["position_balanced_sampling"] = {
            "schema_version": 1,
            "strategy": "checkpoint_cursor_alternating_v1",
            "target_train_position_fractions": {
                "baseline": 0.5,
                "lowered_opening_temperature": 0.5,
            },
            "available_train_positions": {
                "baseline": int(np.sum(train_sampling_groups == 0)),
                "lowered_opening_temperature": int(np.sum(train_sampling_groups == 1)),
            },
            "absolute_sample_cursor_parity_selects_group": True,
        }
    return dataset, validation_dataset, selection


def _build_learner(
    config: V3Config,
    model: torch.nn.Module,
) -> tuple[V3Learner, torch.optim.Optimizer]:
    optimizer = build_adamw(
        model,
        learning_rate=config.learner.learning_rate_for_positions(0),
        weight_decay=config.learner.weight_decay,
    )
    learner = V3Learner(
        model,
        optimizer,
        device=config.runtime.device,
        batch_size=config.learner.batch_size,
        grad_clip_norm=config.learner.grad_clip_norm,
        sample_seed=config.run.seed + 1701,
        num_workers=config.runtime.num_workers,
        amp=config.runtime.learner_amp,
        policy_loss_weight=config.learner.policy_loss_weight,
        wdl_loss_weight=config.learner.wdl_loss_weight,
        opponent_reply_loss_weight=config.learner.opponent_reply_loss_weight,
        future_occupancy_loss_weight=config.learner.future_occupancy_loss_weight,
        moves_left_loss_weight=config.learner.moves_left_loss_weight,
        future_occupancy_class_weights=config.learner.future_occupancy_class_weights,
        learning_rate_schedule=tuple(
            (stage.start_train_positions, stage.learning_rate)
            for stage in config.learner.lr_schedule
        ),
    )
    return learner, optimizer


def _evaluate_validation(
    model: torch.nn.Module,
    dataset: OnlineD4Dataset | None,
    config: V3Config,
) -> dict[str, Any]:
    if dataset is None or len(dataset) == 0:
        return {"status": "skipped", "reason": "active window has no validation game"}
    device = torch.device(config.runtime.device)
    model.eval()
    positions = 0
    policy_positions = 0
    policy_loss_sum = 0.0
    wdl_loss_sum = 0.0
    opponent_reply_loss_sum = 0.0
    opponent_reply_positions = 0
    opponent_reply_correct = 0
    future_occupancy_loss_sum = 0.0
    future_occupancy_weight_sum = 0.0
    future_occupancy_cells = 0
    future_occupancy_correct = 0
    moves_left_loss_sum = 0.0
    moves_left_correct = 0
    brier_sum = 0.0
    correct = 0
    confidence_rows: list[np.ndarray] = []
    correctness_rows: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(dataset), config.learner.batch_size):
            items = [
                dataset[index]
                for index in range(start, min(start + config.learner.batch_size, len(dataset)))
            ]
            boards = torch.stack([item["board"] for item in items]).to(device)
            policy = torch.stack([item["policy"] for item in items]).to(device)
            policy_weight = torch.stack([item["policy_weight"] for item in items]).to(device)
            labels = torch.stack([item["wdl"] for item in items]).to(device)
            legal = torch.stack([item["legal_mask"] for item in items]).to(device)
            roles = torch.stack([item["role_to_play"] for item in items]).to(device)
            rules = torch.stack([item["rule_features"] for item in items]).to(device)
            opponent_reply = torch.stack([item["opponent_reply"] for item in items]).to(device)
            opponent_reply_mask = torch.stack(
                [item["opponent_reply_mask"] for item in items]
            ).to(device)
            future_occupancy = torch.stack(
                [item["future_occupancy"] for item in items]
            ).to(device)
            future_occupancy_mask = torch.stack(
                [item["future_occupancy_mask"] for item in items]
            ).to(device)
            moves_left = torch.stack([item["moves_left"] for item in items]).to(device)
            output = model(
                boards,
                role_to_play=roles,
                rule_features=rules,
            )
            policy_logits = output.policy_logits
            wdl_logits = output.wdl_logits
            policy_logits = policy_logits.float().masked_fill(~legal, -torch.inf)
            wdl_logits = wdl_logits.float()
            safe_log_policy = torch.where(
                legal,
                F.log_softmax(policy_logits, dim=1),
                torch.zeros_like(policy_logits),
            )
            per_position_policy = -(policy.float() * safe_log_policy).sum(dim=1)
            policy_loss_sum += float((per_position_policy * policy_weight).sum().cpu())
            policy_positions += int(policy_weight.sum().cpu())
            wdl_loss_sum += float(F.cross_entropy(wdl_logits, labels, reduction="sum").cpu())
            reply_loss = F.cross_entropy(
                output.opponent_reply_logits.float(), opponent_reply, reduction="none"
            )
            opponent_reply_loss_sum += float(
                (reply_loss * opponent_reply_mask).sum().cpu()
            )
            opponent_reply_positions += int(opponent_reply_mask.sum().cpu())
            opponent_reply_correct += int(
                (
                    output.opponent_reply_logits.argmax(dim=1).eq(opponent_reply)
                    * opponent_reply_mask.bool()
                ).sum().cpu()
            )
            occupancy_class_weights = torch.as_tensor(
                config.learner.future_occupancy_class_weights,
                dtype=torch.float32,
                device=device,
            )
            occupancy_loss = F.cross_entropy(
                output.future_occupancy_logits.float(),
                future_occupancy,
                weight=occupancy_class_weights,
                reduction="none",
            )
            occupancy_mask_float = future_occupancy_mask.float()
            future_occupancy_loss_sum += float(
                (occupancy_loss * occupancy_mask_float).sum().cpu()
            )
            future_occupancy_weight_sum += float(
                (
                    occupancy_class_weights[future_occupancy]
                    * occupancy_mask_float
                ).sum().cpu()
            )
            future_occupancy_cells += int(future_occupancy_mask.sum().cpu())
            future_occupancy_correct += int(
                (
                    output.future_occupancy_logits.argmax(dim=1).eq(future_occupancy)
                    & future_occupancy_mask
                ).sum().cpu()
            )
            moves_left_loss_sum += float(
                F.cross_entropy(
                    output.moves_left_logits.float(), moves_left, reduction="sum"
                ).cpu()
            )
            moves_left_correct += int(
                output.moves_left_logits.argmax(dim=1).eq(moves_left).sum().cpu()
            )
            probabilities = F.softmax(wdl_logits, dim=1)
            one_hot = F.one_hot(labels, num_classes=3).float()
            brier_sum += float(((probabilities - one_hot) ** 2).sum(dim=1).sum().cpu())
            confidence, prediction = probabilities.max(dim=1)
            batch_correct = prediction.eq(labels)
            correct += int(batch_correct.sum().cpu())
            confidence_rows.append(confidence.cpu().numpy())
            correctness_rows.append(batch_correct.float().cpu().numpy())
            positions += len(items)
    confidences = np.concatenate(confidence_rows)
    correctness = np.concatenate(correctness_rows)
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidences >= lower) & (
            confidences <= upper if upper >= 1.0 else confidences < upper
        )
        if np.any(mask):
            calibration_error += float(mask.mean()) * abs(
                float(confidences[mask].mean()) - float(correctness[mask].mean())
            )
    elapsed = max(time.perf_counter() - started, 1e-12)
    policy_loss = policy_loss_sum / max(policy_positions, 1)
    wdl_loss = wdl_loss_sum / positions
    opponent_reply_loss = opponent_reply_loss_sum / max(opponent_reply_positions, 1)
    future_occupancy_loss = future_occupancy_loss_sum / max(
        future_occupancy_weight_sum, 1.0
    )
    moves_left_loss = moves_left_loss_sum / positions
    total_loss = (
        config.learner.policy_loss_weight * policy_loss
        + config.learner.wdl_loss_weight * wdl_loss
        + config.learner.opponent_reply_loss_weight * opponent_reply_loss
        + config.learner.future_occupancy_loss_weight * future_occupancy_loss
        + config.learner.moves_left_loss_weight * moves_left_loss
    )
    return {
        "status": "complete",
        "positions": positions,
        "policy_positions": policy_positions,
        "value_positions": positions,
        "policy_loss": policy_loss,
        "wdl_loss": wdl_loss,
        "opponent_reply_loss": opponent_reply_loss,
        "future_occupancy_loss": future_occupancy_loss,
        "moves_left_loss": moves_left_loss,
        "total_loss": total_loss,
        "brier_score": brier_sum / positions,
        "calibration_error": calibration_error,
        "wdl_accuracy": correct / positions,
        "opponent_reply_accuracy": opponent_reply_correct
        / max(opponent_reply_positions, 1),
        "future_occupancy_accuracy_unweighted": future_occupancy_correct
        / max(future_occupancy_cells, 1),
        "moves_left_accuracy": moves_left_correct / positions,
        "positions_per_second": positions / elapsed,
    }


def _optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps: list[int] = []
    for state in optimizer.state.values():
        raw = state.get("step", 0)
        steps.append(int(raw.item()) if torch.is_tensor(raw) else int(raw))
    return max(steps, default=0)


def _weights_equal(first: torch.nn.Module, second: torch.nn.Module) -> bool:
    first_state = first.state_dict()
    second_state = second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[name].detach().cpu(), second_state[name].detach().cpu())
        for name in first_state
    )


def _verify_resume_equivalence(
    config: V3Config,
    *,
    checkpoint_path: Path,
    continuous_learner: V3Learner,
    continuous_optimizer: torch.optim.Optimizer,
    dataset: OnlineD4Dataset,
    bucket_state: Mapping[str, Any],
    expected_hash: str,
) -> dict[str, Any]:
    continuous_bucket = TrainTokenBucket.from_state_dict(bucket_state)
    continuous_metrics = continuous_learner.train_steps(
        dataset,
        steps=1,
        token_bucket=continuous_bucket,
    )
    continuous_ids = list(continuous_learner.last_sample_ids)
    continuous_step = _optimizer_step(continuous_optimizer)
    continuous_lr = float(continuous_optimizer.param_groups[0]["lr"])

    resumed_model = build_model(config.model)
    resumed_learner, resumed_optimizer = _build_learner(config, resumed_model)
    saved = load_checkpoint(checkpoint_path, map_location=config.runtime.device)
    saved.restore(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_learner.scaler,
        expected_config_hash=expected_hash,
    )
    resumed_learner.load_state_dict(saved.extra_state["learner_state"])
    resumed_bucket = TrainTokenBucket.from_state_dict(saved.extra_state["token_bucket"])
    resumed_metrics = resumed_learner.train_steps(dataset, steps=1, token_bucket=resumed_bucket)

    checks = {
        "one_update_each": continuous_metrics.steps == resumed_metrics.steps == 1,
        "sample_ids": continuous_ids == resumed_learner.last_sample_ids,
        "learning_rate": continuous_lr == float(resumed_optimizer.param_groups[0]["lr"]),
        "optimizer_step": continuous_step == _optimizer_step(resumed_optimizer),
        "global_step": continuous_learner.global_step == resumed_learner.global_step,
        "weights": _weights_equal(continuous_learner.model, resumed_model),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Checkpoint resume diverged from continuous CPU execution: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "next_sample_ids": [list(sample_id) for sample_id in resumed_learner.last_sample_ids],
        "global_step": resumed_learner.global_step,
        "optimizer_step": _optimizer_step(resumed_optimizer),
        "learning_rate": float(resumed_optimizer.param_groups[0]["lr"]),
        "continuous_metrics": continuous_metrics.to_dict(),
        "resumed_metrics": resumed_metrics.to_dict(),
    }


def _load_replay_from_cursor(
    layout: RunLayout,
    cursor: Mapping[str, Any],
    *,
    config: V3Config,
    expected_hash: str,
) -> ReplayShard:
    shards: list[ReplayShard] = []
    ranged_cursor: int | None = None
    ranged_entries = 0
    for entry in cursor.get("shards", []):
        allowed_entry_schemas = (
            {"path", "checksum_sha256"},
            {"path", "checksum_sha256", "position_start", "position_end"},
        )
        if not isinstance(entry, Mapping) or set(entry) not in allowed_entry_schemas:
            raise ValueError(
                "checkpoint replay shard entries need path/checksum and optional position bounds"
            )
        relative = str(entry["path"])
        shard, manifest = load_replay_shard(layout.root / relative)
        if manifest.get("checksum_sha256") != entry["checksum_sha256"]:
            raise ValueError(f"checkpoint replay checksum differs for {relative}")
        if manifest.get("run_id") != config.run.run_id:
            raise ValueError(f"checkpoint replay run_id differs for {relative}")
        if manifest.get("config_hash") != expected_hash:
            raise ValueError(f"checkpoint replay config hash differs for {relative}")
        if "position_start" in entry:
            start = int(entry["position_start"])
            end = int(entry["position_end"])
            if end - start != len(shard):
                raise ValueError(f"checkpoint replay position range differs for {relative}")
            if ranged_cursor is not None and start != ranged_cursor:
                raise ValueError("checkpoint replay shard position ranges are not contiguous")
            ranged_cursor = end
            ranged_entries += 1
        shards.append(shard)
    if not shards:
        raise ValueError("checkpoint replay cursor does not reference any shards")
    if ranged_entries not in (0, len(shards)):
        raise ValueError("checkpoint replay cursor mixes ranged and legacy shard entries")
    replay = concatenate_replay(shards)
    if len(replay) != int(cursor.get("raw_positions", -1)):
        raise ValueError("checkpoint replay cursor raw_positions does not match loaded shards")
    return replay


def _prepare_audit_replays(
    games: list[GameRecord],
    *,
    config: V3Config,
    generation: int,
    saved_at: str,
) -> tuple[Any, dict[int, Mapping[str, Any]], dict[int, str], list[dict[str, Any]]]:
    selections = select_representative_games(
        games,
        limit=config.runtime.storage.representative_games,
    )
    documents: dict[int, Mapping[str, Any]] = {}
    filenames: dict[int, str] = {}
    references: list[dict[str, Any]] = []
    for selection in selections:
        game = selection.game
        document = game_record_to_replay(
            game,
            run_id=config.run.run_id,
            saved_at=saved_at,
        )
        filename = f"game-{int(game.game_id):08d}-{document['id']}.c4replay.json"
        documents[int(game.game_id)] = document
        filenames[int(game.game_id)] = filename
        references.append(
            {
                "game_id": int(game.game_id),
                "replay_id": str(document["id"]),
                "fingerprint": str(document["fingerprint"]),
                "path": f"samples/g{generation:06d}/{filename}",
            }
        )
    return selections, documents, filenames, references


def _write_audit_replays(
    layout: RunLayout,
    *,
    generation: int,
    selections: Any,
    documents: Mapping[int, Mapping[str, Any]],
    filenames: Mapping[int, str],
    checkpoint_path: Path,
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    sample_dir = layout.samples / f"g{generation:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for game_id, document in sorted(documents.items()):
        target = write_replay_atomic(sample_dir / filenames[game_id], document)
        written.append(str(target.relative_to(layout.root)))
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    audit_index = build_audit_index(
        selections,
        documents,
        run_id=run_id,
        generation=generation,
        checkpoint_id=checkpoint_path.name,
        checkpoint_sha256=checkpoint_sha256,
        created_at=created_at,
        filenames=filenames,
    )
    index_path = write_audit_index_atomic(sample_dir / "audit_index.json", audit_index)
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "audit_index": str(index_path.relative_to(layout.root)),
        "audit_index_sha256": _sha256_file(index_path),
        "replays": written,
    }


def _resume_existing_smoke(config: V3Config, layout: RunLayout, expected_hash: str) -> dict[str, Any]:
    latest = _load_latest_generation_commit(
        layout,
        expected_hash=expected_hash,
        allow_missing=False,
    )
    assert latest is not None
    commit, _commit_path = latest
    checkpoint_path = _run_artifact_path(layout, commit["checkpoint"])
    saved = load_checkpoint(checkpoint_path, map_location=config.runtime.device)
    if saved.generation != int(commit["generation"]):
        raise ValueError("committed checkpoint generation does not match its commit")
    replay = _load_replay_from_cursor(
        layout,
        saved.replay_cursor,
        config=config,
        expected_hash=expected_hash,
    )
    dataset, _validation_dataset, _selection = _build_active_datasets(replay, config)
    model = build_model(config.model)
    learner, optimizer = _build_learner(config, model)
    saved.restore(
        model=model,
        optimizer=optimizer,
        scaler=learner.scaler,
        expected_config_hash=expected_hash,
    )
    learner.load_state_dict(saved.extra_state["learner_state"])
    bucket = TrainTokenBucket.from_state_dict(saved.extra_state["token_bucket"])
    formal_state = FormalLoopState.from_dict(saved.extra_state["formal_loop_state"])
    if formal_state.next_generation != saved.generation + 1:
        raise ValueError("formal loop state generation does not follow the checkpoint")
    if formal_state.next_game_id != int(saved.replay_cursor["next_game_id"]):
        raise ValueError("formal loop state next_game_id differs from replay cursor")
    if formal_state.replay_positions != int(saved.replay_cursor["raw_positions"]):
        raise ValueError("formal loop state replay position count differs from replay cursor")
    metrics = learner.train_steps(dataset, steps=1, token_bucket=bucket)
    if metrics.steps != 1:
        raise RuntimeError("Resume probe did not have enough train tokens for one optimizer update.")
    result = {
        "status": "resume-probe-complete",
        "run_dir": str(layout.root),
        "checkpoint": str(checkpoint_path),
        "restored_global_step": saved.global_step,
        "probe_global_step": learner.global_step,
        "formal_loop_state": formal_state.to_dict(),
        "sample_ids": [list(sample_id) for sample_id in learner.last_sample_ids],
        "metrics": metrics.to_dict(),
    }
    _append_metric(layout, {"stage": "resume_probe", **result})
    return result


def run_smoke(config: V3Config) -> dict[str, Any]:
    """Run or resume the complete synchronous smoke workflow."""

    if not isinstance(config, V3Config):
        raise TypeError("run_smoke requires a resolved V3Config")
    if config.runtime.device != "cpu":
        raise ValueError(
            "The executable smoke contract is CPU-only; use `run` to inspect a GPU plan."
        )
    if config.runtime.actor_processes != 1:
        raise ValueError("CPU smoke requires runtime.actor_processes=1.")
    layout = RunLayout.from_root(resolve_run_root(config))
    expected_hash = lineage_config_hash(config)
    _seed_runtime(config)
    if config.run.resume:
        return _resume_existing_smoke(config, layout, expected_hash)
    if layout.run_manifest.exists():
        raise FileExistsError(
            f"V3 run already exists: {layout.root}. Use --resume or choose another --run-dir."
        )

    layout.create()
    code_commit = _git_commit()
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "status": "running",
        "created_at": _utc_now(),
        "config_hash": expected_hash,
        "git_commit": code_commit,
        "code_version": __version__,
    }
    _atomic_write_json(layout.run_manifest, run_manifest)
    artifact_config = replace(config, run=replace(config.run, run_dir=str(layout.root)))
    _atomic_write_json(layout.resolved_config, artifact_config.to_dict())

    try:
        generation = 0
        search_stage = config.selfplay.stage_for_generation(generation)
        accepted_predictor, accepted_model_id = _accepted_predictor(layout, config)
        actor_batch = run_self_play_actor_pool(
            config,
            producer_model_id=accepted_model_id,
            start_game_id=0,
            generation=generation,
        )
        games = list(actor_batch.games)
        producer_model_id = games[0].producer_model_id
        if any(game.producer_model_id != producer_model_id for game in games):
            raise RuntimeError("A self-play batch changed producer model mid-batch.")

        replay_parts: list[ReplayShard] = []
        shard_paths: list[Path] = []
        shard_manifests: list[Mapping[str, Any]] = []
        for shard_index, shard_games in enumerate(
            _game_batches(games, config.replay.shard_games)
        ):
            shard_replay = ReplayShard.from_samples(
                chain.from_iterable(game.samples for game in shard_games)
            )
            if len(shard_replay) == 0:
                raise RuntimeError("Self-play produced an empty replay shard.")
            shard_name = (
                f"g{generation:06d}_s{shard_index:04d}_"
                f"games{shard_games[0].game_id:08d}-{shard_games[-1].game_id:08d}.npz"
            )
            shard_path = layout.raw_replay / shard_name
            shard_manifest = write_replay_shard(
                shard_path,
                shard_replay,
                {
                    "run_id": config.run.run_id,
                    "generation": generation,
                    "producer_model_id": producer_model_id,
                    "seed_range": {
                        "start": shard_games[0].seed,
                        "end": shard_games[-1].seed,
                    },
                    "results": _result_counts(shard_games),
                    "search_config": {
                        "active_stage": asdict(search_stage),
                        "exploration_phases": [
                            asdict(phase) for phase in config.selfplay.exploration_phases
                        ],
                        "opening_temperature_mixture": asdict(
                            config.selfplay.opening_temperature_mixture
                        ),
                        "cpuct": config.selfplay.cpuct,
                        "virtual_loss": config.selfplay.virtual_loss,
                        "mcts_lanes_per_actor": config.runtime.mcts_lanes_per_actor,
                    },
                    "rule_registry_hash": config.selfplay.rule_registry_hash,
                    "config_hash": expected_hash,
                    "git_commit": code_commit,
                },
            )
            replay_parts.append(shard_replay)
            shard_paths.append(shard_path)
            shard_manifests.append(shard_manifest)
        replay = concatenate_replay(replay_parts)
        if not np.any(replay.search_kind == 1):
            raise RuntimeError("Self-play produced no full-search policy target.")

        health = _selfplay_health(
            games,
            expected_search_sims={
                "full": search_stage.full_search_sims,
                "fast": search_stage.fast_search_sims,
            },
            exploration_phases=(
                asdict(phase) for phase in config.selfplay.exploration_phases
            ),
        )
        if config.selfplay.opening_temperature_mixture.enabled:
            mixture_health: dict[str, Any] = {}
            for variant in ("baseline", "lowered_opening_temperature"):
                variant_games = [
                    game for game in games if game.exploration_variant == variant
                ]
                variant_selfplay = config.selfplay.for_exploration_variant(variant)
                mixture_health[variant] = {
                    "games": len(variant_games),
                    "raw_positions": sum(len(game.samples) for game in variant_games),
                    "health": _selfplay_health(
                        variant_games,
                        expected_search_sims={
                            "full": search_stage.full_search_sims,
                            "fast": search_stage.fast_search_sims,
                        },
                        exploration_phases=(
                            asdict(phase)
                            for phase in variant_selfplay.exploration_phases
                        ),
                    ),
                }
            health["opening_temperature_mixture"] = {
                "schema_version": 1,
                "assignment": "alternating_game_id_v1",
                "variants": mixture_health,
            }
        _append_metric(
            layout,
            {
                "stage": "selfplay",
                "producer_model_id": producer_model_id,
                "games": len(games),
                "value_positions": len(replay),
                "policy_positions": int(np.sum(replay.search_kind == 1)),
                "full_search_positions": sum(game.full_search_positions for game in games),
                "fast_search_positions": sum(game.fast_search_positions for game in games),
                "simulations": sum(game.total_simulations for game in games),
                "inference_batches": sum(game.inference_batches for game in games),
                "inference_positions": sum(game.inference_positions for game in games),
                "mean_inference_batch": (
                    sum(game.inference_positions for game in games)
                    / max(sum(game.inference_batches for game in games), 1)
                ),
                "max_inference_batch": max(game.max_inference_batch for game in games),
                "search_stage": asdict(search_stage),
                "actor_runtime": actor_batch.metrics.to_dict(),
                "health": health,
            },
        )

        dataset, validation_dataset, selection = _build_active_datasets(replay, config)
        selection.update(
            {
                "config_hash": expected_hash,
                "input_shards": [
                    {
                        "path": str(path.relative_to(layout.root)),
                        "checksum_sha256": manifest["checksum_sha256"],
                    }
                    for path, manifest in zip(shard_paths, shard_manifests, strict=True)
                ],
            }
        )
        shuffle_manifest_path = layout.shuffle / f"selection_g{generation:06d}.json"
        _atomic_write_json(shuffle_manifest_path, selection)

        model = build_model(config.model)
        learner, optimizer = _build_learner(config, model)
        bucket = TrainTokenBucket(config.replay.train_tokens_per_raw_position)
        bucket.add(len(replay))
        requested_steps = config.learner.max_optimizer_steps_per_cycle
        permitted_positions = bucket.consumable(requested_steps * config.learner.batch_size)
        expected_steps = min(
            requested_steps,
            int(np.ceil(permitted_positions / config.learner.batch_size))
            if permitted_positions
            else 0,
        )
        learner_metrics = learner.train_steps(
            dataset,
            steps=requested_steps,
            token_bucket=bucket,
        )
        if learner_metrics.steps != expected_steps or learner_metrics.steps < 1:
            raise RuntimeError(
                "Learner step cap and token bucket produced an unexpected update count."
            )
        _append_metric(layout, {"stage": "learner", **learner_metrics.to_dict()})
        validation_metrics = _evaluate_validation(model, validation_dataset, config)
        _append_metric(layout, {"stage": "validation", **validation_metrics})

        candidate_model_id = (
            f"candidate-g{generation:06d}-s{learner.global_step:08d}-d{len(replay):08d}"
        )
        candidate_path = layout.candidates / f"{candidate_model_id}.pt"
        _atomic_save_model_artifact(
            candidate_path,
            model=model,
            model_config=model_config_dict(config.model),
            metadata={
                "candidate_model_id": candidate_model_id,
                "parent_model_id": accepted_model_id,
                "global_step": learner.global_step,
                "raw_data_end": len(replay),
                "config_hash": expected_hash,
                "git_commit": code_commit,
            },
        )

        openings = build_openings(
            config.gate.max_opening_pairs,
            run_seed=config.run.seed,
            prefix_lengths=config.gate.opening_depths,
            rule_id=config.selfplay.rule_id,
        )
        opening_manifest_path = layout.manifests / "gate_openings.json"
        write_opening_manifest(opening_manifest_path, openings)
        candidate_predictor = TorchPredictor(model, config.runtime.device)
        gate_search_sims = config.gate.search_sims_for_generation(generation)
        gate_evaluation_runtime: list[dict[str, Any]] = []
        gate_results, gate_control_results, gate_decision, gate_looks = _run_sequential_gate(
            config,
            generation=generation,
            openings=openings,
            candidate_predictor=candidate_predictor,
            incumbent_predictor=accepted_predictor,
            runtime_records=gate_evaluation_runtime,
            allow_random_bootstrap_control=accepted_model_id is None,
        )
        gate_payload = {
            "schema_version": 1,
            "candidate_model_id": candidate_model_id,
            "incumbent_model_id": accepted_model_id or "random",
            "opening_manifest": str(opening_manifest_path.relative_to(layout.root)),
            "search_sims": gate_search_sims,
            "initial_pairs": config.gate.initial_opening_pairs,
            "pair_increment": config.gate.pair_increment,
            "max_pairs": config.gate.max_opening_pairs,
            "sequential_looks": config.gate.sequential_looks(),
            "decision_confidence": config.gate.decision_confidence(),
            "descriptive_confidence": config.gate.confidence,
            "games": [asdict(result) for result in gate_results],
            "role_control_games": [asdict(result) for result in gate_control_results],
            "role_control_baseline": (
                "random_bootstrap"
                if config.gate.role_guard_mode == "relative_noninferiority"
                and accepted_model_id is None
                else (
                    "accepted_champion"
                    if config.gate.role_guard_mode == "relative_noninferiority"
                    else None
                )
            ),
            "looks": gate_looks,
            "evaluation_runtime": gate_evaluation_runtime,
            **gate_decision.to_dict(),
        }
        gate_path = layout.metrics / f"gate_g{generation:06d}.json"
        _atomic_write_json(gate_path, gate_payload)
        _append_metric(
            layout,
            {
                "stage": "gate",
                "candidate_model_id": candidate_model_id,
                "incumbent_model_id": accepted_model_id or "random",
                "verdict": gate_decision.verdict,
                "overall_point_score": gate_decision.summary.overall.point_score,
                "ci_lower": gate_decision.summary.ci_lower,
                "ci_upper": gate_decision.summary.ci_upper,
            },
        )

        accepted_after = accepted_model_id
        candidate_final_path = candidate_path
        if gate_decision.verdict == "accept":
            candidate_final_path = layout.accepted / candidate_path.name
            os.replace(candidate_path, candidate_final_path)
            accepted_after = candidate_model_id
        elif gate_decision.verdict == "reject":
            candidate_final_path = layout.rejected / candidate_path.name
            os.replace(candidate_path, candidate_final_path)
        # Inconclusive candidates stay available for the configured pair increments.

        formal_state = FormalLoopState(
            next_generation=generation,
            next_game_id=0,
            replay_positions=0,
            train_positions_consumed=0,
            last_candidate_train_positions=0,
            accepted_model_id=accepted_model_id,
        ).finish_generation(
            next_game_id=games[-1].game_id + 1,
            replay_positions=len(replay),
            train_positions_consumed=int(bucket.total_positions_consumed),
        )
        formal_state = formal_state.emit_candidate(
            PendingCandidateState(
                candidate_model_id=candidate_model_id,
                candidate_path=str(candidate_final_path.relative_to(layout.root)).replace("\\", "/"),
                incumbent_model_id=accepted_model_id or "random",
                gate_path=str(gate_path.relative_to(layout.root)).replace("\\", "/"),
                opening_manifest=str(opening_manifest_path.relative_to(layout.root)).replace("\\", "/"),
                pairs_evaluated=len(openings),
                max_pairs=config.gate.max_opening_pairs,
            )
        )
        if gate_decision.verdict == "accept":
            formal_state = formal_state.resolve_pending_candidate(accepted=True)
        elif gate_decision.verdict == "reject":
            formal_state = formal_state.resolve_pending_candidate(accepted=False)

        audit_created_at = str(run_manifest["created_at"])
        audit_selections, audit_documents, audit_filenames, audit_references = (
            _prepare_audit_replays(
                games,
                config=config,
                generation=generation,
                saved_at=audit_created_at,
            )
        )
        replay_cursor = {
            "shards": [
                {
                    "path": str(path.relative_to(layout.root)),
                    "checksum_sha256": manifest["checksum_sha256"],
                }
                for path, manifest in zip(shard_paths, shard_manifests, strict=True)
            ],
            "raw_positions": len(replay),
            "window_start": selection["window_start"],
            "window_end": selection["window_end"],
            "next_game_id": games[-1].game_id + 1,
        }
        checkpoint = CheckpointV1.capture(
            model=model,
            optimizer=optimizer,
            global_step=learner.global_step,
            generation=generation,
            replay_cursor=replay_cursor,
            sample_ids=learner.last_sample_ids,
            accepted_model_id=accepted_after,
            candidate_model_id=candidate_model_id,
            config_hash=expected_hash,
            code_version=f"{__version__}+{code_commit[:12]}",
            recent_evaluation=gate_decision.to_dict(),
            scaler=learner.scaler,
            extra_state={
                "learner_state": learner.state_dict(),
                "token_bucket": bucket.state_dict(),
                "model_config": model_config_dict(config.model),
                "train_positions_consumed": bucket.total_positions_consumed,
                "formal_loop_state": formal_state.to_dict(),
                "audit_replays": audit_references,
            },
        )
        checkpoint_target = (
            layout.checkpoints / f"g{generation:06d}-s{learner.global_step:08d}.pt"
        )
        if checkpoint_target.exists():
            raise FileExistsError(f"immutable generation checkpoint already exists: {checkpoint_target}")
        checkpoint_path = save_checkpoint(checkpoint_target, checkpoint)
        audit_artifacts = _write_audit_replays(
            layout,
            generation=generation,
            selections=audit_selections,
            documents=audit_documents,
            filenames=audit_filenames,
            checkpoint_path=checkpoint_path,
            run_id=config.run.run_id,
            created_at=audit_created_at,
        )
        resume_verification = _verify_resume_equivalence(
            config,
            checkpoint_path=checkpoint_path,
            continuous_learner=learner,
            continuous_optimizer=optimizer,
            dataset=dataset,
            bucket_state=bucket.state_dict(),
            expected_hash=expected_hash,
        )
        _append_metric(layout, {"stage": "resume_verification", **resume_verification})

        accepted_relative = (
            f"accepted/{accepted_after}.pt" if accepted_after is not None else None
        )
        if accepted_relative is not None and not (layout.root / accepted_relative).is_file():
            raise RuntimeError("Committed accepted model artifact is missing.")
        generation_commit_path = layout.generation_commits / f"g{generation:06d}.json"
        generation_commit = {
            "schema_version": 1,
            "run_id": config.run.run_id,
            "generation": generation,
            "committed_at": _utc_now(),
            "config_hash": expected_hash,
            "checkpoint": str(checkpoint_path.relative_to(layout.root)),
            "checkpoint_sha256": audit_artifacts["checkpoint_sha256"],
            "accepted_model_id": accepted_after,
            "accepted_model_path": accepted_relative,
            "accepted_model_sha256": (
                _sha256_file(layout.root / accepted_relative)
                if accepted_relative is not None
                else None
            ),
            "candidate_model_id": candidate_model_id,
            "candidate_path": str(candidate_final_path.relative_to(layout.root)),
            "candidate_sha256": _sha256_file(candidate_final_path),
            "gate_verdict": gate_decision.verdict,
            "health_watch_warnings": health["watch_warnings"],
            "replay_shards": replay_cursor["shards"],
            "replay_raw_positions": replay_cursor["raw_positions"],
            "audit_index": audit_artifacts["audit_index"],
            "audit_index_sha256": audit_artifacts["audit_index_sha256"],
            "next_game_id": replay_cursor["next_game_id"],
        }
        if generation_commit_path.exists():
            raise FileExistsError(
                f"immutable generation commit already exists: {generation_commit_path}"
            )
        _atomic_write_json(generation_commit_path, generation_commit)
        generation_commit_sha256 = _sha256_file(generation_commit_path)
        _atomic_write_json(
            layout.manifests / "latest_generation.json",
            {
                "schema_version": 1,
                "generation": generation,
                "commit": str(generation_commit_path.relative_to(layout.root)),
                "commit_sha256": generation_commit_sha256,
            },
        )

        result = {
            "status": "complete",
            "run_id": config.run.run_id,
            "run_dir": str(layout.root),
            "generation": generation,
            "producer_model_id": producer_model_id,
            "games": len(games),
            "raw_positions": len(replay),
            "full_policy_positions": int(np.sum(replay.search_kind == 1)),
            "active_train_positions": selection["train_positions"],
            "active_validation_positions": selection["validation_positions"],
            "optimizer_steps": learner_metrics.steps,
            "learner_metrics": learner_metrics.to_dict(),
            "validation_metrics": validation_metrics,
            "selfplay_health": health,
            "actor_runtime": actor_batch.metrics.to_dict(),
            "formal_loop_state": formal_state.to_dict(),
            "gate_verdict": gate_decision.verdict,
            "gate": gate_decision.to_dict(),
            "checkpoint": str(checkpoint_path),
            "resume_verification": resume_verification,
            "artifacts": {
                "replay": str(shard_paths[0]),
                "replay_shards": [str(path) for path in shard_paths],
                "replay_manifests": [
                    str(replay_manifest_path(path)) for path in shard_paths
                ],
                "shuffle_manifest": str(shuffle_manifest_path),
                "opening_manifest": str(opening_manifest_path),
                "gate": str(gate_path),
                "audit_index": str(layout.root / audit_artifacts["audit_index"]),
                "audit_replays": [
                    str(layout.root / relative) for relative in audit_artifacts["replays"]
                ],
                "generation_commit": str(generation_commit_path),
            },
        }
        run_manifest.update({"status": "complete", "completed_at": _utc_now(), "result": result})
        _atomic_write_json(layout.run_manifest, run_manifest)
        return result
    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(layout.run_manifest, run_manifest)
        raise


def formal_run_status(config: V3Config) -> dict[str, Any]:
    """Return the bounded formal-run plan and its remaining readiness gates."""

    generation = 0
    search_stage = config.selfplay.stage_for_generation(generation)
    gate_sims = config.gate.search_sims_for_generation(generation)
    configured_devices = tuple(
        dict.fromkeys((config.runtime.device, *config.runtime.selfplay_devices))
    )
    if config.runtime.device.startswith("cuda"):
        hardware = plan_hardware(
            configured_devices,
            learner_device=config.runtime.device,
            selfplay_devices=(
                config.runtime.selfplay_devices or (config.runtime.device,)
            ),
            actors=config.runtime.actor_processes,
            mcts_lanes=config.runtime.mcts_lanes_per_actor,
            inference_batch_limit=config.runtime.inference_batch_size,
            cpu_cores=available_cpu_count(),
        ).to_dict()
        hardware["inventory_verified"] = False
        hardware["warnings"] = list(hardware["warnings"])
        hardware["warnings"].append(
            {
                "code": "cuda_inventory_unverified",
                "message": "Offline `run` planning did not probe CUDA device availability.",
            }
        )
    else:
        hardware = {
            "mode": "cpu_reference",
            "learner_device": "cpu",
            "stages_may_overlap": False,
            "ddp_enabled": False,
            "actor_count": 1,
            "warnings": [],
        }
    storage = config.runtime.storage
    retention_policy = RetentionPolicy(
        active_window_margin=storage.active_window_margin,
        keep_recent_by_kind=(
            ("raw_replay", 0),
            ("checkpoint", storage.keep_checkpoints),
            ("accepted", storage.keep_accepted),
            ("rejected", storage.keep_rejected),
        ),
        soft_used_fraction=storage.soft_used_fraction,
        hard_free_bytes=int(storage.hard_free_gib * GIB),
    )
    provisional_auxiliary = (
        tuple(config.learner.future_occupancy_class_weights) == (1.0, 1.0, 1.0)
        and config.learner.opponent_reply_loss_weight == 0.15
        and config.learner.future_occupancy_loss_weight == 0.15
        and config.learner.moves_left_loss_weight == 0.05
    )
    blocking_items: list[str] = []
    if provisional_auxiliary:
        blocking_items.append("P6 auxiliary loss and occupancy class weights are still provisional")
    if storage.mode != "archive_ack_prune":
        blocking_items.append("formal execution requires archive_ack_prune storage mode")
    if storage.hard_free_gib < 10.0:
        blocking_items.append("formal execution requires at least a 10-GiB hard reserve")
    return {
        "status": "bounded-formal-run-available",
        "execution_requires_explicit_flag": True,
        "production_ready": not blocking_items,
        "run_id": config.run.run_id,
        "run_dir": str(resolve_run_root(config)),
        "active_generation_zero_plan": {
            "selfplay": asdict(search_stage),
            "gate_search_sims": gate_sims,
            "gate_sequential_looks": config.gate.sequential_looks(),
            "gate_decision_confidence": config.gate.decision_confidence(),
            "candidate_train_positions": config.gate.candidate_train_positions,
            "bootstrap_candidate_train_positions": (
                config.gate.bootstrap_candidate_train_positions
            ),
            "learning_rate": config.learner.learning_rate_for_positions(0),
        },
        "hardware_plan": hardware,
        "storage_plan": {
            "mode": storage.mode,
            "policy": asdict(retention_policy),
            "bundle_target_gib": storage.bundle_target_gib,
            "representative_games": storage.representative_games,
            "deletion_enabled": "explicit_receipt_gated_command_only",
            "receipt_required_before_prune": True,
        },
        "blocking_items": blocking_items,
        "message": (
            "The synchronous multi-generation scheduler is available only through explicit "
            "`run --execute --max-train-positions ...`; it stops at committed generation "
            "boundaries and refuses execution while any listed readiness gate remains."
        ),
    }


__all__ = [
    "formal_run_status",
    "lineage_config_hash",
    "repository_root",
    "resolve_run_root",
    "run_smoke",
]
