"""Bounded synchronous multi-generation runner for formal V3 lineages.

The runner intentionally favors recoverability over maximum overlap.  Signals
request a drain at the next committed generation boundary; every generation is
wrapped by the no-clobber coordinator lock and checksum-bound draft journal.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
from dataclasses import asdict, replace
from itertools import chain
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from . import __version__
from .actor_runtime import run_self_play_actor_pool
from .checkpoint import CheckpointV1, load_checkpoint, save_checkpoint
from .config import V3Config
from .evaluation import build_openings, write_opening_manifest
from .evaluation_runtime import EvaluationModelSource
from .formal_journal import (
    CoordinatorLock,
    GenerationJournal,
    reconcile_generation_drafts,
)
from .formal_state import FormalLoopState, PendingCandidateState
from .layout import RunLayout
from .model import TorchPredictor, build_model
from .learner import OnlineD4Dataset
from .pipeline import (
    _append_metric,
    _atomic_save_model_artifact,
    _atomic_write_json,
    _build_learner,
    _evaluate_validation,
    _game_batches,
    _git_commit,
    _load_latest_generation_commit,
    _load_model_artifact,
    _load_replay_from_cursor,
    _prepare_audit_replays,
    _result_counts,
    _run_sequential_gate,
    _selfplay_health,
    _sha256_file,
    _sample_id_digest,
    _utc_now,
    _validate_generation_commit,
    _write_audit_replays,
    lineage_config_hash,
    resolve_run_root,
)
from .replay import (
    ReplayShard,
    TrainTokenBucket,
    concatenate_replay,
    growing_window_size,
    replay_manifest_path,
    replay_ready_path,
    stable_split_mask,
    write_replay_shard,
)
from .stability import GenerationStabilityMetrics, assess_stability


class _DrainRequest:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handle(self, signal_number: int, _frame: object) -> None:
        self.requested = True
        self.signal_number = int(signal_number)


def _provisional_auxiliary(config: V3Config) -> bool:
    return (
        tuple(config.learner.future_occupancy_class_weights) == (1.0, 1.0, 1.0)
        and config.learner.opponent_reply_loss_weight == 0.15
        and config.learner.future_occupancy_loss_weight == 0.15
        and config.learner.moves_left_loss_weight == 0.05
    )


def _disk_status(layout: RunLayout, config: V3Config) -> dict[str, Any]:
    usage = shutil.disk_usage(layout.root)
    storage = config.runtime.storage
    hard = int(storage.hard_free_gib * 1024**3)
    staging_headroom = int(storage.bundle_target_gib * 1024**3)
    used_fraction = usage.used / float(usage.total)
    return {
        "capacity_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_fraction": used_fraction,
        "hard_free_bytes": hard,
        "archive_headroom_bytes": hard + staging_headroom,
        "hard_reserve_breached": usage.free <= hard,
        "archive_required": (
            used_fraction >= storage.soft_used_fraction
            or usage.free <= hard + staging_headroom
        ),
    }


def _run_relative(layout: RunLayout, path: Path) -> str:
    return path.resolve().relative_to(layout.root).as_posix()


def _accepted_assets(
    layout: RunLayout,
    config: V3Config,
    commit: Mapping[str, Any] | None,
) -> tuple[torch.nn.Module | None, TorchPredictor | None, str | None]:
    if commit is None or commit.get("accepted_model_id") is None:
        return None, None, None
    path = layout.root / str(commit["accepted_model_path"])
    model = _load_model_artifact(path, config)
    model_id = str(commit["accepted_model_id"])
    return model, TorchPredictor(model, config.runtime.device), model_id


def _record(journal: GenerationJournal, path: Path, kind: str) -> None:
    journal.record_artifact(path, kind=kind)


def _trim_replay_cursor(
    replay: ReplayShard,
    entries: list[dict[str, Any]],
    *,
    retained_position_start: int,
    cumulative_positions: int,
    active_window_positions: int,
    margin: float,
) -> tuple[ReplayShard, list[dict[str, Any]], int]:
    retain_positions = int(math.ceil(active_window_positions * float(margin)))
    desired_start = max(0, cumulative_positions - retain_positions)
    kept = [row for row in entries if int(row["position_end"]) > desired_start]
    if not kept:
        raise RuntimeError("replay retention removed every active shard")
    new_start = int(kept[0]["position_start"])
    if new_start < retained_position_start or new_start > desired_start:
        raise RuntimeError("replay retention margin does not cover the active window")
    local_start = new_start - retained_position_start
    trimmed = replay.take(np.arange(local_start, len(replay), dtype=np.int64))
    if new_start + len(trimmed) != cumulative_positions:
        raise RuntimeError("trimmed replay cursor does not end at the cumulative position")
    return trimmed, kept, new_start


def _build_global_active_datasets(
    replay: ReplayShard,
    config: V3Config,
    *,
    retained_position_start: int,
    cumulative_positions: int,
) -> tuple[OnlineD4Dataset, OnlineD4Dataset | None, dict[str, Any]]:
    if retained_position_start < 0 or retained_position_start + len(replay) != cumulative_positions:
        raise ValueError("retained replay range does not end at the cumulative cursor")
    window_size = growing_window_size(
        cumulative_positions,
        c=config.replay.window_c,
        alpha=config.replay.window_alpha,
        beta=config.replay.window_beta,
    )
    global_window_start = cumulative_positions - window_size
    if retained_position_start > global_window_start:
        raise RuntimeError("retained replay no longer covers the configured active window")
    local_window_start = global_window_start - retained_position_start
    window_indices = np.arange(local_window_start, len(replay), dtype=np.int64)
    validation_fraction = 1.0 - config.replay.train_fraction
    train_mask = stable_split_mask(
        replay.game_id[window_indices],
        split="train",
        validation_fraction=validation_fraction,
        split_seed=config.run.seed,
    )
    validation_mask = ~train_mask
    train_indices = window_indices[train_mask]
    validation_indices = window_indices[validation_mask]
    if len(train_indices) == 0:
        raise RuntimeError("active formal replay window has no training games")
    train_replay = replay.take(train_indices)
    train_games = sorted({int(value) for value in train_replay.game_id})
    validation_games = sorted({int(replay.game_id[index]) for index in validation_indices})
    if set(train_games).intersection(validation_games):
        raise RuntimeError("formal replay split leaked a game between train and validation")
    dataset = OnlineD4Dataset(
        train_replay,
        augmentation_seed=config.run.seed + 701,
        source_positions=train_indices + retained_position_start,
    )
    validation = None
    if len(validation_indices):
        validation = OnlineD4Dataset(
            replay.take(validation_indices),
            augmentation_seed=config.run.seed + 702,
            source_positions=validation_indices + retained_position_start,
        )
    return dataset, validation, {
        "schema_version": 1,
        "raw_positions": cumulative_positions,
        "retained_position_start": retained_position_start,
        "retained_positions": len(replay),
        "window_start": global_window_start,
        "window_end": cumulative_positions,
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


def _resume_ready_commit(layout: RunLayout, expected_hash: str) -> None:
    for row in reconcile_generation_drafts(layout):
        status = row["status"]
        if status == "committed":
            continue
        generation = int(row.get("generation", -1))
        if status == "resume_precommit":
            journal = GenerationJournal.load(layout, generation)
            commit_path = journal.publish_commit()
            _validate_generation_commit(layout, commit_path, expected_hash=expected_hash)
            continue
        raise RuntimeError(
            "unfinished generation draft requires explicit operator recovery: "
            + json.dumps(row, ensure_ascii=False, sort_keys=True)
        )


def _load_training_state(
    config: V3Config,
    layout: RunLayout,
    expected_hash: str,
) -> tuple[
    torch.nn.Module,
    Any,
    Any,
    TrainTokenBucket,
    FormalLoopState,
    ReplayShard | None,
    list[dict[str, Any]],
    int,
    list[GenerationStabilityMetrics],
    dict[str, Any] | None,
]:
    latest = _load_latest_generation_commit(
        layout, expected_hash=expected_hash, allow_missing=True
    )
    if latest is None:
        model = build_model(config.model)
        learner, optimizer = _build_learner(config, model)
        return (
            model,
            learner,
            optimizer,
            TrainTokenBucket(config.replay.train_tokens_per_raw_position),
            FormalLoopState(),
            None,
            [],
            0,
            [],
            None,
        )
    commit, _commit_path = latest
    checkpoint_path = layout.root / str(commit["checkpoint"])
    saved = load_checkpoint(checkpoint_path, map_location=config.runtime.device)
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
        raise ValueError("formal state does not follow the committed checkpoint generation")
    if formal_state.next_game_id != int(saved.replay_cursor["next_game_id"]):
        raise ValueError("formal state next game ID differs from the replay cursor")
    if formal_state.replay_positions != int(
        saved.replay_cursor.get("cumulative_raw_positions", saved.replay_cursor["raw_positions"])
    ):
        raise ValueError("formal state cumulative replay count differs from the replay cursor")
    replay = _load_replay_from_cursor(
        layout,
        saved.replay_cursor,
        config=config,
        expected_hash=expected_hash,
    )
    return (
        model,
        learner,
        optimizer,
        bucket,
        formal_state,
        replay,
        [dict(row) for row in saved.replay_cursor["shards"]],
        int(saved.replay_cursor.get("retained_position_start", 0)),
        [
            GenerationStabilityMetrics.from_mapping(row)
            for row in saved.extra_state.get("stability_history", [])
        ],
        commit,
    )


def _run_generation(
    config: V3Config,
    layout: RunLayout,
    *,
    expected_hash: str,
    code_commit: str,
    run_created_at: str,
    model: torch.nn.Module,
    learner: Any,
    optimizer: Any,
    bucket: TrainTokenBucket,
    formal_state: FormalLoopState,
    replay: ReplayShard | None,
    replay_entries: list[dict[str, Any]],
    retained_position_start: int,
    stability_history: list[GenerationStabilityMetrics],
    latest_commit: Mapping[str, Any] | None,
    remaining_train_positions: int,
) -> tuple[
    dict[str, Any],
    ReplayShard,
    list[dict[str, Any]],
    int,
    list[GenerationStabilityMetrics],
    FormalLoopState,
    dict[str, Any],
]:
    generation = formal_state.next_generation
    journal = GenerationJournal.begin(
        layout,
        run_id=config.run.run_id,
        generation=generation,
        config_hash=expected_hash,
    )
    accepted_model, accepted_predictor, accepted_model_id = _accepted_assets(
        layout, config, latest_commit
    )
    accepted_state = None
    if accepted_model is not None:
        accepted_state = {
            name: tensor.detach().cpu() for name, tensor in accepted_model.state_dict().items()
        }
    search_stage = config.selfplay.stage_for_generation(generation)
    actor_batch = run_self_play_actor_pool(
        config,
        accepted_model_state=accepted_state,
        producer_model_id=accepted_model_id,
        start_game_id=formal_state.next_game_id,
        generation=generation,
    )
    games = list(actor_batch.games)
    producer_model_id = games[0].producer_model_id
    if producer_model_id != (accepted_model_id or "random") or any(
        game.producer_model_id != producer_model_id for game in games
    ):
        raise RuntimeError("self-play producer differs from the committed accepted champion")

    new_parts: list[ReplayShard] = []
    new_entries: list[dict[str, Any]] = []
    new_position_offset = formal_state.replay_positions
    for shard_index, shard_games in enumerate(_game_batches(games, config.replay.shard_games)):
        shard = ReplayShard.from_samples(chain.from_iterable(game.samples for game in shard_games))
        if len(shard) == 0:
            raise RuntimeError("self-play produced an empty replay shard")
        shard_path = layout.raw_replay / (
            f"g{generation:06d}_s{shard_index:04d}_"
            f"games{shard_games[0].game_id:08d}-{shard_games[-1].game_id:08d}.npz"
        )
        position_start = new_position_offset + sum(len(part) for part in new_parts)
        manifest = write_replay_shard(
            shard_path,
            shard,
            {
                "run_id": config.run.run_id,
                "generation": generation,
                "producer_model_id": producer_model_id,
                "seed_range": {"start": shard_games[0].seed, "end": shard_games[-1].seed},
                "results": _result_counts(shard_games),
                "search_config": {
                    "active_stage": asdict(search_stage),
                    "exploration_phases": [asdict(row) for row in config.selfplay.exploration_phases],
                    "cpuct": config.selfplay.cpuct,
                    "virtual_loss": config.selfplay.virtual_loss,
                    "mcts_lanes_per_actor": config.runtime.mcts_lanes_per_actor,
                },
                "position_range": {"start": position_start, "end": position_start + len(shard)},
                "rule_registry_hash": config.selfplay.rule_registry_hash,
                "config_hash": expected_hash,
                "git_commit": code_commit,
            },
        )
        for artifact, kind in (
            (shard_path, "raw_replay"),
            (replay_manifest_path(shard_path), "raw_replay_manifest"),
            (replay_ready_path(shard_path), "raw_replay_ready"),
        ):
            _record(journal, artifact, kind)
        new_parts.append(shard)
        new_entries.append(
            {
                "path": _run_relative(layout, shard_path),
                "checksum_sha256": manifest["checksum_sha256"],
                "position_start": position_start,
                "position_end": position_start + len(shard),
            }
        )
    new_replay = concatenate_replay(new_parts)
    replay = new_replay if replay is None else concatenate_replay((replay, new_replay))
    replay_entries = [*replay_entries, *new_entries]
    if not np.any(new_replay.search_kind == 1):
        raise RuntimeError("self-play produced no full-search policy target")
    health = _selfplay_health(
        games,
        expected_search_sims={"full": search_stage.full_search_sims, "fast": search_stage.fast_search_sims},
        exploration_phases=(
            asdict(phase) for phase in config.selfplay.exploration_phases
        ),
    )
    _append_metric(
        layout,
        {
            "stage": "selfplay",
            "generation": generation,
            "producer_model_id": producer_model_id,
            "games": len(games),
            "new_raw_positions": len(new_replay),
            "cumulative_raw_positions": formal_state.replay_positions + len(new_replay),
            "actor_runtime": actor_batch.metrics.to_dict(),
            "health": health,
        },
    )

    cumulative_positions = formal_state.replay_positions + len(new_replay)
    dataset, validation_dataset, selection = _build_global_active_datasets(
        replay,
        config,
        retained_position_start=retained_position_start,
        cumulative_positions=cumulative_positions,
    )
    selection.update({"config_hash": expected_hash, "input_shards": replay_entries})
    selection_path = layout.shuffle / f"selection_g{generation:06d}.json"
    _atomic_write_json(selection_path, selection)
    _record(journal, selection_path, "shuffle_manifest")
    bucket.add(len(new_replay))
    learner_metrics = learner.train_steps(
        dataset,
        steps=config.learner.max_optimizer_steps_per_cycle,
        token_bucket=bucket,
        position_limit=remaining_train_positions,
    )
    _append_metric(layout, {"stage": "learner", "generation": generation, **learner_metrics.to_dict()})
    validation_metrics = _evaluate_validation(model, validation_dataset, config)
    _append_metric(layout, {"stage": "validation", "generation": generation, **validation_metrics})
    stability_row = GenerationStabilityMetrics(
        generation=generation,
        games=len(games),
        mean_game_length=float(health["game_length"]["mean"]),
        game_length_variance=float(health["game_length"]["variance"]),
        short_game_rate=float(health["game_length"]["short_le_12_rate"]),
        mean_policy_entropy=float(health["mean_policy_entropy"]["full"]),
        value_loss=float(learner_metrics.wdl_loss),
    )
    if accepted_model_id is None:
        # Random bootstrap has a deliberately untrained game-length distribution.
        # Keep its warnings visible, but do not let them satisfy the repeated
        # collapse rule intended for a committed champion.
        stability_history = []
    stability_history = [*stability_history, stability_row]
    stability = assess_stability(stability_history)
    _append_metric(layout, {"stage": "stability", **stability.to_dict()})

    next_state = formal_state.finish_generation(
        next_game_id=games[-1].game_id + 1,
        replay_positions=formal_state.replay_positions + len(new_replay),
        train_positions_consumed=bucket.total_positions_consumed,
    )
    candidate_model_id: str | None = None
    candidate_final_path: Path | None = None
    gate_decision: Any | None = None
    gate_path: Path | None = None
    opening_manifest_path: Path | None = None
    accepted_after = accepted_model_id
    if next_state.candidate_due(config.gate):
        candidate_model_id = (
            f"candidate-g{generation:06d}-s{learner.global_step:08d}-"
            f"d{next_state.replay_positions:08d}"
        )
        candidate_path = layout.candidates / f"{candidate_model_id}.pt"
        _atomic_save_model_artifact(
            candidate_path,
            model=model,
            model_config=asdict(config.model),
            metadata={
                "candidate_model_id": candidate_model_id,
                "parent_model_id": accepted_model_id,
                "global_step": learner.global_step,
                "raw_data_end": next_state.replay_positions,
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
        if not opening_manifest_path.exists():
            write_opening_manifest(opening_manifest_path, openings)
        candidate_predictor = TorchPredictor(model, config.runtime.device)
        gate_evaluation_runtime: list[dict[str, Any]] = []
        candidate_source = EvaluationModelSource(
            "v3_artifact", str(candidate_path.resolve()), candidate_model_id
        )
        incumbent_source = (
            None
            if accepted_model_id is None or latest_commit is None
            else EvaluationModelSource(
                "v3_artifact",
                str((layout.root / str(latest_commit["accepted_model_path"])).resolve()),
                accepted_model_id,
            )
        )
        gate_results, gate_decision, gate_looks = _run_sequential_gate(
            config,
            generation=generation,
            openings=openings,
            candidate_predictor=candidate_predictor,
            incumbent_predictor=accepted_predictor,
            runtime_records=gate_evaluation_runtime,
            candidate_source=candidate_source,
            incumbent_source=incumbent_source,
        )
        gate_path = layout.metrics / f"gate_g{generation:06d}.json"
        _atomic_write_json(
            gate_path,
            {
                "schema_version": 1,
                "candidate_model_id": candidate_model_id,
                "incumbent_model_id": accepted_model_id or "random",
                "opening_manifest": _run_relative(layout, opening_manifest_path),
                "search_sims": config.gate.search_sims_for_generation(generation),
                "initial_pairs": config.gate.initial_opening_pairs,
                "pair_increment": config.gate.pair_increment,
                "max_pairs": config.gate.max_opening_pairs,
                "games": [asdict(result) for result in gate_results],
                "looks": gate_looks,
                "evaluation_runtime": gate_evaluation_runtime,
                **gate_decision.to_dict(),
            },
        )
        candidate_final_path = candidate_path
        if gate_decision.verdict == "accept":
            candidate_final_path = layout.accepted / candidate_path.name
            os.replace(candidate_path, candidate_final_path)
            accepted_after = candidate_model_id
        elif gate_decision.verdict == "reject":
            candidate_final_path = layout.rejected / candidate_path.name
            os.replace(candidate_path, candidate_final_path)
        next_state = next_state.emit_candidate(
            PendingCandidateState(
                candidate_model_id=candidate_model_id,
                candidate_path=_run_relative(layout, candidate_final_path),
                incumbent_model_id=accepted_model_id or "random",
                gate_path=_run_relative(layout, gate_path),
                opening_manifest=_run_relative(layout, opening_manifest_path),
                pairs_evaluated=len(gate_results) // 2,
                max_pairs=config.gate.max_opening_pairs,
            )
        )
        if gate_decision.verdict == "accept":
            next_state = next_state.resolve_pending_candidate(accepted=True)
        elif gate_decision.verdict == "reject":
            next_state = next_state.resolve_pending_candidate(accepted=False)
        for artifact, kind in (
            (opening_manifest_path, "gate_openings"),
            (gate_path, "gate_result"),
            (candidate_final_path, "candidate_model"),
        ):
            _record(journal, artifact, kind)

    if accepted_after != accepted_model_id:
        # Each accepted producer starts a fresh behavioral baseline.  The row
        # above describes games made by the previous producer.
        stability_history = []

    audit_selections, audit_documents, audit_filenames, audit_references = _prepare_audit_replays(
        games,
        config=config,
        generation=generation,
        saved_at=run_created_at,
    )
    replay, replay_entries, retained_position_start = _trim_replay_cursor(
        replay,
        replay_entries,
        retained_position_start=retained_position_start,
        cumulative_positions=next_state.replay_positions,
        active_window_positions=int(selection["window_positions"]),
        margin=config.runtime.storage.active_window_margin,
    )
    replay_cursor = {
        "shards": replay_entries,
        "raw_positions": len(replay),
        "cumulative_raw_positions": next_state.replay_positions,
        "retained_position_start": retained_position_start,
        "window_start": selection["window_start"],
        "window_end": selection["window_end"],
        "next_game_id": next_state.next_game_id,
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
        recent_evaluation=None if gate_decision is None else gate_decision.to_dict(),
        scaler=learner.scaler,
        extra_state={
            "learner_state": learner.state_dict(),
            "token_bucket": bucket.state_dict(),
            "model_config": asdict(config.model),
            "train_positions_consumed": bucket.total_positions_consumed,
            "formal_loop_state": next_state.to_dict(),
            "audit_replays": audit_references,
            "stability_history": [asdict(row) for row in stability_history],
        },
    )
    checkpoint_path = save_checkpoint(
        layout.checkpoints / f"g{generation:06d}-s{learner.global_step:08d}.pt",
        checkpoint,
    )
    _record(journal, checkpoint_path, "checkpoint")
    audit_artifacts = _write_audit_replays(
        layout,
        generation=generation,
        selections=audit_selections,
        documents=audit_documents,
        filenames=audit_filenames,
        checkpoint_path=checkpoint_path,
        run_id=config.run.run_id,
        created_at=run_created_at,
    )
    _record(journal, layout.root / audit_artifacts["audit_index"], "audit_index")
    for relative in audit_artifacts["replays"]:
        _record(journal, layout.root / relative, "audit_replay")

    accepted_relative = None if accepted_after is None else f"accepted/{accepted_after}.pt"
    commit_payload = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "generation": generation,
        "committed_at": _utc_now(),
        "config_hash": expected_hash,
        "checkpoint": _run_relative(layout, checkpoint_path),
        "checkpoint_sha256": audit_artifacts["checkpoint_sha256"],
        "accepted_model_id": accepted_after,
        "accepted_model_path": accepted_relative,
        "accepted_model_sha256": None if accepted_relative is None else _sha256_file(layout.root / accepted_relative),
        "candidate_model_id": candidate_model_id,
        "candidate_path": None if candidate_final_path is None else _run_relative(layout, candidate_final_path),
        "candidate_sha256": None if candidate_final_path is None else _sha256_file(candidate_final_path),
        "gate_verdict": "not_run" if gate_decision is None else gate_decision.verdict,
        "health_watch_warnings": health["watch_warnings"],
        "stability": stability.to_dict(),
        "replay_shards": replay_entries,
        "replay_raw_positions": len(replay),
        "replay_cumulative_positions": next_state.replay_positions,
        "audit_index": audit_artifacts["audit_index"],
        "audit_index_sha256": audit_artifacts["audit_index_sha256"],
        "next_game_id": next_state.next_game_id,
    }
    journal.stage_commit(commit_payload)
    commit_path = journal.publish_commit()
    committed = _validate_generation_commit(layout, commit_path, expected_hash=expected_hash)
    result = {
        "generation": generation,
        "producer_model_id": producer_model_id,
        "games": len(games),
        "new_raw_positions": len(new_replay),
        "cumulative_raw_positions": next_state.replay_positions,
        "optimizer_steps": learner_metrics.steps,
        "train_positions_consumed": next_state.train_positions_consumed,
        "candidate_model_id": candidate_model_id,
        "gate_verdict": commit_payload["gate_verdict"],
        "accepted_model_id": accepted_after,
        "health_watch_warnings": health["watch_warnings"],
        "stability": stability.to_dict(),
        "checkpoint": str(checkpoint_path),
        "commit": str(commit_path),
    }
    _append_metric(layout, {"stage": "generation_commit", **result})
    return (
        result,
        replay,
        replay_entries,
        retained_position_start,
        stability_history,
        next_state,
        committed,
    )


def run_formal(
    config: V3Config,
    *,
    max_train_positions: int,
    max_generations: int | None = None,
) -> dict[str, Any]:
    """Execute a bounded formal run or resume it to the requested bound."""

    if not isinstance(config, V3Config):
        raise TypeError("run_formal requires a resolved V3Config")
    if isinstance(max_train_positions, bool) or int(max_train_positions) < 1:
        raise ValueError("max_train_positions must be a positive integer")
    if max_generations is not None and (
        isinstance(max_generations, bool) or int(max_generations) < 1
    ):
        raise ValueError("max_generations must be a positive integer")
    if _provisional_auxiliary(config):
        raise RuntimeError(
            "formal execution refuses the provisional P6 auxiliary weights; "
            "freeze the calibrated loss and occupancy class weights first"
        )
    if config.runtime.storage.hard_free_gib < 10.0:
        raise RuntimeError("formal execution requires at least a 10-GiB hard free-space reserve")
    if config.runtime.storage.mode != "archive_ack_prune":
        raise RuntimeError("formal execution requires runtime.storage.mode=archive_ack_prune")

    layout = RunLayout.from_root(resolve_run_root(config))
    expected_hash = lineage_config_hash(config)
    if config.run.resume:
        if not layout.run_manifest.is_file():
            raise FileNotFoundError("resume requested but the V3 run manifest is missing")
    elif layout.run_manifest.exists():
        raise FileExistsError(
            f"V3 run already exists: {layout.root}. Use --resume or choose another --run-dir."
        )
    layout.create()
    code_commit = _git_commit()
    if config.run.resume:
        run_manifest = json.loads(layout.run_manifest.read_text(encoding="utf-8"))
        if run_manifest.get("config_hash") != expected_hash:
            raise ValueError("run manifest config hash differs from the requested formal config")
        run_created_at = str(run_manifest["created_at"])
    else:
        run_created_at = _utc_now()
        run_manifest = {
            "schema_version": 1,
            "run_id": config.run.run_id,
            "status": "running",
            "created_at": run_created_at,
            "config_hash": expected_hash,
            "git_commit": code_commit,
            "code_version": __version__,
            "formal": True,
            "max_train_positions": int(max_train_positions),
        }
        _atomic_write_json(layout.run_manifest, run_manifest)
        artifact_config = replace(config, run=replace(config.run, run_dir=str(layout.root)))
        _atomic_write_json(layout.resolved_config, artifact_config.to_dict())

    drain = _DrainRequest()
    previous_handlers: dict[int, Any] = {}
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, drain.handle)
    try:
        with CoordinatorLock(layout.coordinator_lock, run_id=config.run.run_id):
            _resume_ready_commit(layout, expected_hash)
            (
                model,
                learner,
                optimizer,
                bucket,
                formal_state,
                replay,
                replay_entries,
                retained_position_start,
                stability_history,
                latest_commit,
            ) = _load_training_state(config, layout, expected_hash)
            results: list[dict[str, Any]] = []
            stop_reason = "max_train_positions"
            while formal_state.train_positions_consumed < int(max_train_positions):
                if max_generations is not None and len(results) >= int(max_generations):
                    stop_reason = "max_generations"
                    break
                disk = _disk_status(layout, config)
                if disk["hard_reserve_breached"]:
                    raise RuntimeError("disk hard reserve breached before starting a generation")
                if disk["archive_required"]:
                    stop_reason = "archive_required"
                    break
                remaining = int(max_train_positions) - formal_state.train_positions_consumed
                (
                    result,
                    replay,
                    replay_entries,
                    retained_position_start,
                    stability_history,
                    formal_state,
                    latest_commit,
                ) = _run_generation(
                    config,
                    layout,
                    expected_hash=expected_hash,
                    code_commit=code_commit,
                    run_created_at=run_created_at,
                    model=model,
                    learner=learner,
                    optimizer=optimizer,
                    bucket=bucket,
                    formal_state=formal_state,
                    replay=replay,
                    replay_entries=replay_entries,
                    retained_position_start=retained_position_start,
                    stability_history=stability_history,
                    latest_commit=latest_commit,
                    remaining_train_positions=remaining,
                )
                results.append(result)
                if result["stability"]["action"] == "pause":
                    stop_reason = "stability_pause"
                    break
                if formal_state.pending_candidate is not None:
                    stop_reason = "gate_inconclusive_at_max_pairs"
                    break
                if drain.requested:
                    stop_reason = f"drained_after_signal_{drain.signal_number}"
                    break
                if _disk_status(layout, config)["archive_required"]:
                    stop_reason = "archive_required"
                    break
            run_manifest.update(
                {
                    "status": "stopped_at_safe_boundary",
                    "updated_at": _utc_now(),
                    "stop_reason": stop_reason,
                    "formal_loop_state": formal_state.to_dict(),
                    "last_invocation_generations": results,
                    "disk": _disk_status(layout, config),
                }
            )
            _atomic_write_json(layout.run_manifest, run_manifest)
            return {
                "status": "stopped_at_safe_boundary",
                "run_id": config.run.run_id,
                "run_dir": str(layout.root),
                "stop_reason": stop_reason,
                "generations_completed": len(results),
                "formal_loop_state": formal_state.to_dict(),
                "disk": _disk_status(layout, config),
                "results": results,
            }
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
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


__all__ = ["run_formal"]
