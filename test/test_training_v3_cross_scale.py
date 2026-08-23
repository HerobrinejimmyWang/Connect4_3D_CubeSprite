from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from connect4_core.rules import DEFAULT_RULE_REGISTRY
from training.v3.config import ModelConfig
from training.v3.cross_scale import (
    BundleSource,
    CrossScaleSamplePlanner,
    MixedReplayPlanner,
    TransferLedger,
    TransferSchedule,
    TransferStage,
    build_cross_scale_bundle,
    load_cross_scale_replay,
    load_donor_qualification,
    validate_cross_scale_bundle,
    write_donor_qualification,
)
from training.v3.evaluation import Opening
from training.v3.gate import GateGameResult
from training.v3.replay import ReplayShard, sha256_file, write_replay_shard
from training.v3.scaling_experiment import (
    build_scaling_experiment_plan,
    load_scaling_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
SCALING_SPEC = ROOT / "training" / "v3" / "configs" / "dual_track_scaling_v1.json"


def _replay(marker: int) -> ReplayShard:
    board = np.zeros((1, 6, 5, 5), dtype=np.int8)
    board[0, 0, marker % 5, (marker // 5) % 5] = 1
    visits = np.zeros((1, 25), dtype=np.uint32)
    visits[0, marker % 25] = marker + 1
    return ReplayShard(
        board=board,
        visit_counts=visits,
        policy_weight=np.ones((1,), dtype=np.float32),
        wdl=np.zeros((1,), dtype=np.uint8),
        game_id=np.asarray([marker], dtype=np.uint64),
        turn_index=np.ones((1,), dtype=np.uint16),
        player_to_move=np.ones((1,), dtype=np.int8),
        search_kind=np.ones((1,), dtype=np.uint8),
        rule_code=np.zeros((1,), dtype=np.uint16),
        turn_kind=np.zeros((1,), dtype=np.uint8),
        placement_count=np.ones((1,), dtype=np.uint16),
        opponent_reply_column=np.full((1,), -1, dtype=np.int8),
        opponent_reply_mask=np.zeros((1,), dtype=np.uint8),
        terminal_board=board.copy(),
        remaining_turns=np.ones((1,), dtype=np.uint16),
    )


def _metadata(*, generation: int, producer: str = "accepted-donor") -> dict:
    return {
        "run_id": "donor-run",
        "generation": generation,
        "producer_model_id": producer,
        "seed_range": {"start": generation, "end": generation},
        "results": {"p1_wins": 1, "p2_wins": 0, "draws": 0},
        "search_config": {"full_search_sims": 8, "fast_search_sims": 2},
        "rule_registry_hash": DEFAULT_RULE_REGISTRY.registry_hash,
        "config_hash": "source-config-hash",
        "git_commit": "test-tree",
    }


def _artifact(path: Path, *, model_id: str = "accepted-donor") -> Path:
    torch.save(
        {
            "format": "connect4-v3-model",
            "format_version": 1,
            "model_config": asdict(ModelConfig(channels=16, blocks=1)),
            "model_state": {},
            "metadata": {
                "candidate_model_id": model_id,
                "config_hash": "source-config-hash",
            },
        },
        path,
    )
    return path


def _build_fixture(root: Path, *, producer: str = "accepted-donor") -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir()
    artifact = _artifact(source / "accepted-donor.pt")
    strata: dict[str, tuple[BundleSource, ...]] = {}
    for generation, stratum in enumerate(("early", "middle", "late", "strong"), start=1):
        shard = source / f"{stratum}.npz"
        write_replay_shard(
            shard,
            _replay(generation),
            _metadata(generation=generation, producer=producer),
        )
        strata[stratum] = (BundleSource(shard, artifact),)
    bundle = root / "bundle"
    build_cross_scale_bundle(
        bundle,
        donor_run_id="donor-run",
        qualification_donor_model_id="accepted-donor",
        rule_id="classic",
        strata=strata,
    )
    return bundle, artifact


class CrossScaleBundleTests(unittest.TestCase):
    def test_bundle_round_trip_is_data_only_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, _artifact_path = _build_fixture(Path(directory))
            manifest = validate_cross_scale_bundle(bundle)
            self.assertTrue(manifest["data_only"])
            self.assertFalse(manifest["weight_transfer"])
            self.assertEqual(manifest["totals"]["positions"], 4)
            self.assertEqual(set(manifest["strata"]), {"early", "middle", "late", "strong"})
            replay = load_cross_scale_replay(bundle)
            self.assertEqual(len(replay), 4)
            with self.assertRaises(FileExistsError):
                build_cross_scale_bundle(
                    bundle,
                    donor_run_id="donor-run",
                    qualification_donor_model_id="accepted-donor",
                    rule_id="classic",
                    strata={name: () for name in ("early", "middle", "late", "strong")},
                )

    def test_manifest_tamper_and_random_producer_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, _artifact_path = _build_fixture(root)
            manifest_path = bundle / "bundle.manifest.json"
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["totals"]["positions"] += 1
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ready marker"):
                validate_cross_scale_bundle(bundle)

            random_root = root / "random-case"
            random_root.mkdir()
            source = random_root / "source"
            source.mkdir()
            artifact = _artifact(source / "accepted.pt")
            strata = {}
            for generation, stratum in enumerate(("early", "middle", "late", "strong"), 10):
                shard = source / f"{stratum}.npz"
                write_replay_shard(
                    shard,
                    _replay(generation),
                    _metadata(generation=generation, producer="random"),
                )
                strata[stratum] = (BundleSource(shard, artifact),)
            with self.assertRaisesRegex(ValueError, "random-bootstrap"):
                build_cross_scale_bundle(
                    random_root / "bundle",
                    donor_run_id="donor-run",
                    qualification_donor_model_id="accepted-donor",
                    rule_id="classic",
                    strata=strata,
                )


class TransferSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = TransferSchedule(
            (
                TransferStage(0, 0.5),
                TransferStage(100, 0.25),
                TransferStage(1000, 0.05),
            )
        )

    def test_sampling_is_deterministic_and_ledger_resumes_exactly(self) -> None:
        first = MixedReplayPlanner(
            donor_size=17, own_size=31, schedule=self.schedule, seed=99
        )
        second = MixedReplayPlanner(
            donor_size=17, own_size=31, schedule=self.schedule, seed=99
        )
        keys = first.batch(start_cursor=0, count=2000, own_positions_generated=100)
        self.assertEqual(keys, second.batch(start_cursor=0, count=2000, own_positions_generated=100))
        observed = sum(key.origin == "donor" for key in keys) / len(keys)
        self.assertLess(abs(observed - 0.25), 0.04)

        ledger = TransferLedger()
        ledger.record_generated_own(100)
        ledger.record_batch(keys[:20])
        restored = TransferLedger.from_state_dict(ledger.state_dict())
        continuation = first.batch(
            start_cursor=restored.sample_cursor,
            count=10,
            own_positions_generated=restored.own_positions_generated,
        )
        restored.record_batch(continuation)
        self.assertEqual(restored.sample_cursor, 30)
        with self.assertRaisesRegex(ValueError, "cursor"):
            restored.record_batch(keys[:1])

    def test_schedule_cannot_increase_donor_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not increase"):
            TransferSchedule((TransferStage(0, 0.1), TransferStage(5, 0.2)))

    def test_cross_scale_sampler_honors_strata_weights(self) -> None:
        planner = CrossScaleSamplePlanner(
            donor_stratum_sizes={
                "early": 3,
                "middle": 5,
                "late": 7,
                "strong": 11,
            },
            donor_stratum_weights={
                "early": 0.1,
                "middle": 0.2,
                "late": 0.3,
                "strong": 0.4,
            },
            own_size=13,
            schedule=TransferSchedule((TransferStage(0, 1.0),)),
            seed=123,
        )
        keys = planner.batch(start_cursor=0, count=20000, own_positions_generated=0)
        for stratum, expected in (
            ("early", 0.1),
            ("middle", 0.2),
            ("late", 0.3),
            ("strong", 0.4),
        ):
            observed = sum(key.stratum == stratum for key in keys) / len(keys)
            self.assertLess(abs(observed - expected), 0.02)


class ScalingPlanAndQualificationTests(unittest.TestCase):
    def test_dual_track_plan_is_strict_and_keeps_formal_run_disabled(self) -> None:
        spec = load_scaling_experiment(SCALING_SPEC)
        plan = build_scaling_experiment_plan(spec, root=ROOT)
        self.assertFalse(plan["formal_run_enabled"])
        self.assertEqual(len(plan["research_track"]["runs"]), 9)
        self.assertEqual(len(plan["production_track"]["transitions"]), 2)
        self.assertEqual(plan["readiness"]["missing_scale_configs"], [])
        self.assertTrue(
            all(not run["inherited_replay"] for run in plan["research_track"]["runs"])
        )

    def test_qualification_evidence_never_promotes_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, artifact = _build_fixture(root)
            opening_manifest = root / "openings.json"
            opening_manifest.write_text("{}\n", encoding="utf-8")
            opening = Opening("o0", 123, (), "classic", 1)
            results = (
                GateGameResult("o0", 123, True, 1.0),
                GateGameResult("o0", 123, False, 1.0),
            )
            output = root / "qualification.json"
            with mock.patch(
                "training.v3.cross_scale.play_paired_openings", return_value=results
            ):
                write_donor_qualification(
                    output,
                    bundle_dir=bundle,
                    opening_manifest_path=opening_manifest,
                    openings=(opening,),
                    candidate_identity={"model_id": "target", "checksum_sha256": "c" * 64},
                    donor_identity={
                        "model_id": "accepted-donor",
                        "checksum_sha256": sha256_file(artifact),
                    },
                    candidate_predictor=object(),
                    donor_predictor=object(),
                    search_sims=8,
                    cpuct=1.5,
                    confidence=0.95,
                    bootstrap_samples=1000,
                    role_floor=0.45,
                )
            evidence = load_donor_qualification(output)
            self.assertTrue(evidence["qualification_passed"])
            self.assertFalse(evidence["automatic_promotion"])
            self.assertFalse(evidence["replay_generation_authorized"])
            with self.assertRaises(FileExistsError):
                write_donor_qualification(
                    output,
                    bundle_dir=bundle,
                    opening_manifest_path=opening_manifest,
                    openings=(opening,),
                    candidate_identity={"model_id": "target", "checksum_sha256": "c" * 64},
                    donor_identity={
                        "model_id": "accepted-donor",
                        "checksum_sha256": sha256_file(artifact),
                    },
                    candidate_predictor=object(),
                    donor_predictor=object(),
                    search_sims=8,
                    cpuct=1.5,
                    confidence=0.95,
                    bootstrap_samples=1000,
                    role_floor=0.45,
                )


if __name__ == "__main__":
    unittest.main()
