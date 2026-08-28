from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from training.v3.anchored_elo import (
    ANCHOR_SCALE_SCHEMA_VERSION,
    anchored_evaluation_plan,
    build_anchored_report,
    build_pressure_report,
    canonical_anchored_config_hash,
    fit_anchor_scale,
    load_anchored_config,
    load_match_batches,
    summarize_direct_matchup,
    verify_anchor_files,
    verify_opening_suite,
    write_match_batch,
)
from training.v3.search import RandomPredictor
from training.v3.evaluation_runtime import EvaluationModelSource
from tools.run_v3_anchored_eval import _identity_for_anchor, _resolve_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "training" / "v3" / "configs" / "anchored_elo_historical_v1.json"
PRESSURE_CONFIG_PATH = (
    ROOT / "training" / "v3" / "configs" / "anchored_pressure_256v512_historical_v1.json"
)
V2_CONFIG_PATH = ROOT / "training" / "v3" / "configs" / "anchored_elo_historical_v2.json"
V2_PRESSURE_CONFIG_PATH = (
    ROOT / "training" / "v3" / "configs" / "anchored_pressure_256v512_historical_v2.json"
)


def _target_batch(
    config, anchor_id: str, *, score_first: float, score_second: float, profile_id: str = "primary_256"
):
    results = []
    for index in range(12):
        results.extend(
            (
                {
                    "opening_id": f"{anchor_id}-{index:03d}",
                    "seed": index,
                    "model_a_is_first": True,
                    "model_a_score": score_first,
                },
                {
                    "opening_id": f"{anchor_id}-{index:03d}",
                    "seed": index,
                    "model_a_is_first": False,
                    "model_a_score": score_second,
                },
            )
        )
    return {
        "batch_id": f"batch-{anchor_id}",
        "profile": {"profile_id": profile_id},
        "model_a": {"model_id": "target"},
        "model_b": {"model_id": anchor_id},
        "results": results,
    }


class AnchoredEloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_anchored_config(CONFIG_PATH)

    def test_frozen_registry_openings_and_budget_plan_verify(self) -> None:
        anchors = verify_anchor_files(self.config, ROOT)
        openings = verify_opening_suite(self.config, ROOT)
        plan = anchored_evaluation_plan(self.config)
        self.assertEqual(len(anchors), 3)
        self.assertEqual(len(openings), 200)
        self.assertEqual(plan["anchor_calibration_by_profile"]["primary_256"]["initial_games"], 300)
        self.assertEqual(plan["milestone_profiles"]["final"], ["primary_256", "final_512"])
        self.assertEqual(
            canonical_anchored_config_hash(self.config),
            "09c33b7e63eb341f1af2afa1d7481fa997c27efc5211cae34f4b583c6dacbb34",
        )

    def test_pressure_profile_is_a_separate_asymmetric_ruler(self) -> None:
        config = load_anchored_config(PRESSURE_CONFIG_PATH)
        profile = config.profile("pressure_256v512")
        plan = anchored_evaluation_plan(config)
        self.assertEqual(profile.search_sims, 256)
        self.assertEqual(profile.effective_anchor_search_sims, 512)
        self.assertEqual(profile.report_kind, "pressure")
        self.assertNotIn("pressure_256v512", plan["anchor_calibration_by_profile"])
        batches = tuple(
            _target_batch(
                config,
                anchor.anchor_id,
                score_first=1.0,
                score_second=0.5,
                profile_id=profile.profile_id,
            )
            for anchor in config.anchors
        )
        report = build_pressure_report(
            config,
            batches,
            profile_id=profile.profile_id,
            target_model_id="target",
        )
        self.assertEqual(report["target_search_sims"], 256)
        self.assertEqual(report["anchor_search_sims"], 512)
        self.assertIsNone(report["anchored_elo_scale"])
        self.assertEqual(report["aggregate"]["games"], 72)

    def test_v2_registry_adds_frozen_v3_b6_anchor_without_mutating_v1(self) -> None:
        config = load_anchored_config(V2_CONFIG_PATH)
        pressure = load_anchored_config(V2_PRESSURE_CONFIG_PATH)
        plan = anchored_evaluation_plan(config)
        anchor = config.anchor("b6_dynamic_g150")

        self.assertEqual(anchor.predictor_kind, "v3")
        self.assertEqual(
            anchor.checksum_sha256,
            "d30e98568163d70b97173be201a093c4502bdba5a6b61be5445ee5430a55658d",
        )
        self.assertEqual(len(config.anchors), 4)
        self.assertEqual(len(pressure.anchors), 4)
        self.assertEqual(
            plan["anchor_calibration_by_profile"]["primary_256"]["initial_games"],
            600,
        )
        self.assertEqual(
            canonical_anchored_config_hash(config),
            "ee9e491ebdb6bb675d516a4f918d1b860ece1ba9729da01c81486e460037a0f7",
        )
        self.assertEqual(
            canonical_anchored_config_hash(self.config),
            "09c33b7e63eb341f1af2afa1d7481fa997c27efc5211cae34f4b583c6dacbb34",
        )

    def test_v3_anchor_uses_v3_artifact_loader_and_replicated_source(self) -> None:
        config = load_anchored_config(V2_CONFIG_PATH)
        anchor = config.anchor("b6_dynamic_g150")
        sentinel = object()
        loaded_identity = {
            "checksum_sha256": anchor.checksum_sha256,
            "model_config": {"blocks": 6, "channels": 128},
        }
        with patch(
            "tools.run_v3_anchored_eval.load_v3_artifact_predictor",
            return_value=(sentinel, loaded_identity),
        ) as loader:
            predictor, identity = _resolve_model(
                "anchor:b6_dynamic_g150", model_id=None, config=config, device="cpu"
            )

        self.assertIs(predictor, sentinel)
        loader.assert_called_once()
        self.assertEqual(identity["lineage"], "v3_anchor")
        self.assertEqual(identity["model_config"], loaded_identity["model_config"])
        source = EvaluationModelSource.from_identity(identity)
        self.assertEqual(source.kind, "v3_artifact")
        self.assertEqual(source.model_id, "b6_dynamic_g150")
        self.assertEqual(
            _identity_for_anchor(config, "b6_dynamic_g150")["predictor_kind"], "v3"
        )

    def test_opening_checksum_is_portable_across_checkout_newlines(self) -> None:
        source = ROOT / self.config.openings.manifest_path
        normalized = source.read_bytes().replace(b"\r\n", b"\n")
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "openings.json"
            manifest.write_bytes(normalized.replace(b"\n", b"\r\n"))
            portable = replace(
                self.config,
                openings=replace(self.config.openings, manifest_path="openings.json"),
            )
            self.assertEqual(len(verify_opening_suite(portable, directory)), 200)

    def test_all_wins_produce_finite_lower_bound_not_infinite_point_elo(self) -> None:
        row = summarize_direct_matchup((1.0,) * 100, self.config.statistics)
        self.assertEqual(row["rating_status"], "saturated_high")
        self.assertIsNone(row["relative_elo"])
        self.assertTrue(math.isfinite(row["relative_elo_lower"]))
        self.assertGreater(row["relative_elo_lower"], 0.0)
        self.assertEqual(row["evidence_status"], "complete_saturated")

    def test_bradley_terry_recovers_a_known_direct_score(self) -> None:
        observations = [
            *(('strong', 'reference', 1.0) for _ in range(75)),
            *(('strong', 'reference', 0.0) for _ in range(25)),
        ]
        ratings = fit_anchor_scale(
            observations,
            model_ids=("reference", "strong"),
            reference_model_id="reference",
            prior_sigma=800.0,
        )
        self.assertEqual(ratings["reference"], 0.0)
        self.assertGreater(ratings["strong"], 170.0)
        self.assertLess(ratings["strong"], 210.0)

    def test_target_report_keeps_frozen_anchor_ratings(self) -> None:
        batches = tuple(
            _target_batch(self.config, anchor.anchor_id, score_first=1.0, score_second=0.5)
            for anchor in self.config.anchors
        )
        fixed = {
            self.config.anchors[0].anchor_id: 0.0,
            self.config.anchors[1].anchor_id: 100.0,
            self.config.anchors[2].anchor_id: 200.0,
        }
        scale = {
            "schema_version": ANCHOR_SCALE_SCHEMA_VERSION,
            "anchored_config_hash": canonical_anchored_config_hash(self.config),
            "profile_id": "primary_256",
            "ratings": fixed,
            "frozen": True,
        }
        report = build_anchored_report(
            self.config,
            batches,
            profile_id="primary_256",
            target_model_id="target",
            anchor_scale=scale,
        )
        self.assertEqual(report["anchor_scale_ratings"], fixed)
        self.assertFalse(report["promotion_gate_input"])
        self.assertFalse(report["selfplay_replay_input"])
        self.assertGreater(report["anchored_rating"]["estimate"], 200.0)

    def test_match_batch_is_immutable_and_content_hash_is_rechecked(self) -> None:
        base_profile = self.config.profile("primary_256")
        tiny_profile = replace(
            base_profile,
            search_sims=1,
            initial_pairs=1,
            pair_increment=1,
            max_pairs=1,
        )
        config = replace(self.config, profiles=(tiny_profile, self.config.profiles[1]))
        opening = verify_opening_suite(config, ROOT)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.match.json"
            write_match_batch(
                path,
                config=config,
                profile=tiny_profile,
                openings=(opening,),
                opening_manifest_path=ROOT / config.openings.manifest_path,
                model_a={"model_id": "random-target"},
                model_b={"model_id": self.config.anchors[0].anchor_id},
                predictor_a=RandomPredictor(),
                predictor_b=RandomPredictor(),
                milestone="early",
                parallel_games=2,
                inference_batch_size=2,
                inference_batch_timeout_s=0.01,
            )
            loaded = load_match_batches(
                (path,), expected_config_hash=canonical_anchored_config_hash(config)
            )
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["runtime"]["evaluation"]["parallel_games"], 2)
            self.assertTrue(
                all(
                    service["max_batch"] <= service["batch_limit"]
                    for service in loaded[0]["runtime"]["evaluation"]["inference_services"]
                )
            )
            with self.assertRaises(FileExistsError):
                write_match_batch(
                    path,
                    config=config,
                    profile=tiny_profile,
                    openings=(opening,),
                    opening_manifest_path=ROOT / config.openings.manifest_path,
                    model_a={"model_id": "random-target"},
                    model_b={"model_id": self.config.anchors[0].anchor_id},
                    predictor_a=RandomPredictor(),
                    predictor_b=RandomPredictor(),
                    milestone="early",
                )
            text = path.read_text(encoding="utf-8").replace('"model_a_score": 1.0', '"model_a_score": 0.5', 1)
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                load_match_batches(
                    (path,), expected_config_hash=canonical_anchored_config_hash(config)
                )


if __name__ == "__main__":
    unittest.main()
