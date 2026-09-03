from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from training.v3.anchored_elo import (
    ANCHOR_SCALE_SCHEMA_VERSION,
    anchored_evaluation_plan,
    build_anchored_report,
    canonical_anchored_config_hash,
    load_anchored_config,
    verify_anchor_files,
)
from tools.run_v3_anchored_eval import _resolve_model


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "training/v3/configs/anchored_elo_historical_v1.json"
V2 = ROOT / "training/v3/configs/anchored_elo_historical_v2.json"
V3 = ROOT / "training/v3/configs/anchored_elo_historical_v3.json"
V3_PRESSURE = (
    ROOT / "training/v3/configs/anchored_pressure_256v512_historical_v3.json"
)


class AnchoredEloV3Tests(unittest.TestCase):
    def test_six_anchor_registry_extends_without_mutating_v1_or_v2(self) -> None:
        v1 = load_anchored_config(V1)
        v2 = load_anchored_config(V2)
        v3 = load_anchored_config(V3)
        pressure = load_anchored_config(V3_PRESSURE)

        self.assertEqual(canonical_anchored_config_hash(v1), "09c33b7e63eb341f1af2afa1d7481fa997c27efc5211cae34f4b583c6dacbb34")
        self.assertEqual(canonical_anchored_config_hash(v2), "ee9e491ebdb6bb675d516a4f918d1b860ece1ba9729da01c81486e460037a0f7")
        self.assertEqual(canonical_anchored_config_hash(v3), "a874ff835102a1221246b49a4ebe0a8cac8bf983c6795c28563aa4daf2624db7")
        self.assertEqual(canonical_anchored_config_hash(pressure), "7f6d93d68aa5fc3d66b2772487f9eb4842b160cc2cb5d4c5e4fd14782ccbfaf0")
        self.assertEqual(v3.reference_anchor_id, "v2_2_balance")
        self.assertEqual(len(v3.anchors), 6)
        self.assertEqual(tuple(a.anchor_id for a in pressure.anchors), tuple(a.anchor_id for a in v3.anchors))
        self.assertEqual(v3.anchor("b8_role30_g268").predictor_kind, "v3")
        self.assertEqual(v3.anchor("b10_mixed_final_g258").predictor_kind, "v3")

    def test_six_anchor_profiles_require_fresh_fifteen_match_calibration(self) -> None:
        symmetric = load_anchored_config(V3)
        pressure = load_anchored_config(V3_PRESSURE)
        plan = anchored_evaluation_plan(symmetric)
        self.assertEqual(plan["anchor_calibration_by_profile"]["primary_256"]["matchups"], 15)
        self.assertEqual(plan["anchor_calibration_by_profile"]["primary_256"]["initial_games"], 1500)
        self.assertEqual(plan["anchor_calibration_by_profile"]["final_512"]["initial_games"], 1500)
        self.assertNotIn("pressure_256v512", anchored_evaluation_plan(pressure)["anchor_calibration_by_profile"])
        self.assertEqual(pressure.profile("pressure_256v512").effective_anchor_search_sims, 512)

    def test_b8_and_b10_anchors_use_their_own_v3_architecture_metadata(self) -> None:
        config = load_anchored_config(V3)
        sentinel = object()
        identities = (
            {"checksum_sha256": config.anchor("b8_role30_g268").checksum_sha256, "model_config": {"blocks": 8, "channels": 192}},
            {"checksum_sha256": config.anchor("b10_mixed_final_g258").checksum_sha256, "model_config": {"blocks": 10, "channels": 256}},
        )
        with patch(
            "tools.run_v3_anchored_eval.load_v3_artifact_predictor",
            side_effect=((sentinel, identities[0]), (sentinel, identities[1])),
        ) as loader:
            _, b8 = _resolve_model("anchor:b8_role30_g268", model_id=None, config=config, device="cpu")
            _, b10 = _resolve_model("anchor:b10_mixed_final_g258", model_id=None, config=config, device="cpu")
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(b8["model_config"], {"blocks": 8, "channels": 192})
        self.assertEqual(b10["model_config"], {"blocks": 10, "channels": 256})
        self.assertNotEqual(b8["checksum_sha256"], b10["checksum_sha256"])

    def test_checksum_mismatch_is_rejected(self) -> None:
        config = load_anchored_config(V3)
        anchor = config.anchor("b10_mixed_final_g258")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad-a.pt").write_bytes(b"not-the-frozen-model")
            (root / "bad-b.pt").write_bytes(b"also-not-the-frozen-model")
            bad_anchor = replace(anchor, path="bad-a.pt")
            other = replace(config.anchors[0], path="bad-b.pt")
            isolated = replace(
                config,
                anchors=(bad_anchor, other),
                reference_anchor_id=bad_anchor.anchor_id,
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_anchor_files(isolated, root)

    def test_v2_scale_cannot_rate_a_v3_registry(self) -> None:
        v2 = load_anchored_config(V2)
        v3 = load_anchored_config(V3)
        stale_scale = {
            "schema_version": ANCHOR_SCALE_SCHEMA_VERSION,
            "anchored_config_hash": canonical_anchored_config_hash(v2),
            "profile_id": "primary_256",
            "ratings": {anchor.anchor_id: 0.0 for anchor in v2.anchors},
            "frozen": True,
        }
        with self.assertRaisesRegex(ValueError, "config hash"):
            build_anchored_report(
                v3,
                (),
                profile_id="primary_256",
                target_model_id="target",
                anchor_scale=stale_scale,
            )


if __name__ == "__main__":
    unittest.main()
