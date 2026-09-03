from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.v3.anchored_elo import canonical_anchored_config_hash, load_anchored_config
from training.v3.replay import sha256_file
from training.v3.stage2.elo import verify_stage2_elo_protocol


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "training/v3/configs/anchored_elo_historical_v3.json"
PRESSURE = ROOT / "training/v3/configs/anchored_pressure_256v512_historical_v3.json"


class Stage2EloTests(unittest.TestCase):
    def test_protocol_binds_registry_anchors_and_scales(self) -> None:
        registry = load_anchored_config(REGISTRY)
        registry_hash = canonical_anchored_config_hash(registry)
        pressure_hash = canonical_anchored_config_hash(load_anchored_config(PRESSURE))
        ratings = {anchor.anchor_id: float(index) for index, anchor in enumerate(registry.anchors)}
        with tempfile.TemporaryDirectory(prefix="stage2-elo-") as temporary:
            directory = Path(temporary)
            profiles = {}
            for profile_id in ("primary_256", "final_512"):
                path = directory / f"{profile_id}.json"
                path.write_text(
                    json.dumps(
                        {
                            "anchored_config_hash": registry_hash,
                            "profile_id": profile_id,
                            "frozen": True,
                            "ratings": ratings,
                        }
                    ),
                    encoding="utf-8",
                )
                profiles[profile_id] = {
                    "scale_path": str(path),
                    "scale_sha256": sha256_file(path),
                }
            protocol = directory / "protocol.json"
            protocol.write_text(
                json.dumps(
                    {
                        "schema": "connect4-v3-stage2-elo-protocol-v1",
                        "registry_path": str(REGISTRY),
                        "registry_hash": registry_hash,
                        "pressure_registry_path": str(PRESSURE),
                        "pressure_registry_hash": pressure_hash,
                        "profiles": profiles,
                    }
                ),
                encoding="utf-8",
            )
            result = verify_stage2_elo_protocol(protocol, repo_root=ROOT)
            self.assertTrue(result["verified"])
            self.assertEqual(result["anchor_count"], 6)
            self.assertEqual(set(result["profiles"]), {"primary_256", "final_512"})

            primary = Path(profiles["primary_256"]["scale_path"])
            primary.write_text(primary.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_stage2_elo_protocol(protocol, repo_root=ROOT)


if __name__ == "__main__":
    unittest.main()
