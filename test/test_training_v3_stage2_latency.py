from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from tools import benchmark_desktop_cpu_latency as desktop_latency
from tools import benchmark_stage2_cpu_latency as stage2_latency
from training.v3.config import ModelConfig, model_config_dict
from training.v3.model import build_model


class Stage2LatencyBenchmarkTests(unittest.TestCase):
    def test_corpus_matches_desktop_benchmark_exactly(self) -> None:
        desktop = desktop_latency.build_states()
        stage2 = stage2_latency.build_states()
        self.assertEqual(len(stage2), len(desktop))
        for (desktop_board, desktop_player), (stage2_board, stage2_player) in zip(
            desktop, stage2, strict=True
        ):
            self.assertEqual(stage2_player, desktop_player)
            np.testing.assert_array_equal(stage2_board, desktop_board)

    def test_canonical_occupancy_infers_alternating_absolute_roles(self) -> None:
        game = stage2_latency.GameRules()
        roles = []
        for board, player in stage2_latency.build_states()[:8]:
            canonical = game.get_canonical_form(board, player)
            roles.append(tuple(stage2_latency.role_features_for_canonical(canonical)))
        self.assertEqual(roles, [(1.0, 0.0), (0.0, 1.0)] * 4)
        invalid = np.zeros((6, 5, 5), dtype=np.int8)
        invalid[0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "occupancy"):
            stage2_latency.role_features_for_canonical(invalid)

    def _write_model_pair(
        self, root: Path, *, artifact_config: ModelConfig, file_config: ModelConfig
    ) -> tuple[Path, Path]:
        model_path = root / "candidate.model.pt"
        config_path = root / "candidate.config.json"
        model = build_model(artifact_config)
        torch.save(
            {
                "format": "connect4-v3-model",
                "format_version": 1,
                "model_config": model_config_dict(artifact_config),
                "model_state": model.state_dict(),
                "metadata": {
                    "lineage": "v3_stage2_offline",
                    "train_regime": "standard_late",
                    "seed": 271828,
                    "train_positions": 1_000_000,
                },
            },
            model_path,
        )
        config_path.write_text(
            json.dumps(
                {
                    "model": model_config_dict(file_config),
                    "train_regime": "standard_late",
                    "seed": 271828,
                    "target_positions": 1_000_000,
                }
            ),
            encoding="utf-8",
        )
        return model_path, config_path

    def test_artifact_model_config_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-latency-") as temporary:
            root = Path(temporary)
            model_path, config_path = self._write_model_pair(
                root,
                artifact_config=ModelConfig(
                    architecture="column_resnet", channels=8, blocks=1
                ),
                file_config=ModelConfig(
                    architecture="multiview_resnet", channels=8, blocks=1
                ),
            )
            with self.assertRaisesRegex(ValueError, "model_config does not match"):
                stage2_latency.load_model(config_path, model_path)

    def test_loaded_model_records_exact_config_and_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-latency-") as temporary:
            root = Path(temporary)
            config = ModelConfig(architecture="column_resnet", channels=8, blocks=1)
            model_path, config_path = self._write_model_pair(
                root, artifact_config=config, file_config=config
            )
            _predictor, _config, evidence = stage2_latency.load_model(
                config_path, model_path
            )
            self.assertEqual(
                evidence["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest()
            )
            self.assertEqual(
                evidence["artifact_sha256"], hashlib.sha256(model_path.read_bytes()).hexdigest()
            )

    def test_artifact_training_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-latency-") as temporary:
            root = Path(temporary)
            config = ModelConfig(architecture="column_resnet", channels=8, blocks=1)
            model_path, config_path = self._write_model_pair(
                root, artifact_config=config, file_config=config
            )
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            raw["target_positions"] = 3_000_000
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_positions does not match"):
                stage2_latency.load_model(config_path, model_path)

    def test_repeats_and_latency_percentiles_are_recorded(self) -> None:
        fake_predictor = mock.Mock()
        fake_predictor.predict.return_value = (
            np.full(150, 1.0 / 150.0, dtype=np.float32),
            0.0,
        )
        fake_evidence = {"config_sha256": "a" * 64, "artifact_sha256": "b" * 64}
        fake_config = {
            "model": {"architecture": "gravity_resnet"},
            "train_regime": "standard_late",
        }
        fake_search = mock.Mock()
        fake_search.run.return_value = SimpleNamespace(action=0)
        with mock.patch.object(
            stage2_latency,
            "load_model",
            return_value=(fake_predictor, fake_config, fake_evidence),
        ), mock.patch.object(
            stage2_latency, "NumpyMCTS", return_value=fake_search
        ), mock.patch.object(
            stage2_latency, "find_forced_tactical_action", return_value=None
        ):
            result = stage2_latency.run_one(
                "gravity", Path("unused.pt"), Path("unused.json"), 4, repeats=2
            )
        self.assertEqual(result["metadata"]["repeats"], 2)
        self.assertEqual(result["metadata"]["config_sha256"], "a" * 64)
        self.assertEqual(result["summary"]["state_count"], 22)
        self.assertEqual(result["summary"]["measurement_count"], 44)
        self.assertEqual(result["summary"]["searched_measurement_count"], 44)
        for field in ("mean_s", "median_s", "p90_s", "p95_s"):
            self.assertIn(field, result["summary"]["excluding_shortcuts"])

    def test_markdown_header_uses_requested_simulation_counts(self) -> None:
        stats = {"median_s": 1.0, "p90_s": 2.0, "p95_s": 3.0}
        results = {
            "candidate@7": {"summary": {"excluding_shortcuts": stats}},
            "candidate@13": {"summary": {"excluding_shortcuts": stats}},
        }
        markdown = stage2_latency.build_summary_markdown(results, [7, 13])
        self.assertIn("| Architecture | 7 sims | 13 sims |", markdown)
        self.assertNotIn("16 sims", markdown)


if __name__ == "__main__":
    unittest.main()
