from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from training.v3.stage2.geometry import (
    GEOMETRY_CLASSES,
    GEOMETRY_SUITE_SCHEMA,
    TACTICAL_TYPES,
    apply_d4_actions,
    evaluate_geometry_artifact,
    evaluate_geometry_suite,
    load_geometry_suite,
)
from training.v3.config import ModelConfig, model_config_dict
from training.v3.model import build_model


def _board(action: int | None = None) -> list[list[list[int]]]:
    result = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(6)]
    if action is not None:
        result[0][action // 5][action % 5] = 1
    return result


def _case(
    case_id: str,
    *,
    geometry: str = "xy",
    tactical_type: str = "immediate_win",
    marker: int | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "canonical_board": _board(marker),
        "absolute_role": "first" if len(case_id) % 2 else "second",
        "legal_actions": list(range(25)),
        "target_actions": [0],
        "geometry": geometry,
        "tactical_type": tactical_type,
        "wdl_target": "win",
    }


def _suite(cases: list[dict]) -> dict:
    return {
        "schema": GEOMETRY_SUITE_SCHEMA,
        "suite_id": "tiny-geometry-v1",
        "rule_features": [0.0] * 32,
        "cases": cases,
    }


class _UniformModel(torch.nn.Module):
    def forward_search(
        self,
        boards: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> SimpleNamespace:
        self.last_roles = role_to_play.detach().cpu()
        return SimpleNamespace(
            policy_logits=torch.zeros((boards.shape[0], 25), device=boards.device),
            wdl_logits=torch.tensor((2.0, 0.0, -1.0), device=boards.device).repeat(
                boards.shape[0], 1
            ),
        )


class _EquivariantBoardModel(torch.nn.Module):
    def forward_search(
        self,
        boards: torch.Tensor,
        *,
        role_to_play: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> SimpleNamespace:
        del role_to_play, rule_features
        return SimpleNamespace(
            policy_logits=4.0 * boards.sum(dim=1).reshape(boards.shape[0], 25),
            wdl_logits=torch.zeros((boards.shape[0], 3), device=boards.device),
        )


class Stage2GeometrySuiteTest(unittest.TestCase):
    def test_strict_schema_rejects_unknown_geometry_and_tactical_class(self) -> None:
        for field, value, message in (
            ("geometry", "diagonalish", "geometry"),
            ("tactical_type", "maybe_block", "tactical_type"),
        ):
            with self.subTest(field=field):
                raw = _suite([_case("invalid")])
                raw["cases"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    load_geometry_suite(raw)

        raw = _suite([_case("unknown-key")])
        raw["cases"][0]["comment"] = "not part of the frozen schema"
        with self.assertRaisesRegex(ValueError, "unknown=.*comment"):
            load_geometry_suite(raw)

    def test_d4_action_mapping_and_policy_restore_use_replay_convention(self) -> None:
        self.assertEqual(apply_d4_actions((0,), 1), (20,))
        self.assertEqual(apply_d4_actions((20,), 3), (0,))
        for transform in range(8):
            mapped = apply_d4_actions((1, 7, 19), transform)
            restored = apply_d4_actions(mapped, (-(transform % 4)) % 4 if transform < 4 else transform)
            self.assertEqual(restored, (1, 7, 19))

        raw = _suite([_case("equivariant", marker=1)])
        raw["cases"][0]["target_actions"] = [1]
        report = evaluate_geometry_suite(_EquivariantBoardModel(), raw, seed=17)
        consistency = report["overall"]["metrics"]["d4_policy_consistency"]["value"]
        self.assertAlmostEqual(consistency, 1.0, places=6)

    def test_metrics_are_deterministic_and_stratified_by_both_axes(self) -> None:
        cases = [
            _case(
                f"case-{index}",
                geometry=geometry,
                tactical_type=TACTICAL_TYPES[index % len(TACTICAL_TYPES)],
            )
            for index, geometry in enumerate(GEOMETRY_CLASSES)
        ]
        cases[0]["absolute_role"] = "first"
        cases[1]["absolute_role"] = "second"
        model = _UniformModel()
        first = evaluate_geometry_suite(model, _suite(cases), seed=271828)
        second = evaluate_geometry_suite(model, _suite(copy.deepcopy(cases)), seed=271828)
        self.assertEqual(first, second)
        self.assertEqual(first["overall"]["case_count"], 6)
        self.assertEqual(first["bootstrap_samples"], 1_000)
        for geometry in GEOMETRY_CLASSES:
            self.assertEqual(first["by_geometry"][geometry]["case_count"], 1)
        for tactical in TACTICAL_TYPES:
            self.assertEqual(first["by_tactical_type"][tactical]["case_count"], 2)
        self.assertEqual(first["by_geometry_tactical"]["xy/immediate_win"]["case_count"], 1)
        self.assertEqual(first["by_geometry_tactical"]["xy/forced_block"]["case_count"], 0)
        metrics = first["overall"]["metrics"]
        self.assertAlmostEqual(metrics["policy_ce"]["value"], math.log(25.0), places=6)
        self.assertEqual(metrics["policy_top1"]["value"], 1.0)
        self.assertIn("wdl_ece", metrics)
        self.assertIn("ci95", metrics["wdl_brier"])
        self.assertTrue(torch.equal(model.last_roles[0], torch.tensor((1.0, 0.0))))
        self.assertTrue(torch.equal(model.last_roles[8], torch.tensor((0.0, 1.0))))

    def test_artifact_entrypoint_records_identity_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-geometry-") as temporary:
            root = Path(temporary)
            model_config = ModelConfig(channels=8, blocks=1)
            config_path = root / "run.json"
            config_path.write_text(
                json.dumps({"model": model_config_dict(model_config)}), encoding="utf-8"
            )
            artifact_path = root / "model.pt"
            model = build_model(model_config)
            torch.save(
                {
                    "format": "connect4-v3-model",
                    "format_version": 1,
                    "model_config": model_config_dict(model_config),
                    "model_state": model.state_dict(),
                },
                artifact_path,
            )
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps(_suite([_case("one")])), encoding="utf-8")
            report = evaluate_geometry_artifact(
                config_path=config_path,
                artifact_path=artifact_path,
                suite_path=suite_path,
                seed=3,
            )
            self.assertEqual(report["model"], model_config_dict(model_config))
            self.assertEqual(len(report["artifact_sha256"]), 64)

            wrong = ModelConfig(architecture="column_resnet", channels=8, blocks=1)
            config_path.write_text(
                json.dumps({"model": model_config_dict(wrong)}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "model_config differs"):
                evaluate_geometry_artifact(
                    config_path=config_path,
                    artifact_path=artifact_path,
                    suite_path=suite_path,
                    seed=3,
                )


if __name__ == "__main__":
    unittest.main()
