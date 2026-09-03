from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import torch

from training.v3.config import ModelConfig, V3Config, config_hash, model_config_dict
from training.v3.model import (
    ArchitecturePolicyValueNetV3,
    GravityPolicyValueNetV3,
    build_model,
    canonical_board_to_column_features,
    canonical_board_to_voxels,
    classic_rule_features,
)
from training.v3.stage2.calibration import calibrate_architecture_matrix
from training.v3.stage2.selfplay import generate_stage2b_configs


ARCHITECTURES = (
    "gravity_resnet",
    "column_resnet",
    "multiview_resnet",
    "raw3d_resnet",
    "plane3d_fusion_resnet",
    "column3d_fusion_resnet",
    "column_transformer",
    "multiview_transformer",
)


class Stage2ArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.board = torch.zeros((2, 6, 5, 5), dtype=torch.float32)
        self.board[0, 0, 1, 2] = 1
        self.board[1, 0, 3, 4] = -1
        self.roles = torch.tensor(((1.0, 0.0), (0.0, 1.0)))
        self.rules = classic_rule_features(2)

    def test_all_architectures_preserve_fixed_output_contract(self) -> None:
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = build_model(ModelConfig(architecture=architecture, channels=8, blocks=1))
                output = model(self.board, role_to_play=self.roles, rule_features=self.rules)
                self.assertEqual(tuple(output.policy_logits.shape), (2, 25))
                self.assertEqual(tuple(output.wdl_logits.shape), (2, 3))
                self.assertEqual(tuple(output.opponent_reply_logits.shape), (2, 25))
                self.assertEqual(tuple(output.future_occupancy_logits.shape), (2, 3, 6, 5, 5))
                self.assertEqual(tuple(output.moves_left_logits.shape), (2, 301))
                search = model.forward_search(
                    self.board, role_to_play=self.roles, rule_features=self.rules
                )
                self.assertEqual(tuple(search.policy_logits.shape), (2, 25))
                self.assertEqual(tuple(search.wdl_logits.shape), (2, 3))

    def test_builder_keeps_legacy_class_and_uses_family_class_for_stage2(self) -> None:
        self.assertIsInstance(build_model(ModelConfig(channels=8, blocks=1)), GravityPolicyValueNetV3)
        self.assertIsInstance(
            build_model(ModelConfig(architecture="column_resnet", channels=8, blocks=1)),
            ArchitecturePolicyValueNetV3,
        )

    def test_column_and_voxel_axis_contract(self) -> None:
        board = torch.zeros((1, 6, 5, 5))
        board[0, 2, 1, 3] = 1
        board[0, 0, 4, 2] = -1
        columns = canonical_board_to_column_features(board)
        self.assertEqual(tuple(columns.shape), (1, 5, 5, 14))
        self.assertEqual(float(columns[0, 1, 3, 2]), 1.0)
        self.assertEqual(float(columns[0, 4, 2, 6]), 1.0)
        voxels = canonical_board_to_voxels(board)
        self.assertEqual(tuple(voxels.shape), (1, 2, 6, 5, 5))
        self.assertEqual(float(voxels[0, 0, 2, 1, 3]), 1.0)
        self.assertEqual(float(voxels[0, 1, 0, 4, 2]), 1.0)

    def test_strict_architecture_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "one of"):
            ModelConfig(architecture="unknown")
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(architecture="column_transformer", channels=10, attention_heads=4)
        with self.assertRaisesRegex(ValueError, "attention settings"):
            ModelConfig(architecture="column_resnet", attention_heads=2)
        with self.assertRaisesRegex(ValueError, "encoder_channels"):
            ModelConfig(architecture="raw3d_resnet", encoder_channels=16)
        with self.assertRaisesRegex(ValueError, "does not accept"):
            ModelConfig(architecture="gravity_resnet", branch_channels=8)

    def test_legacy_serialization_and_hash_remain_unchanged(self) -> None:
        expected = {
            "architecture": "gravity_resnet",
            "channels": 8,
            "blocks": 1,
            "global_input_schema": "role_rule_v1",
            "output_schema": "policy_wdl_aux_v1",
            "rule_feature_dim": 32,
            "moves_left_classes": 301,
        }
        self.assertEqual(model_config_dict(ModelConfig(channels=8, blocks=1)), expected)
        old = V3Config.from_dict({"model": expected, "learner": {"batch_size": 4}})
        explicit = V3Config.from_dict({"model": dict(expected), "learner": {"batch_size": 4}})
        self.assertEqual(config_hash(old), config_hash(explicit))

    def test_cross_architecture_state_is_strictly_incompatible(self) -> None:
        column = build_model(ModelConfig(architecture="column_resnet", channels=8, blocks=1))
        multiview = build_model(ModelConfig(architecture="multiview_resnet", channels=8, blocks=1))
        with self.assertRaises(RuntimeError):
            multiview.load_state_dict(column.state_dict(), strict=True)

    def test_stage2b_generator_creates_six_cold_start_configs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2b-configs-") as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(
                json.dumps(calibrate_architecture_matrix(anchor_channels=8, anchor_blocks=1)),
                encoding="utf-8",
            )
            finalists_path = root / "finalists.json"
            finalists_path.write_text(
                json.dumps(
                    {"finalists": ["gravity_resnet", "column_resnet", "column_transformer"]}
                ),
                encoding="utf-8",
            )
            base = Path(__file__).resolve().parents[1] / "training/v3/configs/smoke_cpu.json"
            manifest = generate_stage2b_configs(
                base_config_path=base,
                architecture_matrix_path=matrix_path,
                finalists_path=finalists_path,
                output_dir=root / "generated",
            )
            self.assertEqual(len(manifest["runs"]), 6)
            for row in manifest["runs"]:
                config = V3Config.from_dict(json.loads(Path(row["config"]).read_text(encoding="utf-8")))
                self.assertFalse(config.run.resume)
                self.assertEqual(config.run.warm_start_mode, "")


if __name__ == "__main__":
    unittest.main()
