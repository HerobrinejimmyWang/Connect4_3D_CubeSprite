from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from training.v3.config import ModelConfig, V3Config, config_hash, model_config_dict
from training.v3.model import (
    ArchitecturePolicyValueNetV3,
    GravityPolicyValueNetV3,
    build_model,
    canonical_board_to_column_features,
    canonical_board_to_voxels,
    canonical_board_to_winning_diagonal_sections,
    classic_rule_features,
    winning_diagonal_paths,
)
from training.v3.stage2.calibration import (
    calibrate_architecture_matrix,
    calibrate_model_grid,
    component_parameter_breakdown,
)
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
    "multiview_winning_resnet",
    "multiview_winning_transformer",
    "raw3d_to2d_resnet",
    "factorized3d_resnet",
    "plane3d_fusion_v2",
    "column3d_fusion_v2",
    "multiview3d_fusion_resnet",
    "winning3d_fusion_resnet",
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

    def test_winning_diagonal_sections_include_exactly_six_useful_paths(self) -> None:
        paths = winning_diagonal_paths()
        self.assertEqual(len(paths), 6)
        self.assertEqual(sorted(len(path) for path in paths), [4, 4, 4, 4, 5, 5])
        self.assertTrue(all(len(path) >= 4 for path in paths))

        board = torch.zeros((1, 6, 5, 5))
        for path_index, path in enumerate(paths):
            y, x = path[0]
            board[0, path_index, y, x] = 1.0
        sections, valid = canonical_board_to_winning_diagonal_sections(board)
        self.assertEqual(tuple(sections.shape), (1, 6, 2, 6, 5))
        self.assertEqual(tuple(valid.sum(dim=1).tolist()), (4, 5, 4, 4, 5, 4))
        for path_index, path in enumerate(paths):
            y, x = path[0]
            self.assertEqual(float(sections[0, path_index, 0, path_index, 0]), 1.0)
            self.assertEqual(float(board[0, path_index, y, x]), 1.0)

    def test_winning_diagonal_path_set_is_d4_closed(self) -> None:
        expected = {frozenset(path) for path in winning_diagonal_paths()}
        transforms = (
            lambda y, x: (y, x),
            lambda y, x: (x, 4 - y),
            lambda y, x: (4 - y, 4 - x),
            lambda y, x: (4 - x, y),
            lambda y, x: (y, 4 - x),
            lambda y, x: (4 - y, x),
            lambda y, x: (x, y),
            lambda y, x: (4 - x, 4 - y),
        )
        for transform in transforms:
            transformed = {
                frozenset(transform(y, x) for y, x in path)
                for path in winning_diagonal_paths()
            }
            self.assertEqual(transformed, expected)

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
        with self.assertRaisesRegex(ValueError, "volume_blocks"):
            ModelConfig(architecture="column_resnet", volume_blocks=1)
        with self.assertRaisesRegex(ValueError, "collapse_mode"):
            ModelConfig(architecture="raw3d_to2d_resnet", collapse_mode="flat")
        with self.assertRaisesRegex(ValueError, "fusion_mode"):
            ModelConfig(architecture="raw3d_to2d_resnet", fusion_mode="concat")
        with self.assertRaisesRegex(ValueError, "volume_channels"):
            ModelConfig(architecture="raw3d_to2d_resnet", volume_channels=8)
        with self.assertRaisesRegex(ValueError, "post_trunk_mode"):
            ModelConfig(architecture="column_resnet", post_trunk_mode="serial_attention")
        with self.assertRaisesRegex(ValueError, "must be 'serial_attention'"):
            ModelConfig(
                architecture="column3d_fusion_v2",
                post_trunk_mode="unknown",
            )
        with self.assertRaisesRegex(ValueError, "require model.post_trunk_mode"):
            ModelConfig(
                architecture="column3d_fusion_v2",
                post_attention_blocks=2,
            )
        with self.assertRaisesRegex(ValueError, "post_attention_heads"):
            ModelConfig(
                architecture="column3d_fusion_v2",
                channels=10,
                post_trunk_mode="serial_attention",
                post_attention_heads=4,
            )

    def test_balance_v2_3d_controls_are_explicit_and_serialized(self) -> None:
        dense = build_model(
            ModelConfig(
                architecture="raw3d_to2d_resnet",
                channels=8,
                blocks=1,
                branch_channels=4,
                volume_blocks=2,
                collapse_mode="mean",
            )
        )
        factorized = build_model(
            ModelConfig(
                architecture="factorized3d_resnet",
                channels=8,
                blocks=1,
                branch_channels=4,
                volume_blocks=2,
                collapse_mode="learned",
            )
        )
        self.assertTrue(any("conv1.weight" in name for name in dense.state_dict()))
        self.assertTrue(any("spatial1.weight" in name for name in factorized.state_dict()))
        exported = factorized.export_config()
        self.assertEqual(exported["volume_blocks"], 2)
        self.assertEqual(exported["collapse_mode"], "learned")

    def test_balance_v2_fusion_modes_preserve_spatial_contract(self) -> None:
        for fusion_mode in ("concat", "gated", "attention"):
            with self.subTest(fusion_mode=fusion_mode):
                model = build_model(
                    ModelConfig(
                        architecture="winning3d_fusion_resnet",
                        channels=8,
                        blocks=1,
                        branch_channels=4,
                        volume_blocks=1,
                        fusion_mode=fusion_mode,
                    )
                )
                output = model.forward_search(
                    self.board,
                    role_to_play=self.roles,
                    rule_features=self.rules,
                )
                self.assertEqual(tuple(output.policy_logits.shape), (2, 25))
                self.assertEqual(tuple(output.wdl_logits.shape), (2, 3))

    def test_attention_fusion_has_strict_head_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "fusion_attention_heads"):
            ModelConfig(
                architecture="multiview3d_fusion_resnet",
                channels=10,
                blocks=1,
                branch_channels=4,
                volume_channels=4,
                volume_blocks=1,
                fusion_mode="attention",
                fusion_attention_heads=4,
            )
        with self.assertRaisesRegex(ValueError, "only for attention"):
            ModelConfig(
                architecture="multiview3d_fusion_resnet",
                channels=8,
                blocks=1,
                branch_channels=4,
                volume_channels=4,
                volume_blocks=1,
                fusion_mode="concat",
                fusion_attention_heads=2,
            )
        config = ModelConfig(
            architecture="multiview3d_fusion_resnet",
            channels=8,
            blocks=1,
            branch_channels=4,
            volume_channels=6,
            volume_blocks=1,
            fusion_mode="attention",
            fusion_attention_heads=2,
        )
        model = build_model(config)
        output = model.forward_search(
            self.board,
            role_to_play=self.roles,
            rule_features=self.rules,
        )
        self.assertEqual(tuple(output.policy_logits.shape), (2, 25))
        self.assertEqual(model.export_config()["fusion_attention_heads"], 2)

    def test_post_trunk_attention_modes_preserve_all_head_shapes(self) -> None:
        for mode in ("serial_attention", "parallel_attention"):
            with self.subTest(mode=mode):
                config = ModelConfig(
                    architecture="column3d_fusion_v2",
                    channels=8,
                    blocks=1,
                    encoder_channels=8,
                    branch_channels=4,
                    volume_channels=6,
                    volume_blocks=1,
                    fusion_mode="concat",
                    post_trunk_mode=mode,
                    post_attention_blocks=2,
                    post_attention_heads=2,
                    post_attention_mlp_ratio=2.0,
                )
                model = build_model(config)
                output = model(
                    self.board,
                    role_to_play=self.roles,
                    rule_features=self.rules,
                )
                self.assertEqual(tuple(output.policy_logits.shape), (2, 25))
                self.assertEqual(tuple(output.wdl_logits.shape), (2, 3))
                self.assertEqual(tuple(output.opponent_reply_logits.shape), (2, 25))
                self.assertEqual(tuple(output.future_occupancy_logits.shape), (2, 3, 6, 5, 5))
                self.assertEqual(tuple(output.moves_left_logits.shape), (2, 301))
                exported = model.export_config()
                self.assertEqual(exported["post_trunk_mode"], mode)
                self.assertEqual(exported["post_attention_blocks"], 2)
                self.assertEqual(exported["post_attention_heads"], 2)

    def test_post_trunk_changes_hash_and_rejects_baseline_state(self) -> None:
        baseline = ModelConfig(
            architecture="column3d_fusion_v2",
            channels=8,
            blocks=1,
            encoder_channels=8,
            branch_channels=4,
            volume_channels=4,
            volume_blocks=1,
            fusion_mode="concat",
        )
        serial = replace(
            baseline,
            post_trunk_mode="serial_attention",
            post_attention_blocks=2,
            post_attention_heads=2,
            post_attention_mlp_ratio=2.0,
        )
        self.assertNotIn("post_trunk_mode", model_config_dict(baseline))
        self.assertNotEqual(
            config_hash(V3Config(model=baseline)),
            config_hash(V3Config(model=serial)),
        )
        with self.assertRaises(RuntimeError):
            build_model(serial).load_state_dict(build_model(baseline).state_dict(), strict=True)

    def test_v2_fusion_decouples_2d_and_3d_widths_without_changing_trunk(self) -> None:
        narrow = build_model(
            ModelConfig(
                architecture="multiview3d_fusion_resnet",
                channels=24,
                blocks=2,
                branch_channels=8,
                volume_channels=4,
                volume_blocks=1,
                fusion_mode="concat",
            )
        )
        wide = build_model(
            ModelConfig(
                architecture="multiview3d_fusion_resnet",
                channels=24,
                blocks=2,
                branch_channels=8,
                volume_channels=12,
                volume_blocks=1,
                fusion_mode="concat",
            )
        )
        self.assertEqual(narrow.blocks[0].conv1.weight.shape, wide.blocks[0].conv1.weight.shape)
        self.assertEqual(narrow.multiview_encoder.fusion.weight.shape, wide.multiview_encoder.fusion.weight.shape)
        self.assertEqual(narrow.fusion.projection.in_channels, 12)
        self.assertEqual(wide.fusion.projection.in_channels, 20)
        self.assertEqual(narrow.export_config()["volume_channels"], 4)

    def test_balance_grid_calibration_records_3d_allocation(self) -> None:
        row = calibrate_model_grid(
            variant_id="balance.column3d.concat",
            architecture="column3d_fusion_v2",
            anchor_channels=8,
            anchor_blocks=1,
            minimum_channels=8,
            maximum_channels=16,
            branch_fractions=(0.5,),
            volume_blocks=(1,),
        )
        self.assertEqual(row["variant_id"], "balance.column3d.concat")
        self.assertGreater(row["component_parameters"]["volume_3d"], 0)
        self.assertGreater(row["component_parameters"]["fusion"], 0)
        self.assertEqual(
            row["component_parameters"]["total"],
            sum(
                value
                for key, value in row["component_parameters"].items()
                if key != "total"
            ),
        )

        gravity = build_model(ModelConfig(channels=8, blocks=1))
        self.assertEqual(
            component_parameter_breakdown(gravity)["total"],
            sum(parameter.numel() for parameter in gravity.parameters()),
        )

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

        concat = V3Config.from_dict(
            {
                "model": model_config_dict(
                    ModelConfig(
                        architecture="plane3d_fusion_v2",
                        channels=8,
                        blocks=1,
                        branch_channels=4,
                        volume_blocks=1,
                        fusion_mode="concat",
                    )
                )
            }
        )
        gated = V3Config.from_dict(
            {
                "model": model_config_dict(
                    ModelConfig(
                        architecture="plane3d_fusion_v2",
                        channels=8,
                        blocks=1,
                        branch_channels=4,
                        volume_blocks=1,
                        fusion_mode="gated",
                    )
                )
            }
        )
        self.assertNotEqual(config_hash(concat), config_hash(gated))

        legacy_fusion = ModelConfig(
            architecture="multiview3d_fusion_resnet",
            channels=8,
            blocks=1,
            branch_channels=4,
            volume_blocks=1,
            fusion_mode="concat",
        )
        self.assertNotIn("volume_channels", model_config_dict(legacy_fusion))
        explicit_volume = replace(legacy_fusion, volume_channels=4)
        legacy_shapes = {name: value.shape for name, value in build_model(legacy_fusion).state_dict().items()}
        explicit_shapes = {
            name: value.shape for name, value in build_model(explicit_volume).state_dict().items()
        }
        self.assertEqual(legacy_shapes, explicit_shapes)
        self.assertNotEqual(config_hash(V3Config(model=legacy_fusion)), config_hash(V3Config(model=explicit_volume)))

    def test_cross_architecture_state_is_strictly_incompatible(self) -> None:
        column = build_model(ModelConfig(architecture="column_resnet", channels=8, blocks=1))
        multiview = build_model(ModelConfig(architecture="multiview_resnet", channels=8, blocks=1))
        with self.assertRaises(RuntimeError):
            multiview.load_state_dict(column.state_dict(), strict=True)

    def test_balance_v2_checkpoint_round_trip_is_exact(self) -> None:
        config = ModelConfig(
            architecture="plane3d_fusion_v2",
            channels=8,
            blocks=1,
            branch_channels=4,
            volume_blocks=1,
            collapse_mode="mean",
            fusion_mode="gated",
        )
        original = build_model(config).eval()
        restored = build_model(config).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        with torch.inference_mode():
            expected = original.forward_search(
                self.board, role_to_play=self.roles, rule_features=self.rules
            )
            actual = restored.forward_search(
                self.board, role_to_play=self.roles, rule_features=self.rules
            )
        torch.testing.assert_close(actual.policy_logits, expected.policy_logits, rtol=0, atol=0)
        torch.testing.assert_close(actual.wdl_logits, expected.wdl_logits, rtol=0, atol=0)

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

    def test_stage2b_generator_pairs_cold_and_standard_late_model_only_warm_starts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2b-warm-configs-") as temporary:
            root = Path(temporary)
            matrix = calibrate_architecture_matrix(anchor_channels=8, anchor_blocks=1)
            matrix_path = root / "matrix.json"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            finalists = ["gravity_resnet", "column_resnet", "column_transformer"]
            finalists_path = root / "finalists.json"
            finalists_path.write_text(json.dumps({"finalists": finalists}), encoding="utf-8")
            models = {row["architecture"]: row["model"] for row in matrix["architectures"]}
            warm_rows = []
            for architecture in finalists:
                path = root / f"{architecture}.pt"
                model = build_model(ModelConfig(**models[architecture]))
                torch.save(
                    {
                        "format": "connect4-v3-model",
                        "format_version": 1,
                        "model_config": models[architecture],
                        "model_state": model.state_dict(),
                        "metadata": {
                            "lineage": "v3_stage2_offline",
                            "train_regime": "standard_late",
                            "train_positions": 1_000_000,
                        },
                    },
                    path,
                )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                warm_rows.append(
                    {
                        "architecture": architecture,
                        "checkpoint": str(path),
                        "checkpoint_sha256": digest,
                    }
                )
            warm_path = root / "warm.json"
            warm_path.write_text(
                json.dumps(
                    {
                        "schema": "connect4-v3-stage2b-warm-starts-v1",
                        "warm_starts": warm_rows,
                    }
                ),
                encoding="utf-8",
            )
            base = Path(__file__).resolve().parents[1] / "training/v3/configs/smoke_cpu.json"
            manifest = generate_stage2b_configs(
                base_config_path=base,
                architecture_matrix_path=matrix_path,
                finalists_path=finalists_path,
                output_dir=root / "generated",
                warm_starts_path=warm_path,
            )
            self.assertEqual(len(manifest["runs"]), 12)
            self.assertEqual(
                {row["initialization"] for row in manifest["runs"]}, {"cold", "warm"}
            )
            for row in manifest["runs"]:
                config = V3Config.from_dict(json.loads(Path(row["config"]).read_text()))
                if row["initialization"] == "warm":
                    self.assertEqual(
                        config.run.warm_start_mode,
                        "model_only_fresh_optimizer_replay_v1",
                    )
                    self.assertTrue(config.run.warm_start_checkpoint_sha256)
                else:
                    self.assertEqual(config.run.warm_start_mode, "")


if __name__ == "__main__":
    unittest.main()
