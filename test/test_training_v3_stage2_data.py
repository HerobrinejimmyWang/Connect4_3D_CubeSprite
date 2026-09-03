from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.v3.replay import ReplayShard, load_replay_shard, write_replay_shard
from training.v3.anchored_elo import load_v3_artifact_predictor
from training.v3.stage2.data import audit_trajectory, freeze_regime_datasets
from training.v3.stage2.offline import train_offline


def make_shard(generation: int, count: int = 100) -> ReplayShard:
    game_ids = np.arange(generation * 10_000, generation * 10_000 + count, dtype=np.uint64)
    visits = np.zeros((count, 25), dtype=np.uint32)
    visits[:, generation % 25] = 8
    return ReplayShard(
        board=np.zeros((count, 6, 5, 5), dtype=np.int8),
        visit_counts=visits,
        policy_weight=np.ones(count, dtype=np.float32),
        wdl=np.full(count, 1, dtype=np.uint8),
        game_id=game_ids,
        turn_index=np.zeros(count, dtype=np.uint16),
        player_to_move=np.ones(count, dtype=np.int8),
        search_kind=np.ones(count, dtype=np.uint8),
        rule_code=np.zeros(count, dtype=np.uint16),
        turn_kind=np.zeros(count, dtype=np.uint8),
        placement_count=np.zeros(count, dtype=np.uint16),
        opponent_reply_column=np.full(count, -1, dtype=np.int8),
        opponent_reply_mask=np.zeros(count, dtype=np.uint8),
        terminal_board=np.zeros((count, 6, 5, 5), dtype=np.int8),
        remaining_turns=np.zeros(count, dtype=np.uint16),
    )


class Stage2DataTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        source = root / "replay"
        metrics = root / "metrics.jsonl"
        rows = []
        for generation in range(9):
            write_replay_shard(
                source / f"g{generation:03d}.npz",
                make_shard(generation),
                {
                    "run_id": "stage1_b10_standard_test",
                    "generation": generation,
                    "producer_model_id": f"accepted-g{generation:03d}",
                    "seed_range": {"start": generation, "end": generation},
                    "results": {},
                    "search_config": {"full_search_sims": 256},
                    "rule_registry_hash": "a" * 64,
                    "config_hash": "b" * 64,
                    "git_commit": "synthetic-test",
                },
            )
            phase = generation // 3
            rows.append(
                {
                    "generation": generation,
                    "anchored_strength": phase * 10.0 + generation * 0.01,
                    "mean_game_length": 20.0 + phase * 5.0,
                    "short_game_rate": 0.6 - phase * 0.2,
                    "policy_entropy": 2.5 - phase * 0.4,
                    "accepted_cadence": 1.0 - phase * 0.25,
                }
            )
        metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        mixed_source = root / "mixed-replay"
        mixed_metrics = root / "mixed-metrics.jsonl"
        mixed_rows = []
        for generation in range(9, 12):
            mixed_metadata = {
                "run_id": "stage1_b10_mixed_test",
                "generation": generation,
                "producer_model_id": f"accepted-mixed-g{generation:03d}",
                "seed_range": {"start": generation, "end": generation},
                "results": {},
                "search_config": {"full_search_sims": 256},
                "rule_registry_hash": "a" * 64,
                "config_hash": "c" * 64,
                "git_commit": "synthetic-test",
            }
            write_replay_shard(
                mixed_source / f"g{generation:03d}.npz", make_shard(generation), mixed_metadata
            )
            mixed_rows.append(
                {
                    "generation": generation,
                    "anchored_strength": 30.0 + generation,
                    "mean_game_length": 36.0,
                    "short_game_rate": 0.12,
                    "policy_entropy": 1.4,
                    "accepted_cadence": 0.3,
                }
            )
        mixed_metrics.write_text(
            "\n".join(json.dumps(row) for row in mixed_rows) + "\n", encoding="utf-8"
        )
        return source, metrics, mixed_source, mixed_metrics

    def test_audit_and_freeze_are_deterministic_and_game_disjoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-data-") as temporary:
            root = Path(temporary)
            source, metrics, mixed_source, mixed_metrics = self._fixture(root)
            options = {
                "mixed_source_dir": mixed_source,
                "mixed_metrics_path": mixed_metrics,
                "standard_lineage_prefix": "stage1_b10_standard_test",
                "mixed_lineage_prefix": "stage1_b10_mixed_test",
            }
            first = audit_trajectory(source, metrics, **options)
            second = audit_trajectory(source, metrics, **options)
            self.assertEqual(first["segments"], second["segments"])
            self.assertEqual(
                [(first["segments"][name]["generation_start"], first["segments"][name]["generation_end"])
                 for name in ("standard_early", "standard_mid", "standard_late")],
                [(0, 2), (3, 5), (6, 8)],
            )
            self.assertEqual(first["segments"]["mixed_late"]["generation_start"], 9)
            frozen = freeze_regime_datasets(
                first,
                root / "pools",
                train_positions=40,
                validation_positions=5,
                seed=17,
                validation_fraction=0.2,
            )
            for regime in ("standard_early", "standard_mid", "standard_late", "mixed_late"):
                train, _ = load_replay_shard(frozen["regimes"][regime]["train"]["path"])
                validation, _ = load_replay_shard(
                    frozen["regimes"][regime]["validation"]["path"]
                )
                self.assertEqual(len(train), 40)
                self.assertEqual(len(validation), 5)
                self.assertFalse(set(map(int, train.game_id)) & set(map(int, validation.game_id)))

    def test_audit_rejects_missing_metrics_and_checksum_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-data-invalid-") as temporary:
            root = Path(temporary)
            source, metrics, mixed_source, mixed_metrics = self._fixture(root)
            rows = metrics.read_text(encoding="utf-8").splitlines()
            broken = json.loads(rows[0])
            broken.pop("policy_entropy")
            rows[0] = json.dumps(broken)
            metrics.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                audit_trajectory(
                    source,
                    metrics,
                    mixed_source_dir=mixed_source,
                    mixed_metrics_path=mixed_metrics,
                    standard_lineage_prefix="stage1_b10_standard_test",
                    mixed_lineage_prefix="stage1_b10_mixed_test",
                )

            shard = source / "g000.npz"
            content = bytearray(shard.read_bytes())
            content[-1] ^= 1
            shard.write_bytes(bytes(content))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_replay_shard(shard, verify_checksum=True)

    def test_offline_resume_matches_continuous_training(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-offline-resume-") as temporary:
            root = Path(temporary)
            replay_dir = root / "replay"
            metadata = {
                "run_id": "stage2-test",
                "generation": 0,
                "producer_model_id": "frozen-test",
                "seed_range": {"start": 0, "end": 0},
                "results": {},
                "search_config": {},
                "rule_registry_hash": "a" * 64,
                "config_hash": "b" * 64,
                "git_commit": "synthetic-test",
            }
            def frozen_metadata(regime: str) -> dict[str, object]:
                recipe = (
                    "b10_mixed_opening_position_balanced_v1"
                    if regime == "mixed_late"
                    else "b10_standard_v1"
                )
                return {
                    **metadata,
                    "results": {"regime": regime, "data_recipe_id": recipe},
                }

            train_path = replay_dir / "standard-early-train.npz"
            write_replay_shard(
                train_path, make_shard(0, 32), frozen_metadata("standard_early")
            )
            validation_paths = {}
            for index, regime in enumerate(
                ("standard_early", "standard_mid", "standard_late", "mixed_late"), 1
            ):
                validation_path = replay_dir / f"{regime}-validation.npz"
                write_replay_shard(
                    validation_path, make_shard(index, 8), frozen_metadata(regime)
                )
                validation_paths[regime] = validation_path
            base_config = Path(__file__).resolve().parents[1] / "training/v3/configs/smoke_cpu.json"

            def write_config(
                output: Path,
                target: int,
                resume: bool,
                *,
                train_regime: str = "standard_early",
                train_replay: Path = train_path,
                warm_start_checkpoint: Path | None = None,
            ) -> Path:
                config = {
                    "base_config": str(base_config),
                    "model": {"architecture": "column_resnet", "channels": 8, "blocks": 1},
                    "train_regime": train_regime,
                    "train_replay": str(train_replay),
                    "validation_replays": {
                        regime: str(path) for regime, path in validation_paths.items()
                    },
                    "output_dir": str(output),
                    "seed": 1234,
                    "target_positions": target,
                    "device": "cpu",
                    "resume": resume,
                }
                if warm_start_checkpoint is not None:
                    config["warm_start_checkpoint"] = str(warm_start_checkpoint)
                path = output.parent / f"{output.name}-{target}-{int(resume)}.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                return path

            continuous_dir = root / "continuous"
            train_offline(write_config(continuous_dir, 16, False))
            resumed_dir = root / "resumed"
            train_offline(write_config(resumed_dir, 8, False))
            train_offline(write_config(resumed_dir, 16, True))
            continuous = torch.load(continuous_dir / "checkpoint.pt", weights_only=False)
            resumed = torch.load(resumed_dir / "checkpoint.pt", weights_only=False)
            self.assertEqual(continuous["learner_state"], resumed["learner_state"])
            self.assertEqual(continuous["model_state"].keys(), resumed["model_state"].keys())
            for name in continuous["model_state"]:
                self.assertTrue(torch.equal(continuous["model_state"][name], resumed["model_state"][name]), name)
            predictor, identity = load_v3_artifact_predictor(resumed_dir / "model.pt")
            self.assertEqual(
                identity["model_id"],
                "stage2-offline-column_resnet-standard_early-s1234-p16",
            )

            mixed_train_path = replay_dir / "mixed-train.npz"
            write_replay_shard(
                mixed_train_path, make_shard(6, 32), frozen_metadata("mixed_late")
            )
            with self.assertRaisesRegex(ValueError, "promotion-only"):
                train_offline(
                    write_config(
                        root / "invalid-mixed",
                        8,
                        False,
                        train_regime="mixed_late",
                        train_replay=mixed_train_path,
                    )
                )

            standard_late_train_path = replay_dir / "standard-late-train.npz"
            write_replay_shard(
                standard_late_train_path,
                make_shard(7, 32),
                frozen_metadata("standard_late"),
            )
            standard_late_dir = root / "standard-late-parent"
            train_offline(
                write_config(
                    standard_late_dir,
                    8,
                    False,
                    train_regime="standard_late",
                    train_replay=standard_late_train_path,
                )
            )

            promoted_dir = root / "promoted-mixed"
            promoted_report = train_offline(
                write_config(
                    promoted_dir,
                    8,
                    False,
                    train_regime="mixed_late",
                    train_replay=mixed_train_path,
                    warm_start_checkpoint=standard_late_dir / "checkpoint.pt",
                )
            )
            promoted = torch.load(promoted_dir / "checkpoint.pt", weights_only=False)
            self.assertEqual(promoted["learner_state"]["sample_cursor"], 8)
            self.assertTrue(promoted["warm_start_sha256"])
            self.assertEqual(promoted_report["train_regime"], "mixed_late")


if __name__ == "__main__":
    unittest.main()
