from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from training.v3.checkpoint import CheckpointV1, save_checkpoint
from training.v3.config import ModelConfig
from training.v3.evaluation_snapshot import export_evaluation_snapshot
from training.v3.model import build_model


class EvaluationSnapshotTests(unittest.TestCase):
    def test_checkpoint_projection_is_explicitly_evaluation_only(self) -> None:
        config = ModelConfig(channels=16, blocks=1)
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters())
        checkpoint = CheckpointV1.capture(
            model=model,
            optimizer=optimizer,
            global_step=7,
            generation=3,
            replay_cursor={},
            sample_ids=(),
            accepted_model_id="accepted-old",
            candidate_model_id=None,
            config_hash="config-hash",
            code_version="test-code",
            extra_state={
                "model_config": asdict(config),
                "train_positions_consumed": 1234,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = save_checkpoint(root / "g000003-s00000007.pt", checkpoint)
            target = root / "evaluation" / "snapshot.pt"
            result = export_evaluation_snapshot(source, target, model_id="milestone-1234")
            payload = torch.load(target, map_location="cpu", weights_only=False)

            self.assertEqual(payload["format"], "connect4-v3-model")
            self.assertEqual(payload["metadata"]["model_id"], "milestone-1234")
            self.assertTrue(payload["metadata"]["evaluation_only"])
            self.assertFalse(payload["metadata"]["eligible_for_acceptance"])
            self.assertFalse(payload["metadata"]["eligible_for_selfplay"])
            self.assertEqual(payload["metadata"]["train_positions_consumed"], 1234)
            self.assertEqual(result["source_checkpoint_sha256"], payload["metadata"]["source_checkpoint_sha256"])

    def test_snapshot_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "exists.pt"
            target.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                export_evaluation_snapshot(Path(directory) / "missing.pt", target)
            self.assertEqual(target.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
