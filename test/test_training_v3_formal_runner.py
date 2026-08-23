from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from training.v3.config import load_config
from training.v3.formal_runner import run_formal


ROOT = Path(__file__).resolve().parents[1]


class FormalRunnerTests(unittest.TestCase):
    def _config(self, run_dir: Path, *, resume: bool = False):
        base = load_config(ROOT / "training" / "v3" / "configs" / "smoke_cpu.json")
        return replace(
            base,
            run=replace(
                base.run,
                run_id="formal_runner_test",
                run_dir=str(run_dir),
                resume=resume,
            ),
            learner=replace(
                base.learner,
                max_optimizer_steps_per_cycle=1,
                future_occupancy_loss_weight=0.14,
                future_occupancy_class_weights=(1.0, 1.0, 0.9),
            ),
            runtime=replace(
                base.runtime,
                storage=replace(
                    base.runtime.storage,
                    mode="archive_ack_prune",
                    soft_used_fraction=0.99,
                    hard_free_gib=10.0,
                    bundle_target_gib=0.01,
                ),
            ),
        )

    def test_one_generation_commits_and_exact_position_bound_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            result = run_formal(
                self._config(run_dir),
                max_train_positions=5,
                max_generations=1,
            )
            self.assertEqual(result["status"], "stopped_at_safe_boundary")
            self.assertEqual(result["formal_loop_state"]["train_positions_consumed"], 5)
            self.assertEqual(result["formal_loop_state"]["next_generation"], 1)
            self.assertTrue(
                (run_dir / "manifests" / "generations" / "g000000.json").is_file()
            )
            self.assertFalse((run_dir / "manifests" / "coordinator.lock").exists())
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stop_reason"], "max_train_positions")

            resumed = run_formal(
                self._config(run_dir, resume=True),
                max_train_positions=5,
                max_generations=1,
            )
            self.assertEqual(resumed["generations_completed"], 0)
            self.assertEqual(resumed["formal_loop_state"]["train_positions_consumed"], 5)

    def test_second_generation_uses_only_the_previous_committed_champion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            result = run_formal(
                self._config(run_dir),
                max_train_positions=16,
                max_generations=2,
            )
            self.assertEqual(result["generations_completed"], 2)
            first, second = result["results"]
            self.assertNotEqual(first["stability"]["action"], "pause")
            self.assertNotEqual(second["stability"]["action"], "pause")
            expected_producer = first["accepted_model_id"] or "random"
            self.assertEqual(second["producer_model_id"], expected_producer)
            commit = json.loads(
                (run_dir / "manifests" / "generations" / "g000001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertLessEqual(
                commit["replay_raw_positions"], commit["replay_cumulative_positions"]
            )
            self.assertTrue(
                all(
                    {"position_start", "position_end"}.issubset(row)
                    for row in commit["replay_shards"]
                )
            )
            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics" / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            selfplay_rows = [row for row in metrics if row.get("stage") == "selfplay"]
            self.assertEqual(len(selfplay_rows), 2)
            for row in selfplay_rows:
                phases = row["health"]["exploration_by_phase"]
                self.assertEqual([phase["start_ply"] for phase in phases], [0, 12])
                self.assertGreater(phases[0]["position_count"], 0)
                self.assertTrue(0.0 <= phases[0]["selected_top1_rate"] <= 1.0)
            resumed = run_formal(
                self._config(run_dir, resume=True),
                max_train_positions=16,
                max_generations=1,
            )
            self.assertEqual(resumed["generations_completed"], 0)
            self.assertEqual(resumed["formal_loop_state"]["replay_positions"], result["formal_loop_state"]["replay_positions"])

    def test_provisional_auxiliary_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = load_config(ROOT / "training" / "v3" / "configs" / "smoke_cpu.json")
            config = replace(
                base,
                run=replace(base.run, run_dir=str(Path(directory) / "run")),
                runtime=replace(
                    base.runtime,
                    storage=replace(
                        base.runtime.storage,
                        mode="archive_ack_prune",
                        hard_free_gib=10.0,
                    ),
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "provisional P6"):
                run_formal(config, max_train_positions=5)


if __name__ == "__main__":
    unittest.main()
