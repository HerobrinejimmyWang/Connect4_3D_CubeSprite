from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from training.v3.config import OpeningTemperatureMixtureConfig, load_config
from training.v3.formal_runner import _resolve_exhausted_pending_gate, run_formal
from training.v3.formal_state import FormalLoopState, PendingCandidateState
from training.v3.layout import RunLayout


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

    def test_legacy_terminal_inconclusive_is_audited_and_cleared_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = RunLayout.from_root(Path(directory) / "run").create()
            candidate_id = "candidate-g000039-s00003808-d00242393"
            candidate_path = layout.candidates / f"{candidate_id}.pt"
            candidate_path.write_bytes(b"frozen candidate")
            gate_path = layout.metrics / "gate_g000039.json"
            gate_path.write_text(
                json.dumps(
                    {
                        "candidate_model_id": candidate_id,
                        "incumbent_model_id": "accepted-g32",
                        "verdict": "inconclusive",
                        "max_pairs": 2,
                        "games": [{}, {}, {}, {}],
                    }
                ),
                encoding="utf-8",
            )
            pending = PendingCandidateState(
                candidate_model_id=candidate_id,
                candidate_path=f"candidates/{candidate_id}.pt",
                incumbent_model_id="accepted-g32",
                gate_path="metrics/gate_g000039.json",
                opening_manifest="manifests/gate_openings.json",
                pairs_evaluated=2,
                max_pairs=2,
            )
            state = FormalLoopState(
                accepted_model_id="accepted-g32", pending_candidate=pending
            )

            resolved = _resolve_exhausted_pending_gate(layout, state)
            self.assertIsNone(resolved.pending_candidate)
            self.assertEqual(resolved.accepted_model_id, "accepted-g32")
            audit = layout.metrics / f"gate_resolution_{candidate_id}.json"
            payload = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(payload["resolution"], "reject")
            self.assertEqual(
                payload["rejection_basis"], "insufficient_evidence_at_max_pairs"
            )
            self.assertTrue(candidate_path.is_file())
            self.assertEqual(_resolve_exhausted_pending_gate(layout, state), resolved)

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
            self.assertEqual(len(manifest["runtime_invocations"]), 1)
            self.assertEqual(
                manifest["active_runtime"],
                manifest["runtime_invocations"][0]["runtime"],
            )

            resumed = run_formal(
                self._config(run_dir, resume=True),
                max_train_positions=5,
                max_generations=1,
            )
            self.assertEqual(resumed["generations_completed"], 0)
            self.assertEqual(resumed["formal_loop_state"]["train_positions_consumed"], 5)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["runtime_invocations"]), 2)

    def test_formal_mixture_balances_consumed_positions_independently_of_raw_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "mixed"
            base = self._config(run_dir)
            config = replace(
                base,
                selfplay=replace(
                    base.selfplay,
                    opening_temperature_mixture=OpeningTemperatureMixtureConfig(
                        enabled=True
                    ),
                ),
                replay=replace(
                    base.replay,
                    train_fraction=0.999999,
                    window_c=1000,
                ),
            )
            result = run_formal(
                config,
                max_train_positions=8,
                max_generations=1,
            )
            self.assertEqual(result["formal_loop_state"]["train_positions_consumed"], 8)
            metrics = [
                json.loads(line)
                for line in (run_dir / "metrics" / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            learner = next(row for row in metrics if row["stage"] == "learner")
            self.assertEqual(
                learner["sampling_group_positions"],
                {"baseline": 4, "lowered_opening_temperature": 4},
            )
            selfplay = next(row for row in metrics if row["stage"] == "selfplay")
            variants = selfplay["health"]["opening_temperature_mixture"]["variants"]
            self.assertEqual(variants["baseline"]["games"], 2)
            self.assertEqual(variants["lowered_opening_temperature"]["games"], 2)
            selection = json.loads(
                (run_dir / "replay" / "shuffle" / "selection_g000000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                selection["position_balanced_sampling"]["target_train_position_fractions"],
                {"baseline": 0.5, "lowered_opening_temperature": 0.5},
            )

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

    def test_cold_start_relative_role_gate_uses_random_bootstrap_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            base = self._config(run_dir)
            config = replace(
                base,
                gate=replace(
                    base.gate,
                    bootstrap_candidate_train_positions=1,
                    role_floor=0.0,
                    role_hard_reject_floor=0.0,
                    role_guard_mode="relative_noninferiority",
                    role_noninferiority_margin=0.05,
                ),
            )
            result = run_formal(config, max_train_positions=5, max_generations=1)

            self.assertEqual(result["generations_completed"], 1)
            gate_path = run_dir / "metrics" / "gate_g000000.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["incumbent_model_id"], "random")
            self.assertEqual(gate["role_control_baseline"], "random_bootstrap")
            self.assertEqual(len(gate["role_control_games"]), len(gate["games"]))
            self.assertEqual(
                [row["evidence"] for row in gate["evaluation_runtime"]],
                ["candidate_vs_incumbent", "random_bootstrap_control"],
            )

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

    def test_warm_start_preserves_optimizer_and_resets_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_dir = root / "parent"
            parent_result = run_formal(
                self._config(parent_dir),
                max_train_positions=5,
                max_generations=1,
            )
            parent_checkpoint = Path(parent_result["results"][0]["checkpoint"])
            parent_sha256 = hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()
            base = self._config(root / "child")
            child = replace(
                base,
                run=replace(
                    base.run,
                    run_id="formal_runner_warm_child",
                    warm_start_checkpoint=str(parent_checkpoint),
                    warm_start_checkpoint_sha256=parent_sha256,
                    warm_start_mode="optimizer_fresh_replay_v1",
                ),
                selfplay=replace(base.selfplay, opening_full_search_plies=2),
            )
            child_result = run_formal(
                child,
                max_train_positions=5,
                max_generations=1,
            )
            self.assertEqual(child_result["generations_completed"], 1)
            self.assertEqual(child_result["results"][0]["generation"], 0)
            self.assertTrue(
                child_result["results"][0]["producer_model_id"].startswith("warmstart-")
            )
            self.assertEqual(child_result["formal_loop_state"]["train_positions_consumed"], 5)
            manifest = json.loads(
                (root / "child" / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["warm_start"]["replay_policy"], "fresh")


if __name__ == "__main__":
    unittest.main()
