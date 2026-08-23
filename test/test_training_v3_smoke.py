from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from training.v3.config import GateConfig, load_config
from training.v3.cli import main as cli_main
from training.v3.pipeline import _run_sequential_gate, run_smoke
from training.v3.preflight import PreflightError, run_preflight
from training.v3.replay import load_replay_shard, replay_ready_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "training" / "v3" / "configs" / "smoke_cpu.json"


class PreflightTests(unittest.TestCase):
    def test_dependency_failure_is_actionable(self) -> None:
        with mock.patch("training.v3.preflight.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(PreflightError, "requirements-v3.txt"):
                run_preflight("cpu")

    def test_preflight_rejects_cuda_index_outside_visible_inventory(self) -> None:
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=1),
        ):
            with self.assertRaisesRegex(PreflightError, "visible inventory"):
                run_preflight("cuda:2")


class SequentialGateTests(unittest.TestCase):
    def test_gate_appends_only_new_pairs_until_resolved(self) -> None:
        config = load_config(SMOKE_CONFIG)
        config = replace(
            config,
            gate=GateConfig(
                bootstrap_candidate_train_positions=8,
                candidate_train_positions=16,
                initial_opening_pairs=2,
                pair_increment=2,
                max_opening_pairs=6,
                opening_depths=(0, 2, 4, 6),
                search_schedule=config.gate.search_schedule,
                cpuct=config.gate.cpuct,
                bootstrap_samples=config.gate.bootstrap_samples,
                confidence=config.gate.confidence,
                role_floor=config.gate.role_floor,
                accept_threshold=config.gate.accept_threshold,
            ),
        )

        def decision(verdict: str, pairs: int):
            return SimpleNamespace(
                verdict=verdict,
                summary=SimpleNamespace(
                    overall=SimpleNamespace(point_score=0.5),
                    ci_lower=0.4,
                    ci_upper=0.6,
                ),
                pairs=pairs,
            )

        played_slices = []

        def play(openings, **_kwargs):
            played_slices.append(tuple(openings))
            return [object()] * (2 * len(openings))

        with (
            mock.patch("training.v3.pipeline.play_paired_openings", side_effect=play),
            mock.patch(
                "training.v3.pipeline.evaluate_gate",
                side_effect=[
                    decision("inconclusive", 2),
                    decision("inconclusive", 4),
                    decision("accept", 6),
                ],
            ),
        ):
            results, final_decision, looks = _run_sequential_gate(
                config,
                generation=0,
                openings=list(range(6)),
                candidate_predictor=object(),
                incumbent_predictor=None,
            )

        self.assertEqual(played_slices, [(0, 1), (2, 3), (4, 5)])
        self.assertEqual(len(results), 12)
        self.assertEqual(final_decision.verdict, "accept")
        self.assertEqual([look["pairs"] for look in looks], [2, 4, 6])

    def test_cli_print_config_and_guarded_run(self) -> None:
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = cli_main(["print-config", "--config", str(SMOKE_CONFIG)])
        self.assertEqual(code, 0, errors.getvalue())
        self.assertEqual(set(json.loads(output.getvalue())), {
            "run", "model", "selfplay", "replay", "learner", "gate", "runtime"
        })

        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = cli_main(["run", "--config", str(SMOKE_CONFIG)])
        self.assertEqual(code, 0, errors.getvalue())
        status = json.loads(output.getvalue())
        self.assertEqual(status["status"], "bounded-formal-run-available")
        self.assertFalse(status["production_ready"])
        self.assertEqual(
            status["storage_plan"]["deletion_enabled"],
            "explicit_receipt_gated_command_only",
        )
        self.assertTrue(status["blocking_items"])


class EndToEndSmokeTests(unittest.TestCase):
    def test_smoke_rejects_gpu_planning_preset(self) -> None:
        config = load_config(SMOKE_CONFIG)
        gpu_runtime = replace(
            config.runtime,
            device="cuda:0",
            selfplay_devices=("cuda:0",),
            learner_amp=True,
        )
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            run_smoke(replace(config, runtime=gpu_runtime))

    def test_random_replay_train_gate_checkpoint_and_resume(self) -> None:
        config = load_config(SMOKE_CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            config = replace(
                config,
                run=replace(config.run, run_dir=str(run_dir)),
                replay=replace(config.replay, shard_games=2),
            )
            result = run_smoke(config)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["producer_model_id"], "random")
            self.assertEqual(result["games"], 4)
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertGreater(result["full_policy_positions"], 0)
            self.assertLessEqual(result["full_policy_positions"], result["raw_positions"])
            self.assertIn(result["gate_verdict"], {"accept", "reject", "inconclusive"})
            self.assertTrue(result["resume_verification"]["passed"])
            self.assertTrue(all(result["resume_verification"]["checks"].values()))
            self.assertEqual(result["formal_loop_state"]["next_generation"], 1)
            self.assertEqual(result["formal_loop_state"]["next_game_id"], 4)
            self.assertEqual(
                result["formal_loop_state"]["replay_positions"],
                result["raw_positions"],
            )

            replay_paths = [Path(path) for path in result["artifacts"]["replay_shards"]]
            self.assertEqual(len(replay_paths), 2)
            restored = [load_replay_shard(path) for path in replay_paths]
            self.assertEqual(sum(len(replay) for replay, _manifest in restored), result["raw_positions"])
            for replay_path, (_replay, manifest) in zip(replay_paths, restored, strict=True):
                self.assertEqual(manifest["producer_model_id"], "random")
                self.assertEqual(manifest["results"]["games"], 2)
                self.assertEqual(manifest["compressed_bytes"], replay_path.stat().st_size)
                self.assertTrue(replay_ready_path(replay_path).is_file())
            self.assertTrue(Path(result["checkpoint"]).is_file())
            self.assertTrue(Path(result["artifacts"]["audit_index"]).is_file())
            self.assertTrue(result["artifacts"]["audit_replays"])
            self.assertTrue(Path(result["artifacts"]["generation_commit"]).is_file())
            pointer_path = run_dir / "manifests" / "latest_generation.json"
            self.assertTrue(pointer_path.is_file())
            self.assertNotEqual(Path(result["checkpoint"]).name, "latest.pt")
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            self.assertEqual(pointer["schema_version"], 1)
            self.assertEqual(len(pointer["commit_sha256"]), 64)
            self.assertTrue((run_dir / "resolved_config.json").is_file())
            self.assertEqual(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["status"], "complete")

            invalid_pointer = dict(pointer)
            invalid_pointer["generation"] = -1
            pointer_path.write_text(json.dumps(invalid_pointer), encoding="utf-8")
            fallback = run_smoke(replace(config, run=replace(config.run, resume=True)))
            self.assertEqual(fallback["status"], "resume-probe-complete")
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

            replay_bytes = bytearray(replay_paths[0].read_bytes())
            replay_bytes[-1] ^= 1
            replay_paths[0].write_bytes(replay_bytes)
            with self.assertRaisesRegex(ValueError, "no complete committed"):
                run_smoke(replace(config, run=replace(config.run, resume=True)))
            replay_bytes[-1] ^= 1
            replay_paths[0].write_bytes(replay_bytes)

            audit_replay_path = Path(result["artifacts"]["audit_replays"][0])
            audit_replay_bytes = bytearray(audit_replay_path.read_bytes())
            audit_replay_bytes[-2] ^= 1
            audit_replay_path.write_bytes(audit_replay_bytes)
            with self.assertRaisesRegex(ValueError, "no complete committed"):
                run_smoke(replace(config, run=replace(config.run, resume=True)))
            audit_replay_bytes[-2] ^= 1
            audit_replay_path.write_bytes(audit_replay_bytes)

            resumed = run_smoke(replace(config, run=replace(config.run, resume=True)))
            self.assertEqual(resumed["status"], "resume-probe-complete")
            self.assertEqual(resumed["restored_global_step"], 2)
            self.assertEqual(resumed["probe_global_step"], 3)
            self.assertEqual(resumed["formal_loop_state"], result["formal_loop_state"])
            self.assertTrue(resumed["sample_ids"])

            checkpoint_bytes = bytearray(Path(result["checkpoint"]).read_bytes())
            checkpoint_bytes[-1] ^= 1
            Path(result["checkpoint"]).write_bytes(checkpoint_bytes)
            with self.assertRaisesRegex(ValueError, "no complete committed"):
                run_smoke(replace(config, run=replace(config.run, resume=True)))


if __name__ == "__main__":
    unittest.main()
