import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from training.v3.checkpoint import CheckpointV1, load_checkpoint, save_checkpoint
from training.v3.gate import GateGameResult, evaluate_gate, summarize_paired_results
from training.v3.learner import (
    DeterministicKeyBatchSampler,
    OnlineD4Dataset,
    V3Learner,
    build_adamw,
)
from training.v3.replay import (
    ReplayShard,
    SEARCH_FAST,
    SEARCH_FULL,
    TrainTokenBucket,
    apply_d4,
    growing_window_size,
    inverse_d4_index,
    load_replay_shard,
    replay_manifest_path,
    replay_ready_path,
    select_active_replay,
    stable_game_split,
    stable_split_mask,
    write_replay_shard,
)


def _make_replay(count: int = 12, *, search_kind: int = SEARCH_FULL) -> ReplayShard:
    board = np.zeros((count, 6, 5, 5), dtype=np.int8)
    visits = np.zeros((count, 25), dtype=np.uint32)
    for index in range(count):
        row, column = divmod(index % 25, 5)
        board[index, 0, row, column] = 1 if index % 2 == 0 else -1
        visits[index, (index * 7 + 3) % 25] = np.uint32(index + 1)
    players = np.asarray(
        [1 if index % 2 == 0 else -1 for index in range(count)], dtype=np.int8
    )
    return ReplayShard(
        board=board,
        visit_counts=visits,
        policy_weight=np.full(
            (count,), 1.0 if search_kind == SEARCH_FULL else 0.0, dtype=np.float32
        ),
        wdl=np.asarray([index % 3 for index in range(count)], dtype=np.uint8),
        game_id=np.asarray([index // 3 for index in range(count)], dtype=np.uint64),
        turn_index=np.asarray([index % 3 + 1 for index in range(count)], dtype=np.uint16),
        player_to_move=players,
        search_kind=np.full((count,), search_kind, dtype=np.uint8),
        rule_code=np.zeros((count,), dtype=np.uint16),
        turn_kind=np.zeros((count,), dtype=np.uint8),
        placement_count=np.ones((count,), dtype=np.uint16),
        opponent_reply_column=np.full((count,), -1, dtype=np.int8),
        opponent_reply_mask=np.zeros((count,), dtype=np.uint8),
        terminal_board=board * players[:, None, None, None],
        remaining_turns=np.full((count,), 3, dtype=np.uint16),
    )


def _manifest_metadata() -> dict:
    return {
        "run_id": "unit-test",
        "generation": 0,
        "producer_model_id": "random",
        "seed_range": {"start": 100, "end": 103},
        "results": {"p1_wins": 1, "p2_wins": 1, "draws": 2},
        "search_config": {"full_search_sims": 8, "fast_search_sims": 2},
        "rule_registry_hash": "4c21f13e4e5c9529f0a2a3695bb70015893191bdad65a4208076438e79db90ca",
        "config_hash": "abc123",
        "git_commit": "test-tree",
    }


class ReplayShardTests(unittest.TestCase):
    def test_from_samples_keeps_fast_by_default_and_can_filter_to_full(self):
        source = _make_replay(2)
        samples = []
        for index, kind in enumerate(("full", "fast")):
            row = {
                name: value[index]
                for name, value in source.as_dict().items()
            }
            row["search_kind"] = kind
            row["policy_weight"] = 1.0 if kind == "full" else 0.0
            samples.append(row)
        all_positions = ReplayShard.from_samples(samples)
        full_positions = ReplayShard.from_samples(samples, full_only=True)
        np.testing.assert_array_equal(
            all_positions.search_kind, np.asarray([SEARCH_FULL, SEARCH_FAST], dtype=np.uint8)
        )
        self.assertEqual(len(full_positions), 1)
        self.assertEqual(int(full_positions.search_kind[0]), SEARCH_FULL)

    def test_append_only_round_trip_manifest_and_checksum(self):
        replay = _make_replay()
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "shard-000.npz"
            second = Path(temp_dir) / "shard-001.npz"
            manifest = write_replay_shard(first, replay, _manifest_metadata())
            second_manifest = write_replay_shard(second, replay, _manifest_metadata())

            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["sample_count"], len(replay))
            self.assertEqual(manifest["checksum_sha256"], second_manifest["checksum_sha256"])
            self.assertTrue(replay_manifest_path(first).is_file())
            restored, restored_manifest = load_replay_shard(first)
            self.assertEqual(restored_manifest, manifest)
            for name, expected in replay.as_dict().items():
                actual = restored.as_dict()[name]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, expected.dtype)
            with self.assertRaises(FileExistsError):
                write_replay_shard(first, replay, _manifest_metadata())

            incomplete = _manifest_metadata()
            incomplete.pop("git_commit")
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                write_replay_shard(Path(temp_dir) / "incomplete.npz", replay, incomplete)

    def test_checksum_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shard.npz"
            write_replay_shard(path, _make_replay(), _manifest_metadata())
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_replay_shard(path)

    def test_ready_marker_is_required_and_authenticated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shard.npz"
            write_replay_shard(path, _make_replay(), _manifest_metadata())
            marker_path = replay_ready_path(path)
            marker = marker_path.read_text(encoding="utf-8").replace(
                '"manifest_sha256": "', '"manifest_sha256": "0'
            )
            marker_path.write_text(marker, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ready marker"):
                load_replay_shard(path)

    def test_game_level_split_is_stable_and_near_five_percent(self):
        game_ids = np.repeat(np.arange(5000, dtype=np.uint64), 3)
        mask = stable_split_mask(game_ids, split="validation", split_seed=99)
        for start in range(0, len(mask), 3):
            self.assertTrue(np.all(mask[start : start + 3] == mask[start]))
        validation_games = int(mask.reshape(-1, 3)[:, 0].sum())
        self.assertGreater(validation_games, 150)
        self.assertLess(validation_games, 350)
        self.assertEqual(
            stable_game_split(42, split_seed=99),
            stable_game_split(42, split_seed=99),
        )

    def test_growing_window_and_token_bucket(self):
        self.assertEqual(growing_window_size(0, c=1, alpha=0.75, beta=0.4), 0)
        self.assertEqual(growing_window_size(1, c=1, alpha=0.75, beta=0.4), 1)
        self.assertEqual(growing_window_size(100, c=1, alpha=0.75, beta=0.4), 17)
        active = select_active_replay(_make_replay(12), c=1, alpha=0.75, beta=0.4)
        self.assertEqual(len(active), growing_window_size(12, c=1, alpha=0.75, beta=0.4))

        bucket = TrainTokenBucket(tokens_per_position=4.0)
        self.assertEqual(bucket.add(3), 12.0)
        self.assertEqual(bucket.consume(5), 5)
        self.assertEqual(bucket.consume(20), 7)
        self.assertEqual(bucket.consume(1), 0)
        self.assertEqual(bucket.train_data_ratio, 4.0)
        self.assertEqual(TrainTokenBucket.from_state_dict(bucket.state_dict()).state_dict(), bucket.state_dict())

    def test_all_d4_transforms_are_invertible_and_dataset_is_deterministic(self):
        replay = _make_replay(1)
        board = replay.board[0]
        policy = replay.visit_counts[0].reshape(5, 5)
        for transform in range(8):
            transformed_board = apply_d4(board, transform)
            transformed_policy = apply_d4(policy, transform)
            np.testing.assert_array_equal(
                apply_d4(transformed_board, inverse_d4_index(transform)), board
            )
            np.testing.assert_array_equal(
                apply_d4(transformed_policy, inverse_d4_index(transform)), policy
            )

        dataset = OnlineD4Dataset(replay, augmentation_seed=123)
        first = dataset[(0, 77)]
        second = dataset[(0, 77)]
        torch.testing.assert_close(first["board"], second["board"])
        torch.testing.assert_close(first["policy"], second["policy"])
        self.assertEqual(first["d4"], second["d4"])
        self.assertAlmostEqual(float(first["policy"].sum()), 1.0, places=7)
        self.assertEqual(float(first["policy_weight"]), 1.0)
        expected_legal = ((first["board"] != 0).sum(0) < 6).reshape(25)
        self.assertTrue(torch.equal(first["legal_mask"], expected_legal))

        fast_item = OnlineD4Dataset(
            _make_replay(1, search_kind=SEARCH_FAST), augmentation_seed=123
        )[(0, 77)]
        self.assertEqual(float(fast_item["policy_weight"]), 0.0)


class _TinyWDLNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Flatten(), nn.Linear(150, 24), nn.ReLU(), nn.Dropout(0.2))
        self.policy = nn.Linear(24, 25)
        self.wdl = nn.Linear(24, 3)

    def forward(self, board):
        hidden = self.hidden(board)
        return self.policy(hidden), self.wdl(hidden)


def _make_training_stack(initial_state=None, *, num_workers=0):
    model = _TinyWDLNet()
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = build_adamw(model, learning_rate=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    learner = V3Learner(
        model,
        optimizer,
        scheduler=scheduler,
        device="cpu",
        batch_size=4,
        grad_clip_norm=1.0,
        sample_seed=555,
        amp=False,
        num_workers=num_workers,
    )
    return model, optimizer, scheduler, learner


class _OneShotSkippingScaler:
    """Minimal GradScaler stand-in that skips exactly its first optimizer step."""

    def __init__(self) -> None:
        self.scale_value = 1024.0
        self.skip_next_step = True

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        if not self.skip_next_step:
            optimizer.step()

    def update(self):
        if self.skip_next_step:
            self.scale_value /= 2.0
            self.skip_next_step = False


class LearnerCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_losses_metrics_and_exact_resume(self):
        random.seed(10)
        np.random.seed(11)
        torch.manual_seed(12)
        initial_model = _TinyWDLNet()
        initial_state = copy.deepcopy(initial_model.state_dict())
        replay = _make_replay(12)
        dataset = OnlineD4Dataset(replay, augmentation_seed=444)
        model, optimizer, scheduler, learner = _make_training_stack(initial_state)
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(len(replay))

        first_metrics = learner.train_steps(dataset, steps=2, token_bucket=bucket)
        self.assertEqual(first_metrics.steps, 2)
        self.assertEqual(first_metrics.positions, 8)
        self.assertEqual(first_metrics.policy_positions, 8)
        self.assertEqual(first_metrics.value_positions, 8)
        self.assertTrue(np.isfinite(first_metrics.total_loss))
        self.assertGreaterEqual(first_metrics.brier_score, 0.0)
        self.assertLessEqual(first_metrics.calibration_error, 1.0)

        checkpoint = CheckpointV1.capture(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=learner.scaler,
            global_step=learner.global_step,
            generation=3,
            replay_cursor={"sample_cursor": learner.sample_cursor},
            sample_ids=learner.last_sample_ids,
            accepted_model_id="accepted-2",
            candidate_model_id="candidate-3",
            config_hash="config-hash",
            code_version="unit-test",
            recent_evaluation={"verdict": "inconclusive"},
            extra_state={
                "learner": learner.state_dict(),
                "token_bucket": bucket.state_dict(),
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(checkpoint_path, checkpoint)

            expected_metrics = learner.train_steps(dataset, steps=1, token_bucket=bucket)
            expected_state = copy.deepcopy(model.state_dict())
            expected_ids = list(learner.last_sample_ids)
            expected_lr = scheduler.get_last_lr()

            resumed_model, resumed_optimizer, resumed_scheduler, resumed_learner = _make_training_stack(
                initial_state
            )
            loaded = load_checkpoint(checkpoint_path)
            loaded.restore(
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                scaler=resumed_learner.scaler,
                expected_config_hash="config-hash",
            )
            resumed_learner.load_state_dict(loaded.extra_state["learner"])
            resumed_bucket = TrainTokenBucket.from_state_dict(loaded.extra_state["token_bucket"])
            actual_metrics = resumed_learner.train_steps(
                dataset, steps=1, token_bucket=resumed_bucket
            )

            self.assertEqual(resumed_learner.last_sample_ids, expected_ids)
            self.assertEqual(resumed_scheduler.get_last_lr(), expected_lr)
            self.assertEqual(actual_metrics.steps, expected_metrics.steps)
            self.assertAlmostEqual(actual_metrics.total_loss, expected_metrics.total_loss, places=7)
            for name, expected in expected_state.items():
                torch.testing.assert_close(resumed_model.state_dict()[name], expected, rtol=0.0, atol=0.0)

    def test_amp_skipped_step_preserves_budget_cursor_and_sample_order(self):
        torch.manual_seed(13)
        model, optimizer, scheduler, learner = _make_training_stack()
        dataset = OnlineD4Dataset(_make_replay(12), augmentation_seed=444)
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(12)
        initial_model = copy.deepcopy(model.state_dict())
        initial_bucket = bucket.state_dict()
        initial_lr = scheduler.get_last_lr()
        expected_keys = next(
            iter(
                DeterministicKeyBatchSampler(
                    dataset_size=len(dataset),
                    sample_seed=555,
                    start_cursor=0,
                    batch_sizes=(4,),
                )
            )
        )
        expected_ids = [dataset[key]["sample_id"] for key in expected_keys]

        learner.amp_enabled = True
        learner.scaler = _OneShotSkippingScaler()
        skipped = learner.train_steps(dataset, steps=2, token_bucket=bucket)

        self.assertEqual(skipped.steps, 0)
        self.assertEqual(skipped.positions, 0)
        self.assertEqual(learner.global_step, 0)
        self.assertEqual(learner.sample_cursor, 0)
        self.assertEqual(learner.last_sample_ids, [])
        self.assertEqual(bucket.state_dict(), initial_bucket)
        self.assertEqual(scheduler.get_last_lr(), initial_lr)
        for name, expected in initial_model.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0.0, atol=0.0)

        completed = learner.train_steps(dataset, steps=1, token_bucket=bucket)

        self.assertEqual(completed.steps, 1)
        self.assertEqual(completed.positions, 4)
        self.assertEqual(learner.global_step, 1)
        self.assertEqual(learner.sample_cursor, 4)
        self.assertEqual(learner.last_sample_ids, expected_ids)
        self.assertEqual(bucket.total_positions_consumed, 4)
        self.assertNotEqual(scheduler.get_last_lr(), initial_lr)
        self.assertTrue(
            any(
                not torch.equal(model.state_dict()[name], expected)
                for name, expected in initial_model.items()
            )
        )

    def test_training_uses_dataloader_with_deterministic_cursor_keys(self):
        sampler = DeterministicKeyBatchSampler(
            dataset_size=12,
            sample_seed=7,
            start_cursor=20,
            batch_sizes=(4, 3),
        )
        first_pass = list(sampler)
        self.assertEqual(first_pass, list(sampler))
        self.assertEqual([cursor for batch in first_pass for _, cursor in batch], list(range(20, 27)))

        model, optimizer, _, learner = _make_training_stack()
        dataset = OnlineD4Dataset(_make_replay(12), augmentation_seed=8)
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(12)
        real_dataloader = torch.utils.data.DataLoader
        with mock.patch(
            "training.v3.learner.torch.utils.data.DataLoader",
            wraps=real_dataloader,
        ) as dataloader:
            metrics = learner.train_steps(dataset, steps=1, token_bucket=bucket)
        self.assertEqual(metrics.steps, 1)
        dataloader.assert_called_once()
        self.assertEqual(dataloader.call_args.kwargs["num_workers"], 0)
        self.assertNotIn("pin_memory", dataloader.call_args.kwargs)
        self.assertNotIn("prefetch_factor", dataloader.call_args.kwargs)

    def test_position_limit_stops_at_exact_absolute_budget(self):
        model, optimizer, _, learner = _make_training_stack()
        dataset = OnlineD4Dataset(_make_replay(12), augmentation_seed=8)
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(12)
        metrics = learner.train_steps(
            dataset,
            steps=3,
            token_bucket=bucket,
            position_limit=5,
        )
        self.assertEqual(metrics.positions, 5)
        self.assertEqual(metrics.steps, 2)
        self.assertEqual(bucket.total_positions_consumed, 5)

    def test_position_lr_schedule_crossing_is_resume_equivalent(self):
        torch.manual_seed(91)
        initial = _TinyWDLNet().state_dict()

        def make_stack():
            model = _TinyWDLNet()
            model.load_state_dict(initial)
            optimizer = build_adamw(model, learning_rate=1e-3, weight_decay=1e-4)
            learner = V3Learner(
                model,
                optimizer,
                device="cpu",
                batch_size=4,
                grad_clip_norm=1.0,
                sample_seed=777,
                learning_rate_schedule=((0, 1e-3), (4, 5e-4)),
            )
            return model, optimizer, learner

        dataset = OnlineD4Dataset(_make_replay(12), augmentation_seed=333)
        model, optimizer, learner = make_stack()
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(12)
        learner.train_steps(dataset, steps=1, token_bucket=bucket)
        checkpoint = CheckpointV1.capture(
            model=model,
            optimizer=optimizer,
            scaler=learner.scaler,
            global_step=learner.global_step,
            generation=0,
            replay_cursor={},
            sample_ids=learner.last_sample_ids,
            accepted_model_id=None,
            candidate_model_id="candidate",
            config_hash="lr-config",
            code_version="test",
            recent_evaluation=None,
            extra_state={"learner": learner.state_dict(), "bucket": bucket.state_dict()},
        )
        learner.train_steps(dataset, steps=1, token_bucket=bucket)
        expected = copy.deepcopy(model.state_dict())
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-4)

        resumed_model, resumed_optimizer, resumed_learner = make_stack()
        checkpoint.restore(
            model=resumed_model,
            optimizer=resumed_optimizer,
            scaler=resumed_learner.scaler,
            expected_config_hash="lr-config",
        )
        resumed_learner.load_state_dict(checkpoint.extra_state["learner"])
        resumed_bucket = TrainTokenBucket.from_state_dict(checkpoint.extra_state["bucket"])
        resumed_learner.train_steps(dataset, steps=1, token_bucket=resumed_bucket)
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], 5e-4)
        for name, tensor in expected.items():
            torch.testing.assert_close(
                resumed_model.state_dict()[name], tensor, rtol=0.0, atol=0.0
            )

    def test_fast_only_batch_is_value_only_with_differentiable_zero_policy_loss(self):
        model, optimizer, _, learner = _make_training_stack()
        dataset = OnlineD4Dataset(
            _make_replay(8, search_kind=SEARCH_FAST), augmentation_seed=8
        )
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(8)
        metrics = learner.train_steps(dataset, steps=1, token_bucket=bucket)
        self.assertEqual(metrics.steps, 1)
        self.assertEqual(metrics.positions, 4)
        self.assertEqual(metrics.policy_positions, 0)
        self.assertEqual(metrics.value_positions, 4)
        self.assertEqual(metrics.policy_loss, 0.0)
        self.assertTrue(np.isfinite(metrics.wdl_loss))

    def test_worker_dataloader_preserves_key_order(self):
        dataset = OnlineD4Dataset(_make_replay(12), augmentation_seed=8)
        expected_keys = next(
            iter(
                DeterministicKeyBatchSampler(
                    dataset_size=12,
                    sample_seed=555,
                    start_cursor=0,
                    batch_sizes=(4,),
                )
            )
        )
        expected_ids = [dataset[key]["sample_id"] for key in expected_keys]
        model, optimizer, _, learner = _make_training_stack(num_workers=1)
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(12)
        metrics = learner.train_steps(dataset, steps=1, token_bucket=bucket)
        self.assertEqual(metrics.steps, 1)
        self.assertEqual(learner.last_sample_ids, expected_ids)

    def test_atomic_save_failure_preserves_existing_checkpoint(self):
        model, optimizer, _, _ = _make_training_stack()
        checkpoint = CheckpointV1.capture(
            model=model,
            optimizer=optimizer,
            global_step=0,
            generation=0,
            replay_cursor={},
            sample_ids=[],
            accepted_model_id=None,
            candidate_model_id="candidate-0",
            config_hash="hash",
            code_version="test",
            recent_evaluation=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "checkpoint.pt"
            target.write_bytes(b"known-good")
            with mock.patch("training.v3.checkpoint.torch.save", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    save_checkpoint(target, checkpoint)
            self.assertEqual(target.read_bytes(), b"known-good")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_capture_is_detached_from_later_training_and_cursor_mutation(self):
        model, optimizer, _, learner = _make_training_stack()
        dataset = OnlineD4Dataset(_make_replay(8), augmentation_seed=9)
        bucket = TrainTokenBucket(tokens_per_position=4.0)
        bucket.add(8)
        learner.train_steps(dataset, steps=1, token_bucket=bucket)
        replay_cursor = {
            "shards": [{"path": "replay/raw/a.npz", "checksum_sha256": "a" * 64}],
            "raw_positions": 8,
        }
        extra_state = {"nested": {"values": [1, 2]}}
        checkpoint = CheckpointV1.capture(
            model=model,
            optimizer=optimizer,
            scaler=learner.scaler,
            global_step=learner.global_step,
            generation=0,
            replay_cursor=replay_cursor,
            sample_ids=learner.last_sample_ids,
            accepted_model_id=None,
            candidate_model_id="candidate-0",
            config_hash="hash",
            code_version="test",
            recent_evaluation=None,
            extra_state=extra_state,
        )
        captured_model = copy.deepcopy(checkpoint.model_state)
        captured_optimizer = copy.deepcopy(checkpoint.optimizer_state)

        replay_cursor["shards"][0]["path"] = "changed.npz"
        extra_state["nested"]["values"].append(3)
        learner.train_steps(dataset, steps=1, token_bucket=bucket)

        self.assertEqual(checkpoint.replay_cursor["shards"][0]["path"], "replay/raw/a.npz")
        self.assertEqual(checkpoint.extra_state["nested"]["values"], [1, 2])
        for name, tensor in captured_model.items():
            torch.testing.assert_close(checkpoint.model_state[name], tensor, rtol=0.0, atol=0.0)
        self.assertEqual(
            int(next(iter(checkpoint.optimizer_state["state"].values()))["step"]),
            int(next(iter(captured_optimizer["state"].values()))["step"]),
        )


def _paired_scores(pair_scores):
    results = []
    for opening_id, (first, second) in enumerate(pair_scores):
        results.extend(
            [
                GateGameResult(str(opening_id), 1000 + opening_id, True, first),
                GateGameResult(str(opening_id), 1000 + opening_id, False, second),
            ]
        )
    return results


class PairedGateTests(unittest.TestCase):
    def test_accept_reject_and_inconclusive(self):
        accepted = evaluate_gate(
            _paired_scores([(1.0, 1.0)] * 8), bootstrap_samples=500, bootstrap_seed=1
        )
        self.assertEqual(accepted.verdict, "accept")
        self.assertEqual(accepted.summary.candidate_as_first.wins, 8)
        self.assertEqual(accepted.summary.candidate_as_second.wins, 8)

        rejected = evaluate_gate(
            _paired_scores([(0.0, 0.0)] * 8), bootstrap_samples=500, bootstrap_seed=1
        )
        self.assertEqual(rejected.verdict, "reject")

        inconclusive = evaluate_gate(
            _paired_scores([(1.0, 0.0), (0.0, 1.0)] * 4),
            bootstrap_samples=500,
            bootstrap_seed=1,
        )
        self.assertEqual(inconclusive.verdict, "inconclusive")
        self.assertEqual(inconclusive.summary.overall.point_score, 0.5)

    def test_pairing_requires_same_seed_and_exact_color_swap(self):
        bad_seed = [
            GateGameResult("opening", 1, True, 1.0),
            GateGameResult("opening", 2, False, 0.0),
        ]
        with self.assertRaisesRegex(ValueError, "same seed"):
            summarize_paired_results(bad_seed, bootstrap_samples=10)
        missing_swap = [
            GateGameResult("opening", 1, True, 1.0),
            GateGameResult("opening", 1, True, 0.0),
        ]
        with self.assertRaisesRegex(ValueError, "swap"):
            summarize_paired_results(missing_swap, bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
