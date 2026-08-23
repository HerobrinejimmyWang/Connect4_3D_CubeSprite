from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from training.v3.policy_target_quality import (
    compare_replay_policy_targets,
    compare_visit_targets,
    summarize_visit_targets,
)
from training.v3.replay import ReplayShard
from training.v3.stability import GenerationStabilityMetrics, assess_stability


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "test" / "fixtures" / "v3_historical_stability_trace_v1.json"


class HistoricalStabilityRegressionTests(unittest.TestCase):
    def test_historical_collapse_is_stopped_and_recovery_is_not(self) -> None:
        fixture = json.loads(TRACE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["lineage"], "legacy_evidence_only")
        for trace in fixture["traces"]:
            rows = tuple(GenerationStabilityMetrics.from_mapping(row) for row in trace["points"])
            result = assess_stability(rows)
            self.assertEqual(
                result.first_pause_generation,
                trace["expected_first_pause_generation"],
                trace["trace_id"],
            )
            self.assertEqual(result.action, trace["expected_terminal_action"], trace["trace_id"])

    def test_single_indicator_generation_is_watch_not_pause(self) -> None:
        row = GenerationStabilityMetrics(
            generation=1,
            games=240,
            mean_game_length=28.0,
            game_length_variance=100.0,
            short_game_rate=0.11,
            mean_policy_entropy=0.7,
            value_loss=0.5,
        )
        result = assess_stability((row,))
        self.assertEqual(result.action, "watch")
        self.assertIsNone(result.first_pause_generation)

    def test_stable_high_exploration_distribution_stays_watch(self) -> None:
        rows = (
            GenerationStabilityMetrics(
                generation=2,
                games=64,
                mean_game_length=12.859375,
                game_length_variance=20.464599609375,
                short_game_rate=0.5625,
                mean_policy_entropy=2.598018764701392,
                value_loss=0.6571766083759062,
            ),
            GenerationStabilityMetrics(
                generation=3,
                games=64,
                mean_game_length=13.21875,
                game_length_variance=17.3583984375,
                short_game_rate=0.453125,
                mean_policy_entropy=2.6309109264167496,
                value_loss=0.6455787241317984,
            ),
            GenerationStabilityMetrics(
                generation=4,
                games=64,
                mean_game_length=12.96875,
                game_length_variance=20.2802734375,
                short_game_rate=0.59375,
                mean_policy_entropy=2.529447452915516,
                value_loss=0.6309228902839753,
            ),
        )
        result = assess_stability(rows)
        self.assertEqual(result.action, "watch")
        self.assertEqual(result.consecutive_behavioral_alerts, 1)
        self.assertIsNone(result.first_pause_generation)

    def test_material_short_game_deterioration_still_pauses(self) -> None:
        rows = (
            GenerationStabilityMetrics(
                generation=10,
                games=240,
                mean_game_length=16.4,
                game_length_variance=80.0,
                short_game_rate=0.54,
                value_loss=0.46,
            ),
            GenerationStabilityMetrics(
                generation=11,
                games=240,
                mean_game_length=11.3,
                game_length_variance=39.0,
                short_game_rate=0.77,
                value_loss=0.29,
            ),
        )
        result = assess_stability(rows)
        self.assertEqual(result.action, "pause")
        self.assertEqual(result.first_pause_generation, 11)


class PolicyTargetQualityTests(unittest.TestCase):
    def test_visit_target_summary_checks_budget_and_shape(self) -> None:
        counts = np.zeros((2, 25), dtype=np.uint32)
        counts[0, :2] = (192, 64)
        counts[1, :4] = 64
        result = summarize_visit_targets(counts, expected_simulations=256)
        self.assertEqual(result["positions"], 2)
        self.assertEqual(result["exact_budget_fraction"], 1.0)
        self.assertEqual(result["visit_total"]["mean"], 256.0)
        self.assertGreater(result["effective_action_count_mean"], 1.0)

    def test_paired_comparison_reports_delta_without_auto_decision(self) -> None:
        candidate = np.zeros((2, 25), dtype=np.uint32)
        reference = np.zeros((2, 25), dtype=np.uint32)
        candidate[0, :2] = (200, 56)
        reference[0, :2] = (400, 112)
        candidate[1, :2] = (100, 156)
        reference[1, :2] = (300, 212)
        result = compare_visit_targets(candidate, reference)
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["positions"], 2)
        self.assertEqual(result["top1_agreement"], 0.5)
        self.assertGreater(result["mean_reference_top1_regret"], 0.0)

    def test_replay_comparison_rejects_different_positions(self) -> None:
        board = np.zeros((1, 6, 5, 5), dtype=np.int8)
        visits = np.zeros((1, 25), dtype=np.uint32)
        visits[0, 0] = 256
        common = dict(
            board=board,
            visit_counts=visits,
            policy_weight=np.ones(1, dtype=np.float32),
            wdl=np.zeros(1, dtype=np.uint8),
            game_id=np.zeros(1, dtype=np.uint64),
            turn_index=np.zeros(1, dtype=np.uint16),
            player_to_move=np.ones(1, dtype=np.int8),
            search_kind=np.ones(1, dtype=np.uint8),
            rule_code=np.zeros(1, dtype=np.uint16),
            turn_kind=np.zeros(1, dtype=np.uint8),
            placement_count=np.zeros(1, dtype=np.uint16),
            opponent_reply_column=np.full(1, -1, dtype=np.int8),
            opponent_reply_mask=np.zeros(1, dtype=np.uint8),
            terminal_board=board.copy(),
            remaining_turns=np.ones(1, dtype=np.uint16),
        )
        primary = ReplayShard(**common)
        changed = dict(common)
        changed["game_id"] = np.ones(1, dtype=np.uint64)
        reference = ReplayShard(**changed)
        with self.assertRaisesRegex(ValueError, "game_id"):
            compare_replay_policy_targets(primary, reference)


if __name__ == "__main__":
    unittest.main()
