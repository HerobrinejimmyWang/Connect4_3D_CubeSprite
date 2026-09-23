from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from connect4_core.rules import BAL5_R2_RULE_REGISTRY
from training.v3.evaluation import build_openings, load_opening_manifest, write_opening_manifest
from training.v3.gate import GateGameResult
from training.v3.config import GateConfig
from training.v3.multirule_gate import (
    BAL5_R2_RULE_IDS,
    run_multirule_sequential_gate,
    summarize_multirule_gate,
)


def evidence(score: float, *, pairs: int = 10, shared_ids: bool = False):
    result = {}
    for index, rule_id in enumerate(BAL5_R2_RULE_IDS):
        prefix = "shared" if shared_ids else rule_id
        result[rule_id] = tuple(
            GateGameResult(f"{prefix}-{pair}", index * 1000 + pair, role, score)
            for pair in range(pairs)
            for role in (True, False)
        )
    return result


class MultiRuleGateTests(unittest.TestCase):
    def test_sequential_gate_extends_equal_rule_pairs_and_reuses_incumbent_peak(self) -> None:
        openings = {
            rule: tuple(
                SimpleNamespace(opening_id=f"{rule}-opening-{pair}", seed=index * 100 + pair)
                for pair in range(4)
            )
            for index, rule in enumerate(BAL5_R2_RULE_IDS)
        }
        calls = []

        def evaluate(rule_id, rows, opponent_model_id):
            calls.append((rule_id, opponent_model_id, len(rows)))
            score = 0.5 if rows[0].opening_id.endswith(("-0", "-1")) else 1.0
            return tuple(
                GateGameResult(row.opening_id, row.seed, role, score)
                for row in rows
                for role in (True, False)
            )

        current, peaks, decision, looks = run_multirule_sequential_gate(
            openings,
            gate=GateConfig(
                initial_opening_pairs=2,
                pair_increment=2,
                max_opening_pairs=4,
                bootstrap_samples=1000,
            ),
            run_seed=7,
            incumbent_model_id="accepted-root",
            peak_model_ids={rule: "accepted-root" for rule in BAL5_R2_RULE_IDS},
            evaluate_pairs=evaluate,
        )
        self.assertEqual(decision.verdict, "accept")
        self.assertEqual([look["pairs_per_rule"] for look in looks], [2, 4])
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(len(current[rule]) == len(peaks[rule]) == 8 for rule in BAL5_R2_RULE_IDS))

    def test_five_rule_opening_manifests_have_distinct_ids_and_registry(self) -> None:
        all_ids = set()
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, rule_id in enumerate(BAL5_R2_RULE_IDS):
                rows = build_openings(
                    2,
                    run_seed=101 + index * 10000,
                    rule_id=rule_id,
                    registry=BAL5_R2_RULE_REGISTRY,
                    opening_id_prefix=f"{rule_id}-opening",
                )
                path = Path(temp_dir) / f"{rule_id}.json"
                write_opening_manifest(path, rows, registry=BAL5_R2_RULE_REGISTRY)
                self.assertEqual(
                    load_opening_manifest(path, registry=BAL5_R2_RULE_REGISTRY), rows
                )
                self.assertFalse(all_ids.intersection(row.opening_id for row in rows))
                all_ids.update(row.opening_id for row in rows)

    def test_equal_rule_macro_accepts_and_reports_each_rule(self) -> None:
        rows = evidence(1.0)
        summary = summarize_multirule_gate(rows, rows, bootstrap_samples=100)
        self.assertEqual(summary.verdict, "accept")
        self.assertEqual(summary.macro_point_score, 1.0)
        self.assertEqual(summary.opening_pairs_per_rule, 10)
        self.assertEqual(tuple(summary.per_rule), BAL5_R2_RULE_IDS)
        self.assertEqual(summary.to_dict()["schema"], "connect4-v3-bal5-r2-multirule-gate-v1")

    def test_peak_regression_rejects_even_when_macro_improves(self) -> None:
        current = evidence(1.0)
        peaks = evidence(1.0)
        peaks[BAL5_R2_RULE_IDS[0]] = evidence(0.0)[BAL5_R2_RULE_IDS[0]]
        summary = summarize_multirule_gate(current, peaks, bootstrap_samples=100)
        self.assertEqual(summary.verdict, "reject")
        self.assertEqual(summary.hard_regressions, (BAL5_R2_RULE_IDS[0],))

    def test_rule_openings_must_be_isolated_and_equal(self) -> None:
        shared = evidence(1.0, shared_ids=True)
        with self.assertRaisesRegex(ValueError, "share opening IDs"):
            summarize_multirule_gate(shared, shared, bootstrap_samples=10)
        unequal = evidence(1.0)
        unequal[BAL5_R2_RULE_IDS[0]] = evidence(1.0, pairs=9)[BAL5_R2_RULE_IDS[0]]
        with self.assertRaisesRegex(ValueError, "same number"):
            summarize_multirule_gate(unequal, unequal, bootstrap_samples=10)

    def test_peak_evidence_must_match_current_opening_contract(self) -> None:
        current = evidence(1.0)
        peaks = evidence(1.0)
        first_rule = BAL5_R2_RULE_IDS[0]
        peaks[first_rule] = tuple(
            GateGameResult(row.opening_id, row.seed + 1, row.candidate_is_first, row.candidate_score)
            for row in peaks[first_rule]
        )
        with self.assertRaisesRegex(ValueError, "identical openings"):
            summarize_multirule_gate(current, peaks, bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
