from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from connect4_core.rules import BAL5_R2_RULE_REGISTRY
from training.v3.evaluation import build_openings, load_opening_manifest, write_opening_manifest
from training.v3.gate import GateGameResult
from training.v3.config import GateConfig, config_hash, load_config
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
    @staticmethod
    def _staged_gate(*, incumbent_score: float, peak_score: float, pairs: int = 2,
                     max_pairs: int = 4, same_peak: bool = False):
        openings = {
            rule: tuple(SimpleNamespace(opening_id=f"{rule}-{pair}", seed=index * 100 + pair)
                        for pair in range(max_pairs))
            for index, rule in enumerate(BAL5_R2_RULE_IDS)
        }
        calls = []

        def evaluate(rule_id, rows, opponent_model_id):
            calls.append((rule_id, opponent_model_id, tuple(row.opening_id for row in rows)))
            score = incumbent_score if opponent_model_id == "incumbent" else peak_score
            return [GateGameResult(row.opening_id, row.seed, role, score)
                    for row in rows for role in (True, False)]

        result = run_multirule_sequential_gate(
            openings,
            gate=GateConfig(initial_opening_pairs=pairs, pair_increment=2,
                            max_opening_pairs=max_pairs, bootstrap_samples=100,
                            multirule_evaluation_mode="incumbent_first"),
            run_seed=19,
            incumbent_model_id="incumbent",
            peak_model_ids={rule: "incumbent" if same_peak else "peak"
                            for rule in BAL5_R2_RULE_IDS},
            evaluate_pairs=evaluate,
        )
        return result, calls

    def test_staged_macro_reject_skips_peak_with_explicit_unknown(self) -> None:
        (current, peaks, decision, looks), calls = self._staged_gate(
            incumbent_score=0.0, peak_score=1.0
        )
        self.assertEqual(decision.verdict, "reject")
        self.assertEqual(decision.peak_evidence_status, "not_evaluated")
        self.assertIsNone(decision.hard_regressions)
        self.assertIsNone(decision.to_dict()["versus_peak"])
        self.assertEqual(decision.to_dict()["schema"], "connect4-v3-bal5-r2-multirule-gate-v2")
        self.assertEqual(len(calls), 5)
        self.assertEqual([look["pairs_per_rule"] for look in looks], [2])
        self.assertTrue(all(len(peaks[rule]) == 0 and len(current[rule]) == 4
                            for rule in BAL5_R2_RULE_IDS))

    def test_staged_exhaustion_skips_peak(self) -> None:
        (_, peaks, decision, looks), calls = self._staged_gate(
            incumbent_score=0.5, peak_score=1.0
        )
        self.assertEqual(decision.verdict, "reject")
        self.assertEqual(decision.peak_evidence_status, "not_evaluated")
        self.assertEqual([look["pairs_per_rule"] for look in looks], [2, 4])
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(not peaks[rule] for rule in BAL5_R2_RULE_IDS))

    def test_staged_checks_every_peak_before_acceptance(self) -> None:
        (current, peaks, decision, looks), calls = self._staged_gate(
            incumbent_score=1.0, peak_score=0.0
        )
        self.assertEqual(decision.verdict, "reject")
        self.assertEqual(decision.hard_regressions, BAL5_R2_RULE_IDS)
        self.assertEqual(decision.peak_evidence_status, "complete")
        self.assertEqual(len(calls), 10)
        self.assertEqual([look["pairs_per_rule"] for look in looks], [2])
        self.assertTrue(all(len(current[rule]) == len(peaks[rule]) == 4
                            for rule in BAL5_R2_RULE_IDS))

    def test_staged_accept_reuses_identical_peak(self) -> None:
        (_, peaks, decision, _), calls = self._staged_gate(
            incumbent_score=1.0, peak_score=0.0, same_peak=True
        )
        self.assertEqual(decision.verdict, "accept")
        self.assertEqual(decision.hard_regressions, ())
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(len(peaks[rule]) == 4 for rule in BAL5_R2_RULE_IDS))

    def test_staged_accept_completes_distinct_peak_evidence(self) -> None:
        (_, peaks, decision, _), calls = self._staged_gate(
            incumbent_score=1.0, peak_score=1.0
        )
        self.assertEqual(decision.verdict, "accept")
        self.assertEqual(decision.peak_evidence_status, "complete")
        self.assertEqual(set(decision.versus_peak), set(BAL5_R2_RULE_IDS))
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(len(peaks[rule]) == 4 for rule in BAL5_R2_RULE_IDS))

    def test_staged_restart_after_interrupted_peak_replays_same_openings(self) -> None:
        openings = {
            rule: tuple(SimpleNamespace(opening_id=f"{rule}-{pair}", seed=index * 100 + pair)
                        for pair in range(2))
            for index, rule in enumerate(BAL5_R2_RULE_IDS)
        }
        calls = []
        fail_once = [True]

        def evaluate(rule_id, rows, opponent_model_id):
            calls.append((rule_id, opponent_model_id, tuple(row.opening_id for row in rows)))
            if opponent_model_id == "peak" and fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("interrupted")
            return [GateGameResult(row.opening_id, row.seed, role, 1.0)
                    for row in rows for role in (True, False)]

        kwargs = dict(
            gate=GateConfig(initial_opening_pairs=2, pair_increment=2,
                            max_opening_pairs=2, bootstrap_samples=100,
                            multirule_evaluation_mode="incumbent_first"),
            run_seed=19, incumbent_model_id="incumbent",
            peak_model_ids={rule: "peak" for rule in BAL5_R2_RULE_IDS},
            evaluate_pairs=evaluate,
        )
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            run_multirule_sequential_gate(openings, **kwargs)
        first_attempt = tuple(calls)
        current, peaks, decision, _ = run_multirule_sequential_gate(openings, **kwargs)
        self.assertEqual(decision.verdict, "accept")
        self.assertEqual(first_attempt[:5], tuple(calls[len(first_attempt):len(first_attempt) + 5]))
        self.assertTrue(all(len(current[rule]) == len(peaks[rule]) == 4
                            for rule in BAL5_R2_RULE_IDS))

    def test_staged_mode_changes_semantic_hash_and_default_preserves_hash(self) -> None:
        from dataclasses import replace

        single_rule = load_config(Path(__file__).resolve().parents[1] /
                                  "training/v3/configs/smoke_cpu.json")
        config = replace(
            single_rule,
            selfplay=replace(
                single_rule.selfplay,
                multi_rule_ids=BAL5_R2_RULE_IDS,
                rule_registry_hash=BAL5_R2_RULE_REGISTRY.registry_hash,
            ),
        )
        self.assertEqual(config_hash(config), config_hash(replace(
            config, gate=replace(config.gate, multirule_evaluation_mode="complete")
        )))
        staged = replace(
            config, gate=replace(config.gate, multirule_evaluation_mode="incumbent_first")
        )
        self.assertNotEqual(config_hash(config), config_hash(staged))
        with self.assertRaisesRegex(ValueError, "requires five-rule"):
            replace(single_rule, gate=staged.gate)

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
