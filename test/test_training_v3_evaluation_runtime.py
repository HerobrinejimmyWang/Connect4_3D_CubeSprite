from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from training.v3.evaluation import build_openings, play_paired_openings
from training.v3.evaluation_runtime import BatchingPredictor, play_paired_openings_parallel
from training.v3.model import ROLE_FEATURE_COUNT, RULE_FEATURE_COUNT
from training.v3.search import RandomPredictor


class _RecordingPredictor(RandomPredictor):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.batch_sizes: list[int] = []

    def predict_batch(self, canonical_boards, *, role_to_play, rule_features):
        with self._lock:
            self.batch_sizes.append(len(canonical_boards))
        return super().predict_batch(
            canonical_boards,
            role_to_play=role_to_play,
            rule_features=rule_features,
        )


class EvaluationRuntimeTests(unittest.TestCase):
    def test_batching_service_combines_independent_requests_with_a_hard_limit(self) -> None:
        predictor = _RecordingPredictor()
        service = BatchingPredictor(
            predictor,
            service_id="test",
            batch_limit=4,
            batch_timeout_s=0.05,
        )
        board = np.zeros((6, 5, 5), dtype=np.int8)
        role = np.zeros((ROLE_FEATURE_COUNT,), dtype=np.float32)
        role[0] = 1.0
        rules = np.zeros((RULE_FEATURE_COUNT,), dtype=np.float32)
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        service.predict,
                        board,
                        role_to_play=role,
                        rule_features=rules,
                    )
                    for _ in range(4)
                ]
                outputs = [future.result() for future in futures]
        finally:
            metrics = service.close()

        self.assertEqual(len(outputs), 4)
        self.assertEqual(metrics.requests, 4)
        self.assertEqual(metrics.positions, 4)
        self.assertEqual(metrics.max_batch, 4)
        self.assertTrue(all(size <= 4 for size in predictor.batch_sizes))

    def test_parallel_games_preserve_fixed_opening_result_order_and_search_semantics(self) -> None:
        openings = build_openings(2, run_seed=1701, prefix_lengths=(0, 2))
        serial = play_paired_openings(
            openings,
            candidate_predictor=RandomPredictor(),
            incumbent_predictor=RandomPredictor(),
            search_sims=2,
            cpuct=1.5,
        )
        parallel = play_paired_openings_parallel(
            openings,
            candidate_predictor=RandomPredictor(),
            incumbent_predictor=RandomPredictor(),
            search_sims=2,
            cpuct=1.5,
            parallel_games=4,
            inference_batch_size=4,
            inference_batch_timeout_s=0.01,
        )

        self.assertEqual(parallel.games, serial)
        self.assertEqual(parallel.metrics.parallel_games, 4)
        self.assertEqual(parallel.metrics.games, 4)
        self.assertEqual(len(parallel.metrics.inference_services), 2)
        self.assertTrue(
            all(service.max_batch <= service.batch_limit for service in parallel.metrics.inference_services)
        )


if __name__ == "__main__":
    unittest.main()
