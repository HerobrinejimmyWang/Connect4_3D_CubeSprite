from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.run_stage2b_queue import (
    _completed_at_bound,
    _load_terminal_result,
    _stability_paused,
)


class Stage2BQueueTest(unittest.TestCase):
    def test_terminal_json_after_worker_cleanup_warning_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2b-queue-") as temporary:
            log_path = Path(temporary) / "run.log"
            log_path.write_text(
                "worker cleanup warning\n"
                '{"status":"stopped_at_safe_boundary",'
                '"stop_reason":"max_train_positions",'
                '"formal_loop_state":{"train_positions_consumed":1000000}}\n',
                encoding="utf-8",
            )
            result = _load_terminal_result(log_path)
            self.assertTrue(_completed_at_bound(result, 1_000_000))

    def test_partial_or_wrong_bound_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2b-queue-") as temporary:
            log_path = Path(temporary) / "run.log"
            log_path.write_text(
                '{"status":"stopped_at_safe_boundary",'
                '"stop_reason":"max_train_positions",'
                '"results":[{"train_positions_consumed":999999}]} trailing',
                encoding="utf-8",
            )
            self.assertIsNone(_load_terminal_result(log_path))

    def test_stability_pause_is_a_distinct_terminal_outcome(self) -> None:
        result = {
            "status": "stopped_at_safe_boundary",
            "stop_reason": "stability_pause",
            "formal_loop_state": {"train_positions_consumed": 431_524},
        }
        self.assertTrue(_stability_paused(result))
        self.assertFalse(_completed_at_bound(result, 1_000_000))


if __name__ == "__main__":
    unittest.main()
