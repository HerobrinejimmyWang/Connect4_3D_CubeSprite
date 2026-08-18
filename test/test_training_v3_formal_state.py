from __future__ import annotations

import unittest
from dataclasses import replace

from training.v3.config import GateConfig
from training.v3.formal_state import FormalLoopState, PendingCandidateState


def _pending(*, pairs: int = 2, incumbent: str = "random") -> PendingCandidateState:
    return PendingCandidateState(
        candidate_model_id="candidate-1",
        candidate_path="candidates/candidate-1.pt",
        incumbent_model_id=incumbent,
        gate_path="metrics/gate-candidate-1.json",
        opening_manifest="manifests/gate-openings.json",
        pairs_evaluated=pairs,
        max_pairs=6,
    )


class FormalLoopStateTests(unittest.TestCase):
    def test_candidate_cadence_uses_consumed_positions_and_blocks_pending(self) -> None:
        gate = GateConfig(
            bootstrap_candidate_train_positions=8,
            candidate_train_positions=16,
            initial_opening_pairs=2,
            pair_increment=2,
            max_opening_pairs=6,
        )
        state = FormalLoopState(train_positions_consumed=7)
        self.assertFalse(state.candidate_due(gate))
        state = replace(state, train_positions_consumed=8)
        self.assertTrue(state.candidate_due(gate))
        state = state.emit_candidate(_pending())
        self.assertFalse(state.candidate_due(gate))
        state = state.resolve_pending_candidate(accepted=True)
        self.assertEqual(state.accepted_model_id, "candidate-1")
        state = replace(state, train_positions_consumed=23)
        self.assertFalse(state.candidate_due(gate))
        state = replace(state, train_positions_consumed=24)
        self.assertTrue(state.candidate_due(gate))

    def test_generation_and_inconclusive_gate_progress_round_trip(self) -> None:
        state = FormalLoopState().finish_generation(
            next_game_id=4,
            replay_positions=97,
            train_positions_consumed=16,
        )
        state = state.emit_candidate(_pending())
        state = state.extend_pending_gate(4)
        restored = FormalLoopState.from_dict(state.to_dict())
        self.assertEqual(restored, state)
        self.assertEqual(restored.next_generation, 1)
        self.assertEqual(restored.pending_candidate.pairs_evaluated, 4)

    def test_pending_candidate_paths_and_incumbent_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            replace(_pending(), candidate_path="../outside.pt")
        with self.assertRaisesRegex(ValueError, "incumbent"):
            FormalLoopState(accepted_model_id="accepted-a", pending_candidate=_pending())


if __name__ == "__main__":
    unittest.main()
