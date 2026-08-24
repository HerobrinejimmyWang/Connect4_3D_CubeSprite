from __future__ import annotations

import unittest
from dataclasses import replace

from training.v3.config import DynamicExplorationConfig, SelfPlayConfig
from training.v3.dynamic_exploration import prepare_dynamic_exploration
from training.v3.formal_state import FormalLoopState
from training.v3.stability import GenerationStabilityMetrics


def _row(generation: int, *, length: float = 24.0, short: float = 0.05, entropy: float = 2.0):
    return GenerationStabilityMetrics(
        generation=generation,
        games=160,
        mean_game_length=length,
        game_length_variance=80.0,
        short_game_rate=short,
        mean_policy_entropy=entropy,
        value_loss=0.5,
    )


class DynamicExplorationTests(unittest.TestCase):
    def _selfplay(self) -> SelfPlayConfig:
        base = SelfPlayConfig()
        return replace(
            base,
            exploration_phases=(
                replace(base.exploration_phases[0], dirichlet_alpha=0.24, dirichlet_epsilon=0.06),
                replace(base.exploration_phases[1], start_ply=28, temperature=0.5),
            ),
            dynamic_exploration=DynamicExplorationConfig(enabled=True),
        )

    def test_stable_committed_window_advances_one_stage(self) -> None:
        state = FormalLoopState(next_generation=16)
        effective, advanced, decision = prepare_dynamic_exploration(
            self._selfplay(), state, [_row(generation) for generation in range(8, 16)]
        )
        self.assertTrue(decision["transitioned"])
        self.assertEqual(decision["reason"], "stable_window")
        self.assertEqual(advanced.exploration_stage_index, 1)
        self.assertEqual(advanced.exploration_stage_started_generation, 16)
        self.assertEqual(effective.exploration_phases[1].start_ply, 20)

    def test_unstable_window_retains_current_stage(self) -> None:
        rows = [_row(generation) for generation in range(8, 16)]
        rows[-1] = _row(15, length=16.0, short=0.20, entropy=2.5)
        effective, retained, decision = prepare_dynamic_exploration(
            self._selfplay(), FormalLoopState(next_generation=16), rows
        )
        self.assertFalse(decision["transitioned"])
        self.assertEqual(decision["reason"], "distribution_not_stable")
        self.assertEqual(retained.exploration_stage_index, 0)
        self.assertEqual(effective.exploration_phases[1].start_ply, 12)

    def test_legacy_formal_state_schema_defaults_to_first_stage(self) -> None:
        raw = FormalLoopState().to_dict()
        raw.pop("exploration_stage_index")
        raw.pop("exploration_stage_started_generation")
        restored = FormalLoopState.from_dict(raw)
        self.assertEqual(restored.exploration_stage_index, 0)


if __name__ == "__main__":
    unittest.main()
