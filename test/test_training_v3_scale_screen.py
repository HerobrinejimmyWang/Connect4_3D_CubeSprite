from __future__ import annotations

import unittest
from pathlib import Path

import torch

from training.v3.config import load_config
from training.v3.model import build_model, classic_rule_features
from training.v3.scale_screen import load_scale_screen


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "training" / "v3" / "configs" / "stage1_scale_screen_v1.json"


class ScaleScreenContractTests(unittest.TestCase):
    def test_plan_has_staged_seed_budgets_and_frozen_search(self) -> None:
        plan = load_scale_screen(SPEC, root=ROOT)
        self.assertFalse(plan["formal_run_enabled"])
        self.assertEqual(plan["run_count"], 5)
        self.assertEqual([row["scale_id"] for row in plan["levels"]], ["b4c64", "b6c128", "b8c192"])
        self.assertEqual(plan["levels"][0]["seed_budgets"][0]["max_train_positions"], 60000)
        self.assertEqual(plan["frozen_training_contract"]["full_search_sims"], 128)
        self.assertEqual(plan["policy_target_quality"]["primary_search_sims"], 256)
        self.assertEqual(plan["policy_target_quality"]["reference_search_sims"], 512)

    def test_b8_config_builds_all_heads(self) -> None:
        config = load_config(ROOT / "training" / "v3" / "configs" / "stage1_scale_screen_b8c192.json")
        model = build_model(config.model)
        output = model(
            torch.zeros((1, 6, 5, 5), dtype=torch.float32),
            role_to_play=torch.tensor([[1.0, 0.0]]),
            rule_features=classic_rule_features(1),
        )
        self.assertEqual(tuple(output.policy_logits.shape), (1, 25))
        self.assertEqual(tuple(output.wdl_logits.shape), (1, 3))
        self.assertEqual(tuple(output.opponent_reply_logits.shape), (1, 25))
        self.assertEqual(tuple(output.future_occupancy_logits.shape), (1, 3, 6, 5, 5))
        self.assertEqual(tuple(output.moves_left_logits.shape), (1, 301))


if __name__ == "__main__":
    unittest.main()
