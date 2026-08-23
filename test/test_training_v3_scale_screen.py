from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import torch

from training.v3.config import config_hash, load_config
from training.v3.hardware_plan import plan_hardware
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

    def test_dual_3080ti_b4_preset_splits_roles_without_changing_lineage(self) -> None:
        configs = ROOT / "training" / "v3" / "configs"
        baseline = load_config(configs / "stage1_scale_screen_b4c64.json")
        dual = load_config(configs / "stage1_scale_screen_b4c64_2x3080ti.json")
        self.assertEqual(
            config_hash(baseline),
            config_hash(replace(dual, run=baseline.run)),
        )
        self.assertEqual(dual.runtime.device, "cuda:0")
        self.assertEqual(dual.runtime.selfplay_devices, ("cuda:1",))
        plan = plan_hardware(
            (dual.runtime.device, *dual.runtime.selfplay_devices),
            learner_device=dual.runtime.device,
            actors=dual.runtime.actor_processes,
            mcts_lanes=dual.runtime.mcts_lanes_per_actor,
            inference_batch_limit=dual.runtime.inference_batch_size,
            cpu_cores=40,
            cuda_inventory_count=2,
        )
        self.assertEqual(plan.mode, "multi_gpu_role_split")
        self.assertEqual(plan.learner_device, "cuda:0")
        self.assertEqual(plan.selfplay_devices, ("cuda:1",))
        self.assertTrue(plan.stages_may_overlap)


if __name__ == "__main__":
    unittest.main()
