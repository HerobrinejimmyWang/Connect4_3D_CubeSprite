from __future__ import annotations

import unittest

from training.v3.hardware_plan import plan_hardware


class HardwarePlanTests(unittest.TestCase):
    def test_single_gpu_is_staged_bounded_and_no_ddp(self) -> None:
        plan = plan_hardware(
            ["cuda:0"],
            actors=4,
            mcts_lanes=2,
            inference_batch_limit=32,
            cpu_cores=16,
            cuda_inventory_count=1,
        )

        self.assertEqual(plan.mode, "single_gpu_staged")
        self.assertEqual(plan.learner_device, "cuda:0")
        self.assertEqual(plan.selfplay_devices, ("cuda:0",))
        self.assertFalse(plan.stages_may_overlap)
        self.assertFalse(plan.ddp_enabled)
        self.assertEqual(plan.game_queue_capacity, 8)
        self.assertEqual(plan.completed_game_queue_capacity, 8)
        self.assertEqual(plan.checkpoint_queue_capacity, 1)
        self.assertEqual(plan.effective_inference_batch_limit, 8)
        self.assertEqual(plan.inference_services[0].actor_ids, (0, 1, 2, 3))
        self.assertEqual(plan.inference_services[0].request_queue_capacity, 8)
        self.assertIn(
            "inference_batch_concurrency_limited",
            {warning.code for warning in plan.warnings},
        )

    def test_multi_gpu_sorts_inventory_and_splits_roles(self) -> None:
        plan = plan_hardware(
            ["cuda:2", "cuda:0", "cuda:1"],
            actors=8,
            mcts_lanes=2,
            inference_batch_limit=8,
            cpu_cores=32,
            cuda_inventory_count=3,
        )

        self.assertEqual(plan.mode, "multi_gpu_role_split")
        self.assertEqual(plan.cuda_devices, ("cuda:0", "cuda:1", "cuda:2"))
        self.assertEqual(plan.learner_device, "cuda:0")
        self.assertEqual(plan.selfplay_devices, ("cuda:1", "cuda:2"))
        self.assertTrue(plan.stages_may_overlap)
        self.assertFalse(plan.ddp_enabled)
        self.assertEqual(plan.inference_services[0].actor_ids, (0, 2, 4, 6))
        self.assertEqual(plan.inference_services[1].actor_ids, (1, 3, 5, 7))
        self.assertTrue(all(service.effective_batch_limit == 8 for service in plan.inference_services))
        self.assertTrue(all(service.request_queue_capacity == 8 for service in plan.inference_services))
        self.assertEqual(plan.warnings, ())

    def test_explicit_selfplay_can_time_slice_the_learner_gpu(self) -> None:
        plan = plan_hardware(
            ["cuda:0", "cuda:1"],
            learner_device="cuda:0",
            selfplay_devices=["cuda:0", "cuda:1"],
            actors=40,
            mcts_lanes=4,
            inference_batch_limit=32,
            cpu_cores=40,
            cuda_inventory_count=2,
        )

        self.assertEqual(plan.mode, "multi_gpu_staged")
        self.assertEqual(plan.selfplay_devices, ("cuda:0", "cuda:1"))
        self.assertFalse(plan.stages_may_overlap)
        self.assertEqual([service.actor_count for service in plan.inference_services], [20, 20])

    def test_extra_selfplay_devices_are_reported_not_started(self) -> None:
        plan = plan_hardware(
            ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            actors=1,
            mcts_lanes=1,
            inference_batch_limit=1,
            cpu_cores=4,
            cuda_inventory_count=4,
        )

        self.assertEqual(plan.selfplay_devices, ("cuda:1",))
        self.assertEqual(plan.unused_cuda_devices, ("cuda:2", "cuda:3"))
        self.assertIn("idle_selfplay_devices", {warning.code for warning in plan.warnings})

    def test_explicit_learner_device_is_not_reordered(self) -> None:
        plan = plan_hardware(
            ["cuda:0", "cuda:1"],
            learner_device="cuda:1",
            actors=2,
            mcts_lanes=1,
            inference_batch_limit=2,
            cpu_cores=8,
            cuda_inventory_count=2,
        )
        self.assertEqual(plan.learner_device, "cuda:1")
        self.assertEqual(plan.selfplay_devices, ("cuda:0",))

    def test_cpu_oversubscription_tracks_actor_processes_not_vectorized_lanes(self) -> None:
        plan = plan_hardware(
            ["cuda:0", "cuda:1"],
            actors=4,
            mcts_lanes=3,
            inference_batch_limit=12,
            cpu_cores=4,
            cuda_inventory_count=2,
        )

        warning_codes = {warning.code for warning in plan.warnings}
        self.assertIn("cpu_actor_oversubscription", warning_codes)
        self.assertNotIn("cpu_lane_oversubscription", warning_codes)

    def test_invalid_cuda_inventory_is_rejected(self) -> None:
        cases = (
            (["cuda:0", "cuda:0"], 2, "duplicate CUDA device"),
            (["cuda"], 1, "invalid CUDA device"),
            (["cuda:01"], 2, "invalid CUDA device"),
            (["cuda:2"], 2, "outside inventory count"),
        )
        for devices, inventory_count, message in cases:
            with self.subTest(devices=devices):
                with self.assertRaisesRegex(ValueError, message):
                    plan_hardware(
                        devices,
                        actors=1,
                        mcts_lanes=1,
                        inference_batch_limit=1,
                        cpu_cores=2,
                        cuda_inventory_count=inventory_count,
                    )

        with self.assertRaisesRegex(ValueError, "selfplay_devices must be included"):
            plan_hardware(
                ["cuda:0", "cuda:1"],
                learner_device="cuda:0",
                selfplay_devices=["cuda:2"],
                actors=1,
                mcts_lanes=1,
                inference_batch_limit=1,
                cpu_cores=2,
                cuda_inventory_count=3,
            )

    def test_invalid_resource_counts_are_rejected(self) -> None:
        base = {
            "cuda_devices": ["cuda:0"],
            "actors": 1,
            "mcts_lanes": 1,
            "inference_batch_limit": 1,
            "cpu_cores": 2,
            "cuda_inventory_count": 1,
        }
        for field in ("actors", "mcts_lanes", "inference_batch_limit", "cpu_cores"):
            values = dict(base)
            values[field] = 0
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    plan_hardware(**values)


if __name__ == "__main__":
    unittest.main()
