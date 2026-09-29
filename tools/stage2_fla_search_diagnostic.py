"""Profile one fixed Classic state for paired FLA parent-1M/child-2M models.

Uses the desktop NumPy MCTS path, with read-only model snapshots. Tree metrics
describe search concentration and work; they are not playing-strength claims.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "desktop_app" / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from connect4_core import GameRules  # noqa: E402
from cubesprite_backend.search import NumpyMCTS, find_forced_tactical_action  # noqa: E402
from benchmark_stage2_cpu_latency import (  # noqa: E402
    ClassicContextPredictor, build_states, role_features_for_canonical,
)
from stage2_fla_selfplay_queue import atomic_json, sha256  # noqa: E402
from training.v3.config import V3Config, model_config_dict  # noqa: E402
from training.v3.model import LegacyPolicyValueAdapter, TorchPredictor, build_model  # noqa: E402
from training.v3.pipeline import lineage_config_hash  # noqa: E402


SOURCE = ROOT / "training/runs/stage2/fla"
PARENT = SOURCE / "latency_1m_vs_2m/parent_1m"
CHILD = SOURCE / "cpu_terminal_3m/idle10_group60"
NAMES = ("gravity_b8", "raw3d_to2d_b8", "raw3d_to2d_thin_b8c192",
         "raw3d_to2d_thin_b6c128")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entropy(probs: np.ndarray) -> float:
    positive = probs[probs > 0]
    return float(-np.sum(positive * np.log(positive)))


class TimedModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.forward_ns = 0
        self.forward_calls = 0

    def forward_search(self, *args, **kwargs):
        start = time.perf_counter_ns()
        result = self.model.forward_search(*args, **kwargs)
        self.forward_ns += time.perf_counter_ns() - start
        self.forward_calls += 1
        return result


class TimedPredictor:
    def __init__(self, predictor: LegacyPolicyValueAdapter) -> None:
        self.predictor = predictor
        self.predict_ns = 0
        self.predict_calls = 0

    def predict(self, board):
        start = time.perf_counter_ns()
        result = self.predictor.predict(board)
        self.predict_ns += time.perf_counter_ns() - start
        self.predict_calls += 1
        return result


def load(stage: str, name: str, parent: dict, child: dict):
    if stage == "parent_1m":
        source = parent["models"][name]
        config = PARENT / f"{name}.json"
        artifact = PARENT / f"{name}.pt"
        config_sha = source["config_sha256"]
        artifact_sha = source["snapshot_sha256"]
        expected_checkpoint = source["checkpoint_sha256"]
        expected_positions = 1_000_000
    else:
        source = child["models"][name]
        config = CHILD / "inputs" / f"{name}.json"
        artifact = CHILD / "inputs" / f"{name}.pt"
        config_sha = source["config_sha256"]
        artifact_sha = source["snapshot_sha256"]
        expected_checkpoint = source["terminal_sha256"]
        expected_positions = 2_000_000
    if sha256(config) != config_sha or sha256(artifact) != artifact_sha:
        raise ValueError(f"{name} {stage} source SHA-256 mismatch")
    raw = read(config)
    config_obj = V3Config.from_dict(raw)
    payload = torch.load(io.BytesIO(artifact.read_bytes()), map_location="cpu",
                         weights_only=True)
    metadata = payload.get("metadata", {})
    if (payload.get("format") != "connect4-v3-model"
            or payload.get("format_version") != 1
            or payload.get("model_config") != model_config_dict(config_obj.model)
            or metadata.get("source_kind") != "formal_v3_checkpoint"
            or metadata.get("evaluation_only") is not True
            or metadata.get("train_positions_consumed") != expected_positions
            or metadata.get("source_checkpoint_sha256") != expected_checkpoint
            or metadata.get("source_checkpoint_config_hash") !=
            lineage_config_hash(config_obj)):
        raise ValueError(f"{name} {stage} snapshot metadata/config mismatch")
    model = TimedModel(build_model(config_obj.model))
    model.model.load_state_dict(payload["model_state"], strict=True)
    adapter = LegacyPolicyValueAdapter(
        ClassicContextPredictor(TorchPredictor(model, device="cpu")))
    predictor = TimedPredictor(adapter)
    return model, predictor, {
        "architecture": raw["model"]["architecture"],
        "config_sha256": config_sha, "snapshot_sha256": artifact_sha,
        "checkpoint_sha256": expected_checkpoint,
        "train_positions_consumed": expected_positions,
    }


def measure(model: TimedModel, predictor: TimedPredictor,
            game: GameRules, board: np.ndarray, player: int,
            *, sims: int, state_index: int) -> dict:
    model.forward_ns = model.forward_calls = 0
    predictor.predict_ns = predictor.predict_calls = 0
    search = NumpyMCTS(game, predictor, simulations=sims, temperature=0.4,
                       forced_tactics=True, seed=20260908 + state_index)
    started = time.perf_counter_ns()
    result = search.run(board, player)
    response_ns = time.perf_counter_ns() - started
    root = search._key(board, player)
    if root not in search.counts:
        raise ValueError("chosen point bypassed MCTS through a tactical shortcut")
    if predictor.predict_calls != model.forward_calls:
        raise ValueError("predictor and NN call counts differ")
    expansions = len(search.priors) - 1
    terminal_leaves = sims - expansions
    if not 0 <= terminal_leaves <= sims:
        raise ValueError("tree accounting differs from fixed simulation budget")
    valid = game.get_valid_moves(board).astype(bool)
    root_counts = search.counts[root].astype(np.float64)
    root_total = int(root_counts.sum())
    if root_total != sims:
        raise ValueError("root visit count differs from fixed simulation budget")
    root_prob = root_counts / root_total
    root_prior = search.priors[root]
    root_legal = int(valid.sum())
    depth0 = int(np.count_nonzero(board))
    depths = [int(np.count_nonzero(np.frombuffer(key[0], dtype=np.int8))) - depth0
              for key in search.priors]
    mean_traversed_edges = sum(int(values.sum()) for values in search.counts.values()) / sims
    top = int(np.argmax(root_counts))
    top_q = float(search.values[root][top] / root_counts[top]) if root_counts[top] else 0.0
    return {
        "response_s": response_ns / 1e9,
        "predictor_s": predictor.predict_ns / 1e9,
        "nn_forward_s": model.forward_ns / 1e9,
        "tree_and_rules_s": (response_ns - predictor.predict_ns) / 1e9,
        "nn_calls": model.forward_calls,
        "nn_mean_ms": model.forward_ns / max(1, model.forward_calls) / 1e6,
        "expanded_nodes": len(search.priors),
        "new_leaf_expansions": expansions,
        "terminal_leaf_visits": terminal_leaves,
        "terminal_leaf_fraction": terminal_leaves / sims,
        "mean_traversed_edges": mean_traversed_edges,
        "max_expanded_depth": max(depths),
        "mean_expanded_depth": fmean(depths),
        "p90_expanded_depth": float(np.percentile(depths, 90)),
        "legal_root_actions": root_legal,
        "visited_root_actions": int(np.count_nonzero(root_counts)),
        "root_top1_visit_share": float(root_counts.max() / sims),
        "root_visit_entropy": entropy(root_prob),
        "root_effective_actions": math.exp(entropy(root_prob)),
        "root_normalized_entropy": entropy(root_prob) / math.log(root_legal),
        "root_prior_top1_share": float(root_prior[valid].max()),
        "root_prior_effective_actions": math.exp(entropy(root_prior[valid])),
        "root_top1_action": top,
        "root_top1_q": top_q,
        "sampled_action": result.action,
        "reported_value": result.value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-index", type=int, default=12)
    parser.add_argument("--sims", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--idle-s", type=float, default=10.0)
    parser.add_argument("--group-idle-s", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.sims < 1 or args.repeats < 1 or args.idle_s < 0 or args.group_idle_s < 0:
        parser.error("invalid simulation/repeat/idle setting")
    game = GameRules()
    board, player = build_states()[args.state_index]
    if find_forced_tactical_action(game, board, player) is not None:
        parser.error("selected corpus point has a tactical shortcut")
    output = (SOURCE / "latency_1m_vs_2m" /
              f"state{args.state_index}_{args.sims}_idle{args.idle_s:g}_group{args.group_idle_s:g}")
    if not args.execute:
        print(json.dumps({"output": str(output), "names": NAMES,
                          "state_index": args.state_index,
                          "board_occupancy": int(np.count_nonzero(board)),
                          "simulations": args.sims, "repeats": args.repeats}, indent=2))
        return
    target = output / "summary.json"
    if target.exists():
        raise FileExistsError(f"diagnostic result is immutable: {target}")
    parent = read(PARENT / "manifest.json")
    child = read(CHILD / "inputs_remote.json")
    if (parent["schema"] != "connect4-stage2-fla-parent1m-diagnostic-snapshots-v1"
            or child["schema"] != "connect4-stage2-fla-terminal-four-v1"):
        raise ValueError("diagnostic input manifests differ")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for group_index, name in enumerate(NAMES):
        models = {stage: load(stage, name, parent, child)
                  for stage in ("parent_1m", "child_2m")}
        for stage, (model, predictor, _) in models.items():
            predictor.predict(game.get_canonical_form(board, player))
        trials = {stage: [] for stage in models}
        for repeat in range(args.repeats):
            for stage in ("parent_1m", "child_2m"):
                model, predictor, _ = models[stage]
                row = measure(model, predictor, game, board, int(player),
                              sims=args.sims, state_index=args.state_index)
                row.update({"name": name, "stage": stage, "repeat": repeat})
                trials[stage].append(row)
                print(f"{name} {stage} repeat={repeat} response={row['response_s']:.3f}s "
                      f"NN={row['nn_forward_s']:.3f}s calls={row['nn_calls']} "
                      f"nodes={row['expanded_nodes']}", flush=True)
                time.sleep(args.idle_s)
        for stage in ("parent_1m", "child_2m"):
            identity = models[stage][2]
            rows = trials[stage]
            results.append({"name": name, "stage": stage, **identity,
                            "response_mean_s": fmean(r["response_s"] for r in rows),
                            "nn_forward_mean_s": fmean(r["nn_forward_s"] for r in rows),
                            "predictor_mean_s": fmean(r["predictor_s"] for r in rows),
                            "tree_and_rules_mean_s": fmean(r["tree_and_rules_s"] for r in rows),
                            "nn_calls_mean": fmean(r["nn_calls"] for r in rows),
                            "expanded_nodes_mean": fmean(r["expanded_nodes"] for r in rows),
                            "terminal_leaf_fraction_mean": fmean(r["terminal_leaf_fraction"] for r in rows),
                            "root_top1_visit_share_mean": fmean(r["root_top1_visit_share"] for r in rows),
                            "root_effective_actions_mean": fmean(r["root_effective_actions"] for r in rows),
                            "mean_traversed_edges_mean": fmean(r["mean_traversed_edges"] for r in rows),
                            "trials": rows})
        if group_index + 1 < len(NAMES):
            time.sleep(args.group_idle_s)
    atomic_json(target, {
        "schema": "connect4-stage2-fla-fixed-point-search-diagnostic-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_manifest_sha256": sha256(PARENT / "manifest.json"),
        "child_manifest_sha256": sha256(CHILD / "inputs_remote.json"),
        "point": {"corpus": "deterministic-nonterminal-v1",
                  "state_index": args.state_index,
                  "board_occupancy": int(np.count_nonzero(board)),
                  "board_sha256": hashlib.sha256(board.tobytes()).hexdigest(),
                  "player": int(player), "forced_tactics": True},
        "protocol": {"simulations": args.sims, "repeats": args.repeats,
                     "idle_s": args.idle_s, "group_idle_s": args.group_idle_s,
                     "cpuct": 1.0, "temperature": 0.4,
                     "runtime": "native-stage2-pytorch-numpymcts-cpu"},
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor(), "cpu_count": os.cpu_count(),
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads()},
        "interpretation": "search-work and concentration diagnostic; not playing strength",
        "results": results,
    })


if __name__ == "__main__":
    main()
