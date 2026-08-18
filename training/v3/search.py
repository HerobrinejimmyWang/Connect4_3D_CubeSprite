from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from connect4_core import GameRules

from .model import (
    COLUMN_COUNT,
    legal_column_mask,
    wdl_expected_value,
)


SEARCH_FAST = "fast"
SEARCH_FULL = "full"


class Predictor(Protocol):
    def predict(self, canonical_board: np.ndarray) -> tuple[np.ndarray, np.ndarray | float]: ...


class RandomPredictor:
    """Neutral prior/value predictor used only to bootstrap the first generation."""

    def predict(self, canonical_board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del canonical_board
        return (
            np.full(COLUMN_COUNT, 1.0 / COLUMN_COUNT, dtype=np.float32),
            np.asarray((0.5, 0.0, 0.5), dtype=np.float32),
        )

    def predict_batch(self, canonical_boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boards = np.asarray(canonical_boards)
        if boards.ndim != 4 or boards.shape[0] < 1:
            raise ValueError("canonical_boards must be a non-empty board batch")
        count = boards.shape[0]
        return (
            np.full((count, COLUMN_COUNT), 1.0 / COLUMN_COUNT, dtype=np.float32),
            np.tile(np.asarray((0.5, 0.0, 0.5), dtype=np.float32), (count, 1)),
        )


@dataclass
class TreeNode:
    board: np.ndarray | None
    prior: float = 1.0
    action_from_parent: int | None = None
    legacy_action_from_parent: int | None = None
    children: dict[int, "TreeNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_loss_count: int = 0
    expanded: bool = False
    terminal_value: float | None = None
    initial_value: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def apply_virtual_loss(self, amount: float) -> None:
        """Penalize this action in its parent's perspective.

        Values are stored from this node's side-to-move perspective. A parent uses
        ``-child.q_value``, so adding a positive virtual value to the child lowers
        the parent's selection score.
        """
        self.virtual_loss_count += 1
        self.visit_count += 1
        self.value_sum += float(amount)

    def revert_virtual_loss(self, amount: float) -> None:
        if self.virtual_loss_count < 1:
            raise RuntimeError("Cannot revert a virtual loss that was not applied.")
        self.virtual_loss_count -= 1
        self.visit_count -= 1
        self.value_sum -= float(amount)

    def add_backup(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += float(value)


@dataclass(frozen=True)
class SearchResult:
    visit_counts: np.ndarray
    policy: np.ndarray
    root_value: float
    simulations: int
    inference_calls: int
    inference_positions: int
    max_inference_batch: int


def puct_score(parent: TreeNode, child: TreeNode, cpuct: float) -> float:
    parent_visits = max(1, parent.visit_count)
    exploration = float(cpuct) * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    return -child.q_value + exploration


def _normalize_policy(policy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if policy.shape != (COLUMN_COUNT,):
        raise ValueError(f"Predictor policy must have {COLUMN_COUNT} entries, got {policy.shape}.")
    if not np.all(np.isfinite(policy)) or np.any(policy < 0.0):
        raise ValueError("Predictor policy must contain finite non-negative probabilities.")
    masked = np.where(valid, policy, 0.0)
    total = float(masked.sum())
    if total <= 0.0:
        count = int(valid.sum())
        if count == 0:
            return np.zeros(COLUMN_COUNT, dtype=np.float64)
        return valid.astype(np.float64) / count
    return masked / total


def _value_from_prediction(value: np.ndarray | float) -> float:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        return float(np.clip(raw.item(), -1.0, 1.0))
    if raw.shape != (3,):
        raise ValueError(f"Predictor value must be a scalar or three WDL probabilities, got {raw.shape}.")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("Predictor WDL output must contain finite non-negative probabilities.")
    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("Predictor WDL probabilities sum to zero.")
    return float(wdl_expected_value(raw / total))


class MCTS:
    """Deterministic reference MCTS over 25 gravity columns.

    ``num_threads`` controls deterministic virtual-loss lanes. The implementation
    stays single-process and evaluates lanes in a fixed order, making smoke runs
    exactly reproducible while exercising the same selection semantics needed by
    a later parallel inference implementation.
    """

    def __init__(
        self,
        predictor: Predictor,
        *,
        cpuct: float = 1.5,
        virtual_loss: float = 1.0,
        num_threads: int = 1,
        game: GameRules | None = None,
    ) -> None:
        if not hasattr(predictor, "predict"):
            raise TypeError("MCTS predictor must implement predict(canonical_board).")
        if cpuct <= 0.0 or virtual_loss < 0.0 or num_threads < 1:
            raise ValueError("Invalid MCTS cpuct, virtual_loss, or num_threads.")
        self.predictor = predictor
        self.cpuct = float(cpuct)
        self.virtual_loss = float(virtual_loss)
        self.num_threads = int(num_threads)
        self.game = game or GameRules()
        self._get_next_state = getattr(self.game, "get_next_state_fast", self.game.get_next_state)
        self._get_canonical_form = getattr(
            self.game,
            "get_canonical_form_fast",
            self.game.get_canonical_form,
        )
        self._get_game_ended_after_action = getattr(
            self.game,
            "get_game_ended_after_action_fast",
            None,
        )
        self.inference_calls = 0
        self.inference_positions = 0
        self.max_inference_batch = 0

    def search(
        self,
        canonical_board: np.ndarray,
        simulations: int,
        *,
        rng: np.random.Generator,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> SearchResult:
        if simulations < 1:
            raise ValueError("simulations must be positive.")
        raw_board = np.asarray(canonical_board)
        if raw_board.shape != self.game.board_shape:
            raise ValueError(f"Canonical board shape must be {self.game.board_shape}, got {raw_board.shape}.")
        if not np.all(np.isin(raw_board, (-1, 0, 1))):
            raise ValueError("Canonical board cells must contain only -1, 0, or 1.")
        board = np.asarray(raw_board, dtype=np.int8)
        root_terminal = self._terminal_value(board)
        if root_terminal is not None:
            raise ValueError("Cannot search a terminal board.")

        inference_before = self.inference_calls
        inference_positions_before = self.inference_positions
        root = TreeNode(board=np.array(board, copy=True))
        self._expand(root)
        if add_root_noise and root.children and dirichlet_epsilon > 0.0:
            if dirichlet_alpha <= 0.0 or not 0.0 <= dirichlet_epsilon <= 1.0:
                raise ValueError("Invalid Dirichlet noise parameters.")
            actions = sorted(root.children)
            noise = rng.dirichlet(np.full(len(actions), dirichlet_alpha, dtype=np.float64))
            for action, sampled in zip(actions, noise, strict=True):
                child = root.children[action]
                child.prior = (1.0 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * float(sampled)

        completed = 0
        while completed < simulations:
            lane_count = min(self.num_threads, simulations - completed)
            lanes: list[tuple[list[TreeNode], list[TreeNode]]] = []
            for _ in range(lane_count):
                lanes.append(self._select_lane(root))

            pending: list[TreeNode] = []
            seen: set[int] = set()
            for path, _virtual_nodes in lanes:
                leaf = path[-1]
                terminal = self._terminal_value(
                    leaf.board,
                    last_action=leaf.legacy_action_from_parent,
                )
                if terminal is not None:
                    leaf.terminal_value = terminal
                    leaf.initial_value = terminal
                elif not leaf.expanded and id(leaf) not in seen:
                    seen.add(id(leaf))
                    pending.append(leaf)
            self._expand_many(pending)
            values = [path[-1].initial_value for path, _virtual_nodes in lanes]

            for _path, virtual_nodes in lanes:
                for node in reversed(virtual_nodes):
                    node.revert_virtual_loss(self.virtual_loss)
            for (path, _virtual_nodes), value in zip(lanes, values, strict=True):
                self._backup(path, value)
            completed += lane_count

        counts = np.zeros(COLUMN_COUNT, dtype=np.uint32)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        total = int(counts.sum())
        if total != simulations:
            raise RuntimeError(f"MCTS visit accounting error: expected {simulations}, got {total}.")
        policy = counts.astype(np.float64) / float(total)
        return SearchResult(
            visit_counts=counts,
            policy=policy.astype(np.float32),
            root_value=float(root.q_value),
            simulations=simulations,
            inference_calls=self.inference_calls - inference_before,
            inference_positions=self.inference_positions - inference_positions_before,
            max_inference_batch=self.max_inference_batch,
        )

    def _terminal_value(
        self,
        board: np.ndarray | None,
        *,
        last_action: int | None = None,
    ) -> float | None:
        if board is None:
            raise RuntimeError("Cannot evaluate an unmaterialized MCTS node.")
        if last_action is not None and self._get_game_ended_after_action is not None:
            result = float(self._get_game_ended_after_action(board, 1, last_action))
        else:
            result = float(self.game.get_game_ended(board, 1))
        if result == 0.0:
            return None
        if math.isclose(result, 1e-4, rel_tol=0.0, abs_tol=1e-9):
            return 0.0
        return float(np.clip(result, -1.0, 1.0))

    def _expand(self, node: TreeNode) -> float:
        if node.expanded:
            return node.initial_value
        self._expand_many([node])
        return node.initial_value

    def _expand_many(self, nodes: list[TreeNode]) -> None:
        nodes = [node for node in nodes if not node.expanded]
        if not nodes:
            return
        if any(node.board is None for node in nodes):
            raise RuntimeError("Cannot expand an unmaterialized MCTS node.")
        boards = np.stack([node.board for node in nodes], axis=0)
        batch_predict = getattr(self.predictor, "predict_batch", None)
        if callable(batch_predict):
            raw_policies, raw_values = batch_predict(boards)
            policies = np.asarray(raw_policies)
            values = np.asarray(raw_values)
            self.inference_calls += 1
        else:
            predictions = [self.predictor.predict(board) for board in boards]
            policies = np.stack([prediction[0] for prediction in predictions], axis=0)
            values = np.stack(
                [np.asarray(prediction[1]) for prediction in predictions], axis=0
            )
            self.inference_calls += len(nodes)
        if policies.shape != (len(nodes), COLUMN_COUNT):
            raise ValueError(
                "Batch predictor policy must have shape "
                f"[{len(nodes)},{COLUMN_COUNT}], got {policies.shape}."
            )
        if values.shape not in {(len(nodes),), (len(nodes), 3)}:
            raise ValueError(
                "Batch predictor value must have shape [N] or [N,3], "
                f"got {values.shape}."
            )
        self.inference_positions += len(nodes)
        self.max_inference_batch = max(self.max_inference_batch, len(nodes))
        for index, node in enumerate(nodes):
            valid = legal_column_mask(node.board).reshape(-1)
            policy = _normalize_policy(policies[index], valid)
            for action in np.flatnonzero(valid):
                column = int(action)
                node.children[column] = TreeNode(
                    board=None,
                    prior=float(policy[column]),
                    action_from_parent=column,
                )
            node.initial_value = _value_from_prediction(values[index])
            node.expanded = True

    def _select_lane(self, root: TreeNode) -> tuple[list[TreeNode], list[TreeNode]]:
        path = [root]
        virtual_nodes: list[TreeNode] = []
        node = root
        while node.expanded and node.children:
            action, child = max(
                node.children.items(),
                key=lambda item: (puct_score(node, item[1], self.cpuct), -item[0]),
            )
            del action
            self._materialize_child(node, child)
            child.apply_virtual_loss(self.virtual_loss)
            virtual_nodes.append(child)
            path.append(child)
            node = child
            if not node.expanded or node.terminal_value is not None:
                break
        return path, virtual_nodes

    def _materialize_child(self, parent: TreeNode, child: TreeNode) -> np.ndarray:
        if child.board is not None:
            return child.board
        if parent.board is None or child.action_from_parent is None:
            raise RuntimeError("Cannot materialize a child without its parent board and action.")
        row, col = divmod(int(child.action_from_parent), self.game.board_size)
        layer = int(np.count_nonzero(parent.board[:, row, col]))
        legacy_action = layer * self.game.board_size * self.game.board_size + int(
            child.action_from_parent
        )
        next_board, next_player = self._get_next_state(parent.board, 1, legacy_action)
        child.board = np.asarray(
            self._get_canonical_form(next_board, next_player),
            dtype=np.int8,
        )
        child.legacy_action_from_parent = legacy_action
        return child.board

    @staticmethod
    def _backup(path: list[TreeNode], leaf_value: float) -> None:
        value = float(leaf_value)
        for node in reversed(path):
            node.add_backup(value)
            value = -value


def policy_from_visits(
    visit_counts: np.ndarray,
    *,
    temperature: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.asarray(visit_counts, dtype=np.float64).reshape(-1)
    if counts.shape != (COLUMN_COUNT,) or np.any(counts < 0.0):
        raise ValueError(f"visit_counts must contain {COLUMN_COUNT} non-negative entries.")
    valid = np.ones(COLUMN_COUNT, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape != (COLUMN_COUNT,):
        raise ValueError(f"valid_mask must contain {COLUMN_COUNT} entries.")
    counts = np.where(valid, counts, 0.0)
    if temperature == 0.0:
        result = np.zeros(COLUMN_COUNT, dtype=np.float64)
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            raise ValueError("No valid action is available.")
        best = candidates[np.argmax(counts[candidates])]
        result[int(best)] = 1.0
        return result.astype(np.float32)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    weights = np.zeros_like(counts)
    positive = counts > 0.0
    weights[positive] = np.power(counts[positive], 1.0 / temperature)
    total = float(weights.sum())
    if total <= 0.0:
        valid_count = int(valid.sum())
        if valid_count == 0:
            raise ValueError("No valid action is available.")
        weights = valid.astype(np.float64) / valid_count
    else:
        weights /= total
    return weights.astype(np.float32)


__all__ = [
    "MCTS",
    "Predictor",
    "RandomPredictor",
    "SEARCH_FAST",
    "SEARCH_FULL",
    "SearchResult",
    "TreeNode",
    "policy_from_visits",
    "puct_score",
]
