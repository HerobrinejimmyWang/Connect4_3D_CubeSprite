from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from connect4_core.rules import (
    FEATURE_DIM,
    GameOutcome,
    GameState,
    RuleEngine,
    TurnAction,
    TurnKind,
)

from .model import COLUMN_COUNT, ROLE_FEATURE_COUNT, wdl_expected_value


SEARCH_FAST = "fast"
SEARCH_FULL = "full"
SEARCH_NONE = "none"
PASS_ACTION = -1


def role_features_for_player(player_to_move: int) -> np.ndarray:
    """Encode the absolute starter role separately from the canonical board."""

    if int(player_to_move) not in (-1, 1):
        raise ValueError("player_to_move must be +1 (FIRST) or -1 (SECOND).")
    return np.asarray((1.0, 0.0) if int(player_to_move) == 1 else (0.0, 1.0), dtype=np.float32)


def canonical_board_for_state(state: GameState) -> np.ndarray:
    """Return a board where +1 is the side to move without losing absolute role."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState.")
    return np.asarray(state.board * state.player_to_move, dtype=np.int8)


def _validate_predictor_context(
    canonical_boards: np.ndarray,
    role_to_play: np.ndarray,
    rule_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boards = np.asarray(canonical_boards)
    roles = np.asarray(role_to_play, dtype=np.float32)
    rules = np.asarray(rule_features, dtype=np.float32)
    if boards.ndim != 4 or boards.shape[1:] != (6, 5, 5) or boards.shape[0] < 1:
        raise ValueError("canonical_boards must have shape [N,6,5,5] with N >= 1.")
    if roles.shape != (boards.shape[0], ROLE_FEATURE_COUNT):
        raise ValueError(f"role_to_play must have shape [N,{ROLE_FEATURE_COUNT}].")
    if rules.shape != (boards.shape[0], FEATURE_DIM):
        raise ValueError(f"rule_features must have shape [N,{FEATURE_DIM}].")
    if not np.all(np.isfinite(roles)) or not np.all(np.isin(roles, (0.0, 1.0))):
        raise ValueError("role_to_play must be a finite one-hot array.")
    if not np.allclose(roles.sum(axis=1), 1.0):
        raise ValueError("role_to_play must contain exactly one active role per row.")
    if not np.all(np.isfinite(rules)):
        raise ValueError("rule_features must be finite.")
    return boards, roles, rules


class Predictor(Protocol):
    def predict(
        self,
        canonical_board: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | float]: ...

    def predict_batch(
        self,
        canonical_boards: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class RandomPredictor:
    """Neutral prior/value predictor used only to bootstrap the first generation."""

    def predict(
        self,
        canonical_board: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        boards, _roles, _rules = _validate_predictor_context(
            np.asarray(canonical_board)[None, ...],
            np.asarray(role_to_play)[None, ...]
            if np.asarray(role_to_play).ndim == 1
            else np.asarray(role_to_play),
            np.asarray(rule_features)[None, ...]
            if np.asarray(rule_features).ndim == 1
            else np.asarray(rule_features),
        )
        del boards
        return (
            np.full(COLUMN_COUNT, 1.0 / COLUMN_COUNT, dtype=np.float32),
            np.asarray((0.5, 0.0, 0.5), dtype=np.float32),
        )

    def predict_batch(
        self,
        canonical_boards: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        boards, _roles, _rules = _validate_predictor_context(
            canonical_boards, role_to_play, rule_features
        )
        count = boards.shape[0]
        return (
            np.full((count, COLUMN_COUNT), 1.0 / COLUMN_COUNT, dtype=np.float32),
            np.tile(np.asarray((0.5, 0.0, 0.5), dtype=np.float32), (count, 1)),
        )


@dataclass
class TreeNode:
    # ``board`` remains available for diagnostics and the existing PUCT unit
    # tests. Search-created nodes also retain the authoritative absolute state.
    board: np.ndarray | None = None
    state: GameState | None = None
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
        """Penalize this action in its parent's perspective."""

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
    """Deterministic 25-column MCTS over an explicit versioned rule state."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        engine: RuleEngine,
        cpuct: float = 1.5,
        virtual_loss: float = 1.0,
        num_threads: int = 1,
    ) -> None:
        if not hasattr(predictor, "predict"):
            raise TypeError("MCTS predictor must implement predict with role/rule context.")
        if not isinstance(engine, RuleEngine):
            raise TypeError("MCTS requires an explicit RuleEngine.")
        if cpuct <= 0.0 or virtual_loss < 0.0 or num_threads < 1:
            raise ValueError("Invalid MCTS cpuct, virtual_loss, or num_threads.")
        self.predictor = predictor
        self.engine = engine
        self.cpuct = float(cpuct)
        self.virtual_loss = float(virtual_loss)
        self.num_threads = int(num_threads)
        self._rule_features = np.asarray(
            self.engine.registry.features(self.engine.spec), dtype=np.float32
        )
        self.inference_calls = 0
        self.inference_positions = 0
        self.max_inference_batch = 0

    def search(
        self,
        state: GameState,
        simulations: int,
        *,
        rng: np.random.Generator,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> SearchResult:
        if simulations < 1:
            raise ValueError("simulations must be positive.")
        if not isinstance(state, GameState):
            raise TypeError("MCTS.search requires an explicit GameState.")
        if state.rule_id != self.engine.spec.rule_id:
            raise ValueError("MCTS state and RuleEngine use different rules.")
        if state.terminal:
            raise ValueError("Cannot search a terminal state.")
        if self.engine.required_action(state) is not None:
            raise ValueError("A forced-pass state must be advanced without a policy search.")

        inference_before = self.inference_calls
        inference_positions_before = self.inference_positions
        root = TreeNode(board=canonical_board_for_state(state), state=state)
        self._expand(root)
        if add_root_noise and root.children and dirichlet_epsilon > 0.0:
            if dirichlet_alpha <= 0.0 or not 0.0 <= dirichlet_epsilon <= 1.0:
                raise ValueError("Invalid Dirichlet noise parameters.")
            actions = sorted(action for action in root.children if action != PASS_ACTION)
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
                terminal = self._terminal_value(leaf.state)
                if terminal is not None:
                    leaf.terminal_value = terminal
                    leaf.initial_value = terminal
                    leaf.expanded = True
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
            if action != PASS_ACTION:
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

    @staticmethod
    def _terminal_value(state: GameState | None) -> float | None:
        if state is None:
            raise RuntimeError("Cannot evaluate an unmaterialized MCTS node.")
        if state.outcome == GameOutcome.ONGOING:
            return None
        if state.outcome == GameOutcome.DRAW:
            return 0.0
        winner = state.outcome.winner
        if winner is None:
            raise RuntimeError(f"Unsupported terminal outcome {state.outcome!r}.")
        return 1.0 if winner == state.player_to_move else -1.0

    def _expand(self, node: TreeNode) -> float:
        if node.expanded:
            return node.initial_value
        self._expand_many([node])
        return node.initial_value

    def _expand_many(self, nodes: list[TreeNode]) -> None:
        nodes = [node for node in nodes if not node.expanded]
        if not nodes:
            return
        decision_nodes: list[TreeNode] = []
        for node in nodes:
            if node.state is None:
                raise RuntimeError("Cannot expand an MCTS node without an authoritative state.")
            terminal = self._terminal_value(node.state)
            if terminal is not None:
                node.terminal_value = terminal
                node.initial_value = terminal
                node.expanded = True
                continue
            required = self.engine.required_action(node.state)
            if required is not None:
                self._expand_forced_pass(node, required)
                continue
            decision_nodes.append(node)
        if not decision_nodes:
            return

        boards = np.stack([canonical_board_for_state(node.state) for node in decision_nodes], axis=0)
        roles = np.stack(
            [role_features_for_player(node.state.player_to_move) for node in decision_nodes], axis=0
        )
        rules = np.tile(self._rule_features, (len(decision_nodes), 1))
        batch_predict = getattr(self.predictor, "predict_batch", None)
        if callable(batch_predict):
            raw_policies, raw_values = batch_predict(
                boards, role_to_play=roles, rule_features=rules
            )
            policies = np.asarray(raw_policies)
            values = np.asarray(raw_values)
            self.inference_calls += 1
        else:
            predictions = [
                self.predictor.predict(
                    board, role_to_play=role, rule_features=rule
                )
                for board, role, rule in zip(boards, roles, rules, strict=True)
            ]
            policies = np.stack([prediction[0] for prediction in predictions], axis=0)
            values = np.stack([np.asarray(prediction[1]) for prediction in predictions], axis=0)
            self.inference_calls += len(decision_nodes)
        if policies.shape != (len(decision_nodes), COLUMN_COUNT):
            raise ValueError(
                "Batch predictor policy must have shape "
                f"[{len(decision_nodes)},{COLUMN_COUNT}], got {policies.shape}."
            )
        if values.shape not in {(len(decision_nodes),), (len(decision_nodes), 3)}:
            raise ValueError(
                "Batch predictor value must have shape [N] or [N,3], "
                f"got {values.shape}."
            )
        self.inference_positions += len(decision_nodes)
        self.max_inference_batch = max(self.max_inference_batch, len(decision_nodes))
        for index, node in enumerate(decision_nodes):
            assert node.state is not None
            valid = self.engine.legal_column_mask(node.state).astype(bool)
            policy = _normalize_policy(policies[index], valid)
            for action in np.flatnonzero(valid):
                column = int(action)
                node.children[column] = TreeNode(
                    prior=float(policy[column]), action_from_parent=column
                )
            node.initial_value = _value_from_prediction(values[index])
            node.expanded = True

    def _expand_forced_pass(self, node: TreeNode, action: TurnAction) -> None:
        if action.kind != TurnKind.FORCED_PASS:
            raise RuntimeError("RuleEngine.required_action returned a non-pass action.")
        child = TreeNode(prior=1.0, action_from_parent=PASS_ACTION)
        node.children[PASS_ACTION] = child
        self._materialize_child(node, child)
        self._expand_many([child])
        node.initial_value = -child.initial_value
        node.expanded = True

    def _select_lane(self, root: TreeNode) -> tuple[list[TreeNode], list[TreeNode]]:
        path = [root]
        virtual_nodes: list[TreeNode] = []
        node = root
        while node.expanded and node.children:
            _action, child = max(
                node.children.items(),
                key=lambda item: (puct_score(node, item[1], self.cpuct), -item[0]),
            )
            self._materialize_child(node, child)
            child.apply_virtual_loss(self.virtual_loss)
            virtual_nodes.append(child)
            path.append(child)
            node = child
            if not node.expanded or node.terminal_value is not None:
                break
        return path, virtual_nodes

    def _materialize_child(self, parent: TreeNode, child: TreeNode) -> GameState:
        if child.state is not None:
            return child.state
        if parent.state is None or child.action_from_parent is None:
            raise RuntimeError("Cannot materialize a child without its parent state and action.")
        if child.action_from_parent == PASS_ACTION:
            action = TurnAction.forced_pass()
        else:
            action = TurnAction.place(int(child.action_from_parent))
            child.legacy_action_from_parent = self.engine.legacy_action_for_column(
                parent.state, int(child.action_from_parent)
            )
        child.state = self.engine.step(parent.state, action)
        child.board = canonical_board_for_state(child.state)
        child.terminal_value = self._terminal_value(child.state)
        return child.state

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
    valid = (
        np.ones(COLUMN_COUNT, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool).reshape(-1)
    )
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
    "PASS_ACTION",
    "Predictor",
    "RandomPredictor",
    "SEARCH_FAST",
    "SEARCH_FULL",
    "SEARCH_NONE",
    "SearchResult",
    "TreeNode",
    "canonical_board_for_state",
    "policy_from_visits",
    "puct_score",
    "role_features_for_player",
]
