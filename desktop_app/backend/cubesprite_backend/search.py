from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    action: int
    policy: list[float]
    value: float


def find_forced_tactical_action(game, board, player):
    """Return an immediate win or mandatory block without running MCTS."""
    board = np.asarray(board, dtype=np.int8)
    player = int(player)
    valid_actions = np.flatnonzero(game.get_valid_moves(board) > 0)
    for action in valid_actions:
        next_board, _ = game.get_next_state(board, player, int(action))
        if game.check_win(next_board, player):
            return int(action), "win"
    for action in valid_actions:
        next_board, _ = game.get_next_state(board, -player, int(action))
        if game.check_win(next_board, -player):
            return int(action), "block"
    return None


class NumpyMCTS:
    def __init__(
        self,
        game,
        predictor,
        simulations=256,
        cpuct=1.0,
        temperature=0.4,
        seed=None,
        forced_tactics=True,
    ):
        self.game = game
        self.predictor = predictor
        self.simulations = max(1, int(simulations))
        self.cpuct = float(cpuct)
        self.temperature = max(0.0, float(temperature))
        self.forced_tactics = bool(forced_tactics)
        self.rng = np.random.default_rng(seed)
        self.priors: dict[tuple[bytes, int], np.ndarray] = {}
        self.counts: dict[tuple[bytes, int], np.ndarray] = {}
        self.values: dict[tuple[bytes, int], np.ndarray] = {}

    def run(self, board, player):
        board = np.asarray(board, dtype=np.int8)
        valid = self.game.get_valid_moves(board).astype(bool)
        if not np.any(valid):
            raise ValueError("MCTS cannot search a position without legal moves.")
        forced = find_forced_tactical_action(self.game, board, int(player)) if self.forced_tactics else None
        if forced is not None:
            forced_action, kind = forced
            if kind == "win":
                forced_value = 1.0
            else:
                _, forced_value = self.predictor.predict(self.game.get_canonical_form(board, player))
            policy = np.zeros(self.game.get_action_size(), dtype=np.float64)
            policy[forced_action] = 1.0
            return SearchResult(int(forced_action), policy.tolist(), float(forced_value))

        root = self._key(board, player)
        self._expand(board, player)
        for _ in range(self.simulations):
            self._simulate(board, int(player))
        counts = self.counts[root].astype(np.float64)
        counts[~valid] = 0.0
        policy = self._counts_to_policy(counts, valid)
        action = int(self.rng.choice(len(policy), p=policy)) if self.temperature > 0 else int(np.argmax(policy))
        visited = counts > 0
        if np.any(visited):
            q_values = np.divide(
                self.values[root],
                np.maximum(1.0, counts),
                out=np.zeros_like(self.values[root]),
                where=np.maximum(1.0, counts) > 0,
            )
            value = float(np.sum(q_values[visited] * counts[visited]) / np.sum(counts[visited]))
        else:
            _, value = self.predictor.predict(self.game.get_canonical_form(board, player))
        return SearchResult(action, policy.tolist(), float(np.clip(value, -1.0, 1.0)))

    def _simulate(self, board, player):
        terminal = self.game.get_game_ended(board, player)
        if terminal != 0:
            if terminal == 1e-4:
                return 0.0
            return float(terminal)
        key = self._key(board, player)
        if key not in self.priors:
            return self._expand(board, player)
        valid = self.game.get_valid_moves(board).astype(bool)
        counts = self.counts[key]
        q = np.divide(self.values[key], np.maximum(1, counts), out=np.zeros_like(self.values[key]), where=counts > 0)
        score = q + self.cpuct * self.priors[key] * np.sqrt(float(np.sum(counts)) + 1.0) / (1.0 + counts)
        score[~valid] = -np.inf
        action = int(np.argmax(score))
        next_board, next_player = self.game.get_next_state(board, player, action)
        value = -self._simulate(next_board, next_player)
        counts[action] += 1
        self.values[key][action] += value
        return value

    def _expand(self, board, player):
        key = self._key(board, player)
        policy, value = self.predictor.predict(self.game.get_canonical_form(board, player))
        policy = np.asarray(policy, dtype=np.float64)
        valid = self.game.get_valid_moves(board).astype(bool)
        policy[~valid] = 0.0
        total = float(policy.sum())
        policy = policy / total if total > 0 else valid.astype(np.float64) / max(1, int(valid.sum()))
        self.priors[key] = policy
        self.counts[key] = np.zeros(self.game.get_action_size(), dtype=np.int32)
        self.values[key] = np.zeros(self.game.get_action_size(), dtype=np.float64)
        return float(value)

    def _counts_to_policy(self, counts, valid):
        if self.temperature <= 0:
            policy = np.zeros_like(counts)
            if np.any(valid):
                candidates = np.flatnonzero(valid)
                policy[int(candidates[np.argmax(counts[candidates])])] = 1.0
            return policy
        exponent = 1.0 / max(self.temperature, 1e-6)
        weights = np.zeros_like(counts)
        weights[valid] = np.power(np.maximum(counts[valid], 1e-12), exponent)
        total = float(weights.sum())
        return weights / total if total > 0 else valid.astype(np.float64) / max(1, int(valid.sum()))

    @staticmethod
    def _key(board, player):
        return np.asarray(board, dtype=np.int8).tobytes(), int(player)
