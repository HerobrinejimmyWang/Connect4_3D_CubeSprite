from __future__ import annotations

import threading

import numpy as np

from connect4_core import GameRules


class HumanGameController:
    """Thread-safe game state independent from Pygame rendering."""

    def __init__(self, agent, human_player=1, game=None):
        self.game = game or GameRules()
        self.agent = agent
        self.human_player = int(human_player)
        if self.human_player not in (-1, 1):
            raise ValueError("human_player must be +1 (red) or -1 (blue).")
        self.ai_player = -self.human_player
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.board = self.game.get_init_board()
            self.current_player = 1
            self.last_action = None
            self.winner = None
            self.status = "playing"
            self.error = None
            self.ai_thinking = False

    @property
    def is_finished(self):
        return self.status in {"won", "draw", "error"}

    @property
    def is_human_turn(self):
        return self.status == "playing" and not self.ai_thinking and self.current_player == self.human_player

    @property
    def is_ai_turn(self):
        return self.status == "playing" and not self.ai_thinking and self.current_player == self.ai_player

    def snapshot(self):
        with self._lock:
            return {
                "board": self.board.copy(),
                "current_player": self.current_player,
                "human_player": self.human_player,
                "last_action": self.last_action,
                "winner": self.winner,
                "status": self.status,
                "error": self.error,
                "ai_thinking": self.ai_thinking,
            }

    def human_move(self, layer, row, col):
        with self._lock:
            if not self.is_human_turn:
                raise RuntimeError("It is not the human player's turn.")
            action = self.game.coords_to_action(layer, row, col)
            self._apply_action(action, self.human_player)
            return action

    def begin_ai_turn(self):
        with self._lock:
            if not self.is_ai_turn:
                return None
            self.ai_thinking = True
            return self.board.copy(), self.ai_player

    def finish_ai_turn(self, action):
        with self._lock:
            if self.status != "playing" or not self.ai_thinking or self.current_player != self.ai_player:
                return False
            try:
                valid = self.game.get_valid_moves(self.board)
                action = int(action)
                if action < 0 or action >= len(valid) or valid[action] == 0:
                    raise ValueError(f"AI returned illegal action {action}.")
                self._apply_action(action, self.ai_player)
                return True
            except Exception as exc:
                self.fail(exc)
                return False
            finally:
                self.ai_thinking = False

    def fail(self, error):
        with self._lock:
            self.ai_thinking = False
            self.status = "error"
            self.error = str(error)

    def _apply_action(self, action, player):
        self.board, next_player = self.game.get_next_state(self.board, player, action)
        self.last_action = int(action)
        if self.game.check_win(self.board, player):
            self.status = "won"
            self.winner = int(player)
        elif not np.any(self.game.get_valid_moves(self.board)):
            self.status = "draw"
            self.winner = 0
        else:
            self.current_player = int(next_player)

    def close(self):
        close = getattr(self.agent, "close", None)
        if callable(close):
            close()
