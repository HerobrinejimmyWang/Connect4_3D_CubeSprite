from __future__ import annotations

import queue
import threading
from pathlib import Path

import torch

from connect4_core import GameRules
from connect4_runtime import ModelRegistry, load_v22_agent
from game_client.state import HumanGameController


COLOR_BG = (14, 18, 33)
COLOR_PANEL = (25, 32, 53)
COLOR_GRID = (58, 70, 100)
COLOR_TEXT = (230, 234, 244)
COLOR_DIM = (150, 160, 185)
COLOR_ACCENT = (73, 190, 166)
COLOR_RED = (232, 78, 86)
COLOR_BLUE = (69, 137, 230)
COLOR_ERROR = (245, 166, 35)


class GameClientApp:
    def __init__(self, model_roots, width=1220, height=850):
        self.model_roots = [Path(root).resolve() for root in model_roots]
        self.registry = ModelRegistry(self.model_roots)
        self.width, self.height = int(width), int(height)
        self.models = []
        self.selected_index = 0
        self.human_player = 1
        self.device_choice = "auto"
        self.mcts_sims = 256
        self.scene = "launcher"
        self.controller = None
        self.ai_thread = None
        self.ai_results = queue.Queue()
        self.message = ""
        self.running = False
        self._buttons = []
        self._cells = []
        self.refresh_models()

    def refresh_models(self):
        self.models = self.registry.discover()
        self.selected_index = min(self.selected_index, max(0, len(self.models) - 1))
        self.message = f"Found {len(self.models)} model(s)." if self.models else "No completed v2.2 checkpoints found."

    def run(self, max_frames=None):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Pygame is required to run the game client.") from exc

        pygame.init()
        pygame.display.set_caption("Connect4 3D v2.2")
        screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        title_font = pygame.font.SysFont("Microsoft YaHei", 30, bold=True)
        font = pygame.font.SysFont("Microsoft YaHei", 20)
        small_font = pygame.font.SysFont("Microsoft YaHei", 16)
        self.running = True
        frames = 0
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.width, self.height = event.size
                        screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos)
                    elif event.type == pygame.MOUSEWHEEL and self.scene == "launcher":
                        self._move_model_selection(-event.y)
                    elif event.type == pygame.KEYDOWN and self.scene == "launcher":
                        if event.key == pygame.K_UP:
                            self._move_model_selection(-1)
                        elif event.key == pygame.K_DOWN:
                            self._move_model_selection(1)
                self._consume_ai_result()
                self._maybe_start_ai()
                screen.fill(COLOR_BG)
                if self.scene == "launcher":
                    self._draw_launcher(pygame, screen, title_font, font, small_font)
                else:
                    self._draw_game(pygame, screen, title_font, font, small_font)
                pygame.display.flip()
                clock.tick(30)
                frames += 1
                if max_frames is not None and frames >= int(max_frames):
                    self.running = False
        finally:
            self._shutdown_controller()
            pygame.quit()

    def _start_game(self):
        if not self.models:
            self.message = "No model is available. Add a --model-root and refresh."
            return
        self._shutdown_controller()
        item = self.models[self.selected_index]
        device = None if self.device_choice == "auto" else self.device_choice
        try:
            agent = load_v22_agent(
                GameRules(), item.path, name=item.path.stem, device=device,
                num_mcts_sims=self.mcts_sims, num_mcts_threads=2,
            )
            self.controller = HumanGameController(agent, human_player=self.human_player, game=agent.game)
            self.scene = "game"
            self.message = ""
        except Exception as exc:
            self.controller = None
            self.message = f"Model load failed: {exc}"

    def _maybe_start_ai(self):
        if self.controller is None or self.ai_thread is not None:
            return
        request = self.controller.begin_ai_turn()
        if request is None:
            return
        board, player = request

        def worker():
            try:
                action, _ = self.controller.agent.get_action(board, player, temp=0)
                self.ai_results.put(("ok", int(action)))
            except Exception as exc:
                self.ai_results.put(("error", exc))

        self.ai_thread = threading.Thread(target=worker, name="connect4-ai-turn", daemon=True)
        self.ai_thread.start()

    def _consume_ai_result(self):
        try:
            status, payload = self.ai_results.get_nowait()
        except queue.Empty:
            return
        if self.controller is not None:
            if status == "ok":
                self.controller.finish_ai_turn(payload)
            else:
                self.controller.fail(payload)
        self.ai_thread = None

    def _handle_click(self, pos):
        for rect, callback in reversed(self._buttons):
            if rect.collidepoint(pos):
                callback()
                return
        if self.scene != "game" or self.controller is None or not self.controller.is_human_turn:
            return
        for rect, coords in self._cells:
            if rect.collidepoint(pos):
                try:
                    self.controller.human_move(*coords)
                except (ValueError, RuntimeError) as exc:
                    self.message = str(exc)
                return

    def _draw_launcher(self, pygame, screen, title_font, font, small_font):
        self._buttons = []
        self._cells = []
        width, height = screen.get_size()
        screen.blit(title_font.render("Connect4 3D v2.2 - Human vs AI", True, COLOR_TEXT), (40, 28))
        panel = pygame.Rect(40, 88, width - 80, height - 130)
        pygame.draw.rect(screen, COLOR_PANEL, panel, border_radius=12)
        screen.blit(font.render("Model checkpoint", True, COLOR_TEXT), (70, 115))
        list_rect = pygame.Rect(70, 150, width - 140, min(360, height - 390))
        pygame.draw.rect(screen, COLOR_BG, list_rect, border_radius=8)
        row_height = 42
        visible = max(1, list_rect.height // row_height)
        start = min(self.selected_index, max(0, len(self.models) - visible))
        for index, item in enumerate(self.models[start : start + visible], start=start):
            row = pygame.Rect(list_rect.x + 6, list_rect.y + 6 + (index - start) * row_height, list_rect.width - 12, 36)
            pygame.draw.rect(screen, COLOR_ACCENT if index == self.selected_index else COLOR_GRID, row, border_radius=5)
            screen.blit(small_font.render(item.label, True, COLOR_TEXT), (row.x + 10, row.y + 8))
            self._buttons.append((row, lambda i=index: setattr(self, "selected_index", i)))

        y = list_rect.bottom + 24
        side = "Human: Red (+1, first)" if self.human_player == 1 else "Human: Blue (-1, second)"
        device = f"Device: {self.device_choice}"
        controls = [
            (side, lambda: setattr(self, "human_player", -self.human_player)),
            (device, self._cycle_device),
            (f"MCTS simulations: {self.mcts_sims}", self._cycle_sims),
            ("Refresh models", self.refresh_models),
            ("Start game", self._start_game),
            ("Exit", lambda: setattr(self, "running", False)),
        ]
        x = 70
        for label, callback in controls:
            button = pygame.Rect(x, y, max(130, min(230, 18 + len(label) * 10)), 42)
            pygame.draw.rect(screen, COLOR_GRID if label != "Start game" else COLOR_ACCENT, button, border_radius=7)
            screen.blit(small_font.render(label, True, COLOR_TEXT), (button.x + 10, button.y + 11))
            self._buttons.append((button, callback))
            x += button.width + 12
            if x + 180 > width:
                x = 70
                y += 52
        color = COLOR_ERROR if self.message.startswith(("No ", "Model load")) else COLOR_DIM
        screen.blit(small_font.render(self.message, True, color), (70, min(height - 60, y + 58)))

    def _draw_game(self, pygame, screen, title_font, font, small_font):
        self._buttons = []
        self._cells = []
        snapshot = self.controller.snapshot()
        width, height = screen.get_size()
        screen.blit(title_font.render("Connect4 3D v2.2", True, COLOR_TEXT), (30, 20))
        status = self._status_text(snapshot)
        screen.blit(font.render(status, True, COLOR_ERROR if snapshot["status"] == "error" else COLOR_TEXT), (350, 28))

        top = 90
        gap = 18
        panel_width = (width - 80 - 2 * gap) // 3
        panel_height = (height - 190 - gap) // 2
        cell_size = max(20, min((panel_width - 30) // 5, (panel_height - 45) // 5))
        valid = self.controller.game.get_valid_moves(snapshot["board"])
        last_coords = self.controller.game.action_to_coords(snapshot["last_action"]) if snapshot["last_action"] is not None else None
        for layer in range(6):
            panel_x = 30 + (layer % 3) * (panel_width + gap)
            panel_y = top + (layer // 3) * (panel_height + gap)
            panel = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            pygame.draw.rect(screen, COLOR_PANEL, panel, border_radius=10)
            screen.blit(small_font.render(f"Layer {layer + 1}", True, COLOR_TEXT), (panel.x + 12, panel.y + 8))
            origin_x = panel.x + (panel.width - cell_size * 5) // 2
            origin_y = panel.y + 34
            for row in range(5):
                for col in range(5):
                    rect = pygame.Rect(origin_x + col * cell_size, origin_y + row * cell_size, cell_size - 3, cell_size - 3)
                    action = self.controller.game.coords_to_action(layer, row, col)
                    fill = (45, 60, 80) if valid[action] else COLOR_GRID
                    pygame.draw.rect(screen, fill, rect, border_radius=4)
                    value = int(snapshot["board"][layer, row, col])
                    if value:
                        pygame.draw.circle(screen, COLOR_RED if value == 1 else COLOR_BLUE, rect.center, max(5, cell_size // 2 - 5))
                    if last_coords == (layer, row, col):
                        pygame.draw.rect(screen, (255, 220, 90), rect, 3, border_radius=4)
                    if valid[action] and snapshot["current_player"] == snapshot["human_player"] and not snapshot["ai_thinking"]:
                        self._cells.append((rect, (layer, row, col)))

        controls = [
            ("Restart", self.controller.reset),
            ("Back", self._back_to_launcher),
            ("Exit", lambda: setattr(self, "running", False)),
        ]
        x = 30
        y = height - 64
        for label, callback in controls:
            button = pygame.Rect(x, y, 120, 40)
            pygame.draw.rect(screen, COLOR_GRID, button, border_radius=7)
            screen.blit(small_font.render(label, True, COLOR_TEXT), (button.x + 26, button.y + 10))
            self._buttons.append((button, callback))
            x += 134

    def _status_text(self, snapshot):
        if snapshot["status"] == "error":
            return f"Error: {snapshot['error']}"
        if snapshot["status"] == "won":
            return "Red wins" if snapshot["winner"] == 1 else "Blue wins"
        if snapshot["status"] == "draw":
            return "Draw"
        if snapshot["ai_thinking"]:
            return "AI is thinking..."
        return "Red to move" if snapshot["current_player"] == 1 else "Blue to move"

    def _cycle_device(self):
        choices = ["auto", "cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        self.device_choice = choices[(choices.index(self.device_choice) + 1) % len(choices)]

    def _move_model_selection(self, delta):
        if self.models:
            self.selected_index = max(0, min(len(self.models) - 1, self.selected_index + int(delta)))

    def _cycle_sims(self):
        choices = [32, 64, 128, 256, 512]
        self.mcts_sims = choices[(choices.index(self.mcts_sims) + 1) % len(choices)]

    def _back_to_launcher(self):
        if self.ai_thread is not None and self.ai_thread.is_alive():
            self.message = "Wait for the current AI move before returning."
            return
        self._shutdown_controller()
        self.scene = "launcher"

    def _shutdown_controller(self):
        if self.ai_thread is not None:
            self.ai_thread.join()
            self.ai_thread = None
        if self.controller is not None:
            self.controller.close()
            self.controller = None


def default_model_roots(workspace_root):
    workspace_root = Path(workspace_root).resolve()
    candidates = [
        workspace_root / "save_model",
        workspace_root / "training" / "checkpoints",
        workspace_root / "distillation" / "checkpoints",
        workspace_root / "distillation" / "save_model",
    ]
    # Every synced V3 run materializes accepted champions under
    # training/runs/local_archive_validation/<run>/materialized/accepted
    # (B4/B6/B8 scales across runs).  Collect all of them so no accepted
    # champion is missed when a new run is synced back.
    archive_root = workspace_root / "training" / "runs" / "local_archive_validation"
    if archive_root.is_dir():
        for accepted_dir in sorted(archive_root.rglob("materialized/accepted")):
            if accepted_dir.is_dir() and accepted_dir not in candidates:
                candidates.append(accepted_dir)
    active_training_root = workspace_root.parent / "Connect4_3D_AI_v2.2"
    if active_training_root != workspace_root and active_training_root.is_dir():
        candidates.extend(
            [
                active_training_root / "save_model",
                active_training_root / "training" / "checkpoints",
                active_training_root / "distillation" / "checkpoints",
                active_training_root / "distillation" / "save_model",
            ]
        )
    return candidates
