from __future__ import annotations

import threading
from dataclasses import dataclass

from .game import BLUE, DRAW, ONGOING, RED, Layer0State
from .solver import Analysis, StrongSolver


WINDOW = (1080, 720)
BOARD_ORIGIN = (70, 120)
CELL = 104
GRID = CELL * 5

BG = (8, 19, 33)
PANEL = (15, 35, 55)
CELL_BG = (25, 53, 76)
CELL_HOVER = (34, 74, 101)
TEXT = (222, 237, 247)
MUTED = (130, 158, 178)
RED_COLOR = (255, 88, 104)
BLUE_COLOR = (72, 143, 255)
ACCENT = (58, 214, 207)
WARNING = (255, 196, 84)


@dataclass(slots=True)
class PendingAnalysis:
    generation: int
    result: Analysis | None = None
    error: str | None = None
    running: bool = False
    failed: bool = False


class Layer0App:
    def __init__(self, pygame) -> None:
        self.pygame = pygame
        self.state = Layer0State()
        self.solver_color = BLUE
        self.solver = StrongSolver(seed=20260824, timeout=180.0)
        self.analysis: Analysis | None = None
        self.pending = PendingAnalysis(generation=0)
        self.generation = 0
        self.show_hints = True
        self.message = "点击空格落子；P 记录不可见高层落子"
        self.history: list[int | str] = []
        self.font = self._font(28)
        self.small = self._font(19)
        self.tiny = self._font(15)
        self.title = self._font(38, bold=True)

    def _font(self, size: int, *, bold: bool = False):
        return self.pygame.font.SysFont(
            ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Arial"],
            size,
            bold=bold,
        )

    def reset(self, *, swap: bool = False) -> None:
        if swap:
            self.solver_color = -self.solver_color
        self.state = Layer0State()
        self.history.clear()
        self.analysis = None
        self.message = "新对局"
        self.generation += 1
        self.pending = PendingAnalysis(generation=self.generation)

    def _start_analysis(self) -> None:
        if self.state.outcome() != ONGOING or self.pending.running or self.pending.failed:
            return
        if self.pending.generation == self.generation and self.pending.result is not None:
            self.analysis = self.pending.result
            return
        snapshot = self.state
        generation = self.generation
        self.pending = PendingAnalysis(generation=generation, running=True)

        def worker() -> None:
            try:
                result = self.solver.analyze(snapshot)
            except Exception as exc:  # UI must remain usable when an optional backend fails.
                if generation == self.generation:
                    self.pending.error = f"{type(exc).__name__}: {exc}"
                    self.pending.running = False
                    self.pending.failed = True
                return
            if generation == self.generation:
                self.pending.result = result
                self.pending.running = False

        threading.Thread(target=worker, name="layer0-analysis", daemon=True).start()

    def update(self) -> None:
        if self.pending.error:
            self.message = self.pending.error
            self.pending.error = None
        if not self.pending.running and self.pending.result is not None:
            self.analysis = self.pending.result
            if self.state.to_move == self.solver_color:
                move = self.analysis.principal_move
                if move is not None:
                    self._play(move, actor="Solver")
                    return
        self._start_analysis()

    def _play(self, position: int, *, actor: str) -> None:
        try:
            self.state = self.state.play(position)
        except ValueError as exc:
            self.message = str(exc)
            return
        self.history.append(position)
        self.message = f"{actor} 落子 {position}"
        self.generation += 1
        self.analysis = None
        self.pending = PendingAnalysis(generation=self.generation)

    def _pass(self) -> None:
        if self.state.outcome() != ONGOING:
            return
        self.state = self.state.pass_invisible()
        self.history.append("pass")
        self.message = "记录一次不可见的第二层/高层落子"
        self.generation += 1
        self.analysis = None
        self.pending = PendingAnalysis(generation=self.generation)

    def handle_event(self, event) -> bool:
        pygame = self.pygame
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_s:
                self.reset(swap=True)
            elif event.key == pygame.K_h:
                self.show_hints = not self.show_hints
            elif event.key == pygame.K_p:
                self._pass()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            if 70 <= x < 70 + GRID and 120 <= y < 120 + GRID:
                if self.state.to_move != self.solver_color and self.state.outcome() == ONGOING:
                    col = (x - 70) // CELL
                    row = (y - 120) // CELL
                    self._play(row * 5 + col + 1, actor="人类")
            elif 650 <= x <= 810 and 620 <= y <= 670:
                self.reset()
            elif 825 <= x <= 1010 and 620 <= y <= 670:
                self.reset(swap=True)
        return True

    def draw(self, screen) -> None:
        pygame = self.pygame
        screen.fill(BG)
        screen.blit(self.title.render("Layer0 Solver", True, TEXT), (70, 42))
        screen.blit(self.small.render("5×5 · 四连 · D4 对称", True, MUTED), (355, 57))

        mouse = pygame.mouse.get_pos()
        optimal = set(self.analysis.optimal_moves if self.analysis and self.show_hints else ())
        rows = self.state.rows()
        for row in range(5):
            for col in range(5):
                position = row * 5 + col + 1
                rect = pygame.Rect(70 + col * CELL + 4, 120 + row * CELL + 4, CELL - 8, CELL - 8)
                color = CELL_HOVER if rect.collidepoint(mouse) else CELL_BG
                pygame.draw.rect(screen, color, rect, border_radius=14)
                if position in optimal and rows[row][col] == 0:
                    pygame.draw.rect(screen, ACCENT, rect, width=3, border_radius=14)
                piece = rows[row][col]
                if piece:
                    pygame.draw.circle(
                        screen,
                        RED_COLOR if piece == RED else BLUE_COLOR,
                        rect.center,
                        34,
                    )
                    pygame.draw.circle(screen, (255, 255, 255), rect.center, 34, width=2)
                label = self.tiny.render(str(position), True, MUTED)
                screen.blit(label, (rect.x + 8, rect.y + 6))

        panel = pygame.Rect(620, 120, 390, 475)
        pygame.draw.rect(screen, PANEL, panel, border_radius=18)
        turn_name = "红方" if self.state.to_move == RED else "蓝方"
        turn_color = RED_COLOR if self.state.to_move == RED else BLUE_COLOR
        screen.blit(self.font.render(f"轮到 {turn_name}", True, turn_color), (650, 148))
        role = "Solver 执红" if self.solver_color == RED else "Solver 执蓝"
        screen.blit(self.small.render(role, True, TEXT), (650, 190))
        screen.blit(
            self.small.render(
                f"可见落子 {self.state.occupied.bit_count()} / 25   不可见 {self.state.invisible_turns}",
                True,
                MUTED,
            ),
            (650, 226),
        )

        y = 278
        outcome = self.state.outcome()
        if outcome != ONGOING:
            label = "平局" if outcome == DRAW else ("红方获胜" if outcome == RED else "蓝方获胜")
            color = WARNING if outcome == DRAW else RED_COLOR if outcome == RED else BLUE_COLOR
            screen.blit(self.font.render(label, True, color), (650, y))
        elif self.pending.running:
            screen.blit(self.font.render("Solver 思考中…", True, WARNING), (650, y))
        elif self.pending.failed:
            screen.blit(self.font.render("精确求解失败", True, RED_COLOR), (650, y))
            screen.blit(self.tiny.render("未使用估值结果，请重新开始", True, MUTED), (650, y + 44))
        elif self.analysis:
            proof = "已证明" if self.analysis.proven else "搜索估值"
            value = {1: "胜", 0: "和", -1: "负"}[self.analysis.outcome]
            screen.blit(self.font.render(f"{proof}：{value}", True, ACCENT), (650, y))
            screen.blit(
                self.small.render(
                    "最优格：" + ", ".join(map(str, self.analysis.optimal_moves)), True, TEXT
                ),
                (650, y + 44),
            )
            screen.blit(self.tiny.render(self.analysis.note, True, MUTED), (650, y + 80))
            screen.blit(
                self.tiny.render(
                    f"nodes {self.analysis.nodes:,} · cache hits {self.analysis.cache_hits:,}",
                    True,
                    MUTED,
                ),
                (650, y + 105),
            )

        history_text = ",".join(map(str, self.history[-12:])) or "—"
        screen.blit(self.small.render("最近行为", True, TEXT), (650, 445))
        screen.blit(self.tiny.render(history_text, True, MUTED), (650, 476))
        screen.blit(self.tiny.render("P: 高层落子   H: 提示   R: 重开   S: 换边", True, MUTED), (650, 540))

        self._button(screen, pygame.Rect(650, 620, 160, 50), "重新开始")
        self._button(screen, pygame.Rect(825, 620, 185, 50), "交换先后手")
        screen.blit(self.tiny.render(self.message, True, MUTED), (70, 665))

    def _button(self, screen, rect, label: str) -> None:
        pygame = self.pygame
        color = CELL_HOVER if rect.collidepoint(pygame.mouse.get_pos()) else PANEL
        pygame.draw.rect(screen, color, rect, border_radius=10)
        pygame.draw.rect(screen, ACCENT, rect, width=1, border_radius=10)
        text = self.small.render(label, True, TEXT)
        screen.blit(text, text.get_rect(center=rect.center))


def main() -> int:
    try:
        import pygame
    except ImportError:
        print("pygame 未安装。请先运行: python -m pip install -r requirements.txt")
        return 2
    pygame.init()
    pygame.display.set_caption("Connect4 3D - Layer0 Solver")
    screen = pygame.display.set_mode(WINDOW)
    clock = pygame.time.Clock()
    app = Layer0App(pygame)
    running = True
    while running:
        for event in pygame.event.get():
            running = app.handle_event(event)
            if not running:
                break
        app.update()
        app.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    app.solver.close(force=True)
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
