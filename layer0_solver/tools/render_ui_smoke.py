"""Render one pygame frame with SDL's dummy driver for CI/smoke verification."""

from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from layer0_solver.ui import WINDOW, Layer0App


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    app = Layer0App(pygame)
    app.draw(screen)
    output = Path(__file__).resolve().parents[1] / "evidence" / "ui_smoke.png"
    pygame.image.save(screen, output)
    app.solver.close(force=True)
    pygame.quit()
    print(output)


if __name__ == "__main__":
    main()
