"""Compatibility exports for the shared Connect4 rules implementation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connect4_core.game_rules import *  # noqa: F401,F403
