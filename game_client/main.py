from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from game_client.app import GameClientApp, default_model_roots


def parse_args():
    parser = argparse.ArgumentParser(description="Connect4 3D v2.2 human-versus-AI client")
    parser.add_argument("--model-root", action="append", default=[], help="model directory; may be repeated")
    return parser.parse_args()


def main():
    args = parse_args()
    roots = args.model_root or default_model_roots(WORKSPACE_ROOT)
    GameClientApp(roots).run()


if __name__ == "__main__":
    main()
