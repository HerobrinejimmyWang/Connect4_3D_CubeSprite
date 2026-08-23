from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v3.evaluation_snapshot import export_evaluation_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project a formal V3 checkpoint into an immutable evaluation-only model artifact."
    )
    parser.add_argument("--checkpoint", required=True, help="Formal V3 checkpoint path")
    parser.add_argument("--output", required=True, help="New evaluation snapshot path")
    parser.add_argument("--model-id", help="Optional stable evaluation model identifier")
    args = parser.parse_args()
    result = export_evaluation_snapshot(
        args.checkpoint,
        args.output,
        model_id=args.model_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
