from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


MODEL_SUFFIXES = (".pth", ".pth.tar", ".pt", ".ckpt")
SKIPPED_PARTS = {".git", ".venv", "__pycache__", ".cache", ".tmp", "history"}
TEMP_MARKERS = (".tmp", ".partial", ".part", "~")


@dataclass(frozen=True)
class ModelInfo:
    label: str
    path: Path
    root: Path
    size: int
    modified_at: float


class ModelRegistry:
    """Discover completed model files only under explicitly configured roots."""

    def __init__(self, roots, settle_seconds=2.0):
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots if root)
        self.settle_seconds = max(0.0, float(settle_seconds))

    def discover(self):
        now = time.time()
        discovered = []
        seen = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or any(part.lower() in SKIPPED_PARTS for part in path.parts):
                    continue
                lower_name = path.name.lower()
                if not lower_name.endswith(MODEL_SUFFIXES) or any(marker in lower_name for marker in TEMP_MARKERS):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size <= 0 or now - stat.st_mtime < self.settle_seconds:
                    continue
                resolved = path.resolve()
                key = str(resolved).casefold()
                if key in seen:
                    continue
                seen.add(key)
                relative = resolved.relative_to(root)
                discovered.append(
                    ModelInfo(
                        label=f"{root.name}/{relative.as_posix()}",
                        path=resolved,
                        root=root,
                        size=int(stat.st_size),
                        modified_at=float(stat.st_mtime),
                    )
                )
        return sorted(discovered, key=lambda item: (-item.modified_at, item.label.casefold()))
