"""Compatibility adapter for the shared explicit-root model registry."""

from pathlib import Path

from connect4_runtime.model_registry import ModelRegistry


def discover_models(search_root=None):
    if search_root is None:
        return []
    items = ModelRegistry([Path(search_root)], settle_seconds=0).discover()
    return [
        {"label": item.label, "path": str(item.path), "relative_path": str(item.path.relative_to(item.root))}
        for item in items
    ]
