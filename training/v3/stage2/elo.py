"""Verify the immutable Anchored Elo v3 contract used by Stage 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..anchored_elo import canonical_anchored_config_hash, load_anchored_config, verify_anchor_files
from ..replay import sha256_file


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def verify_stage2_elo_protocol(
    protocol_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    raw = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    required = {
        "schema",
        "registry_path",
        "registry_hash",
        "pressure_registry_path",
        "pressure_registry_hash",
        "profiles",
    }
    if set(raw) != required or raw["schema"] != "connect4-v3-stage2-elo-protocol-v1":
        raise ValueError("unsupported Stage 2 Elo protocol")

    registry_path = _resolve(root, str(raw["registry_path"]))
    registry = load_anchored_config(registry_path)
    registry_hash = canonical_anchored_config_hash(registry)
    if registry_hash != raw["registry_hash"]:
        raise ValueError("Stage 2 Elo registry hash mismatch")
    anchors = verify_anchor_files(registry, root)

    pressure_path = _resolve(root, str(raw["pressure_registry_path"]))
    pressure = load_anchored_config(pressure_path)
    pressure_hash = canonical_anchored_config_hash(pressure)
    if pressure_hash != raw["pressure_registry_hash"]:
        raise ValueError("Stage 2 pressure registry hash mismatch")
    if tuple(anchor.anchor_id for anchor in pressure.anchors) != tuple(
        anchor.anchor_id for anchor in registry.anchors
    ):
        raise ValueError("Stage 2 symmetric and pressure registries use different anchors")

    verified_profiles: dict[str, Any] = {}
    expected_anchor_ids = {anchor.anchor_id for anchor in registry.anchors}
    profiles = raw["profiles"]
    if not isinstance(profiles, Mapping) or set(profiles) != {"primary_256", "final_512"}:
        raise ValueError("Stage 2 Elo protocol must bind primary_256 and final_512")
    for profile_id, row in profiles.items():
        if not isinstance(row, Mapping) or set(row) != {"scale_path", "scale_sha256"}:
            raise ValueError(f"invalid Stage 2 Elo profile row: {profile_id}")
        scale_path = _resolve(root, str(row["scale_path"]))
        if sha256_file(scale_path) != row["scale_sha256"]:
            raise ValueError(f"Stage 2 Elo scale checksum mismatch: {profile_id}")
        scale = json.loads(scale_path.read_text(encoding="utf-8"))
        if (
            scale.get("anchored_config_hash") != registry_hash
            or scale.get("profile_id") != profile_id
            or scale.get("frozen") is not True
            or set(scale.get("ratings", {})) != expected_anchor_ids
        ):
            raise ValueError(f"Stage 2 Elo scale contract mismatch: {profile_id}")
        verified_profiles[profile_id] = {
            "scale_path": str(scale_path),
            "scale_sha256": str(row["scale_sha256"]),
            "ratings": dict(scale["ratings"]),
        }
    return {
        "schema": "connect4-v3-stage2-elo-verification-v1",
        "verified": True,
        "registry_path": str(registry_path),
        "registry_hash": registry_hash,
        "pressure_registry_hash": pressure_hash,
        "anchor_count": len(anchors),
        "anchor_ids": [anchor["model_id"] for anchor in anchors],
        "profiles": verified_profiles,
    }


__all__ = ["verify_stage2_elo_protocol"]
