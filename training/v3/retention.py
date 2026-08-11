"""Pure, non-destructive retention planning for V3 run artifacts.

This module never removes, moves, or uploads files.  It accepts catalog facts
and verified archive receipts, then reports which artifacts would be safe to
prune under a policy.  Deletion belongs to a separate, explicitly invoked
command that must revalidate the resulting plan.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Sequence


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIB = 1024**3


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact path must be a non-empty string")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact paths must be run-relative and cannot contain '..'")
    return path.as_posix()


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class RetentionArtifact:
    path: str
    kind: str
    size_bytes: int
    checksum_sha256: str
    sequence: int
    prunable: bool = False
    pinned: bool = False
    position_start: int | None = None
    position_end: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("artifact kind must be a non-empty string")
        if isinstance(self.size_bytes, bool) or int(self.size_bytes) < 0:
            raise ValueError("artifact size_bytes cannot be negative")
        if isinstance(self.sequence, bool) or int(self.sequence) < 0:
            raise ValueError("artifact sequence cannot be negative")
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(
            self,
            "checksum_sha256",
            _sha256(self.checksum_sha256, "artifact checksum_sha256"),
        )
        bounds = (self.position_start, self.position_end)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("position_start and position_end must be provided together")
        if bounds[0] is not None:
            start, end = int(bounds[0]), int(bounds[1])
            if start < 0 or end <= start:
                raise ValueError("artifact position range must be non-empty and non-negative")
            object.__setattr__(self, "position_start", start)
            object.__setattr__(self, "position_end", end)
        if self.kind == "raw_replay" and bounds[0] is None:
            raise ValueError("raw_replay artifacts require a position range")


@dataclass(frozen=True)
class ReceiptEntry:
    path: str
    size_bytes: int
    checksum_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if isinstance(self.size_bytes, bool) or int(self.size_bytes) < 0:
            raise ValueError("receipt size_bytes cannot be negative")
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        object.__setattr__(
            self,
            "checksum_sha256",
            _sha256(self.checksum_sha256, "receipt checksum_sha256"),
        )


@dataclass(frozen=True)
class ArchiveReceipt:
    receipt_id: str
    archive_manifest_sha256: str
    verified: bool
    entries: tuple[ReceiptEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_id, str) or not self.receipt_id.strip():
            raise ValueError("receipt_id must be non-empty")
        object.__setattr__(
            self,
            "archive_manifest_sha256",
            _sha256(self.archive_manifest_sha256, "archive manifest checksum"),
        )
        if type(self.verified) is not bool:
            raise TypeError("receipt verified must be a boolean")
        entries = tuple(self.entries)
        paths = [entry.path for entry in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("archive receipt contains duplicate artifact paths")
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True)
class RetentionPolicy:
    active_window_margin: float = 1.25
    keep_recent_by_kind: tuple[tuple[str, int], ...] = (
        ("raw_replay", 4),
        ("checkpoint", 3),
        ("accepted", 2),
        ("rejected", 1),
    )
    soft_used_fraction: float = 0.80
    hard_free_bytes: int = 20 * GIB

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.active_window_margin)) or self.active_window_margin < 1.0:
            raise ValueError("active_window_margin must be finite and at least 1.0")
        if not 0.0 < float(self.soft_used_fraction) < 1.0:
            raise ValueError("soft_used_fraction must be in (0, 1)")
        if isinstance(self.hard_free_bytes, bool) or int(self.hard_free_bytes) < 0:
            raise ValueError("hard_free_bytes cannot be negative")
        pairs = tuple((str(kind), int(count)) for kind, count in self.keep_recent_by_kind)
        kinds = [kind for kind, _count in pairs]
        if any(not kind for kind in kinds) or len(kinds) != len(set(kinds)):
            raise ValueError("keep_recent_by_kind requires unique non-empty kinds")
        if any(count < 0 for _kind, count in pairs):
            raise ValueError("recent retention counts cannot be negative")
        object.__setattr__(self, "keep_recent_by_kind", pairs)
        object.__setattr__(self, "hard_free_bytes", int(self.hard_free_bytes))

    def recent_count(self, kind: str) -> int:
        return dict(self.keep_recent_by_kind).get(kind, 0)


@dataclass(frozen=True)
class DiskEstimate:
    artifact_bytes: int
    other_used_bytes: int
    total_used_bytes: int
    by_kind: Mapping[str, int]
    capacity_bytes: int | None
    free_bytes: int | None
    used_fraction: float | None


@dataclass(frozen=True)
class RetentionDecision:
    artifact: RetentionArtifact
    action: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RetentionPlan:
    decisions: tuple[RetentionDecision, ...]
    current: DiskEstimate
    projected: DiskEstimate
    eligible_bytes: int
    expanded_window_start: int
    active_window_end: int
    soft_limit_exceeded: bool
    hard_reserve_breached: bool

    @property
    def eligible_paths(self) -> tuple[str, ...]:
        return tuple(
            decision.artifact.path
            for decision in self.decisions
            if decision.action == "eligible_to_prune"
        )

    @property
    def kept_paths(self) -> tuple[str, ...]:
        return tuple(
            decision.artifact.path
            for decision in self.decisions
            if decision.action == "keep"
        )


def estimate_disk_usage(
    artifacts: Iterable[RetentionArtifact],
    *,
    capacity_bytes: int | None = None,
    other_used_bytes: int = 0,
) -> DiskEstimate:
    rows = tuple(artifacts)
    if isinstance(other_used_bytes, bool) or int(other_used_bytes) < 0:
        raise ValueError("other_used_bytes cannot be negative")
    other = int(other_used_bytes)
    if capacity_bytes is not None:
        if isinstance(capacity_bytes, bool) or int(capacity_bytes) <= 0:
            raise ValueError("capacity_bytes must be positive")
        capacity = int(capacity_bytes)
    else:
        capacity = None
    by_kind: dict[str, int] = {}
    for artifact in rows:
        by_kind[artifact.kind] = by_kind.get(artifact.kind, 0) + artifact.size_bytes
    artifact_bytes = sum(by_kind.values())
    used = artifact_bytes + other
    free = None if capacity is None else max(0, capacity - used)
    fraction = None if capacity is None else used / float(capacity)
    return DiskEstimate(
        artifact_bytes=artifact_bytes,
        other_used_bytes=other,
        total_used_bytes=used,
        by_kind=dict(sorted(by_kind.items())),
        capacity_bytes=capacity,
        free_bytes=free,
        used_fraction=fraction,
    )


def _verified_entries(receipts: Sequence[ArchiveReceipt]) -> set[tuple[str, int, str]]:
    return {
        (entry.path, entry.size_bytes, entry.checksum_sha256)
        for receipt in receipts
        if receipt.verified
        for entry in receipt.entries
    }


def _intersects(artifact: RetentionArtifact, start: int, end: int) -> bool:
    if artifact.position_start is None or artifact.position_end is None:
        return False
    return artifact.position_start < end and artifact.position_end > start


def plan_retention(
    artifacts: Iterable[RetentionArtifact],
    policy: RetentionPolicy,
    *,
    active_window_start: int,
    active_window_end: int,
    receipts: Sequence[ArchiveReceipt] = (),
    capacity_bytes: int | None = None,
    other_used_bytes: int = 0,
) -> RetentionPlan:
    """Return a deterministic prune plan without touching the filesystem."""

    rows = tuple(artifacts)
    paths = [artifact.path for artifact in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("retention catalog contains duplicate paths")
    start, end = int(active_window_start), int(active_window_end)
    if start < 0 or end < start:
        raise ValueError("active replay window bounds are invalid")
    window_positions = end - start
    expanded_positions = int(math.ceil(window_positions * policy.active_window_margin))
    expanded_start = max(0, end - expanded_positions)

    recent_paths: set[str] = set()
    kinds = {artifact.kind for artifact in rows}
    for kind in kinds:
        count = policy.recent_count(kind)
        if count <= 0:
            continue
        candidates = sorted(
            (artifact for artifact in rows if artifact.kind == kind),
            key=lambda artifact: (artifact.sequence, artifact.path),
            reverse=True,
        )
        recent_paths.update(artifact.path for artifact in candidates[:count])

    verified = _verified_entries(tuple(receipts))
    decisions: list[RetentionDecision] = []
    for artifact in sorted(rows, key=lambda item: item.path):
        reasons: list[str] = []
        if artifact.pinned:
            reasons.append("pinned")
        if not artifact.prunable:
            reasons.append("not_marked_prunable")
        if artifact.path in recent_paths:
            reasons.append(f"recent:{artifact.kind}")
        if artifact.kind == "raw_replay" and _intersects(artifact, expanded_start, end):
            reasons.append("active_window_margin")
        if (
            artifact.kind == "raw_replay"
            and artifact.position_start is not None
            and artifact.position_start >= end
        ):
            # A producer may have committed shards that the learner cursor has
            # not consumed yet.  Archived is not synonymous with consumed.
            reasons.append("beyond_active_cursor")
        receipt_key = (artifact.path, artifact.size_bytes, artifact.checksum_sha256)
        if receipt_key not in verified:
            reasons.append("verified_archive_receipt_missing_or_mismatched")
        action = "keep" if reasons else "eligible_to_prune"
        if action == "eligible_to_prune":
            reasons.append("archived_and_outside_retention")
        decisions.append(RetentionDecision(artifact, action, tuple(reasons)))

    eligible_bytes = sum(
        decision.artifact.size_bytes
        for decision in decisions
        if decision.action == "eligible_to_prune"
    )
    current = estimate_disk_usage(
        rows,
        capacity_bytes=capacity_bytes,
        other_used_bytes=other_used_bytes,
    )
    projected_rows = tuple(
        decision.artifact for decision in decisions if decision.action != "eligible_to_prune"
    )
    projected = estimate_disk_usage(
        projected_rows,
        capacity_bytes=capacity_bytes,
        other_used_bytes=other_used_bytes,
    )
    soft_exceeded = (
        current.used_fraction is not None
        and current.used_fraction >= policy.soft_used_fraction
    )
    hard_breached = (
        current.free_bytes is not None
        and current.free_bytes <= policy.hard_free_bytes
    )
    return RetentionPlan(
        decisions=tuple(decisions),
        current=current,
        projected=projected,
        eligible_bytes=eligible_bytes,
        expanded_window_start=expanded_start,
        active_window_end=end,
        soft_limit_exceeded=soft_exceeded,
        hard_reserve_breached=hard_breached,
    )


__all__ = [
    "ArchiveReceipt",
    "DiskEstimate",
    "GIB",
    "ReceiptEntry",
    "RetentionArtifact",
    "RetentionDecision",
    "RetentionPlan",
    "RetentionPolicy",
    "estimate_disk_usage",
    "plan_retention",
]
