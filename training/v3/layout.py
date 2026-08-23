"""Filesystem layout for an isolated V3 training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path
    manifests: Path
    raw_replay: Path
    shuffle: Path
    metrics: Path
    checkpoints: Path
    candidates: Path
    accepted: Path
    rejected: Path
    samples: Path
    generation_commits: Path
    generation_drafts: Path
    coordinator_lock: Path
    archive_staging: Path
    archive_receipts: Path
    resolved_config: Path
    run_manifest: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "RunLayout":
        root_path = Path(root).resolve()
        replay_root = root_path / "replay"
        return cls(
            root=root_path,
            manifests=root_path / "manifests",
            raw_replay=replay_root / "raw",
            shuffle=replay_root / "shuffle",
            metrics=root_path / "metrics",
            checkpoints=root_path / "checkpoints",
            candidates=root_path / "candidates",
            accepted=root_path / "accepted",
            rejected=root_path / "rejected",
            samples=root_path / "samples",
            generation_commits=root_path / "manifests" / "generations",
            generation_drafts=root_path / "manifests" / "generation_drafts",
            coordinator_lock=root_path / "manifests" / "coordinator.lock",
            archive_staging=root_path / "archive_staging",
            archive_receipts=root_path / "archive_receipts",
            resolved_config=root_path / "resolved_config.json",
            run_manifest=root_path / "run_manifest.json",
        )

    def create(self) -> "RunLayout":
        for directory in (
            self.root,
            self.manifests,
            self.raw_replay,
            self.shuffle,
            self.metrics,
            self.checkpoints,
            self.candidates,
            self.accepted,
            self.rejected,
            self.samples,
            self.generation_commits,
            self.generation_drafts,
            self.archive_staging,
            self.archive_receipts,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self
