"""No-clobber coordinator lock and pre-commit generation journal.

This module is intentionally non-destructive.  Reconciliation reports whether
an interrupted draft is resumable or needs operator attention; it never deletes
or silently adopts partial artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .layout import RunLayout


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _run_relative(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty run-relative path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the run directory")
    return path.as_posix()


@dataclass(frozen=True)
class DraftArtifact:
    path: str
    checksum_sha256: str
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _run_relative(self.path, "draft artifact path"))
        if (
            not isinstance(self.checksum_sha256, str)
            or len(self.checksum_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checksum_sha256)
        ):
            raise ValueError("draft artifact checksum must be lowercase SHA-256")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("draft artifact kind must be non-empty")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DraftArtifact":
        if set(raw) != {"path", "checksum_sha256", "kind"}:
            raise ValueError("draft artifact schema mismatch")
        return cls(**dict(raw))


@dataclass(frozen=True)
class GenerationDraft:
    run_id: str
    generation: int
    config_hash: str
    phase: str
    artifacts: tuple[DraftArtifact, ...]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if not self.run_id or self.generation < 0:
            raise ValueError("draft run_id/generation are invalid")
        if len(self.config_hash) != 64:
            raise ValueError("draft config_hash must be SHA-256")
        if self.phase not in {"started", "artifacts_ready"}:
            raise ValueError("draft phase is unsupported")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("draft contains duplicate artifact paths")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GenerationDraft":
        expected = {
            "schema_version",
            "run_id",
            "generation",
            "config_hash",
            "phase",
            "artifacts",
            "created_at",
            "updated_at",
        }
        if set(raw) != expected or raw["schema_version"] != 1:
            raise ValueError("generation draft schema mismatch")
        values = dict(raw)
        values.pop("schema_version")
        values["artifacts"] = tuple(DraftArtifact.from_dict(row) for row in values["artifacts"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}


class CoordinatorLock:
    """One-process run lock with explicit ownership and no stale auto-break."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.nonce = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> "CoordinatorLock":
        if self.acquired:
            raise RuntimeError("coordinator lock is already acquired by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "nonce": self.nonce,
            "acquired_at": _utc_now(),
        }
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"coordinator lock already exists: {self.path}; inspect it explicitly before recovery"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self.acquired = True
        return self

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot verify coordinator lock ownership during release") from exc
        if payload.get("nonce") != self.nonce or payload.get("run_id") != self.run_id:
            raise RuntimeError("coordinator lock ownership changed; refusing to remove it")
        self.path.unlink()
        self.acquired = False

    def __enter__(self) -> "CoordinatorLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class GenerationJournal:
    def __init__(self, layout: RunLayout, draft: GenerationDraft) -> None:
        self.layout = layout
        self.draft = draft
        self.path = layout.generation_drafts / f"g{draft.generation:06d}.json"

    @classmethod
    def begin(
        cls,
        layout: RunLayout,
        *,
        run_id: str,
        generation: int,
        config_hash: str,
    ) -> "GenerationJournal":
        layout.create()
        path = layout.generation_drafts / f"g{generation:06d}.json"
        if path.exists() or (layout.generation_commits / path.name).exists():
            raise FileExistsError(f"generation {generation} already has a draft or commit")
        now = _utc_now()
        draft = GenerationDraft(run_id, generation, config_hash, "started", (), now, now)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(draft.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return cls(layout, draft)

    @classmethod
    def load(cls, layout: RunLayout, generation: int) -> "GenerationJournal":
        path = layout.generation_drafts / f"g{generation:06d}.json"
        return cls(layout, GenerationDraft.from_dict(json.loads(path.read_text(encoding="utf-8"))))

    def record_artifact(self, path: str | Path, *, kind: str) -> DraftArtifact:
        artifact_path = Path(path).resolve()
        if not artifact_path.is_file() or not artifact_path.is_relative_to(self.layout.root):
            raise ValueError("journal artifacts must be existing files inside the run")
        relative = artifact_path.relative_to(self.layout.root).as_posix()
        if any(row.path == relative for row in self.draft.artifacts):
            raise ValueError("generation draft already records this artifact")
        artifact = DraftArtifact(relative, _sha256_file(artifact_path), kind)
        self.draft = replace(
            self.draft,
            artifacts=(*self.draft.artifacts, artifact),
            updated_at=_utc_now(),
        )
        _atomic_json(self.path, self.draft.to_dict())
        return artifact

    def mark_artifacts_ready(self) -> None:
        if not self.draft.artifacts:
            raise RuntimeError("cannot mark an empty generation draft ready")
        self.draft = replace(self.draft, phase="artifacts_ready", updated_at=_utc_now())
        _atomic_json(self.path, self.draft.to_dict())

    def stage_commit(self, payload: Mapping[str, Any]) -> Path:
        """Persist the generation commit payload before publication.

        Moving this checksum-bound file into ``manifests/generations`` is the
        transaction's final publication step.  A crash after staging can be
        completed by loading the ready journal and calling ``publish_commit``.
        """

        if self.draft.phase != "started":
            raise RuntimeError("generation commit can only be staged from a started draft")
        if payload.get("run_id") != self.draft.run_id:
            raise ValueError("generation commit run_id differs from its draft")
        if int(payload.get("generation", -1)) != self.draft.generation:
            raise ValueError("generation commit generation differs from its draft")
        if payload.get("config_hash") != self.draft.config_hash:
            raise ValueError("generation commit config hash differs from its draft")
        staged = (
            self.layout.generation_drafts
            / "commit_payloads"
            / f"g{self.draft.generation:06d}.json"
        )
        if staged.exists():
            raise FileExistsError(f"staged generation commit already exists: {staged}")
        _atomic_json(staged, payload)
        self.record_artifact(staged, kind="generation_commit_payload")
        self.mark_artifacts_ready()
        return staged

    def publish_commit(self) -> Path:
        """Publish a staged commit and its latest pointer atomically per file."""

        if self.draft.phase != "artifacts_ready":
            raise RuntimeError("generation artifacts are not ready for commit publication")
        staged_rows = [
            artifact
            for artifact in self.draft.artifacts
            if artifact.kind == "generation_commit_payload"
        ]
        if len(staged_rows) != 1:
            raise RuntimeError("ready generation draft needs exactly one staged commit payload")
        staged = self.layout.root / staged_rows[0].path
        if not staged.is_file() or _sha256_file(staged) != staged_rows[0].checksum_sha256:
            raise RuntimeError("staged generation commit is missing or changed")
        payload = json.loads(staged.read_text(encoding="utf-8"))
        if (
            payload.get("run_id") != self.draft.run_id
            or int(payload.get("generation", -1)) != self.draft.generation
            or payload.get("config_hash") != self.draft.config_hash
        ):
            raise RuntimeError("staged generation commit identity differs from its draft")
        target = self.layout.generation_commits / f"g{self.draft.generation:06d}.json"
        if target.exists():
            raise FileExistsError(f"immutable generation commit already exists: {target}")
        os.replace(staged, target)
        if os.name != "nt":
            descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _atomic_json(
            self.layout.manifests / "latest_generation.json",
            {
                "schema_version": 1,
                "generation": self.draft.generation,
                "commit": target.relative_to(self.layout.root).as_posix(),
                "commit_sha256": _sha256_file(target),
            },
        )
        return target


def reconcile_generation_drafts(layout: RunLayout) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    for path in sorted(layout.generation_drafts.glob("g*.json")):
        try:
            draft = GenerationDraft.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            results.append({"draft": str(path), "status": "blocked_invalid_draft", "error": str(exc)})
            continue
        commit = layout.generation_commits / f"g{draft.generation:06d}.json"
        if commit.is_file():
            results.append({"generation": draft.generation, "status": "committed", "draft": str(path), "commit": str(commit)})
            continue
        failures: list[dict[str, str]] = []
        for artifact in draft.artifacts:
            artifact_path = layout.root / artifact.path
            if not artifact_path.is_file():
                failures.append({"path": artifact.path, "reason": "missing"})
            elif _sha256_file(artifact_path) != artifact.checksum_sha256:
                failures.append({"path": artifact.path, "reason": "checksum_mismatch"})
        if failures:
            status = "blocked_partial_artifacts"
        elif draft.phase == "artifacts_ready":
            status = "resume_precommit"
        else:
            status = "resume_generation"
        results.append(
            {
                "generation": draft.generation,
                "status": status,
                "draft": str(path),
                "artifacts": [asdict(artifact) for artifact in draft.artifacts],
                "failures": failures,
            }
        )
    return tuple(results)


__all__ = [
    "CoordinatorLock",
    "DraftArtifact",
    "GenerationDraft",
    "GenerationJournal",
    "reconcile_generation_drafts",
]
