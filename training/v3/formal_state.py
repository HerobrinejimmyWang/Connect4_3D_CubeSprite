"""Persistent state transitions for the V3 formal generation loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Mapping

from .config import GateConfig


def _relative_artifact_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty run-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the run directory")
    return path.as_posix()


@dataclass(frozen=True)
class PendingCandidateState:
    candidate_model_id: str
    candidate_path: str
    incumbent_model_id: str
    gate_path: str
    opening_manifest: str
    pairs_evaluated: int
    max_pairs: int

    def __post_init__(self) -> None:
        if not self.candidate_model_id or not self.incumbent_model_id:
            raise ValueError("pending candidate and incumbent IDs must be non-empty")
        object.__setattr__(
            self,
            "candidate_path",
            _relative_artifact_path(self.candidate_path, "candidate_path"),
        )
        object.__setattr__(self, "gate_path", _relative_artifact_path(self.gate_path, "gate_path"))
        object.__setattr__(
            self,
            "opening_manifest",
            _relative_artifact_path(self.opening_manifest, "opening_manifest"),
        )
        if self.pairs_evaluated < 1 or self.max_pairs < self.pairs_evaluated:
            raise ValueError("pending gate pair counters are invalid")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PendingCandidateState":
        expected = {
            "candidate_model_id",
            "candidate_path",
            "incumbent_model_id",
            "gate_path",
            "opening_manifest",
            "pairs_evaluated",
            "max_pairs",
        }
        if set(raw) != expected:
            raise ValueError("pending candidate state has an unsupported schema")
        return cls(**dict(raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FormalLoopState:
    next_generation: int = 0
    next_game_id: int = 0
    replay_positions: int = 0
    train_positions_consumed: int = 0
    last_candidate_train_positions: int = 0
    accepted_model_id: str | None = None
    pending_candidate: PendingCandidateState | None = None
    exploration_stage_index: int = 0
    exploration_stage_started_generation: int = 0

    def __post_init__(self) -> None:
        counters = (
            self.next_generation,
            self.next_game_id,
            self.replay_positions,
            self.train_positions_consumed,
            self.last_candidate_train_positions,
            self.exploration_stage_index,
            self.exploration_stage_started_generation,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise ValueError("formal loop counters must be non-negative integers")
        if self.last_candidate_train_positions > self.train_positions_consumed:
            raise ValueError("last candidate cursor cannot exceed consumed train positions")
        if self.exploration_stage_started_generation > self.next_generation:
            raise ValueError("exploration stage cannot start after the next generation")
        if self.accepted_model_id is not None and not self.accepted_model_id:
            raise ValueError("accepted_model_id must be None or non-empty")
        if (
            self.pending_candidate is not None
            and self.pending_candidate.incumbent_model_id != (self.accepted_model_id or "random")
        ):
            raise ValueError("pending candidate incumbent differs from accepted model state")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FormalLoopState":
        legacy = {
            "next_generation",
            "next_game_id",
            "replay_positions",
            "train_positions_consumed",
            "last_candidate_train_positions",
            "accepted_model_id",
            "pending_candidate",
        }
        current = legacy | {
            "exploration_stage_index",
            "exploration_stage_started_generation",
        }
        if frozenset(raw) not in {frozenset(legacy), frozenset(current)}:
            raise ValueError("formal loop state has an unsupported schema")
        values = dict(raw)
        values.setdefault("exploration_stage_index", 0)
        values.setdefault("exploration_stage_started_generation", 0)
        pending = values["pending_candidate"]
        values["pending_candidate"] = (
            None if pending is None else PendingCandidateState.from_dict(pending)
        )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "pending_candidate": (
                None if self.pending_candidate is None else self.pending_candidate.to_dict()
            ),
        }

    def candidate_interval(self, gate: GateConfig) -> int:
        return (
            gate.bootstrap_candidate_train_positions
            if self.accepted_model_id is None
            else gate.candidate_train_positions
        )

    def candidate_due(self, gate: GateConfig) -> bool:
        if self.pending_candidate is not None:
            return False
        return (
            self.train_positions_consumed - self.last_candidate_train_positions
            >= self.candidate_interval(gate)
        )

    def finish_generation(
        self,
        *,
        next_game_id: int,
        replay_positions: int,
        train_positions_consumed: int,
    ) -> "FormalLoopState":
        return replace(
            self,
            next_generation=self.next_generation + 1,
            next_game_id=int(next_game_id),
            replay_positions=int(replay_positions),
            train_positions_consumed=int(train_positions_consumed),
        )

    def advance_exploration_stage(self) -> "FormalLoopState":
        return replace(
            self,
            exploration_stage_index=self.exploration_stage_index + 1,
            exploration_stage_started_generation=self.next_generation,
        )

    def emit_candidate(self, pending: PendingCandidateState) -> "FormalLoopState":
        if self.pending_candidate is not None:
            raise RuntimeError("cannot emit a new candidate while another gate is unresolved")
        if pending.incumbent_model_id != (self.accepted_model_id or "random"):
            raise ValueError("candidate incumbent does not match the accepted model")
        return replace(
            self,
            last_candidate_train_positions=self.train_positions_consumed,
            pending_candidate=pending,
        )

    def extend_pending_gate(self, pairs_evaluated: int) -> "FormalLoopState":
        if self.pending_candidate is None:
            raise RuntimeError("cannot extend a gate without a pending candidate")
        if not self.pending_candidate.pairs_evaluated < pairs_evaluated <= self.pending_candidate.max_pairs:
            raise ValueError("extended gate pairs must increase without exceeding max_pairs")
        return replace(
            self,
            pending_candidate=replace(
                self.pending_candidate,
                pairs_evaluated=int(pairs_evaluated),
            ),
        )

    def resolve_pending_candidate(self, *, accepted: bool) -> "FormalLoopState":
        if self.pending_candidate is None:
            raise RuntimeError("cannot resolve a missing pending candidate")
        accepted_model_id = (
            self.pending_candidate.candidate_model_id if accepted else self.accepted_model_id
        )
        return replace(
            self,
            accepted_model_id=accepted_model_id,
            pending_candidate=None,
        )


__all__ = ["FormalLoopState", "PendingCandidateState"]
