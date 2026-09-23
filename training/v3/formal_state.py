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
    opening_temperature_mixture_start_game_id: int | None = None
    rule_peak_model_ids: tuple[str, ...] = ()
    rule_regression_streaks: tuple[int, ...] = ()

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
            self.opening_temperature_mixture_start_game_id is not None
            and (
                isinstance(self.opening_temperature_mixture_start_game_id, bool)
                or not isinstance(self.opening_temperature_mixture_start_game_id, int)
                or self.opening_temperature_mixture_start_game_id < 0
                or self.opening_temperature_mixture_start_game_id > self.next_game_id
            )
        ):
            raise ValueError("opening mixture game boundary is invalid")
        if (
            self.pending_candidate is not None
            and self.pending_candidate.incumbent_model_id != (self.accepted_model_id or "random")
        ):
            raise ValueError("pending candidate incumbent differs from accepted model state")
        if self.rule_peak_model_ids or self.rule_regression_streaks:
            if len(self.rule_peak_model_ids) != 5 or len(self.rule_regression_streaks) != 5:
                raise ValueError("multi-rule peak state must contain five aligned rules")
            if any(not model_id for model_id in self.rule_peak_model_ids):
                raise ValueError("multi-rule peak model IDs must be non-empty")
            if any(type(streak) is not int or streak < 0 for streak in self.rule_regression_streaks):
                raise ValueError("multi-rule regression streaks must be non-negative integers")

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
        delayed_mixture = current | {"opening_temperature_mixture_start_game_id"}
        multirule = delayed_mixture | {"rule_peak_model_ids", "rule_regression_streaks"}
        keys = set(raw)
        if not legacy.issubset(keys) or not keys.issubset(multirule):
            raise ValueError("formal loop state has an unsupported schema")
        values = dict(raw)
        values.setdefault("exploration_stage_index", 0)
        values.setdefault("exploration_stage_started_generation", 0)
        values.setdefault("opening_temperature_mixture_start_game_id", None)
        values.setdefault("rule_peak_model_ids", ())
        values.setdefault("rule_regression_streaks", ())
        values["rule_peak_model_ids"] = tuple(values["rule_peak_model_ids"])
        values["rule_regression_streaks"] = tuple(values["rule_regression_streaks"])
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

    def start_opening_temperature_mixture(self, game_id: int) -> "FormalLoopState":
        if self.opening_temperature_mixture_start_game_id is not None:
            raise RuntimeError("opening temperature mixture already started")
        if game_id < 0 or game_id > self.next_game_id:
            raise ValueError("opening temperature mixture game boundary is invalid")
        return replace(self, opening_temperature_mixture_start_game_id=int(game_id))

    def initialize_rule_peaks(self) -> "FormalLoopState":
        if self.rule_peak_model_ids:
            return self
        if not self.accepted_model_id:
            raise ValueError("multi-rule peaks need a committed accepted model")
        return replace(
            self,
            rule_peak_model_ids=(self.accepted_model_id,) * 5,
            rule_regression_streaks=(0,) * 5,
        )

    def advance_rule_peaks(
        self, candidate_model_id: str, point_scores: tuple[float, ...]
    ) -> "FormalLoopState":
        if len(point_scores) != 5 or len(self.rule_peak_model_ids) != 5:
            raise ValueError("rule peak update needs five scores and initialized state")
        if not candidate_model_id:
            raise ValueError("candidate model ID must be non-empty")
        models = tuple(
            candidate_model_id if score >= 0.5 else peak
            for peak, score in zip(self.rule_peak_model_ids, point_scores, strict=True)
        )
        streaks = tuple(
            0 if score >= 0.5 else streak + 1
            for streak, score in zip(self.rule_regression_streaks, point_scores, strict=True)
        )
        return replace(self, rule_peak_model_ids=models, rule_regression_streaks=streaks)

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
