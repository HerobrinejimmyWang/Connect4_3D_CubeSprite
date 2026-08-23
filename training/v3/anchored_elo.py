"""Historical-anchor evaluation on the common V3 rules and MCTS kernel.

Legacy checkpoints are external opponents only. They never become V3 lineage
artifacts and their games never enter self-play replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from connect4_core.rules import CLASSIC_RULE, DEFAULT_RULE_REGISTRY

from .config import ModelConfig
from .evaluation import Opening, load_opening_manifest, play_paired_openings
from .model import TorchPredictor, build_model, legacy_policy_to_columns
from .replay import sha256_file
from .search import Predictor


def _portable_text_sha256(path: Path) -> str:
    """Hash tracked text independently of Git checkout newline policy."""

    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


ANCHORED_CONFIG_SCHEMA_VERSION = 1
MATCH_BATCH_SCHEMA_VERSION = 1
ANCHOR_SCALE_SCHEMA_VERSION = 1
ANCHORED_REPORT_SCHEMA_VERSION = 1
ELO_SCALE = 400.0 / math.log(10.0)


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from exc
    if value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the repository")
    return path.as_posix()


@dataclass(frozen=True)
class AnchorSpec:
    anchor_id: str
    label: str
    path: str
    checksum_sha256: str

    def __post_init__(self) -> None:
        if not self.anchor_id or not self.anchor_id.isascii():
            raise ValueError("anchor_id must be non-empty ASCII")
        if not self.label:
            raise ValueError("anchor label must not be empty")
        object.__setattr__(self, "path", _relative_path(self.path, "anchor path"))
        object.__setattr__(
            self,
            "checksum_sha256",
            _sha256(self.checksum_sha256, "anchor checksum_sha256"),
        )


@dataclass(frozen=True)
class EvaluationProfile:
    profile_id: str
    search_sims: int
    cpuct: float
    initial_pairs: int
    pair_increment: int
    max_pairs: int
    milestones: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_id.isascii():
            raise ValueError("profile_id must be non-empty ASCII")
        if self.search_sims < 1 or self.cpuct <= 0.0:
            raise ValueError("profile search_sims and cpuct must be positive")
        if self.initial_pairs < 1 or self.pair_increment < 1:
            raise ValueError("profile pair counts must be positive")
        if self.max_pairs < self.initial_pairs:
            raise ValueError("profile max_pairs must cover initial_pairs")
        if (self.max_pairs - self.initial_pairs) % self.pair_increment:
            raise ValueError("profile pair range must divide exactly by pair_increment")
        milestones = tuple(self.milestones)
        if not milestones or len(set(milestones)) != len(milestones):
            raise ValueError("profile milestones must be non-empty and unique")
        object.__setattr__(self, "milestones", milestones)


@dataclass(frozen=True)
class OpeningSuiteSpec:
    manifest_path: str
    checksum_sha256: str
    count: int
    seed: int
    prefix_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_path",
            _relative_path(self.manifest_path, "opening manifest_path"),
        )
        object.__setattr__(
            self,
            "checksum_sha256",
            _sha256(self.checksum_sha256, "opening checksum_sha256"),
        )
        if self.count < 1 or self.seed < 0:
            raise ValueError("opening count must be positive and seed non-negative")
        lengths = tuple(self.prefix_lengths)
        if not lengths or lengths[0] != 0 or tuple(sorted(set(lengths))) != lengths:
            raise ValueError("opening prefix_lengths must increase strictly from zero")
        object.__setattr__(self, "prefix_lengths", lengths)


@dataclass(frozen=True)
class AnchoredStatisticsConfig:
    confidence: float
    saturation_score: float
    saturation_lower_bound: float
    sentinel_pairs: int
    rating_prior_sigma: float
    max_elo_interval_width: float

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("statistics confidence must be in (0.5, 1)")
        if not 0.5 < self.saturation_score <= 1.0:
            raise ValueError("saturation_score must be in (0.5, 1]")
        if not 0.5 < self.saturation_lower_bound <= self.saturation_score:
            raise ValueError("saturation_lower_bound must be in (0.5, saturation_score]")
        if (
            self.sentinel_pairs < 1
            or self.rating_prior_sigma <= 0.0
            or self.max_elo_interval_width <= 0.0
        ):
            raise ValueError("sentinel_pairs, rating prior, and Elo interval width must be positive")


@dataclass(frozen=True)
class AnchoredEloConfig:
    reference_anchor_id: str
    rule_id: str
    anchors: tuple[AnchorSpec, ...]
    profiles: tuple[EvaluationProfile, ...]
    openings: OpeningSuiteSpec
    statistics: AnchoredStatisticsConfig

    def __post_init__(self) -> None:
        if self.rule_id != CLASSIC_RULE.rule_id:
            raise ValueError("historical anchored Elo v1 supports classic only")
        anchors = tuple(self.anchors)
        profiles = tuple(self.profiles)
        anchor_ids = [item.anchor_id for item in anchors]
        profile_ids = [item.profile_id for item in profiles]
        if len(anchors) < 2 or len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("anchored Elo requires unique historical anchors")
        if self.reference_anchor_id not in anchor_ids:
            raise ValueError("reference_anchor_id is not registered")
        if not profiles or len(profile_ids) != len(set(profile_ids)):
            raise ValueError("anchored Elo profiles must be unique and non-empty")
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "profiles", profiles)

    def anchor(self, anchor_id: str) -> AnchorSpec:
        for anchor in self.anchors:
            if anchor.anchor_id == anchor_id:
                return anchor
        raise KeyError(f"unknown anchor: {anchor_id}")

    def profile(self, profile_id: str) -> EvaluationProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"unknown anchored Elo profile: {profile_id}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_anchored_config(path: str | Path) -> AnchoredEloConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "reference_anchor_id",
        "rule_id",
        "anchors",
        "profiles",
        "openings",
        "statistics",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("anchored Elo config fields do not match schema v1")
    if raw["schema_version"] != ANCHORED_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported anchored Elo config schema")
    anchors = tuple(AnchorSpec(**item) for item in raw["anchors"])
    profiles = tuple(
        EvaluationProfile(
            **{**item, "milestones": tuple(item["milestones"])}
        )
        for item in raw["profiles"]
    )
    openings = OpeningSuiteSpec(
        **{**raw["openings"], "prefix_lengths": tuple(raw["openings"]["prefix_lengths"])}
    )
    statistics = AnchoredStatisticsConfig(**raw["statistics"])
    return AnchoredEloConfig(
        reference_anchor_id=raw["reference_anchor_id"],
        rule_id=raw["rule_id"],
        anchors=anchors,
        profiles=profiles,
        openings=openings,
        statistics=statistics,
    )


def canonical_anchored_config_hash(config: AnchoredEloConfig) -> str:
    encoded = json.dumps(
        config.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def evaluator_code_hash(repository_root: str | Path) -> str:
    """Hash the exact common-search and compatibility source boundary."""

    root = Path(repository_root).resolve()
    relative_paths = (
        "training/v3/anchored_elo.py",
        "training/v3/evaluation.py",
        "training/v3/search.py",
        "training/v3/model.py",
        "connect4_core/rules/specs.py",
        "connect4_core/rules/engine.py",
        "arena/agent.py",
        "training/experimental_models.py",
        "training/model.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"evaluator source is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def verify_anchor_files(
    config: AnchoredEloConfig, repository_root: str | Path
) -> tuple[dict[str, Any], ...]:
    root = Path(repository_root).resolve()
    rows: list[dict[str, Any]] = []
    for anchor in config.anchors:
        path = (root / anchor.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("anchor path escaped repository root")
        if not path.is_file():
            raise FileNotFoundError(f"anchor checkpoint is missing: {path}")
        actual = sha256_file(path)
        if actual != anchor.checksum_sha256:
            raise ValueError(f"anchor checksum mismatch: {anchor.anchor_id}")
        rows.append(
            {
                "model_id": anchor.anchor_id,
                "label": anchor.label,
                "path": str(path),
                "checksum_sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    return tuple(rows)


def verify_opening_suite(
    config: AnchoredEloConfig, repository_root: str | Path
) -> tuple[Opening, ...]:
    root = Path(repository_root).resolve()
    path = (root / config.openings.manifest_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"anchored opening manifest is missing: {path}")
    if _portable_text_sha256(path) != config.openings.checksum_sha256:
        raise ValueError("anchored opening manifest checksum mismatch")
    openings = load_opening_manifest(path)
    if len(openings) != config.openings.count:
        raise ValueError("anchored opening manifest count mismatch")
    if any(opening.rule_id != config.rule_id for opening in openings):
        raise ValueError("anchored opening manifest rule mismatch")
    return openings


def anchored_evaluation_plan(config: AnchoredEloConfig) -> dict[str, Any]:
    profiles = []
    for profile in config.profiles:
        profiles.append(
            {
                **asdict(profile),
                "initial_games_per_target": 2 * profile.initial_pairs * len(config.anchors),
                "maximum_games_per_target": 2 * profile.max_pairs * len(config.anchors),
            }
        )
    calibration_pairs = len(config.anchors) * (len(config.anchors) - 1) // 2
    return {
        "schema_version": ANCHORED_CONFIG_SCHEMA_VERSION,
        "config_hash": canonical_anchored_config_hash(config),
        "rule_id": config.rule_id,
        "reference_anchor_id": config.reference_anchor_id,
        "anchor_ids": [item.anchor_id for item in config.anchors],
        "profiles": profiles,
        "anchor_calibration_by_profile": {
            profile.profile_id: {
                "matchups": calibration_pairs,
                "initial_pairs_per_matchup": profile.initial_pairs,
                "initial_games": 2 * calibration_pairs * profile.initial_pairs,
                "scale_must_be_frozen_before_target_rating": True,
            }
            for profile in config.profiles
        },
        "milestone_profiles": {
            milestone: [
                profile.profile_id
                for profile in config.profiles
                if milestone in profile.milestones
            ]
            for milestone in ("early", "middle", "final")
        },
        "starts_training": False,
        "produces_selfplay_replay": False,
    }


class LegacyCheckpointPredictor:
    """Adapt a frozen 150-action Legacy network to the common V3 Predictor."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        # This import is the explicit external-opponent compatibility boundary.
        from arena.agent import ArenaModelPredictor, load_model_checkpoint

        model, model_config, metadata = load_model_checkpoint(
            model_path=Path(checkpoint_path),
            requested_config={},
            game=None,
            device=device,
        )
        self.model_config = dict(model_config)
        self.metadata = dict(metadata)
        self._predictor = ArenaModelPredictor(
            model=model,
            game_layers=6,
            game_size=5,
            model_layers=int(model_config["board_layers"]),
            model_size=int(model_config["board_size"]),
            input_encoding=(
                "single-channel"
                if int(model_config.get("input_channels", 2)) == 1
                else "two-channel"
            ),
        )

    @staticmethod
    def _validate_context(
        boards: np.ndarray, roles: np.ndarray, rules: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = np.asarray(boards, dtype=np.int8)
        role = np.asarray(roles, dtype=np.float32)
        rule = np.asarray(rules, dtype=np.float32)
        if raw.ndim != 4 or raw.shape[1:] != (6, 5, 5):
            raise ValueError("legacy predictor boards must have shape [N,6,5,5]")
        if role.shape != (raw.shape[0], 2) or rule.shape != (raw.shape[0], 32):
            raise ValueError("legacy predictor role/rule context has the wrong shape")
        classic = np.asarray(DEFAULT_RULE_REGISTRY.features(CLASSIC_RULE), dtype=np.float32)
        if not np.allclose(rule, classic[None, :]):
            raise ValueError("historical anchors support classic rule features only")
        return raw, role, rule

    def predict(
        self,
        canonical_board: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        policies, wdl = self.predict_batch(
            np.asarray(canonical_board)[None, ...],
            role_to_play=np.asarray(role_to_play)[None, ...],
            rule_features=np.asarray(rule_features)[None, ...],
        )
        return policies[0], wdl[0]

    def predict_batch(
        self,
        canonical_boards: np.ndarray,
        *,
        role_to_play: np.ndarray,
        rule_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        boards, _roles, _rules = self._validate_context(
            canonical_boards, role_to_play, rule_features
        )
        legacy_policy, scalar_value = self._predictor.predict(list(boards))
        columns = legacy_policy_to_columns(legacy_policy, boards).astype(np.float32)
        totals = columns.sum(axis=1, keepdims=True)
        columns = np.divide(
            columns,
            totals,
            out=np.full_like(columns, 1.0 / 25.0),
            where=totals > 0.0,
        )
        value = np.clip(np.asarray(scalar_value, dtype=np.float32), -1.0, 1.0)
        wdl = np.stack(((1.0 + value) / 2.0, np.zeros_like(value), (1.0 - value) / 2.0), axis=1)
        return columns, wdl


def load_v3_artifact_predictor(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> tuple[TorchPredictor, dict[str, Any]]:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "connect4-v3-model":
        raise ValueError(f"unsupported V3 model artifact: {path}")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported V3 model artifact version: {path}")
    model_config_raw = payload.get("model_config")
    if not isinstance(model_config_raw, Mapping):
        raise ValueError("V3 artifact model_config is missing")
    model_config = ModelConfig(**dict(model_config_raw))
    model = build_model(model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    identity = {
        "model_id": str(
            payload.get("metadata", {}).get("model_id")
            or payload.get("metadata", {}).get("candidate_model_id")
            or path.stem
        ),
        "label": path.stem,
        "path": str(path.resolve()),
        "checksum_sha256": sha256_file(path),
        "model_config": asdict(model_config),
        "lineage": "v3",
    }
    return TorchPredictor(model, device=device), identity


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def write_match_batch(
    output_path: str | Path,
    *,
    config: AnchoredEloConfig,
    profile: EvaluationProfile,
    openings: Sequence[Opening],
    opening_manifest_path: str | Path,
    model_a: Mapping[str, Any],
    model_b: Mapping[str, Any],
    predictor_a: Predictor,
    predictor_b: Predictor,
    milestone: str,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Run and immutably persist one disjoint paired-opening batch."""

    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"match batch already exists: {target}")
    if config.profile(profile.profile_id) != profile:
        raise ValueError("match profile differs from the frozen anchored config")
    if not milestone:
        raise ValueError("match milestone must not be empty")
    anchor_ids = {anchor.anchor_id for anchor in config.anchors}
    model_ids = {str(model_a.get("model_id")), str(model_b.get("model_id"))}
    anchor_count = len(model_ids.intersection(anchor_ids))
    if anchor_count == 2:
        if milestone != "calibration":
            raise ValueError("anchor-vs-anchor matches must use milestone=calibration")
    elif anchor_count == 1:
        if milestone not in profile.milestones:
            raise ValueError("target-vs-anchor milestone is not enabled for this profile")
    else:
        raise ValueError("anchored evaluation requires at least one historical anchor")
    rows = tuple(openings)
    if not rows:
        raise ValueError("match batch requires at least one opening pair")
    manifest_path = Path(opening_manifest_path)
    if _portable_text_sha256(manifest_path) != config.openings.checksum_sha256:
        raise ValueError("match opening manifest checksum differs from anchored config")
    manifest_rows = {opening.opening_id: opening for opening in load_opening_manifest(manifest_path)}
    if any(manifest_rows.get(opening.opening_id) != opening for opening in rows):
        raise ValueError("match openings are not an exact subset of the frozen suite")
    results = play_paired_openings(
        rows,
        candidate_predictor=predictor_a,
        incumbent_predictor=predictor_b,
        search_sims=profile.search_sims,
        cpuct=profile.cpuct,
    )
    payload = {
        "schema_version": MATCH_BATCH_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchored_config_hash": canonical_anchored_config_hash(config),
        "rule_id": config.rule_id,
        "rule_registry_hash": DEFAULT_RULE_REGISTRY.registry_hash,
        "evaluator_code_hash": evaluator_code_hash(Path(__file__).resolve().parents[2]),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            **dict(runtime or {}),
        },
        "profile": asdict(profile),
        "milestone": milestone,
        "opening_manifest": {
            "path": config.openings.manifest_path,
            "checksum_sha256": config.openings.checksum_sha256,
        },
        "model_a": dict(model_a),
        "model_b": dict(model_b),
        "results": [
            {
                "opening_id": result.opening_id,
                "seed": result.seed,
                "model_a_is_first": result.candidate_is_first,
                "model_a_score": result.candidate_score,
            }
            for result in results
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    payload["batch_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return _atomic_write_json(target, payload)


def load_match_batches(
    paths: Iterable[str | Path], *, expected_config_hash: str
) -> tuple[dict[str, Any], ...]:
    batches: list[dict[str, Any]] = []
    seen_batches: set[str] = set()
    seen_games: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != MATCH_BATCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported match batch schema: {path}")
        if raw.get("anchored_config_hash") != expected_config_hash:
            raise ValueError(f"match batch config hash drift: {path}")
        batch_id = raw.get("batch_id")
        content = dict(raw)
        content.pop("batch_id", None)
        canonical = json.dumps(
            content, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if batch_id != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
            raise ValueError(f"match batch content hash mismatch: {path}")
        if batch_id in seen_batches:
            raise ValueError(f"duplicate match batch: {batch_id}")
        seen_batches.add(batch_id)
        model_a = raw["model_a"]["model_id"]
        model_b = raw["model_b"]["model_id"]
        if model_a == model_b:
            raise ValueError("match batch cannot contain a self-match")
        profile_id = raw["profile"]["profile_id"]
        for result in raw["results"]:
            if set(result) != {
                "opening_id",
                "seed",
                "model_a_is_first",
                "model_a_score",
            }:
                raise ValueError("match result fields do not match schema v1")
            score = float(result["model_a_score"])
            if score not in (0.0, 0.5, 1.0):
                raise ValueError("match result score must be 0, 0.5, or 1")
            first_model = model_a if bool(result["model_a_is_first"]) else model_b
            key = (
                profile_id,
                min(model_a, model_b),
                max(model_a, model_b),
                result["opening_id"],
                first_model,
            )
            if key in seen_games:
                raise ValueError(f"duplicate anchored evaluation game: {key}")
            seen_games.add(key)
        batches.append(raw)
    return tuple(batches)


def wilson_interval(score: float, count: int, confidence: float) -> tuple[float, float]:
    """Finite score interval, including at 0% and 100% observed score."""

    if count < 1 or not 0.0 <= score <= 1.0:
        raise ValueError("Wilson interval needs count >= 1 and score in [0,1]")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / count
    center = (score + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(score * (1.0 - score) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def score_to_elo(score: float) -> float:
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return 400.0 * math.log10(score / (1.0 - score))


def summarize_direct_matchup(
    pair_scores: Sequence[float], statistics: AnchoredStatisticsConfig
) -> dict[str, Any]:
    values = np.asarray(pair_scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or np.any(~np.isin(values, (0.0, 0.25, 0.5, 0.75, 1.0))):
        raise ValueError("pair_scores must contain paired two-game scores")
    score = float(values.mean())
    lower, upper = wilson_interval(score, len(values), statistics.confidence)
    saturated_high = score >= statistics.saturation_score and lower >= statistics.saturation_lower_bound
    saturated_low = score <= 1.0 - statistics.saturation_score and upper <= 1.0 - statistics.saturation_lower_bound
    if saturated_high:
        status = "saturated_high"
    elif saturated_low:
        status = "saturated_low"
    else:
        status = "measured"
    elo_lower = score_to_elo(lower)
    elo_upper = score_to_elo(upper)
    interval_width = elo_upper - elo_lower
    if status in {"saturated_high", "saturated_low"}:
        evidence_status = "complete_saturated"
    elif math.isfinite(interval_width) and interval_width <= statistics.max_elo_interval_width:
        evidence_status = "complete_interval"
    else:
        evidence_status = "extend_if_below_profile_cap"
    return {
        "pairs": int(values.size),
        "games": int(2 * values.size),
        "point_score": score,
        "score_ci_lower": lower,
        "score_ci_upper": upper,
        "relative_elo": None if status != "measured" or score in (0.0, 1.0) else score_to_elo(score),
        "relative_elo_lower": elo_lower,
        "relative_elo_upper": elo_upper,
        "relative_elo_interval_width": interval_width,
        "rating_status": status,
        "evidence_status": evidence_status,
    }


def fit_anchor_scale(
    observations: Sequence[tuple[str, str, float]],
    *,
    model_ids: Sequence[str],
    reference_model_id: str,
    prior_sigma: float,
) -> dict[str, float]:
    """Regularized Bradley-Terry fit; reference rating is exactly zero."""

    ids = tuple(model_ids)
    if reference_model_id not in ids or len(ids) != len(set(ids)):
        raise ValueError("rating model IDs or reference are invalid")
    variables = tuple(model_id for model_id in ids if model_id != reference_model_id)
    index = {model_id: idx for idx, model_id in enumerate(variables)}
    ratings = np.zeros((len(variables),), dtype=np.float64)
    prior_precision = 1.0 / (float(prior_sigma) ** 2)

    def rating(model_id: str) -> float:
        return 0.0 if model_id == reference_model_id else float(ratings[index[model_id]])

    if not observations:
        raise ValueError("anchor scale needs match observations")
    for model_a, model_b, score_a in observations:
        if model_a not in ids or model_b not in ids or model_a == model_b:
            raise ValueError("rating observation references invalid models")
        if score_a not in (0.0, 0.5, 1.0):
            raise ValueError("rating observations must use game scores 0, 0.5, or 1")

    for _iteration in range(100):
        gradient = -prior_precision * ratings
        hessian = -prior_precision * np.eye(len(variables), dtype=np.float64)
        for model_a, model_b, score_a in observations:
            delta = (rating(model_a) - rating(model_b)) / ELO_SCALE
            probability = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, delta))))
            residual = float(score_a) - probability
            weight = probability * (1.0 - probability)
            vector = np.zeros_like(ratings)
            if model_a != reference_model_id:
                vector[index[model_a]] += 1.0
            if model_b != reference_model_id:
                vector[index[model_b]] -= 1.0
            gradient += residual * vector / ELO_SCALE
            hessian -= weight * np.outer(vector, vector) / (ELO_SCALE * ELO_SCALE)
        step = np.linalg.solve(hessian, gradient)
        ratings -= step
        if float(np.max(np.abs(step), initial=0.0)) < 1e-8:
            break
    return {
        model_id: (0.0 if model_id == reference_model_id else float(ratings[index[model_id]]))
        for model_id in ids
    }


def _match_observations(
    batches: Sequence[Mapping[str, Any]], *, profile_id: str
) -> list[tuple[str, str, float]]:
    observations: list[tuple[str, str, float]] = []
    for batch in batches:
        if batch["profile"]["profile_id"] != profile_id:
            continue
        model_a = str(batch["model_a"]["model_id"])
        model_b = str(batch["model_b"]["model_id"])
        observations.extend(
            (model_a, model_b, float(result["model_a_score"]))
            for result in batch["results"]
        )
    return observations


def calibrate_anchor_scale(
    config: AnchoredEloConfig,
    batches: Sequence[Mapping[str, Any]],
    *,
    profile_id: str,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Fit the immutable historical scale from anchor-vs-anchor batches only."""

    config.profile(profile_id)
    anchor_ids = tuple(anchor.anchor_id for anchor in config.anchors)
    anchor_set = set(anchor_ids)
    selected = tuple(
        batch
        for batch in batches
        if batch["profile"]["profile_id"] == profile_id
        and batch["model_a"]["model_id"] in anchor_set
        and batch["model_b"]["model_id"] in anchor_set
    )
    observations = _match_observations(selected, profile_id=profile_id)
    covered_edges = {
        frozenset((model_a, model_b)) for model_a, model_b, _score in observations
    }
    expected_edges = {
        frozenset((anchor_ids[first], anchor_ids[second]))
        for first in range(len(anchor_ids))
        for second in range(first + 1, len(anchor_ids))
    }
    if covered_edges != expected_edges:
        missing = sorted(" vs ".join(sorted(edge)) for edge in expected_edges - covered_edges)
        raise ValueError(f"anchor calibration is missing matchups: {missing}")
    ratings = fit_anchor_scale(
        observations,
        model_ids=anchor_ids,
        reference_model_id=config.reference_anchor_id,
        prior_sigma=config.statistics.rating_prior_sigma,
    )
    return ratings, tuple(str(batch["batch_id"]) for batch in selected)


def _paired_scores(
    batches: Sequence[Mapping[str, Any]],
    *,
    profile_id: str,
    target_model_id: str,
    anchor_model_id: str,
) -> tuple[float, ...]:
    grouped: dict[str, list[tuple[bool, float]]] = {}
    for batch in batches:
        if batch["profile"]["profile_id"] != profile_id:
            continue
        model_a = str(batch["model_a"]["model_id"])
        model_b = str(batch["model_b"]["model_id"])
        if {model_a, model_b} != {target_model_id, anchor_model_id}:
            continue
        target_is_a = model_a == target_model_id
        for result in batch["results"]:
            target_is_first = (
                bool(result["model_a_is_first"])
                if target_is_a
                else not bool(result["model_a_is_first"])
            )
            target_score = (
                float(result["model_a_score"])
                if target_is_a
                else 1.0 - float(result["model_a_score"])
            )
            grouped.setdefault(str(result["opening_id"]), []).append(
                (target_is_first, target_score)
            )
    pair_scores: list[float] = []
    for opening_id in sorted(grouped):
        games = grouped[opening_id]
        if len(games) != 2 or {role for role, _score in games} != {False, True}:
            raise ValueError(f"opening {opening_id} is not an exact role-swapped pair")
        pair_scores.append(sum(score for _role, score in games) / 2.0)
    return tuple(pair_scores)


def _role_scores(
    batches: Sequence[Mapping[str, Any]],
    *,
    profile_id: str,
    target_model_id: str,
    anchor_model_id: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    first: list[float] = []
    second: list[float] = []
    for batch in batches:
        if batch["profile"]["profile_id"] != profile_id:
            continue
        model_a = str(batch["model_a"]["model_id"])
        model_b = str(batch["model_b"]["model_id"])
        if {model_a, model_b} != {target_model_id, anchor_model_id}:
            continue
        target_is_a = model_a == target_model_id
        for result in batch["results"]:
            target_is_first = (
                bool(result["model_a_is_first"])
                if target_is_a
                else not bool(result["model_a_is_first"])
            )
            score = (
                float(result["model_a_score"])
                if target_is_a
                else 1.0 - float(result["model_a_score"])
            )
            (first if target_is_first else second).append(score)
    return tuple(first), tuple(second)


def _score_breakdown(scores: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(score) for score in scores)
    return {
        "games": len(values),
        "wins": sum(score == 1.0 for score in values),
        "draws": sum(score == 0.5 for score in values),
        "losses": sum(score == 0.0 for score in values),
        "point_score": sum(values) / len(values) if values else 0.0,
    }


def _fit_target_rating(
    observations: Sequence[tuple[float, float]], *, prior_sigma: float
) -> tuple[float, float]:
    """Fit one target against fixed anchor ratings, returning rating and SE."""

    if not observations:
        raise ValueError("target rating needs at least one historical-anchor game")
    rating = 0.0
    prior_precision = 1.0 / (prior_sigma * prior_sigma)
    hessian = -prior_precision
    for _iteration in range(100):
        gradient = -prior_precision * rating
        hessian = -prior_precision
        for anchor_rating, score in observations:
            eta = max(-50.0, min(50.0, (rating - anchor_rating) / ELO_SCALE))
            probability = 1.0 / (1.0 + math.exp(-eta))
            gradient += (score - probability) / ELO_SCALE
            hessian -= probability * (1.0 - probability) / (ELO_SCALE * ELO_SCALE)
        step = gradient / hessian
        rating -= step
        if abs(step) < 1e-8:
            break
    standard_error = math.sqrt(max(0.0, -1.0 / hessian))
    return float(rating), float(standard_error)


def build_anchored_report(
    config: AnchoredEloConfig,
    batches: Sequence[Mapping[str, Any]],
    *,
    profile_id: str,
    target_model_id: str,
    anchor_scale: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one target without changing the frozen historical scale."""

    if anchor_scale.get("schema_version") != ANCHOR_SCALE_SCHEMA_VERSION:
        raise ValueError("unsupported anchor scale schema")
    if anchor_scale.get("anchored_config_hash") != canonical_anchored_config_hash(config):
        raise ValueError("anchor scale config hash drift")
    if anchor_scale.get("profile_id") != profile_id or not anchor_scale.get("frozen"):
        raise ValueError("anchor scale profile mismatch or scale is not frozen")
    fixed_ratings = {
        str(key): float(value) for key, value in anchor_scale["ratings"].items()
    }
    if set(fixed_ratings) != {anchor.anchor_id for anchor in config.anchors}:
        raise ValueError("anchor scale does not cover the configured historical anchors")

    direct: dict[str, Any] = {}
    target_observations: list[tuple[float, float]] = []
    for anchor in config.anchors:
        scores = _paired_scores(
            batches,
            profile_id=profile_id,
            target_model_id=target_model_id,
            anchor_model_id=anchor.anchor_id,
        )
        if not scores:
            raise ValueError(f"target has no matches against anchor {anchor.anchor_id}")
        first_scores, second_scores = _role_scores(
            batches,
            profile_id=profile_id,
            target_model_id=target_model_id,
            anchor_model_id=anchor.anchor_id,
        )
        direct[anchor.anchor_id] = {
            **summarize_direct_matchup(scores, config.statistics),
            "target_as_first": _score_breakdown(first_scores),
            "target_as_second": _score_breakdown(second_scores),
        }

    for model_a, model_b, score_a in _match_observations(batches, profile_id=profile_id):
        if model_a == target_model_id and model_b in fixed_ratings:
            target_observations.append((fixed_ratings[model_b], score_a))
        elif model_b == target_model_id and model_a in fixed_ratings:
            target_observations.append((fixed_ratings[model_a], 1.0 - score_a))
    rating, standard_error = _fit_target_rating(
        target_observations, prior_sigma=config.statistics.rating_prior_sigma
    )
    z = NormalDist().inv_cdf(0.5 + config.statistics.confidence / 2.0)
    saturated = any(
        row["rating_status"] == "saturated_high" for row in direct.values()
    )
    return {
        "schema_version": ANCHORED_REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchored_config_hash": canonical_anchored_config_hash(config),
        "profile_id": profile_id,
        "target_model_id": target_model_id,
        "reference_anchor_id": config.reference_anchor_id,
        "anchor_scale_ratings": dict(sorted(fixed_ratings.items())),
        "anchored_rating": {
            "estimate": rating,
            "ci_lower": rating - z * standard_error,
            "ci_upper": rating + z * standard_error,
            "standard_error": standard_error,
            "method": "fixed-anchor regularized Bradley-Terry",
            "prior_sigma_elo": config.statistics.rating_prior_sigma,
            "contains_saturated_matchup": saturated,
        },
        "direct_matchups": direct,
        "source_batch_ids": sorted(
            {
                str(batch["batch_id"])
                for batch in batches
                if batch["profile"]["profile_id"] == profile_id
                and target_model_id
                in {
                    str(batch["model_a"]["model_id"]),
                    str(batch["model_b"]["model_id"]),
                }
            }
        ),
        "promotion_gate_input": False,
        "selfplay_replay_input": False,
    }


def write_anchor_scale(
    path: str | Path,
    *,
    config: AnchoredEloConfig,
    profile_id: str,
    ratings: Mapping[str, float],
    source_batch_ids: Sequence[str],
) -> Path:
    if Path(path).exists():
        raise FileExistsError(f"frozen anchor scale already exists: {path}")
    if set(ratings) != {anchor.anchor_id for anchor in config.anchors}:
        raise ValueError("anchor scale ratings do not cover the frozen anchor registry")
    if float(ratings[config.reference_anchor_id]) != 0.0:
        raise ValueError("reference anchor rating must be exactly zero")
    payload = {
        "schema_version": ANCHOR_SCALE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchored_config_hash": canonical_anchored_config_hash(config),
        "profile_id": profile_id,
        "reference_anchor_id": config.reference_anchor_id,
        "ratings": {key: float(value) for key, value in sorted(ratings.items())},
        "source_batch_ids": sorted(set(source_batch_ids)),
        "frozen": True,
    }
    return _atomic_write_json(path, payload)


__all__ = [
    "ANCHORED_CONFIG_SCHEMA_VERSION",
    "ANCHOR_SCALE_SCHEMA_VERSION",
    "MATCH_BATCH_SCHEMA_VERSION",
    "AnchorSpec",
    "AnchoredEloConfig",
    "AnchoredStatisticsConfig",
    "EvaluationProfile",
    "LegacyCheckpointPredictor",
    "OpeningSuiteSpec",
    "anchored_evaluation_plan",
    "build_anchored_report",
    "calibrate_anchor_scale",
    "canonical_anchored_config_hash",
    "evaluator_code_hash",
    "fit_anchor_scale",
    "load_anchored_config",
    "load_match_batches",
    "load_v3_artifact_predictor",
    "score_to_elo",
    "summarize_direct_matchup",
    "verify_anchor_files",
    "verify_opening_suite",
    "wilson_interval",
    "write_anchor_scale",
    "write_match_batch",
]
