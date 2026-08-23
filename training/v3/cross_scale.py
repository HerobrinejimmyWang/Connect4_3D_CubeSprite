"""Cross-scale replay transfer and donor-qualification contracts.

This module deliberately does not mutate a formal V3 run.  It creates and
validates immutable data-only bundles, plans deterministic donor/own sampling,
and writes qualification evidence that a future formal scheduler may consume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from connect4_core.rules import DEFAULT_RULE_REGISTRY

from .evaluation import Opening, play_paired_openings
from .evaluation_runtime import play_paired_openings_parallel
from .gate import GateGameResult, evaluate_gate
from .replay import (
    ARRAY_SCHEMA,
    REPLAY_SCHEMA_VERSION,
    ReplayShard,
    concatenate_replay,
    load_replay_shard,
    replay_manifest_path,
    replay_ready_path,
    sha256_file,
    validate_replay_shard_artifacts,
)


BUNDLE_FORMAT = "cross_scale_replay_bundle_v1"
BUNDLE_SCHEMA_VERSION = 1
SOURCE_SPEC_SCHEMA_VERSION = 1
QUALIFICATION_FORMAT = "cross_scale_donor_qualification_v1"
QUALIFICATION_SCHEMA_VERSION = 1
STRATA = ("early", "middle", "late", "strong")
ACTION_SCHEMA = "gravity_columns_25"
ROLE_SCHEMA = "absolute_first_second_v1"
TARGET_SCHEMA = "policy_wdl_aux_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def replay_semantic_contract(rule_id: str) -> dict[str, Any]:
    """Return the model-size-independent contract required for replay transfer."""

    spec = DEFAULT_RULE_REGISTRY.get(rule_id)
    return {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "arrays": {
            name: {"dtype": dtype.name, "shape": ["N", *shape]}
            for name, (dtype, shape) in ARRAY_SCHEMA.items()
        },
        "action_schema": ACTION_SCHEMA,
        "role_schema": ROLE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "rule_id": spec.rule_id,
        "rule_code": spec.rule_code,
        "rule_version": spec.rule_version,
        "rule_registry_hash": DEFAULT_RULE_REGISTRY.registry_hash,
    }


def replay_semantic_hash(rule_id: str) -> str:
    return _sha256_json(replay_semantic_contract(rule_id))


@dataclass(frozen=True)
class BundleSource:
    shard_path: Path
    accepted_model_artifact: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_path", Path(self.shard_path).resolve())
        object.__setattr__(
            self,
            "accepted_model_artifact",
            Path(self.accepted_model_artifact).resolve(),
        )


def load_bundle_source_spec(
    path: str | Path,
) -> tuple[str, str, str, dict[str, tuple[BundleSource, ...]]]:
    """Load the strict input catalog used to build one immutable bundle."""

    source_path = Path(path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "donor_run_id",
        "qualification_donor_model_id",
        "rule_id",
        "strata",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("cross-scale source spec fields do not match schema V1")
    if raw["schema_version"] != SOURCE_SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported cross-scale source spec schema")
    donor_run_id = raw["donor_run_id"]
    qualification_donor_model_id = raw["qualification_donor_model_id"]
    rule_id = raw["rule_id"]
    if not isinstance(donor_run_id, str) or not donor_run_id:
        raise ValueError("donor_run_id must be a non-empty string")
    if not isinstance(rule_id, str) or not rule_id:
        raise ValueError("rule_id must be a non-empty string")
    if not isinstance(qualification_donor_model_id, str) or not qualification_donor_model_id:
        raise ValueError("qualification_donor_model_id must be a non-empty string")
    DEFAULT_RULE_REGISTRY.get(rule_id)
    strata_raw = raw["strata"]
    if not isinstance(strata_raw, dict) or set(strata_raw) != set(STRATA):
        raise ValueError(f"source spec strata must contain exactly {list(STRATA)}")
    strata: dict[str, tuple[BundleSource, ...]] = {}
    for stratum in STRATA:
        rows = strata_raw[stratum]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"source stratum {stratum!r} must be a non-empty list")
        decoded: list[BundleSource] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "shard_path",
                "accepted_model_artifact",
            }:
                raise ValueError(f"source rows in {stratum!r} do not match schema V1")
            shard = Path(row["shard_path"])
            artifact = Path(row["accepted_model_artifact"])
            if not shard.is_absolute():
                shard = source_path.parent / shard
            if not artifact.is_absolute():
                artifact = source_path.parent / artifact
            decoded.append(BundleSource(shard, artifact))
        strata[stratum] = tuple(decoded)
    return donor_run_id, qualification_donor_model_id, rule_id, strata


def _accepted_artifact_attestation(
    artifact_path: Path,
    *,
    producer_model_id: str,
    config_hash: str,
) -> dict[str, Any]:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"accepted producer artifact not found: {artifact_path}")
    import torch

    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "connect4-v3-model":
        raise ValueError(f"producer artifact is not a V3 model: {artifact_path}")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported producer artifact version: {artifact_path}")
    metadata = payload.get("metadata")
    model_config = payload.get("model_config")
    if not isinstance(metadata, Mapping) or not isinstance(model_config, Mapping):
        raise ValueError("producer artifact is missing metadata or model_config")
    artifact_model_id = metadata.get("candidate_model_id") or metadata.get("model_id")
    if artifact_model_id != producer_model_id:
        raise ValueError("replay producer_model_id does not match its accepted artifact")
    if metadata.get("config_hash") != config_hash:
        raise ValueError("replay config_hash does not match its accepted artifact")
    required_model_contract = {
        "global_input_schema": "role_rule_v1",
        "output_schema": TARGET_SCHEMA,
        "rule_feature_dim": 32,
        "moves_left_classes": 301,
    }
    if any(model_config.get(key) != value for key, value in required_model_contract.items()):
        raise ValueError("producer artifact does not use the frozen replay/model contract")
    return {
        "model_id": producer_model_id,
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "config_hash": config_hash,
        "model": {
            "architecture": model_config.get("architecture"),
            "channels": model_config.get("channels"),
            "blocks": model_config.get("blocks"),
        },
    }


def _bundle_content_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "bundle_content_sha256"}


def build_cross_scale_bundle(
    output_dir: str | Path,
    *,
    donor_run_id: str,
    qualification_donor_model_id: str,
    rule_id: str,
    strata: Mapping[str, Sequence[BundleSource]],
) -> Path:
    """Copy authenticated replay triplets into one immutable portable bundle."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"cross-scale bundle already exists: {target}")
    if not donor_run_id:
        raise ValueError("donor_run_id must not be empty")
    if not qualification_donor_model_id:
        raise ValueError("qualification_donor_model_id must not be empty")
    if set(strata) != set(STRATA) or any(not strata[name] for name in STRATA):
        raise ValueError(f"bundle strata must contain four non-empty groups: {list(STRATA)}")
    spec = DEFAULT_RULE_REGISTRY.get(rule_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temp_name:
        staging = Path(temp_name)
        entries: list[dict[str, Any]] = []
        producer_attestations: dict[str, dict[str, Any]] = {}
        seen_shard_checksums: set[str] = set()
        source_config_hashes: set[str] = set()
        for stratum in STRATA:
            for index, source in enumerate(strata[stratum]):
                shard, source_manifest = load_replay_shard(source.shard_path)
                if source_manifest["run_id"] != donor_run_id:
                    raise ValueError("bundle source run_id differs from donor_run_id")
                if source_manifest["rule_registry_hash"] != DEFAULT_RULE_REGISTRY.registry_hash:
                    raise ValueError("bundle source rule registry differs from executable rules")
                if len(shard) == 0 or np.any(shard.rule_code != spec.rule_code):
                    raise ValueError("bundle shard is empty or contains a different rule")
                producer = str(source_manifest["producer_model_id"])
                if producer == "random":
                    raise ValueError("random-bootstrap replay cannot enter a cross-scale bundle")
                checksum = str(source_manifest["checksum_sha256"])
                if checksum in seen_shard_checksums:
                    raise ValueError("a replay shard appears more than once in the bundle")
                seen_shard_checksums.add(checksum)
                config_hash = str(source_manifest["config_hash"])
                source_config_hashes.add(config_hash)
                attestation = _accepted_artifact_attestation(
                    source.accepted_model_artifact,
                    producer_model_id=producer,
                    config_hash=config_hash,
                )
                previous = producer_attestations.setdefault(producer, attestation)
                if previous != attestation:
                    raise ValueError("one producer_model_id resolves to multiple artifacts")

                destination_name = source.shard_path.name
                relative_shard = Path("shards") / stratum / destination_name
                destination = staging / relative_shard
                if destination.exists():
                    raise ValueError("bundle source shard filenames collide within one stratum")
                destination.parent.mkdir(parents=True, exist_ok=True)
                for source_file, destination_file in (
                    (source.shard_path, destination),
                    (replay_manifest_path(source.shard_path), replay_manifest_path(destination)),
                    (replay_ready_path(source.shard_path), replay_ready_path(destination)),
                ):
                    shutil.copy2(source_file, destination_file)
                copied_manifest = validate_replay_shard_artifacts(destination)
                if copied_manifest != source_manifest:
                    raise RuntimeError("copied replay manifest changed during bundle creation")
                entries.append(
                    {
                        "stratum": stratum,
                        "shard_path": relative_shard.as_posix(),
                        "sample_count": int(source_manifest["sample_count"]),
                        "game_count": int(len(set(int(value) for value in shard.game_id))),
                        "checksum_sha256": checksum,
                        "source_generation": int(source_manifest["generation"]),
                        "producer_model_id": producer,
                        "source_config_hash": config_hash,
                        "search_config": source_manifest["search_config"],
                    }
                )

        if len(source_config_hashes) != 1:
            raise ValueError("one donor bundle must come from exactly one source config lineage")
        strong_producers = {
            entry["producer_model_id"] for entry in entries if entry["stratum"] == "strong"
        }
        if qualification_donor_model_id not in strong_producers:
            raise ValueError("qualification donor must produce at least one strong stratum shard")
        strata_summary = {
            stratum: {
                "shards": sum(entry["stratum"] == stratum for entry in entries),
                "games": sum(entry["game_count"] for entry in entries if entry["stratum"] == stratum),
                "positions": sum(
                    entry["sample_count"] for entry in entries if entry["stratum"] == stratum
                ),
            }
            for stratum in STRATA
        }
        manifest: dict[str, Any] = {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "donor_run_id": donor_run_id,
            "qualification_donor_model_id": qualification_donor_model_id,
            "data_only": True,
            "weight_transfer": False,
            "semantic_contract": replay_semantic_contract(rule_id),
            "semantic_hash": replay_semantic_hash(rule_id),
            "source_config_hash": next(iter(source_config_hashes)),
            "producer_attestations": [
                producer_attestations[key] for key in sorted(producer_attestations)
            ],
            "strata": strata_summary,
            "entries": entries,
            "totals": {
                "shards": len(entries),
                "games": sum(entry["game_count"] for entry in entries),
                "positions": sum(entry["sample_count"] for entry in entries),
            },
        }
        manifest["bundle_content_sha256"] = _sha256_json(_bundle_content_payload(manifest))
        manifest_path = _atomic_write_json(staging / "bundle.manifest.json", manifest)
        _atomic_write_json(
            staging / "bundle.ready.json",
            {
                "format": BUNDLE_FORMAT,
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "manifest_sha256": sha256_file(manifest_path),
                "bundle_content_sha256": manifest["bundle_content_sha256"],
            },
        )
        os.replace(staging, target)
    return target


def validate_cross_scale_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    manifest_path = root / "bundle.manifest.json"
    ready_path = root / "bundle.ready.json"
    if not root.is_dir() or not manifest_path.is_file() or not ready_path.is_file():
        raise ValueError("cross-scale bundle is missing its manifest or ready marker")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    expected_manifest_keys = {
        "format",
        "schema_version",
        "created_at",
        "donor_run_id",
        "qualification_donor_model_id",
        "data_only",
        "weight_transfer",
        "semantic_contract",
        "semantic_hash",
        "source_config_hash",
        "producer_attestations",
        "strata",
        "entries",
        "totals",
        "bundle_content_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("cross-scale bundle manifest fields do not match schema V1")
    if manifest.get("format") != BUNDLE_FORMAT or manifest.get("schema_version") != 1:
        raise ValueError("unsupported cross-scale bundle")
    expected_ready = {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "manifest_sha256": sha256_file(manifest_path),
        "bundle_content_sha256": manifest.get("bundle_content_sha256"),
    }
    if ready != expected_ready:
        raise ValueError("cross-scale bundle ready marker does not authenticate its manifest")
    if manifest.get("bundle_content_sha256") != _sha256_json(_bundle_content_payload(manifest)):
        raise ValueError("cross-scale bundle content hash mismatch")
    semantic = manifest.get("semantic_contract")
    if not isinstance(semantic, dict) or semantic.get("rule_id") is None:
        raise ValueError("cross-scale bundle semantic contract is missing")
    rule_id = str(semantic["rule_id"])
    if semantic != replay_semantic_contract(rule_id):
        raise ValueError("cross-scale bundle semantic contract differs from executable V3")
    if manifest.get("semantic_hash") != replay_semantic_hash(rule_id):
        raise ValueError("cross-scale bundle semantic hash mismatch")
    if manifest.get("data_only") is not True or manifest.get("weight_transfer") is not False:
        raise ValueError("cross-scale bundle V1 must be data-only")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("cross-scale bundle contains no replay entries")
    seen_paths: set[str] = set()
    seen_checksums: set[str] = set()
    computed_strata = {name: {"shards": 0, "games": 0, "positions": 0} for name in STRATA}
    for entry in entries:
        expected_keys = {
            "stratum",
            "shard_path",
            "sample_count",
            "game_count",
            "checksum_sha256",
            "source_generation",
            "producer_model_id",
            "source_config_hash",
            "search_config",
        }
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ValueError("cross-scale bundle entry fields do not match schema V1")
        if entry["stratum"] not in STRATA:
            raise ValueError("cross-scale bundle has an unknown stratum")
        relative = Path(entry["shard_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen_paths:
            raise ValueError("cross-scale bundle shard path is unsafe or duplicated")
        seen_paths.add(relative.as_posix())
        shard_path = (root / relative).resolve()
        if not shard_path.is_relative_to(root):
            raise ValueError("cross-scale bundle shard escaped its root")
        shard, shard_manifest = load_replay_shard(shard_path)
        checksum = str(entry["checksum_sha256"])
        if checksum in seen_checksums:
            raise ValueError("cross-scale bundle contains duplicate replay content")
        seen_checksums.add(checksum)
        if shard_manifest["checksum_sha256"] != checksum or len(shard) != entry["sample_count"]:
            raise ValueError("cross-scale bundle entry differs from its replay shard")
        if shard_manifest["run_id"] != manifest["donor_run_id"]:
            raise ValueError("cross-scale bundle replay run_id mismatch")
        if shard_manifest["producer_model_id"] != entry["producer_model_id"]:
            raise ValueError("cross-scale bundle replay producer mismatch")
        if shard_manifest["config_hash"] != entry["source_config_hash"]:
            raise ValueError("cross-scale bundle replay config hash mismatch")
        if entry["source_config_hash"] != manifest["source_config_hash"]:
            raise ValueError("cross-scale bundle contains multiple config lineages")
        if shard_manifest["rule_registry_hash"] != semantic["rule_registry_hash"]:
            raise ValueError("cross-scale bundle replay rule registry mismatch")
        if np.any(shard.rule_code != int(semantic["rule_code"])):
            raise ValueError("cross-scale bundle replay contains a different rule")
        summary = computed_strata[entry["stratum"]]
        summary["shards"] += 1
        summary["games"] += int(entry["game_count"])
        summary["positions"] += len(shard)
        if int(entry["game_count"]) != len(set(int(value) for value in shard.game_id)):
            raise ValueError("cross-scale bundle entry game count mismatch")
    if computed_strata != manifest.get("strata") or any(
        computed_strata[name]["shards"] < 1 for name in STRATA
    ):
        raise ValueError("cross-scale bundle strata summary mismatch")
    totals = {
        "shards": len(entries),
        "games": sum(item["games"] for item in computed_strata.values()),
        "positions": sum(item["positions"] for item in computed_strata.values()),
    }
    if totals != manifest.get("totals"):
        raise ValueError("cross-scale bundle totals mismatch")
    attestations = manifest.get("producer_attestations")
    if not isinstance(attestations, list) or not attestations:
        raise ValueError("cross-scale bundle lacks accepted-producer attestations")
    expected_attestation_keys = {
        "model_id",
        "artifact_file",
        "artifact_sha256",
        "config_hash",
        "model",
    }
    if any(not isinstance(row, dict) or set(row) != expected_attestation_keys for row in attestations):
        raise ValueError("cross-scale producer attestation fields do not match schema V1")
    if any(
        len(str(row["artifact_sha256"])) != 64
        or row["config_hash"] != manifest["source_config_hash"]
        for row in attestations
    ):
        raise ValueError("cross-scale producer attestation hash/config mismatch")
    attestation_ids = {row.get("model_id") for row in attestations if isinstance(row, dict)}
    producer_ids = {entry["producer_model_id"] for entry in entries}
    if attestation_ids != producer_ids:
        raise ValueError("cross-scale bundle producer attestations do not cover its entries")
    if manifest.get("qualification_donor_model_id") not in {
        entry["producer_model_id"] for entry in entries if entry["stratum"] == "strong"
    }:
        raise ValueError("cross-scale qualification donor is not a strong-stratum producer")
    return manifest


def load_cross_scale_replay(
    bundle_dir: str | Path,
    *,
    strata: Iterable[str] = STRATA,
) -> ReplayShard:
    manifest = validate_cross_scale_bundle(bundle_dir)
    selected = tuple(strata)
    if not selected or len(set(selected)) != len(selected) or any(name not in STRATA for name in selected):
        raise ValueError("selected bundle strata must be unique known names")
    root = Path(bundle_dir).resolve()
    shards = [
        load_replay_shard(root / entry["shard_path"])[0]
        for entry in manifest["entries"]
        if entry["stratum"] in selected
    ]
    return concatenate_replay(shards)


@dataclass(frozen=True)
class TransferStage:
    start_own_positions: int
    donor_fraction: float

    def __post_init__(self) -> None:
        if self.start_own_positions < 0:
            raise ValueError("start_own_positions must be non-negative")
        if not math.isfinite(self.donor_fraction) or not 0.0 <= self.donor_fraction <= 1.0:
            raise ValueError("donor_fraction must be finite and in [0, 1]")


@dataclass(frozen=True)
class TransferSchedule:
    stages: tuple[TransferStage, ...]

    def __post_init__(self) -> None:
        if not self.stages or self.stages[0].start_own_positions != 0:
            raise ValueError("transfer schedule must start at zero")
        starts = tuple(stage.start_own_positions for stage in self.stages)
        fractions = tuple(stage.donor_fraction for stage in self.stages)
        if starts != tuple(sorted(set(starts))):
            raise ValueError("transfer schedule boundaries must be strictly increasing")
        if any(right > left for left, right in zip(fractions, fractions[1:])):
            raise ValueError("donor_fraction must not increase")

    def donor_fraction_for(self, own_positions_generated: int) -> float:
        if own_positions_generated < 0:
            raise ValueError("own_positions_generated must be non-negative")
        selected = self.stages[0]
        for stage in self.stages:
            if stage.start_own_positions > own_positions_generated:
                break
            selected = stage
        return selected.donor_fraction


@dataclass(frozen=True)
class MixedReplayKey:
    origin: str
    index: int
    cursor: int


@dataclass(frozen=True)
class TransferSampleKey:
    origin: str
    stratum: str | None
    index: int
    cursor: int


class MixedReplayPlanner:
    """Deterministically choose donor/own sample keys without touching trainer state."""

    def __init__(
        self,
        *,
        donor_size: int,
        own_size: int,
        schedule: TransferSchedule,
        seed: int,
    ) -> None:
        if donor_size < 0 or own_size < 0 or donor_size + own_size == 0:
            raise ValueError("mixed replay pools must contain at least one sample")
        self.donor_size = int(donor_size)
        self.own_size = int(own_size)
        self.schedule = schedule
        self.seed = int(seed)

    def _word(self, cursor: int, stream: int) -> int:
        if cursor < 0:
            raise ValueError("sample cursor must be non-negative")
        payload = f"{self.seed}:{cursor}:{stream}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def key(self, cursor: int, *, own_positions_generated: int) -> MixedReplayKey:
        fraction = self.schedule.donor_fraction_for(own_positions_generated)
        donor_selected = self._word(cursor, 0) / float(2**64) < fraction
        if self.donor_size == 0:
            donor_selected = False
        elif self.own_size == 0:
            donor_selected = True
        origin = "donor" if donor_selected else "own"
        size = self.donor_size if donor_selected else self.own_size
        return MixedReplayKey(origin, self._word(cursor, 1) % size, int(cursor))

    def batch(
        self,
        *,
        start_cursor: int,
        count: int,
        own_positions_generated: int,
    ) -> tuple[MixedReplayKey, ...]:
        if count < 0:
            raise ValueError("sample count must be non-negative")
        return tuple(
            self.key(cursor, own_positions_generated=own_positions_generated)
            for cursor in range(start_cursor, start_cursor + count)
        )


class StratifiedDonorPlanner:
    """Choose donor strata by declared weights, independent of stratum size."""

    def __init__(
        self,
        *,
        stratum_sizes: Mapping[str, int],
        stratum_weights: Mapping[str, float],
        seed: int,
    ) -> None:
        if set(stratum_sizes) != set(STRATA) or set(stratum_weights) != set(STRATA):
            raise ValueError("donor stratum sizes and weights must cover all four strata")
        if any(type(size) is not int or size < 1 for size in stratum_sizes.values()):
            raise ValueError("every donor stratum must contain at least one sample")
        weights = tuple(float(stratum_weights[name]) for name in STRATA)
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
            raise ValueError("donor stratum weights must be finite and positive")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("donor stratum weights must sum to one")
        self.stratum_sizes = {name: int(stratum_sizes[name]) for name in STRATA}
        self.stratum_weights = {name: float(stratum_weights[name]) for name in STRATA}
        self.seed = int(seed)

    def _word(self, cursor: int, stream: int) -> int:
        if cursor < 0:
            raise ValueError("sample cursor must be non-negative")
        payload = f"{self.seed}:donor:{cursor}:{stream}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def key(self, cursor: int) -> TransferSampleKey:
        draw = self._word(cursor, 0) / float(2**64)
        cumulative = 0.0
        stratum = STRATA[-1]
        for name in STRATA:
            cumulative += self.stratum_weights[name]
            if draw < cumulative:
                stratum = name
                break
        index = self._word(cursor, 1) % self.stratum_sizes[stratum]
        return TransferSampleKey("donor", stratum, index, int(cursor))


class CrossScaleSamplePlanner:
    """Compose online donor/own decay with fixed donor-stage stratification."""

    def __init__(
        self,
        *,
        donor_stratum_sizes: Mapping[str, int],
        donor_stratum_weights: Mapping[str, float],
        own_size: int,
        schedule: TransferSchedule,
        seed: int,
    ) -> None:
        self.donor = StratifiedDonorPlanner(
            stratum_sizes=donor_stratum_sizes,
            stratum_weights=donor_stratum_weights,
            seed=seed,
        )
        self.origin = MixedReplayPlanner(
            donor_size=sum(donor_stratum_sizes.values()),
            own_size=own_size,
            schedule=schedule,
            seed=seed,
        )

    def key(self, cursor: int, *, own_positions_generated: int) -> TransferSampleKey:
        origin = self.origin.key(cursor, own_positions_generated=own_positions_generated)
        if origin.origin == "donor":
            return self.donor.key(cursor)
        return TransferSampleKey("own", None, origin.index, origin.cursor)

    def batch(
        self,
        *,
        start_cursor: int,
        count: int,
        own_positions_generated: int,
    ) -> tuple[TransferSampleKey, ...]:
        if count < 0:
            raise ValueError("sample count must be non-negative")
        return tuple(
            self.key(cursor, own_positions_generated=own_positions_generated)
            for cursor in range(start_cursor, start_cursor + count)
        )


@dataclass
class TransferLedger:
    own_positions_generated: int = 0
    donor_positions_consumed: int = 0
    own_positions_consumed: int = 0
    sample_cursor: int = 0

    def record_generated_own(self, count: int) -> None:
        if count < 0:
            raise ValueError("generated own positions cannot be negative")
        self.own_positions_generated += int(count)

    def record_batch(self, keys: Sequence[MixedReplayKey]) -> None:
        if any(key.cursor != self.sample_cursor + index for index, key in enumerate(keys)):
            raise ValueError("mixed replay keys do not continue the ledger cursor")
        self.donor_positions_consumed += sum(key.origin == "donor" for key in keys)
        self.own_positions_consumed += sum(key.origin == "own" for key in keys)
        self.sample_cursor += len(keys)

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, raw: Mapping[str, Any]) -> "TransferLedger":
        if set(raw) != {
            "own_positions_generated",
            "donor_positions_consumed",
            "own_positions_consumed",
            "sample_cursor",
        }:
            raise ValueError("transfer ledger fields do not match schema V1")
        values = {key: int(value) for key, value in raw.items()}
        if any(value < 0 for value in values.values()):
            raise ValueError("transfer ledger values must be non-negative")
        if values["donor_positions_consumed"] + values["own_positions_consumed"] != values["sample_cursor"]:
            raise ValueError("transfer ledger consumed counts do not match sample_cursor")
        return cls(**values)


def write_donor_qualification(
    output_path: str | Path,
    *,
    bundle_dir: str | Path,
    opening_manifest_path: str | Path,
    openings: Sequence[Opening],
    candidate_identity: Mapping[str, Any],
    donor_identity: Mapping[str, Any],
    candidate_predictor: Any,
    donor_predictor: Any,
    search_sims: int,
    cpuct: float,
    confidence: float,
    bootstrap_samples: int,
    role_floor: float,
    accept_threshold: float = 0.5,
    parallel_games: int = 1,
    inference_batch_size: int = 1,
    inference_batch_timeout_s: float = 0.001,
) -> Path:
    """Run and immutably write a cross-size qualification gate artifact."""

    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"donor qualification artifact already exists: {target}")
    bundle = validate_cross_scale_bundle(bundle_dir)
    if not openings:
        raise ValueError("donor qualification requires at least one opening pair")
    if any(opening.rule_id != bundle["semantic_contract"]["rule_id"] for opening in openings):
        raise ValueError("qualification openings do not match the donor bundle rule")
    required_identity = {"model_id", "checksum_sha256"}
    for label, identity in (("candidate", candidate_identity), ("donor", donor_identity)):
        if not required_identity.issubset(identity) or any(not identity[key] for key in required_identity):
            raise ValueError(f"{label} identity needs model_id and checksum_sha256")
    if candidate_identity["model_id"] == donor_identity["model_id"]:
        raise ValueError("candidate and donor must be different models")
    if donor_identity["model_id"] != bundle["qualification_donor_model_id"]:
        raise ValueError("qualification donor differs from the bundle's frozen donor")
    donor_attestation = next(
        row
        for row in bundle["producer_attestations"]
        if row["model_id"] == bundle["qualification_donor_model_id"]
    )
    if donor_identity["checksum_sha256"] != donor_attestation["artifact_sha256"]:
        raise ValueError("qualification donor checksum differs from the bundle attestation")
    if parallel_games < 1 or inference_batch_size < 1 or inference_batch_timeout_s < 0.0:
        raise ValueError("qualification evaluation parallel/batch settings are invalid")
    if parallel_games == 1 and inference_batch_size == 1:
        started = time.perf_counter()
        results = play_paired_openings(
            openings,
            candidate_predictor=candidate_predictor,
            incumbent_predictor=donor_predictor,
            search_sims=search_sims,
            cpuct=cpuct,
        )
        wall_seconds = time.perf_counter() - started
        evaluation_runtime: dict[str, Any] = {
            "parallel_games": 1,
            "games": len(results),
            "wall_seconds": wall_seconds,
            "games_per_second": len(results) / max(wall_seconds, 1e-12),
            "inference_services": [],
        }
    else:
        evaluated = play_paired_openings_parallel(
            openings,
            candidate_predictor=candidate_predictor,
            incumbent_predictor=donor_predictor,
            search_sims=search_sims,
            cpuct=cpuct,
            parallel_games=parallel_games,
            inference_batch_size=inference_batch_size,
            inference_batch_timeout_s=inference_batch_timeout_s,
        )
        results = evaluated.games
        evaluation_runtime = evaluated.metrics.to_dict()
    decision = evaluate_gate(
        results,
        confidence=confidence,
        bootstrap_seed=int(openings[0].seed) + 9109,
        bootstrap_samples=bootstrap_samples,
        accept_threshold=accept_threshold,
        role_floor=role_floor,
    )
    payload: dict[str, Any] = {
        "format": QUALIFICATION_FORMAT,
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "bundle_content_sha256": bundle["bundle_content_sha256"],
        "semantic_hash": bundle["semantic_hash"],
        "opening_manifest_path": str(Path(opening_manifest_path).resolve()),
        "opening_manifest_sha256": sha256_file(opening_manifest_path),
        "opening_ids": [opening.opening_id for opening in openings],
        "candidate": dict(candidate_identity),
        "donor": dict(donor_identity),
        "search": {"simulations": int(search_sims), "cpuct": float(cpuct)},
        "evaluation_runtime": evaluation_runtime,
        "statistics": {
            "confidence": float(confidence),
            "bootstrap_samples": int(bootstrap_samples),
            "role_floor": float(role_floor),
            "accept_threshold": float(accept_threshold),
        },
        "results": [asdict(result) for result in results],
        "decision": decision.to_dict(),
        "qualification_passed": decision.verdict == "accept",
        "automatic_promotion": False,
        "replay_generation_authorized": False,
    }
    payload["content_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    _atomic_write_json(target, payload)
    return target


def load_donor_qualification(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != QUALIFICATION_FORMAT or payload.get("schema_version") != 1:
        raise ValueError("unsupported donor qualification artifact")
    expected = _sha256_json({key: value for key, value in payload.items() if key != "content_sha256"})
    if payload.get("content_sha256") != expected:
        raise ValueError("donor qualification content hash mismatch")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 2 * len(payload.get("opening_ids", [])):
        raise ValueError("donor qualification lacks exact paired-opening evidence")
    expected_result_keys = {"opening_id", "seed", "candidate_is_first", "candidate_score"}
    if any(not isinstance(row, dict) or set(row) != expected_result_keys for row in results):
        raise ValueError("donor qualification result fields do not match the gate schema")
    coerced = tuple(GateGameResult(**row) for row in results)
    opening_ids = payload["opening_ids"]
    for opening_id in opening_ids:
        rows = [row for row in coerced if row.opening_id == opening_id]
        if len(rows) != 2 or {row.candidate_is_first for row in rows} != {True, False}:
            raise ValueError("donor qualification is missing a role-swapped opening pair")
    if {row.opening_id for row in coerced} != set(opening_ids):
        raise ValueError("donor qualification results reference unexpected openings")
    if payload.get("automatic_promotion") is not False or payload.get("replay_generation_authorized") is not False:
        raise ValueError("qualification evidence must not mutate training state")
    if payload.get("qualification_passed") != (payload.get("decision", {}).get("verdict") == "accept"):
        raise ValueError("qualification pass flag differs from its gate decision")
    return payload


__all__ = [
    "ACTION_SCHEMA",
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BundleSource",
    "CrossScaleSamplePlanner",
    "MixedReplayKey",
    "MixedReplayPlanner",
    "QUALIFICATION_FORMAT",
    "STRATA",
    "StratifiedDonorPlanner",
    "TransferLedger",
    "TransferSchedule",
    "TransferSampleKey",
    "TransferStage",
    "build_cross_scale_bundle",
    "load_bundle_source_spec",
    "load_cross_scale_replay",
    "load_donor_qualification",
    "replay_semantic_contract",
    "replay_semantic_hash",
    "validate_cross_scale_bundle",
    "write_donor_qualification",
]
