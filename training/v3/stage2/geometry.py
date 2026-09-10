"""Deterministic geometry-stratified evaluation for Stage 2A-R3.

The suite is deliberately independent from replay and training.  Boards are
canonical (+1 is the side to move), while ``absolute_role`` retains whether
that side is the absolute first or second player.  Policy actions always use
the V3 row-major 5x5 column contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from connect4_core.rules import FEATURE_DIM

from ..replay import BOARD_SHAPE, COLUMN_COUNT, apply_d4, inverse_d4_index, sha256_file
from ..search import role_features_for_player
from ..config import ModelConfig, model_config_dict
from ..model import build_model


GEOMETRY_SUITE_SCHEMA = "connect4-v3-stage2-geometry-suite-v1"
GEOMETRY_REPORT_SCHEMA = "connect4-v3-stage2-geometry-report-v1"
GEOMETRY_CLASSES = (
    "xy",
    "z",
    "xz",
    "yz",
    "winning_section",
    "space_diagonal",
)
TACTICAL_TYPES = ("immediate_win", "forced_block", "double_threat")
WDL_TARGETS = {"win": 0, "draw": 1, "loss": 2}
BOOTSTRAP_SAMPLES = 1_000
ECE_BINS = 10

_SUITE_KEYS = frozenset({"schema", "suite_id", "rule_features", "cases"})
_CASE_KEYS = frozenset(
    {
        "case_id",
        "canonical_board",
        "absolute_role",
        "legal_actions",
        "target_actions",
        "geometry",
        "tactical_type",
        "wdl_target",
    }
)


@dataclass(frozen=True)
class GeometryCase:
    case_id: str
    canonical_board: np.ndarray
    absolute_role: str
    legal_actions: tuple[int, ...]
    target_actions: tuple[int, ...]
    geometry: str
    tactical_type: str
    wdl_target: str


@dataclass(frozen=True)
class GeometrySuite:
    suite_id: str
    rule_features: np.ndarray
    cases: tuple[GeometryCase, ...]
    content_sha256: str


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{where} keys differ: missing={missing}, unknown={unknown}")


def _json_payload(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(raw, Mapping):
        raise ValueError("geometry suite root must be a JSON object")
    return raw


def _strict_board(value: Any, where: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != BOARD_SHAPE[0]:
        raise ValueError(f"{where} must be a JSON array with shape [6,5,5]")
    normalized: list[list[list[int]]] = []
    for z, layer in enumerate(value):
        if not isinstance(layer, list) or len(layer) != BOARD_SHAPE[1]:
            raise ValueError(f"{where}[{z}] must contain 5 rows")
        normalized_layer: list[list[int]] = []
        for y, row in enumerate(layer):
            if not isinstance(row, list) or len(row) != BOARD_SHAPE[2]:
                raise ValueError(f"{where}[{z}][{y}] must contain 5 cells")
            normalized_row: list[int] = []
            for x, cell in enumerate(row):
                if isinstance(cell, bool) or not isinstance(cell, Integral) or int(cell) not in (-1, 0, 1):
                    raise ValueError(f"{where}[{z}][{y}][{x}] must be integer -1/0/1")
                normalized_row.append(int(cell))
            normalized_layer.append(normalized_row)
        normalized.append(normalized_layer)
    board = np.asarray(normalized, dtype=np.int8)
    occupied = board != 0
    for y in range(5):
        for x in range(5):
            column = occupied[:, y, x]
            if np.any(column[1:] & ~column[:-1]):
                raise ValueError(f"{where} contains a gravity hole at column ({y},{x})")
    board.setflags(write=False)
    return board


def _strict_actions(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty JSON array")
    actions: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{where} entries must be integer actions")
        action = int(item)
        if not 0 <= action < COLUMN_COUNT:
            raise ValueError(f"{where} actions must be in [0,24]")
        actions.append(action)
    if actions != sorted(set(actions)):
        raise ValueError(f"{where} must be strictly increasing and unique")
    return tuple(actions)


def _strict_rule_features(value: Any) -> np.ndarray:
    if not isinstance(value, list) or len(value) != FEATURE_DIM:
        raise ValueError(f"rule_features must be a JSON array with {FEATURE_DIM} entries")
    features: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real) or not np.isfinite(float(item)):
            raise ValueError("rule_features entries must be finite JSON numbers")
        features.append(float(item))
    result = np.asarray(features, dtype=np.float32)
    result.setflags(write=False)
    return result


def _normalized_payload(suite_id: str, rules: np.ndarray, cases: Sequence[GeometryCase]) -> dict[str, Any]:
    return {
        "schema": GEOMETRY_SUITE_SCHEMA,
        "suite_id": suite_id,
        "rule_features": [float(value) for value in rules],
        "cases": [
            {
                "case_id": case.case_id,
                "canonical_board": case.canonical_board.tolist(),
                "absolute_role": case.absolute_role,
                "legal_actions": list(case.legal_actions),
                "target_actions": list(case.target_actions),
                "geometry": case.geometry,
                "tactical_type": case.tactical_type,
                "wdl_target": case.wdl_target,
            }
            for case in cases
        ],
    }


def load_geometry_suite(source: str | Path | Mapping[str, Any]) -> GeometrySuite:
    """Load and strictly validate the versioned JSON geometry suite."""

    raw = _json_payload(source)
    _exact_keys(raw, _SUITE_KEYS, "geometry suite")
    if raw["schema"] != GEOMETRY_SUITE_SCHEMA:
        raise ValueError(f"geometry suite schema must be {GEOMETRY_SUITE_SCHEMA!r}")
    suite_id = raw["suite_id"]
    if not isinstance(suite_id, str) or not suite_id or suite_id.strip() != suite_id:
        raise ValueError("suite_id must be a non-empty trimmed string")
    rules = _strict_rule_features(raw["rule_features"])
    raw_cases = raw["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty JSON array")

    cases: list[GeometryCase] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(raw_cases):
        where = f"cases[{index}]"
        if not isinstance(value, Mapping):
            raise ValueError(f"{where} must be a JSON object")
        _exact_keys(value, _CASE_KEYS, where)
        case_id = value["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id.strip() != case_id:
            raise ValueError(f"{where}.case_id must be a non-empty trimmed string")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id: {case_id!r}")
        seen_ids.add(case_id)
        board = _strict_board(value["canonical_board"], f"{where}.canonical_board")
        role = value["absolute_role"]
        if role not in ("first", "second"):
            raise ValueError(f"{where}.absolute_role must be 'first' or 'second'")
        legal = _strict_actions(value["legal_actions"], f"{where}.legal_actions")
        targets = _strict_actions(value["target_actions"], f"{where}.target_actions")
        if not set(targets).issubset(legal):
            raise ValueError(f"{where}.target_actions must be a subset of legal_actions")
        heights = np.count_nonzero(board, axis=0).reshape(-1)
        if any(heights[action] >= BOARD_SHAPE[0] for action in legal):
            raise ValueError(f"{where}.legal_actions contains a full column")
        geometry = value["geometry"]
        if geometry not in GEOMETRY_CLASSES:
            raise ValueError(f"{where}.geometry must be one of {GEOMETRY_CLASSES}")
        tactical_type = value["tactical_type"]
        if tactical_type not in TACTICAL_TYPES:
            raise ValueError(f"{where}.tactical_type must be one of {TACTICAL_TYPES}")
        wdl_target = value["wdl_target"]
        if wdl_target not in WDL_TARGETS:
            raise ValueError(f"{where}.wdl_target must be one of {tuple(WDL_TARGETS)}")
        cases.append(
            GeometryCase(
                case_id=case_id,
                canonical_board=board,
                absolute_role=role,
                legal_actions=legal,
                target_actions=targets,
                geometry=geometry,
                tactical_type=tactical_type,
                wdl_target=wdl_target,
            )
        )

    canonical = json.dumps(
        _normalized_payload(suite_id, rules, cases),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return GeometrySuite(
        suite_id=suite_id,
        rule_features=rules,
        cases=tuple(cases),
        content_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def apply_d4_actions(actions: Sequence[int], transform: int) -> tuple[int, ...]:
    """Map row-major V3 column actions through the replay D4 convention."""

    mask = np.zeros((5, 5), dtype=np.uint8)
    checked: list[int] = []
    for item in actions:
        if isinstance(item, bool) or not isinstance(item, Integral) or not 0 <= int(item) < 25:
            raise ValueError("D4 actions must be integer columns in [0,24]")
        checked.append(int(item))
    if len(set(checked)) != len(checked):
        raise ValueError("D4 actions must be unique")
    mask.reshape(-1)[checked] = 1
    return tuple(int(index) for index in np.flatnonzero(apply_d4(mask, transform).reshape(-1)))


def _extract_outputs(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, Mapping):
        try:
            policy_logits = output["policy_logits"]
            wdl_logits = output["wdl_logits"]
        except KeyError as exc:
            raise ValueError("model mapping output needs policy_logits and wdl_logits") from exc
    elif hasattr(output, "policy_logits") and hasattr(output, "wdl_logits"):
        policy_logits = output.policy_logits
        wdl_logits = output.wdl_logits
    elif isinstance(output, (tuple, list)) and len(output) >= 2:
        policy_logits, wdl_logits = output[:2]
    else:
        raise TypeError("model must return V3 policy_logits and wdl_logits")
    if not torch.is_tensor(policy_logits) or tuple(policy_logits.shape[1:]) != (25,):
        raise ValueError("policy_logits must have shape [N,25]")
    if not torch.is_tensor(wdl_logits) or tuple(wdl_logits.shape[1:]) != (3,):
        raise ValueError("wdl_logits must have shape [N,3]")
    if policy_logits.shape[0] != wdl_logits.shape[0]:
        raise ValueError("policy and WDL batch dimensions differ")
    return policy_logits, wdl_logits


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _ece(confidence: np.ndarray, correct: np.ndarray) -> float:
    if confidence.size == 0:
        return 0.0
    bins = np.minimum((confidence * ECE_BINS).astype(np.int64), ECE_BINS - 1)
    total = float(confidence.size)
    result = 0.0
    for bin_index in range(ECE_BINS):
        active = bins == bin_index
        if np.any(active):
            result += float(active.sum()) / total * abs(
                float(correct[active].mean()) - float(confidence[active].mean())
            )
    return result


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"geometry-bootstrap-v1:{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _summarize(
    rows: Mapping[str, np.ndarray], indices: np.ndarray, *, seed: int, label: str
) -> dict[str, Any]:
    count = int(indices.size)
    if count == 0:
        return {"case_count": 0, "metrics": {}}

    mean_metrics = (
        "policy_top1",
        "policy_top3",
        "policy_target_rank",
        "policy_ce",
        "wdl_ce",
        "wdl_brier",
        "d4_policy_consistency",
    )

    def aggregate(sample: np.ndarray) -> dict[str, float]:
        result = {name: float(rows[name][sample].mean()) for name in mean_metrics}
        result["wdl_ece"] = _ece(rows["wdl_confidence"][sample], rows["wdl_correct"][sample])
        return result

    point = aggregate(indices)
    rng = np.random.default_rng(_derived_seed(seed, label))
    bootstrap: dict[str, list[float]] = {name: [] for name in point}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = indices[rng.integers(0, count, size=count)]
        values = aggregate(sample)
        for name, value in values.items():
            bootstrap[name].append(value)
    metrics = {}
    for name, value in point.items():
        low, high = np.quantile(np.asarray(bootstrap[name]), (0.025, 0.975))
        metrics[name] = {"value": value, "ci95": [float(low), float(high)]}
    return {"case_count": count, "metrics": metrics}


def evaluate_geometry_suite(
    model: torch.nn.Module,
    suite: GeometrySuite | str | Path | Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Evaluate a model on all cases and their eight horizontal D4 views."""

    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    checked = suite if isinstance(suite, GeometrySuite) else load_geometry_suite(suite)
    seed = int(seed)
    boards: list[np.ndarray] = []
    roles: list[np.ndarray] = []
    rules: list[np.ndarray] = []
    legal_rows: list[np.ndarray] = []
    for case in checked.cases:
        role = role_features_for_player(1 if case.absolute_role == "first" else -1)
        for transform in range(8):
            boards.append(apply_d4(case.canonical_board, transform))
            roles.append(role)
            rules.append(checked.rule_features)
            legal = np.zeros(25, dtype=bool)
            legal[list(apply_d4_actions(case.legal_actions, transform))] = True
            legal_rows.append(legal)

    device = _model_device(model)
    board_tensor = torch.as_tensor(np.stack(boards), dtype=torch.float32, device=device)
    role_tensor = torch.as_tensor(np.stack(roles), dtype=torch.float32, device=device)
    rule_tensor = torch.as_tensor(np.stack(rules), dtype=torch.float32, device=device)
    legal_tensor = torch.as_tensor(np.stack(legal_rows), dtype=torch.bool, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model.forward_search(
                board_tensor,
                role_to_play=role_tensor,
                rule_features=rule_tensor,
            )
            policy_logits, wdl_logits = _extract_outputs(output)
            expected = len(checked.cases) * 8
            if policy_logits.shape[0] != expected:
                raise ValueError(f"model returned {policy_logits.shape[0]} rows, expected {expected}")
            if not torch.isfinite(policy_logits).all() or not torch.isfinite(wdl_logits).all():
                raise ValueError("model returned non-finite policy or WDL logits")
            masked_logits = policy_logits.float().masked_fill(~legal_tensor, -torch.inf)
            policy = torch.softmax(masked_logits, dim=1).cpu().numpy().astype(np.float64)
            wdl = torch.softmax(wdl_logits.float(), dim=1).cpu().numpy().astype(np.float64)
    finally:
        model.train(was_training)

    collected: dict[str, list[float]] = {
        "policy_top1": [],
        "policy_top3": [],
        "policy_target_rank": [],
        "policy_ce": [],
        "wdl_ce": [],
        "wdl_brier": [],
        "wdl_confidence": [],
        "wdl_correct": [],
        "d4_policy_consistency": [],
    }
    case_rows: list[dict[str, Any]] = []
    epsilon = np.finfo(np.float64).tiny
    for case_index, case in enumerate(checked.cases):
        start = case_index * 8
        base_policy = policy[start]
        ordering = np.lexsort((np.arange(25), -base_policy))
        rank_by_action = np.empty(25, dtype=np.int64)
        rank_by_action[ordering] = np.arange(1, 26)
        target_set = set(case.target_actions)
        predicted = int(ordering[0])
        top3 = set(int(action) for action in ordering[:3])
        target_rank = min(int(rank_by_action[action]) for action in case.target_actions)
        policy_ce = -float(np.log(np.maximum(base_policy[list(case.target_actions)], epsilon)).mean())

        wdl_target = WDL_TARGETS[case.wdl_target]
        base_wdl = wdl[start]
        one_hot = np.zeros(3, dtype=np.float64)
        one_hot[wdl_target] = 1.0
        d4_scores: list[float] = []
        for transform in range(1, 8):
            restored = apply_d4(
                policy[start + transform].reshape(5, 5), inverse_d4_index(transform)
            ).reshape(25)
            d4_scores.append(1.0 - 0.5 * float(np.abs(base_policy - restored).sum()))
        values = {
            "policy_top1": float(predicted in target_set),
            "policy_top3": float(bool(top3 & target_set)),
            "policy_target_rank": float(target_rank),
            "policy_ce": policy_ce,
            "wdl_ce": -float(np.log(max(base_wdl[wdl_target], epsilon))),
            "wdl_brier": float(np.square(base_wdl - one_hot).sum()),
            "wdl_confidence": float(base_wdl.max()),
            "wdl_correct": float(int(base_wdl.argmax()) == wdl_target),
            "d4_policy_consistency": float(np.mean(d4_scores)),
        }
        for name, value in values.items():
            collected[name].append(value)
        case_rows.append(
            {
                "case_id": case.case_id,
                "geometry": case.geometry,
                "tactical_type": case.tactical_type,
                "predicted_action": predicted,
                "target_rank": target_rank,
            }
        )

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in collected.items()}
    all_indices = np.arange(len(checked.cases), dtype=np.int64)
    by_geometry = {
        geometry: _summarize(
            arrays,
            np.asarray(
                [i for i, case in enumerate(checked.cases) if case.geometry == geometry],
                dtype=np.int64,
            ),
            seed=seed,
            label=f"geometry:{geometry}",
        )
        for geometry in GEOMETRY_CLASSES
    }
    by_tactical = {
        tactical: _summarize(
            arrays,
            np.asarray(
                [i for i, case in enumerate(checked.cases) if case.tactical_type == tactical],
                dtype=np.int64,
            ),
            seed=seed,
            label=f"tactical:{tactical}",
        )
        for tactical in TACTICAL_TYPES
    }
    by_cross = {}
    for geometry in GEOMETRY_CLASSES:
        for tactical in TACTICAL_TYPES:
            label = f"{geometry}/{tactical}"
            indices = np.asarray(
                [
                    i
                    for i, case in enumerate(checked.cases)
                    if case.geometry == geometry and case.tactical_type == tactical
                ],
                dtype=np.int64,
            )
            by_cross[label] = _summarize(arrays, indices, seed=seed, label=f"cross:{label}")

    return {
        "schema": GEOMETRY_REPORT_SCHEMA,
        "suite_schema": GEOMETRY_SUITE_SCHEMA,
        "suite_id": checked.suite_id,
        "suite_sha256": checked.content_sha256,
        "seed": seed,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "ece_bins": ECE_BINS,
        "coverage": {
            "geometry_present": sorted({case.geometry for case in checked.cases}),
            "geometry_missing": sorted(set(GEOMETRY_CLASSES) - {case.geometry for case in checked.cases}),
            "tactical_present": sorted({case.tactical_type for case in checked.cases}),
            "tactical_missing": sorted(set(TACTICAL_TYPES) - {case.tactical_type for case in checked.cases}),
        },
        "overall": _summarize(arrays, all_indices, seed=seed, label="overall"),
        "by_geometry": by_geometry,
        "by_tactical_type": by_tactical,
        "by_geometry_tactical": by_cross,
        "cases": case_rows,
    }


def evaluate_geometry_artifact(
    *,
    config_path: str | Path,
    artifact_path: str | Path,
    suite_path: str | Path,
    seed: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a strict Stage 2 offline artifact and evaluate the frozen suite."""

    config_target = Path(config_path)
    artifact_target = Path(artifact_path)
    raw_config = json.loads(config_target.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping) or not isinstance(raw_config.get("model"), Mapping):
        raise ValueError("geometry evaluation config must contain a model mapping")
    model_config = ModelConfig(**dict(raw_config["model"]))
    payload = torch.load(artifact_target, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("geometry model artifact must contain a mapping")
    if payload.get("format") != "connect4-v3-model" or payload.get("format_version") != 1:
        raise ValueError("unsupported geometry model artifact")
    if payload.get("model_config") != model_config_dict(model_config):
        raise ValueError("geometry artifact model_config differs from the evaluation config")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("geometry model artifact is missing model_state")
    model = build_model(model_config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    report = evaluate_geometry_suite(model, suite_path, seed)
    report["model"] = model_config_dict(model_config)
    report["config_sha256"] = sha256_file(config_target)
    report["artifact_sha256"] = sha256_file(artifact_target)
    return report


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "GEOMETRY_CLASSES",
    "GEOMETRY_REPORT_SCHEMA",
    "GEOMETRY_SUITE_SCHEMA",
    "GeometryCase",
    "GeometrySuite",
    "TACTICAL_TYPES",
    "WDL_TARGETS",
    "apply_d4_actions",
    "evaluate_geometry_suite",
    "evaluate_geometry_artifact",
    "load_geometry_suite",
]
