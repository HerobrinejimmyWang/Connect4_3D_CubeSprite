"""Fail-closed evidence check for starting Stage 2A-R3 training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..replay import sha256_file
from .round3 import build_round3_design


EVIDENCE_SCHEMA = "connect4-v3-stage2-r2-primary-evidence-v1"
RECEIPT_SCHEMA = "connect4-v3-stage2-round3-readiness-v1"
_TERMINAL_STATUS = "stopped_at_safe_boundary"
_COMPLETE_OUTCOME = "complete_at_bound"
_GUARD_OUTCOME = "safe_guard_stop"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    name: str,
) -> None:
    optional = optional or set()
    missing = required.difference(value)
    unknown = set(value).difference(required | optional)
    if missing or unknown:
        raise ValueError(f"{name} missing={sorted(missing)} unknown={sorted(unknown)}")


def _lower_sha256(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def _resolve_artifact(
    raw: Any,
    *,
    repo_root: Path,
    name: str,
) -> dict[str, str]:
    artifact = _mapping(raw, name)
    _exact_keys(artifact, required={"path", "sha256"}, name=name)
    relative = Path(str(artifact["path"]))
    path = relative if relative.is_absolute() else repo_root / relative
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    expected = _lower_sha256(artifact["sha256"], f"{name}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected={expected} actual={actual}")
    return {"path": str(path), "sha256": actual}


def _required_pairs(dependency: Mapping[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    rows = dependency.get("required_lines")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Round 3 design dependency.required_lines must be non-empty")
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"dependency.required_lines[{index}]")
        _exact_keys(row, required={"architecture", "initializations"}, name="required line")
        architecture = str(row["architecture"])
        initializations = row["initializations"]
        if not architecture or initializations != ["cold", "warm"]:
            raise ValueError("every required architecture must declare cold then warm")
        for initialization in initializations:
            pair = (architecture, str(initialization))
            if pair in pairs:
                raise ValueError(f"duplicate required readiness line: {pair}")
            pairs.add(pair)
    return pairs


def _validate_terminal(
    raw: Any,
    *,
    target_positions: int,
    name: str,
) -> tuple[dict[str, Any], int]:
    terminal = _mapping(raw, name)
    _exact_keys(
        terminal,
        required={
            "outcome",
            "status",
            "stop_reason",
            "train_positions_consumed",
            "safe_boundary",
        },
        optional={"guard_reason"},
        name=name,
    )
    if terminal["status"] != _TERMINAL_STATUS or terminal["safe_boundary"] is not True:
        raise ValueError(f"{name} is not an explicit safe-boundary terminal result")
    consumed = terminal["train_positions_consumed"]
    if not isinstance(consumed, int) or isinstance(consumed, bool) or consumed <= 0:
        raise ValueError(f"{name}.train_positions_consumed must be a positive integer")
    outcome = terminal["outcome"]
    if outcome == _COMPLETE_OUTCOME:
        if terminal["stop_reason"] != "max_train_positions" or consumed < target_positions:
            raise ValueError(f"{name} does not prove completion at {target_positions} positions")
        if terminal.get("guard_reason"):
            raise ValueError(f"{name} complete_at_bound cannot also carry a guard reason")
    elif outcome == _GUARD_OUTCOME:
        guard_reason = terminal.get("guard_reason")
        if not isinstance(guard_reason, str) or not guard_reason.strip():
            raise ValueError(f"{name} safe_guard_stop requires a non-empty guard_reason")
        if terminal["stop_reason"] == "max_train_positions":
            raise ValueError(f"{name} safe_guard_stop cannot use max_train_positions")
    else:
        raise ValueError(f"{name}.outcome must be {_COMPLETE_OUTCOME} or {_GUARD_OUTCOME}")
    return dict(terminal), consumed


def _validate_elo(raw: Any, *, terminal_positions: int, name: str) -> dict[str, Any]:
    elo = _mapping(raw, name)
    _exact_keys(
        elo,
        required={
            "artifact",
            "status",
            "fixed_openings",
            "color_swapped",
            "opening_pairs",
            "observed_terminal_positions",
        },
        name=name,
    )
    if elo["status"] != "complete":
        raise ValueError(f"{name} is not complete")
    if elo["fixed_openings"] is not True or elo["color_swapped"] is not True:
        raise ValueError(f"{name} must use fixed openings with colors swapped")
    pairs = elo["opening_pairs"]
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
        raise ValueError(f"{name}.opening_pairs must be a positive integer")
    if elo["observed_terminal_positions"] != terminal_positions:
        raise ValueError(f"{name} was not frozen against the line terminal position")
    return dict(elo)


def verify_round3_readiness(
    evidence_path: str | Path,
    *,
    repo_root: str | Path,
    design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify all primary-seed dependencies and return a hash-bound receipt.

    The checker reads evidence only.  It does not launch a process, modify the
    Stage 2B queue, or infer missing evidence from filenames.
    """

    root = Path(repo_root).resolve()
    source = Path(evidence_path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    evidence = _mapping(raw, "evidence")
    _exact_keys(
        evidence,
        required={
            "schema",
            "dependency_id",
            "primary_seed",
            "target_positions",
            "lines",
            "queue_state",
            "archive",
        },
        optional={"notes"},
        name="evidence",
    )
    round3 = dict(design) if design is not None else build_round3_design()
    if round3.get("schema") != "connect4-v3-stage2-round3-design-v1":
        raise ValueError("unsupported Round 3 design schema")
    dependency = _mapping(round3.get("dependency"), "design.dependency")
    if evidence["schema"] != EVIDENCE_SCHEMA:
        raise ValueError("unsupported Round 2 primary evidence schema")
    if evidence["dependency_id"] != dependency.get("id"):
        raise ValueError("evidence dependency id differs from the Round 3 design")
    primary_seed = dependency.get("primary_seed")
    target_positions = dependency.get("target_positions")
    if evidence["primary_seed"] != primary_seed or evidence["target_positions"] != target_positions:
        raise ValueError("evidence seed/position scope differs from the Round 3 design")
    if not isinstance(primary_seed, int) or not isinstance(target_positions, int):
        raise ValueError("Round 3 dependency seed and target positions must be integers")

    required_pairs = _required_pairs(dependency)
    raw_lines = evidence["lines"]
    if not isinstance(raw_lines, list):
        raise ValueError("evidence.lines must be a list")
    observed_pairs: set[tuple[str, str]] = set()
    receipt_lines: list[dict[str, Any]] = []
    for index, raw_line in enumerate(raw_lines):
        line_name = f"evidence.lines[{index}]"
        line = _mapping(raw_line, line_name)
        _exact_keys(
            line,
            required={
                "run_id",
                "architecture",
                "initialization",
                "seed",
                "terminal",
                "checkpoint",
                "checkpoint_config_hash",
                "report",
                "report_status",
                "elo",
            },
            name=line_name,
        )
        if line["seed"] != primary_seed:
            raise ValueError(f"{line_name} is not from the primary seed")
        pair = (str(line["architecture"]), str(line["initialization"]))
        if pair not in required_pairs:
            raise ValueError(f"unexpected readiness line: {pair}")
        if pair in observed_pairs:
            raise ValueError(f"duplicate readiness line: {pair}")
        observed_pairs.add(pair)
        run_id = str(line["run_id"])
        if not run_id:
            raise ValueError(f"{line_name}.run_id must be non-empty")
        terminal, consumed = _validate_terminal(
            line["terminal"], target_positions=target_positions, name=f"{line_name}.terminal"
        )
        checkpoint = _resolve_artifact(
            line["checkpoint"], repo_root=root, name=f"{line_name}.checkpoint"
        )
        config_hash = _lower_sha256(
            line["checkpoint_config_hash"], f"{line_name}.checkpoint_config_hash"
        )
        if line["report_status"] != "complete":
            raise ValueError(f"{line_name}.report_status is not complete")
        report = _resolve_artifact(line["report"], repo_root=root, name=f"{line_name}.report")
        elo = _validate_elo(line["elo"], terminal_positions=consumed, name=f"{line_name}.elo")
        elo_artifact = _resolve_artifact(
            elo["artifact"], repo_root=root, name=f"{line_name}.elo.artifact"
        )
        receipt_lines.append(
            {
                "run_id": run_id,
                "architecture": pair[0],
                "initialization": pair[1],
                "seed": primary_seed,
                "terminal": terminal,
                "checkpoint": checkpoint,
                "checkpoint_config_hash": config_hash,
                "report": report,
                "elo": {
                    "artifact": elo_artifact,
                    "opening_pairs": elo["opening_pairs"],
                    "observed_terminal_positions": elo["observed_terminal_positions"],
                },
            }
        )
    if observed_pairs != required_pairs:
        missing = sorted(required_pairs.difference(observed_pairs))
        raise ValueError(f"evidence lacks required cold/warm lines: {missing}")

    queue_state = _resolve_artifact(
        evidence["queue_state"], repo_root=root, name="evidence.queue_state"
    )
    archive = _mapping(evidence["archive"], "evidence.archive")
    _exact_keys(
        archive,
        required={"status", "index", "receipts"},
        name="evidence.archive",
    )
    archive_index = _resolve_artifact(
        archive["index"], repo_root=root, name="evidence.archive.index"
    )
    raw_receipts = archive["receipts"]
    if not isinstance(raw_receipts, list):
        raise ValueError("evidence.archive.receipts must be a list")
    receipts = []
    for index, raw_receipt in enumerate(raw_receipts):
        receipt = _mapping(raw_receipt, f"evidence.archive.receipts[{index}]")
        _exact_keys(
            receipt,
            required={"artifact", "verified"},
            name=f"evidence.archive.receipts[{index}]",
        )
        if receipt["verified"] is not True:
            raise ValueError("every archive receipt must be explicitly verified")
        receipts.append(
            _resolve_artifact(
                receipt["artifact"],
                repo_root=root,
                name=f"evidence.archive.receipts[{index}].artifact",
            )
        )
    if archive["status"] == "verified_receipts":
        if not receipts:
            raise ValueError("verified_receipts archive status requires at least one receipt")
    elif archive["status"] == "no_archives_required":
        if receipts:
            raise ValueError("no_archives_required cannot list archive receipts")
    else:
        raise ValueError("unsupported evidence.archive.status")

    receipt_lines.sort(key=lambda row: (row["architecture"], row["initialization"]))
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "ready",
        "dependency_id": dependency["id"],
        "primary_seed": primary_seed,
        "target_positions": target_positions,
        "design_sha256": hashlib.sha256(
            json.dumps(round3, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "evidence": {"path": str(source), "sha256": sha256_file(source)},
        "queue_state": queue_state,
        "archive": {
            "status": archive["status"],
            "index": archive_index,
            "receipts": receipts,
        },
        "lines": receipt_lines,
        "execution_authorized": True,
        "execution_authorization": (
            "This receipt satisfies the declared R2 evidence dependency only; normal bounded "
            "execution and hardware/storage preflight remain mandatory."
        ),
    }
    receipt["content_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return receipt


__all__ = [
    "EVIDENCE_SCHEMA",
    "RECEIPT_SCHEMA",
    "verify_round3_readiness",
]
