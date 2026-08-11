"""Dependency and interpreter checks performed before a V3 command starts."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass


SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX = (3, 13)


class PreflightError(RuntimeError):
    """Raised when the current interpreter cannot run the V3 pipeline."""


@dataclass(frozen=True)
class PreflightReport:
    python: str
    numpy: str
    torch: str
    device: str


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise PreflightError(
            f"Missing required dependency '{name}'. Install "
            "training/requirements-v3.txt in this interpreter."
        )


def run_preflight(device: str) -> PreflightReport:
    version = sys.version_info[:2]
    if not (SUPPORTED_PYTHON_MIN <= version <= SUPPORTED_PYTHON_MAX):
        raise PreflightError(
            "V3 supports Python 3.11 through 3.13; "
            f"the current interpreter is {sys.version.split()[0]}."
        )
    _require_module("numpy")
    _require_module("torch")

    import numpy as np
    import torch

    normalized_device = str(device).lower()
    if normalized_device != "cpu" and not normalized_device.startswith("cuda"):
        raise PreflightError("device must be 'cpu' or a CUDA device such as 'cuda:0'.")
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        raise PreflightError(
            f"CUDA device '{device}' was requested, but torch.cuda.is_available() is false."
        )
    if normalized_device.startswith("cuda"):
        try:
            parsed_device = torch.device(normalized_device)
        except (RuntimeError, ValueError) as exc:
            raise PreflightError(f"Invalid CUDA device string: {device!r}.") from exc
        index = 0 if parsed_device.index is None else int(parsed_device.index)
        device_count = int(torch.cuda.device_count())
        if index < 0 or index >= device_count:
            raise PreflightError(
                f"CUDA device '{device}' is outside the visible inventory of {device_count} device(s)."
            )
    return PreflightReport(
        python=sys.version.split()[0],
        numpy=np.__version__,
        torch=torch.__version__,
        device=normalized_device,
    )
