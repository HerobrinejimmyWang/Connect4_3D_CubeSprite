"""Scratch-directory helper for the Stage 2 archive tests.

``tempfile.TemporaryDirectory`` and ``tempfile.mkdtemp`` create their directory
with mode ``0o700``.  Under the DSH file sandbox, writes *inside* a ``0o700``
directory are denied with ``WinError 5`` even though the creating user owns it,
so every archive test failed in ``setUp`` before it could exercise anything.

Creating the scratch directory with default permissions behaves identically in
a normal shell and stays runnable under the sandbox.  Cleanup still uses the
extended-length path helper, because archive tests deliberately build paths
longer than the legacy ``MAX_PATH`` limit.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from tools.migrate_stage2_archive import _fs_path


def make_scratch_dir(prefix: str = "stage2-archive-") -> Path:
    """Create a unique scratch directory with default (non-0o700) permissions."""

    base = Path(tempfile.gettempdir())
    root = base / f"{prefix}{os.getpid()}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def remove_scratch_dir(root: Path) -> None:
    """Remove a scratch directory, tolerating Windows long paths."""

    shutil.rmtree(_fs_path(root), ignore_errors=True)


__all__ = ["make_scratch_dir", "remove_scratch_dir"]
