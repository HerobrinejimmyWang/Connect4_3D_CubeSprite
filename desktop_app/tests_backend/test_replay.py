"""Mirror the canonical replay contract tests in both documented test suites."""

from pathlib import Path
from runpy import run_path


CANONICAL_TEST = Path(__file__).resolve().parents[1] / "backend" / "tests" / "test_replay.py"
ReplayServiceTests = run_path(
    str(CANONICAL_TEST),
    run_name="cubesprite_replay_contract_tests",
)["ReplayServiceTests"]
