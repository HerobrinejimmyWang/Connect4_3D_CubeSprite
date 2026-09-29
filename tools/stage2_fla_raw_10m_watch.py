"""Wait for search-budget evidence, then stage the guarded raw B8 V3 fork."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training/runs/stage2/fla3_pre/raw_b8_pool_10m"
MATCH = ROOT / "training/runs/stage2/fla2_r2/search_budget_256_vs_512_accepted_g31/match_summary_50pairs.json"
STAGED = BASE / "staged_code"
HANDOFF = BASE / "handoff_state.json"
SOURCE = {
    "config.py": ("training/v3/config.py", "3ea1297159450aaba0eb2abbebfb5fe73f1b2757b97725cd204338cde87b6d83", "995612b0d55c701ecffcc48ca8363a9d695c4793e36298a36042967ff0bf2d74"),
    "multirule_gate.py": ("training/v3/multirule_gate.py", "c74c957e57257f8692eb0c7e91e052f55a68763b0d6faacaaa7a0535b081b7da", "3829bd9e1ee7458f6d46da3db94400ab559f6e1d1d28f8a6734ca4a59596657f"),
    "pipeline.py": ("training/v3/pipeline.py", "fb94ada878596f0075214faa0e62cf5b3bfcbf36edaea60da9feebd7094bae6c", "b9d903bb43923d5510d3228c0a654b25ef13f21e74402e9a6759efa26a238ec5"),
    "formal_runner.py": ("training/v3/formal_runner.py", "700a1a77a17c09c2c09fccc163b049ef12af75da87f39c9c4d1c4688b6a2de3b", "23d92be32e7b8de276a48d8c0149aafae4a0e3cf8eff422a62c9fbe4c5164bff"),
    "test_training_v3_multirule_gate.py": ("test/test_training_v3_multirule_gate.py", "71d0a265660625952833578e745f8c2261c98e71fead697f72f4fbbc2a190fe1", "7c834d0689bd722b5e20b7816f97ea3a3f43f18bfc1a6ed8c6c4589666882e9e"),
    "test_training_v3_formal_runner.py": ("test/test_training_v3_formal_runner.py", "52b6919d5e91cff67b2ad81880e363ba906e50e26f503033a3a238f10f92851a", "bfc2a39807b3dac6d8c151b73731000d5a16ac3ac404be852948c748c6dba0b0"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(status: str, **fields: object) -> None:
    HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    temp = HANDOFF.with_suffix(".tmp")
    temp.write_text(json.dumps({"status": status, **fields}, indent=2, sort_keys=True) + "\n")
    temp.replace(HANDOFF)


def search_budget_running() -> bool:
    process = subprocess.run(["pgrep", "-f", "stage2_fla_raw_search_budget.py matches"],
                             capture_output=True, text=True)
    return process.returncode == 0


def gpu_idle() -> bool:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    return not apps.stdout.strip()


def upgrade_code() -> None:
    backup = BASE / "preupgrade_code"
    backup.mkdir(parents=True, exist_ok=True)
    for name, (relative, old_hash, staged_hash) in SOURCE.items():
        source = STAGED / name
        target = ROOT / relative
        if sha(source) != staged_hash:
            raise ValueError(f"staged V3 code SHA differs: {name}")
        if old_hash is not None and sha(target) != old_hash:
            raise ValueError(f"remote V3 code changed after preflight: {relative}")
    for name, (relative, _old_hash, staged_hash) in SOURCE.items():
        target = ROOT / relative
        if target.exists():
            shutil.copy2(target, backup / name)
        staged_target = target.with_suffix(target.suffix + ".fla3tmp")
        shutil.copy2(STAGED / name, staged_target)
        os.replace(staged_target, target)
        if sha(target) != staged_hash:
            raise ValueError(f"installed V3 code SHA differs: {name}")


def checked(command: list[str], log_name: str) -> None:
    log = BASE / "logs" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = BASE / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": str(temp_dir)}
    with log.open("w", encoding="utf-8") as output:
        done = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                              stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError(f"command exited {done.returncode}: {log}")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "handoff.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if HANDOFF.exists():
            raise FileExistsError("handoff watcher already started")
        try:
            state("waiting_search_budget")
            while not MATCH.exists():
                if not search_budget_running():
                    raise RuntimeError("search-budget match exited before five-rule summary")
                time.sleep(30)
            match_hash = sha(MATCH)
            state("waiting_idle_gpus", match_summary_sha256=match_hash)
            while not gpu_idle():
                time.sleep(30)
            if sha(MATCH) != match_hash:
                raise ValueError("search-budget summary changed before handoff")
            state("upgrading_gate", match_summary_sha256=match_hash)
            upgrade_code()
            checked([sys.executable, "-m", "unittest", "discover", "-s", "test",
                     "-p", "test_training_v3_multirule_gate.py"], "gate_unit.log")
            checked([sys.executable, "-m", "unittest", "discover", "-s", "test",
                     "-p", "test_training_v3_formal_runner.py"], "formal_unit.log")
            state("preparing", match_summary_sha256=match_hash)
            checked([sys.executable, "-u", "tools/stage2_fla_raw_10m_continue.py", "prepare"],
                    "prepare.log")
            state("launched", match_summary_sha256=match_hash)
            checked([sys.executable, "-u", "tools/stage2_fla_raw_10m_continue.py", "launch"],
                    "launcher.log")
            state("complete", match_summary_sha256=match_hash)
        except Exception as exc:
            state("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
