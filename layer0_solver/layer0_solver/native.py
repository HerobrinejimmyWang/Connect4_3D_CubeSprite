from __future__ import annotations

import json
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .game import Layer0State


@dataclass(frozen=True, slots=True)
class NativeResult:
    outcome: int
    optimal_moves: tuple[int, ...]
    nodes: int
    cache_hits: int
    cache_size: int
    cache_reset: bool
    seconds: float


class NativeSolver:
    """Adapter for the optional C++ exhaustive backend."""

    def __init__(self, executable: str | Path | None = None, *, timeout: float = 8.0) -> None:
        root = Path(__file__).resolve().parents[1]
        self.executable = (
            Path(executable).resolve()
            if executable is not None
            else root / "build" / "layer0_native.exe"
        )
        self.timeout = float(timeout)

    @property
    def available(self) -> bool:
        return self.executable.is_file()

    def analyze(self, state: Layer0State) -> NativeResult:
        if not self.available:
            raise FileNotFoundError(
                f"Native solver not found at {self.executable}; run build_native.ps1 first."
            )
        try:
            completed = subprocess.run(
                [
                    str(self.executable),
                    "--state",
                    str(state.red_bits),
                    str(state.blue_bits),
                    str(state.to_move),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"native solver exceeded {self.timeout:.1f}s") from exc
        payload = json.loads(completed.stdout)
        if payload.get("terminal"):
            raise ValueError("native solver received a terminal state")
        return NativeResult(
            outcome=int(payload["value"]),
            optimal_moves=tuple(int(move) for move in payload["optimal_moves"]),
            nodes=int(payload["nodes"]),
            cache_hits=int(payload["cache_hits"]),
            cache_size=int(payload["cache_size"]),
            cache_reset=bool(payload.get("cache_reset", False)),
            seconds=float(payload["seconds"]),
        )


class PersistentNativeSolver(NativeSolver):
    """Long-lived native process that reuses the exact transposition table."""

    def __init__(self, executable: str | Path | None = None, *, timeout: float = 180.0) -> None:
        super().__init__(executable, timeout=timeout)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            if not self.available:
                raise FileNotFoundError(
                    f"Native solver not found at {self.executable}; run build_native.ps1 first."
                )
            self._process = subprocess.Popen(
                [str(self.executable), "--server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

    def analyze(self, state: Layer0State) -> NativeResult:
        with self._lock:
            self.start()
            process = self._process
            assert process is not None and process.stdin is not None and process.stdout is not None
            process.stdin.write(f"{state.red_bits} {state.blue_bits} {state.to_move}\n")
            process.stdin.flush()

            response: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

            def read_response() -> None:
                try:
                    response.put(process.stdout.readline())
                except BaseException as exc:  # Propagate pipe failures to the caller.
                    response.put(exc)

            reader = threading.Thread(target=read_response, daemon=True)
            reader.start()
            try:
                item = response.get(timeout=self.timeout)
            except queue.Empty as exc:
                self.close(force=True)
                raise TimeoutError(f"persistent native solver exceeded {self.timeout:.1f}s") from exc
            if isinstance(item, BaseException):
                self.close(force=True)
                raise RuntimeError("failed reading native solver response") from item
            if not item:
                stderr = process.stderr.read() if process.stderr is not None else ""
                code = process.poll()
                self.close(force=True)
                raise RuntimeError(f"native solver exited with code {code}: {stderr.strip()}")
            payload = json.loads(item)
            if payload.get("terminal"):
                raise ValueError("native solver received a terminal state")
            return NativeResult(
                outcome=int(payload["value"]),
                optimal_moves=tuple(int(move) for move in payload["optimal_moves"]),
                nodes=int(payload["nodes"]),
                cache_hits=int(payload["cache_hits"]),
                cache_size=int(payload["cache_size"]),
                cache_reset=bool(payload.get("cache_reset", False)),
                seconds=float(payload["seconds"]),
            )

    def close(self, *, force: bool = False) -> None:
        if force:
            process, self._process = self._process, None
            if process is None:
                return
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            return
        with self._lock:
            process, self._process = self._process, None
            if process is None:
                return
            if process.poll() is None and not force and process.stdin is not None:
                try:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                    process.wait(timeout=2.0)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    force = True
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()

    def __enter__(self) -> PersistentNativeSolver:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
