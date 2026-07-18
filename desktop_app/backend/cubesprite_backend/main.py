from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__
from .service import CubeSpriteService, ServiceError


WRITE_LOCK = threading.Lock()
PENDING_LOCK = threading.Lock()
PENDING_IDS = set()
MAX_REQUEST_CHARS = 1_048_576


def write_message(payload):
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with WRITE_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def reject_nonfinite_json(value):
    """Reject Python's non-standard NaN/Infinity JSON extensions.

    JavaScript's ``JSON.parse`` rejects these tokens. Accepting them here could
    therefore make an error response impossible to parse on the Tauri side.
    """

    raise ValueError(f"Non-finite JSON value {value} is not permitted.")


def process_request(service, request):
    request_id = request.get("id") if isinstance(request, dict) else None
    try:
        if not isinstance(request, dict):
            raise ServiceError("INVALID_ENVELOPE", "A request must be a JSON object.")
        if isinstance(request.get("v"), bool) or request.get("v") != 1 or request.get("type") != "request":
            raise ServiceError("INVALID_ENVELOPE", "Expected a protocol v1 request envelope.")
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool) or request_id == "":
            raise ServiceError("INVALID_REQUEST_ID", "Request id must be a non-empty string or integer.")
        if not isinstance(request.get("command"), str) or not request["command"]:
            raise ServiceError("INVALID_COMMAND", "command must be a non-empty string.")
        if "params" in request and request["params"] is not None and not isinstance(request["params"], dict):
            raise ServiceError("INVALID_PARAMS", "params must be a JSON object.")
        command = str(request.get("command", ""))
        if command == "system.initialize":
            result = service.initialize()
        else:
            result = service.handle(command, request.get("params") or {})
        write_message({"v": 1, "type": "response", "id": request_id, "ok": True, "result": result})
    except ServiceError as exc:
        write_message(
            {
                "v": 1,
                "type": "response",
                "id": request_id,
                "ok": False,
                "error": {"code": exc.code, "message": str(exc), "details": exc.details},
            }
        )
    except Exception as exc:  # Keep protocol alive; diagnostics belong on stderr.
        print(f"Unhandled backend error: {exc!r}", file=sys.stderr, flush=True)
        write_message(
            {
                "v": 1,
                "type": "response",
                "id": request_id,
                "ok": False,
                "error": {"code": "INTERNAL_ERROR", "message": str(exc), "details": None},
            }
        )


def resolve_resource_dir(value):
    if value:
        return Path(value).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "resources"
    return Path(__file__).resolve().parents[2] / "src-tauri" / "resources"


def resolve_data_dir(value):
    if not value:
        return None
    return Path(value).expanduser().resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Connect4 3D CubeSprite backend")
    parser.add_argument("--resource-dir", default=os.environ.get("CUBESPRITE_RESOURCE_DIR"))
    parser.add_argument("--data-dir", default=os.environ.get("CUBESPRITE_DATA_DIR"))
    args = parser.parse_args(argv)
    try:
        service = CubeSpriteService(
            resolve_resource_dir(args.resource_dir),
            data_dir=resolve_data_dir(args.data_dir),
        )
    except ServiceError as exc:
        write_message(
            {
                "v": 1,
                "type": "event",
                "event": "backend.fatal",
                "data": {"code": exc.code, "message": str(exc), "details": exc.details},
            }
        )
        return 2
    write_message(
        {
            "v": 1,
            "type": "event",
            "event": "backend.ready",
            "data": {"backend_version": __version__, "protocol_version": 1},
        }
    )
    with (
        ThreadPoolExecutor(max_workers=4, thread_name_prefix="cubesprite") as executor,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="cubesprite-analysis") as analysis_executor,
    ):
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            if len(raw_line) > MAX_REQUEST_CHARS:
                write_message(
                    {
                        "v": 1,
                        "type": "response",
                        "id": None,
                        "ok": False,
                        "error": {"code": "REQUEST_TOO_LARGE", "message": "JSON request is too large.", "details": None},
                    }
                )
                continue
            try:
                request = json.loads(raw_line, parse_constant=reject_nonfinite_json)
            except (json.JSONDecodeError, ValueError) as exc:
                write_message(
                    {
                        "v": 1,
                        "type": "response",
                        "id": None,
                        "ok": False,
                        "error": {"code": "INVALID_JSON", "message": str(exc), "details": None},
                    }
                )
                continue
            request_id = request.get("id") if isinstance(request, dict) else None
            track_request_id = (
                isinstance(request_id, (str, int))
                and not isinstance(request_id, bool)
                and request_id != ""
            )
            request_executor = (
                analysis_executor
                if isinstance(request, dict) and request.get("command") == "replay.analyze"
                else executor
            )
            if not track_request_id:
                request_executor.submit(process_request, service, request)
                continue
            with PENDING_LOCK:
                duplicate = request_id in PENDING_IDS
                if not duplicate:
                    PENDING_IDS.add(request_id)
            if duplicate:
                write_message(
                    {
                        "v": 1,
                        "type": "response",
                        "id": request_id,
                        "ok": False,
                        "error": {"code": "DUPLICATE_REQUEST_ID", "message": "Request id is already in flight.", "details": None},
                    }
                )
                continue
            future = request_executor.submit(process_request, service, request)

            def clear_pending(_future, pending_id=request_id):
                with PENDING_LOCK:
                    PENDING_IDS.discard(pending_id)

            future.add_done_callback(clear_pending)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
