"""Local observation requests. They grant no hardware or motion authority."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_current_request: ContextVar[tuple[Path, dict] | None] = ContextVar("capture_request", default=None)


def process_identity(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def requested_activity(state_dir: Path) -> dict | None:
    episodes = state_dir / "episodes.json"
    if episodes.exists():
        active = json.loads(episodes.read_text()).get("active")
        if active is not None:
            return {"id": active["id"], "kind": "episode", "label": active.get("task", "")}
    for path in sorted((state_dir / "requests").glob("*.json")):
        try:
            request = json.loads(path.read_text())
            if request["kind"] in {"calibration", "diagnostic"} and request[
                "process_identity"
            ] == process_identity(request["pid"]):
                return {key: request[key] for key in ("id", "kind", "label")}
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return None


@contextmanager
def capture_activity(data_dir: Path, kind: str, label: str):
    if kind not in {"calibration", "diagnostic"}:
        raise ValueError("Unsupported capture activity")
    folder = data_dir / "mcap/requests"
    existing = _current_request.get()
    if existing is not None and existing[0] == folder.resolve():
        yield existing[1]
        return
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    identity = uuid.uuid4().hex
    path = folder / f"{identity}.json"
    temporary = path.with_suffix(".tmp")
    request = {
        "id": identity,
        "kind": kind,
        "label": label,
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "requested_at_ns": time.time_ns(),
    }
    try:
        with temporary.open("x") as handle:
            json.dump(request, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        token = _current_request.set((folder.resolve(), request))
        try:
            yield request
        finally:
            _current_request.reset(token)
    finally:
        path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
