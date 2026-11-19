"""Crash-safe file writes: temp file in the same directory, fsync, rename."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

TEMP_PREFIX = ".tmp-"
_IS_WINDOWS = os.name == "nt"
_UNSUPPORTED_DIRECTORY_FSYNC = frozenset(
    value
    for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS")
    if (value := getattr(errno, name, None)) is not None
)
_WINDOWS_DIRECTORY_HANDLE_UNSUPPORTED = frozenset(
    value
    for name in ("EACCES", "EPERM", "EBADF")
    if (value := getattr(errno, name, None)) is not None
)
_PENDING_DIRECTORY_FSYNC_GUARD = threading.Lock()
_PENDING_DIRECTORY_FSYNC: set[Path] = set()


def _directory_fsync_unsupported(exc: OSError) -> bool:
    if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC:
        return True
    return _IS_WINDOWS and exc.errno in _WINDOWS_DIRECTORY_HANDLE_UNSUPPORTED


def fsync_directory(path: Path) -> None:
    """Flush directory metadata, ignoring only platform/filesystem non-support.

    Storage failures such as ``EIO`` and ``ENOSPC`` propagate. Callers must
    not report a durable write when the filesystem accepted directory handles
    but failed to persist their metadata.
    """

    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        if _directory_fsync_unsupported(exc):
            return
        raise
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        if not _directory_fsync_unsupported(exc):
            raise
    finally:
        os.close(dir_fd)


def _durability_fsync(path: Path) -> None:
    """Flush a parent and retain a failed obligation for a later retry."""

    path = Path(path).absolute()
    try:
        fsync_directory(path)
    except OSError:
        with _PENDING_DIRECTORY_FSYNC_GUARD:
            _PENDING_DIRECTORY_FSYNC.add(path)
        raise
    else:
        with _PENDING_DIRECTORY_FSYNC_GUARD:
            _PENDING_DIRECTORY_FSYNC.discard(path)


def _retry_directory_fsyncs() -> set[Path]:
    retried: set[Path] = set()
    with _PENDING_DIRECTORY_FSYNC_GUARD:
        pending = tuple(_PENDING_DIRECTORY_FSYNC)
    for path in pending:
        try:
            path.stat()
        except FileNotFoundError:
            with _PENDING_DIRECTORY_FSYNC_GUARD:
                _PENDING_DIRECTORY_FSYNC.discard(path)
            continue
        except OSError:
            pass  # the retained fsync below reports the storage failure
        _durability_fsync(path)
        retried.add(path)
    return retried


def durable_makedirs(path: Path) -> None:
    """Create ``path`` and missing parents, persisting every new directory entry."""

    path = Path(path)
    retried = _retry_directory_fsyncs()
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not missing:
        # Preserve Path.mkdir's error when ``path`` exists but is not a directory.
        path.mkdir(parents=True, exist_ok=True)
        # A prior call may have created this directory and then failed its
        # parent fsync. Existence is not evidence that the entry is durable.
        if path.parent != path and path.parent.absolute() not in retried:
            _durability_fsync(path.parent)
        return
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _durability_fsync(directory.parent)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=TEMP_PREFIX, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    # Persist the rename itself; only unsupported directory fsync is ignored.
    fsync_directory(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
