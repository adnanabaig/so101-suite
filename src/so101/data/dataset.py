"""Dataset discovery + management for both LeRobot and fallback layouts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote

from ..config import get_settings
from ..hardware.camera_spec import is_depth_key
from ..logging import get_logger

log = get_logger(__name__)
_HUB_ENV_LOCK = threading.Lock()
_DATASET_LOCKS_GUARD = threading.Lock()
_DATASET_LOCKS: dict[str, DatasetLock] = {}
# Shared holds on a dataset's operation lock (training reads, Hub pushes).
# Never take _DATASET_LOCKS_GUARD while holding this guard.
_DATASET_READ_GUARD = threading.Lock()
_DATASET_READERS: dict[str, int] = {}
_LEROBOT_IMAGE_PREFIX = "observation.images."
FALLBACK_PUBLISH_TRANSACTION = ".fallback-publish.json"
# Words the dataset API uses after a repo id (/api/datasets/<id>/qc, /push,
# /episodes/<n>/...) or in place of one (/api/datasets/trash). An id with one
# of them as a component would be routed to that endpoint instead of the
# dataset, so no component may be one of these.
RESERVED_REPO_ID_COMPONENTS = ("episodes", "push", "qc", "trash")
_COMPONENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_PLAIN_REPO_ID = re.compile(rf"^{_COMPONENT}(?:/{_COMPONENT})*$")
_FREE_COMPONENT = rf"(?!(?:{'|'.join(RESERVED_REPO_ID_COMPONENTS)})(?:/|\Z)){_COMPONENT}"
_REPO_ID = re.compile(rf"^{_FREE_COMPONENT}(?:/{_FREE_COMPONENT})*$")


def repo_id_problem(repo_id: str) -> str | None:
    """Why ``repo_id`` is not a valid dataset id, or ``None`` when it is."""

    if len(repo_id) <= 200 and _REPO_ID.fullmatch(repo_id):
        return None
    if len(repo_id) <= 200 and _PLAIN_REPO_ID.fullmatch(repo_id):
        reserved = next(c for c in repo_id.split("/") if c in RESERVED_REPO_ID_COMPONENTS)
        return (
            f"repo_id {repo_id!r} uses the reserved component {reserved!r}: "
            f"{', '.join(RESERVED_REPO_ID_COMPONENTS)} are dataset API route names and "
            "cannot be part of a dataset id; choose another name"
        )
    return (
        "repo_id must contain one or more slash-separated alphanumeric, dot, "
        "underscore, or hyphen components (at most 200 characters)"
    )


def dataset_root(repo_id: str) -> Path:
    """Resolve a logical repo id beneath the configured dataset directory."""

    problem = repo_id_problem(repo_id)
    if problem is not None:
        raise ValueError(problem)
    datasets_dir = get_settings().datasets_dir
    encoded = datasets_dir / f"v2--{quote(repo_id, safe='')}"
    if encoded.is_symlink():
        raise ValueError(f"dataset root {encoded} is a symlink; refusing to follow it")
    if encoded.exists():
        return encoded

    # Backward-compatible lookup for earlier percent-encoded and flattened
    # layouts, but only when a manifest proves exact logical ownership.
    old_candidates = (
        datasets_dir / quote(repo_id, safe=""),
        datasets_dir / repo_id.replace("/", "__"),
    )
    for legacy in old_candidates:
        if legacy.is_symlink():
            raise ValueError(f"dataset root {legacy} is a symlink; refusing to follow it")
        if legacy == encoded or not legacy.exists():
            continue
        manifest_paths = (legacy / "meta.json", legacy / "meta" / "info.json")
        for manifest_path in manifest_paths:
            if not manifest_path.is_file():
                continue
            try:
                if json.loads(manifest_path.read_text()).get("repo_id") == repo_id:
                    return legacy
            except (json.JSONDecodeError, OSError):
                break
    return encoded


class DatasetBusyError(RuntimeError):
    """Dataset is being recorded, finalized, trained, exported, or removed."""


class DatasetNotFoundError(FileNotFoundError):
    """No local dataset exists for a logical repo id."""


class DatasetIdConflictError(RuntimeError):
    """A new repo id equals an existing dataset's id except for letter case."""


def validate_fallback_episode_tree(directory: Path) -> None:
    """Reject every symlink below one fallback episode without following it."""

    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError(f"fallback episode path is a symlink: {directory}")
    if not directory.is_dir():
        raise ValueError(f"fallback episode is not a directory: {directory}")

    def check(current: Path) -> None:
        with os.scandir(current) as entries:
            for entry in entries:
                path = current / entry.name
                if path.is_symlink():
                    raise ValueError(f"fallback episode contains a symlink: {path}")
                if path.is_dir():
                    check(path)

    check(directory)


def resolved_dataset_path(repo_id: str) -> Path:
    """The folder an operation on ``repo_id`` touches.

    That is the folder of the local dataset whose manifest names ``repo_id``
    (what :func:`get_dataset` finds, and what deletion, labels, training and
    pushes read or move), else where recording would create it.
    """

    root = dataset_root(repo_id)  # validates the id (ValueError)
    matches = {Path(info.path) for info in list_datasets() if info.repo_id == repo_id}

    # A publication marker deliberately hides a fallback dataset from normal
    # discovery.  Its validated journal still binds the logical id to the same
    # folder, including imported/noncanonical folders.  Without this lookup a
    # restart would lock and write the canonical folder instead of recovering
    # the hidden one.
    datasets_dir = get_settings().datasets_dir
    if datasets_dir.is_dir():
        for child in sorted(datasets_dir.iterdir()):
            if child.is_symlink() or not child.is_dir() or child.name.startswith("."):
                continue
            marker = child / FALLBACK_PUBLISH_TRANSACTION
            if not marker.exists() and not marker.is_symlink():
                continue
            if _marker_root_repo_id(child) == repo_id:
                matches.add(child)

    if len(matches) > 1:
        locations = ", ".join(str(path) for path in sorted(matches))
        raise DatasetIdConflictError(
            f"dataset {repo_id!r} is present in more than one folder ({locations}); "
            "refusing an ambiguous operation until the duplicate is repaired"
        )
    return next(iter(matches), root)


def _lock_key(repo_id: str) -> str:
    # Keyed on the folder, not the id, so every id that reaches one folder
    # shares one lock. Case-folded: on macOS and Windows "Alice/Pick" and
    # "alice/pick" are one folder (and on Linux they share a lock, harmlessly).
    path = resolved_dataset_path(repo_id).resolve()
    return os.path.normcase(str(path)).casefold()


# Lock files live here, one per dataset folder, named by a hash of the lock key
# (so every case variant of an id meets the same file). They are never
# deleted: deleting a locked file would let another process lock a new one.
LOCKS_DIRNAME = ".locks"
_USE_MSVCRT = os.name == "nt"
# Windows locks a byte range; lock one byte far past the (empty) file.
_WINDOWS_LOCK_OFFSET = 1 << 30
# flock reports EWOULDBLOCK; msvcrt.locking reports EACCES (or EDEADLOCK).
_LOCK_HELD_ERRNOS = {errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, errno.EDEADLK}
_LOCK_POLL_S = 0.05


def _try_file_lock(fd: int, *, shared: bool) -> bool:
    """Non-blocking lock on ``fd``; False while another open file holds it."""

    try:
        if _USE_MSVCRT:  # pragma: no cover - Windows (msvcrt has no shared locks)
            import msvcrt

            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _LOCK_HELD_ERRNOS:
            return False
        raise
    return True


def _file_unlock(fd: int) -> None:
    if _USE_MSVCRT:  # pragma: no cover - Windows
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class DatasetLock:
    """A dataset's operation lock, exclusive across threads and processes.

    A thread lock excludes the threads of this process; an OS file lock
    (``fcntl.flock`` on macOS and Linux, ``msvcrt.locking`` on Windows) on
    ``<datasets dir>/.locks/<hash>.lock`` excludes other processes, so a
    ``so101 datasets rm`` cannot move a dataset an API recording is writing.
    The OS drops the file lock when its holder exits, crashes included.

    ``acquire(shared=True)`` takes the file lock shared, for read holds: other
    processes may read at the same time but not write. Windows has no shared
    file locks, so there readers in different processes exclude each other.
    A platform with no locking API falls back to the process lock (logged);
    filesystem/open/locking errors on a supported platform fail closed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread = threading.Lock()
        self._fd: int | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1, *, shared: bool = False) -> bool:
        deadline = time.monotonic() + timeout if blocking and timeout >= 0 else None
        if not self._thread.acquire(blocking, timeout):
            return False
        fd: int | None = None
        try:
            fd = self._open()
            while fd is not None:
                try:
                    if _try_file_lock(fd, shared=shared):
                        break
                except ImportError as exc:  # pragma: no cover - unsupported OS
                    log.warning(
                        "no file-locking API for %s (%s); locking per process only",
                        self.path,
                        exc,
                    )
                    os.close(fd)
                    fd = None
                    break
                if not blocking or (deadline is not None and time.monotonic() >= deadline):
                    os.close(fd)
                    fd = None
                    self._thread.release()
                    return False
                time.sleep(_LOCK_POLL_S)
        except BaseException:
            if fd is not None:
                os.close(fd)
            self._thread.release()
            raise
        self._fd = fd
        return True

    def _open(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)

    def release(self) -> None:
        fd, self._fd = self._fd, None
        try:
            if fd is not None:
                try:
                    _file_unlock(fd)
                except OSError:
                    pass  # closing the descriptor releases it anyway
                finally:
                    os.close(fd)
        finally:
            self._thread.release()

    def locked(self) -> bool:
        """Whether this process holds the lock (other processes are not visible)."""

        return self._thread.locked()

    def __enter__(self) -> DatasetLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def dataset_lock_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return get_settings().datasets_dir / LOCKS_DIRNAME / f"{digest}.lock"


def _lock_for_key(key: str) -> DatasetLock:
    with _DATASET_LOCKS_GUARD:
        lock = _DATASET_LOCKS.get(key)
        if lock is None:
            lock = _DATASET_LOCKS[key] = DatasetLock(dataset_lock_path(key))
        return lock


def dataset_operation_target(repo_id: str) -> tuple[Path, DatasetLock]:
    """Resolve one folder once and return both it and its operation lock.

    A recorder must never choose its destination with :func:`dataset_root` and
    then independently choose a lock through discovery: an imported folder can
    legitimately have a noncanonical name.  Resolving once makes the bytes and
    the lock share one identity.
    """

    path = resolved_dataset_path(repo_id)
    key = os.path.normcase(str(path.resolve())).casefold()
    return path, _lock_for_key(key)


def dataset_operation_lock(repo_id: str) -> DatasetLock:
    """The exclusive per-dataset lock held by recording, deletion and edits.

    Shared by every thread of this process and, through its lock file, by
    other so101 processes (the server, the CLI, ``so101 sim``).
    """

    return dataset_operation_target(repo_id)[1]


def dataset_busy_message(repo_id: str) -> str:
    """Explain why ``repo_id``'s operation lock is taken (for writers that lost the race)."""

    key = _lock_key(repo_id)
    readers = _DATASET_READERS.get(key, 0)
    if readers:
        jobs = "job" if readers == 1 else "jobs"
        return (
            f"dataset {repo_id!r} is currently in use by {readers} training {jobs} or Hub "
            "push; wait for it to finish or cancel it"
        )
    if not _lock_for_key(key).locked():
        return (
            f"dataset {repo_id!r} is currently in use by another so101 process (recording, "
            "training, a Hub push or an edit); wait for it to finish"
        )
    return f"dataset {repo_id!r} is currently in use (recording, export or an edit in progress)"


@contextmanager
def dataset_read_hold(repo_id: str) -> Iterator[None]:
    """Hold ``repo_id`` for reading: shared with other readers, exclusive of writers.

    Training jobs and Hub pushes read a dataset while it must not change;
    several may hold it at once. The first reader in this process takes the
    operation lock (its lock file shared, so readers in other processes may
    join) and the last one releases it, so recording, deletion and restore,
    here or in another process, are refused (and say why) while any reader is
    active.

    Raises :class:`DatasetBusyError` when a writer owns the dataset.
    """

    key = _lock_key(repo_id)
    lock = _lock_for_key(key)  # takes _DATASET_LOCKS_GUARD; not under ours
    with _DATASET_READ_GUARD:
        count = _DATASET_READERS.get(key, 0)
        if count == 0 and not lock.acquire(blocking=False, shared=True):
            where = "" if lock.locked() else " by another so101 process"
            raise DatasetBusyError(
                f"dataset {repo_id!r} is being recorded, exported or edited{where} right now; "
                "try again when that finishes"
            )
        _DATASET_READERS[key] = count + 1
    try:
        yield
    finally:
        with _DATASET_READ_GUARD:
            remaining = _DATASET_READERS[key] - 1
            if remaining:
                _DATASET_READERS[key] = remaining
            else:
                del _DATASET_READERS[key]
                lock.release()


@dataclass
class DatasetInfo:
    repo_id: str
    path: str
    format: str  # "lerobot" | "so101-fallback-v1/v2"
    num_episodes: int
    num_frames: int
    fps: int
    task: str
    camera_keys: list[str]  # color cameras only: what a viewer may show as RGB
    # Depth streams (RealSense ``+depth``): "<entry>#depth" keys in the fallback
    # format, ``is_depth_map`` image features in LeRobot datasets.
    depth_keys: list[str] = field(default_factory=list)


def _split_depth(keys: list[str], depth: set[str]) -> dict[str, list[str]]:
    return {
        "camera_keys": [key for key in keys if key not in depth],
        "depth_keys": [key for key in keys if key in depth],
    }


@dataclass
class TrashedDatasetInfo:
    id: str
    repo_id: str
    format: str
    num_episodes: int
    num_frames: int
    deleted_at: float
    size_bytes: int


def _fallback_transaction_repo_id(root: Path) -> str | None:
    """Return a pending journal's validated dataset identity, never its new counters."""

    path = root / FALLBACK_PUBLISH_TRANSACTION
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    expected = {
        "schema_version",
        "episode",
        "stage",
        "destination",
        "previous_metadata",
        "next_metadata",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema_version") != 1
    ):
        return None
    episode = payload.get("episode")
    stage = payload.get("stage")
    destination = payload.get("destination")
    previous = payload.get("previous_metadata")
    following = payload.get("next_metadata")
    if (
        type(episode) is not int
        or episode < 0
        or not isinstance(stage, str)
        or Path(stage).name != stage
        or not stage.startswith(".episode-staging-")
        or destination != f"episode_{episode:06d}"
        or (previous is not None and not isinstance(previous, dict))
        or not isinstance(following, dict)
        or following.get("format") != "so101-fallback-v2"
        or type(following.get("episodes")) is not int
        or following.get("episodes") != episode + 1
    ):
        return None
    previous_episodes = 0 if previous is None else previous.get("episodes", 0)
    if type(previous_episodes) is not int or previous_episodes != episode:
        return None
    repo_id = following.get("repo_id")
    if (
        not isinstance(repo_id, str)
        or repo_id_problem(repo_id) is not None
        or (previous is not None and previous.get("repo_id") != repo_id)
    ):
        return None
    if root.name.startswith("v2--") and root.name != f"v2--{quote(repo_id, safe='')}":
        return None
    return repo_id


def _marker_root_repo_id(root: Path) -> str | None:
    """Best validated identity for a marker-hidden root, even with a corrupt marker."""

    journal_id = _fallback_transaction_repo_id(root)
    if journal_id is not None:
        return journal_id
    encoded_id: str | None = None
    if root.name.startswith("v2--"):
        decoded = unquote(root.name.removeprefix("v2--"))
        if repo_id_problem(decoded) is None and root.name == f"v2--{quote(decoded, safe='')}":
            encoded_id = decoded
    manifest_id: str | None = None
    try:
        metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        metadata = None
    repo_id = metadata.get("repo_id") if isinstance(metadata, dict) else None
    if isinstance(repo_id, str) and repo_id_problem(repo_id) is None:
        manifest_id = repo_id
    if encoded_id is not None and manifest_id is not None and encoded_id != manifest_id:
        return None
    return manifest_id or encoded_id


@dataclass(frozen=True)
class _FileSnapshot:
    """Bytes plus the identity of the exact file description they came from."""

    data: bytes
    token: tuple[int, int, int, int, bytes]


def _read_file_snapshot(path: Path) -> _FileSnapshot:
    """Read ``path`` with a token that changes on replacement or content change."""

    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise OSError(f"dataset metadata changed while it was read: {path}")
    return _FileSnapshot(
        data=data,
        token=(*after_identity, hashlib.sha256(data).digest()),
    )


def _fallback_marker_present(root: Path) -> bool:
    marker = root / FALLBACK_PUBLISH_TRANSACTION
    return marker.exists() or marker.is_symlink()


def _read_fallback(root: Path) -> DatasetInfo | None:
    transaction = root / FALLBACK_PUBLISH_TRANSACTION
    if _fallback_marker_present(root):
        log.warning(
            "skipping fallback dataset %s while publication transaction %s needs recovery",
            root,
            transaction,
        )
        return None
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return None
    try:
        snapshot = _read_file_snapshot(meta_path)
        meta = json.loads(snapshot.data.decode("utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError) as exc:
        log.warning("skipping unreadable fallback dataset manifest %s: %s", meta_path, exc)
        return None
    if not isinstance(meta, dict):
        log.warning("skipping non-object fallback dataset manifest %s", meta_path)
        return None
    try:
        frames = int(meta["total_frames"])
    except (KeyError, TypeError, ValueError):
        frames = 0
        for ep in sorted(root.glob("episode_*")):
            try:
                validate_fallback_episode_tree(ep)
            except (OSError, ValueError) as exc:
                log.warning("skipping unsafe fallback dataset episode %s: %s", ep, exc)
                return None
            fj = ep / "frames.jsonl"
            if fj.exists():
                try:
                    with fj.open() as handle:
                        frames += sum(1 for line in handle if line.strip())
                except OSError as exc:
                    log.warning("could not count frames in %s: %s", fj, exc)
    try:
        episodes = int(meta.get("episodes", 0))
        fps = int(meta.get("fps", 30))
    except (TypeError, ValueError) as exc:
        log.warning("skipping fallback dataset with invalid counters %s: %s", meta_path, exc)
        return None
    repo_id = str(meta.get("repo_id", root.name))
    problem = repo_id_problem(repo_id)
    if problem is not None:
        log.warning("skipping fallback dataset %s: %s", root, problem)
        return None
    raw_cameras = meta.get("camera_keys", [])
    camera_keys = [str(item) for item in raw_cameras] if isinstance(raw_cameras, list) else []
    info = DatasetInfo(
        repo_id=repo_id,
        path=str(root),
        format=str(meta.get("format", "so101-fallback-v1")),
        num_episodes=episodes,
        num_frames=frames,
        fps=fps,
        task=str(meta.get("task", "")),
        **_split_depth(camera_keys, {key for key in camera_keys if is_depth_key(key)}),
    )
    # The marker alone is not a generation: absent -> present -> absent can
    # happen while a failed publication rolls back.  Require both an absent
    # marker and the same manifest identity/content that supplied ``info``.
    if _fallback_marker_present(root):
        log.warning(
            "discarding fallback dataset snapshot %s because publication began while reading",
            root,
        )
        return None
    try:
        current = _read_file_snapshot(meta_path)
    except OSError as exc:
        log.warning("discarding fallback dataset snapshot %s: %s", root, exc)
        return None
    if current.token != snapshot.token:
        log.warning(
            "discarding fallback dataset snapshot %s because its manifest generation changed",
            root,
        )
        return None
    return info


def _read_lerobot(root: Path) -> DatasetInfo | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text())
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        log.warning("skipping unreadable LeRobot dataset manifest %s: %s", info_path, exc)
        return None
    if not isinstance(info, dict):
        log.warning("skipping non-object LeRobot dataset manifest %s", info_path)
        return None
    try:
        episodes = int(info.get("total_episodes", 0))
        frames = int(info.get("total_frames", 0))
        fps = int(info.get("fps", 30))
    except (TypeError, ValueError) as exc:
        log.warning("skipping LeRobot dataset with invalid counters %s: %s", info_path, exc)
        return None
    inferred_repo_id = (
        unquote(root.name.removeprefix("v2--")) if root.name.startswith("v2--") else root.name
    )
    repo_id = str(info.get("repo_id", inferred_repo_id))
    problem = repo_id_problem(repo_id)
    if problem is not None:
        log.warning("skipping LeRobot dataset %s: %s", root, problem)
        return None
    features = info.get("features", {})
    if not isinstance(features, dict):
        features = {}
    images = [str(key) for key in features if "images" in str(key)]
    depth_maps = {
        key
        for key in images
        if isinstance(features[key], dict)
        and isinstance(features[key].get("info"), dict)
        and (
            features[key]["info"].get("is_depth_map")
            or features[key]["info"].get("video.is_depth_map")
        )
    }
    split = _split_depth(images, depth_maps)
    return DatasetInfo(
        repo_id=repo_id,
        path=str(root),
        format="lerobot",
        num_episodes=episodes,
        num_frames=frames,
        fps=fps,
        task="",
        # Camera names may contain dots (rtsp hosts), so strip only the prefix.
        camera_keys=[key.removeprefix(_LEROBOT_IMAGE_PREFIX) for key in split["camera_keys"]],
        depth_keys=[key.removeprefix(_LEROBOT_IMAGE_PREFIX) for key in split["depth_keys"]],
    )


def _read_incomplete_only(root: Path) -> DatasetInfo | None:
    """Discover a dataset whose first recording was interrupted before meta.json."""

    if _fallback_marker_present(root):
        return None

    candidates = sorted(root.glob("incomplete_*/INCOMPLETE.json"))
    metadata: dict = {}
    if candidates:
        try:
            metadata = json.loads(candidates[-1].read_text())
        except (json.JSONDecodeError, OSError, TypeError):
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    repo_id = str(metadata.get("repo_id", ""))
    if not repo_id and root.name.startswith("v2--"):
        repo_id = unquote(root.name.removeprefix("v2--"))
    if not repo_id or not _REPO_ID.fullmatch(repo_id):
        return None
    if not candidates and not list(root.glob(".episode-staging-*")):
        return None
    try:
        fps = int(metadata.get("fps", 30))
    except (TypeError, ValueError):
        fps = 30
    keys = [str(item) for item in metadata.get("camera_keys", [])]
    info = DatasetInfo(
        repo_id=repo_id,
        path=str(root),
        format=str(metadata.get("format", "so101-fallback-v2")),
        num_episodes=0,
        num_frames=0,
        fps=fps,
        task=str(metadata.get("task", "")),
        **_split_depth(keys, {key for key in keys if is_depth_key(key)}),
    )
    if _fallback_marker_present(root):
        return None
    return info


def list_datasets() -> list[DatasetInfo]:
    """Scan the datasets dir for both layouts."""
    root = get_settings().datasets_dir
    out: list[DatasetInfo] = []
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if child.is_symlink() or not child.is_dir() or child.name.startswith("."):
            continue
        info = _read_lerobot(child) or _read_fallback(child) or _read_incomplete_only(child)
        if info:
            out.append(info)
    return out


def get_dataset(repo_id: str) -> DatasetInfo | None:
    """The local dataset whose manifest names ``repo_id``, or ``None``.

    Only the logical id matches. A folder name never does: the legacy
    flattened folder ``bob__reach`` holds ``bob/reach``, and the id
    ``bob__reach`` must not reach it around ``bob/reach``'s lock.
    """

    if repo_id_problem(repo_id) is not None:
        return None
    for ds in list_datasets():
        if ds.repo_id == repo_id:
            return ds
    return None


def case_alias(repo_id: str) -> str | None:
    """The existing dataset id that equals ``repo_id`` except for letter case.

    ``None`` when no dataset matches ignoring case, or one matches exactly.
    An unresolved publication marker raises ``DatasetIdConflictError`` rather
    than allowing a new id around data whose owner cannot be established.
    """

    folded = repo_id.casefold()
    alias: str | None = None
    for ds in list_datasets():
        if ds.repo_id == repo_id:
            return None
        if alias is None and ds.repo_id.casefold() == folded:
            alias = ds.repo_id
    root = get_settings().datasets_dir
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_symlink() or not child.is_dir() or child.name.startswith("."):
                continue
            marker = child / FALLBACK_PUBLISH_TRANSACTION
            if not marker.exists() and not marker.is_symlink():
                continue
            pending = _marker_root_repo_id(child)
            if pending is None:
                raise DatasetIdConflictError(
                    f"cannot verify the dataset identity hidden by publication marker {marker}; "
                    f"refusing admission of {repo_id!r} until the marker is repaired"
                )
            if pending == repo_id:
                return None
            if pending is not None and alias is None and pending.casefold() == folded:
                alias = pending
    return alias


def case_alias_message(repo_id: str, existing: str) -> str:
    return (
        f"dataset {existing!r} already exists and {repo_id!r} differs from it only in letter "
        "case; on macOS and Windows both ids are one folder. Use "
        f"{existing!r}, or a name that differs in more than case"
    )


def refuse_case_alias(repo_id: str) -> None:
    """Refuse a new repo id that equals an existing one ignoring case.

    Raises :class:`DatasetIdConflictError` naming the existing id.
    """

    existing = case_alias(repo_id)
    if existing is not None:
        raise DatasetIdConflictError(case_alias_message(repo_id, existing))


def require_dataset(repo_id: str) -> DatasetInfo:
    """Return the local dataset or raise :class:`DatasetNotFoundError` with a fix."""

    dataset_root(repo_id)  # validates the repo id format (ValueError)
    info = get_dataset(repo_id)
    if info is None:
        existing = case_alias(repo_id)
        hint = f" (did you mean {existing!r}?)" if existing is not None else ""
        raise DatasetNotFoundError(
            f"dataset {repo_id!r} not found under {get_settings().datasets_dir}{hint}; "
            "record it first or check `so101 datasets ls`"
        )
    return info


def delete_dataset(repo_id: str) -> bool:
    operation_lock = dataset_operation_lock(repo_id)
    if not operation_lock.acquire(blocking=False):
        raise DatasetBusyError(dataset_busy_message(repo_id))
    try:
        ds = get_dataset(repo_id)
        if not ds:
            return False
        source = Path(ds.path)
        trash = get_settings().datasets_dir / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        destination = trash / f"{source.name}.{timestamp}"
        while destination.exists():
            timestamp += 1
            destination = trash / f"{source.name}.{timestamp}"
        os.replace(source, destination)
        log.info("moved dataset %s to recoverable trash at %s", repo_id, destination)
        return True
    finally:
        operation_lock.release()


def _directory_size(root: Path) -> int:
    size = 0
    try:
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
    except OSError:
        pass
    return size


def list_trashed_datasets() -> list[TrashedDatasetInfo]:
    """Return recoverable dataset deletions without following trash symlinks."""

    trash = get_settings().datasets_dir / ".trash"
    if not trash.is_dir():
        return []
    entries: list[TrashedDatasetInfo] = []
    for child in sorted(trash.iterdir(), reverse=True):
        if child.is_symlink() or not child.is_dir():
            continue
        info = _read_lerobot(child) or _read_fallback(child) or _read_incomplete_only(child)
        if info is None:
            continue
        suffix = child.name.rsplit(".", 1)[-1]
        try:
            deleted_at = int(suffix) / 1000.0
        except ValueError:
            deleted_at = child.stat().st_mtime
        entries.append(
            TrashedDatasetInfo(
                id=child.name,
                repo_id=info.repo_id,
                format=info.format,
                num_episodes=info.num_episodes,
                num_frames=info.num_frames,
                deleted_at=deleted_at,
                size_bytes=_directory_size(child),
            )
        )
    return entries


def restore_trashed_dataset(trash_id: str) -> DatasetInfo:
    """Atomically restore one dataset from recoverable trash."""

    if not trash_id or len(trash_id) > 300 or "/" in trash_id or "\\" in trash_id:
        raise ValueError("invalid trash entry id")
    trash = get_settings().datasets_dir / ".trash"
    source = trash / trash_id
    if source.is_symlink() or not source.is_dir() or source.parent != trash:
        raise FileNotFoundError(f"trash entry {trash_id!r} not found")
    info = _read_lerobot(source) or _read_fallback(source) or _read_incomplete_only(source)
    if info is None:
        raise ValueError(f"trash entry {trash_id!r} is not a readable dataset")

    operation_lock = dataset_operation_lock(info.repo_id)
    if not operation_lock.acquire(blocking=False):
        raise DatasetBusyError(dataset_busy_message(info.repo_id))
    try:
        if get_dataset(info.repo_id) is not None:
            raise FileExistsError(f"dataset {info.repo_id!r} already exists")
        existing = case_alias(info.repo_id)
        if existing is not None:
            raise FileExistsError(case_alias_message(info.repo_id, existing))
        destination = dataset_root(info.repo_id)
        if destination.exists():
            raise FileExistsError(f"restore destination already exists: {destination.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        restored = (
            _read_lerobot(destination)
            or _read_fallback(destination)
            or _read_incomplete_only(destination)
        )
        if restored is None:  # pragma: no cover - source was validated immediately above
            raise RuntimeError("restored dataset became unreadable")
        log.info("restored dataset %s from %s", info.repo_id, source)
        return restored
    finally:
        operation_lock.release()


def push_to_hub(repo_id: str, private: bool = True) -> str:
    """Push a local LeRobot dataset to the Hugging Face Hub."""

    # A push only reads the dataset: it may run beside training, never beside
    # recording or deletion.
    with dataset_read_hold(repo_id):
        return _push_to_hub_locked(repo_id, private)


def _push_to_hub_locked(repo_id: str, private: bool) -> str:
    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError("SO101_HF_TOKEN not set")
    dataset_info = get_dataset(repo_id)
    if dataset_info is None:
        raise RuntimeError(f"dataset {repo_id!r} not found")
    if dataset_info.format != "lerobot":
        raise NotImplementedError(
            "Only validated LeRobot-format datasets can be pushed. Convert the "
            f"{dataset_info.format} development dataset first."
        )
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("LeRobot dataset support is not installed") from exc

    dataset = LeRobotDataset(
        repo_id,
        root=Path(dataset_info.path),
        token=settings.hf_token,
    )
    # LeRobot 0.6 creates its own HfApi(), which reads HF_TOKEN. Serialize the
    # temporary environment override and always restore the caller's process.
    with _HUB_ENV_LOCK:
        previous = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = settings.hf_token
        try:
            dataset.push_to_hub(private=private)
        finally:
            if previous is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = previous
    url = f"https://huggingface.co/datasets/{repo_id}"
    log.info("pushed dataset %s -> %s", repo_id, url)
    return url
