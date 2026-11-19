"""Who may open the serial bus while the Suite is installed.

A running Suite server (RobotManager) owns both serial buses and the cameras.
A CLI tool from the bring-up kit opens a bus only when both hold:

1. Nothing answers ``GET /api/robot/status`` at ``SO101_HOST:SO101_PORT``.
   Anything that answers, including a non-Suite program or a timeout, counts
   as "a server is running" (fail closed).
2. It holds one exclusive, host-global lock for each canonical serial device
   it may open (``fcntl.flock`` on macOS and Linux, ``msvcrt.locking`` on
   Windows), so two Suite processes cannot share a physical bus even when they
   use different data folders or checkouts.

On the real backend the server takes the device locks for its whole lifetime
(:func:`take_server_bus_lock`, from the app lifespan), and bounded bring-up
commands such as ``calibrate`` take them through :mod:`.standalone`. Owned
real motion is server-only. A server on
a port other than ``SO101_PORT`` is invisible to the probe, but the device
locks still exclude it.

On POSIX, a device locks its configured/canonical paths, character-device
``st_rdev`` (or regular-file inode test seam), and the captured node inode.
Thus aliases, hardlinks and duplicate character nodes collide, while replacing
or reusing a tty path is refused immediately before and after the driver open.
A missing device takes an exclusive namespace gate. Locks and the crash-safe
energized-intent journal live in an installer-created root-owned, setgid+sticky
hardware-group directory, never an age-cleaned temporary directory. Files are
opened without following links/reparse points and descriptor/path identity is
checked around each open. Windows uses stable Local AppData, independent of
TEMP, and refuses filesystem aliases/reparse points for motor ports.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import httpx

BUS_LOCK_FILENAME = "bus.lock"
DEVICE_LOCK_VERSION = "v1"
DEVICE_LOCK_PREFIX = f"so101-motor-bus-{DEVICE_LOCK_VERSION}"
GLOBAL_ENERGIZED_INTENT_KIND = "so101.global-energized-intent"
GLOBAL_ENERGIZED_INTENT_SCHEMA_VERSION = 2
_GLOBAL_CLEAR_KIND = "so101.global-energized-clear"
_GLOBAL_ALIAS_KIND = "so101.global-device-alias"
_GLOBAL_MARKER_MAX_BYTES = 16 * 1024
STATUS_PATH = "/api/robot/status"
#: Seconds to wait for the status reply once connected.
PROBE_READ_TIMEOUT_S = 0.5
#: Seconds to wait for the TCP connect. Windows reports a refused localhost
#: connection only after retrying the SYN (about 1-2 s; unverified here, no
#: Windows host), so a 0.5 s connect timeout would read as "a server is running".
PROBE_CONNECT_TIMEOUT_S = 3.0 if os.name == "nt" else 0.5
#: Windows locks a byte range; lock one byte far past the holder record so other
#: processes can still read who holds it.
_WINDOWS_LOCK_OFFSET = 1 << 30
_USE_MSVCRT = os.name == "nt"
OwnershipReason = Literal["server", "lock", "unsupported"]


class BusOwnershipError(RuntimeError):
    """The bus belongs to someone else; nothing was opened."""

    def __init__(
        self,
        reason: OwnershipReason,
        detail: str,
        remediation: str,
        holder: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason: OwnershipReason = reason
        self.detail = detail
        self.remediation = remediation
        self.holder = holder or {}


class GlobalEnergizedIntentError(RuntimeError):
    """A host-global energized-intent marker could not be trusted or persisted."""

    def __init__(self, detail: str, *, path: Path | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.path = path


_held_device_guard = threading.Lock()
_held_device_sets: dict[tuple[int, tuple[str, ...]], int] = {}
_held_device_identities: dict[tuple[int, tuple[str, ...]], tuple[DeviceIdentity, ...]] = {}
_enable_intent_tokens: dict[tuple[int, str], set[str]] = {}


@dataclass(frozen=True)
class ServerProbe:
    """Result of probing the local Suite server."""

    running: bool
    url: str
    detail: str
    http_status: int | None = None
    #: Parsed ``/api/robot/status`` body when it was a JSON object.
    status: dict[str, Any] | None = field(default=None)


def server_base_url(settings: Any) -> str:
    """Base URL clients use for this machine's server (mirrors ``so101 estop``)."""
    host = str(getattr(settings, "host", "") or "").strip()
    if host in {"", "0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(settings.port)}"


def probe_local_server(
    settings: Any,
    *,
    timeout: float | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ServerProbe:
    """Is a server answering at the configured address? Fails closed.

    ``timeout`` overrides both the connect and the read timeout.
    """
    url = server_base_url(settings) + STATUS_PATH
    if timeout is None:
        limits = httpx.Timeout(PROBE_READ_TIMEOUT_S, connect=PROBE_CONNECT_TIMEOUT_S)
        waited = max(PROBE_READ_TIMEOUT_S, PROBE_CONNECT_TIMEOUT_S)
    else:
        limits, waited = httpx.Timeout(timeout), timeout
    try:
        with httpx.Client(timeout=limits, trust_env=False, transport=transport) as client:
            response = client.get(url)
    except httpx.ConnectError as exc:
        return ServerProbe(running=False, url=url, detail=f"nothing is listening ({exc})")
    except httpx.TimeoutException as exc:
        return ServerProbe(
            running=True,
            url=url,
            detail=f"the port did not answer within {waited:g} s ({type(exc).__name__}); "
            "treating it as in use",
        )
    except httpx.HTTPError as exc:
        return ServerProbe(
            running=True,
            url=url,
            detail=f"the probe failed ({type(exc).__name__}: {exc}); treating the port as in use",
        )
    status: dict[str, Any] | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        status = payload
    looks_like_suite = status is not None and "estopped" in status and "mode" in status
    detail = f"HTTP {response.status_code} from {url}"
    if not looks_like_suite:
        detail += " (not a Suite status reply; if another program uses this port, stop it)"
    return ServerProbe(
        running=True,
        url=url,
        detail=detail,
        http_status=response.status_code,
        status=status if looks_like_suite else None,
    )


@dataclass(frozen=True)
class DeviceIdentity:
    """Stable identity for one configured physical serial endpoint."""

    configured_port: str
    configured_path: str
    canonical_port: str
    endpoint_key: str
    node_key: str
    path_key: str
    key: str
    lock_keys: tuple[str, ...]
    resolved: bool


def canonical_device_identity(port: str) -> DeviceIdentity:
    """Resolve a serial-port spelling to the physical endpoint it names.

    POSIX endpoints are pinned by ``st_rdev`` for character devices and by
    ``(st_dev, st_ino)`` for regular-file test seams. Configured and canonical
    path tokens are locked as well, so aliases/hardlinks collide and a node
    replacement at the same path cannot slip past a live owner. Missing paths
    are marked unresolved for the conservative namespace gate.
    """

    configured = str(port).strip()
    if not configured:
        raise BusOwnershipError(
            "unsupported",
            "a real motor-bus port is empty; no device-global lock identity can be made",
            "Set SO101_FOLLOWER_PORT and SO101_LEADER_PORT to the real serial devices.",
        )
    expanded = os.path.expanduser(configured)
    if _USE_MSVCRT:
        # Windows device names are case-insensitive.  realpath/abspath also
        # normalize ordinary paths used by tests. Bare COM names must not be
        # made relative to the checkout: COM3 is the same device everywhere.
        device_name = configured.upper()
        if device_name.startswith("\\\\.\\"):
            device_name = device_name[4:]
        if device_name.startswith("COM") and device_name[3:].isdigit():
            canonical = f"\\\\.\\{device_name}"
            endpoint_key = f"windows-device:{canonical}"
            path_key = f"windows-path:{canonical}"
            return DeviceIdentity(
                configured_port=configured,
                configured_path=canonical,
                canonical_port=canonical,
                endpoint_key=endpoint_key,
                node_key=endpoint_key,
                path_key=path_key,
                key=endpoint_key,
                lock_keys=tuple(sorted({endpoint_key, path_key})),
                resolved=True,
            )
        absolute = os.path.normcase(os.path.abspath(expanded))
        _reject_windows_reparse_chain(Path(absolute).parent)
        try:
            link_info = os.lstat(absolute)
        except (FileNotFoundError, NotADirectoryError):
            endpoint_key = f"windows-unresolved:{absolute}"
            path_key = f"windows-path:{absolute}"
            return DeviceIdentity(
                configured_port=configured,
                configured_path=absolute,
                canonical_port=absolute,
                endpoint_key=endpoint_key,
                node_key=endpoint_key,
                path_key=path_key,
                key=endpoint_key,
                lock_keys=tuple(sorted({endpoint_key, path_key})),
                resolved=False,
            )
        if _stat_is_link_or_reparse(link_info):
            raise BusOwnershipError(
                "unsupported",
                f"Windows motor port {configured!r} is a symlink or reparse point",
                "Use a direct COM device name; Windows motor-port aliases are refused.",
            )
        endpoint_key = f"windows-file:{link_info.st_dev}:{link_info.st_ino}"
        node_key = f"windows-node:{link_info.st_dev}:{link_info.st_ino}"
        path_key = f"windows-path:{absolute}"
        return DeviceIdentity(
            configured_port=configured,
            configured_path=absolute,
            canonical_port=absolute,
            endpoint_key=endpoint_key,
            node_key=node_key,
            path_key=path_key,
            key=endpoint_key,
            lock_keys=tuple(sorted({endpoint_key, node_key, path_key})),
            resolved=True,
        )

    absolute = os.path.abspath(expanded)
    canonical = os.path.realpath(absolute)
    configured_key = f"posix-alias:{absolute}"
    path_key = f"posix-path:{canonical}"
    try:
        info = os.stat(absolute, follow_symlinks=True)
    except (FileNotFoundError, NotADirectoryError):
        endpoint_key = f"posix-unresolved:{canonical}"
        node_key = endpoint_key
        resolved = False
    except OSError as exc:
        raise BusOwnershipError(
            "unsupported",
            f"cannot identify real motor-bus port {configured!r}: {type(exc).__name__}: {exc}",
            "Fix the serial-port path and its parent-directory permissions, then retry.",
        ) from exc
    else:
        node_key = f"posix-node:{info.st_dev}:{info.st_ino}"
        if stat.S_ISCHR(info.st_mode):
            endpoint_key = f"posix-char:{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}"
        else:
            endpoint_key = f"posix-inode:{info.st_dev}:{info.st_ino}"
        resolved = True
    return DeviceIdentity(
        configured_port=configured,
        configured_path=absolute,
        canonical_port=canonical,
        endpoint_key=endpoint_key,
        node_key=node_key,
        path_key=path_key,
        key=endpoint_key,
        lock_keys=tuple(sorted({configured_key, path_key, endpoint_key, node_key})),
        resolved=resolved,
    )


def _stat_is_link_or_reparse(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & reparse)


def configured_bus_ports(settings: Any) -> tuple[str, ...]:
    """Distinct configured real motor ports in follower/leader order."""

    ports: list[str] = []
    for name in ("follower_port", "leader_port"):
        value = str(getattr(settings, name, "") or "").strip()
        if value:
            ports.append(value)
    if not ports:
        raise BusOwnershipError(
            "unsupported",
            "no real motor-bus ports are configured; no device-global lock can be taken",
            "Set SO101_FOLLOWER_PORT and SO101_LEADER_PORT, then retry.",
        )
    if len(ports) > 1 and ports[0] == ports[1]:
        raise BusOwnershipError(
            "unsupported",
            f"follower and leader motor ports are both configured as {ports[0]!r}",
            "Give the follower and leader distinct physical serial adapters, then retry.",
        )
    return tuple(ports)


def device_lock_identities(
    settings: Any, *, ports: tuple[str, ...] | None = None
) -> tuple[DeviceIdentity, ...]:
    """Canonical, de-duplicated device identities in deadlock-safe order."""

    requested = ports if ports is not None else configured_bus_ports(settings)
    by_key: dict[str, DeviceIdentity] = {}
    for port in requested:
        identity = canonical_device_identity(port)
        existing = by_key.get(identity.key)
        if existing is None:
            by_key[identity.key] = identity
        else:
            raise BusOwnershipError(
                "unsupported",
                f"configured motor ports {existing.configured_port!r} and "
                f"{identity.configured_port!r} resolve to the same endpoint "
                f"{identity.canonical_port}",
                "Give the follower and leader distinct physical serial adapters, then retry.",
            )
    if not by_key:
        raise BusOwnershipError(
            "unsupported",
            "no real motor-bus ports were supplied; no device-global lock can be taken",
            "Supply at least one configured real serial port, then retry.",
        )
    return tuple(by_key[key] for key in sorted(by_key))


def device_lock_key(settings: Any, *, ports: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """All lock tokens for the requested ports, in global acquisition order."""

    identities = device_lock_identities(settings, ports=ports)
    return tuple(sorted({key for identity in identities for key in identity.lock_keys}))


def locked_device_port(port: str) -> str:
    """Return the canonical port captured by this process's device lock.

    Hardware constructors call this after the ownership gate.  If a custom
    symlink or udev alias is retargeted after lock acquisition, they keep using
    the endpoint that was actually locked instead of following the changed
    alias to an unlocked bus.  Calls outside a :class:`DeviceBusLock` retain
    their original port for low-level tests and non-Suite library use.
    """

    try:
        return _locked_identity_for_port(port).canonical_port
    except LookupError:
        return str(port)


def require_device_lock(port: str) -> str:
    """Return pinned endpoint or fail before any real serial port is opened."""

    try:
        return _locked_identity_for_port(port).canonical_port
    except LookupError as exc:
        raise BusOwnershipError(
            "unsupported",
            f"refusing to open real motor port {port!r}: no adopted DeviceBusLock covers it",
            "Acquire the matching DeviceBusLock before opening the real bus.",
        ) from exc


def _locked_identity_for_port(port: str) -> DeviceIdentity:
    requested = str(port)
    if _USE_MSVCRT:
        normalized = os.path.normcase(requested)
        absolute = os.path.normcase(os.path.abspath(os.path.expanduser(requested)))
    else:
        normalized = requested
        absolute = os.path.abspath(os.path.expanduser(requested))
    pid = os.getpid()
    with _held_device_guard:
        held = [
            identities
            for (owner_pid, _lock_key), identities in _held_device_identities.items()
            if owner_pid == pid
        ]
    for identities in held:
        for identity in identities:
            configured = identity.configured_port
            configured_absolute = os.path.abspath(os.path.expanduser(configured))
            if _USE_MSVCRT:
                configured = os.path.normcase(configured)
                configured_absolute = os.path.normcase(configured_absolute)
            if normalized in {configured, identity.canonical_port} or absolute in {
                configured_absolute,
                identity.canonical_port,
            }:
                _revalidate_identity_endpoint(identity)
                return identity
    if held:
        covered = sorted(
            {identity.configured_port for identities in held for identity in identities}
        )
        raise BusOwnershipError(
            "unsupported",
            f"real motor port {requested!r} is not covered by this process's held "
            f"device lock(s) for {', '.join(covered)}",
            "Use the configured SO101_FOLLOWER_PORT/SO101_LEADER_PORT, or acquire the "
            "matching device-global lock before opening that bus.",
        )
    raise LookupError(requested)


def _revalidate_identity_endpoint(identity: DeviceIdentity) -> None:
    """Refuse if the endpoint captured by the lock was replaced before open."""

    if not identity.resolved:
        raise BusOwnershipError(
            "unsupported",
            f"motor port {identity.configured_port!r} was unresolved when its lock was acquired",
            "Connect the adapter, then restart the owning command so it can pin the endpoint.",
        )
    if identity.endpoint_key.startswith("windows-device:"):
        return
    try:
        info = os.lstat(identity.canonical_port)
    except OSError as exc:
        raise BusOwnershipError(
            "unsupported",
            f"locked motor endpoint {identity.canonical_port} disappeared before open: {exc}",
            "Stop and restart after the configured adapter is present.",
        ) from exc
    if _stat_is_link_or_reparse(info):
        raise BusOwnershipError(
            "unsupported",
            f"locked motor endpoint {identity.canonical_port} became a symlink/reparse point",
            "Stop and restart with a direct, stable device endpoint.",
        )
    if os.name == "posix" and stat.S_ISCHR(info.st_mode):
        actual = f"posix-char:{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}"
        actual_node = f"posix-node:{info.st_dev}:{info.st_ino}"
    elif os.name == "posix":
        actual = f"posix-inode:{info.st_dev}:{info.st_ino}"
        actual_node = f"posix-node:{info.st_dev}:{info.st_ino}"
    else:
        actual = f"windows-file:{info.st_dev}:{info.st_ino}"
        actual_node = f"windows-node:{info.st_dev}:{info.st_ino}"
    if actual != identity.endpoint_key or actual_node != identity.node_key:
        raise BusOwnershipError(
            "unsupported",
            f"locked motor endpoint {identity.canonical_port} was replaced "
            f"({identity.endpoint_key}/{identity.node_key} -> {actual}/{actual_node})",
            "Stop and restart so the replacement endpoint gets its own device lock.",
        )


def _stable_user_state_root() -> Path:
    """Stable system state independent of TEMP, cwd and SO101_DATA_DIR."""

    if _USE_MSVCRT:
        local = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
        if not local:
            raise BusOwnershipError(
                "unsupported",
                "LOCALAPPDATA is unavailable; no stable Windows motor-lock root exists",
                "Restore the Windows Local AppData known-folder environment and retry.",
            )
        base = Path(local)
        if not base.is_absolute():
            raise BusOwnershipError(
                "unsupported",
                f"LOCALAPPDATA is not absolute: {base}",
                "Restore the Windows Local AppData known-folder environment and retry.",
            )
        _reject_windows_reparse_chain(base)
        return base / "SO101" / "device-ownership-v2"
    if sys.platform == "darwin":
        return Path("/Library/Application Support/SO101/device-ownership-v2")
    return Path("/var/lib/so101/device-ownership-v2")


def _ensure_private_state_dir(path: Path) -> Path:
    try:
        if _USE_MSVCRT:
            _reject_windows_reparse_chain(path.parent)
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            _reject_windows_reparse_chain(path)
        info = os.lstat(path)
    except OSError as exc:
        raise BusOwnershipError(
            "unsupported",
            f"the motor-device state directory {path} is unavailable: {exc}",
            "Re-run the installer so it can create the shared device-ownership directory.",
        ) from exc
    if _stat_is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise BusOwnershipError(
            "unsupported",
            f"the motor-device state directory {path} is a symlink/reparse point "
            "or not a directory",
            "Replace it with a direct system directory and re-run the installer.",
        )
    if os.name == "posix":
        mode = stat.S_IMODE(info.st_mode)
        if info.st_uid != 0 or mode & 0o007 or (mode & 0o070) != 0o070 or (mode & 0o3000) != 0o3000:
            raise BusOwnershipError(
                "unsupported",
                f"the shared motor-device state directory {path} has unsafe ownership/mode",
                "Re-run the installer (expected root ownership, hardware group, mode 3770).",
            )
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            raise BusOwnershipError(
                "unsupported",
                f"this user cannot access shared motor-device state directory {path}",
                "Join the configured hardware group (normally dialout), log out/in, and retry.",
            )
    return path


def _reject_windows_reparse_chain(path: Path) -> None:
    if not _USE_MSVCRT:
        return
    current = path
    while True:
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise BusOwnershipError(
                "unsupported",
                f"cannot inspect Windows state path {current}: {exc}",
                "Repair Local AppData and retry.",
            ) from exc
        else:
            if _stat_is_link_or_reparse(info):
                raise BusOwnershipError(
                    "unsupported",
                    f"Windows state path {current} is a symlink or reparse point",
                    "Use a direct Local AppData directory; reparse points are refused.",
                )
        parent = current.parent
        if parent == current:
            return
        current = parent


def _device_lock_root() -> Path:
    return _ensure_private_state_dir(_stable_user_state_root() / "locks")


def _validate_device_lock_root(root: Path) -> None:
    _ensure_private_state_dir(root)


def _validate_lock_namespace_root(root: Path) -> None:
    _validate_device_lock_root(root)


def _device_lock_path_for_key(key: str) -> Path:
    """Opaque host-global lock path for one canonical/configured identity token."""

    root = _device_lock_root()
    _validate_lock_namespace_root(root)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return root / f"{DEVICE_LOCK_PREFIX}-{digest}.lock"


def device_lock_path(identity: DeviceIdentity) -> Path:
    """Compatibility path for the identity's canonical token."""

    return _device_lock_path_for_key(identity.key)


def _device_namespace_lock_path() -> Path:
    root = _device_lock_root()
    _validate_lock_namespace_root(root)
    return root / f"{DEVICE_LOCK_PREFIX}-namespace.lock"


def _global_marker_root() -> Path:
    return _ensure_private_state_dir(_stable_user_state_root() / "intent-events")


def _validate_global_marker_root(root: Path) -> None:
    _validate_device_lock_root(root)


def _intent_identities(settings: Any, device_name: str | None) -> tuple[DeviceIdentity, ...]:
    name = str(device_name or "").strip().lower()
    ports: tuple[str, ...]
    if name == "follower":
        ports = (str(settings.follower_port),)
    elif name == "leader":
        ports = (str(settings.leader_port),)
    else:
        ports = configured_bus_ports(settings)
    return device_lock_identities(settings, ports=ports)


def _intent_key_map(settings: Any, device_name: str | None = None) -> dict[str, DeviceIdentity]:
    return {
        key: identity
        for identity in _intent_identities(settings, device_name)
        for key in identity.lock_keys
    }


def global_energized_intent_paths(
    settings: Any, *, device_name: str | None = None
) -> tuple[Path, ...]:
    """Immutable journal records currently relevant to selected device(s)."""

    identities = _intent_identities(settings, device_name)
    relevant = _relevant_scope_digests(identities)
    return tuple(
        path
        for path in sorted(_global_marker_root().glob("*.json"))
        if set(_event_scopes(path.name)) & relevant
    )


def _require_device_keys_held(keys: set[str]) -> None:
    """Require this process to own one device-lock set covering ``keys``."""

    pid = os.getpid()
    with _held_device_guard:
        held_sets = [
            set(lock_key)
            for (owner_pid, lock_key), count in _held_device_sets.items()
            if owner_pid == pid and count > 0
        ]
    if not any(keys <= held for held in held_sets):
        raise GlobalEnergizedIntentError(
            "global energized intent may only be read or changed while this process holds "
            "the matching motor-device lock"
        )


def device_lock_is_held(settings: Any, *, device_name: str | None = None) -> bool:
    """Whether this process owns a lock set covering the configured device scope."""

    try:
        keys = {
            key
            for identity in _intent_identities(settings, device_name)
            for key in identity.lock_keys
        }
    except BusOwnershipError:
        return False
    pid = os.getpid()
    with _held_device_guard:
        return any(
            owner_pid == pid and count > 0 and keys <= set(lock_key)
            for (owner_pid, lock_key), count in _held_device_sets.items()
        )


def _marker_open_flags(*, writable: bool) -> int:
    flags = (os.O_WRONLY if writable else os.O_RDONLY) | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _validate_marker_fd(fd: int, path: Path) -> os.stat_result:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise GlobalEnergizedIntentError(
            f"refusing unsafe global energized-intent marker {path}: expected one regular file",
            path=path,
        )
    return info


def _fsync_marker_parent(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(path.parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _atomic_create_marker(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable fsynced record; never truncate or overwrite."""

    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if len(data) > _GLOBAL_MARKER_MAX_BYTES:
        raise GlobalEnergizedIntentError(
            f"global energized-intent marker payload exceeds {_GLOBAL_MARKER_MAX_BYTES} bytes",
            path=path,
        )
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = _marker_open_flags(writable=True) | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(temp, flags, 0o640)
    except OSError as exc:
        raise GlobalEnergizedIntentError(
            f"cannot create temporary energized-intent record {temp}: {exc}", path=path
        ) from exc
    try:
        _validate_marker_fd(fd, temp)
        if os.name == "posix":
            os.fchmod(fd, 0o640)
        _write_all(fd, data)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    os.close(fd)
    try:
        if os.name == "nt":
            # Windows rename refuses an existing destination. The root and
            # temporary file were already checked for reparse points.
            os.rename(temp, path)
        else:
            os.link(temp, path, follow_symlinks=False)
            temp.unlink()
        _fsync_marker_parent(path)
    except OSError as exc:
        try:
            temp.unlink()
        except OSError:
            pass
        raise GlobalEnergizedIntentError(
            f"could not atomically publish energized-intent record {path}: {exc}", path=path
        ) from exc


def _safe_read_marker(path: Path) -> dict[str, Any]:
    try:
        before = os.lstat(path)
        if _stat_is_link_or_reparse(before):
            raise OSError("symlink/reparse point refused")
        fd = os.open(path, _marker_open_flags(writable=False))
    except OSError as exc:
        raise GlobalEnergizedIntentError(
            f"cannot safely read marker {path}: {exc}", path=path
        ) from exc
    try:
        info = _validate_marker_fd(fd, path)
        current = os.lstat(path)
        if _stat_is_link_or_reparse(current) or (
            current.st_dev,
            current.st_ino,
        ) != (info.st_dev, info.st_ino):
            raise GlobalEnergizedIntentError(f"marker {path} changed while opened", path=path)
        raw = os.read(fd, _GLOBAL_MARKER_MAX_BYTES + 1)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    try:
        if len(raw) > _GLOBAL_MARKER_MAX_BYTES:
            raise ValueError("record is too large")
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise GlobalEnergizedIntentError(f"marker {path} is unreadable ({exc})", path=path) from exc
    if not isinstance(payload, dict):
        raise GlobalEnergizedIntentError(f"marker {path} is not a JSON object", path=path)
    return payload


def _scope_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _event_name(kind: str, keys: set[str], event_id: str) -> str:
    scopes = ".".join(sorted(_scope_digest(key) for key in keys))
    return f"{kind}--{scopes}--{event_id}.json"


def _event_scopes(name: str) -> tuple[str, ...]:
    parts = name.split("--")
    if len(parts) != 3 or not parts[2].endswith(".json"):
        return ()
    return tuple(item for item in parts[1].split(".") if item)


def _alias_root() -> Path:
    path = _global_marker_root() / "aliases"
    try:
        path.mkdir(mode=0o3770, exist_ok=True)
    except OSError as exc:
        raise GlobalEnergizedIntentError(f"cannot create alias journal {path}: {exc}") from exc
    _validate_global_marker_root(path)
    return path


def _alias_key(path: str) -> str:
    prefix = "windows-alias" if _USE_MSVCRT else "posix-alias"
    normalized = os.path.normcase(path) if _USE_MSVCRT else path
    return f"{prefix}:{normalized}"


def _ensure_alias_descriptor(identity: DeviceIdentity) -> None:
    alias = identity.configured_path
    key = _alias_key(alias)
    payload = {
        "kind": _GLOBAL_ALIAS_KIND,
        "schema_version": GLOBAL_ENERGIZED_INTENT_SCHEMA_VERSION,
        "alias_key": key,
        "alias_path": alias,
    }
    digest = _scope_digest(key)
    path = _alias_root() / f"alias-{digest}.json"
    if path.exists():
        try:
            existing = _safe_read_marker(path)
        except GlobalEnergizedIntentError:
            existing = None
        if existing == payload:
            return
        # Keep the immutable corrupt/conflicting record for fail-closed
        # diagnosis, but publish one recoverable descriptor. This exceptional
        # path may grow; the ordinary per-command path is deterministic.
        path = _alias_root() / f"alias-{digest}-{uuid.uuid4().hex}.json"
    _atomic_create_marker(path, payload)


def _discovered_alias_keys(identities: tuple[DeviceIdentity, ...]) -> set[str]:
    endpoints = {identity.endpoint_key for identity in identities}
    aliases: set[str] = set()
    valid_digests: set[str] = set()
    corrupt_digests: set[str] = set()
    for path in sorted(_alias_root().glob("alias-*.json")):
        parts = path.stem.split("-")
        digest = parts[1] if len(parts) >= 2 else ""
        try:
            payload = _safe_read_marker(path)
        except GlobalEnergizedIntentError:
            if digest:
                corrupt_digests.add(digest)
            continue
        if (
            payload.get("kind") != _GLOBAL_ALIAS_KIND
            or payload.get("schema_version") != GLOBAL_ENERGIZED_INTENT_SCHEMA_VERSION
            or not isinstance(payload.get("alias_path"), str)
            or not isinstance(payload.get("alias_key"), str)
            or digest != _scope_digest(str(payload.get("alias_key")))
        ):
            if digest:
                corrupt_digests.add(digest)
            continue
        valid_digests.add(digest)
        try:
            current = canonical_device_identity(payload["alias_path"])
        except BusOwnershipError:
            continue
        if current.resolved and current.endpoint_key in endpoints:
            aliases.add(str(payload["alias_key"]))
    # Valid intent records redundantly carry alias paths, so a torn descriptor
    # is recoverable without mutating or trusting it.
    for path in sorted(_global_marker_root().glob("intent--*.json")):
        try:
            payload = _safe_read_marker(path)
        except GlobalEnergizedIntentError:
            continue
        if not _valid_intent(payload):
            continue
        for alias_path in payload["alias_paths"]:
            alias_key = _alias_key(alias_path)
            valid_digests.add(_scope_digest(alias_key))
            try:
                current = canonical_device_identity(alias_path)
            except BusOwnershipError:
                continue
            if current.resolved and current.endpoint_key in endpoints:
                aliases.add(alias_key)
    unresolved_corrupt = corrupt_digests - valid_digests
    if unresolved_corrupt:
        raise GlobalEnergizedIntentError(
            "alias journal contains an unreadable descriptor with no redundant valid record"
        )
    return aliases


def _relevant_keys(identities: tuple[DeviceIdentity, ...]) -> set[str]:
    keys = {key for identity in identities for key in identity.lock_keys}
    keys.update(_discovered_alias_keys(identities))
    return keys


def _relevant_scope_digests(identities: tuple[DeviceIdentity, ...]) -> set[str]:
    return {_scope_digest(key) for key in _relevant_keys(identities)}


def _valid_intent(payload: dict[str, Any]) -> bool:
    keys = payload.get("device_keys")
    aliases = payload.get("alias_paths")
    return (
        payload.get("kind") == GLOBAL_ENERGIZED_INTENT_KIND
        and payload.get("schema_version") == GLOBAL_ENERGIZED_INTENT_SCHEMA_VERSION
        and payload.get("backend") == "real"
        and isinstance(payload.get("event_id"), str)
        and isinstance(payload.get("endpoint_key"), str)
        and isinstance(payload.get("node_key"), str)
        and isinstance(keys, list)
        and all(isinstance(key, str) and key for key in keys)
        and keys == sorted(set(keys))
        and payload.get("endpoint_key") in keys
        and payload.get("node_key") in keys
        and isinstance(aliases, list)
        and all(isinstance(alias, str) for alias in aliases)
        and isinstance(payload.get("purpose"), str)
        and payload.get("device_name") in {"", "follower", "leader"}
        and isinstance(payload.get("written_at"), str)
        and payload.get("energized") is True
    )


def _valid_clear(payload: dict[str, Any]) -> bool:
    targets = payload.get("target_clears")
    corrupt = payload.get("corrupt_clears")
    return (
        payload.get("kind") == _GLOBAL_CLEAR_KIND
        and payload.get("schema_version") == GLOBAL_ENERGIZED_INTENT_SCHEMA_VERSION
        and isinstance(payload.get("event_id"), str)
        and isinstance(targets, dict)
        and all(
            isinstance(event_id, str)
            and isinstance(keys, list)
            and all(isinstance(key, str) for key in keys)
            for event_id, keys in targets.items()
        )
        and isinstance(corrupt, dict)
        and all(
            isinstance(name, str)
            and isinstance(scopes, list)
            and all(isinstance(scope, str) for scope in scopes)
            for name, scopes in corrupt.items()
        )
    )


@dataclass(frozen=True)
class _JournalState:
    active: tuple[dict[str, Any], ...]
    corrupt: tuple[Path, ...]
    relevant_keys: frozenset[str]


def _journal_state(identities: tuple[DeviceIdentity, ...]) -> _JournalState:
    relevant_keys = _relevant_keys(identities)
    relevant_scopes = {_scope_digest(key) for key in relevant_keys}
    intents: list[dict[str, Any]] = []
    clears: list[dict[str, Any]] = []
    corrupt: list[Path] = []
    for path in sorted(_global_marker_root().glob("*.json")):
        scopes = set(_event_scopes(path.name))
        if not scopes & relevant_scopes:
            continue
        try:
            payload = _safe_read_marker(path)
        except GlobalEnergizedIntentError:
            corrupt.append(path)
            continue
        if _valid_intent(payload) and scopes == {
            _scope_digest(key) for key in payload["device_keys"]
        }:
            payload = {**payload, "_path": str(path)}
            intents.append(payload)
        elif _valid_clear(payload):
            clears.append(payload)
        else:
            corrupt.append(path)
    cleared: dict[str, set[str]] = {}
    corrupt_clears: dict[str, set[str]] = {}
    for clear in clears:
        for event_id, keys in clear["target_clears"].items():
            cleared.setdefault(event_id, set()).update(keys)
        for name, scopes in clear["corrupt_clears"].items():
            corrupt_clears.setdefault(name, set()).update(scopes)
    active: list[dict[str, Any]] = []
    for intent in intents:
        remaining = set(intent["device_keys"]) - cleared.get(intent["event_id"], set())
        if remaining & relevant_keys:
            active.append({**intent, "remaining_device_keys": sorted(remaining)})
    uncleared_corrupt = []
    for path in corrupt:
        remaining = set(_event_scopes(path.name)) - corrupt_clears.get(path.name, set())
        if remaining & relevant_scopes:
            uncleared_corrupt.append(path)
    return _JournalState(tuple(active), tuple(uncleared_corrupt), frozenset(relevant_keys))


def _append_intent_records(
    settings: Any,
    *,
    identities: tuple[DeviceIdentity, ...],
    purpose: str,
    device_name: str,
) -> tuple[dict[str, Any], ...]:
    written: list[dict[str, Any]] = []
    pending_tokens: list[tuple[DeviceIdentity, str]] = []
    for identity in identities:
        _ensure_alias_descriptor(identity)
        event_id = uuid.uuid4().hex
        record_device = device_name
        if not record_device:
            if identity.configured_port == str(getattr(settings, "follower_port", "")):
                record_device = "follower"
            elif identity.configured_port == str(getattr(settings, "leader_port", "")):
                record_device = "leader"
        payload = {
            "kind": GLOBAL_ENERGIZED_INTENT_KIND,
            "schema_version": GLOBAL_ENERGIZED_INTENT_SCHEMA_VERSION,
            "event_id": event_id,
            "device_keys": sorted(identity.lock_keys),
            "endpoint_key": identity.endpoint_key,
            "node_key": identity.node_key,
            "alias_paths": [identity.configured_path],
            "configured_port": identity.configured_port,
            "canonical_port": identity.canonical_port,
            "backend": "real",
            "follower_id": str(getattr(settings, "follower_id", "") or ""),
            "leader_id": str(getattr(settings, "leader_id", "") or ""),
            "purpose": str(purpose),
            "device_name": str(record_device),
            "pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "energized": True,
        }
        path = _global_marker_root() / _event_name("intent", set(identity.lock_keys), event_id)
        _atomic_create_marker(path, payload)
        pending_tokens.append((identity, event_id))
        written.append(payload)
    with _held_device_guard:
        for identity, event_id in pending_tokens:
            _enable_intent_tokens.setdefault((os.getpid(), identity.endpoint_key), set()).add(
                event_id
            )
    return tuple(written)


def write_global_energized_intent(
    settings: Any,
    *,
    purpose: str,
    device_name: str = "follower",
) -> tuple[dict[str, Any], ...]:
    """Append immutable intent records and mint raw torque-enable capability."""

    identities = _intent_identities(settings, device_name)
    current_keys = {key for identity in identities for key in identity.lock_keys}
    _require_device_keys_held(current_keys)
    return _append_intent_records(
        settings,
        identities=identities,
        purpose=purpose,
        device_name=device_name,
    )


def clear_global_energized_intent(
    settings: Any,
    *,
    purpose: str = "confirmed torque-off",
    device_name: str = "follower",
    force_current_aliases: bool = False,
    recover_corrupt_current: bool = True,
) -> None:
    """Append one scoped clear event; never overwrite an intent record."""

    identities = _intent_identities(settings, device_name)
    current_keys = {key for identity in identities for key in identity.lock_keys}
    _require_device_keys_held(current_keys)
    for identity in identities:
        _ensure_alias_descriptor(identity)
    state = _journal_state(identities)
    target_clears: dict[str, list[str]] = {}
    exact_endpoints = {(identity.endpoint_key, identity.node_key) for identity in identities}
    for intent in state.active:
        if (intent.get("endpoint_key"), intent.get("node_key")) in exact_endpoints:
            target_clears[str(intent["event_id"])] = list(intent["device_keys"])
            continue
        overlap = set(intent["remaining_device_keys"]) & set(state.relevant_keys)
        if overlap and not force_current_aliases:
            raise GlobalEnergizedIntentError(
                "an active alias marker belongs to a different pinned endpoint; "
                "ordinary confirmed-off recovery cannot clear it",
                path=Path(str(intent["_path"])),
            )
        if overlap:
            target_clears[str(intent["event_id"])] = sorted(overlap)
    corrupt_clears: dict[str, list[str]] = {}
    if state.corrupt:
        if not recover_corrupt_current:
            raise GlobalEnergizedIntentError(
                f"{len(state.corrupt)} relevant energized-intent record(s) are corrupt",
                path=state.corrupt[0],
            )
        current_scopes = {_scope_digest(key) for key in current_keys}
        node_scopes = {_scope_digest(identity.node_key) for identity in identities}
        relevant_scopes = {_scope_digest(key) for key in state.relevant_keys}
        for path in state.corrupt:
            record_scopes = set(_event_scopes(path.name))
            if record_scopes & node_scopes:
                overlap = record_scopes & current_scopes
            elif force_current_aliases:
                overlap = record_scopes & relevant_scopes
            else:
                overlap = set()
            if overlap:
                corrupt_clears[path.name] = sorted(overlap)
    if target_clears or corrupt_clears:
        event_id = uuid.uuid4().hex
        payload = {
            "kind": _GLOBAL_CLEAR_KIND,
            "schema_version": GLOBAL_ENERGIZED_INTENT_SCHEMA_VERSION,
            "event_id": event_id,
            "target_clears": target_clears,
            "corrupt_clears": corrupt_clears,
            "purpose": str(purpose),
            "pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        path = _global_marker_root() / _event_name("clear", current_keys, event_id)
        _atomic_create_marker(path, payload)
    with _held_device_guard:
        for identity in identities:
            _enable_intent_tokens.pop((os.getpid(), identity.endpoint_key), None)
    _compact_inactive_journal(identities)


def _compact_inactive_journal(identities: tuple[DeviceIdentity, ...]) -> None:
    """Best-effort bounded cleanup after a fully cleared, stable device scope.

    Intent records are removed and directory-fsynced before their clear
    records. A crash can therefore leave redundant clears, but can never
    resurrect an intent by durably deleting its clear first. Complex alias,
    partial-clear and corrupt recovery histories remain append-only.
    """

    try:
        state = _journal_state(identities)
        if state.active or state.corrupt:
            return
        root = _global_marker_root()
        current_keys = {key for identity in identities for key in identity.lock_keys}
        current_scopes = {_scope_digest(key) for key in current_keys}
        intents: dict[str, Path] = {}
        clears: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(root.glob("*.json")):
            scopes = set(_event_scopes(path.name))
            if not scopes or not scopes <= current_scopes:
                continue
            payload = _safe_read_marker(path)
            if _valid_intent(payload):
                if set(payload["device_keys"]) <= current_keys:
                    intents[str(payload["event_id"])] = path
            elif _valid_clear(payload):
                clears.append((path, payload))
            else:
                return
        if not intents:
            return
        for path in intents.values():
            path.unlink()
        _fsync_marker_parent(root / ".intent-compaction")
        removable_clears = []
        removed_ids = set(intents)
        for path, payload in clears:
            targets = payload["target_clears"]
            if (
                not payload["corrupt_clears"]
                and targets
                and set(targets) <= removed_ids
                and all(set(keys) <= current_keys for keys in targets.values())
            ):
                removable_clears.append(path)
        for path in removable_clears:
            path.unlink()
        if removable_clears:
            _fsync_marker_parent(root / ".clear-compaction")
    except (OSError, GlobalEnergizedIntentError):
        # Compaction is maintenance after the durable clear. Leaving any
        # immutable history behind is safe and must not turn a confirmed-off
        # operation into a false failure.
        return


def read_global_energized_intent(settings: Any) -> dict[str, Any] | None:
    """Read active journal intent for configured devices under their lock."""

    identities = _intent_identities(settings, None)
    current_keys = {key for identity in identities for key in identity.lock_keys}
    _require_device_keys_held(current_keys)
    state = _journal_state(identities)
    if state.corrupt:
        raise GlobalEnergizedIntentError(
            f"energized-intent journal contains {len(state.corrupt)} corrupt relevant record(s); "
            "treat the arm as energized",
            path=state.corrupt[0],
        )
    if not state.active:
        return None
    first = state.active[0]
    return {
        **first,
        "device_keys": sorted(
            {key for intent in state.active for key in intent["remaining_device_keys"]}
        ),
        "markers": list(state.active),
        "energized": True,
    }


def require_enable_intent(port: str) -> None:
    """Raw real-torque capability: held pinned lock plus fresh durable intent."""

    try:
        identity = _locked_identity_for_port(port)
    except LookupError as exc:
        raise BusOwnershipError(
            "unsupported",
            f"refusing torque-enable on {port!r}: no adopted DeviceBusLock covers it",
            "Acquire the matching DeviceBusLock and durable energized intent first.",
        ) from exc
    with _held_device_guard:
        tokens = _enable_intent_tokens.get((os.getpid(), identity.endpoint_key), set())
    if not tokens:
        raise BusOwnershipError(
            "unsupported",
            f"refusing torque-enable on {port!r}: no durable global intent token was adopted",
            "Acquire the matching DeviceBusLock and write global energized intent first.",
        )


def enable_intent_is_adopted(settings: Any, *, device_name: str = "follower") -> bool:
    """No-disk capability query for re-use after a durable intent already exists."""

    identities = _intent_identities(settings, device_name)
    current_keys = {key for identity in identities for key in identity.lock_keys}
    _require_device_keys_held(current_keys)
    with _held_device_guard:
        return all(
            bool(_enable_intent_tokens.get((os.getpid(), identity.endpoint_key)))
            for identity in identities
        )


def bus_lock_path(settings: Any) -> Path:
    """Compatibility view of the first configured device-global lock path."""

    return device_lock_path(device_lock_identities(settings)[0])


def read_lock_holder(path: Path) -> dict[str, Any]:
    """Best-effort read of the holder record written by :class:`BusLock`."""
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
    except (OSError, ValueError):
        return {}
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return {}
        return _read_lock_holder_fd(fd)
    finally:
        os.close(fd)


def _read_lock_holder_fd(fd: int) -> dict[str, Any]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 16 * 1024)
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (OSError, UnicodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _lock_fd(fd: int, *, exclusive: bool = True) -> None:
    """Non-blocking advisory lock on ``fd``; OSError when incompatible."""
    if _USE_MSVCRT:
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, mode | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    if _USE_MSVCRT:
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class BusLock:
    """Non-blocking advisory lock on one safely opened file.

    The lock file is never deleted (deleting a locked file lets a second
    process lock a fresh one); the OS releases the lock when the holder exits,
    including on a crash.
    """

    def __init__(
        self,
        path: Path,
        *,
        purpose: str,
        shared: bool = False,
        resource: str | None = None,
        holder_fields: dict[str, Any] | None = None,
        exclusive: bool = True,
        write_holder: bool = True,
        read_holder_on_contention: bool = True,
    ) -> None:
        self.path = Path(path)
        self.purpose = purpose
        self.shared = shared
        self.resource = resource or str(self.path)
        self.holder_fields = dict(holder_fields or {})
        self.exclusive = exclusive
        self.write_holder = write_holder
        self.read_holder_on_contention = read_holder_on_contention
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> BusLock:
        if self._fd is not None:
            return self
        if not self.shared:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            before = os.lstat(self.path)
        except FileNotFoundError:
            before = None
        except OSError as exc:
            raise BusOwnershipError(
                "unsupported",
                f"cannot inspect motor-bus lock {self.path}: {exc}",
                "Repair the shared device-ownership directory and retry.",
            ) from exc
        if before is not None and _stat_is_link_or_reparse(before):
            raise BusOwnershipError(
                "unsupported",
                f"refusing symlink/reparse-point motor-bus lock {self.path}",
                "Remove the unsafe object and re-run the installer.",
            )
        try:
            # Existing files are opened without O_CREAT; creation is exclusive.
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            try:
                fd = os.open(
                    self.path,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o660,
                )
            except FileExistsError:
                # Another process created the permanent file after our first
                # open. It never gets deleted by Suite code, so one retry is
                # enough and does not introduce an unlink race.
                try:
                    fd = os.open(self.path, flags)
                except OSError as exc:
                    raise BusOwnershipError(
                        "unsupported",
                        f"cannot safely open the motor-bus lock for {self.resource}: "
                        f"{type(exc).__name__}: {exc}",
                        "Fix the reported lock-file permissions and retry.",
                    ) from exc
            except OSError as exc:
                raise BusOwnershipError(
                    "unsupported",
                    f"cannot safely create the motor-bus lock for {self.resource}: "
                    f"{type(exc).__name__}: {exc}",
                    "Remove a conflicting object at the reported lock path and retry.",
                ) from exc
        except OSError as exc:
            raise BusOwnershipError(
                "unsupported",
                f"cannot safely open the motor-bus lock for {self.resource}: "
                f"{type(exc).__name__}: {exc}",
                "Remove a conflicting symlink or non-file at the reported lock path, fix its "
                "permissions, and retry.",
            ) from exc
        try:
            info = os.fstat(fd)
            current = os.lstat(self.path)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise BusOwnershipError(
                    "unsupported",
                    f"refusing unsafe motor-bus lock {self.path}: expected one regular file",
                    "Remove the conflicting object and retry; lock files must not be symlinks, "
                    "devices or hard links.",
                )
            if _stat_is_link_or_reparse(current) or (
                current.st_dev,
                current.st_ino,
            ) != (info.st_dev, info.st_ino):
                raise BusOwnershipError(
                    "unsupported",
                    f"motor-bus lock {self.path} changed while it was opened",
                    "Remove the unsafe object and re-run the installer.",
                )
            if os.name == "posix" and info.st_uid == os.geteuid():
                os.fchmod(fd, 0o660)
            _lock_fd(fd, exclusive=self.exclusive)
        except BusOwnershipError:
            os.close(fd)
            raise
        except ImportError as exc:  # neither fcntl nor msvcrt: not a supported OS
            os.close(fd)
            raise BusOwnershipError(
                "unsupported",
                f"no file locking on this platform ({exc})",
                "Run the bring-up kit on Windows, Linux or macOS.",
            ) from exc
        except OSError as exc:
            holder = _read_lock_holder_fd(fd) if self.read_holder_on_contention else {}
            os.close(fd)
            who = (
                f"pid {holder.get('pid')} ({holder.get('purpose', 'unknown purpose')}, "
                f"since {holder.get('since', '?')})"
                if holder
                else "another process"
            )
            raise BusOwnershipError(
                "lock",
                f"{self.resource} is held by {who} (device-global lock {self.path})",
                "Wait for the other bring-up command to finish, or stop it.",
                holder,
            ) from exc
        if self.write_holder:
            record = {
                "pid": os.getpid(),
                "purpose": self.purpose,
                "since": datetime.now(timezone.utc).isoformat(),
                **self.holder_fields,
            }
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
                os.fsync(fd)
            except OSError:
                pass  # the lock itself is what matters
        self._fd = fd
        return self

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock_fd(fd)
        except OSError:
            pass  # closing the descriptor releases it anyway
        finally:
            os.close(fd)

    def __enter__(self) -> BusLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class DeviceBusLock:
    """One deadlock-safe lock set covering canonical physical motor buses."""

    def __init__(
        self,
        settings: Any,
        *,
        purpose: str,
        ports: tuple[str, ...] | None = None,
    ) -> None:
        self.purpose = purpose
        self._requested_ports = ports if ports is not None else configured_bus_ports(settings)
        self.identities = device_lock_identities(settings, ports=self._requested_ports)
        self.lock_key = device_lock_key(settings, ports=self._requested_ports)
        self.paths = tuple(_device_lock_path_for_key(key) for key in self.lock_key)
        self._uses_namespace_gate = os.name == "posix"
        self._namespace_exclusive = any(not identity.resolved for identity in self.identities)
        # Compatibility for diagnostics and callers that showed one bus.lock.
        self.path = self.paths[0]
        self._locks: list[BusLock] = []
        self._registered = False

    @property
    def held(self) -> bool:
        expected = len(self.lock_key) + int(self._uses_namespace_gate)
        return bool(self._locks) and len(self._locks) == expected

    @property
    def description(self) -> str:
        return ", ".join(identity.canonical_port for identity in self.identities)

    def acquire(self) -> DeviceBusLock:
        if self.held:
            return self
        try:
            if self._uses_namespace_gate:
                # A resolved device takes a shared namespace gate and then its
                # own exclusive lock, so disjoint ports can proceed.  A
                # missing/dangling configured path has no physical identity
                # yet and therefore takes the gate exclusively: no process can
                # later lock a newly appeared target under another spelling.
                namespace = BusLock(
                    _device_namespace_lock_path(),
                    purpose=self.purpose,
                    shared=True,
                    resource="the physical motor-bus lock namespace",
                    holder_fields={
                        "configured_ports": [
                            identity.configured_port for identity in self.identities
                        ],
                    },
                    exclusive=self._namespace_exclusive,
                    write_holder=self._namespace_exclusive,
                    # Shared holders cannot safely rewrite one holder record,
                    # so namespace contention deliberately reports "another
                    # process" rather than risk naming a stale PID.
                    read_holder_on_contention=False,
                ).acquire()
                self._locks.append(namespace)
            identity_by_key = {
                key: identity for identity in self.identities for key in identity.lock_keys
            }
            for key in self.lock_key:
                identity = identity_by_key[key]
                lock = BusLock(
                    _device_lock_path_for_key(key),
                    purpose=self.purpose,
                    shared=True,
                    resource=f"physical motor bus {identity.canonical_port}",
                    holder_fields={
                        "configured_port": identity.configured_port,
                        "canonical_port": identity.canonical_port,
                        "device_identity": key,
                    },
                ).acquire()
                self._locks.append(lock)

            # Catch an alias that changed while the lock set was being taken.
            # Device links in /dev are privileged on the supported hosts, but
            # custom paths may live in a caller-writable directory.
            after = device_lock_identities(object(), ports=self._requested_ports)
            after_key = tuple(sorted({key for identity in after for key in identity.lock_keys}))
            if after_key != self.lock_key:
                raise BusOwnershipError(
                    "unsupported",
                    "a configured motor-bus path changed while its device-global lock was "
                    "being acquired",
                    "Use stable /dev paths (the installed udev aliases on Linux), then retry.",
                )
            with _held_device_guard:
                registry_key = (os.getpid(), self.lock_key)
                _held_device_sets[registry_key] = _held_device_sets.get(registry_key, 0) + 1
                _held_device_identities[registry_key] = self.identities
            self._registered = True
        except BaseException:
            self.release()
            raise
        return self

    def release(self) -> None:
        if self._registered:
            with _held_device_guard:
                registry_key = (os.getpid(), self.lock_key)
                remaining = _held_device_sets.get(registry_key, 0) - 1
                if remaining > 0:
                    _held_device_sets[registry_key] = remaining
                else:
                    _held_device_sets.pop(registry_key, None)
                    _held_device_identities.pop(registry_key, None)
                    for identity in self.identities:
                        _enable_intent_tokens.pop((os.getpid(), identity.endpoint_key), None)
            self._registered = False
        locks, self._locks = self._locks, []
        for lock in reversed(locks):
            lock.release()

    def __enter__(self) -> DeviceBusLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


ServerProber = Callable[[Any], ServerProbe]


def server_running_error(probe: ServerProbe) -> BusOwnershipError:
    return BusOwnershipError(
        "server",
        f"a Suite server appears to be running ({probe.detail}); it owns the serial buses",
        "Stop the server (Ctrl-C in its terminal) and re-run, or use the server's "
        "POST /api/preflight, which goes through RobotManager.",
    )


def acquire_bus_access(
    settings: Any,
    *,
    purpose: str,
    server_probe: ServerProber = probe_local_server,
    ports: tuple[str, ...] | None = None,
) -> tuple[DeviceBusLock, ServerProbe]:
    """Probe for a server, then lock the canonical devices. Raises on refusal."""
    probe = server_probe(settings)
    if probe.running:
        raise server_running_error(probe)
    lock = DeviceBusLock(settings, purpose=purpose, ports=ports).acquire()
    return lock, probe


#: Holder purpose the server writes into each device-lock record.
SERVER_LOCK_PURPOSE = "so101 server (Suite, RobotManager)"


class ServerBusLockError(RuntimeError):
    """The real-backend server cannot start: a motor-device lock is held."""


def take_server_bus_lock(settings: Any) -> DeviceBusLock | None:
    """The server's device locks, held for its lifetime; None off the real backend.

    Raises :class:`ServerBusLockError` naming the holder when a CLI command or
    another server has locked either configured physical port.
    """
    from ..hardware.backend import current_backend

    if current_backend(settings) != "real":
        return None
    try:
        return DeviceBusLock(settings, purpose=SERVER_LOCK_PURPOSE).acquire()
    except BusOwnershipError as exc:
        if exc.reason == "lock":
            message = (
                f"cannot start the Suite server on the real backend: {exc.detail}, which owns "
                "the serial bus. Let that command finish or stop it (Ctrl-C in its terminal), "
                "then start the server again."
            )
        else:
            message = (
                f"cannot safely start the Suite server on the real backend: {exc.detail}. "
                f"{exc.remediation}"
            )
        raise ServerBusLockError(message) from exc
