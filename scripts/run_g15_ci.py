#!/usr/bin/env python3
"""Run the one-shot, hardware-free G15 software CI gate and seal its evidence.

This script intentionally uses only the Python 3.12 standard library so it can
run before ``make bootstrap``.  It accepts exactly one source commit and one
fresh evidence directory, runs ``make bootstrap`` once, then (only when that
passes) runs ``make ci`` once.  It never runs doctor, preflight, HIL, an
installed-artifact rehearsal, or any robot/camera command.

The exit status is zero only when both commands pass and the checkout remains
at the expected clean commit.  Even then ``result.json`` says
``release_ready=false``: real G15 install, doctor, and HIL evidence are separate
release gates.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol

SCHEMA = "so101.g15_ci_evidence/v1"
ENVIRONMENT_SCHEMA = "so101.g15_ci_environment/v1"
EVIDENCE_DIR_MODE = 0o700
EVIDENCE_FILE_MODE = 0o600
HASH_MANIFEST = "SHA256SUMS"
RESULT_FILE = "result.json"
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
NODE_VERSION_RE = re.compile(r"v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:[-+].*)?\Z")
REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("bootstrap", ("make", "bootstrap"), "bootstrap.log"),
    ("ci", ("make", "ci"), "ci.log"),
)
TRUSTED_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SYSTEM_TEMP_ROOT = Path("/tmp")
DMI_SYS_VENDOR = Path("/sys/class/dmi/id/sys_vendor")
DMI_PRODUCT_NAME = Path("/sys/class/dmi/id/product_name")
G15_PRODUCT_RE = re.compile(r"(?:\bzephyrus\s+g15\b|\bga503[a-z0-9_-]*\b)", re.IGNORECASE)
REQUIRED_TOOLS = ("python", "node", "npm", "git", "make")
ALLOWED_PACKAGE_ENV = frozenset(
    {
        "NPM_CONFIG_CAFILE",
        "NPM_CONFIG_HTTPS_PROXY",
        "NPM_CONFIG_PROXY",
        "NPM_CONFIG_REGISTRY",
        "PIP_CERT",
        "PIP_CLIENT_CERT",
        "PIP_EXTRA_INDEX_URL",
        "PIP_INDEX_URL",
        "PIP_PROXY",
        "PIP_RETRIES",
        "PIP_TIMEOUT",
        "PIP_TRUSTED_HOST",
    }
)
FRESH_PATHS = (
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
    "coverage.json",
    ".coverage",
    "frontend/node_modules",
    "frontend/dist",
    "frontend/.vite",
    "frontend/coverage",
    "frontend/playwright-report",
    "frontend/test-results",
)
FRESH_GLOBS = ("frontend/*.tsbuildinfo",)
ALLOWED_BASE_ENV = frozenset(
    {
        "ALL_PROXY",
        "COLORTERM",
        "CURL_CA_BUNDLE",
        "FTP_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "NODE_AUTH_TOKEN",
        "NODE_EXTRA_CA_CERTS",
        "NO_COLOR",
        "NO_PROXY",
        "NPM_TOKEN",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TZ",
    }
)
SECRET_ENV_PARTS = (
    "AUTH",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "API_KEY",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "COOKIE",
    "NETRC",
)
REDACTION = b"[REDACTED]"
SIGNAL_GRACE_SECONDS = 10.0
PUBLIC_PACKAGE_PATHS = frozenset(
    {"api", "cpu", "index", "packages", "pypi", "repository", "simple", "v1", "v2", "v3", "whl"}
)


class GateError(RuntimeError):
    """A refusal or evidence error safe to report without a traceback."""


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class Probe(Protocol):
    def __call__(
        self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]
    ) -> ProbeResult: ...


@dataclasses.dataclass(frozen=True)
class HostFacts:
    system: str
    architecture: str
    kernel_release: str
    python_version: str
    python_implementation: str
    python_major: int
    python_minor: int
    python_isolated: bool
    os_release: Mapping[str, str]
    effective_uid: int
    system_vendor: str
    product_name: str


@dataclasses.dataclass(frozen=True)
class ToolInfo:
    name: str
    invocation_path: str
    resolved_path: str
    sha256: str
    mode: str
    owner_uid: int

    def public(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CommandResult:
    command_id: str
    argv: tuple[str, ...]
    log_path: str
    started_at: str
    finished_at: str
    duration_ns: int
    exit_code: int | None
    spawn_error: str | None = None
    interrupted_by: str | None = None
    redactions_applied: int = 0

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and self.spawn_error is None and self.interrupted_by is None

    def public(self, log_file: Path) -> dict[str, Any]:
        metadata = _file_metadata(log_file, log_file.parent)
        return {
            "id": self.command_id,
            "argv": list(self.argv),
            "cwd": "$REPO",
            "invocation_count": 1,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ns": self.duration_ns,
            "exit_code": self.exit_code,
            "interrupted_by": self.interrupted_by,
            "spawn_error": self.spawn_error,
            "status": "passed" if self.passed else "failed",
            "stdout_stderr": {
                **metadata,
                "stderr_merged": True,
                "redactions_applied": self.redactions_applied,
            },
        }


class Executor(Protocol):
    def __call__(
        self,
        command_id: str,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        relay: SignalRelay,
        redact_patterns: Sequence[bytes],
        scratch_dir: Path,
        now: Callable[[], dt.datetime],
        monotonic_ns: Callable[[], int],
    ) -> CommandResult: ...


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _iso(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_root_owned_path(path: Path) -> None:
    """Require a root-owned, non-writable executable and every resolved parent."""

    current = path
    while True:
        metadata = current.stat()
        if metadata.st_uid != 0:
            raise GateError(f"required tool path is not root-owned: {path}")
        if stat.S_IMODE(metadata.st_mode) & 0o022:
            raise GateError(f"required tool path is group/other-writable: {path}")
        if current == current.parent:
            return
        current = current.parent


def _resolve_system_tool(name: str, *, search_path: str = TRUSTED_PATH) -> ToolInfo:
    requested_name = sys.executable if name == "python" else name
    if name == "python":
        invocation = Path(requested_name).absolute()
    else:
        found = shutil.which(requested_name, path=search_path)
        if found is None:
            raise GateError(f"required system tool is missing from the trusted PATH: {name}")
        invocation = Path(found).absolute()
    try:
        resolved = invocation.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise GateError(f"cannot resolve required system tool {name}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise GateError(f"required system tool is not an executable regular file: {name}")
    _assert_root_owned_path(resolved)
    _assert_root_owned_path(invocation.parent.resolve(strict=True))
    invocation_metadata = invocation.lstat()
    if invocation_metadata.st_uid != 0:
        raise GateError(f"required system tool entry is not root-owned: {invocation}")
    return ToolInfo(
        name=name,
        invocation_path=str(invocation),
        resolved_path=str(resolved),
        sha256=_sha256(resolved),
        mode=f"{stat.S_IMODE(metadata.st_mode):04o}",
        owner_uid=metadata.st_uid,
    )


def resolve_toolchain() -> dict[str, ToolInfo]:
    return {name: _resolve_system_tool(name) for name in REQUIRED_TOOLS}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    """Write one private evidence file, fsync it, and atomically publish it."""

    if path.parent.is_symlink():
        raise GateError(f"evidence parent may not be a symlink: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, EVIDENCE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(EVIDENCE_FILE_MODE)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _write_atomic_json(path: Path, payload: Any) -> bytes:
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    _write_atomic_bytes(path, serialized)
    return serialized


def _prepare_evidence(requested: Path, repo_root: Path) -> Path:
    expanded = requested.expanduser().absolute()
    if expanded.is_symlink():
        raise GateError(f"evidence directory may not be a symlink: {expanded}")
    resolved = expanded.resolve(strict=False)
    repository = repo_root.resolve()
    try:
        resolved.relative_to(repository)
    except ValueError:
        pass
    else:
        raise GateError("evidence directory must be outside the source checkout")

    if resolved.exists():
        metadata = resolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise GateError(f"evidence path is not an ordinary directory: {resolved}")
        if stat.S_IMODE(metadata.st_mode) != EVIDENCE_DIR_MODE:
            raise GateError(
                f"existing evidence directory must have mode 0700, not "
                f"{stat.S_IMODE(metadata.st_mode):04o}: {resolved}"
            )
        if any(resolved.iterdir()):
            raise GateError(
                f"evidence directory is not empty: {resolved}; use a new empty directory"
            )
    else:
        resolved.mkdir(parents=True, mode=EVIDENCE_DIR_MODE)
    metadata = resolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GateError(f"evidence path is not an ordinary directory: {resolved}")
    if stat.S_IMODE(metadata.st_mode) != EVIDENCE_DIR_MODE:
        raise GateError(f"evidence directory is not mode 0700: {resolved}")
    return resolved


def _validate_system_temp(path: Path = SYSTEM_TEMP_ROOT) -> None:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise GateError(f"fixed system temporary root is not an ordinary directory: {path}")
    if metadata.st_uid != 0 or not metadata.st_mode & stat.S_ISVTX:
        raise GateError(f"fixed system temporary root is not root-owned and sticky: {path}")


def _parse_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except OSError as exc:
        raise GateError(f"cannot read {path}: {exc.strerror or exc}") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        try:
            parsed = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise GateError(f"cannot parse {path} key {key!r}") from exc
        values[key] = parsed[0] if len(parsed) == 1 else raw_value
    return values


def _read_dmi_fact(path: Path) -> str:
    """Read one kernel-provided machine identity field without shelling out."""

    try:
        value = path.read_text(encoding="utf-8", errors="strict").strip()
    except (OSError, UnicodeError) as exc:
        raise GateError(f"cannot read machine identity {path}: {exc}") from exc
    if not value or any(ord(character) < 0x20 for character in value):
        raise GateError(f"machine identity {path} is empty or malformed")
    return value


def inspect_host() -> HostFacts:
    version = sys.version_info
    get_effective_uid = getattr(os, "geteuid", lambda: -1)
    return HostFacts(
        system=platform.system(),
        architecture=platform.machine(),
        kernel_release=platform.release(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        python_major=version.major,
        python_minor=version.minor,
        python_isolated=bool(sys.flags.isolated),
        os_release=_parse_os_release(),
        effective_uid=get_effective_uid(),
        system_vendor=_read_dmi_fact(DMI_SYS_VENDOR),
        product_name=_read_dmi_fact(DMI_PRODUCT_NAME),
    )


def _default_probe(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> ProbeResult:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeResult(127, b"", f"{type(exc).__name__}: {exc}".encode("utf-8", "replace"))
    return ProbeResult(completed.returncode, completed.stdout, completed.stderr)


def _decode(value: bytes) -> str:
    return value.decode("utf-8", "replace").strip()


def _first_line(value: bytes) -> str:
    text = _decode(value)
    return text.splitlines()[0] if text else ""


def _validate_expected_commit(value: str) -> str:
    stripped = value.strip()
    if not COMMIT_RE.fullmatch(stripped):
        raise GateError("--expected-commit must be a full 40-character lowercase Git commit")
    return stripped


def _forbidden_so101_keys(environ: Mapping[str, str]) -> list[str]:
    return sorted(key for key in environ if key.upper().startswith("SO101_"))


def _allowed_base_environment_key(key: str) -> bool:
    return key.upper() in ALLOWED_BASE_ENV or key.upper().startswith("LC_")


def _child_environment(
    source: Mapping[str, str], isolation_root: Path, *, python_path: str
) -> tuple[dict[str, str], list[str], list[str]]:
    preserved_base_env = {
        key: value for key, value in source.items() if _allowed_base_environment_key(key)
    }
    preserved_package_env = {
        key.upper(): value for key, value in source.items() if key.upper() in ALLOWED_PACKAGE_ENV
    }
    preserved_names = set(preserved_base_env) | {
        key for key in source if key.upper() in ALLOWED_PACKAGE_ENV
    }
    scrubbed = sorted(set(source) - preserved_names)
    env = dict(preserved_base_env)
    home = isolation_root / "home"
    cache = isolation_root / "cache"
    config = isolation_root / "config"
    data = isolation_root / "xdg-data"
    state = isolation_root / "state"
    runtime = isolation_root / "runtime"
    temporary = isolation_root / "tmp"
    raw_logs = isolation_root / "raw-logs"
    for directory in (home, cache, config, data, state, runtime, temporary, raw_logs):
        directory.mkdir(mode=0o700)
    npm_userconfig = config / "npmrc"
    _write_atomic_bytes(npm_userconfig, b"")
    no_hardware = isolation_root / "NO-HARDWARE-PORTS"
    env.update(preserved_package_env)
    env.update(
        {
            "CI": "true",
            "HOME": str(home),
            "NPM_CONFIG_CACHE": str(cache / "npm"),
            "NPM_CONFIG_SCRIPT_SHELL": "/bin/sh",
            "NPM_CONFIG_USERCONFIG": str(npm_userconfig),
            "PATH": TRUSTED_PATH,
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHON": python_path,
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONUNBUFFERED": "1",
            "SHELL": "/bin/sh",
            "SO101_BACKEND": "mock",
            "SO101_CAMERAS": "",
            "SO101_FOLLOWER_PORT": str(no_hardware / "follower-MUST-NOT-OPEN"),
            "SO101_HOME": str(isolation_root / "so101-home"),
            "SO101_LEADER_PORT": str(no_hardware / "leader-MUST-NOT-OPEN"),
            "SO101_MOCK": "1",
            "SO101_OPENAI_API_KEY": "",
            "SO101_SIM_CAMERAS": "",
            "SO101_WANDB_ENABLED": "false",
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "TMPDIR": str(temporary),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "XDG_DATA_HOME": str(data),
            "XDG_RUNTIME_DIR": str(runtime),
            "XDG_STATE_HOME": str(state),
        }
    )
    return env, scrubbed, sorted(preserved_package_env)


def _secret_variants(value: str | None) -> set[str]:
    if not value:
        return set()
    decoded = {value}
    frontier = {value}
    for _ in range(3):
        discovered = {
            candidate
            for item in frontier
            for candidate in (urllib.parse.unquote(item), urllib.parse.unquote_plus(item))
        } - decoded
        if not discovered:
            break
        decoded.update(discovered)
        frontier = discovered
    variants: set[str] = set()
    for item in decoded:
        if len(item) < 3:
            continue
        quoted = urllib.parse.quote(item, safe="")
        quoted_plus = urllib.parse.quote_plus(item, safe="")
        variants.update(
            {
                item,
                item.lower(),
                item.upper(),
                quoted,
                quoted.lower(),
                quoted.upper(),
                quoted_plus,
                quoted_plus.lower(),
                quoted_plus.upper(),
            }
        )
    return variants


def _sensitive_url_parts(value: str) -> set[str]:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return set()
    if not parsed.scheme:
        return set()
    parts = _secret_variants(value)
    for part in (parsed.username, parsed.password):
        if part:
            parts.update(_secret_variants(part))
    for _name, query_value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        if query_value:
            parts.update(_secret_variants(query_value))
    if parsed.fragment:
        parts.update(_secret_variants(parsed.fragment))
    for raw_segment in parsed.path.split("/"):
        segment = urllib.parse.unquote(raw_segment)
        if raw_segment and len(segment) >= 3 and segment.lower() not in PUBLIC_PACKAGE_PATHS:
            parts.update(_secret_variants(raw_segment))
    return parts


def _secret_patterns(environ: Mapping[str, str]) -> tuple[bytes, ...]:
    values: set[str] = set()
    for key, value in environ.items():
        upper = key.upper()
        if any(part in upper for part in SECRET_ENV_PARTS):
            values.update(_secret_variants(value))
        if upper in ALLOWED_PACKAGE_ENV:
            # Pip's list-valued settings are whitespace separated. Downloaders
            # may echo one URL at a time, decoded, re-encoded, or case-folded.
            values.update(_secret_variants(value))
            for candidate in value.split():
                values.update(_secret_variants(candidate))
                values.update(_sensitive_url_parts(candidate))
        else:
            for candidate in value.split():
                values.update(_sensitive_url_parts(candidate))
    encoded = {
        value.encode("utf-8")
        for value in values
        if len(value) >= 3 and "\n" not in value and "\r" not in value
    }
    return tuple(sorted(encoded, key=lambda item: (-len(item), item)))


def _sanitize_text(text: str, patterns: Sequence[bytes]) -> str:
    data = text.encode("utf-8", "replace")
    for pattern in patterns:
        data = data.replace(pattern, REDACTION)
    return data.decode("utf-8", "replace")


def _redact_stream(incoming: BinaryIO, outgoing: BinaryIO, patterns: Sequence[bytes]) -> int:
    if not patterns:
        shutil.copyfileobj(incoming, outgoing)
        return 0
    keep = max(len(pattern) for pattern in patterns) - 1
    buffer = b""
    redactions = 0

    def flush_safe(*, final: bool) -> None:
        nonlocal buffer, redactions
        while buffer:
            safe_limit = len(buffer) if final else len(buffer) - keep
            if safe_limit <= 0:
                return
            found: tuple[int, bytes] | None = None
            for pattern in patterns:
                index = buffer.find(pattern)
                if index < 0:
                    continue
                if (
                    found is None
                    or index < found[0]
                    or (index == found[0] and len(pattern) > len(found[1]))
                ):
                    found = (index, pattern)
            if found is None or found[0] >= safe_limit:
                outgoing.write(buffer[:safe_limit])
                buffer = buffer[safe_limit:]
                return
            index, pattern = found
            outgoing.write(buffer[:index])
            outgoing.write(REDACTION)
            buffer = buffer[index + len(pattern) :]
            redactions += 1

    for chunk in iter(lambda: incoming.read(64 * 1024), b""):
        buffer += chunk
        flush_safe(final=False)
    flush_safe(final=True)
    return redactions


def _redact_file(path: Path, patterns: Sequence[bytes]) -> int:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.redact.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, EVIDENCE_FILE_MODE)
        with path.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            count = _redact_stream(incoming, outgoing, patterns)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, path)
        path.chmod(EVIDENCE_FILE_MODE)
        _fsync_directory(path.parent)
        return count
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _publish_redacted_file(source: Path, destination: Path, patterns: Sequence[bytes]) -> int:
    """Atomically publish a redacted log; raw command output stays outside evidence."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.publish.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, EVIDENCE_FILE_MODE)
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            count = _redact_stream(incoming, outgoing, patterns)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, destination)
        destination.chmod(EVIDENCE_FILE_MODE)
        _fsync_directory(destination.parent)
        return count
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


class SignalRelay:
    """Remember the first terminal signal and forward it to the active process group."""

    def __init__(self, *, install_handlers: bool = True) -> None:
        self.install_handlers = install_handlers
        self.first_signal: int | None = None
        self.signal_at_ns: int | None = None
        self.active_pid: int | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> SignalRelay:
        if self.install_handlers:
            for signum in (signal.SIGINT, signal.SIGTERM):
                try:
                    self._previous[signum] = signal.signal(signum, self._handle)
                except ValueError:
                    self._previous.clear()
                    break
        return self

    def __exit__(self, *_: object) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()

    @property
    def name(self) -> str | None:
        return signal.Signals(self.first_signal).name if self.first_signal is not None else None

    def record(self, signum: int, *, forward: bool = True) -> None:
        if self.first_signal is None:
            self.first_signal = signum
            self.signal_at_ns = time.monotonic_ns()
        pid = self.active_pid
        if not forward or pid is None:
            return
        forwarded = signum if signum == self.first_signal else signal.SIGKILL
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, forwarded)

    def _handle(self, signum: int, _frame: object) -> None:
        self.record(signum)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=SIGNAL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=SIGNAL_GRACE_SECONDS)


def execute_command(
    command_id: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    relay: SignalRelay,
    redact_patterns: Sequence[bytes] = (),
    scratch_dir: Path = SYSTEM_TEMP_ROOT,
    now: Callable[[], dt.datetime] = _utc_now,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> CommandResult:
    """Run one fixed command once, with merged output in a private full log."""

    started_at = _iso(now())
    started_ns = monotonic_ns()
    exit_code: int | None = None
    spawn_error: str | None = None
    process: subprocess.Popen[bytes] | None = None
    scratch_metadata = scratch_dir.lstat()
    if stat.S_ISLNK(scratch_metadata.st_mode) or not stat.S_ISDIR(scratch_metadata.st_mode):
        raise GateError(f"command scratch path is not an ordinary directory: {scratch_dir}")
    raw_descriptor, raw_name = tempfile.mkstemp(
        prefix=f"so101-g15-ci-{command_id}-", dir=scratch_dir
    )
    raw_path = Path(raw_name)
    os.fchmod(raw_descriptor, EVIDENCE_FILE_MODE)
    stream = os.fdopen(raw_descriptor, "wb")
    redactions_applied = 0
    try:
        print(f"g15-ci: starting {' '.join(argv)}; sealed output -> {log_path}", flush=True)
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=None,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            relay.active_pid = process.pid
            if relay.first_signal is not None:
                relay.record(relay.first_signal)
            next_notice_ns = monotonic_ns() + 60_000_000_000
            while True:
                try:
                    exit_code = process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    current_ns = monotonic_ns()
                    if current_ns >= next_notice_ns:
                        elapsed = (current_ns - started_ns) / 1_000_000_000
                        print(
                            f"g15-ci: {command_id} still running after {elapsed:.0f}s; "
                            f"output -> {log_path}",
                            flush=True,
                        )
                        next_notice_ns = current_ns + 60_000_000_000
                    if (
                        relay.first_signal is not None
                        and relay.signal_at_ns is not None
                        and current_ns - relay.signal_at_ns
                        >= int(SIGNAL_GRACE_SECONDS * 1_000_000_000)
                    ):
                        with contextlib.suppress(ProcessLookupError, PermissionError):
                            os.killpg(process.pid, signal.SIGKILL)
        except OSError as exc:
            spawn_error = f"{type(exc).__name__}: {exc}"
            stream.write((spawn_error + "\n").encode("utf-8", "replace"))
        except BaseException:
            if process is not None:
                _terminate_process_group(process)
            raise
    finally:
        relay.active_pid = None
        stream_error: BaseException | None = None
        try:
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException as exc:
            stream_error = exc
        finally:
            stream.close()
        try:
            redactions_applied = _publish_redacted_file(raw_path, log_path, redact_patterns)
        finally:
            raw_path.unlink(missing_ok=True)
        if stream_error is not None:
            raise stream_error
    finished_ns = monotonic_ns()
    return CommandResult(
        command_id=command_id,
        argv=tuple(argv),
        log_path=log_path.name,
        started_at=started_at,
        finished_at=_iso(now()),
        duration_ns=max(0, finished_ns - started_ns),
        exit_code=exit_code,
        spawn_error=spawn_error,
        interrupted_by=relay.name,
        redactions_applied=redactions_applied,
    )


def _file_metadata(path: Path, root: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GateError(f"evidence contains a non-regular file: {path}")
    if stat.S_IMODE(metadata.st_mode) != EVIDENCE_FILE_MODE:
        raise GateError(f"evidence file is not mode 0600: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": metadata.st_size,
        "mode": "0600",
        "sha256": _sha256(path),
    }


def _existing_evidence_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise GateError(f"evidence contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != EVIDENCE_DIR_MODE:
                raise GateError(f"evidence directory is not mode 0700: {path}")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise GateError(f"evidence contains a special file: {path}")
        if path.name.startswith("."):
            raise GateError(f"evidence contains an unpublished temporary file: {path}")
        files.append(path)
    return files


def _finalize_result(root: Path, result: dict[str, Any]) -> None:
    """Write a checksum manifest, then atomically publish result.json last.

    SHA256SUMS conventionally excludes itself.  It covers every other regular
    evidence file, including the exact result bytes computed before publication.
    """

    if (root / RESULT_FILE).exists() or (root / HASH_MANIFEST).exists():
        raise GateError("result or checksum manifest already exists")
    payload_files = _existing_evidence_files(root)
    entries = [_file_metadata(path, root) for path in payload_files]
    result["evidence"] = {
        "algorithm": "sha256",
        "files_before_result": entries,
        "hash_manifest": HASH_MANIFEST,
        "manifest_excludes_itself": True,
        "result_is_in_manifest": True,
    }
    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    checksums = {entry["path"]: entry["sha256"] for entry in entries}
    checksums[RESULT_FILE] = hashlib.sha256(result_bytes).hexdigest()
    manifest = "".join(f"{checksums[name]}  {name}\n" for name in sorted(checksums))
    _write_atomic_bytes(root / HASH_MANIFEST, manifest.encode("utf-8"))
    _write_atomic_bytes(root / RESULT_FILE, result_bytes)


def _failure(
    stage: str,
    kind: str,
    message: str,
    *,
    command: CommandResult | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"stage": stage, "kind": kind, "message": message}
    if command is not None:
        payload.update(
            {
                "command_id": command.command_id,
                "exit_code": command.exit_code,
                "log": command.log_path,
            }
        )
    return payload


def _version_fact(result: ProbeResult, patterns: Sequence[bytes]) -> dict[str, Any]:
    output = _first_line(result.stdout if result.stdout else result.stderr)
    return {
        "available": result.returncode == 0,
        "exit_code": result.returncode,
        "version": _sanitize_text(output, patterns) if output else None,
    }


def _has_local_env_files(repo_root: Path) -> bool:
    candidates = list(repo_root.glob(".env*")) + list((repo_root / "frontend").glob(".env*"))
    return any(path.name != ".env.example" for path in candidates)


def _fresh_checkout_blockers(repo_root: Path) -> list[str]:
    blockers = [relative for relative in FRESH_PATHS if (repo_root / relative).exists()]
    for pattern in FRESH_GLOBS:
        blockers.extend(path.relative_to(repo_root).as_posix() for path in repo_root.glob(pattern))
    if _has_local_env_files(repo_root):
        blockers.append("local .env file(s)")
    return sorted(blockers)


def _status_entry_count(status: bytes) -> int:
    return sum(1 for item in status.split(b"\0") if item)


def _command_failure(record: CommandResult) -> tuple[str, str]:
    if record.interrupted_by:
        return "signal", f"{record.command_id} interrupted by {record.interrupted_by}"
    if record.spawn_error:
        return "spawn_error", f"could not start {' '.join(record.argv)}"
    return "command_exit", f"{' '.join(record.argv)} exited {record.exit_code}"


def _command_return_code(record: CommandResult) -> int:
    if record.interrupted_by:
        try:
            return 128 + signal.Signals[record.interrupted_by].value
        except (KeyError, ValueError):
            return 1
    if record.exit_code is not None and 0 < record.exit_code <= 255:
        return record.exit_code
    return 1


def run_gate(
    expected_commit: str,
    evidence_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    environ: Mapping[str, str] | None = None,
    host_facts: HostFacts | None = None,
    probe: Probe = _default_probe,
    executor: Executor = execute_command,
    tool_resolver: Callable[[], Mapping[str, ToolInfo]] = resolve_toolchain,
    now: Callable[[], dt.datetime] = _utc_now,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    relay: SignalRelay | None = None,
) -> int:
    """Run the gate.  Injection points keep every policy path deterministic in tests."""

    expected = _validate_expected_commit(expected_commit)
    root = repo_root.resolve()
    evidence = _prepare_evidence(evidence_dir, root)
    source_env = dict(os.environ if environ is None else environ)
    patterns = _secret_patterns(source_env)
    started_at = _iso(now())
    run_started_ns = monotonic_ns()
    host = host_facts or inspect_host()
    actual_commit: str | None = None
    clean_at_start: bool | None = None
    clean_at_end: bool | None = None
    source_unchanged: bool | None = None
    commands: list[dict[str, Any]] = []
    primary_failure: dict[str, Any] | None = None
    status = "running"
    return_code = 1
    tools: dict[str, Any] = {
        "python": {
            "available": True,
            "implementation": host.python_implementation,
            "isolated": host.python_isolated,
            "version": host.python_version,
        }
    }
    toolchain: dict[str, ToolInfo] = {}
    policy: dict[str, Any] = {
        "hardware_access_allowed": False,
        "doctor_allowed": False,
        "hil_allowed": False,
        "commands_are_fixed": True,
        "retries": 0,
        "so101_environment_variables_present": [],
        "child_backend": "mock",
        "child_cameras": "disabled",
        "child_serial_ports": "nonexistent sentinels",
        "scrubbed_environment_keys": [],
        "preserved_package_environment_keys": [],
    }
    active_relay = relay or SignalRelay()

    def refuse(kind: str, message: str) -> None:
        nonlocal primary_failure, return_code, status
        if primary_failure is None:
            primary_failure = _failure("prerequisites", kind, _sanitize_text(message, patterns))
            status = "refused"
            return_code = 2

    try:
        forbidden = _forbidden_so101_keys(source_env)
        policy["so101_environment_variables_present"] = forbidden
        if forbidden:
            joined = ", ".join(forbidden[:10])
            suffix = "" if len(forbidden) <= 10 else f" (+{len(forbidden) - 10} more)"
            refuse(
                "so101_environment",
                f"refusing inherited SO101_* settings (names only): {joined}{suffix}",
            )

        if primary_failure is None and host.system != "Linux":
            refuse("platform", f"requires Linux, found {host.system or 'unknown'}")
        if primary_failure is None and host.architecture.lower() not in {"x86_64", "amd64"}:
            refuse("architecture", f"requires x86_64, found {host.architecture or 'unknown'}")
        if primary_failure is None and not host.system_vendor.strip().casefold().startswith("asus"):
            refuse(
                "machine_vendor",
                f"requires an ASUS Zephyrus G15, found vendor {host.system_vendor!r}",
            )
        if primary_failure is None and G15_PRODUCT_RE.search(host.product_name) is None:
            refuse(
                "machine_model",
                f"requires an ASUS Zephyrus G15 (GA503), found product {host.product_name!r}",
            )
        if primary_failure is None and (
            host.os_release.get("ID", "").lower() != "ubuntu"
            or host.os_release.get("VERSION_ID") != "24.04"
        ):
            refuse(
                "operating_system",
                "requires Ubuntu 24.04 exactly; "
                f"found ID={host.os_release.get('ID')!r} "
                f"VERSION_ID={host.os_release.get('VERSION_ID')!r}",
            )
        if primary_failure is None and (host.python_major, host.python_minor) != (3, 12):
            refuse("python", f"requires Python 3.12, found {host.python_version}")
        if primary_failure is None and not host.python_isolated:
            refuse("python_isolation", "runner must start with Python isolated mode (-I)")
        if primary_failure is None and host.effective_uid == 0:
            refuse("root_user", "run as the normal G15 user, not root or sudo")

        if primary_failure is None:
            try:
                toolchain = dict(tool_resolver())
                missing_tools = sorted(set(REQUIRED_TOOLS) - set(toolchain))
                if missing_tools:
                    raise GateError("tool resolver omitted: " + ", ".join(missing_tools))
                tools["python"].update(toolchain["python"].public())
                _validate_system_temp()
            except (GateError, OSError) as exc:
                refuse("toolchain", str(exc))

        isolation_parent = SYSTEM_TEMP_ROOT if SYSTEM_TEMP_ROOT.is_dir() else None
        with tempfile.TemporaryDirectory(
            prefix="so101-g15-ci-", dir=isolation_parent
        ) as isolation_name:
            isolation = Path(isolation_name)
            python_path = (
                toolchain["python"].invocation_path if "python" in toolchain else sys.executable
            )
            child_env, scrubbed, preserved_package_env = _child_environment(
                source_env, isolation, python_path=python_path
            )
            policy["scrubbed_environment_keys"] = scrubbed
            policy["preserved_package_environment_keys"] = preserved_package_env

            if primary_failure is None:
                node = probe(
                    (toolchain["node"].invocation_path, "--version"),
                    cwd=root,
                    env=child_env,
                )
                tools["node"] = {
                    **toolchain["node"].public(),
                    **_version_fact(node, patterns),
                }
                node_text = _first_line(node.stdout)
                match = NODE_VERSION_RE.fullmatch(node_text)
                if node.returncode != 0 or match is None or int(match.group("major")) != 24:
                    found = _sanitize_text(
                        node_text or _first_line(node.stderr) or "unavailable", patterns
                    )
                    refuse("node", f"requires Node 24, found {found}")

            if primary_failure is None:
                version_probes = (
                    ("npm", (toolchain["npm"].invocation_path, "--version")),
                    ("git", (toolchain["git"].invocation_path, "--version")),
                    ("make", (toolchain["make"].invocation_path, "--version")),
                )
                for name, argv in version_probes:
                    version_result = probe(argv, cwd=root, env=child_env)
                    tools[name] = {
                        **toolchain[name].public(),
                        **_version_fact(version_result, patterns),
                    }
                npm_version = str(tools["npm"].get("version") or "")
                git_version = str(tools["git"].get("version") or "")
                make_version = str(tools["make"].get("version") or "")
                if not NODE_VERSION_RE.fullmatch(npm_version):
                    refuse("npm", f"requires npm with a semantic version, found {npm_version!r}")
                elif not git_version.startswith("git version "):
                    refuse("git", f"requires Git, found {git_version!r}")
                elif not make_version.startswith("GNU Make "):
                    refuse("make", f"requires GNU Make, found {make_version!r}")

            if primary_failure is None:
                commit_probe = probe(
                    (
                        toolchain["git"].invocation_path,
                        "-C",
                        str(root),
                        "rev-parse",
                        "--verify",
                        "HEAD^{commit}",
                    ),
                    cwd=root,
                    env=child_env,
                )
                candidate = _first_line(commit_probe.stdout).lower()
                if commit_probe.returncode != 0 or not COMMIT_RE.fullmatch(candidate):
                    refuse("git", "could not resolve the checkout's HEAD commit")
                else:
                    actual_commit = candidate
                    if actual_commit != expected:
                        refuse(
                            "commit",
                            f"expected commit {expected}, found {actual_commit}",
                        )

            if primary_failure is None:
                status_probe = probe(
                    (
                        toolchain["git"].invocation_path,
                        "-C",
                        str(root),
                        "status",
                        "--porcelain=v1",
                        "-z",
                        "--untracked-files=all",
                        "--ignore-submodules=none",
                    ),
                    cwd=root,
                    env=child_env,
                )
                clean_at_start = status_probe.returncode == 0 and not status_probe.stdout
                if status_probe.returncode != 0:
                    refuse("git_status", "could not inspect the checkout for changes")
                elif not clean_at_start:
                    refuse(
                        "dirty_checkout",
                        f"checkout has {_status_entry_count(status_probe.stdout)} changed or "
                        "untracked "
                        "entries; paths are intentionally omitted",
                    )

            if primary_failure is None:
                blockers = _fresh_checkout_blockers(root)
                if blockers:
                    refuse(
                        "reused_checkout",
                        "checkout contains ignored bootstrap/CI state or local environment files: "
                        + ", ".join(blockers),
                    )

            environment_payload = {
                "schema": ENVIRONMENT_SCHEMA,
                "captured_at": _iso(now()),
                "host": {
                    "system": host.system,
                    "architecture": host.architecture,
                    "kernel_release": host.kernel_release,
                    "system_vendor": host.system_vendor,
                    "product_name": host.product_name,
                    "os_release": {
                        key.lower(): host.os_release.get(key)
                        for key in ("ID", "VERSION_ID", "VERSION_CODENAME")
                    },
                    "root_user": host.effective_uid == 0,
                },
                "tools": tools,
                "source": {
                    "expected_commit": expected,
                    "actual_commit": actual_commit,
                    "clean_at_start": clean_at_start,
                },
                "policy": policy,
            }
            _write_atomic_json(evidence / "environment.json", environment_payload)

            with active_relay:
                if primary_failure is None and active_relay.first_signal is not None:
                    primary_failure = _failure(
                        "prerequisites",
                        "signal",
                        f"interrupted by {active_relay.name} before bootstrap",
                    )
                    status = "interrupted"
                    return_code = 128 + active_relay.first_signal

                commands_to_run = (
                    tuple(
                        (command_id, (toolchain["make"].invocation_path, argv[1]), log_name)
                        for command_id, argv, log_name in COMMANDS
                    )
                    if toolchain
                    else ()
                )
                for command_id, argv, log_name in commands_to_run:
                    if primary_failure is not None:
                        break
                    if active_relay.first_signal is not None:
                        primary_failure = _failure(
                            command_id,
                            "signal",
                            f"interrupted by {active_relay.name} before {command_id}",
                        )
                        status = "interrupted"
                        return_code = 128 + active_relay.first_signal
                        break
                    log_path = evidence / log_name
                    try:
                        record = executor(
                            command_id,
                            argv,
                            cwd=root,
                            env=child_env,
                            log_path=log_path,
                            relay=active_relay,
                            redact_patterns=patterns,
                            scratch_dir=isolation / "raw-logs",
                            now=now,
                            monotonic_ns=monotonic_ns,
                        )
                    except (Exception, KeyboardInterrupt) as exc:
                        if not log_path.exists():
                            _write_atomic_bytes(
                                log_path,
                                (
                                    _sanitize_text(f"{type(exc).__name__}: {exc}", patterns) + "\n"
                                ).encode(),
                            )
                        record = CommandResult(
                            command_id=command_id,
                            argv=tuple(argv),
                            log_path=log_name,
                            started_at=_iso(now()),
                            finished_at=_iso(now()),
                            duration_ns=0,
                            exit_code=None,
                            spawn_error=_sanitize_text(f"{type(exc).__name__}: {exc}", patterns),
                            interrupted_by=active_relay.name,
                        )
                    if record.interrupted_by is None and active_relay.first_signal is not None:
                        record.interrupted_by = active_relay.name
                    record.redactions_applied += _redact_file(log_path, patterns)
                    commands.append(record.public(log_path))
                    if not record.passed:
                        kind, message = _command_failure(record)
                        primary_failure = _failure(
                            command_id,
                            kind,
                            _sanitize_text(message, patterns),
                            command=record,
                        )
                        if record.interrupted_by:
                            status = "interrupted"
                        else:
                            status = "failed"
                        return_code = _command_return_code(record)

                if primary_failure is None and active_relay.first_signal is not None:
                    primary_failure = _failure(
                        "postflight",
                        "signal",
                        f"interrupted by {active_relay.name} before source postflight",
                    )
                    status = "interrupted"
                    return_code = 128 + active_relay.first_signal

                if primary_failure is None:
                    final_commit_probe = probe(
                        (
                            toolchain["git"].invocation_path,
                            "-C",
                            str(root),
                            "rev-parse",
                            "--verify",
                            "HEAD^{commit}",
                        ),
                        cwd=root,
                        env=child_env,
                    )
                    final_commit = _first_line(final_commit_probe.stdout).lower()
                    final_status_probe = probe(
                        (
                            toolchain["git"].invocation_path,
                            "-C",
                            str(root),
                            "status",
                            "--porcelain=v1",
                            "-z",
                            "--untracked-files=all",
                            "--ignore-submodules=none",
                        ),
                        cwd=root,
                        env=child_env,
                    )
                    clean_at_end = (
                        final_status_probe.returncode == 0 and not final_status_probe.stdout
                    )
                    source_unchanged = (
                        final_commit_probe.returncode == 0
                        and final_commit == expected
                        and clean_at_end
                    )
                    if active_relay.first_signal is not None:
                        primary_failure = _failure(
                            "postflight",
                            "signal",
                            f"interrupted by {active_relay.name} during source postflight",
                        )
                        status = "interrupted"
                        return_code = 128 + active_relay.first_signal
                    elif not source_unchanged:
                        primary_failure = _failure(
                            "postflight",
                            "source_changed",
                            "bootstrap/CI did not leave the checkout clean at the expected commit",
                        )
                        status = "failed"
                        return_code = 1
                    else:
                        status = "passed"
                        return_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        if primary_failure is None:
            primary_failure = _failure(
                "runner",
                "runner_error",
                _sanitize_text(f"{type(exc).__name__}: {exc}", patterns),
            )
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        return_code = 130 if isinstance(exc, KeyboardInterrupt) else 1

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": status,
        "passed": status == "passed",
        "release_ready": False,
        "hardware_touched": False,
        "hardware_validation_required": True,
        "started_at": started_at,
        "finished_at": _iso(now()),
        "duration_ns": max(0, monotonic_ns() - run_started_ns),
        "source": {
            "expected_commit": expected,
            "actual_commit": actual_commit,
            "clean_at_start": clean_at_start,
            "clean_at_end": clean_at_end,
            "unchanged": source_unchanged,
        },
        "policy": policy,
        "commands": commands,
        "failure": primary_failure,
        "next_gate": (
            "Build and install a verified handoff; doctor and HIL remain mandatory."
            if status == "passed"
            else None
        ),
    }
    try:
        _finalize_result(evidence, result)
    except Exception as exc:
        print(f"g15-ci: FAIL: could not finalize evidence: {exc}", file=sys.stderr)
        return 1

    if status == "passed":
        print(
            f"g15-ci: software CI passed; evidence: {evidence}\n"
            "g15-ci: release_ready=false; do not connect hardware on this evidence alone."
        )
    else:
        message = primary_failure["message"] if primary_failure else "unknown failure"
        print(f"g15-ci: {status.upper()}: {message}; evidence: {evidence}", file=sys.stderr)
    return return_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="full 40-character commit supplied by the candidate manifest (never infer it)",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="new or empty mode-0700 directory outside the checkout",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_gate(args.expected_commit, args.evidence_dir)
    except GateError as exc:
        print(f"g15-ci: REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
