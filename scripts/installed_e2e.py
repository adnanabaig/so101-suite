#!/usr/bin/env python3
"""Run the software-only product workflow from one installed, UI-bearing wheel.

This is the M3 installed-artifact gate.  It creates one clean virtual
environment, installs the exact wheel selected from DIST_DIR (with the sim
extra when requested), and runs isolated mock and MuJoCo workflows from that
environment.  The checkout is never placed on ``sys.path``.

Every backend gets a fresh temporary SO101_HOME, data, datasets and outputs
tree, empty camera configuration and deliberately nonexistent serial ports.
Before the first mutating HTTP request, the harness proves that the server
reports the requested mock/sim backend and refuses if it is real or latched.

The evidence directory is retained.  The virtual environment, server state
and generated datasets/checkpoints are removed unless ``--keep-work`` is
given.  A failed run still writes its command output, server logs and failure
summary to the evidence directory.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

ASSET_REFERENCE = re.compile(rb"(?:src|href)=[\"'](/assets/[^\"'?#]+)[\"']")
DROPPED_ENV = frozenset({"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"})
TERMINAL_JOBS = frozenset({"succeeded", "failed", "cancelled", "interrupted", "corrupt"})
TERMINAL_EVAL = frozenset({"done", "aborted", "failed"})
POLL_SECONDS = 0.1
SERVER_START_SECONDS = 90.0
ASYNC_SECONDS = 60.0
COLLAPSE_SECONDS = 20.0
PROCESS_STOP_SECONDS = 20.0
OPERATOR = "installed-e2e"
UI_E2E_COMMAND = ("npm", "run", "test:e2e")
E2E_HEADER = "X-SO101-E2E-Token"
E2E_KIND = "so101.installed_e2e"
EVIDENCE_DIR_MODE = 0o700
EVIDENCE_FILE_MODE = 0o600
HASH_MANIFEST = "EVIDENCE_SHA256SUMS"
RUNTIME_UNSAFE_ENV = frozenset(
    {
        "BASH_ENV",
        "DEBUG",
        "DYLD_INSERT_LIBRARIES",
        "ENV",
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LD_PRELOAD",
        "NODE_DEBUG",
        "NODE_OPTIONS",
        "NODE_PATH",
        "PWDEBUG",
        "ZDOTDIR",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "NO_PROXY",
    }
)
SECRET_ENV_PARTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "API_KEY",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "KEY",
    "JWT",
    "NETRC",
    "AUTH_SOCK",
    "COOKIE",
)
PIP_INSTALL_ENV = frozenset(
    {
        "PIP_CERT",
        "PIP_CLIENT_CERT",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_NO_INDEX",
        "PIP_PROXY",
        "PIP_TRUSTED_HOST",
    }
)
PIP_SECRET_ENV = PIP_INSTALL_ENV - {"PIP_NO_INDEX"}


class GateError(RuntimeError):
    """A gate condition failed; its message is suitable for the evidence log."""


class SafetyRefusal(GateError):
    """The target was not proved to be a fresh, non-hardware backend."""


COMMON_WORKFLOW_STAGES = (
    "identity",
    "preflight",
    "record",
    "inspect_label_qc",
    "deadman",
    "watchdog_halt",
    "lease_expiry",
    "train",
    "rollout",
    "halt",
    "collapse",
    "evaluate",
    "estop",
    "energized_shutdown",
)
SIM_WORKFLOW_STAGES = ("sim_diagnostics", "sim_calibration_refusal")


@dataclass
class WorkflowLedger:
    backend: str
    completed: list[str] = field(default_factory=list)

    @property
    def required(self) -> tuple[str, ...]:
        if self.backend == "sim":
            return (
                *COMMON_WORKFLOW_STAGES[:2],
                *SIM_WORKFLOW_STAGES,
                *COMMON_WORKFLOW_STAGES[2:],
            )
        return COMMON_WORKFLOW_STAGES

    def complete(self, stage: str) -> None:
        index = len(self.completed)
        expected = self.required[index] if index < len(self.required) else None
        if stage != expected:
            raise GateError(
                f"{self.backend} workflow stage {stage!r} arrived out of order; "
                f"expected {expected!r} after {self.completed}"
            )
        self.completed.append(stage)

    def verify_complete(self) -> None:
        if tuple(self.completed) != self.required:
            missing = self.required[len(self.completed) :]
            raise GateError(f"{self.backend} workflow is incomplete; missing {missing}")


@dataclass(frozen=True)
class WheelArtifact:
    path: Path
    sha256: str
    version: str
    index_html: bytes
    ui_stamp: bytes
    assets: dict[str, bytes]
    ui_sources_sha256: str
    ui_git_head: str | None

    def public(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "filename": self.path.name,
            "sha256": self.sha256,
            "version": self.version,
            "ui_index_sha256": hashlib.sha256(self.index_html).hexdigest(),
            "ui_stamp_sha256": hashlib.sha256(self.ui_stamp).hexdigest(),
            "ui_sources_sha256": self.ui_sources_sha256,
            "ui_git_head": self.ui_git_head,
            "ui_assets": {
                name: hashlib.sha256(data).hexdigest() for name, data in self.assets.items()
            },
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_ui_wheel(dist_dir: Path) -> WheelArtifact:
    """Select exactly one ``so101`` wheel and prove its stamped UI is complete."""

    wheels = sorted(dist_dir.glob("so101-*.whl"))
    if len(wheels) != 1:
        found = ", ".join(path.name for path in wheels) or "none"
        raise GateError(f"expected exactly one so101 wheel in {dist_dir}, found {found}")
    wheel = wheels[0].resolve()
    try:
        with zipfile.ZipFile(wheel) as archive:
            entries = archive.namelist()
            if len(entries) != len(set(entries)):
                raise GateError(f"{wheel.name} contains duplicate archive member names")
            names = set(entries)
            index_name = "so101/web/index.html"
            stamp_name = "so101/web/so101-ui-stamp.json"
            missing = [name for name in (index_name, stamp_name) if name not in names]
            if missing:
                raise GateError(f"{wheel.name} is not UI-bearing; missing {', '.join(missing)}")
            index = archive.read(index_name)
            try:
                stamp_bytes = archive.read(stamp_name)
                stamp = json.loads(stamp_bytes)
                manifest = stamp["files"]
            except (KeyError, TypeError, ValueError) as exc:
                raise GateError(f"{wheel.name} has an unreadable UI build stamp: {exc}") from exc
            if not isinstance(stamp, dict) or stamp.get("format") != 1:
                raise GateError(f"{wheel.name}'s UI build stamp is not format 1")
            if set(stamp) != {"format", "sources", "git_head", "files"}:
                raise GateError(f"{wheel.name}'s UI build stamp has unexpected or missing fields")
            if not isinstance(manifest, dict):
                raise GateError(f"{wheel.name}'s UI build stamp has no file manifest")
            if not 2 <= len(manifest) <= 512:
                raise GateError(f"{wheel.name}'s UI build stamp has an invalid file count")
            sources = stamp.get("sources")
            if not isinstance(sources, str) or re.fullmatch(r"[0-9a-f]{64}", sources) is None:
                raise GateError(f"{wheel.name}'s UI build stamp has no valid sources digest")
            git_head = stamp.get("git_head")
            if not isinstance(git_head, str) or re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
                raise GateError(f"{wheel.name}'s UI build stamp has no exact git commit")
            web_prefix = "so101/web/"
            packed_files = {
                name.removeprefix(web_prefix)
                for name in names
                if name.startswith(web_prefix) and not name.endswith("/") and name != stamp_name
            }
            listed_files = set(manifest)
            if listed_files != packed_files:
                raise GateError(
                    f"{wheel.name}'s UI manifest is incomplete: "
                    f"unlisted={sorted(packed_files - listed_files)}, "
                    f"missing={sorted(listed_files - packed_files)}"
                )
            web_bytes: dict[str, bytes] = {}
            for relative, expected in manifest.items():
                if not isinstance(relative, str):
                    raise GateError(f"{wheel.name}'s UI manifest has a non-string path")
                pure = PurePosixPath(relative)
                if not relative or pure.is_absolute() or ".." in pure.parts or "\\" in relative:
                    raise GateError(f"{wheel.name}'s UI manifest has unsafe path {relative!r}")
                if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                    raise GateError(f"{wheel.name}'s UI manifest has invalid hash for {relative!r}")
                data = archive.read(web_prefix + relative)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise GateError(f"{wheel.name}'s {relative} does not match its UI build stamp")
                web_bytes[relative] = data
            index = web_bytes["index.html"]
            references = sorted(set(ASSET_REFERENCE.findall(index)))
            if not references:
                raise GateError(f"{wheel.name}'s UI index references no /assets/ file")
            assets = {
                f"/{relative}": data
                for relative, data in web_bytes.items()
                if relative != "index.html"
            }
            for raw in references:
                public = raw.decode("utf-8")
                member = f"so101/web/{public.lstrip('/')}"
                if member not in names:
                    raise GateError(f"{wheel.name}'s UI is missing referenced {public}")
                relative = member.removeprefix("so101/web/")
                if assets.get(public) != web_bytes[relative]:
                    raise GateError(f"{wheel.name}'s UI reference is not in its complete manifest")
            metadata_names = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
            if len(metadata_names) != 1:
                raise GateError(
                    f"{wheel.name} must contain one dist-info/METADATA, found {len(metadata_names)}"
                )
            metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
            if metadata.get("Name", "").lower() != "so101" or not metadata.get("Version"):
                raise GateError(f"{wheel.name} has invalid so101 package metadata")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise GateError(f"cannot inspect wheel {wheel}: {exc}") from exc
    return WheelArtifact(
        path=wheel,
        sha256=_sha256(wheel),
        version=str(metadata["Version"]),
        index_html=index,
        ui_stamp=stamp_bytes,
        assets=assets,
        ui_sources_sha256=sources,
        ui_git_head=git_head,
    )


def prepare_evidence(path: Path) -> Path:
    """Create a new evidence directory; stale results can never look current."""

    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise GateError(f"evidence directory may not be a symlink: {requested}")
    target = requested.resolve()
    if target.exists():
        if not target.is_dir():
            raise GateError(f"evidence path is not a directory: {target}")
        if any(target.iterdir()):
            raise GateError(
                f"evidence directory is not empty: {target}; choose a new directory so stale "
                "artifacts cannot be mistaken for this run"
            )
    target.mkdir(parents=True, exist_ok=True, mode=EVIDENCE_DIR_MODE)
    target.chmod(EVIDENCE_DIR_MODE)
    return target


def _secure_directory(path: Path) -> None:
    if path.is_symlink():
        raise GateError(f"evidence directory may not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=EVIDENCE_DIR_MODE)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GateError(f"evidence path is not an ordinary directory: {path}")
    path.chmod(EVIDENCE_DIR_MODE)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    _secure_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, EVIDENCE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(EVIDENCE_FILE_MODE)


def _copy_evidence_file(source: Path, destination: Path, *, exclusive: bool = False) -> Path:
    _secure_directory(destination.parent)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, EVIDENCE_FILE_MODE)
    try:
        with os.fdopen(descriptor, "wb") as outgoing, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except (Exception, KeyboardInterrupt):
        destination.unlink(missing_ok=True)
        raise
    destination.chmod(EVIDENCE_FILE_MODE)
    return destination


def snapshot_ui_wheel(dist_dir: Path, snapshot_dir: Path) -> tuple[Path, Path]:
    """Copy the sole candidate once, before any parsing or hashing.

    The build output may live in a directory another process can rewrite.  All
    artifact inspection and evidence publication therefore use this private,
    read-only snapshot rather than reopening the selected build path.
    """

    wheels = sorted(dist_dir.glob("so101-*.whl"))
    if len(wheels) != 1:
        found = ", ".join(path.name for path in wheels) or "none"
        raise GateError(f"expected exactly one so101 wheel in {dist_dir}, found {found}")
    source = wheels[0].resolve()
    _secure_directory(snapshot_dir)
    snapshot = _copy_evidence_file(source, snapshot_dir / source.name, exclusive=True)
    snapshot.chmod(0o400)
    return source, snapshot


def _secure_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise GateError(f"evidence contains an unexpected symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(EVIDENCE_DIR_MODE)
        elif stat.S_ISREG(metadata.st_mode):
            path.chmod(EVIDENCE_FILE_MODE)
        else:
            raise GateError(f"evidence contains a special file: {path}")


def write_evidence_hashes(root: Path) -> dict[str, str]:
    """Hash every retained regular file except the hash manifest itself."""

    hashes = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / HASH_MANIFEST
    }
    _write_text(
        root / HASH_MANIFEST,
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
    )
    return hashes


def assert_secret_absent(root: Path, secret: str) -> None:
    if not secret:
        return
    needle = secret.encode()
    leaked: list[Path] = []
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise GateError(f"evidence contains an unexpected symlink: {path}")
        if stat.S_ISREG(metadata.st_mode) and needle in path.read_bytes():
            leaked.append(path)
    if leaked:
        raise GateError(
            "ephemeral E2E token leaked into evidence: "
            + ", ".join(str(path.relative_to(root)) for path in leaked)
        )


def redact_secret_file(path: Path, secret: str) -> bool:
    if not secret or not path.is_file():
        return False
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GateError(f"cannot redact non-regular evidence file: {path}")
    needle = secret.encode()
    data = path.read_bytes()
    if needle not in data:
        return False
    replacement = b"*" * len(needle)
    redacted = data.replace(needle, replacement)
    _secure_directory(path.parent)
    flags = os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, EVIDENCE_FILE_MODE)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(redacted)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(EVIDENCE_FILE_MODE)
    if needle in path.read_bytes():
        raise GateError(f"could not redact E2E token from {path}")
    return True


def redact_secret_tree(root: Path, secret: str) -> list[str]:
    redacted: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise GateError(f"evidence contains an unexpected symlink: {path}")
        if path.is_file() and redact_secret_file(path, secret):
            redacted.append(path.relative_to(root).as_posix())
    return redacted


def copy_wheel_to_evidence(artifact: WheelArtifact, evidence: Path) -> WheelArtifact:
    destination = evidence / "wheel" / artifact.path.name
    _copy_evidence_file(artifact.path, destination, exclusive=True)
    copied_hash = _sha256(destination)
    if copied_hash != artifact.sha256:
        raise GateError(f"evidence wheel copy changed: {copied_hash} != selected {artifact.sha256}")
    return replace(artifact, path=destination)


def _secret_environment_key(key: str) -> bool:
    upper = key.upper()
    return any(part in upper for part in SECRET_ENV_PARTS)


def _credential_url(value: str) -> bool:
    return bool(_package_url_secret_values(value))


_URL_SECRET_HINT = re.compile(
    r"(?:access|api|auth|bearer|credential|key|pass|secret|signature|token)", re.IGNORECASE
)
_PUBLIC_PACKAGE_PATHS = frozenset(
    {
        "api",
        "cpu",
        "index",
        "packages",
        "pypi",
        "repository",
        "simple",
        "v1",
        "v2",
        "v3",
        "whl",
    }
)


def _secret_variants(value: str | None) -> set[str]:
    if not value:
        return set()
    decoded_forms = {value}
    frontier = {value}
    for _ in range(3):
        next_frontier = {
            decoded
            for item in frontier
            for decoded in (urllib.parse.unquote(item), urllib.parse.unquote_plus(item))
        } - decoded_forms
        if not next_frontier:
            break
        decoded_forms.update(next_frontier)
        frontier = next_frontier
    forms: set[str] = set()
    for item in decoded_forms:
        if len(item) < 3:
            continue
        encoded = urllib.parse.quote(item, safe="")
        encoded_plus = urllib.parse.quote_plus(item, safe="")
        forms.update(
            {
                item,
                item.lower(),
                item.upper(),
                encoded,
                encoded.lower(),
                encoded.upper(),
                encoded_plus,
                encoded_plus.lower(),
                encoded_plus.upper(),
            }
        )
    return forms


def _package_url_secret_values(value: str) -> set[str]:
    """Return credentials embedded in a package URL, including transformed forms.

    Pip and proxies may echo a URL with userinfo removed or percent escapes decoded,
    so redacting only the original environment string and password is insufficient.
    Ordinary public index paths such as ``/simple`` are not treated as credentials.
    """

    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return set()
    if not parsed.scheme or not parsed.netloc:
        return set()

    secrets: set[str] = set()
    secrets.update(_secret_variants(parsed.username))
    secrets.update(_secret_variants(parsed.password))

    for pair in parsed.query.split("&"):
        raw_key, separator, raw_item = pair.partition("=")
        if separator and raw_item:
            secrets.update(_secret_variants(raw_item))
        if _URL_SECRET_HINT.search(urllib.parse.unquote(raw_key)):
            secrets.update(_secret_variants(raw_key))
    for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        if item:
            secrets.update(_secret_variants(item))
        if _URL_SECRET_HINT.search(key):
            secrets.update(_secret_variants(key))

    if parsed.fragment:
        secrets.update(_secret_variants(parsed.fragment))
        for pair in parsed.fragment.split("&"):
            raw_key, separator, raw_item = pair.partition("=")
            if separator and raw_item:
                secrets.update(_secret_variants(raw_item))
            if _URL_SECRET_HINT.search(urllib.parse.unquote(raw_key)):
                secrets.update(_secret_variants(raw_key))
        for key, item in urllib.parse.parse_qsl(parsed.fragment, keep_blank_values=True):
            if item:
                secrets.update(_secret_variants(item))
            if _URL_SECRET_HINT.search(key):
                secrets.update(_secret_variants(key))

    for raw_segment in parsed.path.split("/"):
        segment = urllib.parse.unquote(raw_segment)
        looks_opaque = len(segment) >= 20 and any(char.isdigit() for char in segment)
        unknown_path_value = len(segment) >= 3 and segment.lower() not in _PUBLIC_PACKAGE_PATHS
        if raw_segment and (_URL_SECRET_HINT.search(segment) or looks_opaque or unknown_path_value):
            secrets.update(_secret_variants(raw_segment))

    if secrets:
        secrets.update(_secret_variants(value))
    return secrets


def runtime_base_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Host/runtime variables without generic credentials or developer Python state."""

    source = os.environ if base is None else base
    return {
        key: value
        for key, value in source.items()
        if key not in DROPPED_ENV
        and key not in RUNTIME_UNSAFE_ENV
        and not key.startswith("PYTHON")
        and not key.startswith("SO101_")
        and not key.startswith("PIP_")
        and not _secret_environment_key(key)
        and not _credential_url(value)
    }


def clean_env(root: Path, backend: str, *, e2e_token: str = "") -> dict[str, str]:
    """Installed-process environment with explicit no-hardware sentinels."""

    if backend not in {"mock", "sim"}:
        raise SafetyRefusal(f"installed E2E refuses backend {backend!r}; use mock or sim")
    if not e2e_token:
        raise SafetyRefusal("installed E2E needs a nonblank ephemeral token")
    env = runtime_base_env()
    sentinels = root / "NO-HARDWARE-PORTS"
    env.update(
        {
            "SO101_BACKEND": backend,
            "SO101_MOCK": "1" if backend == "mock" else "0",
            "SO101_HOME": str(root / "home"),
            "SO101_DATA_DIR": str(root / "data"),
            "SO101_DATASETS_DIR": str(root / "datasets"),
            "SO101_OUTPUTS_DIR": str(root / "outputs"),
            "SO101_FOLLOWER_PORT": str(sentinels / "follower-MUST-NOT-OPEN"),
            "SO101_LEADER_PORT": str(sentinels / "leader-MUST-NOT-OPEN"),
            "SO101_CAMERAS": "",
            "SO101_SIM_CAMERAS": "",
            "SO101_SIM_ENGINE": "mujoco" if backend == "sim" else "kinematic",
            "SO101_SIM_TASK": "reach",
            "SO101_SIM_SEED": "23",
            "SO101_SIM_REALTIME": "false",
            "SO101_WANDB_ENABLED": "false",
            "SO101_HF_TOKEN": "",
            "SO101_OPENAI_API_KEY": "",
            "SO101_E2E_TOKEN": e2e_token,
        }
    )
    return env


def install_env(home: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Isolated venv/pip environment with only explicit index/transport settings."""

    source = os.environ if base is None else base
    env = {
        key: value
        for key, value in source.items()
        if key not in DROPPED_ENV
        and key not in RUNTIME_UNSAFE_ENV
        and not key.startswith("PYTHON")
        and not key.startswith("SO101_")
        and not key.startswith("PIP_")
        and not _secret_environment_key(key)
        and not _credential_url(value)
    }
    env.update({key: source[key] for key in PIP_INSTALL_ENV if source.get(key)})
    env.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
        }
    )
    return env


def sensitive_environment_values(base: Mapping[str, str] | None = None) -> list[str]:
    source = os.environ if base is None else base
    values: set[str] = set()
    for key, value in source.items():
        if not value:
            continue
        if _secret_environment_key(key) and len(value) >= 6:
            values.add(value)
        if key in PIP_INSTALL_ENV:
            if key in PIP_SECRET_ENV:
                values.update(_secret_variants(value))
            # PIP_EXTRA_INDEX_URL and PIP_FIND_LINKS may contain several URLs.
            # Preserve them for installation, but enumerate every credential-like
            # component that a downloader or proxy could print in transformed form.
            for candidate in value.split():
                values.update(_package_url_secret_values(candidate))
    return sorted(values, key=len, reverse=True)


def _sanitize_text(text: str, values: Sequence[str]) -> str:
    sanitized = text
    for value in values:
        if value:
            sanitized = sanitized.replace(value, "***REDACTED***")
    return sanitized


class DescendantMonitor:
    def __init__(self, process: subprocess.Popen[Any]) -> None:
        self.process = process
        self.observed: set[int] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="e2e-process-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.observed.update(_descendant_pids(self.process.pid))
            if self._stop.wait(0.5):
                return

    def finish(self) -> set[int]:
        self._stop.set()
        self._thread.join(timeout=1)
        self.observed.update(_descendant_pids(self.process.pid))
        return set(self.observed)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log: Path,
    timeout: float,
    redact_values: Sequence[str] = (),
    return_combined: bool = False,
) -> str:
    """Run one command, persist all output, and make a nonzero exit self-explanatory."""

    command_text = _sanitize_text(shlex.join(command), redact_values)
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    monitor = DescendantMonitor(process)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        descendants = monitor.finish() | _signal_process_tree(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            descendants |= _descendant_pids(process.pid)
            _signal_process_tree(process, signal.SIGKILL, descendants)
            stdout, stderr = process.communicate(timeout=5)
        try:
            _cleanup_process_group(process.pid)
        finally:
            _cleanup_descendants(descendants)
        output = _sanitize_text(stdout + stderr, redact_values)
        _write_text(log, f"$ {command_text}\nTIMEOUT after {timeout:g}s\n{output}")
        raise GateError(
            f"`{command_text}` timed out after {timeout:g}s\n{output}".rstrip()
        ) from exc
    observed_descendants = monitor.finish()
    try:
        _cleanup_process_group(process.pid)
    finally:
        _cleanup_descendants(observed_descendants)
    stdout = _sanitize_text(stdout, redact_values)
    output = _sanitize_text(stdout + stderr, redact_values)
    _write_text(log, f"$ {command_text}\nexit: {process.returncode}\n{output}")
    if process.returncode != 0:
        raise GateError(f"`{command_text}` exited {process.returncode}\n{output}".rstrip())
    return output if return_combined else stdout


T = TypeVar("T")


def wait_for(
    description: str,
    read: Callable[[], T],
    ready: Callable[[T], bool],
    timeout: float,
    *,
    interval: float = POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    pause: Callable[[float], None] = time.sleep,
    failed: Callable[[T], str | None] | None = None,
) -> T:
    """Poll observable state until it proves completion; never retry an operation."""

    deadline = clock() + timeout
    last = read()
    while not ready(last):
        problem = failed(last) if failed is not None else None
        if problem:
            raise GateError(f"failed while waiting for {description}: {problem}; state: {last!r}")
        if clock() >= deadline:
            raise GateError(f"timed out waiting for {description}; last state: {last!r}")
        pause(min(interval, max(0.0, deadline - clock())))
        last = read()
    return last


def guard_problems(
    status: Mapping[str, Any],
    control: Mapping[str, Any],
    expected_backend: str,
    *,
    jobs: Sequence[Mapping[str, Any]] = (),
    eval_state: Mapping[str, Any] | None = None,
) -> list[str]:
    """Reasons no state-changing request may be sent to this server."""

    problems: list[str] = []
    backend = status.get("backend")
    if expected_backend not in {"mock", "sim"}:
        problems.append(f"requested backend {expected_backend!r} is not a rehearsal backend")
    if backend != expected_backend:
        problems.append(f"server backend is {backend!r}, expected {expected_backend!r}")
    if backend == "real":
        problems.append("server reports the real backend")
    if status.get("estopped") is not False:
        problems.append(f"E-stop state is not explicitly clear ({status.get('estopped')!r})")
    if status.get("torque_unconfirmed") is not False:
        problems.append(
            f"torque-off confirmation is not explicitly clear "
            f"({status.get('torque_unconfirmed')!r})"
        )
    if control.get("safety_state") != "disarmed":
        problems.append(f"control safety_state is {control.get('safety_state')!r}, not 'disarmed'")
    if status.get("mode") != "idle" or control.get("armed") is not False:
        problems.append("the server is not idle and disarmed")
    if status.get("stop_state") != "none" or status.get("torque_state") != "off":
        problems.append("the server is holding or energized instead of fresh and torque-off")
    if "lease" not in control or control.get("lease") is not None:
        problems.append("an external control lease is active")
    if status.get("latch_persistence") != "saved":
        problems.append(
            f"latch persistence is {status.get('latch_persistence')!r}, not durably saved"
        )
    active_jobs = [
        str(job.get("id", "?"))
        for job in jobs
        if job.get("status") in {"pending", "queued", "running"}
    ]
    if active_jobs:
        problems.append(f"training jobs are active: {active_jobs}")
    if eval_state is not None and eval_state.get("active"):
        problems.append(f"an evaluation is active ({eval_state.get('phase')!r})")
    return problems


def assert_safe_backend(
    status: Mapping[str, Any],
    control: Mapping[str, Any],
    expected_backend: str,
    *,
    jobs: Sequence[Mapping[str, Any]] = (),
    eval_state: Mapping[str, Any] | None = None,
) -> None:
    problems = guard_problems(
        status,
        control,
        expected_backend,
        jobs=jobs,
        eval_state=eval_state,
    )
    if problems:
        raise SafetyRefusal("no mutation was sent: " + "; ".join(problems))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _process_group_alive(group_id: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _descendant_pids(root_pid: int) -> set[int]:
    """Best-effort process descendants, including children that created a new group."""

    if os.name != "posix":
        return set()
    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    children: dict[int, set[int]] = {}
    for line in listing.splitlines():
        try:
            pid, parent = (int(value) for value in line.split())
        except (TypeError, ValueError):
            continue
        children.setdefault(parent, set()).add(pid)
    descendants: set[int] = set()
    pending = list(children.get(root_pid, set()))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children.get(pid, set()))
    return descendants


def _signal_process_tree(
    process: subprocess.Popen[Any],
    sig: signal.Signals,
    descendants: set[int] | None = None,
) -> set[int]:
    descendants = _descendant_pids(process.pid) if descendants is None else descendants
    if process.poll() is not None and not _process_group_alive(process.pid):
        return descendants
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass
    for pid in sorted(descendants, reverse=True):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    return descendants


def _cleanup_process_group(group_id: int, timeout: float = 2.0) -> None:
    """Remove descendants left in a command/server session after its leader exits."""

    if not _process_group_alive(group_id):
        return
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while _process_group_alive(group_id) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _process_group_alive(group_id):
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            return
    deadline = time.monotonic() + timeout
    while _process_group_alive(group_id) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _process_group_alive(group_id):
        raise GateError(f"process group {group_id} still has live descendants after SIGKILL")


def _cleanup_descendants(pids: set[int], timeout: float = 2.0) -> None:
    if os.name != "posix" or not pids:
        return

    def alive() -> set[int]:
        remaining: set[int] = set()
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                pass
            remaining.add(pid)
        return remaining

    remaining = alive()
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while (remaining := alive()) and time.monotonic() < deadline:
        time.sleep(0.02)
    if remaining:
        raise GateError(f"descendant processes survived cleanup: {sorted(remaining)}")


def stop_process(
    process: subprocess.Popen[bytes],
    timeout: float = PROCESS_STOP_SECONDS,
    *,
    known_descendants: set[int] | None = None,
) -> str:
    """Stop a server gracefully, escalating to kill only after the bounded wait."""

    descendants = set(known_descendants or ())
    if process.poll() is not None:
        try:
            _cleanup_process_group(process.pid)
        finally:
            _cleanup_descendants(descendants)
        raise GateError(f"installed server exited unexpectedly with code {process.returncode}")
    descendants |= _signal_process_tree(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        descendants |= _descendant_pids(process.pid)
        _signal_process_tree(process, signal.SIGKILL, descendants)
        process.wait(timeout=timeout)
        try:
            _cleanup_process_group(process.pid)
        finally:
            _cleanup_descendants(descendants)
        raise GateError(
            f"installed server ignored SIGTERM for {timeout:g}s; process group was killed "
            f"({process.returncode})"
        ) from None
    try:
        _cleanup_process_group(process.pid)
    finally:
        _cleanup_descendants(descendants)
    if process.returncode != 0:
        raise GateError(
            f"installed server exited nonzero during graceful shutdown: {process.returncode}"
        )
    return f"process group terminated ({process.returncode})"


class Api:
    """Small logged HTTP client for one throwaway installed server."""

    def __init__(
        self,
        base: str,
        log: Path,
        *,
        expected_backend: str,
        expected_version: str,
        e2e_token: str = "",
    ) -> None:
        if expected_backend not in {"mock", "sim"}:
            raise SafetyRefusal(f"E2E API refuses backend {expected_backend!r}")
        if not e2e_token:
            raise SafetyRefusal("API client needs the ephemeral E2E token")
        self.base = base.rstrip("/")
        self.log = log
        self.expected_backend = expected_backend
        self.expected_version = expected_version
        self._e2e_token = e2e_token
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.sequence = 0
        self._mutations_allowed = False

    def authorize_mutations(
        self,
        status: Mapping[str, Any],
        control: Mapping[str, Any],
        expected_backend: str,
        *,
        jobs: Sequence[Mapping[str, Any]] = (),
        eval_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Unlock non-GET methods only after the non-hardware guard passes."""

        if expected_backend != self.expected_backend:
            raise SafetyRefusal(
                f"API client expects {self.expected_backend!r}, not {expected_backend!r}"
            )
        assert_safe_backend(
            status,
            control,
            expected_backend,
            jobs=jobs,
            eval_state=eval_state,
        )
        self._mutations_allowed = True

    def lock_mutations(self) -> None:
        self._mutations_allowed = False

    def reauthorize_external_mutation(self) -> None:
        if not self._mutations_allowed:
            raise SafetyRefusal("external mutation attempted before the backend guard")
        self._reauthorize()

    def _record(
        self,
        method: str,
        path: str,
        request_body: Any,
        status: int | None,
        content_type: str,
        response_body: bytes,
        error: str = "",
    ) -> None:
        _secure_directory(self.log.parent)
        self.sequence += 1
        try:
            response: Any = json.loads(response_body) if response_body else None
        except ValueError:
            response = response_body.decode("utf-8", "replace")[:4000]
        entry = {
            "sequence": self.sequence,
            "method": method,
            "path": path,
            "request": request_body,
            "status": status,
            "content_type": content_type,
            "response": response,
            "error": error or None,
        }
        descriptor = os.open(
            self.log,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            EVIDENCE_FILE_MODE,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.log.chmod(EVIDENCE_FILE_MODE)

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None,
    ) -> tuple[int, str, bytes]:
        payload = None if body is None else json.dumps(body, allow_nan=False).encode()
        headers = {"Content-Type": "application/json"}
        if self._e2e_token:
            headers[E2E_HEADER] = self._e2e_token
        request = urllib.request.Request(
            self.base + path,
            data=payload,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=10) as response:
                status = response.status
                content_type = response.headers.get("content-type", "")
                data = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            content_type = exc.headers.get("content-type", "")
            data = exc.read()
        except (OSError, http.client.HTTPException) as exc:
            self._record(method, path, body, None, "", b"", str(exc))
            raise GateError(f"{method} {path}: {exc}") from exc
        self._record(method, path, body, status, content_type, data)
        return status, content_type, data

    @staticmethod
    def _decode_json(method: str, path: str, data: bytes) -> Any:
        try:
            return json.loads(data) if data else None
        except ValueError as exc:
            raise GateError(f"{method} {path}: response is not JSON: {data[:500]!r}") from exc

    def _reauthorize(self) -> None:
        """Bind the next mutation to the expected non-real server."""

        status, _content_type, data = self._request("GET", "/api/e2e/identity", None)
        if status != 200:
            raise SafetyRefusal(
                f"identity reauthorization returned HTTP {status}; no mutation sent"
            )
        identity = self._decode_json("GET", "/api/e2e/identity", data)
        expected = {
            "ok": True,
            "kind": E2E_KIND,
            "schema_version": 1,
            "backend": self.expected_backend,
            "mock": self.expected_backend == "mock",
            "version": self.expected_version,
        }
        if identity != expected:
            raise SafetyRefusal(f"identity changed before mutation: {identity!r} != {expected!r}")

    def raw(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        *,
        expected_status: int = 200,
    ) -> tuple[int, str, bytes]:
        normalized = method.upper()
        if normalized not in {"GET", "HEAD", "OPTIONS"}:
            if not self._mutations_allowed:
                raise SafetyRefusal(
                    f"no mutation was sent: {normalized} {path} was attempted before the "
                    "server passed the non-hardware backend guard"
                )
            self._reauthorize()
        status, content_type, data = self._request(normalized, path, body)
        if status != expected_status:
            raise GateError(
                f"{normalized} {path}: HTTP {status}, expected {expected_status}: "
                f"{data.decode(errors='replace')}"
            )
        return status, content_type, data

    def json(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        *,
        expected_status: int = 200,
    ) -> Any:
        _status, _content_type, data = self.raw(
            method,
            path,
            body,
            expected_status=expected_status,
        )
        return self._decode_json(method, path, data)


def wait_for_server(api: Api, server: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    """Wait for health without hiding an early process exit."""

    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise GateError(f"installed server exited with code {server.returncode} during startup")
        try:
            identity = api.json("GET", "/api/e2e/identity")
            if identity.get("kind") != E2E_KIND or identity.get("backend") != api.expected_backend:
                raise SafetyRefusal(f"unexpected installed identity: {identity!r}")
            health = api.json("GET", "/api/health")
        except SafetyRefusal:
            raise
        except GateError as exc:
            if "HTTP " in str(exc):
                raise SafetyRefusal(f"server identity/health was rejected: {exc}") from exc
            last = str(exc)
            time.sleep(POLL_SECONDS)
            continue
        if isinstance(health, dict) and health.get("ok") is True:
            return health
        last = repr(health)
        time.sleep(POLL_SECONDS)
    raise GateError(f"installed server did not become healthy in {timeout:g}s; last: {last}")


def verify_served_ui(api: Api, artifact: WheelArtifact) -> dict[str, Any]:
    status, content_type, index = api.raw("GET", "/")
    if status != 200 or not content_type.startswith("text/html"):
        raise GateError(f"GET / returned {status} {content_type!r}, not the installed UI")
    if index != artifact.index_html:
        raise GateError("GET / did not serve the index.html carried by the selected wheel")
    served: dict[str, str] = {"/": hashlib.sha256(index).hexdigest()}
    stamp_status, _stamp_type, stamp = api.raw("GET", "/so101-ui-stamp.json")
    if stamp_status != 200 or stamp != artifact.ui_stamp:
        raise GateError("GET /so101-ui-stamp.json differs from the selected wheel")
    served["/so101-ui-stamp.json"] = hashlib.sha256(stamp).hexdigest()
    for path, expected in artifact.assets.items():
        asset_status, _asset_type, data = api.raw("GET", path)
        if asset_status != 200 or data != expected:
            raise GateError(f"GET {path} did not serve the bytes carried by the selected wheel")
        served[path] = hashlib.sha256(data).hexdigest()
    return served


def authorize_fresh_rehearsal(api: Api, backend: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read every activity guard, then unlock this client for mutations."""

    status = api.json("GET", "/api/robot/status")
    control = api.json("GET", "/api/control/state")
    jobs = api.json("GET", "/api/training/jobs")
    eval_state = api.json("GET", "/api/eval/state")
    api.authorize_mutations(
        status,
        control,
        backend,
        jobs=jobs,
        eval_state=eval_state,
    )
    return status, control


def system_browser_provenance(
    env: Mapping[str, str],
    *,
    cwd: Path,
    evidence: Path,
) -> dict[str, str]:
    configured = env.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "").strip()
    if not configured:
        raise GateError("system browser provenance needs an explicit executable override")
    candidates = [configured]
    browser = next(
        (
            Path(candidate).expanduser().resolve()
            for candidate in candidates
            if Path(candidate).expanduser().is_file()
            and os.access(Path(candidate).expanduser(), os.X_OK)
        ),
        None,
    )
    if browser is None:
        raise GateError("installed UI E2E requires an explicit system Chrome/Chromium executable")
    version_output = _run(
        [str(browser), "--version"],
        cwd=cwd,
        env=env,
        log=evidence / "browser-version.log",
        timeout=30,
        return_combined=True,
    )
    version = version_output.strip().splitlines()[0] if version_output.strip() else ""
    if not version or len(version) > 500:
        raise GateError(f"system browser returned an invalid version: {version!r}")
    provenance = {"executable": str(browser), "version": version}
    _write_json(evidence / "browser.json", provenance)
    return provenance


def parse_browser_metadata(log: Path) -> dict[str, Any]:
    prefix = "SO101_E2E_BROWSER_METADATA="
    lines = [
        line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)
    ]
    if len(lines) != 1:
        raise GateError(f"UI E2E launcher emitted {len(lines)} browser metadata lines")
    try:
        metadata = json.loads(lines[0].removeprefix(prefix))
    except ValueError as exc:
        raise GateError("UI E2E browser metadata is invalid JSON") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("kind") != "so101.e2e_browser"
        or metadata.get("source") not in {"system_override", "playwright_pinned"}
        or not isinstance(metadata.get("executable_path"), str)
        or not metadata["executable_path"]
        or not isinstance(metadata.get("version"), str)
        or not metadata["version"]
    ):
        raise GateError(f"UI E2E browser metadata is incomplete: {metadata!r}")
    return metadata


def run_ui_hook(
    *,
    base: str,
    e2e_token: str,
    expected_version: str,
    cwd: Path,
    evidence: Path,
) -> None:
    """Run an optional browser suite against the fresh installed mock server."""

    if not e2e_token:
        raise SafetyRefusal("UI E2E hook needs the ephemeral mock identity token")
    if not expected_version or any(char.isspace() for char in expected_version):
        raise GateError("UI E2E hook needs the installed artifact's package version")
    if cwd.name != "frontend" or not (cwd / "package.json").is_file():
        raise GateError(f"UI E2E working directory is not the frontend package: {cwd}")
    base_env = runtime_base_env()
    explicit_browser = base_env.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "").strip()
    env = dict(base_env)
    for key in list(env):
        if key in {"DEBUG", "PWDEBUG", "NODE_DEBUG"} or key.startswith(("PW_", "PLAYWRIGHT_")):
            env.pop(key, None)
    browser = None
    if explicit_browser:
        browser = system_browser_provenance(base_env, cwd=cwd, evidence=evidence)
        env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = browser["executable"]
    env["SO101_E2E_BASE_URL"] = base
    env["SO101_E2E_TOKEN"] = e2e_token
    env["SO101_E2E_EXPECTED_VERSION"] = expected_version
    artifact_leaf = evidence / "playwright"
    if artifact_leaf.is_symlink():
        raise GateError(f"unsafe Playwright artifact symlink: {artifact_leaf}")
    artifacts = artifact_leaf.resolve()
    if artifacts.name != "playwright":
        raise GateError(f"unsafe Playwright artifact directory: {artifacts}")
    if artifacts.exists() and any(artifacts.iterdir()):
        raise GateError(f"Playwright artifact directory is not empty: {artifacts}")
    _secure_directory(artifacts)
    env["PLAYWRIGHT_ARTIFACTS_DIR"] = str(artifacts)
    log = evidence / "ui-e2e.log"
    try:
        _run(
            UI_E2E_COMMAND,
            cwd=cwd,
            env=env,
            log=log,
            timeout=900,
            redact_values=(e2e_token,),
        )
        metadata = parse_browser_metadata(log)
        if browser is not None and (
            metadata["source"] != "system_override"
            or Path(metadata["executable_path"]).resolve() != Path(browser["executable"])
            or metadata["version"] != browser["version"]
        ):
            raise GateError("UI E2E launcher did not use the recorded system browser override")
        if browser is None and metadata["source"] != "playwright_pinned":
            raise GateError("UI E2E launcher did not use its pinned Playwright browser")
        _write_json(evidence / "browser.json", metadata)
    except Exception as exc:
        raise GateError(_sanitize_text(f"fixed UI E2E failed: {exc}", (e2e_token,))) from None
    finally:
        redacted = ["ui-e2e.log"] if redact_secret_file(log, e2e_token) else []
        redacted.extend(f"playwright/{name}" for name in redact_secret_tree(artifacts, e2e_token))
        if redacted:
            raise GateError(
                "UI E2E command exposed its ephemeral token; redacted " + ", ".join(redacted)
            )


def newest_event_id(api: Api) -> int:
    events = api.json("GET", "/api/events?limit=1").get("events", [])
    return int(events[0]["id"]) if events else 0


def events_after(api: Api, event_id: int) -> list[dict[str, Any]]:
    result = api.json("GET", f"/api/events?after_id={event_id}&limit=500")
    return [dict(event) for event in result.get("events", [])]


def persisted_events_after(data_dir: Path, event_id: int) -> list[dict[str, Any]]:
    database = data_dir / "control_plane.sqlite3"
    if not database.is_file():
        raise GateError(f"control-plane event database is missing after shutdown: {database}")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        with connection:
            rows = connection.execute(
                """
                SELECT id, created_at, type, level, source, message, detail_json
                FROM events WHERE id > ? ORDER BY id ASC
                """,
                (event_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise GateError(f"cannot read shutdown evidence from {database}: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "type": row["type"],
            "level": row["level"],
            "source": row["source"],
            "message": row["message"],
            "detail": json.loads(row["detail_json"]),
        }
        for row in rows
    ]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _relative_child(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise GateError(f"{label} escaped its temporary root: {resolved} is not under {root}")
    return resolved


def _wilson(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    margin /= denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


PROBE_CODE = r"""
import hashlib, importlib.resources, json, pathlib, sys
import so101
from so101.config import Settings
from so101.hardware.backend import current_backend

s = Settings()
web = importlib.resources.files("so101") / "web"
index = (web / "index.html").read_bytes()
print(json.dumps({
    "backend": current_backend(s),
    "cameras": s.cameras,
    "data_dir": str(s.data_dir),
    "datasets_dir": str(s.datasets_dir),
    "follower_port": s.follower_port,
    "index_sha256": hashlib.sha256(index).hexdigest(),
    "leader_port": s.leader_port,
    "module_file": so101.__file__,
    "outputs_dir": str(s.outputs_dir),
    "prefix": sys.prefix,
    "sim_cameras": s.sim_cameras,
    "version": so101.__version__,
}))
"""

TELEMETRY_PROBE_CODE = r"""
import asyncio, inspect, json, os, sys
import websockets

async def main():
    uri, minimum, mode, minimum_frames, timeout = sys.argv[1:]
    headers = {"X-SO101-E2E-Token": os.environ["SO101_E2E_TOKEN"]}
    parameters = inspect.signature(websockets.connect).parameters
    header_name = "additional_headers" if "additional_headers" in parameters else "extra_headers"
    kwargs = {header_name: headers, "open_timeout": float(timeout), "close_timeout": 2}
    async with asyncio.timeout(float(timeout)):
        async with websockets.connect(uri, **kwargs) as connection:
            while True:
                raw = await connection.recv()
                frame = json.loads(raw)
                if int(frame.get("sample_seq", -1)) < int(minimum):
                    continue
                if mode and frame.get("mode") != mode:
                    continue
                if int(frame.get("episode_frames", -1)) < int(minimum_frames):
                    continue
                print(json.dumps(frame, allow_nan=False))
                return

asyncio.run(main())
"""


class TelemetryProbe:
    def __init__(
        self,
        python: Path,
        base: str,
        *,
        cwd: Path,
        env: Mapping[str, str],
        evidence: Path,
    ) -> None:
        self.python = python
        self.uri = base.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        self.uri += "/ws/telemetry"
        self.cwd = cwd
        self.env = dict(env)
        self.evidence = evidence
        self.sequence = 0

    def wait(
        self,
        *,
        minimum_sequence: int = 0,
        mode: str = "",
        minimum_episode_frames: int = -1,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        self.sequence += 1
        output = _run(
            [
                str(self.python),
                "-c",
                TELEMETRY_PROBE_CODE,
                self.uri,
                str(minimum_sequence),
                mode,
                str(minimum_episode_frames),
                str(timeout),
            ],
            cwd=self.cwd,
            env=self.env,
            log=self.evidence / f"telemetry-{self.sequence:03d}.log",
            timeout=timeout + 5,
        )
        try:
            frame = json.loads(output)
        except ValueError as exc:
            raise GateError(f"telemetry probe returned invalid JSON: {output!r}") from exc
        if not isinstance(frame, dict):
            raise GateError(f"telemetry probe returned {frame!r}, not an object")
        return frame


def _record_mock_dataset(
    api: Api,
    telemetry: TelemetryProbe,
    repo_id: str,
    episodes: int = 3,
) -> list[dict[str, Any]]:
    """Record the synthetic leader through the real HTTP recording state machine."""

    _heartbeat(api)
    api.json(
        "POST",
        "/api/recording/start",
        {
            "repo_id": repo_id,
            "task": "installed E2E mock reach",
            "fps": 30,
            "operator_id": OPERATOR,
        },
    )
    latest = telemetry.wait(mode="recording", minimum_sequence=1)
    _assert(latest.get("torque_state") == "on", "recording telemetry is not energized")
    saved: list[dict[str, Any]] = []
    for episode in range(episodes):
        _heartbeat(api)
        api.json("POST", "/api/recording/episode/start")
        latest = telemetry.wait(
            mode="recording",
            minimum_sequence=int(latest.get("sample_seq", 0)) + 1,
            minimum_episode_frames=12,
        )
        _heartbeat(api)
        result = api.json("POST", "/api/recording/episode/save")
        _assert(result.get("episode") == episode, "recording returned the wrong episode index")
        _assert(result.get("frames", 0) >= 10, "recording saved too few frames")
        saved.append(result)
    api.json("POST", "/api/recording/stop")
    halted = wait_for(
        "recording stop to Halt",
        lambda: api.json("GET", "/api/robot/status"),
        lambda status: status.get("mode") == "idle" and status.get("stop_state") == "halted",
        ASYNC_SECONDS,
    )
    _assert(halted.get("torque_state") == "on", "recording stop did not hold with torque on")
    return saved


def analyze_recorded_motion(dataset_root: Path, episode: int = 0) -> dict[str, Any]:
    """Prove a published fallback episode contains changing actions and measured motion."""

    path = dataset_root / f"episode_{episode:06d}" / "frames.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError) as exc:
        raise GateError(f"recorded action evidence is unreadable: {path}: {exc}") from exc
    if len(rows) < 10:
        raise GateError(f"recorded episode has only {len(rows)} frames; expected at least 10")

    def vectors(key: str) -> list[list[float]]:
        values: list[list[float]] = []
        for index, row in enumerate(rows):
            raw = row.get(key) if isinstance(row, dict) else None
            if not isinstance(raw, list) or len(raw) != 6:
                raise GateError(f"frame {index} has invalid {key}: {raw!r}")
            vector = [float(value) for value in raw]
            if any(not math.isfinite(value) for value in vector):
                raise GateError(f"frame {index} has non-finite {key}")
            values.append(vector)
        return values

    actions = vectors("action")
    states = vectors("observation.state")

    def maximum_arm_span(values: Sequence[Sequence[float]]) -> float:
        return max(
            max(row[joint] for row in values) - min(row[joint] for row in values)
            for joint in range(5)
        )

    action_span = maximum_arm_span(actions)
    measured_span = maximum_arm_span(states)
    max_action = max(abs(row[joint]) for row in actions for joint in range(5))
    _assert(max_action > 0.5, "recorded arm commands are effectively all zero")
    _assert(action_span > 0.1, "recorded arm commands do not change")
    _assert(measured_span > 0.05, "recorded arm state shows no measured motion")
    return {
        "episode": episode,
        "frames": len(rows),
        "frames_file": path.relative_to(dataset_root).as_posix(),
        "frames_sha256": _sha256(path),
        "max_abs_arm_action": round(max_action, 6),
        "max_arm_action_span": round(action_span, 6),
        "max_measured_arm_span": round(measured_span, 6),
        "first_action": actions[0],
        "last_action": actions[-1],
        "first_state": states[0],
        "last_state": states[-1],
    }


def probe_install(
    python: Path,
    venv: Path,
    artifact: WheelArtifact,
    root: Path,
    backend: str,
    env: Mapping[str, str],
    evidence: Path,
) -> dict[str, Any]:
    output = _run(
        [str(python), "-c", PROBE_CODE],
        cwd=root,
        env=env,
        log=evidence / "installed-probe.log",
        timeout=120,
    )
    try:
        probe = json.loads(output)
    except ValueError as exc:
        raise GateError(f"installed package probe did not return JSON: {output}") from exc
    module = Path(str(probe.get("module_file", ""))).resolve()
    _assert(module.is_relative_to(venv.resolve()), f"so101 imported from {module}, not {venv}")
    _assert(probe.get("prefix") == str(venv), f"probe used unexpected prefix {probe.get('prefix')}")
    _assert(probe.get("version") == artifact.version, "installed version differs from wheel")
    _assert(probe.get("backend") == backend, "installed settings selected the wrong backend")
    _assert(probe.get("cameras") == "", "real camera configuration was not empty")
    _assert(probe.get("sim_cameras") == "", "sim camera configuration was not empty")
    _assert(
        "NO-HARDWARE-PORTS" in str(probe.get("follower_port"))
        and "NO-HARDWARE-PORTS" in str(probe.get("leader_port")),
        "serial-port sentinels were not in force",
    )
    _assert(
        probe.get("index_sha256") == hashlib.sha256(artifact.index_html).hexdigest(),
        "installed UI differs from the selected wheel",
    )
    _write_json(evidence / "installed-probe.json", probe)
    return probe


def _copy_artifacts(
    mode_evidence: Path,
    mode_root: Path,
    detail: Mapping[str, Any],
    qc: Mapping[str, Any],
    model: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    _write_json(mode_evidence / "dataset.json", detail)
    _write_json(mode_evidence / "dataset-qc.json", qc)
    _write_json(mode_evidence / "model.json", model)
    outputs = mode_root / "outputs"
    checkpoint = _relative_child(Path(str(model["path"])), outputs, "model checkpoint")
    checkpoint_copy = mode_evidence / "checkpoint"
    shutil.copytree(checkpoint, checkpoint_copy, symlinks=True)
    _secure_tree(checkpoint_copy)
    report_path = _relative_child(Path(str(report["_path"])), outputs, "evaluation report")
    _copy_evidence_file(report_path, mode_evidence / "eval-report.json")
    journal = report_path.with_suffix(".jsonl")
    if not journal.is_file():
        raise GateError(f"evaluation journal is missing: {journal}")
    _copy_evidence_file(journal, mode_evidence / journal.name)
    dataset_root = _relative_child(Path(str(detail["path"])), mode_root / "datasets", "dataset")
    dataset_copy = mode_evidence / "dataset"
    shutil.copytree(dataset_root, dataset_copy, symlinks=True)
    _secure_tree(dataset_copy)
    return {
        "dataset_files": sum(path.is_file() for path in dataset_copy.rglob("*")),
        "checkpoint_sha256": {
            str(path.relative_to(checkpoint_copy)): _sha256(path)
            for path in sorted(checkpoint_copy.rglob("*"))
            if path.is_file()
        },
        "eval_report_sha256": _sha256(mode_evidence / "eval-report.json"),
    }


def _dataset_workflow(
    backend: str,
    api: Api,
    telemetry: TelemetryProbe,
    so101: Path,
    mode_root: Path,
    env: Mapping[str, str],
    evidence: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo_id = f"installed_e2e/{backend}_reach"
    if backend == "mock":
        recording_results = _record_mock_dataset(api, telemetry, repo_id, episodes=3)
        _write_json(evidence / "recording-results.json", recording_results)
        expected_episodes = 3
    else:
        api.reauthorize_external_mutation()
        _run(
            [
                str(so101),
                "sim",
                "demos",
                repo_id,
                "--task",
                "reach",
                "--episodes",
                "3",
                "--seed",
                "23",
                "--cameras",
                "",
            ],
            cwd=mode_root,
            env=env,
            log=evidence / "create-dataset.log",
            timeout=300,
        )
        expected_episodes = 3
    route = urllib.parse.quote(repo_id, safe="/")
    detail = api.json("GET", f"/api/datasets/{route}")
    _assert(detail.get("repo_id") == repo_id, "dataset inspection returned another dataset")
    _assert(detail.get("num_episodes") == expected_episodes, "dataset episode count is wrong")
    _assert(detail.get("format") == "so101-fallback-v2", "unexpected dataset format")
    dataset_root = _relative_child(
        Path(str(detail["path"])),
        mode_root / "datasets",
        "recorded dataset",
    )
    action_evidence = analyze_recorded_motion(dataset_root)
    _write_json(evidence / "action-evidence.json", action_evidence)
    qc = api.json("GET", f"/api/datasets/{route}/qc")
    _assert(qc.get("overall") == "pass", f"dataset QC did not pass: {qc}")
    # Exercise human labeling in both formats/states.  For sim this overrides
    # one automatic predicate label, which must remain recorded in history.
    detail = api.json(
        "POST",
        f"/api/datasets/{route}/episodes/0/outcome",
        {"outcome": "success", "notes": "installed E2E accepted demonstration"},
    )
    detail = api.json(
        "POST",
        f"/api/datasets/{route}/episodes/1/outcome",
        {"outcome": "discarded", "notes": "installed E2E selection check"},
    )
    by_episode = {entry["episode"]: entry for entry in detail.get("episodes", [])}
    _assert(by_episode[0].get("outcome") == "success", "success label was not stored")
    _assert(by_episode[0].get("outcome_source") == "human", "success label is not human")
    _assert(by_episode[1].get("outcome") == "discarded", "discarded label was not stored")
    _assert(by_episode[1].get("outcome_source") == "human", "discarded label is not human")
    _assert(detail.get("outcome_counts", {}).get("success", 0) >= 1, "no success is labeled")
    _assert(
        detail.get("outcome_counts", {}).get("discarded", 0) == 1,
        "discarded outcome count is wrong",
    )
    if backend == "sim":
        try:
            label_record = json.loads(
                (dataset_root / "episode_000001" / "episode.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise GateError(f"sim outcome history is unreadable: {exc}") from exc
        history = label_record.get("outcome_history")
        _assert(
            isinstance(history, list)
            and any(
                isinstance(previous, dict) and previous.get("source") == "auto:sim_predicate"
                for previous in history
            ),
            "human sim label did not retain its automatic predicate-label history",
        )
        _write_json(evidence / "sim-label-history.json", label_record)
    return repo_id, detail, qc, action_evidence


def _train_and_find_model(
    backend: str, api: Api, repo_id: str, outputs: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    extra = {"hidden": "16", "eval_every": "10", "patience": "3", "seed": "7"}
    if backend == "mock":
        extra["state_only"] = "true"
    job = api.json(
        "POST",
        "/api/training/jobs",
        {
            "repo_id": repo_id,
            "policy": "bc_baseline",
            "steps": 40,
            "batch_size": 16,
            "episodes": "kept",
            "extra_args": extra,
        },
    )
    job_id = str(job["id"])
    job = wait_for(
        f"training job {job_id}",
        lambda: api.json("GET", f"/api/training/jobs/{job_id}"),
        lambda state: state.get("status") in TERMINAL_JOBS,
        ASYNC_SECONDS,
    )
    logs = api.json("GET", f"/api/training/jobs/{job_id}/logs")
    if job.get("status") != "succeeded":
        raise GateError(f"training job {job_id} ended {job.get('status')}: {job.get('error')}")
    models = api.json("GET", "/api/models")
    matches = [model for model in models if model.get("dataset") == repo_id]
    _assert(bool(matches), f"model registry has no checkpoint trained on {repo_id}")
    model = next((entry for entry in matches if entry.get("checkpoint") == "last"), matches[0])
    _assert(model.get("kind") == "numpy_bc_mlp", "registry did not identify numpy baseline")
    _assert(
        bool(model.get("privileged_state")) == (backend == "sim"),
        "registry reported the wrong privileged-state capability",
    )
    checkpoint = _relative_child(Path(str(model["path"])), outputs, "model checkpoint")
    try:
        manifest = json.loads((checkpoint / "policy_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"trained checkpoint manifest is unreadable: {exc}") from exc
    trained = manifest.get("dataset") or {}
    episodes = trained.get("episodes")
    _assert(trained.get("selection") == "kept", "checkpoint lost the episode selection")
    if not isinstance(episodes, list) or 0 not in episodes:
        raise GateError("success episode was not trained on")
    _assert(1 not in episodes, "discarded episode entered the training set")
    _assert(
        any(entry.get("episode") == 1 for entry in trained.get("skipped", [])),
        "checkpoint does not explain why the discarded episode was skipped",
    )
    model = dict(model)
    model["e2e_episode_selection"] = {
        "selection": trained.get("selection"),
        "episodes": episodes,
        "skipped": trained.get("skipped"),
    }
    return job | {"logs": logs}, model


def _heartbeat(api: Api) -> None:
    api.json("POST", "/api/robot/operator-heartbeat", {"operator_id": OPERATOR})


def _manual_control_contracts(api: Api) -> dict[str, Any]:
    """Exercise deadman rejection and an energized lease-expiry Halt."""

    cursor = newest_event_id(api)
    lease = api.json(
        "POST",
        "/api/control/lease",
        {"source": "installed-e2e", "ttl_seconds": 1.2},
        expected_status=201,
    )
    lease_id = str(lease["id"])
    armed = api.json("POST", "/api/control/arm", {"lease_id": lease_id})
    _assert(armed.get("armed") is True, "manual control did not arm")
    rejected = api.json(
        "POST",
        "/api/control/command",
        {
            "lease_id": lease_id,
            "source": "installed-e2e",
            "sequence": 1,
            "sent_at": time.time(),
            "ttl_ms": 500,
            "deadman": False,
            "targets": {"shoulder_pan": 20.0},
        },
        expected_status=409,
    )
    _assert("deadman" in str(rejected.get("detail", "")).lower(), "deadman release was accepted")
    accepted = api.json(
        "POST",
        "/api/control/command",
        {
            "lease_id": lease_id,
            "source": "installed-e2e",
            "sequence": 1,
            "sent_at": time.time(),
            "ttl_ms": 500,
            "deadman": True,
            "targets": {"shoulder_pan": 20.0},
        },
    )
    _assert(accepted.get("accepted") is True, "valid deadman command was not accepted")
    watchdog = wait_for(
        "energized command watchdog Halt",
        lambda: api.json("GET", "/api/control/state"),
        lambda state: state.get("safety_state") == "watchdog_timeout",
        2.0,
    )
    _assert(watchdog.get("armed") is False, "watchdog Halt stayed armed")
    watchdog_halt = api.json("GET", "/api/robot/status")
    _assert(
        watchdog_halt.get("stop_state") == "halted" and watchdog_halt.get("torque_state") == "on",
        "watchdog did not Halt the energized arm",
    )
    expired = wait_for(
        "manual lease expiry",
        lambda: api.json("GET", "/api/control/state"),
        lambda state: state.get("lease") is None and state.get("safety_state") == "lease_expired",
        3.0,
    )
    _assert(expired.get("armed") is False, "expired lease stayed armed")
    halted = api.json("GET", "/api/robot/status")
    _assert(
        halted.get("stop_state") == "halted" and halted.get("torque_state") == "on",
        "lease expiry did not Halt with torque on",
    )
    events = events_after(api, cursor)
    _assert(
        any(event.get("type") == "control.watchdog_timeout" for event in events),
        "watchdog timeout audit event is missing",
    )
    _assert(
        any(event.get("type") == "control.lease_expired" for event in events),
        "lease expiry audit event is missing",
    )
    return {
        "deadman_rejection": rejected,
        "accepted_command": accepted,
        "watchdog_control": watchdog,
        "watchdog_halt": watchdog_halt,
        "expired_control": expired,
        "halted_status": halted,
        "events": events,
    }


def _sim_contracts(api: Api) -> dict[str, Any]:
    diagnostics = api.json("GET", "/api/diagnostics")
    checks = {entry.get("id"): entry for entry in diagnostics.get("checks", [])}
    backend = checks.get("runtime.backend") or {}
    _assert(backend.get("status") == "pass", f"sim backend diagnostics failed: {backend}")
    _assert("SO101_BACKEND=sim" in str(backend.get("detail")), "diagnostics do not name sim")
    text = json.dumps(diagnostics, sort_keys=True)
    _assert("NO-HARDWARE-PORTS" not in text, "sim diagnostics inspected sentinel serial ports")
    refused = api.json(
        "POST",
        "/api/calibration/start",
        {"arm": "follower", "sample_seconds": 12.0},
        expected_status=409,
    )
    _assert(
        "sim backend" in str(refused.get("detail", "")).lower(),
        f"sim calibration refusal is unclear: {refused}",
    )
    return {"diagnostics": diagnostics, "calibration_refusal": refused}


def _rollout_halt_collapse(
    api: Api,
    telemetry: TelemetryProbe,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    _heartbeat(api)
    cursor = newest_event_id(api)
    sequence = int(telemetry.wait().get("sample_seq", 0))
    api.json(
        "POST",
        "/api/models/rollout/start",
        {"model_path": model["name"], "operator_id": OPERATOR},
    )
    live = telemetry.wait(
        minimum_sequence=sequence + 3,
        mode="rollout",
    )
    _assert(live.get("torque_state") == "on", "rollout telemetry is not energized")
    live_status = api.json("GET", "/api/robot/status")
    _assert(
        live_status.get("active_model") == model["path"],
        "rollout did not expose active model",
    )
    _heartbeat(api)
    api.json("POST", "/api/models/rollout/stop")
    halted = wait_for(
        "Halt to hold the stopped rollout",
        lambda: api.json("GET", "/api/robot/status"),
        lambda status: status.get("mode") == "idle" and status.get("stop_state") == "halted",
        ASYNC_SECONDS,
    )
    _assert(halted.get("torque_state") == "on", "Halt did not keep torque on")
    _assert(isinstance(halted.get("halt"), dict), "Halt status has no hold detail")
    halt_events = events_after(api, cursor)
    _assert(any(event.get("type") == "safety.halt" for event in halt_events), "no Halt event")

    collapse_cursor = newest_event_id(api)
    api.json("POST", "/api/robot/collapse", {"reason": "installed E2E Collapse"})
    collapsed = wait_for(
        "Collapse to reach rest and turn torque off",
        lambda: api.json("GET", "/api/robot/status"),
        lambda status: status.get("stop_state") == "none",
        COLLAPSE_SECONDS,
    )
    _assert(collapsed.get("torque_state") == "off", "Collapse did not confirm torque off")
    _assert(not collapsed.get("torque_unconfirmed"), "Collapse left torque unconfirmed")
    _assert(not collapsed.get("estopped"), "Collapse unexpectedly latched E-stop")
    collapse_events = events_after(api, collapse_cursor)
    types = [event.get("type") for event in collapse_events]
    _assert("safety.collapse.start" in types, "Collapse did not record its start")
    _assert("safety.collapse.fallback" not in types, "Collapse used its torque-cut fallback")
    starts = [event for event in collapse_events if event.get("type") == "safety.collapse.start"]
    start_detail = (starts[-1].get("detail") or {}) if starts else {}
    distance = start_detail.get("distance_deg")
    _assert(
        isinstance(distance, (int, float)) and distance > 1.0,
        f"Collapse was vacuous; commanded distance was {distance!r}",
    )
    done = [event for event in collapse_events if event.get("type") == "safety.collapse.done"]
    _assert(bool(done), "Collapse never proved that rest was reached")
    detail = done[-1].get("detail") or {}
    _assert(detail.get("torque_off") is True, "Collapse done event lacks confirmed torque-off")
    _assert(not detail.get("fault"), f"Collapse done event carries a fault: {detail.get('fault')}")
    duration = detail.get("duration_s")
    _assert(
        isinstance(duration, (int, float)) and duration > 0.05,
        f"Collapse did not perform a measurable descent ({duration!r}s)",
    )
    error = detail.get("final_error_deg")
    _assert(isinstance(error, (int, float)) and error <= 3.0, "Collapse final error is too high")
    control = api.json("GET", "/api/control/state")
    _assert(control.get("safety_state") != "fault", "Collapse left a runtime fault latched")
    return {
        "halt_status": halted,
        "halt_events": halt_events,
        "collapse_status": collapsed,
        "collapse_events": collapse_events,
    }


def _evaluate(api: Api, model: Mapping[str, Any], backend: str, outputs: Path) -> dict[str, Any]:
    _heartbeat(api)
    api.json(
        "POST",
        "/api/eval/start",
        {
            "model_path": model["name"],
            "n_episodes": 2,
            "max_seconds": 0.15,
            "operator_id": OPERATOR,
        },
    )
    for number, success in enumerate((True, False), start=1):

        def awaiting_score(value: dict[str, Any], number: int = number) -> bool:
            return value.get("phase") == "awaiting_score" and value.get("episode") == number

        state = wait_for(
            f"evaluation trial {number} to await its score",
            lambda: api.json("GET", "/api/eval/state"),
            awaiting_score,
            ASYNC_SECONDS,
            failed=lambda value: (
                f"evaluation ended {value.get('phase')!r}: {value.get('error')}"
                if value.get("phase") in TERMINAL_EVAL
                else None
            ),
        )
        _assert(state.get("active") is True, "evaluation became inactive before scoring")
        _heartbeat(api)
        api.json(
            "POST",
            "/api/eval/score",
            {"operator_id": OPERATOR, "success": success},
        )
        if number == 1:
            reset = wait_for(
                "evaluation workspace-reset gate",
                lambda: api.json("GET", "/api/eval/state"),
                lambda value: value.get("phase") == "awaiting_reset",
                ASYNC_SECONDS,
                failed=lambda value: (
                    f"evaluation ended {value.get('phase')!r}: {value.get('error')}"
                    if value.get("phase") in TERMINAL_EVAL
                    else None
                ),
            )
            _assert(reset.get("scored") == 1, "first evaluation score was not durable")
            _heartbeat(api)
            api.json("POST", "/api/eval/continue", {"operator_id": OPERATOR})
    terminal = wait_for(
        "evaluation report publication",
        lambda: api.json("GET", "/api/eval/state"),
        lambda value: value.get("phase") in TERMINAL_EVAL and not value.get("active"),
        ASYNC_SECONDS,
    )
    _assert(terminal.get("phase") == "done", f"evaluation ended {terminal}")
    report_path = _relative_child(Path(str(terminal.get("report_path"))), outputs, "eval report")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"evaluation report is unreadable: {report_path}: {exc}") from exc
    _assert(report.get("source") == backend, "evaluation report names the wrong backend")
    _assert(report.get("phase") == "done" and not report.get("error"), "eval report is not clean")
    _assert(report.get("requested_episodes") == 2, "eval report requested count is wrong")
    _assert(report.get("scored") == 2 and report.get("n") == 2, "eval denominator is wrong")
    _assert(report.get("successes") == 1, "eval success count is wrong")
    _assert(report.get("success_rate") == 0.5, "eval success rate is wrong")
    _assert(report.get("ci95") == _wilson(1, 2), "eval Wilson interval is wrong")
    _assert(bool(report.get("label")), "eval report does not name the policy implementation")
    _assert(
        bool(report.get("privileged_state")) == (backend == "sim"),
        "eval report has the wrong privileged-state flag",
    )
    report["_path"] = str(report_path)
    return report


def _estop(api: Api, model: Mapping[str, Any]) -> dict[str, Any]:
    # A completed eval leaves a Halt holding.  Starting another registered
    # rollout deliberately takes over that hold; E-stop then has energized
    # motors to cut, rather than passing vacuously on an already-off arm.
    _heartbeat(api)
    cursor = newest_event_id(api)
    api.json(
        "POST",
        "/api/models/rollout/start",
        {"model_path": model["name"], "operator_id": OPERATOR},
    )
    wait_for(
        "rollout before E-stop",
        lambda: api.json("GET", "/api/robot/status"),
        lambda status: (
            status.get("mode") == "rollout"
            and status.get("torque_state") == "on"
            and isinstance(status.get("fps"), (int, float))
            and status["fps"] > 0
        ),
        ASYNC_SECONDS,
        failed=lambda status: (
            "rollout returned to idle before E-stop" if status.get("mode") == "idle" else None
        ),
    )
    api.json("POST", "/api/robot/estop")
    status = wait_for(
        "E-stop torque removal",
        lambda: api.json("GET", "/api/robot/status"),
        lambda value: value.get("estopped") is True and value.get("mode") == "idle",
        ASYNC_SECONDS,
    )
    _assert(status.get("torque_state") == "off", "E-stop did not confirm all torque off")
    _assert(not status.get("torque_unconfirmed"), "E-stop left torque unconfirmed")
    _assert(status.get("latch_persistence") == "saved", "E-stop latch was not durably saved")
    events = events_after(api, cursor)
    _assert(any(event.get("type") == "safety.estop" for event in events), "no E-stop event")
    # This is a fresh, isolated server and the harness set this E-stop itself;
    # it therefore owns the deliberate reset.  It never resets a preexisting latch.
    api.json(
        "POST",
        "/api/robot/estop/reset",
        {"operator_id": OPERATOR, "confirm": True},
    )
    reset = api.json("GET", "/api/robot/status")
    _assert(not reset.get("estopped"), "owned E-stop did not reset")
    _assert(reset.get("torque_state") == "off", "E-stop reset unexpectedly enabled torque")
    _assert(reset.get("latch_persistence") == "saved", "E-stop reset was not durably saved")
    return {"latched_status": status, "reset_status": reset, "events": events}


def _energize_for_shutdown(
    api: Api,
    telemetry: TelemetryProbe,
    model: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Leave a sampled rollout energized so SIGTERM must perform a real Collapse."""

    _heartbeat(api)
    cursor = newest_event_id(api)
    sequence = int(telemetry.wait().get("sample_seq", 0))
    api.json(
        "POST",
        "/api/models/rollout/start",
        {"model_path": model["name"], "operator_id": OPERATOR},
    )
    live = telemetry.wait(
        minimum_sequence=sequence + 3,
        mode="rollout",
    )
    _assert(live.get("torque_state") == "on", "shutdown rollout is not energized")
    _heartbeat(api)
    return cursor, live


def verify_energized_shutdown(data_dir: Path, cursor: int) -> dict[str, Any]:
    events = persisted_events_after(data_dir, cursor)
    starts = [
        event
        for event in events
        if event.get("type") == "safety.collapse.start"
        and (event.get("detail") or {}).get("reason") == "service shutdown"
    ]
    dones = [event for event in events if event.get("type") == "safety.collapse.done"]
    fallbacks = [event for event in events if event.get("type") == "safety.collapse.fallback"]
    _assert(bool(starts), "energized server shutdown never started its service Collapse")
    distance = (starts[-1].get("detail") or {}).get("distance_deg")
    _assert(
        isinstance(distance, (int, float)) and distance > 1.0,
        f"energized shutdown Collapse was vacuous ({distance!r} deg)",
    )
    _assert(not fallbacks, f"energized server shutdown used fallback: {fallbacks}")
    _assert(bool(dones), "energized server shutdown did not finish at rest")
    detail = dones[-1].get("detail") or {}
    _assert(detail.get("torque_off") is True, "shutdown did not confirm torque off")
    _assert(not detail.get("fault"), f"shutdown Collapse carried a fault: {detail.get('fault')}")
    return {"events": events, "start": starts[-1], "done": dones[-1]}


def run_mode(
    backend: str,
    *,
    python: Path,
    so101: Path,
    venv: Path,
    artifact: WheelArtifact,
    work: Path,
    evidence_root: Path,
    ledger_secrets: list[str],
    ui_e2e: bool = False,
    ui_cwd: Path | None = None,
) -> dict[str, Any]:
    mode_root = work / backend
    mode_root.mkdir(parents=True)
    evidence = evidence_root / backend
    _secure_directory(evidence)
    e2e_token = secrets.token_urlsafe(32)
    ledger_secrets.append(e2e_token)
    env = clean_env(mode_root, backend, e2e_token=e2e_token)
    probe_env = dict(env)
    probe_env["SO101_E2E_TOKEN"] = ""
    ledger = WorkflowLedger(backend)
    probe = probe_install(python, venv, artifact, mode_root, backend, probe_env, evidence)
    _run(
        [str(so101), "info"],
        cwd=mode_root,
        env=probe_env,
        log=evidence / "so101-info.log",
        timeout=120,
    )

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = evidence / "server.log"
    _write_text(server_log, "")
    api = Api(
        base,
        evidence / "http.jsonl",
        expected_backend=backend,
        expected_version=artifact.version,
        e2e_token=e2e_token,
    )
    telemetry = TelemetryProbe(
        python,
        base,
        cwd=mode_root,
        env=env,
        evidence=evidence,
    )
    shutdown_cursor: int | None = None
    summary: dict[str, Any] | None = None
    stops_payload: dict[str, Any] = {}
    with server_log.open("wb") as output:
        server = subprocess.Popen(
            [str(so101), "serve", "--host", "127.0.0.1", "--port", str(port)],
            cwd=mode_root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        server_monitor = DescendantMonitor(server)
        try:
            health = wait_for_server(api, server, SERVER_START_SECONDS)
            # The only gate between read-only startup checks and every mutation below.
            status, _control = authorize_fresh_rehearsal(api, backend)
            _assert(health.get("backend") == backend, "health endpoint names the wrong backend")
            served_ui = verify_served_ui(api, artifact)
            ledger.complete("identity")

            if ui_e2e:
                if backend != "mock":
                    raise GateError("the installed UI E2E hook runs only on the mock backend")
                run_ui_hook(
                    base=base,
                    e2e_token=e2e_token,
                    expected_version=artifact.version,
                    cwd=(ui_cwd or Path.cwd()).resolve(),
                    evidence=evidence,
                )
                # A browser suite is allowed to mutate its reserved dataset,
                # but it must hand the server back idle and unlatched before
                # this harness proceeds with its separate namespace.
                api.lock_mutations()
                status, _control = authorize_fresh_rehearsal(api, backend)

            preflight = api.json("POST", "/api/preflight?arm=both&save=true")
            _assert(preflight.get("backend") == backend, "preflight names the wrong backend")
            _assert(preflight.get("rehearsal") is True, "preflight claimed hardware evidence")
            _assert(
                preflight.get("ready_for_hardware") is False,
                "rehearsal preflight claimed the system was ready for hardware",
            )
            _write_json(evidence / "preflight.json", preflight)
            ledger.complete("preflight")
            sim_contracts = _sim_contracts(api) if backend == "sim" else None
            if sim_contracts is not None:
                _write_json(evidence / "sim-contracts.json", sim_contracts)
                ledger.complete("sim_diagnostics")
                ledger.complete("sim_calibration_refusal")

            repo_id, dataset, qc, action_evidence = _dataset_workflow(
                backend,
                api,
                telemetry,
                so101,
                mode_root,
                env,
                evidence,
            )
            ledger.complete("record")
            ledger.complete("inspect_label_qc")
            manual_control = _manual_control_contracts(api)
            _write_json(evidence / "manual-control-contracts.json", manual_control)
            ledger.complete("deadman")
            ledger.complete("watchdog_halt")
            ledger.complete("lease_expiry")
            job, model = _train_and_find_model(backend, api, repo_id, mode_root / "outputs")
            _write_json(evidence / "training-job.json", job)
            ledger.complete("train")
            stops = _rollout_halt_collapse(api, telemetry, model)
            ledger.complete("rollout")
            ledger.complete("halt")
            ledger.complete("collapse")
            report = _evaluate(api, model, backend, mode_root / "outputs")
            ledger.complete("evaluate")
            estop = _estop(api, model)
            ledger.complete("estop")
            artifacts = _copy_artifacts(evidence, mode_root, dataset, qc, model, report)
            stops_payload = stops | {"estop": estop}
            shutdown_cursor, shutdown_live = _energize_for_shutdown(api, telemetry, model)
            summary = {
                "backend": backend,
                "rehearsal": True,
                "health": health,
                "initial_status": status,
                "probe": probe,
                "served_ui": served_ui,
                "ui_hook": shlex.join(UI_E2E_COMMAND) if ui_e2e else None,
                "preflight_overall": preflight.get("overall"),
                "dataset": repo_id,
                "dataset_episodes": dataset.get("num_episodes"),
                "dataset_qc": qc.get("overall"),
                "action_evidence": action_evidence,
                "manual_control": {
                    "deadman_rejected": True,
                    "watchdog_halted": True,
                    "lease_expired_to_halt": True,
                },
                "sim_contracts": sim_contracts,
                "training_status": job.get("status"),
                "model": model,
                "evaluation": {
                    key: report.get(key)
                    for key in (
                        "source",
                        "label",
                        "privileged_state",
                        "successes",
                        "n",
                        "success_rate",
                        "ci95",
                        "phase",
                    )
                },
                "stops": {
                    "halt": "held with torque on",
                    "collapse": "rest reached without fallback; torque off confirmed",
                    "estop": "latched; torque off confirmed; owned latch reset",
                    "energized_shutdown": {
                        "started_from_mode": shutdown_live.get("mode"),
                        "started_with_torque": shutdown_live.get("torque_state"),
                    },
                },
                "artifact_hashes": artifacts,
            }
        finally:
            try:
                outcome = stop_process(server, known_descendants=server_monitor.finish())
            except Exception as exc:
                _write_text(evidence / "server-cleanup.txt", f"FAILED: {exc}\n")
                raise
            else:
                _write_text(evidence / "server-cleanup.txt", outcome + "\n")
            finally:
                output.flush()
                os.fsync(output.fileno())
                redacted = redact_secret_tree(evidence, e2e_token)
                if redacted:
                    _write_json(evidence / "token-redactions.json", {"redacted": redacted})
                    raise GateError(
                        "ephemeral E2E token reached evidence and was redacted: "
                        + ", ".join(redacted)
                    )
    if summary is None or shutdown_cursor is None:
        raise GateError(f"{backend} workflow ended without energized-shutdown evidence")
    shutdown = verify_energized_shutdown(mode_root / "data", shutdown_cursor)
    ledger.complete("energized_shutdown")
    ledger.verify_complete()
    stops_payload["energized_shutdown"] = shutdown
    summary["stops"]["energized_shutdown"]["verified"] = True
    summary["workflow_stages"] = ledger.completed
    _write_json(evidence / "stops.json", stops_payload)
    _write_json(evidence / "summary.json", summary)
    store = mode_root / "data" / "control_plane.sqlite3"
    _copy_evidence_file(store, evidence / "control_plane.sqlite3")
    redacted = redact_secret_tree(evidence, e2e_token)
    if redacted:
        raise GateError(
            "ephemeral E2E token reached evidence and was redacted: " + ", ".join(redacted)
        )
    assert_secret_absent(evidence, e2e_token)
    return summary


def run_gate(
    dist_dir: Path,
    evidence_dir: Path,
    backends: Sequence[str],
    *,
    keep_work: bool = False,
    ui_e2e: bool = False,
    ui_cwd: Path | None = None,
) -> int:
    evidence = prepare_evidence(evidence_dir)
    ledger_secrets = sensitive_environment_values()
    work = Path(tempfile.mkdtemp(prefix="so101-installed-e2e-"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": time.time(),
        "rehearsal_only": True,
        "release_ready": False,
        "hardware_touched": False,
        "backends": list(backends),
    }
    exit_code = 1
    try:
        if ui_e2e and "mock" not in backends:
            raise GateError("--ui-e2e needs the mock backend")
        source, snapshot = snapshot_ui_wheel(dist_dir, work / "wheel-snapshot")
        selected = select_ui_wheel(snapshot.parent)
        source_path = str(source)
        artifact = copy_wheel_to_evidence(selected, evidence)
        result["wheel"] = artifact.public()
        _write_json(
            evidence / "wheel.json",
            {**artifact.public(), "selected_from": source_path, "installed_from_evidence": True},
        )
        _write_text(evidence / "wheel.sha256", f"{artifact.sha256}  {artifact.path.name}\n")
        _write_json(
            evidence / "environment.json",
            {
                "platform": platform.platform(),
                "python": sys.version,
                "python_executable": sys.executable,
                "backends": list(backends),
                "wheel": artifact.public(),
            },
        )
        venv = work / "venv"
        command_dir = work / "command-workdir"
        command_dir.mkdir()
        installer_home = work / "installer-home"
        _secure_directory(installer_home)
        env = install_env(installer_home)
        _run(
            [sys.executable, "-m", "venv", str(venv)],
            cwd=command_dir,
            env=env,
            log=evidence / "create-venv.log",
            timeout=300,
        )
        python = venv / "bin" / "python"
        so101 = venv / "bin" / "so101"
        _assert(_sha256(artifact.path) == artifact.sha256, "evidence wheel changed before install")
        install_target = str(artifact.path) + ("[sim]" if "sim" in backends else "")
        _run(
            [str(python), "-m", "pip", "install", "--no-input", install_target],
            cwd=command_dir,
            env=env,
            log=evidence / "install.log",
            timeout=1200,
            redact_values=ledger_secrets,
        )
        _assert(_sha256(artifact.path) == artifact.sha256, "evidence wheel changed during install")
        _run(
            [str(so101), "--help"],
            cwd=command_dir,
            env=env,
            log=evidence / "so101-help.log",
            timeout=120,
        )
        freeze = _run(
            [str(python), "-m", "pip", "freeze", "--all"],
            cwd=command_dir,
            env=env,
            log=evidence / "packages.log",
            timeout=120,
            redact_values=ledger_secrets,
        )
        result["packages_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
        modes: dict[str, Any] = {}
        result["modes"] = modes
        for backend in backends:
            modes[backend] = run_mode(
                backend,
                python=python,
                so101=so101,
                venv=venv,
                artifact=artifact,
                work=work,
                evidence_root=evidence,
                ledger_secrets=ledger_secrets,
                ui_e2e=ui_e2e and backend == "mock",
                ui_cwd=ui_cwd,
            )
        leftovers = sorted(path.name for path in command_dir.iterdir())
        _assert(
            not leftovers,
            f"installed commands wrote into their working directory: {leftovers}",
        )
        result["status"] = "passed"
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        detail = _sanitize_text(f"{type(exc).__name__}: {exc}", ledger_secrets)
        result.update(status="failed", error=detail)
        _write_text(
            evidence / "failure.log", _sanitize_text(traceback.format_exc(), ledger_secrets)
        )
        print(f"installed_e2e: FAIL: {detail}\nevidence: {evidence}", file=sys.stderr)
    finally:
        if keep_work:
            result["work_cleanup"] = {"removed": False, "path": str(work)}
            print(f"installed_e2e: kept work directory {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
            removed = not work.exists()
            result["work_cleanup"] = {"removed": removed, "path": str(work)}
            if not removed:
                cleanup_error = f"temporary work directory could not be removed: {work}"
                previous = result.get("error")
                result.update(
                    status="failed",
                    error=f"{previous}; {cleanup_error}" if previous else cleanup_error,
                )
                exit_code = 1
        redacted_paths: set[str] = set()
        for secret in ledger_secrets:
            redacted_paths.update(redact_secret_tree(evidence, secret))
        if redacted_paths:
            result.update(
                status="failed",
                error="sensitive environment value reached evidence and was redacted",
            )
            exit_code = 1
            _write_json(
                evidence / "secret-redactions.json",
                {"redacted": sorted(redacted_paths)},
            )
        result["finished_at"] = time.time()
        _write_json(evidence / "result.json", result)
        final_redactions: set[str] = set()
        for secret in ledger_secrets:
            final_redactions.update(redact_secret_tree(evidence, secret))
        if final_redactions:
            result.update(
                status="failed",
                error="sensitive value reached final evidence and was redacted",
            )
            exit_code = 1
            _write_json(
                evidence / "secret-redactions.json",
                {"redacted": sorted(redacted_paths | final_redactions)},
            )
            _write_json(evidence / "result.json", result)
        try:
            _secure_tree(evidence)
            write_evidence_hashes(evidence)
        except Exception as exc:
            previous = result.get("error")
            result.update(
                status="failed",
                error=(
                    f"{previous}; evidence hashing failed: {exc}"
                    if previous
                    else f"evidence hashing failed: {exc}"
                ),
            )
            exit_code = 1
            print(f"installed_e2e: FAIL: evidence hashing failed: {exc}", file=sys.stderr)
            _write_json(evidence / "result.json", result)
            # Best effort only after the gate is irreversibly red.
            try:
                _secure_tree(evidence)
                write_evidence_hashes(evidence)
            except Exception:
                pass
        if result.get("status") == "passed":
            ran = "/".join(str(item) for item in result.get("backends", []))
            print(f"installed_e2e: {ran} installed-artifact gate passed; evidence: {evidence}")
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dist_dir", type=Path, help="directory containing exactly one wheel")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="new or empty directory that will retain logs and artifacts",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "sim", "both"),
        default="both",
        help="software backend(s) to exercise (default: both)",
    )
    parser.add_argument("--keep-work", action="store_true", help="keep the throwaway venv/data")
    parser.add_argument(
        "--ui-e2e",
        action="store_true",
        help="run the fixed npm run test:e2e contract against the fresh installed mock server",
    )
    parser.add_argument(
        "--ui-cwd",
        type=Path,
        default=Path.cwd() / "frontend",
        help="frontend package directory for --ui-e2e (default: ./frontend)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    backends = ("mock", "sim") if args.backend == "both" else (args.backend,)
    previous_umask = os.umask(0o077)
    try:
        return run_gate(
            args.dist_dir,
            args.evidence_dir,
            backends,
            keep_work=args.keep_work,
            ui_e2e=args.ui_e2e,
            ui_cwd=args.ui_cwd,
        )
    except (GateError, ValueError) as exc:
        # prepare_evidence can fail before run_gate has somewhere safe to log.
        print(f"installed_e2e: FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    raise SystemExit(main())
