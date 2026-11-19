#!/usr/bin/env python3
"""Check worker dependency resolution with pip's non-installing dry run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIREMENT_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([A-Za-z0-9_,.-]+)\])?"
    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>).+)?\Z"
)
FORBIDDEN_DISTRIBUTIONS = frozenset(
    {"so101", "pyrealsense2", "pyrealsense2-macosx", "feetech-servo-sdk", "pyserial"}
)
FORBIDDEN_EXTRAS = frozenset({"robot", "realsense", "feetech", "serial"})


class DependencyError(ValueError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def validate_requirements(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise DependencyError(f"cannot read requirements file: {exc}") from exc
    requirements: list[str] = []
    for number, original in enumerate(text.splitlines(), start=1):
        line = original.split("#", 1)[0].strip()
        if not line:
            continue
        if any(token in line for token in ("/", "\\", "@", ";", "--")):
            raise DependencyError(
                f"line {number}: URLs, paths, markers and pip options are prohibited"
            )
        match = REQUIREMENT_RE.fullmatch(line)
        if match is None:
            raise DependencyError(f"line {number}: unsupported requirement syntax")
        distribution = match.group(1).lower().replace("_", "-")
        extras = {item.lower() for item in (match.group(2) or "").split(",") if item}
        if distribution in FORBIDDEN_DISTRIBUTIONS:
            raise DependencyError(f"line {number}: prohibited worker distribution {distribution!r}")
        forbidden = extras & FORBIDDEN_EXTRAS
        if forbidden:
            raise DependencyError(f"line {number}: prohibited worker extras {sorted(forbidden)}")
        requirements.append(line)
    if not requirements:
        raise DependencyError("requirements file has no requirements")
    return requirements


def build_command(
    requirements: Path,
    report: Path,
    *,
    platform_name: str | None = None,
    python_version: str | None = None,
    abi: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--no-input",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--report",
        str(report),
    ]
    if platform_name:
        command.extend(("--platform", platform_name, "--implementation", "cp"))
    if python_version:
        command.extend(("--python-version", python_version))
    if abi:
        command.extend(("--abi", abi))
    command.extend(("--requirement", str(requirements)))
    return command


def _safe_report(path: Path) -> list[dict[str, Any]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    installs = report.get("install", []) if isinstance(report, dict) else []
    packages = []
    for item in installs:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        archive = item.get("download_info", {}).get("archive_info", {})
        hashes = archive.get("hashes", {}) if isinstance(archive, dict) else {}
        if isinstance(metadata, dict):
            packages.append(
                {
                    "name": metadata.get("name"),
                    "version": metadata.get("version"),
                    "sha256": hashes.get("sha256") if isinstance(hashes, dict) else None,
                }
            )
    return sorted(packages, key=lambda item: (str(item["name"]), str(item["version"])))


def _diagnostic(stderr: str) -> str | None:
    # Avoid persisting embedded URL credentials if pip printed an index address.
    cleaned = re.sub(r"(https?://)[^\s/@:]+(?::[^\s/@]*)?@", r"\1<redacted>@", stderr)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines[-20:])[:8000] or None


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    default_requirements = Path(__file__).resolve().parent.parent / "requirements-worker.txt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=default_requirements)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--print-command", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--platform")
    parser.add_argument("--python-version")
    parser.add_argument("--abi")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        requirements = validate_requirements(args.requirements)
    except DependencyError as exc:
        parser.error(str(exc))

    placeholder = Path("<PIP_REPORT.json>")
    shown = build_command(
        args.requirements,
        placeholder,
        platform_name=args.platform,
        python_version=args.python_version,
        abi=args.abi,
    )
    if not args.run:
        print(shlex.join(shown))
        return 0

    with tempfile.TemporaryDirectory(prefix="so101-thor-deps-") as raw_directory:
        report_path = Path(raw_directory) / "pip-report.json"
        command = build_command(
            args.requirements,
            report_path,
            platform_name=args.platform,
            python_version=args.python_version,
            abi=args.abi,
        )
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
        payload = {
            "schema_version": 1,
            "captured_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "requirements_sha256": hashlib.sha256(args.requirements.read_bytes()).hexdigest(),
            "requirements": requirements,
            "platform": args.platform,
            "python_version": args.python_version,
            "abi": args.abi,
            "dry_run": True,
            "wheel_only": True,
            "ok": result.returncode == 0,
            "packages": _safe_report(report_path),
            "diagnostic": _diagnostic(result.stderr),
        }
    content = canonical_json_bytes(payload)
    if args.output:
        _atomic_write(args.output, content)
    else:
        sys.stdout.buffer.write(content)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
