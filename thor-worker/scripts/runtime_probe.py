#!/usr/bin/env python3
"""Record the exact host/container compatibility tuple without probing devices."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
IMAGE_RE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class ProbeError(ValueError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _text_file(path: str, *, strip_nul: bool = False) -> str | None:
    try:
        value = Path(path).read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return None
    if strip_nul:
        value = value.replace("\x00", "")
    value = value.strip()
    return value or None


def _os_release() -> dict[str, str]:
    allowed = {"ID", "VERSION_ID", "PRETTY_NAME"}
    values: dict[str, str] = {}
    raw = _text_file("/etc/os-release")
    if raw is None:
        return values
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in allowed:
            values[key.lower()] = value.strip().strip('"')
    return values


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _command_line(command: list[str]) -> str | None:
    output = _command_output(command)
    if output is None:
        return None
    lines = output.splitlines()
    return lines[0].strip() if lines and lines[0].strip() else None


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _cuda_version() -> str | None:
    path = Path("/usr/local/cuda/version.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict):
        cuda = value.get("cuda")
        if isinstance(cuda, dict) and isinstance(cuda.get("version"), str):
            return cuda["version"]
        if isinstance(value.get("version"), str):
            return value["version"]
    output = _command_output(["nvcc", "--version"])
    if output is None:
        return None
    match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)", output)
    return match.group(1) if match else output.splitlines()[-1].strip()


def _dpkg_version(package: str) -> str | None:
    return _command_line(["dpkg-query", "-W", "-f=${Version}", package])


def _memory_bytes() -> int | None:
    raw = _text_file("/proc/meminfo")
    if raw is None:
        return None
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "MemTotal:":
            try:
                return int(fields[1]) * 1024
            except ValueError:
                return None
    return None


def _torch_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "version": None,
        "cuda_build": None,
        "cuda_available": False,
        "cudnn": None,
        "device_names": [],
        "compute_capabilities": [],
    }
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return facts
    torch_version = getattr(torch, "__version__", None)
    facts["version"] = str(torch_version) if torch_version is not None else None
    version_module = getattr(torch, "version", None)
    cuda_build = getattr(version_module, "cuda", None)
    facts["cuda_build"] = str(cuda_build) if cuda_build is not None else None
    try:
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if facts["cuda_available"]:
            count = int(torch.cuda.device_count())
            facts["device_names"] = [
                str(torch.cuda.get_device_name(index)) for index in range(count)
            ]
            facts["compute_capabilities"] = [
                ".".join(str(part) for part in torch.cuda.get_device_capability(index))
                for index in range(count)
            ]
    except Exception:
        facts["cuda_available"] = False
        facts["device_names"] = []
        facts["compute_capabilities"] = []
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:
        cudnn = None
    facts["cudnn"] = str(cudnn) if cudnn is not None else None
    return facts


def validate_identity(source_commit: str, container_image: str) -> None:
    if not SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise ProbeError("source commit must be exactly 40 lowercase hexadecimal characters")
    if source_commit == "0" * 40:
        raise ProbeError("source commit may not use the all-zero placeholder")
    if container_image != "HOST-NATIVE" and not IMAGE_RE.fullmatch(container_image):
        raise ProbeError("container image must be HOST-NATIVE or pinned with @sha256:<64 hex>")
    if container_image != "HOST-NATIVE" and container_image.endswith("0" * 64):
        raise ProbeError("container image may not use the all-zero digest placeholder")
    baked_commit = os.environ.get("THOR_WORKER_SOURCE_COMMIT")
    if baked_commit is not None and baked_commit != source_commit:
        raise ProbeError("source commit does not match the revision baked into this worker image")


def collect(
    *,
    source_commit: str,
    container_image: str,
    target: str,
    wheelhouse_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    validate_identity(source_commit, container_image)
    if wheelhouse_manifest_sha256 is not None and (
        not DIGEST_RE.fullmatch(wheelhouse_manifest_sha256)
        or wheelhouse_manifest_sha256 == "0" * 64
    ):
        raise ProbeError("wheelhouse manifest must be a non-placeholder SHA-256 digest")
    if wheelhouse_manifest_sha256 is not None and container_image != "HOST-NATIVE":
        baked_mode = os.environ.get("THOR_WORKER_QUALIFICATION_ONLY")
        baked_digest = os.environ.get("THOR_WORKER_WHEELHOUSE_MANIFEST_SHA256")
        if baked_mode != "0" or baked_digest != wheelhouse_manifest_sha256:
            raise ProbeError(
                "locked qualification requires the same wheelhouse digest baked into the image"
            )
    torch = _torch_facts()
    l4t = _text_file("/etc/nv_tegra_release")
    jetpack = _dpkg_version("nvidia-jetpack")
    tensorrt = _module_version("tensorrt") or _dpkg_version("tensorrt")
    cudnn = torch["cudnn"] or _dpkg_version("libcudnn9") or _dpkg_version("libcudnn8")
    runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cuda": _cuda_version() or torch["cuda_build"],
        "cudnn": cudnn,
        "tensorrt": tensorrt,
        "torch": torch,
        "lerobot": _package_version("lerobot"),
        "mujoco": _package_version("mujoco"),
        "numpy": _package_version("numpy"),
    }
    host = {
        "architecture": platform.machine(),
        "machine_model": _text_file("/proc/device-tree/model", strip_nul=True),
        "kernel": platform.release(),
        "os_release": _os_release(),
        "l4t_release": l4t.splitlines()[0] if l4t else None,
        "jetpack": jetpack,
        "memory_bytes": _memory_bytes(),
    }
    required: dict[str, object] = {
        "host.architecture": host["architecture"],
        "host.kernel": host["kernel"],
        "runtime.python": runtime["python"],
        "runtime.torch": torch["version"],
        "runtime.lerobot": runtime["lerobot"],
    }
    if target == "thor":
        required.update(
            {
                "host.machine_model": host["machine_model"],
                "host.l4t_release": host["l4t_release"],
                "host.jetpack": host["jetpack"],
                "runtime.cuda": runtime["cuda"],
                "runtime.cudnn": runtime["cudnn"],
                "runtime.tensorrt": runtime["tensorrt"],
            }
        )
        if host["architecture"] != "aarch64":
            required["host.architecture(aarch64)"] = None
        if not host["machine_model"] or "thor" not in str(host["machine_model"]).casefold():
            required["host.machine_model(Jetson Thor)"] = None
        if not torch["cuda_available"]:
            required["runtime.torch.cuda_available"] = None
    elif target == "g15":
        required.update(
            {
                "runtime.cuda": runtime["cuda"],
                "runtime.cudnn": runtime["cudnn"],
            }
        )
        if host["architecture"] not in {"x86_64", "amd64"}:
            required["host.architecture(x86_64)"] = None
        if not torch["cuda_available"]:
            required["runtime.torch.cuda_available"] = None
    missing = sorted(key for key, value in required.items() if value is None or value == "")
    return {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "target": target,
        "source_commit": source_commit,
        "container_image": container_image,
        "qualification": {
            "mode": "locked" if wheelhouse_manifest_sha256 else "qualification-only",
            "wheelhouse_manifest_sha256": wheelhouse_manifest_sha256,
        },
        "host": host,
        "runtime": runtime,
        "complete_for_target": not missing,
        "missing_required": missing,
    }


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--target", choices=("thor", "g15", "other"), default="thor")
    parser.add_argument("--wheelhouse-manifest-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = collect(
            source_commit=args.source_commit,
            container_image=args.container_image,
            target=args.target,
            wheelhouse_manifest_sha256=args.wheelhouse_manifest_sha256,
        )
    except ProbeError as exc:
        parser.error(str(exc))
    content = canonical_json_bytes(result)
    if args.output:
        _atomic_write(args.output, content)
    else:
        sys.stdout.buffer.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
