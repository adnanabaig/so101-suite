#!/usr/bin/env python3
"""Fail-closed validation for the source/wheel selected by ``install.sh``.

The workstation installer must not silently install a wheel from another source
revision, a stale/altered web build, or an API-only artifact.  This helper uses
only the standard library so it can run before the virtual environment exists.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tomllib
import zipfile
from collections import Counter
from collections.abc import Mapping
from email.message import Message
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

UI_STAMP = "so101-ui-stamp.json"
UI_SOURCE_DIRS = ("src", "public")
UI_SOURCE_FILES = (
    "index.html",
    "package.json",
    "package-lock.json",
    "vite.config.ts",
    "build-stamp.ts",
    "tsconfig.json",
    "tsconfig.node.json",
)
SKIP_NAMES = frozenset({".DS_Store"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class InputError(RuntimeError):
    """The selected installation artifact is not the requested workstation."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files_under(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def _ui_sources_digest(frontend: Path) -> str | None:
    if not (frontend / "src").is_dir():
        return None
    source_files: dict[str, bytes] = {}
    for directory in UI_SOURCE_DIRS:
        source_files.update(
            {
                f"{directory}/{name}": data
                for name, data in _files_under(frontend / directory).items()
            }
        )
    for name in UI_SOURCE_FILES:
        path = frontend / name
        if path.is_file() and path.name not in SKIP_NAMES:
            source_files[name] = path.read_bytes()
    hashes = {name: _sha256(data) for name, data in source_files.items()}
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(f"{name}\0{hashes[name]}\n".encode())
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    head = result.stdout.strip()
    return head if GIT_SHA.fullmatch(head) else None


def _parse_stamp(data: bytes) -> dict[str, Any]:
    try:
        stamp = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{UI_STAMP} is not valid JSON: {exc}") from None
    if not isinstance(stamp, dict) or set(stamp) != {"format", "sources", "git_head", "files"}:
        raise InputError(f"{UI_STAMP} has unexpected or missing fields")
    if stamp["format"] != 1:
        raise InputError(f"{UI_STAMP} is not format 1")
    sources, head, manifest = stamp["sources"], stamp["git_head"], stamp["files"]
    if not isinstance(sources, str) or not SHA256.fullmatch(sources):
        raise InputError(f"{UI_STAMP} sources is not a SHA-256")
    if head is not None and (not isinstance(head, str) or not GIT_SHA.fullmatch(head)):
        raise InputError(f"{UI_STAMP} git_head is not a commit id")
    if not isinstance(manifest, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) and SHA256.fullmatch(digest)
        for name, digest in manifest.items()
    ):
        raise InputError(f"{UI_STAMP} files is not a path-to-SHA-256 map")
    return stamp


def _validate_ui(
    files: Mapping[str, bytes],
    *,
    root: Path,
    label: str,
    allow_no_ui: bool,
    expected_git_head: str | None,
) -> None:
    if not files:
        if allow_no_ui:
            return
        raise InputError(
            f"{label} has no built web UI; rebuild it with `make frontend-deps build-ui wheel` "
            "(`make ci-wheel` runs the release check), or for an intentional API/CLI-only "
            "development install use --no-launcher --allow-no-ui"
        )
    if "index.html" not in files:
        raise InputError(f"{label} web UI has no index.html")
    stamp_data = files.get(UI_STAMP)
    if stamp_data is None:
        raise InputError(f"{label} web UI has no {UI_STAMP}")
    stamp = _parse_stamp(stamp_data)
    manifest: dict[str, str] = stamp["files"]
    actual = {name: _sha256(data) for name, data in files.items() if name != UI_STAMP}
    missing = sorted(set(manifest) - set(actual))
    extra = sorted(set(actual) - set(manifest))
    changed = sorted(name for name in set(actual) & set(manifest) if actual[name] != manifest[name])
    if missing or extra or changed:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing[:3]))
        if extra:
            detail.append("unlisted " + ", ".join(extra[:3]))
        if changed:
            detail.append("altered " + ", ".join(changed[:3]))
        raise InputError(f"{label} web UI does not match its build stamp ({'; '.join(detail)})")

    sources = _ui_sources_digest(root / "frontend")
    if sources is not None and stamp["sources"] != sources:
        raise InputError(f"{label} web UI was built from other frontend sources, so it is stale")
    head = expected_git_head or _git_head(root)
    if head is not None and stamp["git_head"] != head:
        built = stamp["git_head"] or "an unknown revision"
        raise InputError(f"{label} web UI was built at {built}, not source revision {head}")


def _wheel_files(path: Path) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            if len(names) != len(set(names)):
                raise InputError(f"{path} contains duplicate file names")
            for name in names:
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise InputError(f"{path} contains an unsafe path: {name}")
            return {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile) as exc:
        raise InputError(f"{path}: it is not a readable wheel ({exc})") from None


def _normal_requirement(value: str) -> str:
    requirement, separator, marker = value.partition(";")
    compact = re.sub(r"\s+", "", requirement)
    parsed = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[([^]]+)\])?(.*)", compact)
    if parsed is not None:
        name, extras, constraint = parsed.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        normalized_extras = ""
        if extras:
            normalized_extras = (
                "[" + ",".join(sorted(item.lower() for item in extras.split(","))) + "]"
            )
        if constraint and not constraint.startswith("@"):
            constraint = ",".join(sorted(constraint.split(",")))
        compact = name + normalized_extras + constraint
    if not separator:
        return compact
    return f"{compact};{_normal_marker(marker)}"


def _normal_extra(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _normal_specifiers(value: str) -> str:
    return ",".join(sorted(re.sub(r"\s+", "", value).split(",")))


def _normal_marker(value: str) -> str:
    clauses = re.split(r"\s+and\s+", value.replace('"', "'").strip())
    normalized = [re.sub(r"[()\s]+", "", clause) for clause in clauses]
    return "and".join(sorted(filter(None, normalized)))


def _project_contract(
    root: Path, selected_extras: tuple[str, ...]
) -> tuple[Counter[str], Counter[str], dict[str, str], str | None]:
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        optional = project.get("optional-dependencies", {})
        declared = {
            _normal_extra(str(extra)): dependencies for extra, dependencies in optional.items()
        }
        missing = sorted(set(selected_extras) - set(declared))
        if missing:
            raise InputError(
                "selected suite extras are not declared by this source: " + ", ".join(missing)
            )
        requires = Counter(
            _normal_requirement(str(item)) for item in project.get("dependencies", [])
        )
        for extra, dependencies in declared.items():
            for item in dependencies:
                requirement, separator, marker = str(item).partition(";")
                clauses = [f"extra == '{extra}'"]
                if separator:
                    clauses.append(marker)
                normalized = (
                    _normal_requirement(requirement) + ";" + _normal_marker(" and ".join(clauses))
                )
                requires[normalized] += 1
        provides = Counter(declared.keys())
        scripts = {
            str(name): str(target).strip() for name, target in project.get("scripts", {}).items()
        }
        requires_python = project.get("requires-python")
        if requires_python is not None:
            requires_python = _normal_specifiers(str(requires_python))
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise InputError(
            f"cannot read the package contract from {root / 'pyproject.toml'}: {exc}"
        ) from None
    return requires, provides, scripts, requires_python


def _entry_points(data: bytes, wheel: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None, delimiters=("=",), strict=True)
    parser.optionxform = str
    try:
        parser.read_string(data.decode("utf-8"))
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise InputError(f"{wheel} has an invalid entry_points.txt ({exc})") from None
    if parser.sections() != ["console_scripts"]:
        raise InputError(f"{wheel} entry_points.txt has unexpected executable groups")
    return {name: target.strip() for name, target in parser.items("console_scripts")}


def _validate_record(files: Mapping[str, bytes], record_name: str, wheel: Path) -> None:
    try:
        rows = list(csv.reader(io.StringIO(files[record_name].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise InputError(f"{wheel} has an invalid RECORD ({exc})") from None
    if any(len(row) != 3 for row in rows):
        raise InputError(f"{wheel} RECORD has a row without path, hash and size")
    paths = [row[0] for row in rows]
    if len(paths) != len(set(paths)) or set(paths) != set(files):
        raise InputError(f"{wheel} RECORD does not list exactly the wheel files")
    for path, encoded_hash, size in rows:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise InputError(f"{wheel} RECORD contains an unsafe path: {path}")
        if path == record_name:
            if encoded_hash or size:
                raise InputError(f"{wheel} RECORD must leave its own hash and size empty")
            continue
        expected = base64.urlsafe_b64encode(hashlib.sha256(files[path]).digest()).rstrip(b"=")
        if encoded_hash != "sha256=" + expected.decode("ascii") or size != str(len(files[path])):
            raise InputError(f"{wheel} RECORD has the wrong hash or size for {path}")


def _validate_wheel_metadata(
    root: Path,
    metadata: Message,
    files: Mapping[str, bytes],
    dist_info: str,
    wheel: Path,
    selected_extras: tuple[str, ...],
) -> None:
    expected_requires, expected_provides, expected_scripts, expected_python = _project_contract(
        root, selected_extras
    )
    actual_requires = Counter(
        _normal_requirement(value) for value in metadata.get_all("Requires-Dist", [])
    )
    actual_provides = Counter(
        _normal_extra(value) for value in metadata.get_all("Provides-Extra", [])
    )
    if actual_requires != expected_requires:
        raise InputError(f"{wheel} does not exactly match this source folder's dependencies")
    if actual_provides != expected_provides:
        raise InputError(f"{wheel} does not exactly match this source folder's provided extras")
    python_fields = metadata.get_all("Requires-Python", [])
    if len(python_fields) > 1:
        raise InputError(f"{wheel} has duplicate Requires-Python metadata")
    actual_python = _normal_specifiers(python_fields[0]) if python_fields else None
    if actual_python != expected_python:
        raise InputError(f"{wheel} does not match this source folder's Requires-Python")

    entry_name = f"{dist_info}/entry_points.txt"
    if expected_scripts:
        if entry_name not in files:
            raise InputError(f"{wheel} has no entry_points.txt for the source console scripts")
        if _entry_points(files[entry_name], wheel) != expected_scripts:
            raise InputError(f"{wheel} entry_points.txt does not match this source folder")
    elif entry_name in files:
        raise InputError(f"{wheel} adds executable entry points the source does not declare")

    wheel_name = f"{dist_info}/WHEEL"
    record_name = f"{dist_info}/RECORD"
    if wheel_name not in files or record_name not in files:
        raise InputError(f"{wheel} has no complete WHEEL/RECORD metadata")
    _validate_record(files, record_name, wheel)
    wheel_metadata = Parser().parsestr(files[wheel_name].decode("utf-8", "replace"))
    allowed_headers = {"Wheel-Version", "Generator", "Root-Is-Purelib", "Tag"}
    if set(wheel_metadata.keys()) - allowed_headers:
        raise InputError(f"{wheel} WHEEL metadata has unexpected fields")
    if (
        wheel_metadata.get_all("Wheel-Version", []) != ["1.0"]
        or wheel_metadata.get_all("Root-Is-Purelib", []) != ["true"]
        or wheel_metadata.get_all("Tag", []) != ["py3-none-any"]
        or len(wheel_metadata.get_all("Generator", [])) > 1
    ):
        raise InputError(f"{wheel} WHEEL metadata does not describe the expected pure Python wheel")

    allowed_dist_info = {"METADATA", "WHEEL", "RECORD", "entry_points.txt"}
    unexpected = sorted(
        name
        for name in files
        if name.startswith(f"{dist_info}/")
        and name.removeprefix(f"{dist_info}/") not in allowed_dist_info
        and not name.startswith(f"{dist_info}/licenses/")
    )
    if unexpected:
        raise InputError(f"{wheel} has unexpected dist-info metadata: {', '.join(unexpected)}")


def _validate_wheel(
    root: Path,
    wheel: Path,
    *,
    wheel_version: str,
    source_version: str,
    allow_no_ui: bool,
    selected_extras: tuple[str, ...],
    expected_git_head: str | None,
) -> None:
    files = _wheel_files(wheel)
    metadata_names = [
        name for name in files if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_names) != 1:
        raise InputError(f"{wheel} has no single *.dist-info/METADATA")
    metadata_name = metadata_names[0]
    metadata = Parser().parsestr(files[metadata_name].decode("utf-8", "replace"), headersonly=True)
    if metadata.get_all("Metadata-Version", []) != ["2.4"]:
        raise InputError(f"{wheel} does not have one Metadata-Version: 2.4 field")
    if metadata.get_all("Name", []) != ["so101"]:
        raise InputError(f"{wheel} METADATA names {metadata.get('Name') or 'nothing'}, not so101")
    if metadata.get_all("Version", []) != [wheel_version]:
        raise InputError(
            f"{wheel} is named version {wheel_version}, but its METADATA says "
            f"{metadata.get('Version') or 'nothing'}"
        )
    if source_version and wheel_version != source_version:
        raise InputError(
            f"{wheel} is version {wheel_version}, but this source folder is version "
            f"{source_version}; use a wheel built from this source"
        )
    dist_info = metadata_name.split("/", 1)[0]
    expected_dist_info = f"so101-{wheel_version.replace('-', '_')}.dist-info"
    if dist_info != expected_dist_info:
        raise InputError(f"{wheel} uses {dist_info}, not {expected_dist_info}")
    _validate_wheel_metadata(root, metadata, files, dist_info, wheel, selected_extras)
    top_levels = {name.split("/", 1)[0] for name in files}
    unexpected = sorted(top_levels - {"so101", dist_info})
    if unexpected:
        raise InputError(f"{wheel} contains unexpected top-level files: {', '.join(unexpected)}")

    packaged = {
        name.removeprefix("so101/"): data
        for name, data in files.items()
        if name.startswith("so101/") and not name.startswith("so101/web/")
    }
    expected = _files_under(root / "src" / "so101")
    if "__init__.py" not in expected:
        raise InputError(f"source folder {root} has no src/so101/__init__.py")
    extra = sorted(set(packaged) - set(expected))
    missing = sorted(set(expected) - set(packaged))
    changed = sorted(
        name for name in set(packaged) & set(expected) if packaged[name] != expected[name]
    )
    if extra or missing or changed:
        detail = (extra or missing or changed)[0]
        kind = "extra" if extra else "missing" if missing else "changed"
        raise InputError(
            f"{wheel} does not match this source folder ({kind} package file {detail})"
        )

    ui = {
        name.removeprefix("so101/web/"): data
        for name, data in files.items()
        if name.startswith("so101/web/")
    }
    _validate_ui(
        ui,
        root=root,
        label=str(wheel),
        allow_no_ui=allow_no_ui,
        expected_git_head=expected_git_head,
    )


def verify(
    root: Path,
    target: Path,
    *,
    wheel_version: str,
    source_version: str,
    allow_no_ui: bool,
    selected_extras: tuple[str, ...],
    expected_git_head: str | None = None,
) -> str:
    if expected_git_head is not None and GIT_SHA.fullmatch(expected_git_head) is None:
        raise InputError("the expected source revision is not a full Git commit id")
    _project_contract(root, selected_extras)
    if wheel_version:
        _validate_wheel(
            root,
            target,
            wheel_version=wheel_version,
            source_version=source_version,
            allow_no_ui=allow_no_ui,
            selected_extras=selected_extras,
            expected_git_head=expected_git_head,
        )
        return f"so101 {wheel_version}; wheel matches this source and its UI build stamp"

    if not (root / "src" / "so101" / "__init__.py").is_file():
        raise InputError(f"source folder {root} has no src/so101/__init__.py")
    ui = _files_under(root / "frontend" / "dist")
    _validate_ui(
        ui,
        root=root,
        label=f"source folder {root}",
        allow_no_ui=allow_no_ui,
        expected_git_head=expected_git_head,
    )
    return f"source folder {root}; UI build stamp verified"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--wheel-version", default="")
    parser.add_argument("--source-version", default="")
    parser.add_argument("--extras", default="")
    parser.add_argument("--allow-no-ui", action="store_true")
    parser.add_argument("--expected-git-head")
    args = parser.parse_args(argv)
    try:
        answer = verify(
            args.root,
            args.target,
            wheel_version=args.wheel_version,
            source_version=args.source_version,
            allow_no_ui=args.allow_no_ui,
            selected_extras=tuple(
                _normal_extra(item.strip()) for item in args.extras.split(",") if item.strip()
            ),
            expected_git_head=args.expected_git_head,
        )
    except InputError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
