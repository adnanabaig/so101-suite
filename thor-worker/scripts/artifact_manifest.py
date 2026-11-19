#!/usr/bin/env python3
"""Create and verify canonical immutable artifact manifests.

This module intentionally uses only the standard library so the G15 and Thor
can verify a transfer before installing any workload dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "artifact-manifest.json"
SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
KINDS = frozenset({"dataset", "checkpoint", "benchmark", "evaluation", "other"})
PROHIBITED_NAMES = frozenset(
    {".lock", "lock", ".incomplete", "incomplete", ".partial", "partial", ".tmp"}
)
PROHIBITED_SUFFIXES = (".lock", ".incomplete", ".partial", ".tmp")


class ManifestError(ValueError):
    """The artifact is not a complete, canonical bundle."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ManifestError(f"non-standard JSON number is prohibited: {value}")


def _require_exact_keys(value: object, expected: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{location} must be an object")
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ManifestError(f"{location} keys differ (missing={missing}, extra={extra})")
    return value


def _validate_component(component: str) -> None:
    if component in {"", ".", ".."}:
        raise ManifestError("artifact paths may not contain empty, '.' or '..' components")
    if unicodedata.normalize("NFC", component) != component:
        raise ManifestError(f"artifact path component is not NFC-normalized: {component!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in component):
        raise ManifestError("artifact paths may not contain control characters")
    lowered = component.casefold()
    if lowered in PROHIBITED_NAMES or lowered.endswith(PROHIBITED_SUFFIXES):
        raise ManifestError(f"incomplete or lock marker is prohibited: {component!r}")


def validate_relative_path(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise ManifestError("artifact file path must be a non-empty string")
    if path.startswith("/") or "\\" in path:
        raise ManifestError(f"artifact path must be relative POSIX syntax: {path!r}")
    components = path.split("/")
    for component in components:
        _validate_component(component)
    normalized = "/".join(components)
    if normalized == MANIFEST_NAME:
        raise ManifestError(f"{MANIFEST_NAME} may not list itself")
    return normalized


def _validate_manifest(value: object) -> dict[str, Any]:
    manifest = _require_exact_keys(value, {"schema_version", "artifact", "files"}, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(f"unsupported schema_version: {manifest['schema_version']!r}")

    artifact = _require_exact_keys(
        manifest["artifact"], {"kind", "name", "source_commit", "created_utc"}, "artifact"
    )
    if artifact["kind"] not in KINDS:
        raise ManifestError(f"unsupported artifact kind: {artifact['kind']!r}")
    if not isinstance(artifact["name"], str) or not NAME_RE.fullmatch(artifact["name"]):
        raise ManifestError("artifact name must use 1-128 ASCII letters, digits, '.', '_' or '-'")
    if not isinstance(artifact["source_commit"], str) or not SOURCE_COMMIT_RE.fullmatch(
        artifact["source_commit"]
    ):
        raise ManifestError("source_commit must be exactly 40 lowercase hexadecimal characters")
    if artifact["source_commit"] == "0" * 40:
        raise ManifestError("source_commit may not use the all-zero placeholder")
    created_utc = artifact["created_utc"]
    if not isinstance(created_utc, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created_utc
    ):
        raise ManifestError("created_utc must be UTC with whole seconds (YYYY-MM-DDTHH:MM:SSZ)")
    try:
        datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ManifestError(f"created_utc is not a real timestamp: {exc}") from exc

    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ManifestError("files must be a non-empty array")
    validated_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(files):
        entry = _require_exact_keys(raw_entry, {"path", "size", "sha256"}, f"files[{index}]")
        path = validate_relative_path(entry["path"])
        if path in seen:
            raise ManifestError(f"duplicate artifact path: {path!r}")
        seen.add(path)
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ManifestError(f"invalid size for {path!r}")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
            raise ManifestError(f"invalid sha256 for {path!r}")
        validated_files.append({"path": path, "size": size, "sha256": digest})
    if [entry["path"] for entry in validated_files] != sorted(seen):
        raise ManifestError("files must be sorted lexicographically by path")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": artifact,
        "files": validated_files,
    }


def _bundle_root(root: Path) -> Path:
    root = root.absolute()
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise ManifestError(f"artifact root does not exist: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise ManifestError("artifact root may not be a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ManifestError("artifact root must be a directory")
    return root


def _scan_bundle(root: Path, *, allow_manifest: bool) -> list[str]:
    files: list[str] = []

    def visit(directory: Path, prefix: str) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ManifestError(f"cannot scan artifact directory {directory}: {exc}") from exc
        for entry in entries:
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if allow_manifest and rel == MANIFEST_NAME:
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError as exc:
                    raise ManifestError(f"cannot inspect {rel!r}: {exc}") from exc
                if not stat.S_ISREG(mode):
                    raise ManifestError(f"manifest must be an ordinary file: {rel!r}")
                continue
            validate_relative_path(rel)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ManifestError(f"cannot inspect {rel!r}: {exc}") from exc
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                raise ManifestError(f"symlinks are prohibited: {rel!r}")
            if stat.S_ISDIR(mode):
                visit(Path(entry.path), rel)
            elif stat.S_ISREG(mode):
                files.append(rel)
            else:
                raise ManifestError(f"special files are prohibited: {rel!r}")

    visit(root, "")
    return sorted(files)


def _open_regular_no_follow(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot securely open {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManifestError(f"not an ordinary file: {path}")
        lexical = _stat_path_no_follow(path)
        if stat.S_ISLNK(lexical.st_mode) or (lexical.st_dev, lexical.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ManifestError(f"file identity changed while opening: {path}")
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _file_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _read_regular_beneath(
    root: Path,
    relative: str,
    *,
    on_chunk: Callable[[bytes], None] | None = None,
) -> tuple[int, str, os.stat_result]:
    """Read one file via a no-follow descriptor walk anchored at ``root``."""
    relative = validate_relative_path(relative)
    components = relative.split("/")
    root = _bundle_root(root)
    root_before = root.lstat()
    descriptors: list[int] = []
    directory_records: list[tuple[int, str, os.stat_result, int]] = []
    file_descriptor: int | None = None
    try:
        root_descriptor = os.open(root, _directory_flags())
        descriptors.append(root_descriptor)
        root_opened = os.fstat(root_descriptor)
        if (root_before.st_dev, root_before.st_ino) != (
            root_opened.st_dev,
            root_opened.st_ino,
        ):
            raise ManifestError("artifact root identity changed while opening")
        current_descriptor = root_descriptor
        for component in components[:-1]:
            entry = os.stat(component, dir_fd=current_descriptor, follow_symlinks=False)
            if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
                raise ManifestError(f"artifact ancestor is not an ordinary directory: {relative!r}")
            child_descriptor = os.open(component, _directory_flags(), dir_fd=current_descriptor)
            child = os.fstat(child_descriptor)
            if (entry.st_dev, entry.st_ino) != (child.st_dev, child.st_ino):
                os.close(child_descriptor)
                raise ManifestError(f"artifact ancestor identity changed: {relative!r}")
            descriptors.append(child_descriptor)
            directory_records.append(
                (current_descriptor, component, entry, child_descriptor)
            )
            current_descriptor = child_descriptor

        leaf = components[-1]
        lexical = os.stat(leaf, dir_fd=current_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
            raise ManifestError(f"artifact payload is not an ordinary file: {relative!r}")
        file_descriptor = os.open(leaf, _file_flags(), dir_fd=current_descriptor)
        before = os.fstat(file_descriptor)
        if _identity(lexical) != _identity(before):
            raise ManifestError(f"artifact payload identity changed: {relative!r}")

        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
        after = os.fstat(file_descriptor)
        if _identity(before) != _identity(after) or total != after.st_size:
            raise ManifestError(f"artifact payload changed while reading: {relative!r}")
        current_leaf = os.stat(leaf, dir_fd=current_descriptor, follow_symlinks=False)
        if _identity(current_leaf) != _identity(before):
            raise ManifestError(f"artifact payload path was replaced: {relative!r}")
        for parent_descriptor, name, expected, child_descriptor in directory_records:
            current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            opened = os.fstat(child_descriptor)
            if stat.S_ISLNK(current.st_mode) or (
                current.st_dev,
                current.st_ino,
            ) != (expected.st_dev, expected.st_ino) or (
                opened.st_dev,
                opened.st_ino,
            ) != (expected.st_dev, expected.st_ino):
                raise ManifestError(f"artifact ancestor was replaced: {relative!r}")
        root_after = root.lstat()
        if (root_after.st_dev, root_after.st_ino) != (
            root_opened.st_dev,
            root_opened.st_ino,
        ):
            raise ManifestError("artifact root was replaced while reading")
        return total, digest.hexdigest(), before
    except OSError as exc:
        raise ManifestError(
            f"cannot securely read {relative!r} beneath artifact root: {exc}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def stream_verified_file(
    root: Path,
    entry: dict[str, Any],
    on_chunk: Callable[[bytes], None],
) -> int:
    """Stream a manifest entry and reject any byte not bound to its size/hash."""
    validated = _require_exact_keys(entry, {"path", "size", "sha256"}, "file entry")
    relative = validate_relative_path(validated["path"])
    if (
        isinstance(validated["size"], bool)
        or not isinstance(validated["size"], int)
        or validated["size"] < 0
    ):
        raise ManifestError(f"invalid size for {relative!r}")
    if not isinstance(validated["sha256"], str) or not DIGEST_RE.fullmatch(
        validated["sha256"]
    ):
        raise ManifestError(f"invalid sha256 for {relative!r}")
    size, digest, _identity_at_read = _read_regular_beneath(
        root,
        relative,
        on_chunk=on_chunk,
    )
    if size != validated["size"] or digest != validated["sha256"]:
        raise ManifestError(f"timed payload bytes differ from manifest: {relative!r}")
    return size


def _stat_path_no_follow(path: Path) -> os.stat_result:
    """Stat a leaf through an identity-checked, non-symlink parent directory."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_before = path.parent.lstat()
        if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
            raise ManifestError(f"file parent is not an ordinary directory: {path.parent}")
        parent_descriptor = os.open(path.parent, flags)
    except OSError as exc:
        raise ManifestError(f"cannot securely open parent directory for {path}: {exc}") from exc
    try:
        opened_parent = os.fstat(parent_descriptor)
        if (parent_before.st_dev, parent_before.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ):
            raise ManifestError(f"parent directory identity changed for {path}")
        lexical = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        parent_after = path.parent.lstat()
        if (parent_after.st_dev, parent_after.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ):
            raise ManifestError(f"parent directory was replaced for {path}")
        return lexical
    except OSError as exc:
        raise ManifestError(f"cannot securely stat {path}: {exc}") from exc
    finally:
        os.close(parent_descriptor)


def _assert_path_identity(path: Path, expected: os.stat_result) -> None:
    current = _stat_path_no_follow(path)
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise ManifestError(f"file path is no longer an ordinary file: {path}")
    if _identity(current) != _identity(expected):
        raise ManifestError(f"file path was replaced or changed while hashing: {path}")


def _read_regular(
    path: Path, *, limit: int | None = None
) -> tuple[bytes, os.stat_result]:
    descriptor, before = _open_regular_no_follow(path)
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if limit is not None and total > limit:
                raise ManifestError(f"file exceeds {limit} bytes: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or total != after.st_size:
        raise ManifestError(f"file changed while being read: {path}")
    _assert_path_identity(path, before)
    return b"".join(chunks), before


def _parse_manifest(raw: bytes) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"manifest is not UTF-8: {exc}") from exc
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest JSON: {exc}") from exc
    manifest = _validate_manifest(value)
    if raw != canonical_json_bytes(manifest):
        raise ManifestError("manifest bytes are not in canonical JSON form")
    return manifest


def verify_bundle(root: Path) -> dict[str, Any]:
    root = _bundle_root(root)
    manifest_path = root / MANIFEST_NAME
    try:
        raw, manifest_identity = _read_regular(manifest_path, limit=MAX_MANIFEST_BYTES)
    except FileNotFoundError as exc:
        raise ManifestError(f"missing {MANIFEST_NAME}") from exc
    manifest = _parse_manifest(raw)
    discovered = _scan_bundle(root, allow_manifest=True)
    declared = [entry["path"] for entry in manifest["files"]]
    if discovered != declared:
        missing = sorted(set(declared) - set(discovered))
        undeclared = sorted(set(discovered) - set(declared))
        raise ManifestError(f"bundle contents differ (missing={missing}, undeclared={undeclared})")
    identities: dict[str, os.stat_result] = {}
    for entry in manifest["files"]:
        size, digest, identity = _read_regular_beneath(root, entry["path"])
        if size != entry["size"]:
            raise ManifestError(f"size mismatch for {entry['path']!r}")
        if digest != entry["sha256"]:
            raise ManifestError(f"sha256 mismatch for {entry['path']!r}")
        identities[entry["path"]] = identity
    if _scan_bundle(root, allow_manifest=True) != declared:
        raise ManifestError("bundle contents changed during verification")
    for relative, identity in identities.items():
        _assert_path_identity(root / relative, identity)
    _assert_path_identity(manifest_path, manifest_identity)
    return manifest


def _write_manifest(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def create_bundle_manifest(
    root: Path, *, kind: str, name: str, source_commit: str, created_utc: str | None = None
) -> dict[str, Any]:
    root = _bundle_root(root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ManifestError(f"refusing to replace existing {MANIFEST_NAME}")
    files = _scan_bundle(root, allow_manifest=False)
    if not files:
        raise ManifestError("artifact bundle must contain at least one payload file")
    timestamp = created_utc or (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    entries = []
    for relative in files:
        size, digest, _identity_at_hash = _read_regular_beneath(root, relative)
        entries.append({"path": relative, "size": size, "sha256": digest})
    manifest = _validate_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": {
                "kind": kind,
                "name": name,
                "source_commit": source_commit,
                "created_utc": timestamp,
            },
            "files": entries,
        }
    )
    _write_manifest(manifest_path, canonical_json_bytes(manifest))
    # Re-scan and re-hash after publication so concurrent mutation fails closed.
    return verify_bundle(root)


def manifest_sha256(root: Path) -> str:
    raw, _identity_at_read = _read_regular(
        _bundle_root(root) / MANIFEST_NAME,
        limit=MAX_MANIFEST_BYTES,
    )
    return hashlib.sha256(raw).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a canonical manifest")
    create.add_argument("root", type=Path)
    create.add_argument("--kind", required=True, choices=sorted(KINDS))
    create.add_argument("--name", required=True)
    create.add_argument("--source-commit", required=True)

    verify = subparsers.add_parser("verify", help="verify a complete bundle")
    verify.add_argument("root", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest = create_bundle_manifest(
                args.root, kind=args.kind, name=args.name, source_commit=args.source_commit
            )
        else:
            manifest = verify_bundle(args.root)
        summary = {
            "ok": True,
            "artifact": manifest["artifact"],
            "files": len(manifest["files"]),
            "manifest_sha256": manifest_sha256(args.root),
        }
        sys.stdout.buffer.write(canonical_json_bytes(summary))
        return 0
    except (ManifestError, OSError) as exc:
        print(f"artifact manifest error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
