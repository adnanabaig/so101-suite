"""The SHA-256 manifest that seals a finalized HIL run directory.

``MANIFEST.sha256`` uses the ``sha256sum`` / ``shasum -a 256`` line format, so an
operator can check a run without the Suite::

    cd <data_dir>/hil/<run_id> && shasum -a 256 -c MANIFEST.sha256

It lists every regular file in the run directory except itself and hidden files
(names starting with ``.``: in-flight ``.tmp-`` writes, Finder's ``.DS_Store``).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from ..atomic_io import atomic_write_text
from .schemas import ManifestVerification

MANIFEST_NAME = "MANIFEST.sha256"
_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hidden(relative: PurePosixPath) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def evidence_files(run_dir: Path) -> list[str]:
    """Relative POSIX paths of the files a manifest covers, sorted."""
    root = Path(run_dir)
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _hidden(relative) or relative.as_posix() == MANIFEST_NAME:
            continue
        found.append(relative.as_posix())
    return sorted(found)


def write_manifest(run_dir: Path) -> Path:
    """Hash every evidence file and write the manifest atomically (last)."""
    root = Path(run_dir)
    lines = [f"{sha256_file(root / name)}  {name}\n" for name in evidence_files(root)]
    path = root / MANIFEST_NAME
    atomic_write_text(path, "".join(lines))
    return path


def _safe_relative(text: str) -> str | None:
    relative = PurePosixPath(text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    if "\\" in text or _hidden(relative) or relative.as_posix() != text:
        return None
    return text


def verify_manifest(run_dir: Path) -> ManifestVerification:
    """Re-hash every listed file; report missing, changed, unlisted and bad lines."""
    root = Path(run_dir)
    path = root / MANIFEST_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ManifestVerification(ok=False, manifest_present=False)

    expected: dict[str, str] = {}
    malformed: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _LINE.match(line)
        name = _safe_relative(match.group(2)) if match else None
        if match is None or name is None or name in expected or name == MANIFEST_NAME:
            malformed.append(f"line {number}: {line[:120]}")
            continue
        expected[name] = match.group(1)

    missing: list[str] = []
    mismatched: list[str] = []
    for name, digest in expected.items():
        target = root / name
        if target.is_symlink() or not target.is_file():
            missing.append(name)
        elif sha256_file(target) != digest:
            mismatched.append(name)
    extra = [name for name in evidence_files(root) if name not in expected]
    ok = bool(expected) and not (missing or mismatched or extra or malformed)
    return ManifestVerification(
        ok=ok,
        manifest_present=True,
        checked=len(expected),
        missing=missing,
        mismatched=mismatched,
        extra=extra,
        malformed=malformed,
    )
