"""Hatch build hook: ship the built operator UI (frontend/dist) with the package.

Wheel: frontend/dist/ is copied to so101/web/, where an installed so101 can find
it with importlib.resources. Sdist: frontend/dist/ stays at frontend/dist/, so a
wheel built from the sdist (python -m build does exactly that) bundles it too.
The hook only maps files into the archives; it writes nothing into the source tree.

If the UI is not built (no frontend/dist/index.html), the build prints a warning
and produces a package without the UI, so `pip install .` still works for
backend-only work and the API serves its JSON placeholder at /. With
SO101_REQUIRE_UI=1 (make wheel, make ci-wheel and the CI wheel job) a missing UI
fails the build instead. Editable installs are left alone: they run from the
checkout, which serves frontend/dist directly.

The UI build stamp. `npm run build` ends by writing
frontend/dist/so101-ui-stamp.json (frontend/build-stamp.ts, a plugin in
frontend/vite.config.ts)::

    {"format": 1, "sources": DIGEST, "git_head": SHA or null,
     "files": {"index.html": SHA256, "assets/...": SHA256, ...}}

``files`` is the build's manifest: every file in dist except the stamp.
``sources`` is the digest of the UI's inputs under frontend/: every file in
UI_SOURCE_DIRS and the UI_SOURCE_FILES, except .DS_Store. The digest is the
SHA-256 of ``"<path>\\0<sha256 of the file>\\n"`` for each input, sorted by
path (POSIX, relative to frontend/). A built UI is bundled only when the stamp
lists exactly the files in frontend/dist with their hashes and, where the
frontend sources are present (a checkout, not an unpacked sdist), its digest
matches them. Otherwise the build fails: a stale, hand-edited or hand-copied
dist would ship. frontend/build-stamp.ts computes the same digest; both test
suites pin one known answer.

Package files. In a git checkout, a file under src/so101 that git neither
tracks nor ignores fails the build, because hatchling would pack it (an
untracked local hack would ship). A tracked file with uncommitted changes
only warns: the package then carries the working tree, not the commit.

scripts/check_dist.py and scripts/make_handoff.py import the UI-stamp helpers
from this file, also where hatchling is not installed (the CI wheel job
installs only `build`, which runs hatchling in its own environment).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ModuleNotFoundError:  # imported for the UI-stamp helpers only; no build runs
    BuildHookInterface = object  # type: ignore[assignment,misc]

UI_DIR = os.path.join("frontend", "dist")
# Where each target puts the built UI.
UI_TARGETS = {"wheel": "so101/web", "sdist": "frontend/dist"}
REQUIRE_UI_ENV = "SO101_REQUIRE_UI"
PACKAGE_SRC = "src/so101"

# The UI build stamp; keep these in step with frontend/build-stamp.ts.
UI_STAMP = "so101-ui-stamp.json"
UI_STAMP_FORMAT = 1
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
UI_SKIP_NAMES = frozenset({".DS_Store"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_LISTED = 20  # problems listed per kind before "... and N more"


def ui_required() -> bool:
    return os.environ.get(REQUIRE_UI_ENV, "").strip().lower() in {"1", "true", "yes"}


# --- the UI build stamp -------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(directory: Path, prefix: str) -> Iterator[tuple[str, Path]]:
    """Regular files under ``directory`` as (``prefix/...`` POSIX path, path).

    Follows symlinks, like frontend/build-stamp.ts (statSync).
    """
    for entry in sorted(directory.iterdir()):
        relative = f"{prefix}/{entry.name}" if prefix else entry.name
        if entry.is_dir():
            yield from _walk(entry, relative)
        elif entry.is_file():
            yield relative, entry


def is_ui_source(path: str) -> bool:
    """Whether ``path`` (POSIX, relative to frontend/) is an input of the UI build."""
    parts = path.split("/")
    if parts[-1] in UI_SKIP_NAMES:
        return False
    return path in UI_SOURCE_FILES or (len(parts) > 1 and parts[0] in UI_SOURCE_DIRS)


def _sources_digest(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in hashes if is_ui_source(path)):
        digest.update(f"{path}\0{hashes[path]}\n".encode())
    return digest.hexdigest()


def ui_sources_digest(files: Mapping[str, bytes]) -> str:
    """The stamp's ``sources`` digest of ``files`` (frontend/-relative path -> contents).

    Paths that are not UI inputs are ignored, so a whole tree may be passed.
    """
    return _sources_digest({path: sha256_hex(data) for path, data in files.items()})


def ui_sources_digest_of(frontend: Path) -> str:
    """The stamp's ``sources`` digest of the frontend directory on disk."""
    hashes: dict[str, str] = {}
    for name in UI_SOURCE_DIRS:
        if (frontend / name).is_dir():
            hashes.update((path, _file_sha256(file)) for path, file in _walk(frontend / name, name))
    for name in UI_SOURCE_FILES:
        if (frontend / name).is_file():
            hashes[name] = _file_sha256(frontend / name)
    return _sources_digest(hashes)


def dist_hashes(dist: Path) -> dict[str, str]:
    """SHA-256 of every file under ``dist`` except the stamp: the build's manifest."""
    return {path: _file_sha256(file) for path, file in _walk(dist, "") if path != UI_STAMP}


def parse_ui_stamp(data: bytes) -> dict[str, Any]:
    """The stamp in ``data``; ValueError says what is wrong with it."""
    try:
        stamp = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not JSON ({exc})") from None
    if not isinstance(stamp, dict) or stamp.get("format") != UI_STAMP_FORMAT:
        raise ValueError(f"not a format-{UI_STAMP_FORMAT} stamp")
    sources, head, files = stamp.get("sources"), stamp.get("git_head"), stamp.get("files")
    if not isinstance(sources, str) or not _SHA256.match(sources):
        raise ValueError("'sources' is not a SHA-256")
    if head is not None and (not isinstance(head, str) or not _GIT_SHA.match(head)):
        raise ValueError("'git_head' is neither null nor a commit id")
    if not isinstance(files, dict) or not all(
        isinstance(path, str) and isinstance(value, str) and _SHA256.match(value)
        for path, value in files.items()
    ):
        raise ValueError("'files' is not a map of paths to SHA-256s")
    if "index.html" not in files:
        raise ValueError("'files' has no index.html")
    return stamp


def _listed(problems: list[str], label: str, paths: list[str]) -> None:
    problems += [f"{path} {label}" for path in paths[:_LISTED]]
    if len(paths) > _LISTED:
        problems.append(f"... and {len(paths) - _LISTED} more {label}")


def ui_build_problems(
    hashes: Mapping[str, str], stamp_data: bytes | None, sources: str | None
) -> tuple[dict[str, Any] | None, list[str]]:
    """Check one built UI against its stamp.

    ``hashes``: SHA-256 of every file of the build except the stamp (from disk,
    a wheel or an sdist). ``stamp_data``: the stamp's bytes, None if missing.
    ``sources``: the current sources digest, or None when the sources are not
    at hand. Returns the parsed stamp (None if unusable) and the problems.
    """
    if stamp_data is None:
        return None, [
            f"no {UI_STAMP}: the UI was not built by `npm run build` in this checkout "
            "(or by an older one without the build stamp)"
        ]
    try:
        stamp = parse_ui_stamp(stamp_data)
    except ValueError as exc:
        return None, [f"{UI_STAMP} is not a valid build stamp: {exc}"]
    problems: list[str] = []
    if sources is not None and stamp["sources"] != sources:
        problems.append(
            f"the UI was built from other frontend sources (stamp {stamp['sources'][:12]}, "
            f"sources now {sources[:12]}), so it is stale"
        )
    manifest: dict[str, str] = stamp["files"]
    _listed(problems, "is not in the build's manifest", sorted(set(hashes) - set(manifest)))
    _listed(problems, "is in the build's manifest but missing", sorted(set(manifest) - set(hashes)))
    changed = sorted(path for path in set(hashes) & set(manifest) if hashes[path] != manifest[path])
    _listed(problems, "differs from the build's manifest", changed)
    return stamp, problems


def ui_dist_problems(frontend: Path) -> list[str]:
    """Problems with frontend/dist as a build of the frontend sources next to it.

    The sources digest is checked only when frontend/src exists: an unpacked
    sdist has the built UI but not its sources (the sdist build checked them).
    """
    dist = frontend / "dist"
    stamp_path = dist / UI_STAMP
    stamp_data = stamp_path.read_bytes() if stamp_path.is_file() else None
    sources = ui_sources_digest_of(frontend) if (frontend / "src").is_dir() else None
    return ui_build_problems(dist_hashes(dist), stamp_data, sources)[1]


# --- package files ------------------------------------------------------------------


def package_changes(root: Path) -> tuple[list[str], list[str]] | None:
    """(untracked, modified) files under src/so101, or None if not a git checkout.

    Ignored files are left out: hatchling skips them too.
    """
    if not (root / ".git").exists():  # a directory, or a file in a git worktree
        return None
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
            + ["--", PACKAGE_SRC],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError):
        return None
    untracked: list[str] = []
    modified: list[str] = []
    entries = iter(listed.split("\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if code == "??":
            untracked.append(path)
        else:
            modified.append(path)
        if "R" in code or "C" in code:
            next(entries, None)  # the rename's or copy's source path
    return sorted(untracked), sorted(modified)


def _shown(paths: list[str]) -> str:
    return ", ".join(paths[:10]) + (f" and {len(paths) - 10} more" if len(paths) > 10 else "")


class UiBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        target = UI_TARGETS.get(self.target_name)
        if target is None or version == "editable":
            return
        self._check_package_files()
        source = os.path.join(self.root, UI_DIR)
        if os.path.isfile(os.path.join(source, "index.html")):
            problems = ui_dist_problems(Path(self.root) / "frontend")
            if problems:
                self.app.abort(
                    f"error: {UI_DIR} is not a build of the current frontend sources, so "
                    f"the {self.target_name} would ship a stale or altered UI:\n  "
                    + "\n  ".join(problems)
                    + "\nRun `make build-ui` (cd frontend && npm run build), or delete "
                    f"{UI_DIR} to build without the UI."
                )
                return  # abort() exits; the return only satisfies type checkers
            build_data["force_include"][source] = target
            return
        missing = f"{UI_DIR}/index.html does not exist, so the UI is not built"
        if ui_required():
            self.app.abort(
                f"error: {REQUIRE_UI_ENV}=1 and {missing}. Run `make build-ui` "
                "(cd frontend && npm run build) before building the package."
            )
            return  # abort() exits; the return only satisfies type checkers
        self.app.display_warning(
            f"warning: {missing}. Building the {self.target_name} without the operator "
            f"UI; run `make build-ui` first to bundle it ({REQUIRE_UI_ENV}=1 makes this "
            "an error)."
        )

    def _check_package_files(self) -> None:
        changes = package_changes(Path(self.root))
        if changes is None:
            return
        untracked, modified = changes
        if untracked:
            self.app.abort(
                f"error: git does not track {_shown(untracked)}, and the {self.target_name} "
                f"would carry it: hatchling packs every file under {PACKAGE_SRC} that is not "
                "ignored. Commit it, delete it, or add it to .gitignore."
            )
            return  # abort() exits; the return only satisfies type checkers
        if modified:
            self.app.display_warning(
                f"warning: uncommitted changes in {_shown(modified)}; the "
                f"{self.target_name} carries the working tree, not the commit."
            )
