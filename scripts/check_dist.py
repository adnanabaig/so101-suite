#!/usr/bin/env python3
"""Check what the built sdist and wheel contain (make ci-wheel, the CI wheel job).

    python scripts/check_dist.py DIST_DIR [--ui required|absent|auto] [--root REPO]

Exits 1, listing every problem, unless:

- DIST_DIR holds exactly one so101 wheel and one so101 sdist;
- every tracked file under src/so101 is in both (so101/... and src/so101/...),
  and so is every file under src/so101/simulation/assets/ when that directory
  exists, tracked or not (a .gitignore'd asset would otherwise drop silently);
- neither holds a package file that is not one of those: in a git checkout an
  untracked file under src/so101 (a local hack) fails, outside one a file that
  is not on disk does (hatch_build.py already refuses to build with untracked
  files, so this checks what was actually packed);
- the wheel holds only so101/ and its .dist-info, declares the so101 console
  script and Requires-Python >=3.12,<3.14; the sdist holds only the package source
  and the files needed to build the wheel;
- the built UI is in both (so101/web/ in the wheel, frontend/dist/ in the sdist)
  with --ui required, and in neither with --ui absent. A bundled UI must carry
  its build stamp (so101-ui-stamp.json, see hatch_build.py), list exactly the
  files of the stamp's manifest with their hashes (no stale index.html, no
  leftover chunk), and have been built from the checkout's frontend sources;
- neither contains tests, node_modules, caches, .env files, keys or
  credentials, data or model files, text that looks like a token or private
  key, or the value of any secret set in the checkout's .env files.
"""

from __future__ import annotations

import argparse
import configparser
import functools
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from email.parser import Parser
from pathlib import Path, PurePosixPath
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "so101"
PACKAGE_SRC = "src/so101"
ASSETS_SRC = "src/so101/simulation/assets"
UI_SRC = "frontend/dist"
UI_WHEEL = "so101/web"
# pyproject.toml's requires-python. Hatchling writes it normalized and sorted
# ("<3.14,>=3.12"), so the check compares the specifiers, not the text.
REQUIRES_PYTHON = ">=3.12,<3.14"
CONSOLE_SCRIPT = "so101.cli:app"
# Everything an sdist may hold at its top level: the package source, the built
# UI, and what hatchling adds to rebuild the wheel.
SDIST_TOP_LEVEL = frozenset(
    {"src", "frontend", "PKG-INFO", "pyproject.toml", "hatch_build.py", "README.md", "LICENSE"}
    | {".gitignore"}  # hatchling copies it into every sdist
)

# Directory names that never ship, at any depth.
FORBIDDEN_DIRS = frozenset(
    {
        "tests",
        "test",
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        "htmlcov",
        ".idea",
        ".vscode",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "conftest.py",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".coverage",
        "coverage.json",
        ".ds_store",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".parquet",
        ".mp4",
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
    }
)
ENV_FILE = re.compile(r"^\.env(\..+)?$")
TEST_FILE = re.compile(r"^(test_.*|.*_test)\.py$")
SSH_KEY = re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$")
SECRET_PATTERNS = {
    "a private key": re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"),
    "an OpenAI-style API key": re.compile(rb"\bsk-[A-Za-z0-9_-]{32,}"),
    "a Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}"),
    "an AWS access key id": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "a GitHub token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"),
    "a Slack token": re.compile(rb"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    "a Google API key": re.compile(rb"\bAIza[0-9A-Za-z_-]{35}"),
}
# .env keys whose values are secrets; their values must not appear in a package.
SECRET_ENV_KEY = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY)$", re.IGNORECASE)
MIN_SECRET_LENGTH = 8
# Junk that hatchling never packs, so it is not expected in a package either.
JUNK_DIRS = frozenset({"__pycache__", ".git", ".venv", ".mypy_cache", ".pytest_cache"})
JUNK_NAMES = frozenset({".DS_Store"})


class DistError(Exception):
    """The dist directory cannot be checked at all."""


@functools.cache
def _hatch_build() -> ModuleType:
    """The build hook, for its UI-stamp helpers (it imports hatchling, a dev dependency)."""
    spec = importlib.util.spec_from_file_location("so101_hatch_build", REPO_ROOT / "hatch_build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.problems.append(message)

    def ok(self, message: str) -> None:
        self.notes.append(message)


def find_artifacts(dist_dir: Path) -> tuple[Path, Path]:
    """The one so101 wheel and the one so101 sdist in ``dist_dir``."""
    if not dist_dir.is_dir():
        raise DistError(f"{dist_dir} is not a directory; build first (make wheel)")
    wheels = sorted(dist_dir.glob(f"{PACKAGE}-*.whl"))
    sdists = sorted(dist_dir.glob(f"{PACKAGE}-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        found = ", ".join(path.name for path in wheels + sdists) or "nothing"
        raise DistError(
            f"expected one {PACKAGE} wheel and one sdist in {dist_dir}, found {found}. "
            "Empty the directory and build again."
        )
    return wheels[0], sdists[0]


def read_wheel(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()
        }


def read_sdist(path: Path, report: Report) -> dict[str, bytes]:
    """Regular files in the sdist, keyed by their path under the top-level directory."""
    expected_top = path.name.removesuffix(".tar.gz")
    files: dict[str, bytes] = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            top, _, rest = member.name.partition("/")
            if top != expected_top:
                report.fail(f"sdist: {member.name} is outside {expected_top}/")
                continue
            if member.issym() or member.islnk():
                report.fail(f"sdist: {member.name} is a link")
            elif member.isfile() and rest:
                extracted = archive.extractfile(member)
                files[rest] = extracted.read() if extracted else b""
    return files


def forbidden_reason(path: str) -> str | None:
    """Why ``path`` (inside a package) must not ship, or None."""
    parts = PurePosixPath(path).parts
    for part in parts[:-1]:
        if part.lower() in FORBIDDEN_DIRS:
            return f"inside a {part}/ directory"
    name = parts[-1]
    lowered = name.lower()
    if ENV_FILE.match(lowered) and lowered != ".env.example":
        return "an environment file"
    if lowered in FORBIDDEN_NAMES:
        return "a file that never ships"
    if TEST_FILE.match(lowered):
        return "a test file"
    if SSH_KEY.match(lowered):
        return "an SSH key"
    suffix = PurePosixPath(lowered).suffix
    if suffix in FORBIDDEN_SUFFIXES:
        return f"a {suffix} file"
    return None


def secret_findings(data: bytes, env_secrets: dict[str, bytes]) -> list[str]:
    """What in ``data`` looks like a secret. Binary files (with NUL bytes) are skipped."""
    if b"\0" in data:
        return []
    found = [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(data)]
    found += [
        f"the value of {key} from .env" for key, value in env_secrets.items() if value in data
    ]
    return found


def env_file_secrets(root: Path) -> dict[str, bytes]:
    """Secret values set in the checkout's .env files (never printed)."""
    secrets: dict[str, bytes] = {}
    candidates = [root / ".env", *sorted((root / "frontend").glob(".env*"))]
    for env_file in candidates:
        if not env_file.is_file() or env_file.name == ".env.example":
            continue
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            value = value.strip().strip("'\"")
            if sep and SECRET_ENV_KEY.search(key.strip()) and len(value) >= MIN_SECRET_LENGTH:
                secrets[key.strip()] = value.encode()
    return secrets


def _files_on_disk(directory: Path, root: Path) -> list[str]:
    """Files under ``directory`` as root-relative POSIX paths, without junk."""
    found = []
    for current, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in JUNK_DIRS)
        for name in sorted(files):
            if name in JUNK_NAMES or name.endswith((".pyc", ".pyo")):
                continue
            found.append((Path(current) / name).relative_to(root).as_posix())
    return found


def _git_tracked(root: Path, pathspec: str) -> list[str] | None:
    """``git ls-files`` under ``root``, or None if ``root`` is not a git checkout's top."""
    if not (root / ".git").exists():  # a directory, or a file in a git worktree
        return None
    try:
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", pathspec],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [name for name in listed.split("\0") if name]


def package_source_files(root: Path) -> tuple[list[str], str]:
    """Files under src/so101 that both packages must contain, and where the list came from."""
    tracked = _git_tracked(root, PACKAGE_SRC)
    if tracked is None:
        return _files_on_disk(root / PACKAGE_SRC, root), "files on disk (not a git checkout)"
    # A tracked file deleted in the working tree is not expected to ship.
    return [name for name in tracked if (root / name).is_file()], "git ls-files"


def _require(report: Report, label: str, names: set[str], expected: list[str]) -> int:
    missing = [name for name in expected if name not in names]
    for name in missing[:20]:
        report.fail(f"{label}: missing {name}")
    if len(missing) > 20:
        report.fail(f"{label}: {len(missing) - 20} more files missing")
    return len(expected) - len(missing)


def _check_forbidden(report: Report, label: str, files: dict[str, bytes], root: Path) -> None:
    env_secrets = env_file_secrets(root)
    clean = True
    for name, data in sorted(files.items()):
        reason = forbidden_reason(name)
        if reason:
            report.fail(f"{label}: {name} is {reason}")
            clean = False
        for finding in secret_findings(data, env_secrets):
            report.fail(f"{label}: {name} contains {finding}")
            clean = False
    if clean:
        report.ok(
            f"{label}: nothing forbidden (tests, node_modules, caches, .env, keys, "
            f"data/model files, token patterns, {len(env_secrets)} .env secret values)"
        )


def _check_wheel_metadata(report: Report, wheel_path: Path, wheel: dict[str, bytes]) -> None:
    problems_before = len(report.problems)
    if not wheel_path.name.endswith("-py3-none-any.whl"):
        report.fail(f"wheel: {wheel_path.name} is not a pure-Python (py3-none-any) wheel")
    top_level = {name.split("/", 1)[0] for name in wheel}
    dist_infos = sorted(name for name in top_level if name.endswith(".dist-info"))
    if len(dist_infos) != 1:
        report.fail(f"wheel: expected one .dist-info directory, found {dist_infos}")
        return
    dist_info = dist_infos[0]
    for name in sorted(top_level - {PACKAGE, dist_info}):
        report.fail(f"wheel: unexpected top-level entry {name}")
    metadata = Parser().parsestr(wheel.get(f"{dist_info}/METADATA", b"").decode())
    if metadata.get("Name") != PACKAGE:
        report.fail(f"wheel: METADATA Name is {metadata.get('Name')!r}, expected {PACKAGE!r}")
    if _specifiers(metadata.get("Requires-Python") or "") != _specifiers(REQUIRES_PYTHON):
        report.fail(
            f"wheel: METADATA Requires-Python is {metadata.get('Requires-Python')!r}, "
            f"expected {REQUIRES_PYTHON!r}"
        )
    entry_points = configparser.ConfigParser(interpolation=None)
    entry_points.read_string(wheel.get(f"{dist_info}/entry_points.txt", b"").decode())
    script = entry_points.get("console_scripts", PACKAGE, fallback=None)
    if script != CONSOLE_SCRIPT:
        report.fail(f"wheel: console script {PACKAGE} = {script!r}, expected {CONSOLE_SCRIPT!r}")
    if f"{dist_info}/RECORD" not in wheel:
        report.fail("wheel: no RECORD")
    if len(report.problems) == problems_before:
        report.ok(
            f"wheel: {dist_info}, py3-none-any, so101 = {CONSOLE_SCRIPT}, "
            f"Requires-Python {REQUIRES_PYTHON}"
        )


def _specifiers(text: str) -> frozenset[str]:
    """``">=3.12, <3.14"`` -> ``{">=3.12", "<3.14"}``: a version specifier set, order-free."""
    return frozenset(part.replace(" ", "") for part in text.split(",") if part.strip())


def _check_sdist_layout(report: Report, sdist: dict[str, bytes]) -> None:
    for required in ("PKG-INFO", "pyproject.toml", "hatch_build.py"):
        if required not in sdist:
            report.fail(f"sdist: missing {required}")
    for name in sorted(sdist):
        top = name.split("/", 1)[0]
        if top not in SDIST_TOP_LEVEL:
            report.fail(f"sdist: unexpected {name} (only the package source ships)")
        elif top == "src" and not name.startswith(f"{PACKAGE_SRC}/"):
            report.fail(f"sdist: unexpected {name} (only {PACKAGE_SRC}/ ships from src/)")
        elif top == "frontend" and not name.startswith(f"{UI_SRC}/"):
            report.fail(f"sdist: unexpected {name} (only the built {UI_SRC}/ ships)")


def _check_unexpected(
    report: Report,
    label: str,
    files: dict[str, bytes],
    prefix: str,
    expected: set[str],
    origin: str,
) -> None:
    """Package files in ``files`` (under ``prefix``, the UI aside) that are not ``expected``."""
    what = (
        f"not a file git tracks under {PACKAGE_SRC} (untracked files must not ship)"
        if origin == "git ls-files"
        else f"not in {PACKAGE_SRC} on disk"
    )
    extra = [
        name
        for name in sorted(files)
        if name.startswith(prefix)
        and not name.startswith((f"{UI_WHEEL}/", f"{UI_SRC}/"))
        and name not in expected
        and forbidden_reason(name) is None  # reported as forbidden instead
    ]
    for name in extra[:20]:
        report.fail(f"{label}: {name} is {what}")
    if len(extra) > 20:
        report.fail(f"{label}: {len(extra) - 20} more unexpected files")


def _ui_files(files: dict[str, bytes], prefix: str) -> dict[str, bytes]:
    return {
        name.removeprefix(f"{prefix}/"): data
        for name, data in files.items()
        if name.startswith(f"{prefix}/")
    }


def _check_ui(
    report: Report, root: Path, ui: str, wheel: dict[str, bytes], sdist: dict[str, bytes]
) -> None:
    in_wheel, in_sdist = _ui_files(wheel, UI_WHEEL), _ui_files(sdist, UI_SRC)
    if ui == "absent":
        for name in sorted(f"{UI_WHEEL}/{n}" for n in in_wheel) + sorted(
            f"{UI_SRC}/{n}" for n in in_sdist
        ):
            report.fail(f"--ui absent, but the packages contain {name}")
        if not in_wheel and not in_sdist:
            report.ok("UI: not bundled, as required by --ui absent")
        return
    if ui == "auto" and not in_wheel and not in_sdist:
        report.ok("UI: not bundled (--ui auto)")
        return
    # Bundled (or required): it must be the stamped build of the checkout's sources.
    hb = _hatch_build()
    frontend = root / "frontend"
    problems_before = len(report.problems)
    sources = None
    if (frontend / "src").is_dir():
        sources = hb.ui_sources_digest_of(frontend)
    else:
        report.fail(
            f"{frontend / 'src'} does not exist, so the UI's build stamp cannot be checked "
            "against the frontend sources; pass the checkout as --root"
        )
    stamps = []
    for label, files, prefix in (("wheel", in_wheel, UI_WHEEL), ("sdist", in_sdist, UI_SRC)):
        if "index.html" not in files:
            report.fail(f"{label}: missing {prefix}/index.html")
            continue
        hashes = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in files.items()
            if name != hb.UI_STAMP
        }
        stamp, problems = hb.ui_build_problems(hashes, files.get(hb.UI_STAMP), sources)
        for problem in problems:
            report.fail(f"{label}: {prefix}: {problem}")
        stamps.append(stamp)
    if len(stamps) == 2 and None not in stamps and stamps[0] != stamps[1]:
        report.fail(f"the wheel's and the sdist's {hb.UI_STAMP} differ")
    if len(report.problems) == problems_before and stamps and stamps[0] is not None:
        report.ok(
            f"UI: {len(in_wheel) - 1} built files under {UI_WHEEL}/, exactly the manifest "
            f"of {hb.UI_STAMP}, built from the current frontend sources "
            f"({stamps[0]['sources'][:12]}, git {stamps[0]['git_head'] or 'unknown'})"
        )


def check(dist_dir: Path, root: Path = REPO_ROOT, ui: str = "auto") -> Report:
    report = Report()
    wheel_path, sdist_path = find_artifacts(dist_dir)
    wheel = read_wheel(wheel_path)
    sdist = read_sdist(sdist_path, report)
    report.ok(
        f"artifacts: {wheel_path.name} ({len(wheel)} files), {sdist_path.name} ({len(sdist)})"
    )

    sources, origin = package_source_files(root)
    if not sources:
        report.fail(f"no files under {root / PACKAGE_SRC}; is --root the repo root?")
    as_wheel = [name.removeprefix("src/") for name in sources]
    in_wheel = _require(report, "wheel", set(wheel), as_wheel)
    _require(report, "sdist", set(sdist), sources)
    problems_before = len(report.problems)
    _check_unexpected(report, "wheel", wheel, f"{PACKAGE}/", set(as_wheel), origin)
    _check_unexpected(report, "sdist", sdist, f"{PACKAGE_SRC}/", set(sources), origin)
    extra = "" if len(report.problems) > problems_before else ", and no other package files"
    report.ok(f"package: {in_wheel}/{len(sources)} {PACKAGE_SRC} files in both ({origin}){extra}")

    assets_dir = root / ASSETS_SRC
    if assets_dir.is_dir():
        assets = _files_on_disk(assets_dir, root)
        found = _require(report, "wheel", set(wheel), [a.removeprefix("src/") for a in assets])
        _require(report, "sdist", set(sdist), assets)
        report.ok(f"simulation assets: {found}/{len(assets)} files from {ASSETS_SRC}/ in both")
    else:
        report.ok(f"simulation assets: {ASSETS_SRC}/ does not exist here; nothing to check")

    _check_wheel_metadata(report, wheel_path, wheel)
    _check_sdist_layout(report, sdist)
    _check_ui(report, root, ui, wheel, sdist)
    _check_forbidden(report, "wheel", wheel, root)
    _check_forbidden(report, "sdist", sdist, root)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dist_dir", type=Path, help="directory holding the built wheel + sdist")
    parser.add_argument(
        "--ui",
        choices=("required", "absent", "auto"),
        default="auto",
        help="whether the built frontend must be bundled (default: auto, either way)",
    )
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repo root (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    try:
        report = check(args.dist_dir, args.root, args.ui)
    except DistError as exc:
        print(f"check_dist: {exc}", file=sys.stderr)
        return 1
    for note in report.notes:
        print(f"  ok    {note}")
    for problem in report.problems:
        print(f"  FAIL  {problem}")
    if report.problems:
        print(f"check_dist: {len(report.problems)} problem(s) in {args.dist_dir}", file=sys.stderr)
        return 1
    print(f"check_dist: {args.dist_dir} ok (--ui {args.ui})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
