"""hatch_build.py and the [tool.hatch] config: what goes into the sdist and wheel.

The hook tests call UiBuildHook directly. The build tests run hatchling in-process
on a temp copy of the project (no network, no pip), sdist first and then the
wheel from the unpacked sdist, the way `python -m build` does it, and then run
scripts/check_dist.py on the result.

A test UI build writes the build stamp itself, the way frontend/build-stamp.ts
does after `npm run build`; the known-answer test pins the digest both compute.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
from hatchling.builders.sdist import SdistBuilder
from hatchling.builders.wheel import WheelBuilder

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VERSION = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))["project"][
    "version"
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve types via sys.modules
    spec.loader.exec_module(module)
    return module


hatch_build = _load("so101_hatch_build", REPO_ROOT / "hatch_build.py")
check_dist = _load("check_dist", REPO_ROOT / "scripts" / "check_dist.py")


def write_stamp(frontend: Path, git_head: str | None = None) -> dict:
    """What frontend/build-stamp.ts writes at the end of `npm run build`."""
    stamp = {
        "format": 1,
        "sources": hatch_build.ui_sources_digest_of(frontend),
        "git_head": git_head,
        "files": hatch_build.dist_hashes(frontend / "dist"),
    }
    (frontend / "dist" / hatch_build.UI_STAMP).write_text(json.dumps(stamp), encoding="utf-8")
    return stamp


def _build_ui(root: Path) -> Path:
    frontend = root / "frontend"
    main = frontend / "src" / "main.tsx"
    if not main.exists():
        main.parent.mkdir(parents=True)
        main.write_text("export {}\n", encoding="utf-8")
    dist = frontend / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>SO-101</title>", encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log('ui')", encoding="utf-8")
    write_stamp(frontend)
    return dist


def _run_hook(root: Path, target: str, version: str = "standard") -> dict[str, str]:
    hook = hatch_build.UiBuildHook(str(root), {}, None, None, str(root / "out"), target)
    build_data: dict = {"force_include": {}}
    hook.initialize(version, build_data)
    return build_data["force_include"]


def test_wheel_bundles_the_built_ui_as_so101_web(tmp_path):
    dist = _build_ui(tmp_path)
    assert _run_hook(tmp_path, "wheel") == {str(dist): "so101/web"}


def test_sdist_keeps_the_built_ui_at_frontend_dist(tmp_path):
    # A wheel built from this sdist finds frontend/dist at the sdist root.
    dist = _build_ui(tmp_path)
    assert _run_hook(tmp_path, "sdist") == {str(dist): "frontend/dist"}


def test_editable_installs_do_not_copy_the_ui(tmp_path):
    _build_ui(tmp_path)
    assert _run_hook(tmp_path, "wheel", version="editable") == {}


def test_other_build_targets_are_left_alone(tmp_path):
    _build_ui(tmp_path)
    assert _run_hook(tmp_path, "binary") == {}


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_missing_ui_warns_and_builds_without_it(tmp_path, capsys, target):
    assert _run_hook(tmp_path, target) == {}
    err = capsys.readouterr().err
    assert "frontend/dist/index.html does not exist" in err
    assert f"Building the {target} without the operator UI" in err
    assert "SO101_REQUIRE_UI=1" in err


def test_a_dist_dir_without_index_html_counts_as_not_built(tmp_path, capsys):
    (tmp_path / "frontend" / "dist" / "assets").mkdir(parents=True)
    assert _run_hook(tmp_path, "wheel") == {}
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_missing_ui_fails_the_build_when_required(tmp_path, capsys, monkeypatch, target):
    monkeypatch.setenv("SO101_REQUIRE_UI", "1")
    with pytest.raises(SystemExit) as exc:
        _run_hook(tmp_path, target)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "error: SO101_REQUIRE_UI=1 and frontend/dist/index.html does not exist" in err
    assert "make build-ui" in err


def test_required_ui_that_is_built_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("SO101_REQUIRE_UI", "1")
    dist = _build_ui(tmp_path)
    assert _run_hook(tmp_path, "wheel") == {str(dist): "so101/web"}


# --- The UI build stamp (review finding M12) ------------------------------------------

# The same tree and known answers as frontend/build-stamp.test.ts.
STAMP_FIXTURE = {
    "index.html": "<!doctype html><title>SO-101</title>\n",
    "package.json": '{"name": "so101-frontend"}\n',
    "src/main.tsx": "export {};\n",
    "src/lib/a.ts": "export const a = 1;\n",
    "src/.DS_Store": "junk",
    "public/manifest.webmanifest": "{}\n",
    "README.md": "not an input\n",
    "node_modules/x/index.js": "not an input\n",
    "dist/index.html": "<html></html>\n",
    "dist/assets/index-abc123.js": "console.log(1);\n",
    "dist/so101-ui-stamp.json": "{}\n",
}
SOURCES_DIGEST = "c778c8d01b2435e5ac63ba58349054c906b9aac296f8be8f4c7fb5513763a2d1"
MANIFEST = {
    "assets/index-abc123.js": "b603d946eb2b396ca4ecf65c223daff659dbe6f1cfeac235b7c61d3ba6964cae",
    "index.html": "b0693dc92f76e08bf1485b3dd9b514a2e31dfd6f39422a6b60edb722671dc98f",
}


def test_the_stamp_digest_is_the_one_build_stamp_ts_computes(tmp_path):
    for name, text in STAMP_FIXTURE.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert hatch_build.ui_sources_digest_of(tmp_path) == SOURCES_DIGEST
    contents = {name: text.encode() for name, text in STAMP_FIXTURE.items()}
    assert hatch_build.ui_sources_digest(contents) == SOURCES_DIGEST  # from an archive
    assert hatch_build.dist_hashes(tmp_path / "dist") == MANIFEST


def test_the_stamp_constants_match_build_stamp_ts():
    source = (REPO_ROOT / "frontend" / "build-stamp.ts").read_text(encoding="utf-8")
    assert f'export const UI_STAMP = "{hatch_build.UI_STAMP}";' in source
    for name in hatch_build.UI_SOURCE_DIRS + hatch_build.UI_SOURCE_FILES:
        assert f'"{name}"' in source, name
    # The build stamp's own source and the config that loads it are inputs too.
    assert {"build-stamp.ts", "vite.config.ts"} <= set(hatch_build.UI_SOURCE_FILES)
    assert "so101BuildStamp(FRONTEND)" in (REPO_ROOT / "frontend" / "vite.config.ts").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_a_ui_built_from_older_sources_fails_the_build(tmp_path, capsys, target):
    _build_ui(tmp_path)
    (tmp_path / "frontend" / "src" / "main.tsx").write_text("export const x = 1\n")
    with pytest.raises(SystemExit):
        _run_hook(tmp_path, target)
    err = capsys.readouterr().err
    assert "frontend/dist is not a build of the current frontend sources" in err
    assert "built from other frontend sources" in err and "make build-ui" in err


def test_a_ui_without_a_stamp_fails_the_build(tmp_path, capsys):
    dist = _build_ui(tmp_path)
    (dist / hatch_build.UI_STAMP).unlink()
    with pytest.raises(SystemExit):
        _run_hook(tmp_path, "wheel")
    assert "no so101-ui-stamp.json" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda dist: (dist / "assets" / "old-chunk-DEADBEEF.js").write_text("old"),
            "assets/old-chunk-DEADBEEF.js is not in the build's manifest",
        ),
        (
            lambda dist: (dist / "index.html").write_text("<!doctype html><title>old</title>"),
            "index.html differs from the build's manifest",
        ),
        (
            lambda dist: (dist / "assets" / "index-abc123.js").unlink(),
            "assets/index-abc123.js is in the build's manifest but missing",
        ),
        (
            lambda dist: (dist / hatch_build.UI_STAMP).write_text("{not json"),
            "so101-ui-stamp.json is not a valid build stamp",
        ),
    ],
    ids=["extra-chunk", "stale-index", "missing-asset", "corrupt-stamp"],
)
def test_a_dist_that_is_not_exactly_the_stamped_build_fails(tmp_path, capsys, change, message):
    dist = _build_ui(tmp_path)
    change(dist)
    with pytest.raises(SystemExit):
        _run_hook(tmp_path, "wheel")
    assert message in capsys.readouterr().err


def test_from_an_unpacked_sdist_only_the_manifest_is_checked(tmp_path, capsys):
    # An sdist has frontend/dist but not frontend/src: the sdist build checked
    # the sources digest, and the manifest still guards the files.
    dist = _build_ui(tmp_path)
    shutil.rmtree(tmp_path / "frontend" / "src")
    assert _run_hook(tmp_path, "wheel") == {str(dist): "so101/web"}
    (dist / "assets" / "old-chunk-DEADBEEF.js").write_text("old")
    with pytest.raises(SystemExit):
        _run_hook(tmp_path, "wheel")
    assert "old-chunk-DEADBEEF.js is not in the build's manifest" in capsys.readouterr().err


# --- Package files in a git checkout (review finding M12) ------------------------------

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env={**os.environ, **GIT_ENV},
    )


def _git_checkout(root: Path) -> Path:
    package = root / "src" / "so101"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (root / ".gitignore").write_text("*.local.py\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return package


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_an_untracked_package_file_fails_the_build(tmp_path, capsys, target):
    package = _git_checkout(tmp_path)
    (package / "zz_local_hack.py").write_text("print('hack')\n")
    with pytest.raises(SystemExit):
        _run_hook(tmp_path, target)
    err = capsys.readouterr().err
    assert "git does not track src/so101/zz_local_hack.py" in err
    assert "Commit it, delete it, or add it to .gitignore" in err


def test_ignored_and_tracked_package_files_build_and_changes_only_warn(tmp_path, capsys):
    package = _git_checkout(tmp_path)
    (package / "settings.local.py").write_text("x = 1\n")  # ignored: hatchling skips it too
    assert _run_hook(tmp_path, "wheel") == {}
    assert "git does not track" not in capsys.readouterr().err
    (package / "__init__.py").write_text('__version__ = "0.2.1"\n')
    _run_hook(tmp_path, "wheel")
    err = capsys.readouterr().err
    assert "uncommitted changes in src/so101/__init__.py" in err


def test_outside_a_git_checkout_the_package_files_are_not_checked(tmp_path, capsys):
    (tmp_path / "src" / "so101").mkdir(parents=True)
    (tmp_path / "src" / "so101" / "zz_local_hack.py").write_text("")
    assert _run_hook(tmp_path, "wheel") == {}
    assert "git does not track" not in capsys.readouterr().err


# --- Real hatchling builds of a temp copy of the project ----------------------------


def _project_copy(root: Path) -> Path:
    root.mkdir()
    for name in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE", ".gitignore"):
        shutil.copy2(REPO_ROOT / name, root / name)
    shutil.copytree(
        REPO_ROOT / "src" / "so101",
        root / "src" / "so101",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # Things that must never ship.
    (root / "tests").mkdir()
    (root / "tests" / "test_example.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("# notes\n", encoding="utf-8")
    (root / ".env").write_text("SO101_HF_TOKEN=hf_not_a_real_token\n", encoding="utf-8")
    (root / "frontend" / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "frontend" / "node_modules" / "left-pad" / "index.js").write_text("", encoding="utf-8")
    (root / "frontend" / "src").mkdir(parents=True)
    (root / "frontend" / "src" / "main.tsx").write_text("export {}\n", encoding="utf-8")
    return root


def _build_sdist_then_wheel(root: Path, out: Path) -> tuple[Path, Path]:
    out.mkdir()
    sdist = Path(next(SdistBuilder(str(root)).build(directory=str(out))))
    unpacked = out / "unpacked"
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    (sdist_root,) = unpacked.iterdir()
    wheel = Path(next(WheelBuilder(str(sdist_root)).build(directory=str(out))))
    return sdist, wheel


def _sdist_names(sdist: Path) -> set[str]:
    with tarfile.open(sdist) as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
    return {name.split("/", 1)[1] for name in names}


def _wheel_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return {name for name in archive.namelist() if not name.endswith("/")}


def test_built_packages_carry_the_ui_and_nothing_from_outside_the_package(tmp_path):
    root = _project_copy(tmp_path / "project")
    _build_ui(root)
    sdist, wheel = _build_sdist_then_wheel(root, tmp_path / "dist")

    sdist_names = _sdist_names(sdist)
    assert {"pyproject.toml", "hatch_build.py", "PKG-INFO", "src/so101/cli.py"} <= sdist_names
    assert {"frontend/dist/index.html", "frontend/dist/assets/index-abc123.js"} <= sdist_names
    top_level = {name.split("/", 1)[0] for name in sdist_names}
    assert not top_level & {"tests", "docs", "scripts", ".env"}
    assert not any(
        name.startswith(("frontend/src", "frontend/node_modules")) for name in sdist_names
    )

    wheel_names = _wheel_names(wheel)
    assert {"so101/web/index.html", "so101/web/assets/index-abc123.js"} <= wheel_names
    assert "so101/cli.py" in wheel_names
    assert "so101/simulation/scenarios.yaml" in wheel_names  # package data ships too
    assert {name.split("/", 1)[0] for name in wheel_names} == {
        "so101",
        f"so101-{PROJECT_VERSION}.dist-info",
    }

    report = check_dist.check(tmp_path / "dist", root, "required")
    assert report.problems == []


def test_a_real_build_refuses_the_reviewers_repro(tmp_path):
    # A stale index.html plus a leftover chunk shipped, and so did an untracked
    # zz_local_hack.py; now each one stops the build.
    root = _project_copy(tmp_path / "project")
    dist = _build_ui(root)
    (dist / "assets" / "old-chunk-DEADBEEF.js").write_text("old", encoding="utf-8")
    with pytest.raises(SystemExit):
        _build_sdist_then_wheel(root, tmp_path / "dist1")
    (dist / "assets" / "old-chunk-DEADBEEF.js").unlink()
    _git(root, "init", "-q")
    _git(root, "add", "src", "pyproject.toml", "hatch_build.py", "README.md", "LICENSE")
    _git(root, "commit", "-q", "-m", "init")
    (root / "src" / "so101" / "zz_local_hack.py").write_text("print('hack')\n")
    with pytest.raises(SystemExit):
        _build_sdist_then_wheel(root, tmp_path / "dist2")


def test_built_packages_without_a_ui_are_ui_less_not_broken(tmp_path, capsys):
    root = _project_copy(tmp_path / "project")
    sdist, wheel = _build_sdist_then_wheel(root, tmp_path / "dist")
    assert not any(name.startswith("frontend/") for name in _sdist_names(sdist))
    wheel_names = _wheel_names(wheel)
    assert "so101/cli.py" in wheel_names
    assert not any(name.startswith("so101/web/") for name in wheel_names)
    assert "without the operator UI" in capsys.readouterr().err

    report = check_dist.check(tmp_path / "dist", root, "absent")
    assert report.problems == []
