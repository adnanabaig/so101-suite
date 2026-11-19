"""scripts/check_dist.py: the sdist/wheel contents gate of the CI wheel job.

The packages here are hand-built archives, so each test controls exactly what is
in them. tests/test_hatch_build.py runs the same check on real hatchling output.
The bundled UI carries the build stamp that frontend/build-stamp.ts writes,
computed here from FRONTEND_SOURCES and UI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_INFO = "so101-0.2.0.dist-info"
WHEEL = "so101-0.2.0-py3-none-any.whl"
SDIST = "so101-0.2.0.tar.gz"
SOURCES = {
    "src/so101/__init__.py": b'__version__ = "0.2.0"\n',
    "src/so101/cli.py": b"app = object()\n",
    "src/so101/simulation/scenarios.yaml": b"scenarios: []\n",
}
# The checkout's frontend sources the UI below was built from.
FRONTEND_SOURCES = {"src/main.tsx": b"export {}\n", "index.html": b"<!doctype html>\n"}
BUILT_UI = {
    "index.html": b"<!doctype html><title>SO-101</title>",
    "assets/index-abc123.js": b"console.log('ui')",
}
# Built at runtime so this file never contains a complete token-shaped string.
FAKE_TOKENS = {
    "a Hugging Face token": "hf" + "_" + "A1b2C3d4" * 5,
    "an OpenAI-style API key": "sk" + "-" + "proj-" + "x9Y8z7W6" * 5,
    "an AWS access key id": "AKIA" + "Q" * 16,
    "a GitHub token": "ghp" + "_" + "a" * 36,
    "a Slack token": "xoxb" + "-" + "1234567890-abcdef",
    "a Google API key": "AIza" + "B" * 35,
    "a private key": "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
}


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "check_dist.py"
    spec = importlib.util.spec_from_file_location("check_dist", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve types via sys.modules
    spec.loader.exec_module(module)
    return module


cd = _load_script()


def _load_hatch_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_dist_test_hb", REPO_ROOT / "hatch_build.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The UI-stamp helpers, for building test packages the way the UI build does.
hb = _load_hatch_build()


def _stamp(ui: dict[str, bytes], sources: dict[str, bytes] = FRONTEND_SOURCES) -> bytes:
    """The so101-ui-stamp.json of a build of ``sources`` that produced ``ui``."""
    stamp = {
        "format": 1,
        "sources": hb.ui_sources_digest(sources),
        "git_head": "a" * 40,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in ui.items()},
    }
    return json.dumps(stamp).encode()


UI = {**BUILT_UI, "so101-ui-stamp.json": _stamp(BUILT_UI)}


def _project(root: Path, *, ui: bool = True, sources: dict[str, bytes] = SOURCES) -> Path:
    """A checkout: the package source and the frontend sources (and its dist with ``ui``)."""
    for name, data in sources.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    for name, data in FRONTEND_SOURCES.items():
        (root / "frontend" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "frontend" / name).write_bytes(data)
    if ui:
        for name, data in UI.items():
            path = root / "frontend" / "dist" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return root


def _wheel_files(*, ui: bool = True) -> dict[str, bytes]:
    files = {name.removeprefix("src/"): data for name, data in SOURCES.items()}
    if ui:
        files.update({f"so101/web/{name}": data for name, data in UI.items()})
    files[f"{DIST_INFO}/METADATA"] = (
        b"Metadata-Version: 2.4\nName: so101\nVersion: 0.2.0\nRequires-Python: <3.14,>=3.12\n"
    )
    files[f"{DIST_INFO}/entry_points.txt"] = b"[console_scripts]\nso101 = so101.cli:app\n"
    files[f"{DIST_INFO}/WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[f"{DIST_INFO}/RECORD"] = b""
    return files


def _sdist_files(*, ui: bool = True) -> dict[str, bytes]:
    files = dict(SOURCES)
    if ui:
        files.update({f"frontend/dist/{name}": data for name, data in UI.items()})
    files.update(
        {
            "PKG-INFO": b"Metadata-Version: 2.4\nName: so101\n",
            "pyproject.toml": b"[project]\nname = 'so101'\n",
            "hatch_build.py": b"",
            "README.md": b"# SO-101\n",
            "LICENSE": b"MIT\n",
            ".gitignore": b"dist/\n",
        }
    )
    return files


def _write_dist(
    dist: Path,
    wheel: dict[str, bytes],
    sdist: dict[str, bytes],
    *,
    wheel_name: str = WHEEL,
    sdist_name: str = SDIST,
) -> Path:
    dist.mkdir(exist_ok=True)
    with zipfile.ZipFile(dist / wheel_name, "w") as archive:
        for name, data in wheel.items():
            archive.writestr(name, data)
    top = sdist_name.removesuffix(".tar.gz")
    with tarfile.open(dist / sdist_name, "w:gz") as archive:
        for name, data in sdist.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return dist


def _check(
    tmp_path: Path,
    wheel=None,
    sdist=None,
    *,
    ui="required",
    project_ui=True,
    sources=SOURCES,
    **names,
):
    root = _project(tmp_path / "repo", ui=project_ui, sources=sources)
    dist = _write_dist(
        tmp_path / "dist",
        _wheel_files() if wheel is None else wheel,
        _sdist_files() if sdist is None else sdist,
        **names,
    )
    return cd.check(dist, root, ui)


def test_complete_packages_pass(tmp_path):
    report = _check(tmp_path)
    assert report.problems == []
    notes = "\n".join(report.notes)
    assert "package: 3/3 src/so101 files in both" in notes and "no other package files" in notes
    assert (
        "UI: 2 built files under so101/web/, exactly the manifest of so101-ui-stamp.json" in notes
    )
    assert "src/so101/simulation/assets/ does not exist here" in notes


def test_a_package_file_missing_from_either_archive_fails(tmp_path):
    wheel = _wheel_files()
    del wheel["so101/cli.py"]
    sdist = _sdist_files()
    del sdist["src/so101/simulation/scenarios.yaml"]
    assert _check(tmp_path, wheel, sdist).problems == [
        "wheel: missing so101/cli.py",
        "sdist: missing src/so101/simulation/scenarios.yaml",
    ]


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("so101/tests/helpers.py", "inside a tests/ directory"),
        ("so101/conftest.py", "a file that never ships"),
        ("so101/test_cli.py", "a test file"),
        ("so101/cli_test.py", "a test file"),
        ("so101/.env", "an environment file"),
        ("so101/.env.local", "an environment file"),
        ("so101/web/node_modules/react/index.js", "inside a node_modules/ directory"),
        ("so101/__pycache__/cli.cpython-312.pyc", "inside a __pycache__/ directory"),
        ("so101/id_ed25519", "an SSH key"),
        ("so101/certs/server.pem", "a .pem file"),
        ("so101/data/episode_000.parquet", "a .parquet file"),
        ("so101/models/policy.safetensors", "a .safetensors file"),
        ("so101/.pypirc", "a file that never ships"),
    ],
)
def test_files_that_must_never_ship_fail(tmp_path, name, reason):
    wheel = {**_wheel_files(), name: b"x"}
    expected = {f"wheel: {name} is {reason}"}
    if name.startswith("so101/web/"):  # also outside the UI build's manifest
        expected.add(
            f"wheel: so101/web: {name.removeprefix('so101/web/')} is not in the build's manifest"
        )
    problems = _check(tmp_path, wheel).problems
    assert sorted(problems) == sorted(expected)


def test_env_example_may_ship(tmp_path):
    example = b"SO101_MOCK=1\n"
    wheel = {**_wheel_files(), "so101/.env.example": example}
    sdist = {**_sdist_files(), "src/so101/.env.example": example}
    sources = {**SOURCES, "src/so101/.env.example": example}
    assert _check(tmp_path, wheel, sdist, sources=sources).problems == []


@pytest.mark.parametrize("label", sorted(FAKE_TOKENS))
def test_text_that_looks_like_a_secret_fails(tmp_path, label):
    bundle = f"const config = {{ key: '{FAKE_TOKENS[label]}' }};".encode()
    wheel = {**_wheel_files(), "so101/web/assets/index-abc123.js": bundle}
    sdist = {**_sdist_files(), "frontend/dist/assets/index-abc123.js": bundle}
    problems = _check(tmp_path, wheel, sdist).problems
    assert f"wheel: so101/web/assets/index-abc123.js contains {label}" in problems
    assert f"sdist: frontend/dist/assets/index-abc123.js contains {label}" in problems


def test_secret_scan_skips_binary_files(tmp_path):
    mesh = b"\0binary STL\0" + FAKE_TOKENS["an AWS access key id"].encode()
    wheel = {**_wheel_files(), "so101/simulation/mesh.stl": mesh}
    sdist = {**_sdist_files(), "src/so101/simulation/mesh.stl": mesh}
    sources = {**SOURCES, "src/so101/simulation/mesh.stl": mesh}
    assert _check(tmp_path, wheel, sdist, sources=sources).problems == []


def test_a_secret_value_from_the_checkouts_env_file_fails_without_being_printed(tmp_path):
    secret = "correct-horse-battery-staple"
    root = _project(tmp_path / "repo")
    (root / ".env").write_text(f"SO101_OPENAI_API_KEY='{secret}'\nSO101_MOCK=1\n")
    wheel = {**_wheel_files(), "so101/cli.py": f"KEY = '{secret}'\n".encode()}
    dist = _write_dist(tmp_path / "dist", wheel, _sdist_files())
    problems = cd.check(dist, root, "required").problems
    assert problems == ["wheel: so101/cli.py contains the value of SO101_OPENAI_API_KEY from .env"]
    assert not any(secret in problem for problem in problems)


def test_required_ui_missing_from_the_packages_fails(tmp_path):
    problems = _check(tmp_path, _wheel_files(ui=False), _sdist_files(ui=False)).problems
    assert problems == [
        "wheel: missing so101/web/index.html",
        "sdist: missing frontend/dist/index.html",
    ]


def test_the_ui_is_checked_against_its_stamp_not_the_checkouts_dist(tmp_path):
    # The packages carry the build and its manifest; whatever frontend/dist
    # holds now (here nothing) does not matter.
    assert _check(tmp_path, project_ui=False).problems == []


def test_a_bundled_ui_needs_the_frontend_sources_in_the_checkout(tmp_path):
    root = _project(tmp_path / "repo")
    shutil.rmtree(root / "frontend" / "src")
    dist = _write_dist(tmp_path / "dist", _wheel_files(), _sdist_files())
    problems = cd.check(dist, root, "required").problems
    assert len(problems) == 1
    assert "frontend/src does not exist, so the UI's build stamp cannot be checked" in problems[0]


# --- The UI build stamp (review finding M12) ------------------------------------------


def _with_ui(ui: dict[str, bytes]) -> tuple[dict[str, bytes], dict[str, bytes]]:
    wheel = {n: d for n, d in _wheel_files().items() if not n.startswith("so101/web/")}
    sdist = {n: d for n, d in _sdist_files().items() if not n.startswith("frontend/dist/")}
    wheel.update({f"so101/web/{name}": data for name, data in ui.items()})
    sdist.update({f"frontend/dist/{name}": data for name, data in ui.items()})
    return wheel, sdist


def test_a_stale_index_and_a_leftover_chunk_fail(tmp_path):
    # The reviewer's repro: check_dist used to print "UI: 48/48 ... dist ok".
    ui = {
        **UI,
        "index.html": b"<!doctype html><title>stale</title>",
        "assets/old-chunk-DEADBEEF.js": b"old",
    }
    problems = _check(tmp_path, *_with_ui(ui)).problems
    for label, prefix in (("wheel", "so101/web"), ("sdist", "frontend/dist")):
        assert f"{label}: {prefix}: index.html differs from the build's manifest" in problems
        assert (
            f"{label}: {prefix}: assets/old-chunk-DEADBEEF.js is not in the build's manifest"
            in problems
        )


def test_a_ui_built_from_other_sources_fails(tmp_path):
    older = {**FRONTEND_SOURCES, "src/main.tsx": b"export const old = 1\n"}
    ui = {**BUILT_UI, "so101-ui-stamp.json": _stamp(BUILT_UI, older)}
    problems = _check(tmp_path, *_with_ui(ui)).problems
    assert len(problems) == 2
    assert all("built from other frontend sources" in problem for problem in problems)


def test_a_ui_without_its_stamp_fails(tmp_path):
    problems = _check(tmp_path, *_with_ui(BUILT_UI)).problems
    assert problems == [
        "wheel: so101/web: no so101-ui-stamp.json: the UI was not built by `npm run build` in "
        "this checkout (or by an older one without the build stamp)",
        "sdist: frontend/dist: no so101-ui-stamp.json: the UI was not built by `npm run build` "
        "in this checkout (or by an older one without the build stamp)",
    ]


def test_the_wheel_and_sdist_must_carry_the_same_stamp(tmp_path):
    wheel, _ = _with_ui(UI)
    other = {**BUILT_UI, "so101-ui-stamp.json": _stamp(BUILT_UI).replace(b"a" * 40, b"b" * 40)}
    _, sdist = _with_ui(other)
    assert _check(tmp_path, wheel, sdist).problems == [
        "the wheel's and the sdist's so101-ui-stamp.json differ"
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_an_untracked_package_file_in_the_packages_fails(tmp_path):
    root = _project(tmp_path / "repo")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", *SOURCES], check=True, capture_output=True)
    (root / "src" / "so101" / "zz_local_hack.py").write_bytes(b"print('hack')\n")
    wheel = {**_wheel_files(), "so101/zz_local_hack.py": b"print('hack')\n"}
    sdist = {**_sdist_files(), "src/so101/zz_local_hack.py": b"print('hack')\n"}
    dist = _write_dist(tmp_path / "dist", wheel, sdist)
    assert cd.check(dist, root, "required").problems == [
        "wheel: so101/zz_local_hack.py is not a file git tracks under src/so101 "
        "(untracked files must not ship)",
        "sdist: src/so101/zz_local_hack.py is not a file git tracks under src/so101 "
        "(untracked files must not ship)",
    ]


def test_outside_git_a_package_file_not_on_disk_fails(tmp_path):
    wheel = {**_wheel_files(), "so101/ghost.py": b""}
    assert _check(tmp_path, wheel).problems == ["wheel: so101/ghost.py is not in src/so101 on disk"]


def test_absent_ui_must_not_be_bundled(tmp_path):
    problems = _check(tmp_path, ui="absent").problems
    assert "--ui absent, but the packages contain so101/web/index.html" in problems
    assert "--ui absent, but the packages contain frontend/dist/index.html" in problems
    assert _check(tmp_path / "b", _wheel_files(ui=False), _sdist_files(ui=False), ui="absent")


def test_absent_ui_passes_without_one(tmp_path):
    report = _check(tmp_path, _wheel_files(ui=False), _sdist_files(ui=False), ui="absent")
    assert report.problems == []


@pytest.mark.parametrize("ui", [True, False])
def test_auto_accepts_packages_with_or_without_the_ui(tmp_path, ui):
    report = _check(tmp_path, _wheel_files(ui=ui), _sdist_files(ui=ui), ui="auto")
    assert report.problems == []


def test_auto_still_requires_index_html_when_a_ui_is_bundled(tmp_path):
    wheel = _wheel_files()
    del wheel["so101/web/index.html"]
    assert _check(tmp_path, wheel, ui="auto").problems == ["wheel: missing so101/web/index.html"]


def test_auto_checks_a_bundled_ui_against_its_stamp_too(tmp_path):
    ui = {**UI, "assets/old-chunk-DEADBEEF.js": b"old"}
    problems = _check(tmp_path, *_with_ui(ui), ui="auto").problems
    assert "wheel: so101/web: assets/old-chunk-DEADBEEF.js is not in the build's manifest" in (
        problems
    )


def test_every_simulation_asset_on_disk_must_ship(tmp_path):
    root = _project(tmp_path / "repo")
    assets = root / "src" / "so101" / "simulation" / "assets"
    (assets / "meshes").mkdir(parents=True)
    (assets / "so101.xml").write_bytes(b"<mujoco/>")
    (assets / "meshes" / "base.stl").write_bytes(b"\0\0mesh")
    (assets / ".DS_Store").write_bytes(b"junk")  # never expected
    wheel = {**_wheel_files(), "so101/simulation/assets/so101.xml": b"<mujoco/>"}
    sdist = {**_sdist_files(), "src/so101/simulation/assets/so101.xml": b"<mujoco/>"}
    dist = _write_dist(tmp_path / "dist", wheel, sdist)
    report = cd.check(dist, root, "required")
    assert "wheel: missing so101/simulation/assets/meshes/base.stl" in report.problems
    assert "sdist: missing src/so101/simulation/assets/meshes/base.stl" in report.problems
    assert "simulation assets: 1/2 files from src/so101/simulation/assets/ in both" in report.notes


def test_only_the_package_and_its_build_files_may_ship(tmp_path):
    wheel = {**_wheel_files(), "scripts/smoke.py": b"", "frontend/index.html": b""}
    sdist = {
        **_sdist_files(),
        "docs/guide.md": b"",
        "frontend/src/main.tsx": b"",
        "src/other/x.py": b"",
    }
    problems = _check(tmp_path, wheel, sdist).problems
    assert "wheel: unexpected top-level entry scripts" in problems
    assert "wheel: unexpected top-level entry frontend" in problems
    assert "sdist: unexpected docs/guide.md (only the package source ships)" in problems
    assert (
        "sdist: unexpected frontend/src/main.tsx (only the built frontend/dist/ ships)" in problems
    )
    assert "sdist: unexpected src/other/x.py (only src/so101/ ships from src/)" in problems


def test_wheel_metadata_must_match_the_project(tmp_path):
    wheel = _wheel_files()
    wheel[f"{DIST_INFO}/METADATA"] = b"Name: so101\nRequires-Python: >=3.11\n"
    wheel[f"{DIST_INFO}/entry_points.txt"] = b"[console_scripts]\nso101 = so101.main:run\n"
    problems = _check(
        tmp_path, wheel, wheel_name="so101-0.2.0-cp312-cp312-macosx_11_0_arm64.whl"
    ).problems
    assert any("is not a pure-Python (py3-none-any) wheel" in p for p in problems)
    assert "wheel: METADATA Requires-Python is '>=3.11', expected '>=3.12,<3.14'" in problems
    assert "wheel: console script so101 = 'so101.main:run', expected 'so101.cli:app'" in problems


@pytest.mark.parametrize(
    ("requires", "ok"),
    [("<3.14,>=3.12", True), (">=3.12, <3.14", True), (">=3.12", False), (">=3.12,<3.15", False)],
)
def test_requires_python_needs_both_bounds_in_any_order(tmp_path, requires, ok):
    wheel = _wheel_files()
    wheel[f"{DIST_INFO}/METADATA"] = f"Name: so101\nRequires-Python: {requires}\n".encode()
    problems = _check(tmp_path, wheel).problems
    assert any("Requires-Python" in problem for problem in problems) is not ok


def test_sdist_members_outside_its_top_level_directory_fail(tmp_path):
    dist = _write_dist(tmp_path / "dist", _wheel_files(), _sdist_files())
    with tarfile.open(dist / SDIST, "w:gz") as archive:
        info = tarfile.TarInfo("evil/payload.py")
        archive.addfile(info, io.BytesIO(b""))
    report = cd.check(dist, _project(tmp_path / "repo"), "auto")
    assert "sdist: evil/payload.py is outside so101-0.2.0/" in report.problems


@pytest.mark.parametrize(
    "names",
    [[WHEEL], [SDIST], [WHEEL, SDIST, "so101-0.0.9-py3-none-any.whl"], []],
)
def test_the_dist_dir_must_hold_exactly_one_wheel_and_one_sdist(tmp_path, capsys, names):
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in names:
        (dist / name).write_bytes(b"")
    assert cd.main([str(dist), "--root", str(_project(tmp_path / "repo"))]) == 1
    assert "expected one so101 wheel and one sdist" in capsys.readouterr().err


def test_main_prints_every_check_and_exits_zero_when_clean(tmp_path, capsys):
    root = _project(tmp_path / "repo")
    dist = _write_dist(tmp_path / "dist", _wheel_files(), _sdist_files())
    assert cd.main([str(dist), "--ui", "required", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert out.count("  ok    ") >= 7
    assert "FAIL" not in out
    assert "ok (--ui required)" in out


def test_main_lists_each_problem_and_exits_one(tmp_path, capsys):
    root = _project(tmp_path / "repo")
    wheel = {**_wheel_files(), "so101/.env": b"SO101_MOCK=1\n"}
    dist = _write_dist(tmp_path / "dist", wheel, _sdist_files())
    assert cd.main([str(dist), "--root", str(root)]) == 1
    captured = capsys.readouterr()
    assert "  FAIL  wheel: so101/.env is an environment file" in captured.out
    assert "check_dist: 1 problem(s)" in captured.err


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_in_a_git_checkout_only_tracked_files_are_expected(tmp_path):
    root = _project(tmp_path / "repo")
    (root / "src" / "so101" / "scratch.py").write_bytes(b"# untracked\n")

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    git("init", "-q")
    git("add", *SOURCES)
    files, origin = cd.package_source_files(root)
    assert origin == "git ls-files"
    assert sorted(files) == sorted(SOURCES)
    # Not a git checkout (an exported tree): everything on disk is expected.
    shutil.rmtree(root / ".git")
    files, origin = cd.package_source_files(root)
    assert origin.startswith("files on disk")
    assert "src/so101/scratch.py" in files


def test_check_dist_runs_where_hatchling_is_not_installed(tmp_path):
    # The CI wheel job installs only `build` (hatchling runs in its isolated
    # environment), and check_dist uses hatch_build.py's UI-stamp helpers.
    root = _project(tmp_path / "repo")
    dist = _write_dist(tmp_path / "dist", _wheel_files(), _sdist_files())
    script = REPO_ROOT / "scripts" / "check_dist.py"
    argv = ["check_dist.py", str(dist), "--ui", "required", "--root", str(root)]
    code = (
        f"import runpy, sys; sys.modules['hatchling'] = None; sys.argv = {argv!r}; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok (--ui required)" in result.stdout
