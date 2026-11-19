"""Makefile contracts: the stop targets, the server address, and what CI runs.

Most checks use `make -n` (print the commands, run nothing), so no server or
hardware is involved.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "Makefile"
STOPS = ("estop", "halt", "collapse")
CI_TARGETS = ("ci", "ci-backend", "ci-thor-worker", "ci-frontend", "ci-smoke", "ci-wheel")

# The variables a caller may override (`NAME ?= ...`), e.g. PY and SMOKE_PORT.
OVERRIDABLE = frozenset(
    re.findall(r"^([A-Za-z_]\w*)[ \t]*\?=", MAKEFILE.read_text(encoding="utf-8"), re.M)
)
# A parent make (e.g. `make ci SMOKE_PORT=8814`) exports these: its flags and
# jobserver state, and every variable set on its command line.
INHERITED = frozenset({"MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEFILES"}) | OVERRIDABLE

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="GNU make is not installed")


def _make(*args: str, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None):
    # SO101_HOME stays: it points at the test's own data folder, so a recipe
    # that reads settings never sees the developer's <data home>/.env.
    clean = {
        key: value
        for key, value in os.environ.items()
        if key not in INHERITED and (key == "SO101_HOME" or not key.startswith("SO101_"))
    }
    return subprocess.run(
        ["make", *args],
        cwd=cwd,
        env={**clean, **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
    )


def _dry_run(*args: str, env: dict[str, str] | None = None) -> str:
    proc = _make("-n", *args, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def _help_sections() -> dict[str, dict[str, str]]:
    proc = _make("help")
    assert proc.returncode == 0, proc.stderr
    text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
    sections: dict[str, dict[str, str]] = {}
    current = ""
    for line in text.splitlines():
        if line and not line.startswith(" "):
            current = line.strip()
            sections[current] = {}
        elif line.strip():
            target, _, description = line.strip().partition(" ")
            sections.setdefault(current, {})[target] = description.strip()
    return sections


def test_help_lists_each_stop_under_a_safety_heading_with_its_motor_effect():
    sections = _help_sections()
    safety = [heading for heading in sections if heading.startswith("Safety")]
    assert len(safety) == 1, sections.keys()
    targets = sections[safety[0]]
    assert list(targets) == list(STOPS)
    for stop in STOPS:
        assert "motor" in targets[stop], f"`make help` must say what {stop} does to the motors"
    assert "torque" in targets["estop"]
    assert "hold" in targets["halt"]
    assert "rest pose" in targets["collapse"]


@pytest.mark.parametrize("stop", STOPS)
@pytest.mark.parametrize(
    ("variables", "env", "url"),
    [
        ((), {}, "http://127.0.0.1:8000"),
        (("HOST=0.0.0.0",), {}, "http://127.0.0.1:8000"),
        (("HOST=::", "PORT=9001"), {}, "http://127.0.0.1:9001"),
        (("HOST=192.168.1.20", "PORT=8800"), {}, "http://192.168.1.20:8800"),
        (("HOST=::1",), {}, "http://[::1]:8000"),
        ((), {"SO101_HOST": "0.0.0.0", "SO101_PORT": "8123"}, "http://127.0.0.1:8123"),
        ((), {"SO101_HOST": "10.0.0.5", "SO101_PORT": "8123"}, "http://10.0.0.5:8123"),
    ],
)
def test_stop_targets_call_the_cli_against_the_configured_server(stop, variables, env, url):
    out = _dry_run(stop, *variables, env=env)
    assert re.search(rf"-m so101\.cli {stop} --url {re.escape(url)} ", out), out
    # The failure path is loud: it names the URL and says what to do next.
    assert f"NOT CONFIRMED by {url}" in out


def test_server_targets_and_smoke_share_the_configured_address():
    out = _dry_run("api", "suite", "smoke", env={"SO101_HOST": "0.0.0.0", "SO101_PORT": "8123"})
    assert out.count("--host 0.0.0.0 --port 8123") == 2  # api and suite bind it
    assert "smoke_suite.py --base http://127.0.0.1:8123" in out  # the client uses loopback


def test_stop_targets_are_never_run_by_ci():
    for target in CI_TARGETS:
        out = _dry_run(target)  # recursive $(MAKE) calls are expanded under -n
        for stop in STOPS:
            assert not re.search(rf"(so101\.cli|so101|make) +{stop}\b", out), (target, stop)


def test_api_never_auto_reloads_and_api_dev_is_mock_only():
    assert "--reload" not in _dry_run("api")
    assert "--reload" not in _dry_run("suite")
    dev = _dry_run("api-dev")
    assert "--reload" in dev
    assert "Settings().mock" in dev  # the guard runs before uvicorn
    assert dev.index("Settings().mock") < dev.index("--reload")


@pytest.mark.parametrize(("mock", "exit_code"), [("0", 2), ("1", 0)])
def test_dev_checks_for_mock_hardware_before_starting_anything(mock, exit_code):
    # MAKE=true stands in for the recursive make that starts the API and Vite.
    # make prints that command when it runs it, so the output shows whether
    # anything started before the guard refused.
    pythonpath = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH")])
    )
    proc = _make(
        "dev",
        "MAKE=true",
        f"PY={sys.executable}",
        env={"SO101_MOCK": mock, "PYTHONPATH": pythonpath},
    )
    assert proc.returncode == exit_code, proc.stdout + proc.stderr
    started = any(line.startswith("true ") for line in proc.stdout.splitlines())
    if exit_code:
        assert "error: dev auto-reloads" in proc.stderr
        assert not started, "make dev started the API and Vite before the guard refused"
    else:
        assert started, proc.stdout


def test_every_make_target_the_docs_mention_exists():
    # Three audit bugs (B2, B11, B14) were docs that drifted from the Makefile.
    targets = set(re.findall(r"^([A-Za-z0-9_.-]+):", MAKEFILE.read_text(encoding="utf-8"), re.M))
    docs = [REPO_ROOT / "README.md", REPO_ROOT / "CONTRIBUTING.md"]
    docs += sorted((REPO_ROOT / "docs").glob("*.md"))
    mentioned = {}
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        fenced = re.findall(r"```.*?```", text, re.S)
        inline = re.findall(r"`[^`\n]+`", re.sub(r"```.*?```", "", text, flags=re.S))
        for code in fenced + inline:
            pattern = r"(?<![\w/-])make[ \t]+(?:-C[ \t]+\S+[ \t]+)?((?:[a-z][\w-]*[ \t]*)+)"
            for match in re.finditer(pattern, code):
                for target in match.group(1).split():
                    mentioned.setdefault(target, doc.name)
    assert "suite" in mentioned and "ci" in mentioned  # the scan finds commands
    missing = {target: doc for target, doc in mentioned.items() if target not in targets}
    assert not missing, f"docs mention make targets the Makefile lacks: {missing}"


def test_install_targets_only_ever_use_the_venv_interpreter(tmp_path):
    venv = tmp_path / "no-such-venv"
    for target in ("install", "install-robot"):
        out = _dry_run(target, f"VENV={venv}")
        pip_lines = [line for line in out.splitlines() if "pip install" in line]
        assert pip_lines, out
        assert all(line.startswith(f"{venv}/bin/python -m pip ") for line in pip_lines), out


def test_coverage_target_measures_then_enforces_the_floors():
    out = _dry_run("coverage")
    assert "pytest --cov" in out
    assert "--cov-report=json:coverage.json" in out
    assert out.index("pytest --cov") < out.index("scripts/check_coverage.py coverage.json")


@pytest.mark.parametrize("target", ["test", "coverage"])
def test_full_runs_enforce_the_skip_budget(target: str):
    out = _dry_run(target)
    pytest_line = next(line for line in out.splitlines() if " -m pytest" in line)
    assert pytest_line.endswith("--skip-budget tests/skip-budget.toml"), pytest_line
    assert (REPO_ROOT / "tests" / "skip-budget.toml").is_file()


def test_ci_wheel_builds_the_ui_first_then_checks_what_it_built():
    out = _dry_run("ci-wheel", "FRONTEND_INSTALL=0")
    steps = [
        "npm run build",
        "BUILT_UI_PATHS_REQUIRED=1",
        "test_c1_every_file_in_the_built_ui_passes_the_path_check",
        "rm -rf dist",
        "SO101_REQUIRE_UI=1",
        "check_dist.py dist --ui required",
        "wheel_smoke.py dist --ui required",
    ]
    positions = [out.find(step) for step in steps]
    assert -1 not in positions and positions == sorted(positions), out


def test_verify_is_a_fast_check_not_the_full_release_gate():
    help_text = _help_sections()["Quality"]["verify"]
    assert "Fast local" in help_text
    assert "release gate is make ci" in help_text
    assert "Full software release gate" not in MAKEFILE.read_text(encoding="utf-8")


def test_wheel_builds_the_ui_first_so_it_never_packs_a_stale_one():
    # frontend-deps is a file target: an up-to-date install keeps the dry run
    # from printing npm ci, as in a checkout where it already ran.
    out = _dry_run("wheel", "-o", "frontend/node_modules/.package-lock.json")
    steps = [
        "npm run build",
        "rm -rf dist",
        "SO101_REQUIRE_UI=1",
        "-m build --outdir dist .",
        "check_dist.py dist --ui required",
    ]
    positions = [out.find(step) for step in steps]
    assert -1 not in positions and positions == sorted(positions), out


def test_a_parent_makes_overrides_do_not_reach_the_make_under_test(monkeypatch):
    # make exports the variables set on its command line, so under
    # `make ci SMOKE_PORT=8814 PY=...` the tests inherit them.
    assert {"PY", "SMOKE_PORT", "FRONTEND_INSTALL", "HOST", "PORT"} <= OVERRIDABLE
    monkeypatch.setenv("SMOKE_PORT", "1234")
    monkeypatch.setenv("PY", "/inherited/python")
    out = _dry_run("ci-smoke")
    assert "scripts/run_smoke.py --port 0" in out
    assert "/inherited/python" not in out


def test_ci_smoke_uses_a_throwaway_server_on_the_given_port():
    assert "scripts/run_smoke.py --port 0" in _dry_run("ci-smoke")
    assert "scripts/run_smoke.py --port 8814" in _dry_run("ci-smoke", "SMOKE_PORT=8814")


def test_thor_host_gate_uses_the_root_selected_interpreter_without_hardware():
    out = _dry_run("ci-thor-worker", f"PY={sys.executable}")
    assert "-C thor-worker test" in out
    assert "import sys; print(sys.executable)" in out
    assert "docker" not in out.lower() and "cuda" not in out.lower()


def test_every_script_the_makefile_runs_exists():
    scripts = set(re.findall(r"scripts/[\w./-]+\.py", MAKEFILE.read_text(encoding="utf-8")))
    assert {"scripts/run_smoke.py", "scripts/check_dist.py", "scripts/wheel_smoke.py"} <= scripts
    assert sorted(path for path in scripts if not (REPO_ROOT / path).is_file()) == []


def _frontend_copy(tmp_path: Path, scripts: dict[str, str]) -> Path:
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    (tmp_path / "frontend").mkdir()
    package = {"name": "x", "private": True, "scripts": scripts}
    (tmp_path / "frontend" / "package.json").write_text(json.dumps(package), encoding="utf-8")
    return tmp_path


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_ci_frontend_fails_loudly_when_a_script_is_missing(tmp_path):
    copy = _frontend_copy(tmp_path, {"dev": "vite", "build": "vite build"})
    proc = _make("ci-frontend", cwd=copy)
    assert proc.returncode != 0
    assert "has no lint, typecheck, test script" in proc.stderr
    assert "npm ci" not in proc.stdout  # it fails before installing anything


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_frontend_script_check_passes_when_every_ci_script_exists(tmp_path):
    copy = _frontend_copy(
        tmp_path,
        {"lint": "eslint .", "typecheck": "tsc --noEmit", "test": "vitest run", "build": "x"},
    )
    proc = _make("frontend-scripts", cwd=copy)
    assert proc.returncode == 0, proc.stderr


def test_ci_frontend_runs_typecheck_before_tests_and_build():
    out = _dry_run("ci-frontend", "FRONTEND_INSTALL=0")
    steps = ["npm run lint", "npm run typecheck", "npm run test", "npm run build"]
    positions = [out.find(step) for step in steps]
    assert -1 not in positions and positions == sorted(positions), out


def test_frontend_installs_refuse_a_symlinked_node_modules(tmp_path):
    copy = _frontend_copy(tmp_path, {"lint": "a", "test": "b", "build": "c"})
    target = tmp_path / "elsewhere"
    target.mkdir()
    (copy / "frontend" / "node_modules").symlink_to(target)
    proc = _make("frontend-ci-install", cwd=copy)
    assert proc.returncode != 0
    assert "frontend/node_modules is a symlink; npm ci would delete" in proc.stderr
    assert "npm ci" not in proc.stdout  # refused before npm ran
    assert target.exists()


@pytest.mark.parametrize(
    ("version", "ok"),
    [((3, 11, 9), False), ((3, 12, 3), True), ((3, 13, 1), True), ((3, 14, 0), False)],
)
def test_the_venv_python_check_matches_requires_python(version, ok):
    """PY_VERSION_OK accepts exactly pyproject.toml's requires-python (>=3.12,<3.14)."""
    import tomllib

    from packaging.specifiers import SpecifierSet

    (code,) = re.findall(r"^PY_VERSION_OK = (.*)$", MAKEFILE.read_text(encoding="utf-8"), re.M)
    fake_sys = type("FakeSys", (), {"version_info": version, "exit": staticmethod(sys.exit)})
    with pytest.raises(SystemExit) as exited:
        exec(code.replace("import sys; ", ""), {"sys": fake_sys})
    assert (not exited.value.code) is ok
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires = SpecifierSet(pyproject["project"]["requires-python"])
    assert requires.contains(".".join(map(str, version))) is ok
