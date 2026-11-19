"""Docker images and compose file (audit B18).

Static checks only: Docker is not run by the tests (and was not available when
these files changed). hadolint and the compose-spec schema were run by hand;
see CONTRIBUTING.md.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER = REPO_ROOT / "docker"
COMPOSE = REPO_ROOT / "docker-compose.yml"
IMAGES = ("api", "suite", "ui")
PYTHON_IMAGES = ("api", "suite")


def _final_stage(name: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) for the final build stage, continuations joined."""
    text = (DOCKER / f"Dockerfile.{name}").read_text(encoding="utf-8")
    text = re.sub(r"\\\n\s*", " ", text)
    instructions = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keyword, _, arguments = line.partition(" ")
            instructions.append((keyword.upper(), arguments.strip()))
    last_from = max(i for i, (keyword, _) in enumerate(instructions) if keyword == "FROM")
    return instructions[last_from:]


def _arguments(name: str, keyword: str) -> list[str]:
    return [arguments for kw, arguments in _final_stage(name) if kw == keyword]


@pytest.mark.parametrize("name", IMAGES)
def test_images_run_as_an_unprivileged_numeric_user(name):
    users = _arguments(name, "USER")
    assert users, f"Dockerfile.{name} runs as root"
    # The last USER is the one the container runs as. Numeric, so that a runtime
    # can check it is not root without reading the image's /etc/passwd.
    assert users[-1] == "1000:1000"


def test_ui_image_installs_its_dependencies_as_the_user_it_runs_as():
    # The dev server writes Vite's cache into node_modules/.vite, so npm ci must
    # not run as root. COPY ignores USER, hence --chown.
    stage = _final_stage("ui")
    keywords = [keyword for keyword, _ in stage]
    assert keywords.index("USER") < stage.index(("RUN", "npm ci"))
    copies = _arguments("ui", "COPY")
    assert copies and all(copy.startswith("--chown=node:node ") for copy in copies)


@pytest.mark.parametrize("name", PYTHON_IMAGES)
def test_python_images_report_api_health_with_urllib(name):
    (healthcheck,) = _arguments(name, "HEALTHCHECK")
    assert "urllib.request.urlopen('http://127.0.0.1:8000/api/health'" in healthcheck
    (command,) = _arguments(name, "CMD")
    assert '"--port", "8000"' in command  # the port the health check polls
    assert "--reload" not in command


@pytest.mark.parametrize("name", PYTHON_IMAGES)
def test_python_images_copy_the_build_hook_pyproject_declares(name):
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "custom" in pyproject["tool"]["hatch"]["build"]["hooks"]
    copies = " ".join(_arguments(name, "COPY"))
    assert "hatch_build.py" in copies, "pip install fails without the build hook"


def test_compose_ui_waits_for_a_healthy_api():
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert compose["services"]["ui"]["depends_on"] == {"api": {"condition": "service_healthy"}}


def test_compose_passes_no_devices_and_publishes_on_loopback_only():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "/dev/tty" not in text  # real hardware runs on the host
    for name, service in yaml.safe_load(text)["services"].items():
        assert "devices" not in service and not service.get("privileged"), name
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), (name, port)


@pytest.mark.skipif(shutil.which("make") is None, reason="GNU make is not installed")
def test_make_docker_creates_the_bind_mounted_dirs_first():
    # Without a parent make's flags and jobserver state (e.g. under `make ci`).
    env = {k: v for k, v in os.environ.items() if k not in {"MAKEFLAGS", "MFLAGS", "MAKELEVEL"}}
    proc = subprocess.run(
        ["make", "-n", "docker"], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    mkdir = next(line for line in lines if line.startswith("mkdir -p "))
    assert lines.index(mkdir) < next(i for i, line in enumerate(lines) if "docker compose" in line)
    created = set(mkdir.split()[2:])
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    for service in compose["services"].values():
        for volume in service.get("volumes", []):
            host = volume.split(":")[0]
            assert host.removeprefix("./") in created, volume
