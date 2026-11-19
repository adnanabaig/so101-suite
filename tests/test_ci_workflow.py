""".github/workflows/ci.yml: its structure, pinned actions, and parity with `make ci`.

actionlint (with shellcheck) checks the workflow's syntax and shell scripts;
see CONTRIBUTING.md. These tests pin down the choices it cannot see.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"
JOBS = ("backend", "frontend", "smoke", "wheel")
STOPS = ("estop", "halt", "collapse")


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _steps(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return workflow["jobs"][job]["steps"]


def _runs(workflow: dict[str, Any], job: str) -> list[str]:
    return [step["run"] for step in _steps(workflow, job) if "run" in step]


def _make_targets(workflow: dict[str, Any], job: str) -> list[str]:
    return [
        target
        for run in _runs(workflow, job)
        for target in re.findall(r"(?m)^\s*make\s+([\w-]+)", run)
    ]


def _uses(workflow: dict[str, Any], job: str, action: str) -> list[dict[str, Any]]:
    return [
        step
        for step in _steps(workflow, job)
        if str(step.get("uses", "")).startswith(f"actions/{action}@")
    ]


def _makefile_prerequisites(target: str) -> list[str]:
    match = re.search(rf"(?m)^{re.escape(target)}:([^#\n]*)", MAKEFILE.read_text("utf-8"))
    assert match, f"no {target} target in the Makefile"
    return match.group(1).split()


def _makefile_recipe(target: str) -> list[str]:
    lines = MAKEFILE.read_text("utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{target}:"))
    recipe = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        recipe.append(line.strip())
    return recipe


def test_triggers_permissions_and_concurrency(workflow):
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads a bare `on` as True
    assert {"push", "pull_request"} <= set(triggers)
    assert "pull_request_target" not in triggers  # would run fork code with secrets
    assert workflow["permissions"] == {"contents": "read"}
    assert "concurrency" in workflow


def test_the_four_jobs_run_on_ubuntu_with_a_timeout(workflow):
    assert tuple(workflow["jobs"]) == JOBS
    for name, job in workflow["jobs"].items():
        assert job["runs-on"] == "ubuntu-latest", name
        assert 0 < job["timeout-minutes"] <= 60, name


def test_every_action_is_pinned_to_a_commit_with_its_release_named():
    lines = [line for line in WORKFLOW.read_text("utf-8").splitlines() if "uses:" in line]
    assert lines
    for line in lines:
        assert re.search(r"uses: actions/[\w-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+$", line), line


def test_checkouts_do_not_keep_the_token(workflow):
    for job in JOBS:
        (checkout,) = _uses(workflow, job, "checkout")
        assert checkout["with"]["persist-credentials"] is False, job


def test_python_matches_requires_python_and_caches_pip(workflow):
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    assert project["requires-python"] == ">=3.12,<3.14"
    for job in ("backend", "smoke", "wheel"):
        assert workflow["jobs"][job]["strategy"] == {
            "fail-fast": False,
            "matrix": {"python-version": ["3.12", "3.13"]},
        }
        (setup,) = _uses(workflow, job, "setup-python")
        assert setup["with"] == {
            "python-version": "${{ matrix.python-version }}",
            "cache": "pip",
            "cache-dependency-path": "pyproject.toml",
        }, job


def test_node_is_the_lts_line_and_caches_npm(workflow):
    for job in ("frontend", "wheel"):
        (setup,) = _uses(workflow, job, "setup-node")
        assert setup["with"] == {
            "node-version": "24",
            "cache": "npm",
            "cache-dependency-path": "frontend/package-lock.json",
        }, job


def test_backend_installs_dev_and_sim_with_best_effort_egl(workflow):
    job = workflow["jobs"]["backend"]
    assert job["env"]["MUJOCO_GL"] == "egl"
    (apt,) = [step for step in job["steps"] if "apt-get install" in step.get("run", "")]
    assert apt["continue-on-error"] is True  # missing GL must not fail the job
    assert "libegl1" in apt["run"]
    assert any('pip install -e ".[dev,sim]"' in run for run in _runs(workflow, "backend"))


def test_make_ci_runs_the_same_steps_as_the_workflow(workflow):
    ci_jobs = [line.split()[1] for line in _makefile_recipe("ci") if line.startswith("$(MAKE)")]
    assert ci_jobs == ["ci-backend", "ci-frontend", "ci-smoke", "ci-wheel"]
    # The backend job runs ci-backend's prerequisites as separate steps, for
    # separate logs; the other jobs run their ci-* target.
    assert _make_targets(workflow, "backend") == _makefile_prerequisites("ci-backend")
    assert _make_targets(workflow, "frontend") == ["ci-frontend"]
    assert _make_targets(workflow, "smoke") == ["ci-smoke"]
    assert _make_targets(workflow, "wheel") == ["ci-wheel"]


def test_the_wheel_job_has_python_node_and_build(workflow):
    assert _uses(workflow, "wheel", "setup-python")
    assert _uses(workflow, "wheel", "setup-node")
    assert any('pip install -e ".[dev]"' in run for run in _runs(workflow, "wheel"))


def test_no_job_runs_a_stop(workflow):
    for job in JOBS:
        for run in _runs(workflow, job):
            for stop in STOPS:
                assert not re.search(rf"\b{stop}\b", run), (job, stop)
