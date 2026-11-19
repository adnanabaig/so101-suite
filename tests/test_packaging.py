"""Packaging metadata contracts (pyproject.toml)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _extra(name: str) -> dict[str, Requirement]:
    requirements = [
        Requirement(spec) for spec in PYPROJECT["project"]["optional-dependencies"][name]
    ]
    return {requirement.name.lower(): requirement for requirement in requirements}


def test_every_python_dependency_has_a_compatible_upper_bound():
    groups = {"default": PYPROJECT["project"]["dependencies"]}
    groups.update(PYPROJECT["project"]["optional-dependencies"])
    for group, specs in groups.items():
        for spec in specs:
            requirement = Requirement(spec)
            operators = {item.operator for item in requirement.specifier}
            assert operators & {"<", "<=", "==", "===", "~="}, (group, spec)


def test_robot_extra_installs_a_single_opencv_distribution():
    """LeRobot 0.6.1 requires opencv-python-headless>=4.9,<4.14. Listing
    opencv-python too puts two wheels in one cv2/ package (audit B6)."""
    robot = _extra("robot")
    assert "opencv-python" not in robot
    headless = robot["opencv-python-headless"]
    assert headless.specifier.contains("4.13.0.92")
    assert not headless.specifier.contains("4.14.0")
    assert not headless.specifier.contains("5.0.0.93")


def test_the_package_version_is_the_project_version():
    """The handoff wheel's name comes from pyproject.toml; so101.__version__ must agree."""
    import so101

    assert so101.__version__ == PYPROJECT["project"]["version"] == "0.2.2rc5"

    frontend = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((REPO_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))
    assert frontend["version"] == lock["version"] == lock["packages"][""]["version"]
    assert frontend["version"].replace("-rc.", "rc") == so101.__version__


def test_requires_python_is_312_to_313():
    """LeRobot 0.6.1 needs 3.12+; pyrealsense2 2.56 has no cp314 wheels."""
    from packaging.specifiers import SpecifierSet

    from so101 import platform_info

    requires = SpecifierSet(PYPROJECT["project"]["requires-python"])
    assert [v for v in ("3.11.9", "3.12.3", "3.13.1", "3.14.0") if requires.contains(v)] == [
        "3.12.3",
        "3.13.1",
    ]
    assert (platform_info.MIN_PYTHON, platform_info.MAX_PYTHON) == ((3, 12), (3, 13))
