"""Every SO101_REST_POSE example in .env.example, config.py and the docs is a rest pose.

An operator who pastes a documented example gives Collapse that target, so each
one must pass ``check_rest_pose`` in the suite's own model (review finding M10).
An example is a value in the SO101_REST_POSE format (six comma-separated
numbers) in backticks or quotes, or a non-empty ``SO101_REST_POSE=`` line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from so101.hardware.rest_pose import parse_rest_pose_setting

REPO_ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"-?\d+(?:\.\d+)?"
QUOTED = re.compile(rf"[`\"']({NUMBER}(?:,{NUMBER}){{5}})[`\"']")
ASSIGNED = re.compile(r"^\s*(?:export\s+)?SO101_REST_POSE=(\S+)", re.M)
# Where an example can be: the settings template, the settings class and the docs.
SOURCES = (".env.example", "src/so101/config.py", "README.md", "CONTRIBUTING.md", "docs/*.md")
# These document SO101_REST_POSE with an example; an empty match would make the
# physics check below vacuous.
MUST_HAVE_AN_EXAMPLE = {".env.example", "src/so101/config.py", "docs/bring-up.md"}


def _examples() -> list[tuple[str, str]]:
    found = []
    for pattern in SOURCES:
        for path in sorted(REPO_ROOT.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            where = path.relative_to(REPO_ROOT).as_posix()
            values = QUOTED.findall(text) + [value.strip("\"'") for value in ASSIGNED.findall(text)]
            found += [(where, value) for value in values]
    return found


def test_the_documented_examples_are_found_and_parse():
    examples = _examples()
    assert MUST_HAVE_AN_EXAMPLE <= {where for where, _ in examples}, examples
    for where, value in examples:
        assert parse_rest_pose_setting(value) is not None, (where, value)


def test_every_documented_example_passes_check_rest_pose():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("the rest-pose example check needs the sim extra (mujoco)")
    from so101.simulation.rest_pose import check_rest_pose

    poses: dict[str, list[str]] = {}
    for where, value in _examples():
        poses.setdefault(value, []).append(where)
    failures = []
    for value, places in poses.items():
        pose = parse_rest_pose_setting(value)
        assert pose is not None
        report = check_rest_pose(pose)
        if not report.ok:
            failed = sorted(name for name, ok in report.checks.items() if not ok)
            failures.append(
                f"{value} ({', '.join(places)}): fails {failed}; "
                f"contacts at the target {list(report.commanded_contacts)}"
            )
    assert failures == []
