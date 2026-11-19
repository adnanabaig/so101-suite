"""The skip budget (tests/skip_budget.py, tests/skip-budget.toml; finding 5).

A full run (make test, make coverage, the CI backend job) must fail when a test
skips for an unlisted reason, when one reason skips more tests than its cap,
or when the total passes max_skipped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from skip_budget import BudgetError, judge, load_budget

TESTS = Path(__file__).resolve().parent
BUDGET = TESTS / "skip-budget.toml"


def _budget(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "budget.toml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_the_pinned_budget_loads_and_its_caps_add_up_to_max_skipped() -> None:
    budget = load_budget(BUDGET)
    assert budget.allowed, "every allowed skip reason is listed"
    assert budget.max_skipped == sum(allowance.max for allowance in budget.allowed)
    assert all(allowance.why.strip() for allowance in budget.allowed)


def test_the_pinned_budget_names_the_known_skips() -> None:
    budget = load_budget(BUDGET)
    known = [
        "could not import 'draccus': No module named 'draccus'",
        "LeRobot is not installed here; robot hosts run this check",
        "set SO101_TEST_LEROBOT_SDIST to the path of lerobot-0.6.1.tar.gz to check the source",
        "no JPEG encoder installed; the dependency-free fallback is PNG",
        "could not import 'pyarrow': No module named 'pyarrow'",
        "frontend/dist is not built",
        "pyrealsense2 is not installed (the realsense extra)",
        "the rest-pose example check needs the sim extra (mujoco)",
    ]
    verdict = judge(budget, [(f"t{index}", reason) for index, reason in enumerate(known)])
    assert verdict.unlisted == []


def test_within_the_budget_is_ok(tmp_path: Path) -> None:
    budget = load_budget(
        _budget(
            tmp_path,
            """
            max_skipped = 2
            [[allowed]]
            reason = "needs LeRobot"
            max = 2
            why = "robot hosts run it"
            """,
        )
    )
    verdict = judge(budget, [("a", "needs LeRobot (robot extra)"), ("b", "needs LeRobot")])
    assert verdict.ok
    assert verdict.lines(budget.path)[0].endswith("2 skipped, at most 2 allowed: ok")


def test_an_unlisted_reason_a_cap_and_the_total_each_fail(tmp_path: Path) -> None:
    budget = load_budget(
        _budget(
            tmp_path,
            """
            max_skipped = 2
            [[allowed]]
            reason = "needs LeRobot"
            max = 1
            why = "robot hosts run it"
            [[allowed]]
            reason = "needs git"
            max = 2
            why = "git is everywhere we run"
            """,
        )
    )
    unlisted = judge(budget, [("a", "no OpenGL here")])
    assert not unlisted.ok and unlisted.unlisted == [("a", "no OpenGL here")]
    over_cap = judge(budget, [("a", "needs LeRobot"), ("b", "needs LeRobot")])
    assert not over_cap.ok and [a.reason for a, _ in over_cap.over] == ["needs LeRobot"]
    over_total = judge(budget, [("a", "needs LeRobot"), ("b", "needs git"), ("c", "needs git")])
    assert not over_total.ok and over_total.total == 3
    text = "\n".join(over_total.lines(budget.path))
    assert "FAILED" in text and "1 skip(s) over max_skipped" in text


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("max_skipped = -1\n", "max_skipped"),
        ("max_skipped = 1\nextra = 2\n", "unknown keys"),
        ('max_skipped = 1\n[[allowed]]\nreason = "x"\nmax = 1\n', "exactly reason, max and why"),
        ('max_skipped = 1\n[[allowed]]\nreason = "("\nmax = 1\nwhy = "w"\n', "regular expr"),
        ('max_skipped = 1\n[[allowed]]\nreason = "x"\nmax = 0\nwhy = "w"\n', "max must"),
        ('max_skipped = 1\n[[allowed]]\nreason = "x"\nmax = 1\nwhy = " "\n', "why must"),
    ],
)
def test_a_bad_budget_file_is_refused(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(BudgetError, match=message):
        load_budget(_budget(tmp_path, text))


def _run_pytest(tmp_path: Path, budget: str) -> subprocess.CompletedProcess[str]:
    """A separate pytest session on a tiny suite, with the budget plugin loaded."""

    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_tiny.py").write_text(
        textwrap.dedent(
            """
            import pytest

            def test_runs():
                pass

            @pytest.mark.skipif(True, reason="needs LeRobot")
            def test_lerobot():
                pass

            def test_gl():
                pytest.skip("no OpenGL here")

            @pytest.mark.xfail(reason="known bug", strict=True)
            def test_known_bug():
                assert False
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_missing.py").write_text(
        'import pytest\npytest.importorskip("no_such_module_for_the_budget")\n', encoding="utf-8"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TESTS), *sys.path])}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "skip_budget",
            "-p",
            "no:cacheprovider",
            "--skip-budget",
            str(_budget(tmp_path, budget)),
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


ALL_THREE = """
max_skipped = 3
[[allowed]]
reason = "needs LeRobot"
max = 1
why = "robot hosts run it"
[[allowed]]
reason = "no OpenGL here"
max = 1
why = "headless runners"
[[allowed]]
reason = "could not import 'no_such_module_for_the_budget'"
max = 1
why = "a collection-time skip counts too"
"""


def test_a_session_within_the_budget_passes(tmp_path: Path) -> None:
    result = _run_pytest(tmp_path, ALL_THREE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 skipped, at most 3 allowed: ok" in result.stdout  # the xfail is not a skip


def test_a_session_with_an_unlisted_skip_fails(tmp_path: Path) -> None:
    budget = ALL_THREE.replace("max_skipped = 3", "max_skipped = 2").replace(
        '[[allowed]]\nreason = "no OpenGL here"\nmax = 1\nwhy = "headless runners"\n', ""
    )
    result = _run_pytest(tmp_path, budget)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "unlisted skip: test_tiny.py::test_gl: no OpenGL here" in result.stdout
    assert "FAILED" in result.stdout


def test_a_session_over_max_skipped_fails(tmp_path: Path) -> None:
    result = _run_pytest(tmp_path, ALL_THREE.replace("max_skipped = 3", "max_skipped = 2"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "3 skipped, at most 2 allowed: FAILED" in result.stdout


def test_without_the_option_nothing_is_checked(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_one.py").write_text(
        "import pytest\n\ndef test_skip():\n    pytest.skip('anything')\n", encoding="utf-8"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TESTS), *sys.path])}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "skip_budget", "-p", "no:cacheprovider", "."],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skip budget" not in result.stdout
