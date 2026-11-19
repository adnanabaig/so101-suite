"""The skip budget: a full test run fails when tests skip beyond a pinned list.

``make test`` and ``make coverage`` (the CI backend job) run pytest with
``--skip-budget tests/skip-budget.toml``. The file pins the most tests that
may skip (``max_skipped``) and lists every allowed skip reason, each with its
own cap. The run fails (exit 1) when:

- a test skips for a reason no entry matches (a new skip must be listed on
  purpose),
- more tests skip for one entry than its ``max``, or
- more tests skip in total than ``max_skipped``.

Without ``--skip-budget`` nothing is checked, so running a few tests by hand
is unaffected. Pure Python and ``tomllib``: the same on every OS.

The hooks are imported by ``tests/conftest.py``; ``-p skip_budget`` (with
``tests/`` on ``sys.path``) loads them in any other pytest run.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

OPTION = "--skip-budget"
_SKIPPED_PREFIX = "Skipped: "


@dataclass(frozen=True)
class Allowance:
    reason: str
    pattern: re.Pattern[str]
    max: int
    why: str


@dataclass(frozen=True)
class Budget:
    path: Path
    max_skipped: int
    allowed: tuple[Allowance, ...]


@dataclass
class Verdict:
    total: int
    max_skipped: int
    counts: dict[str, int] = field(default_factory=dict)
    unlisted: list[tuple[str, str]] = field(default_factory=list)
    over: list[tuple[Allowance, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unlisted and not self.over and self.total <= self.max_skipped

    def lines(self, path: Path) -> list[str]:
        head = f"skip budget ({path}): {self.total} skipped, at most {self.max_skipped} allowed"
        if self.ok:
            return [f"{head}: ok"]
        lines = [f"{head}: FAILED"]
        if self.total > self.max_skipped:
            lines.append(f"  {self.total - self.max_skipped} skip(s) over max_skipped")
        for allowance, count in self.over:
            lines.append(
                f"  {count} skipped for {allowance.reason!r}, at most {allowance.max} allowed"
            )
        for nodeid, reason in self.unlisted:
            lines.append(f"  unlisted skip: {nodeid}: {reason}")
        lines.append(
            "  A skip hides coverage. Fix what makes the test skip, or list the reason in the "
            "budget file on purpose (see CONTRIBUTING.md, Skip budget)."
        )
        return lines


class BudgetError(ValueError):
    """The budget file cannot be used."""


def load_budget(path: Path) -> Budget:
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BudgetError(f"cannot read the skip budget {path}: {exc}") from exc
    unknown = set(data) - {"max_skipped", "allowed"}
    if unknown:
        raise BudgetError(f"{path}: unknown keys {sorted(unknown)}")
    max_skipped = data.get("max_skipped")
    if not isinstance(max_skipped, int) or isinstance(max_skipped, bool) or max_skipped < 0:
        raise BudgetError(f"{path}: max_skipped must be a whole number >= 0")
    allowed = []
    for index, entry in enumerate(data.get("allowed", [])):
        where = f"{path}: allowed[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"reason", "max", "why"}:
            raise BudgetError(f"{where} needs exactly reason, max and why")
        reason, cap, why = entry["reason"], entry["max"], entry["why"]
        if not isinstance(reason, str) or not reason.strip():
            raise BudgetError(f"{where}: reason must be a non-empty regular expression")
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise BudgetError(f"{where}: max must be a whole number >= 1")
        if not isinstance(why, str) or not why.strip():
            raise BudgetError(f"{where}: why must say why the skip is acceptable")
        try:
            pattern = re.compile(reason)
        except re.error as exc:
            raise BudgetError(f"{where}: bad regular expression {reason!r}: {exc}") from exc
        allowed.append(Allowance(reason, pattern, cap, why))
    return Budget(Path(path), max_skipped, tuple(allowed))


def skip_reason(report: Any) -> str:
    """The skip message of a skipped test or collection report."""
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        message = str(longrepr[2])
    else:
        message = str(longrepr or "")
    return message[len(_SKIPPED_PREFIX) :] if message.startswith(_SKIPPED_PREFIX) else message


def judge(budget: Budget, skips: Iterable[tuple[str, str]]) -> Verdict:
    """Match each ``(nodeid, reason)`` skip against the budget's entries."""
    skips = list(skips)
    verdict = Verdict(total=len(skips), max_skipped=budget.max_skipped)
    for allowance in budget.allowed:
        verdict.counts[allowance.reason] = 0
    for nodeid, reason in skips:
        match = next((a for a in budget.allowed if a.pattern.search(reason)), None)
        if match is None:
            verdict.unlisted.append((nodeid, reason))
        else:
            verdict.counts[match.reason] += 1
    verdict.over = [
        (allowance, verdict.counts[allowance.reason])
        for allowance in budget.allowed
        if verdict.counts[allowance.reason] > allowance.max
    ]
    return verdict


class SkipBudgetPlugin:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.skips: list[tuple[str, str]] = []
        self.verdict: Verdict | None = None

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped and not hasattr(report, "wasxfail"):
            self.skips.append((report.nodeid, skip_reason(report)))

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.skips.append((report.nodeid, skip_reason(report)))

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self.verdict = judge(self.budget, self.skips)
        if not self.verdict.ok and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self.verdict is None:
            return
        for line in self.verdict.lines(self.budget.path):
            terminalreporter.write_line(line, red=not self.verdict.ok, bold=not self.verdict.ok)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        OPTION,
        metavar="TOML",
        default=None,
        help="fail the run when tests skip beyond this budget file (make test, make coverage)",
    )


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption(OPTION)
    if not path:
        return
    try:
        budget = load_budget(Path(path))
    except BudgetError as exc:
        raise pytest.UsageError(str(exc)) from exc
    config.pluginmanager.register(SkipBudgetPlugin(budget), "so101-skip-budget")
