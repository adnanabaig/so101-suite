#!/usr/bin/env python3
"""Enforce per-file and per-function coverage floors from a coverage.py JSON report.

Usage (what ``make coverage`` and CI run)::

    pytest --cov --cov-report=json:coverage.json
    python scripts/check_coverage.py coverage.json

Floors live in ``coverage-floors.toml`` at the repo root. The metric is
coverage.py's ``percent_covered``, i.e. the "Cover" column of
``coverage report``. With branch coverage on, it combines statements and
branches. A floor whose target is missing from the report is a failure, not
a skip: a renamed module must carry its floor along, not silently drop it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOORS = REPO_ROOT / "coverage-floors.toml"
PACKAGE = "so101"
ALLOWED_SECTIONS = {"settings", "files", "functions"}
RATCHET_HINT_POINTS = 5.0


class FloorsError(ValueError):
    """The floors file or the coverage report cannot be used."""


@dataclass(frozen=True)
class Result:
    target: str
    floor: float
    actual: float | None  # None: target missing from the report

    @property
    def ok(self) -> bool:
        return self.actual is not None and round(self.actual, 2) >= self.floor


def package_key(path: str) -> str:
    """Map any spelling of a module path to ``so101/...``.

    The report may say ``src/so101/x.py``, an absolute path, or a
    site-packages path; floors are written relative to the repo root.
    """
    parts = Path(path.replace("\\", "/")).parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == PACKAGE:
            return "/".join(parts[index:])
    return "/".join(parts)


def _floor_value(raw: Any, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise FloorsError(f"{where}: floor must be a number, got {raw!r}")
    value = float(raw)
    if not 0.0 <= value <= 100.0:
        raise FloorsError(f"{where}: floor must be within 0..100, got {value}")
    return value


def load_floors(path: Path) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FloorsError(f"floors file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise FloorsError(f"{path}: invalid TOML: {exc}") from exc
    unknown = set(data) - ALLOWED_SECTIONS
    if unknown:
        raise FloorsError(f"{path}: unknown section(s) {sorted(unknown)}")
    settings = dict(data.get("settings", {}))
    files = {
        name: _floor_value(raw, f"[files] {name}") for name, raw in data.get("files", {}).items()
    }
    functions: dict[str, float] = {}
    for name, raw in data.get("functions", {}).items():
        if "::" not in name:
            raise FloorsError(f"[functions] {name}: expected '<file>::<qualified name>'")
        functions[name] = _floor_value(raw, f"[functions] {name}")
    if not files and not functions:
        raise FloorsError(f"{path}: no floors defined")
    return settings, files, functions


def evaluate(
    report: dict[str, Any],
    settings: dict[str, Any],
    file_floors: dict[str, float],
    function_floors: dict[str, float],
) -> list[Result]:
    meta = report.get("meta", {})
    if "files" not in report:
        raise FloorsError("not a coverage.py JSON report (no 'files' key)")
    want_branch = settings.get("branch")
    if want_branch is not None and bool(meta.get("branch_coverage")) != bool(want_branch):
        raise FloorsError(
            f"floors assume branch coverage = {bool(want_branch)}, but the report has "
            f"branch_coverage = {bool(meta.get('branch_coverage'))}; "
            "run pytest with the [tool.coverage] config (make coverage)"
        )
    measured = {package_key(name): data for name, data in report["files"].items()}

    results: list[Result] = []
    for target, floor in file_floors.items():
        data = measured.get(package_key(target))
        actual = None if data is None else float(data["summary"]["percent_covered"])
        results.append(Result(target, floor, actual))

    for target, floor in function_floors.items():
        file_name, qualname = target.split("::", 1)
        data = measured.get(package_key(file_name))
        actual = None
        if data is not None:
            if "functions" not in data:
                raise FloorsError(
                    "function floors need coverage.py >= 7.5 (JSON 'functions' regions)"
                )
            region = data["functions"].get(qualname)
            if region is not None:
                actual = float(region["summary"]["percent_covered"])
        results.append(Result(target, floor, actual))
    return results


def render(results: list[Result]) -> str:
    width = max(len(result.target) for result in results)
    lines = [f"{'target':<{width}}  {'floor':>6}  {'actual':>6}  status"]
    for result in results:
        if result.actual is None:
            status, actual = "MISSING from report", "-"
        else:
            actual = f"{result.actual:6.2f}"
            if not result.ok:
                status = "FAIL"
            elif result.actual - result.floor >= RATCHET_HINT_POINTS:
                status = "ok (consider ratcheting up)"
            else:
                status = "ok"
        lines.append(f"{result.target:<{width}}  {result.floor:6.2f}  {actual:>6}  {status}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", nargs="?", default="coverage.json", type=Path)
    parser.add_argument("--floors", default=DEFAULT_FLOORS, type=Path)
    args = parser.parse_args(argv)

    try:
        settings, file_floors, function_floors = load_floors(args.floors)
        try:
            report = json.loads(args.report.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FloorsError(
                f"coverage report not found: {args.report} (run `make coverage`)"
            ) from exc
        results = evaluate(report, settings, file_floors, function_floors)
    except FloorsError as exc:
        print(f"check_coverage: error: {exc}", file=sys.stderr)
        return 2

    print(f"Coverage floors ({args.floors.name}):")
    print(render(results))
    failures = [result for result in results if not result.ok]
    if not failures:
        print(f"All {len(results)} coverage floors hold.")
        return 0
    for result in failures:
        if result.actual is None:
            detail = f"{result.target} is not in the coverage report; was it renamed or moved?"
        else:
            detail = f"{result.target} is {result.actual:.2f}%, below its {result.floor:.2f}% floor"
        print(f"FAIL: {detail}", file=sys.stderr)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::error title=Coverage floor::{detail}")
    print(
        f"{len(failures)} of {len(results)} coverage floors failed. Add tests; "
        "do not lower a floor to make a change pass.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
