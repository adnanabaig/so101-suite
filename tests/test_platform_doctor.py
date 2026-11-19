"""``so101 doctor`` shows the platform section after the suite's checks, and ``--json``."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from rich.console import Console
from typer.testing import CliRunner

from so101 import cli, platform_info
from so101.cli import app
from so101.platform_info import PlatformCheck
from so101.services.routers import operations
from so101.services.schemas import DiagnosticCheck, DiagnosticsResponse

runner = CliRunner()


@pytest.fixture
def machine(monkeypatch):
    """Doctor with one suite check and the given platform checks; returns a setter."""

    def set_up(suite_status: str, *platform: tuple[str, str]) -> None:
        response = DiagnosticsResponse(
            generated_at=datetime.now(timezone.utc),
            overall=suite_status,
            checks=[
                DiagnosticCheck(
                    id="backend",
                    label="Backend",
                    status=suite_status,
                    detail="mock backend",
                    remediation="",
                )
            ],
            ports=[],
            events=[],
        )
        monkeypatch.setattr(operations, "diagnostics", lambda: response)
        report = platform_info.collect(platform_info.Probes(torch=lambda: None))
        monkeypatch.setattr(platform_info, "collect", lambda: report)
        checks = [
            PlatformCheck(f"platform.test.{i}", f"Platform {i}", status, detail, "Fix it.")
            for i, (status, detail) in enumerate(platform)
        ]
        monkeypatch.setattr(platform_info, "checks", lambda report: checks)
        monkeypatch.setattr(cli, "console", Console(width=240))

    return set_up


def test_doctor_prints_the_platform_section_after_the_suite_checks(machine):
    machine("pass", ("pass", "Python 3.12.3 on linux x86_64"), ("warn", "no udev rules [x]"))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert result.stdout.index("mock backend") < result.stdout.index("Python 3.12.3")
    assert "no udev rules [x]" in result.stdout  # data, not Rich markup
    assert "SO-101 diagnostics · WARN" in result.stdout


@pytest.mark.parametrize(
    ("suite", "platform", "overall", "code"),
    [
        ("pass", "pass", "pass", 0),
        ("pass", "warn", "warn", 0),
        ("warn", "pass", "warn", 0),
        ("pass", "fail", "fail", 1),
        ("fail", "pass", "fail", 1),
    ],
)
def test_doctor_json_has_every_check_and_the_worst_status(machine, suite, platform, overall, code):
    machine(suite, (platform, "Disk space"))

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == code, result.output
    payload = json.loads(result.stdout)
    assert payload["overall"] == overall
    assert [check["id"] for check in payload["checks"]] == ["backend", "platform.test.0"]
    assert payload["checks"][1] == {
        "id": "platform.test.0",
        "label": "Platform 0",
        "status": platform,
        "detail": "Disk space",
        "remediation": "Fix it.",
    }
    assert payload["diagnostics"]["overall"] == suite
    assert payload["platform"]["os"] in {"linux", "macos", "windows"}
    assert "realsense" in payload["platform"]


def test_doctor_exits_1_on_a_platform_failure(machine):
    machine("pass", ("fail", "Python 3.11 is too old"))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Python 3.11 is too old" in result.stdout
