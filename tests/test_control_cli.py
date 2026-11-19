"""CLI contracts owned by the control-plane stream.

- ``so101 estop`` must never look like it worked when it did not (C11 item 6,
  devex-B7 CLI half, docs/audit/v0.1/devex.md).
- ``so101 doctor`` exits 1 when a check fails and prints every detail and
  remediation verbatim, not as Rich markup (devex-B13).
"""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from so101 import cli, config
from so101.cli import app
from so101.services.schemas import DiagnosticCheck, DiagnosticsResponse

runner = CliRunner()
NOT_DELIVERED = "E-STOP NOT DELIVERED to {url} — use the power switch"


@pytest.fixture
def posts(monkeypatch) -> list[str]:
    """Record every URL the CLI posts to and answer 200."""

    calls: list[str] = []

    def fake_post(url, *args, **kwargs):
        calls.append(url)
        return httpx.Response(
            200,
            json={"ok": True, "detail": "E-STOP engaged; all motion stopped"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def _settings(monkeypatch, host: str, port: int) -> None:
    monkeypatch.setenv("SO101_HOST", host)
    monkeypatch.setenv("SO101_PORT", str(port))
    config.get_settings.cache_clear()


def test_estop_posts_to_the_given_url(posts: list[str]):
    result = runner.invoke(app, ["estop", "--url", "http://robot.local:9000/"])

    assert result.exit_code == 0, result.output
    assert posts == ["http://robot.local:9000/api/robot/estop"]
    assert "E-STOP engaged" in result.stdout


def test_estop_503_says_latched_with_torque_off_unconfirmed(monkeypatch):
    def fake_post(url, *args, **kwargs):
        return httpx.Response(
            503,
            json={"detail": "E-STOP is latched, but torque-off is not confirmed yet"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = runner.invoke(app, ["estop", "--url", "http://robot.local:9000"])

    assert result.exit_code == 1
    assert "E-STOP LATCHED at http://robot.local:9000" in result.stderr
    assert "cut physical power" in result.stderr
    assert "NOT DELIVERED" not in result.stderr


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "http://127.0.0.1:8123"),
        ("::", "http://127.0.0.1:8123"),
        ("robot.local", "http://robot.local:8123"),
        ("::1", "http://[::1]:8123"),
    ],
)
def test_estop_default_url_comes_from_settings(posts: list[str], monkeypatch, host, expected):
    _settings(monkeypatch, host, 8123)

    result = runner.invoke(app, ["estop"])

    assert result.exit_code == 0, result.output
    assert posts == [f"{expected}/api/robot/estop"]


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
    ],
    ids=["connect-error", "timeout"],
)
def test_estop_transport_failure_is_loud(monkeypatch, failure):
    _settings(monkeypatch, "0.0.0.0", 8123)

    def failing_post(url, *args, **kwargs):
        raise failure

    monkeypatch.setattr(httpx, "post", failing_post)

    result = runner.invoke(app, ["estop"])

    assert result.exit_code != 0
    assert NOT_DELIVERED.format(url="http://127.0.0.1:8123") in result.stderr
    assert "NOT DELIVERED" not in result.stdout


def test_estop_non_2xx_is_loud_and_shows_the_server_detail(monkeypatch):
    # Any non-2xx but 503 (latched, torque-off unconfirmed; tested above) means
    # the stop may not have happened.
    def refused(url, *args, **kwargs):
        return httpx.Response(
            500,
            json={"detail": "internal error before the latch"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", refused)

    result = runner.invoke(app, ["estop", "--url", "http://robot.local:9000"])

    assert result.exit_code != 0
    assert NOT_DELIVERED.format(url="http://robot.local:9000") in result.stderr
    assert "500" in result.stderr
    assert "internal error before the latch" in result.stderr


# --- devex-B13: so101 doctor ------------------------------------------------------


def _diagnose(monkeypatch, *checks: tuple[str, str, str]) -> None:
    """Make ``so101 doctor`` see these (status, detail, remediation) checks."""

    from datetime import datetime, timezone

    from rich.console import Console

    from so101.services.routers import operations

    statuses = [status for status, _, _ in checks]
    overall = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    response = DiagnosticsResponse(
        generated_at=datetime.now(timezone.utc),
        overall=overall,
        checks=[
            DiagnosticCheck(
                id=f"check.{index}",
                label=f"Check {index}",
                status=status,
                detail=detail,
                remediation=remediation,
            )
            for index, (status, detail, remediation) in enumerate(checks)
        ],
        ports=[],
        events=[],
    )
    monkeypatch.setattr(operations, "diagnostics", lambda: response)
    # Wide enough that no cell wraps; the width is not what is under test.
    monkeypatch.setattr(cli, "console", Console(width=240))


def test_devex_b13_doctor_exits_1_when_a_check_fails(monkeypatch):
    _diagnose(
        monkeypatch,
        ("pass", "Explicit deterministic mock backend.", ""),
        ("fail", "Real mode is selected but LeRobot is not installed.", "Install it."),
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.stdout


@pytest.mark.parametrize("status", ["pass", "warn"])
def test_devex_b13_doctor_exits_0_without_a_failure(monkeypatch, status):
    _diagnose(monkeypatch, (status, "Storage is low.", "Free space."))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output


def test_devex_b13_doctor_prints_the_remediation_verbatim(monkeypatch):
    _diagnose(
        monkeypatch,
        (
            "fail",
            "Real mode is selected but LeRobot is not installed.",
            "Install `.[robot]` or explicitly set SO101_MOCK=1.",
        ),
    )

    result = runner.invoke(app, ["doctor"])

    assert "Install `.[robot]` or explicitly set SO101_MOCK=1." in result.stdout


def test_devex_b13_doctor_survives_markup_like_details(monkeypatch):
    # A port path or exception text that looks like a closing tag used to
    # raise rich.errors.MarkupError instead of printing the report.
    _diagnose(monkeypatch, ("warn", "no reply on [/dev/ttyACM0] [bold]", "Check [red]it[/red]."))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "no reply on [/dev/ttyACM0] [bold]" in result.stdout
    assert "Check [red]it[/red]." in result.stdout
