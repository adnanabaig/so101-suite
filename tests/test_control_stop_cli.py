"""``so101 halt`` and ``so101 collapse`` mirror ``so101 estop`` (design doc §5).

Anything but a confirmed stop is loud: a non-zero exit and a stderr message
that names the E-stop and the power switch. ``so101 calibrate`` refuses the
sim backend, as the calibration route does.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from so101 import cli as cli_module
from so101 import config
from so101.cli import app as cli_app

runner = CliRunner()


@pytest.fixture
def posts(monkeypatch) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    def fake_post(url, *args, **kwargs):
        calls.append((url, kwargs.get("json")))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


@pytest.mark.parametrize(("command", "path"), [("halt", "halt"), ("collapse", "collapse")])
def test_cli_stop_posts_to_the_given_url(posts, command: str, path: str):
    result = runner.invoke(cli_app, [command, "--url", "http://robot.local:9000/"])

    assert result.exit_code == 0, result.output
    assert posts[0][0] == f"http://robot.local:9000/api/robot/{path}"
    assert posts[0][1]["reason"].startswith(f"so101 {command}")
    assert command.upper() in result.stdout


@pytest.mark.parametrize("command", ["halt", "collapse"])
def test_cli_stop_default_url_comes_from_settings(posts, monkeypatch, command: str):
    monkeypatch.setenv("SO101_HOST", "0.0.0.0")
    monkeypatch.setenv("SO101_PORT", "8123")
    config.get_settings.cache_clear()

    assert runner.invoke(cli_app, [command]).exit_code == 0
    assert posts[0][0] == f"http://127.0.0.1:8123/api/robot/{command}"


@pytest.mark.parametrize("command", ["halt", "collapse"])
@pytest.mark.parametrize(
    "answer",
    [
        httpx.ConnectError("connection refused"),
        httpx.Response(503, json={"detail": "the halt could not hold; torque cut"}),
    ],
    ids=["connect-error", "503"],
)
def test_cli_stop_failures_are_loud(monkeypatch, command: str, answer):
    def post(url, *args, **kwargs):
        if isinstance(answer, Exception):
            raise answer
        answer.request = httpx.Request("POST", url)
        return answer

    monkeypatch.setattr(httpx, "post", post)

    result = runner.invoke(cli_app, [command, "--url", "http://robot.local:9000"])

    assert result.exit_code == 1
    assert f"{command.upper()} NOT CONFIRMED by http://robot.local:9000" in result.stderr
    assert "estop" in result.stderr
    assert "NOT CONFIRMED" not in result.stdout


def test_cli_calibrate_refuses_the_sim_backend(monkeypatch):
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", "sim")
    config.get_settings.cache_clear()

    result = runner.invoke(cli_app, ["calibrate", "follower"])

    assert result.exit_code == 1
    assert "sim backend" in result.stderr


def test_cli_calibrate_real_requires_external_confirmation_before_bus_access(monkeypatch):
    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("SO101_BACKEND", "real")
    config.get_settings.cache_clear()
    gate_calls: list[str] = []

    def gate(purpose: str):
        gate_calls.append(purpose)
        raise AssertionError("the standalone gate must not run before confirmation")

    monkeypatch.setattr(cli_module, "_standalone_gate", gate)
    try:
        result = runner.invoke(cli_app, ["calibrate", "follower"], input="n\n")
    finally:
        config.get_settings.cache_clear()

    assert result.exit_code == 1
    assert "hands-on joint sweep" in result.stderr
    assert "padded mechanical catch" in result.stderr
    assert "physical cutoff" in result.stderr
    assert "pinch points" in result.stderr
    assert "midpoint captured; now sweep" in result.stderr
    assert gate_calls == []
