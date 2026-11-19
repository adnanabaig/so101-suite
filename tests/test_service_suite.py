"""Availability and stopping decisions preserve the motor safety boundary."""

import importlib.util
import json
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "suite", Path(__file__).resolve().parents[1] / "scripts/service_suite.py"
)
assert spec and spec.loader
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def test_online_services_do_not_hide_motor_fault_and_stale_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(
        suite.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="active\n")
    )

    def fetch(url):
        if url.endswith("control/state"):
            return {"armed": False, "safety_state": "fault"}
        if url.endswith("robot/status"):
            return {"backend": "real", "torque_state": "off"}
        return {"live": True}

    monkeypatch.setattr(suite, "fetch", fetch)
    path = tmp_path / "data/real/mcap/status.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "active": True,
                "updated_at": time.time(),
                "bytes": 1024,
                "telemetry_received_at": time.time(),
            }
        )
    )
    report = suite.status()
    assert report["all_local_online"]
    assert report["motion_state"]["safety_state"] == "fault"
    assert not report["upload"]["enabled"]
    path.write_text(json.dumps({"active": True, "updated_at": 0, "bytes": 1024}))
    assert not suite.status()["all_local_online"]


def test_down_refuses_energized_arm(monkeypatch):
    monkeypatch.setattr("sys.argv", ["services", "down"])
    monkeypatch.setattr(suite, "fetch", lambda url: {"torque_state": "on", "armed": True})
    monkeypatch.setattr(
        suite.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not stop services")
    )
    with pytest.raises(RuntimeError, match="torque OFF"):
        suite.main()


def restart_state(monkeypatch):
    replies = {
        "robot/status": {
            "torque_state": "off",
            "torque_unconfirmed": False,
            "mode": "idle",
            "stop_state": "none",
            "operation_owner": None,
        },
        "control/state": {"armed": False, "lease": None},
        "mcap/status": {"active": None, "recorder": {"active": False, "recording": False}},
    }
    monkeypatch.setattr(
        suite,
        "fetch",
        lambda url: next(value for suffix, value in replies.items() if url.endswith(suffix)),
    )
    return replies


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("robot/status", "torque_state", "on"),
        ("robot/status", "torque_state", "unconfirmed"),
        ("robot/status", "mode", "calibrating"),
        ("robot/status", "mode", "teleop"),
        ("robot/status", "stop_state", "halt"),
        ("robot/status", "operation_owner", "operator"),
        ("control/state", "armed", True),
        ("control/state", "lease", {"owner": "operator"}),
        ("mcap/status", "active", {"id": "episode"}),
    ],
)
def test_restart_refuses_active_operations(monkeypatch, section, key, value):
    replies = restart_state(monkeypatch)
    replies[section][key] = value
    monkeypatch.setattr("sys.argv", ["services", "restart"])
    monkeypatch.setattr(
        suite.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not restart")
    )
    with pytest.raises(RuntimeError, match="Refresh refused"):
        suite.main()


def test_restart_refuses_active_recorder(monkeypatch):
    replies = restart_state(monkeypatch)
    replies["mcap/status"]["recorder"]["active"] = True
    with pytest.raises(RuntimeError, match="Refresh refused"):
        suite.require_idle_restart()


def test_restart_allows_observed_idle_torque_off(monkeypatch):
    restart_state(monkeypatch)
    suite.require_idle_restart()


def test_restart_fails_closed_when_control_unavailable(monkeypatch):
    restart_state(monkeypatch)

    def unavailable(url):
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr(suite, "fetch", unavailable)
    with pytest.raises(RuntimeError, match="unresponsive"):
        suite.require_idle_restart()


@pytest.mark.parametrize(
    "motor_state,recorder_state,allowed",
    [
        ("inactive", "inactive", True),
        ("failed", "inactive", True),
        ("active", "inactive", False),
        ("inactive", "active", False),
        ("", "inactive", False),
    ],
)
def test_restart_when_api_is_not_running(monkeypatch, motor_state, recorder_state, allowed):
    def unavailable(url):
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(suite, "fetch", unavailable)
    monkeypatch.setattr(
        suite.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout=motor_state if command[-1] == "so101-motor.service" else recorder_state
        ),
    )
    if allowed:
        suite.require_idle_restart()
    else:
        with pytest.raises(RuntimeError):
            suite.require_idle_restart()


def test_restart_stops_then_reinstalls_and_starts(monkeypatch, tmp_path):
    restart_state(monkeypatch)
    monkeypatch.setattr("sys.argv", ["services", "restart", "--json", "--open"])
    monkeypatch.setattr(suite.Path, "home", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(suite.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(suite.shutil, "copyfile", lambda *args: None)
    monkeypatch.setattr(suite, "status", lambda: {"all_local_online": True})
    opened = []
    monkeypatch.setattr(suite.webbrowser, "open", opened.append)
    suite.main()
    assert [command[2] for command in calls] == ["stop", "daemon-reload", "start"]
    assert all(unit in calls[0] and unit in calls[-1] for unit in suite.UNITS)
    assert opened[0].startswith("http://127.0.0.1:8000/teleop?refresh=")
