"""scripts/run_smoke.py: the throwaway mock API behind `make ci-smoke`.

These tests start no server: Popen, the health check and the smoke suite are
replaced with fakes. The real run is `make ci-smoke` and the CI smoke job.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("run_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rs = _load_script()


class FakeServer:
    """Stands in for the uvicorn Popen; records how it was started and stopped."""

    instances: list[FakeServer] = []

    def __init__(self, command, *, cwd, env, stdout, stderr, ignores_sigterm=False):
        self.command, self.cwd, self.env = command, Path(cwd), env
        stdout.write(b"INFO:     fake uvicorn log line\n")
        self.returncode: int | None = None
        self.ignores_sigterm = ignores_sigterm
        self.terminated = self.killed = False
        FakeServer.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignores_sigterm:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("uvicorn", timeout)
        return self.returncode


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Fake server, health and smoke suite; returns a dict the test can adjust."""
    FakeServer.instances = []
    home = tmp_path / "home"
    state = {"health": {"ok": True, "mock": True}, "smoke_code": 0, "smoke_commands": []}

    def mkdtemp(prefix):
        home.mkdir()
        return str(home)

    def fake_run(command, timeout):
        state["smoke_commands"].append(command)
        return subprocess.CompletedProcess(command, state["smoke_code"])

    monkeypatch.setattr(rs.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(rs, "port_in_use", lambda host, port: False)
    monkeypatch.setattr(rs.subprocess, "Popen", FakeServer)
    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    monkeypatch.setattr(rs, "wait_for_health", lambda base, server, timeout: state["health"])
    state["home"] = home
    return state


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1:8814"),
        ("0.0.0.0", "http://127.0.0.1:8814"),
        ("::", "http://127.0.0.1:8814"),
        ("::1", "http://[::1]:8814"),
        ("localhost", "http://localhost:8814"),
    ],
)
def test_client_url(host, url):
    assert rs.client_url(host, 8814) == url


def test_the_server_gets_mock_mode_and_temp_data_and_no_developer_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("SO101_FOLLOWER_PORT", "/dev/ttyACM0")
    monkeypatch.setenv("SO101_DATA_DIR", "/home/me/so101-data")
    monkeypatch.setenv("SO101_HOME", "/home/me/.local/share/so101")  # its .env is real
    monkeypatch.setenv("UNRELATED", "kept")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", f"src{rs.os.pathsep}{rs.os.pathsep}/abs/lib")
    env = rs.server_env(tmp_path / "home")
    assert env["SO101_MOCK"] == "1"
    assert "SO101_FOLLOWER_PORT" not in env
    assert env["SO101_DATA_DIR"] == str(tmp_path / "home" / "data")
    assert env["SO101_DATASETS_DIR"] == str(tmp_path / "home" / "datasets")
    assert env["SO101_OUTPUTS_DIR"] == str(tmp_path / "home" / "outputs")
    # The installed suite's <data folder>/.env must not reach the smoke server.
    assert env["SO101_HOME"] == str(tmp_path / "home")
    assert env["UNRELATED"] == "kept"
    # The server runs from the temp dir: relative entries are made absolute and
    # the empty entry (the current directory) is dropped.
    assert env["PYTHONPATH"].split(rs.os.pathsep) == [str((tmp_path / "src").resolve()), "/abs/lib"]


def test_an_empty_pythonpath_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")
    assert "PYTHONPATH" not in rs.server_env(tmp_path)


def test_runs_the_smoke_suite_against_a_mock_server_and_stops_it(harness, capsys):
    assert rs.run("127.0.0.1", 8814, timeout=5) == 0
    (server,) = FakeServer.instances
    assert server.command[1:3] == ["-m", "uvicorn"]
    assert server.command[3:] == ["--factory", rs.APP, "--host", "127.0.0.1", "--port", "8814"]
    assert "--reload" not in server.command
    assert server.cwd == harness["home"]  # so no .env from the checkout is read
    assert server.env["SO101_MOCK"] == "1"
    assert harness["smoke_commands"] == [
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "smoke_suite.py"),
            "--base",
            "http://127.0.0.1:8814",
        ]
    ]
    assert server.terminated and not server.killed
    assert not harness["home"].exists()
    assert (
        "run_smoke: passed; the server on http://127.0.0.1:8814 stopped" in capsys.readouterr().out
    )


@pytest.mark.parametrize("mock", [False, None])
def test_refuses_a_server_that_is_not_in_mock_mode(harness, capsys, mock):
    harness["health"] = {"ok": True, "mock": mock}
    assert rs.run("127.0.0.1", 8814, timeout=5) == 1
    err = capsys.readouterr().err
    assert f"/api/health reports mock={mock!r}, not true" in err
    assert "fake uvicorn log line" in err  # the server log tail
    assert harness["smoke_commands"] == []
    assert FakeServer.instances[0].terminated
    assert not harness["home"].exists()


def test_a_failing_smoke_suite_fails_with_its_exit_code_and_the_server_log(harness, capsys):
    harness["smoke_code"] = 2
    assert rs.run("127.0.0.1", 8814, timeout=5) == 2
    err = capsys.readouterr().err
    assert "smoke_suite.py exited 2; the server stopped" in err
    assert "fake uvicorn log line" in err
    assert FakeServer.instances[0].terminated
    assert not harness["home"].exists()


def test_a_startup_failure_stops_the_server_and_shows_the_log(harness, monkeypatch, capsys):
    def no_answer(base, server, timeout):
        raise rs.SmokeServerError(f"no answer from {base}/api/health within {timeout:g}s")

    monkeypatch.setattr(rs, "wait_for_health", no_answer)
    assert rs.run("127.0.0.1", 8814, timeout=5) == 1
    err = capsys.readouterr().err
    assert "no answer from http://127.0.0.1:8814/api/health within 5s" in err
    assert "fake uvicorn log line" in err
    assert FakeServer.instances[0].terminated


def test_a_server_that_ignores_sigterm_is_killed(harness, monkeypatch):
    monkeypatch.setattr(
        rs.subprocess, "Popen", lambda *a, **kw: FakeServer(*a, **kw, ignores_sigterm=True)
    )
    assert rs.run("127.0.0.1", 8814, timeout=5) == 0
    (server,) = FakeServer.instances
    assert server.terminated and server.killed


def test_a_busy_port_fails_before_starting_anything(harness, monkeypatch, capsys):
    monkeypatch.setattr(rs, "port_in_use", lambda host, port: True)
    assert rs.run("127.0.0.1", 8814, timeout=5) == 1
    assert "127.0.0.1:8814 is already in use" in capsys.readouterr().err
    assert FakeServer.instances == []
    assert not harness["home"].exists()


def test_port_zero_picks_a_free_port(harness, monkeypatch):
    monkeypatch.setattr(rs, "free_port", lambda host: 8814)
    assert rs.run("127.0.0.1", 0, timeout=5) == 0
    assert FakeServer.instances[0].command[-2:] == ["--port", "8814"]


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


class _Alive:
    returncode = None

    def poll(self):
        return None


def test_wait_for_health_returns_the_health_payload(monkeypatch):
    replies = iter([OSError("connection refused"), b'{"ok": true, "mock": true}'])

    def urlopen(url, timeout):
        assert url == "http://127.0.0.1:8814/api/health"
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)

    monkeypatch.setattr(rs.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(rs.time, "sleep", lambda seconds: None)
    health = rs.wait_for_health("http://127.0.0.1:8814", _Alive(), timeout=5)
    assert health == {"ok": True, "mock": True}


def test_wait_for_health_fails_when_the_server_exits():
    class Exited:
        returncode = 3

        def poll(self):
            return 3

    with pytest.raises(rs.SmokeServerError, match="exited with code 3 on startup"):
        rs.wait_for_health("http://127.0.0.1:8814", Exited(), timeout=5)


def test_wait_for_health_gives_up_after_the_timeout(monkeypatch):
    clock = iter(range(100))
    monkeypatch.setattr(rs.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rs.time, "sleep", lambda seconds: None)

    def refused(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(rs.urllib.request, "urlopen", refused)
    with pytest.raises(rs.SmokeServerError, match="no answer from .* within 5s"):
        rs.wait_for_health("http://127.0.0.1:8814", _Alive(), timeout=5)


def test_wait_for_health_rejects_a_non_object(monkeypatch):
    monkeypatch.setattr(rs.urllib.request, "urlopen", lambda url, timeout: _Response(b"[]"))
    with pytest.raises(rs.SmokeServerError, match="returned \\[\\], not an object"):
        rs.wait_for_health("http://127.0.0.1:8814", _Alive(), timeout=5)


@pytest.mark.parametrize("port", ["-1", "65536"])
def test_main_rejects_an_out_of_range_port(port, capsys):
    with pytest.raises(SystemExit) as exc:
        rs.main(["--port", port])
    assert exc.value.code == 2
    assert "--port must be 0-65535" in capsys.readouterr().err


def test_the_server_is_built_by_the_app_factory():
    from so101.services import app

    assert rs.APP == app.APP_FACTORY  # with --factory: importing the module builds nothing
