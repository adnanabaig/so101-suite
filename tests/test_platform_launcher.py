"""so101.launcher without starting a real server or browser."""

from __future__ import annotations

import io
import json
import os
import sys
import types
from pathlib import Path

import pytest

from so101 import launcher


@pytest.fixture
def clean_env(monkeypatch):
    """Let main() change these; monkeypatch puts the originals back afterwards."""

    for key in ("MUJOCO_GL", "SO101_HOST", "SO101_PORT", "SO101_HOME", "DISPLAY"):
        # setenv first so undo removes whatever main() sets, even if it was unset.
        monkeypatch.setenv(key, "unset")
        monkeypatch.delenv(key)
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.chdir(Path.cwd())  # restore the working directory main() changes


# --- pieces ------------------------------------------------------------------


def test_resolve_home_prefers_the_argument_then_so101_home(tmp_path):
    home = tmp_path / "user"
    assert launcher.resolve_home(str(tmp_path / "x"), "linux", {}, home) == tmp_path / "x"
    env = {"SO101_HOME": str(tmp_path / "y")}
    assert launcher.resolve_home(None, "linux", env, home) == tmp_path / "y"
    assert launcher.resolve_home(None, "linux", {}, home) == home / ".local/share/so101"


def test_resolve_home_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert launcher.resolve_home("rel", "linux", {}, tmp_path) == tmp_path / "rel"


def test_prepare_environment_drops_a_mujoco_gl_that_breaks_import():
    env = {"MUJOCO_GL": "egl"}
    notes = launcher.prepare_environment(env, "windows", lambda name: None)
    assert "MUJOCO_GL" not in env
    assert env["PYTHONUTF8"] == "1"
    assert any("Ignoring MUJOCO_GL=egl" in note for note in notes)


def test_prepare_environment_picks_egl_on_headless_linux():
    env: dict[str, str] = {}
    notes = launcher.prepare_environment(env, "linux", lambda name: "libEGL.so.1")
    assert env["MUJOCO_GL"] == "egl"
    assert any("MUJOCO_GL=egl" in note for note in notes)


def test_prepare_environment_leaves_good_settings_alone():
    env = {"MUJOCO_GL": "osmesa", "PYTHONUTF8": "0", "DISPLAY": ":0"}
    assert (
        launcher.prepare_environment(env, "linux", lambda name: "libEGL.so.1", scripts_dir="") == []
    )
    assert env == {"MUJOCO_GL": "osmesa", "PYTHONUTF8": "0", "DISPLAY": ":0"}
    mac: dict[str, str] = {}
    assert launcher.prepare_environment(mac, "macos", lambda name: None, scripts_dir="") == []
    assert "MUJOCO_GL" not in mac


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://127.0.0.1:8000/"),
        ("0.0.0.0", "http://127.0.0.1:8000/"),
        ("::", "http://127.0.0.1:8000/"),
        ("", "http://127.0.0.1:8000/"),
        ("::1", "http://[::1]:8000/"),
        ("192.168.1.20", "http://192.168.1.20:8000/"),
    ],
)
def test_suite_url(host, expected):
    assert launcher.suite_url(host, 8000) == expected


def test_can_open_browser_needs_a_display_only_on_linux():
    assert launcher.can_open_browser("windows", {}) is True
    assert launcher.can_open_browser("macos", {}) is True
    assert launcher.can_open_browser("linux", {}) is False
    assert launcher.can_open_browser("linux", {"WAYLAND_DISPLAY": "wayland-0"}) is True


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_health_ok_accepts_only_an_ok_suite():
    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return FakeResponse(json.dumps({"ok": True, "version": "0.2.0"}).encode())

    assert launcher.health_ok("http://127.0.0.1:8000/", urlopen=urlopen) is True
    assert calls == [("http://127.0.0.1:8000/api/health", 1.0)]

    def other(body: bytes, status: int = 200):
        return lambda url, timeout: FakeResponse(body, status)

    assert launcher.health_ok("u/", urlopen=other(b'{"ok": false}')) is False
    assert launcher.health_ok("u/", urlopen=other(b"<html>")) is False
    assert launcher.health_ok("u/", urlopen=other(b'["ok"]')) is False
    assert launcher.health_ok("u/", urlopen=other(b'{"ok": true}', 503)) is False

    def refused(url, timeout):
        raise ConnectionRefusedError(61, "Connection refused")

    assert launcher.health_ok("u/", urlopen=refused) is False


def test_open_when_ready_opens_once_the_suite_answers():
    answers = iter([False, False, True])
    opened = []
    assert launcher.open_when_ready(
        "http://127.0.0.1:8000/",
        poll=lambda url: next(answers),
        opener=opened.append,
        sleep=lambda s: None,
    )
    assert opened == ["http://127.0.0.1:8000/"]


def test_open_when_ready_gives_up_at_the_deadline():
    now = iter([0.0, 1.0, 2.0, 99.0])
    opened = []
    assert not launcher.open_when_ready(
        "u",
        timeout=5.0,
        poll=lambda url: False,
        opener=opened.append,
        sleep=lambda s: None,
        clock=lambda: next(now),
    )
    assert opened == []


def test_pause_only_when_asked(monkeypatch):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "")
    launcher._pause(False)
    assert asked == []
    launcher._pause(True)
    assert asked == ["Press Enter to close this window."]

    def closed(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed)
    launcher._pause(True)  # no console: returns quietly


# --- main() ---------------------------------------------------------------------


@pytest.fixture
def no_server(monkeypatch):
    """Record serve/browser calls instead of starting uvicorn or a browser."""

    calls: dict[str, list] = {"serve": [], "open": [], "thread": []}
    monkeypatch.setattr(launcher, "_serve", lambda host, port: calls["serve"].append((host, port)))
    monkeypatch.setattr(
        launcher, "webbrowser", types.SimpleNamespace(open=lambda url: calls["open"].append(url))
    )
    monkeypatch.setattr(launcher, "health_ok", lambda url: False)

    class Thread:
        def __init__(self, target, args, daemon):
            calls["thread"].append((target, args, daemon))

        def start(self):
            pass

    monkeypatch.setattr(launcher, "threading", types.SimpleNamespace(Thread=Thread))
    return calls


def test_main_runs_the_server_from_the_data_folder(tmp_path, clean_env, no_server, capsys):
    home = tmp_path / "home"
    assert launcher.main(["--home", str(home), "--port", "8818"]) == 0

    assert Path.cwd() == home.resolve()
    assert os.environ["SO101_PORT"] == "8818"
    assert no_server["serve"] == [("127.0.0.1", 8818)]
    [(target, args, daemon)] = no_server["thread"]
    assert target is launcher.open_when_ready and args == ("http://127.0.0.1:8818/",) and daemon
    out = capsys.readouterr().out
    assert str(home) in out and "rest pose" in out


def test_main_creates_the_settings_file_once_and_makes_the_folder_so101_home(
    tmp_path, clean_env, no_server, capsys
):
    from so101 import platform_info

    home = tmp_path / "suite"
    assert launcher.main(["--home", str(home), "--no-browser"]) == 0
    assert os.environ["SO101_HOME"] == str(home.resolve())  # Settings reads <home>/.env
    env = home / ".env"
    assert platform_info.parse_dotenv(env.read_text(encoding="utf-8")) == (
        platform_info.storage_settings(home.resolve())
    )
    assert f"Created {env.resolve()}" in capsys.readouterr().out

    env.write_text("SO101_DATA_DIR=/mnt/big/so101\n", encoding="utf-8")  # the user's choice
    assert launcher.main(["--home", str(home), "--no-browser"]) == 0
    assert env.read_text(encoding="utf-8") == "SO101_DATA_DIR=/mnt/big/so101\n"  # untouched
    assert "Created" not in capsys.readouterr().out


def test_main_passes_host_through_settings(tmp_path, clean_env, no_server):
    assert launcher.main(["--home", str(tmp_path), "--host", "0.0.0.0", "--no-browser"]) == 0
    assert os.environ["SO101_HOST"] == "0.0.0.0"
    assert no_server["serve"] == [("0.0.0.0", 8000)]
    assert no_server["thread"] == []


def test_main_only_opens_the_browser_when_already_running(
    tmp_path, clean_env, no_server, monkeypatch, capsys
):
    monkeypatch.setattr(launcher, "health_ok", lambda url: True)
    assert launcher.main(["--home", str(tmp_path)]) == 0
    assert no_server["serve"] == []
    assert no_server["open"] == ["http://127.0.0.1:8000/"]
    assert "already running" in capsys.readouterr().out


def test_main_explains_a_port_in_use_and_pauses(tmp_path, clean_env, monkeypatch, capsys):
    def busy(host, port):
        raise SystemExit(launcher.STARTUP_FAILURE)

    paused = []
    monkeypatch.setattr(launcher, "_serve", busy)
    monkeypatch.setattr(launcher, "health_ok", lambda url: False)
    monkeypatch.setattr(launcher, "_pause", paused.append)
    code = launcher.main(["--home", str(tmp_path), "--no-browser", "--pause-on-error"])
    assert code == launcher.STARTUP_FAILURE
    assert paused == [True]
    assert "SO101_PORT" in capsys.readouterr().err


def test_main_treats_ctrl_c_and_clean_exit_as_success(tmp_path, clean_env, monkeypatch):
    monkeypatch.setattr(launcher, "health_ok", lambda url: False)

    def interrupted(host, port):
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher, "_serve", interrupted)
    assert launcher.main(["--home", str(tmp_path), "--no-browser"]) == 0

    def clean(host, port):
        raise SystemExit(0)

    monkeypatch.setattr(launcher, "_serve", clean)
    assert launcher.main(["--home", str(tmp_path), "--no-browser"]) == 0


def test_main_reports_a_crash_and_reraises(tmp_path, clean_env, monkeypatch, capsys):
    def crash(host, port):
        raise ImportError("No module named 'uvicorn'")

    paused = []
    monkeypatch.setattr(launcher, "_serve", crash)
    monkeypatch.setattr(launcher, "health_ok", lambda url: False)
    monkeypatch.setattr(launcher, "_pause", paused.append)
    with pytest.raises(ImportError):
        launcher.main(["--home", str(tmp_path), "--no-browser", "--pause-on-error"])
    assert paused == [True]
    assert "the server failed" in capsys.readouterr().err


def test_main_fails_cleanly_when_the_data_folder_is_unusable(tmp_path, clean_env, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    assert launcher.main(["--home", str(blocker / "sub")]) == 1
    assert "can't use the data folder" in capsys.readouterr().err


def test_prepare_environment_picks_egl_with_the_nvidia_driver_on_a_desktop():
    env = {"DISPLAY": ":0"}
    egl = lambda name: "libEGL.so.1" if name == "EGL" else None  # noqa: E731
    notes = launcher.prepare_environment(env, "linux", egl, exists=lambda path: True)
    assert env["MUJOCO_GL"] == "egl"
    assert any("NVIDIA" in note for note in notes)
    plain = {"DISPLAY": ":0"}
    assert (
        launcher.prepare_environment(plain, "linux", egl, exists=lambda path: False, scripts_dir="")
        == []
    )
    assert "MUJOCO_GL" not in plain


def test_prepare_environment_puts_the_venv_scripts_first_on_path():
    env = {"PATH": os.pathsep.join(["/usr/bin", "/bin"])}
    notes = launcher.prepare_environment(
        env, "linux", lambda name: None, scripts_dir="/app/venv/bin"
    )
    assert env["PATH"].split(os.pathsep) == ["/app/venv/bin", "/usr/bin", "/bin"]
    assert any("/app/venv/bin first on PATH" in note for note in notes)
    again = launcher.prepare_environment(
        env, "linux", lambda name: None, scripts_dir="/app/venv/bin"
    )
    assert env["PATH"].split(os.pathsep) == ["/app/venv/bin", "/usr/bin", "/bin"]
    assert not any("PATH" in note for note in again)

    empty: dict[str, str] = {}
    launcher.prepare_environment(
        empty, "windows", lambda name: None, scripts_dir="C:\\app\\Scripts"
    )
    assert empty["PATH"] == "C:\\app\\Scripts"
    default: dict[str, str] = {"PATH": "/usr/bin"}
    launcher.prepare_environment(default, "linux", lambda name: None)
    assert default["PATH"].split(os.pathsep)[0] == os.path.dirname(sys.executable)
