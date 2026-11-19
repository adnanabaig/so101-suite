"""One-command startup setup stays separate from calibration and motion."""

from __future__ import annotations

import importlib.util
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("start_local", REPO / "scripts/start_local.py")
assert spec is not None and spec.loader is not None
start = importlib.util.module_from_spec(spec)
spec.loader.exec_module(start)


@pytest.fixture
def root(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "config/rigs").mkdir(parents=True)
    shutil.copyfile(
        REPO / "config/rigs/so101.env.example", checkout / "config/rigs/so101.env.example"
    )
    (checkout / "frontend/dist").mkdir(parents=True)
    (checkout / "frontend/dist/index.html").write_text("UI")
    monkeypatch.setattr(start, "ROOT", checkout)
    monkeypatch.setattr(start, "port_busy", lambda: False)
    monkeypatch.setattr(start, "camera_entries", lambda: "camera_1=/dev/v4l/by-id/example")
    monkeypatch.setattr(start.os, "geteuid", lambda: 1000)
    return checkout


def test_profile_is_private_and_existing_settings_are_preserved(root):
    assert start.create_profile(root, "camera_1=/stable/camera")
    path = root / ".env"
    content = path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "SO101_BACKEND=real" in content
    assert "SO101_CAMERAS=camera_1=/stable/camera\n" in content
    assert f"SO101_DATA_DIR={root}/data/real" in content
    path.write_text("operator's existing settings\n")
    assert not start.create_profile(root, "replacement")
    assert path.read_text() == "operator's existing settings\n"
    assert not list(root.glob(".env-start-*"))


def test_profile_refuses_symlink_without_changing_target(root):
    target = root / "operator.env"
    target.write_text("keep")
    (root / ".env").symlink_to(target)
    with pytest.raises(RuntimeError, match="symlinked .env"):
        start.create_profile(root, "")
    assert target.read_text() == "keep"


def test_camera_discovery_uses_capture_paths_and_deduplicates_aliases(tmp_path):
    endpoint = tmp_path / "video0"
    endpoint.touch()
    directory = tmp_path / "by-id"
    directory.mkdir()
    (directory / "camera-video-index0").symlink_to(endpoint)
    (directory / "alias-video-index0").symlink_to(endpoint)
    (directory / "camera-video-index1").symlink_to(endpoint)
    (directory / "missing-video-index0").symlink_to(tmp_path / "missing")
    assert start.camera_entries(directory) == f"camera_1={directory}/alias-video-index0"


def test_namespace_provisioning_uses_installer_permissions_and_rechecks(monkeypatch):
    states = iter((False, True))
    monkeypatch.setattr(start, "namespace_ready", lambda: next(states))
    calls = []
    monkeypatch.setattr(start.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    start.provision_namespace()
    assert calls == [
        (
            ["sudo", "install", "-d", "-o", "root", "-g", "dialout", "-m", "3770"]
            + [str(path) for path in start.DEVICE_DIRS],
            {"check": True},
        )
    ]


def test_namespace_ready_skips_sudo(monkeypatch):
    monkeypatch.setattr(start, "namespace_ready", lambda: True)
    monkeypatch.setattr(start.subprocess, "run", lambda *a, **kw: pytest.fail("unnecessary sudo"))
    start.provision_namespace()


def test_check_is_read_only_even_when_setup_is_missing(root, monkeypatch):
    monkeypatch.setattr(start, "namespace_ready", lambda: False)
    monkeypatch.setattr(start, "create_profile", lambda *a: pytest.fail("profile write"))
    monkeypatch.setattr(start, "provision_namespace", lambda: pytest.fail("sudo"))
    monkeypatch.setattr(start.os, "execv", lambda *a: pytest.fail("server started"))
    assert start.main(["--check"]) == 1
    assert not (root / ".env").exists()


def test_busy_port_refuses_before_any_setup(root, monkeypatch):
    monkeypatch.setattr(start, "port_busy", lambda: True)
    monkeypatch.setattr(start, "create_profile", lambda *a: pytest.fail("profile write"))
    monkeypatch.setattr(start, "provision_namespace", lambda: pytest.fail("sudo"))
    with pytest.raises(RuntimeError, match="occupied"):
        start.main([])


def test_failed_provisioning_never_starts_server(root, monkeypatch):
    def denied():
        raise subprocess.CalledProcessError(1, ["sudo", "install"])

    monkeypatch.setattr(start, "provision_namespace", denied)
    monkeypatch.setattr(start.os, "execv", lambda *a: pytest.fail("server started"))
    monkeypatch.setattr(start.subprocess, "run", lambda *a, **kw: pytest.fail("server info"))
    with pytest.raises(subprocess.CalledProcessError):
        start.main([])


def test_launch_only_runs_info_then_the_non_reloading_real_launcher(root, monkeypatch):
    monkeypatch.setattr(start, "provision_namespace", lambda: None)
    monkeypatch.setattr(start.os, "getgroups", lambda: [start.grp.getgrnam("dialout").gr_gid])
    calls = []
    monkeypatch.setattr(start.subprocess, "run", lambda command, **kw: calls.append(command))

    class Launched(Exception):
        pass

    def launch(path, argv):
        calls.append((path, argv))
        raise Launched

    monkeypatch.setattr(start.os, "execv", launch)
    with pytest.raises(Launched):
        start.main([])
    launcher = str(root / "config/launch-real.sh")
    assert calls == [[launcher, "info"], (launcher, [launcher])]
