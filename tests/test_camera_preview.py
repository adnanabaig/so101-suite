"""Camera-only preview never constructs or operates a motor device."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from so101.services.app import create_app
from so101.services.robot_manager import ControlConflict, Mode, get_manager


def test_camera_preview_opens_publishes_and_stops_without_motor_access(monkeypatch):
    manager = get_manager()
    initial_devices = (manager._follower, manager._leader)
    monkeypatch.setattr(manager, "follower", lambda: pytest.fail("preview opened follower"))
    monkeypatch.setattr(manager, "leader", lambda: pytest.fail("preview opened leader"))
    cameras = manager.cameras()
    observed = threading.Event()
    original_read = cameras.read

    def read():
        frames = original_read()
        # Read #2 proves read #1 was already published under the manager lock.
        if manager.get_frame(cameras.streams[0]) is not None:
            observed.set()
        return frames

    monkeypatch.setattr(cameras, "read", read)
    manager.start_camera_preview()
    try:
        assert observed.wait(2), "preview did not publish a frame"
        assert manager.camera_preview_status()["active"] is True
        assert manager.get_frame(cameras.streams[0]) is not None
        assert (manager._follower, manager._leader) == initial_devices
        with pytest.raises(ControlConflict, match="stop camera preview"):
            manager.claim_calibration("follower")
        with pytest.raises(RuntimeError, match="stop camera preview"):
            manager._start_thread(Mode.TELEOP, lambda: None, "operator")
    finally:
        manager.stop_camera_preview()
    assert manager.camera_preview_status()["active"] is False
    assert manager.get_frame(cameras.streams[0]) is None
    assert not cameras._cams


@pytest.mark.parametrize("busy", ["calibration", "mode", "armed"])
def test_preview_refuses_robot_activity_before_opening_cameras(monkeypatch, busy):
    manager = get_manager()
    monkeypatch.setattr(manager, "cameras", lambda: pytest.fail("busy preview opened cameras"))
    if busy == "calibration":
        manager.cal_in_progress = True
    elif busy == "mode":
        manager.mode = Mode.TELEOP
    else:
        manager._armed = True
    try:
        with pytest.raises(RuntimeError, match="busy|external control"):
            manager.start_camera_preview()
    finally:
        manager.cal_in_progress = False
        manager.mode = Mode.IDLE
        manager._armed = False


def test_preview_camera_error_is_visible_without_robot_fault(monkeypatch):
    manager = get_manager()
    initial_devices = (manager._follower, manager._leader)
    cameras = manager.cameras()
    monkeypatch.setattr(
        cameras, "open", lambda: (_ for _ in ()).throw(RuntimeError("camera unplugged"))
    )
    manager.start_camera_preview()
    manager.stop_camera_preview()
    state = manager.camera_preview_status()
    assert state["active"] is False and state["error"] == "camera unplugged"
    assert manager._fault is None
    assert (manager._follower, manager._leader) == initial_devices


def test_preview_api_status_and_control_routes(monkeypatch):
    manager = get_manager()
    monkeypatch.setattr(manager, "follower", lambda: pytest.fail("preview API opened follower"))
    monkeypatch.setattr(manager, "leader", lambda: pytest.fail("preview API opened leader"))
    with TestClient(create_app()) as client:
        assert client.get("/api/cameras/preview/status").json()["active"] is False
        assert client.post("/api/cameras/preview/start").status_code == 200
        assert client.post("/api/cameras/preview/start").status_code == 409
        assert client.post("/api/cameras/preview/stop").json()["active"] is False


def test_preview_construction_failure_releases_reservation(monkeypatch):
    manager = get_manager()

    def broken():
        raise RuntimeError("camera construction failed")

    monkeypatch.setattr(manager, "cameras", broken)
    monkeypatch.setattr(manager, "follower", lambda: pytest.fail("opened follower"))
    monkeypatch.setattr(manager, "leader", lambda: pytest.fail("opened leader"))
    manager.start_camera_preview()
    manager.stop_camera_preview()
    assert manager.camera_preview_status()["active"] is False
    assert manager.camera_preview_status()["error"] == "camera construction failed"


def test_preview_cleanup_failure_keeps_capture_reserved(monkeypatch):
    manager = get_manager()
    cameras = manager.cameras()

    def broken_close():
        raise RuntimeError("camera release failed")

    monkeypatch.setattr(cameras, "close", broken_close)
    monkeypatch.setattr(manager, "follower", lambda: pytest.fail("opened follower"))
    monkeypatch.setattr(manager, "leader", lambda: pytest.fail("opened leader"))
    manager.start_camera_preview()
    manager.stop_camera_preview()
    status = manager.camera_preview_status()
    assert status["active"] is True
    assert "camera cleanup failed: camera release failed" in status["error"]
    with pytest.raises(RuntimeError, match="already running or stopping"):
        manager.start_camera_preview()
