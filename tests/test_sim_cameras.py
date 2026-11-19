"""Teleop recording under SO101_BACKEND=sim films with the simulated cameras.

The default ``.env`` keeps ``SO101_CAMERAS=0``. Under the sim backend that
entry is ignored (one log line), no OpenCV or RealSense module is imported,
and the recording API records the MuJoCo cameras ``front`` and ``wrist_cam``
under their own feature names, as ``so101 sim demos`` does.
"""

from __future__ import annotations

import importlib.abc
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101 import config  # noqa: E402
from so101.data.dataset import get_dataset  # noqa: E402
from so101.hardware import cameras as cameras_module  # noqa: E402
from so101.hardware.cameras import CameraUnavailableError  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import get_manager, reset_manager  # noqa: E402
from so101.simulation import devices  # noqa: E402

OPERATOR = "sim-camera-operator"


class _BlockCameraStacks(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        if fullname.split(".")[0] in ("cv2", "pyrealsense2"):
            self.attempts.append(fullname)
            raise ImportError(f"blocked import of {fullname}")
        return None


@pytest.fixture
def sim_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", "sim")
    monkeypatch.setenv("SO101_CAMERAS", "0")  # the .env default
    monkeypatch.setenv("SO101_SIM_IMAGE_SIZE", "32x24")
    monkeypatch.setenv("SO101_SIM_TASK", "reach")
    monkeypatch.setattr(cameras_module, "_IGNORED_UNDER_SIM", set())
    blocker = _BlockCameraStacks()
    for name in ("cv2", "pyrealsense2"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    config.get_settings.cache_clear()
    reset_manager()
    devices.reset_sim_runtime()
    try:
        devices.make_sim_camera("front").release()
    except CameraUnavailableError as exc:
        pytest.skip(f"no offscreen rendering here: {exc}")
    yield blocker
    reset_manager()
    devices.reset_sim_runtime()
    config.get_settings.cache_clear()


def _wait_for(predicate, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_recording_under_sim_films_the_simulated_cameras(
    sim_env: _BlockCameraStacks, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        caplog.at_level("WARNING", logger="so101.hardware.cameras"),
        TestClient(create_app()) as client,
    ):
        manager = get_manager()
        manager.operator_heartbeat(OPERATOR)
        started = client.post(
            "/api/recording/start",
            json={"repo_id": "sim/cameras", "task": "reach", "fps": 15, "operator_id": OPERATOR},
        )
        assert started.status_code == 200, started.text
        assert client.get("/api/robot/status").json()["cameras"] == ["front", "wrist_cam"]
        assert client.post("/api/recording/episode/start").status_code == 200
        assert _wait_for(lambda: manager.telemetry()["episode_frames"] >= 3)
        saved = client.post("/api/recording/episode/save")
        assert saved.status_code == 200, saved.text
        assert _wait_for(lambda: set(manager.status()["live_cameras"]) == {"front", "wrist_cam"})
        manager.soft_stop("test")
        assert _wait_for(lambda: manager.status()["mode"] == "idle")
    info = get_dataset("sim/cameras")
    assert info is not None and info.camera_keys == ["front", "wrist_cam"]
    chunk = next(Path(info.path).rglob("chunk_*.npy"))
    frames = np.load(chunk)
    assert frames.shape[1:] == (24, 32, 3) and frames.dtype == np.uint8
    assert sim_env.attempts == []  # no OpenCV or RealSense import was even tried
    assert caplog.text.count("ignoring real camera entries 0;") == 1
