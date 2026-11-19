"""CameraManager: stream order, mock RealSense color and depth, the sim refusal, info."""

from __future__ import annotations

import importlib.abc
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from so101.config import get_settings
from so101.hardware import cameras as cameras_module
from so101.hardware.cameras import (
    MOCK_DEPTH_FAR_MM,
    MOCK_DEPTH_NEAR_MM,
    CameraDevice,
    CameraManager,
    CameraUnavailableError,
    DepthCameraDevice,
)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_mock_opencv_entries_are_unchanged() -> None:
    frames = CameraManager(["0", "front=/dev/video2"]).read()
    assert list(frames) == ["0", "front=/dev/video2"]
    assert all(
        frame.shape == (480, 640, 3) and frame.dtype == np.uint8 for frame in frames.values()
    )


def test_mock_realsense_color_uses_the_entry_mode() -> None:
    frames = CameraManager(["realsense:1234:320x240@15"]).read()
    assert frames["realsense:1234:320x240@15"].shape == (240, 320, 3)


def test_mock_depth_is_a_tabletop_with_a_moving_object_and_holes() -> None:
    entries = ["a=realsense:1111:320x240+depth", "0", "b=realsense:2222+depth"]
    manager = CameraManager(entries)
    frames = manager.read()
    assert (
        list(frames)
        == manager.streams
        == [
            *entries,
            "a=realsense:1111:320x240+depth#depth",
            "b=realsense:2222+depth#depth",
        ]
    )
    color, depth = frames[entries[0]], frames[entries[0] + "#depth"]
    assert depth.shape == (240, 320, 1) and depth.dtype == np.uint16
    assert color.shape[:2] == depth.shape[:2]
    valid = depth[depth > 0]
    assert (depth == 0).any() and valid.min() >= MOCK_DEPTH_NEAR_MM
    assert valid.max() <= MOCK_DEPTH_FAR_MM
    # The box drawn in blue in the color frame stands proud of the table in depth.
    blue = (color[:, :, 2] == 255) & (color[:, :, 0] == 0)
    rows = np.linspace(MOCK_DEPTH_FAR_MM, MOCK_DEPTH_NEAR_MM, 240)[:, None].repeat(320, axis=1)
    box = blue & (depth[:, :, 0] > 0)
    assert box.any() and (depth[:, :, 0][box] < rows[box]).all()
    assert frames[entries[2] + "#depth"].shape == (480, 640, 1)


def test_info_before_and_after_open() -> None:
    entry = "w=realsense:1234+depth"
    manager = CameraManager([entry])
    before = manager.info(entry + "#depth")
    assert before == {
        "entry": entry,
        "kind": "realsense",
        "name": "w",
        "streams": [entry, entry + "#depth"],
        "backend": "mock",
        "open": False,
        "device": {},
    }
    with manager:
        described = manager.describe()[entry]
    assert described["open"] is True and described["device"]["synthetic"] is True
    assert described["device"]["streams"]["depth"]["unit"] == "mm"
    with pytest.raises(KeyError):
        manager.info("nope")
    plain = CameraManager(["0"])
    plain.open()
    assert plain.info("0")["device"]["streams"]["color"]["format"] == "rgb8"
    plain.close()


def test_a_device_without_depth_cannot_serve_a_depth_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[str] = []

    class ColorOnly:
        def read(self) -> np.ndarray:
            return np.zeros((2, 2, 3), np.uint8)

        def release(self) -> None:
            released.append("x")

    monkeypatch.setattr(CameraManager, "_open_one", lambda self, cid: ColorOnly())
    with pytest.raises(CameraUnavailableError, match="asks for \\+depth"):
        CameraManager(["realsense+depth"]).open()
    assert released == ["x"]


def test_close_releases_every_camera_even_if_one_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[str] = []

    class Camera:
        def __init__(self, cid: str) -> None:
            self.cid = cid

        def read(self) -> np.ndarray:
            return np.zeros((2, 2, 3), np.uint8)

        def release(self) -> None:
            released.append(self.cid)
            if self.cid == "0":
                raise RuntimeError("stuck")

    monkeypatch.setattr(CameraManager, "_open_one", lambda self, cid: Camera(cid))
    manager = CameraManager(["0", "1"])
    manager.open()
    manager.close()
    assert released == ["0", "1"] and manager.read().keys() == {"0", "1"}


def test_malformed_entries_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="camera entry '0' is listed twice"):
        CameraManager(["0", "0"])


def test_empty_camera_list_reads_nothing() -> None:
    assert CameraManager([]).read() == {} and CameraManager([]).streams == []


def test_default_entries_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, SO101_CAMERAS="1,realsense+depth")
    assert CameraManager().streams == ["1", "realsense+depth", "realsense+depth#depth"]


def test_error_text_carries_the_remediation() -> None:
    error = CameraUnavailableError("it broke.", remediation="Replug it.")
    assert str(error) == "it broke. Replug it." and error.detail == "it broke."
    assert str(CameraUnavailableError("plain")) == "plain"


def test_protocols() -> None:
    assert issubclass(cameras_module._MockCamera, CameraDevice)
    assert not issubclass(cameras_module._MockCamera, DepthCameraDevice)
    assert issubclass(cameras_module._MockDepthCamera, DepthCameraDevice)


def test_sim_camera_satisfies_the_camera_protocol_without_depth() -> None:
    pytest.importorskip("mujoco")
    from so101.simulation.devices import SimCamera

    assert issubclass(SimCamera, CameraDevice) and not issubclass(SimCamera, DepthCameraDevice)


# --- SO101_BACKEND=sim ---------------------------------------------------------------------------


class _ImportRecorder(importlib.abc.MetaPathFinder):
    """Records (and blocks) imports of camera stacks."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        if fullname.split(".")[0] in ("cv2", "pyrealsense2"):
            self.attempts.append(fullname)
            raise ImportError(f"blocked import of {fullname}")
        return None


@pytest.fixture
def sim_backend(monkeypatch: pytest.MonkeyPatch) -> _ImportRecorder:
    """SO101_BACKEND=sim with the default SO101_CAMERAS=0, and cv2/pyrealsense2 imports blocked."""
    _settings(monkeypatch, SO101_MOCK="0", SO101_BACKEND="sim", SO101_CAMERAS="0")
    recorder = _ImportRecorder()
    for name in ("cv2", "pyrealsense2"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    monkeypatch.setattr(cameras_module, "_IGNORED_UNDER_SIM", set())
    return recorder


class _SimFrames:
    """Stands in for make_sim_camera: records what was opened."""

    opened: list[str] = []

    def __init__(self, cid: str) -> None:
        self.cid = cid
        _SimFrames.opened.append(cid)

    def read(self) -> np.ndarray:
        return np.full((24, 32, 3), 7, np.uint8)

    def release(self) -> None:
        pass


def test_sim_backend_opens_sim_cameras_and_ignores_real_entries(
    sim_backend: _ImportRecorder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    devices = pytest.importorskip("so101.simulation.devices")
    _SimFrames.opened = []
    monkeypatch.setattr(devices, "make_sim_camera", _SimFrames)
    with caplog.at_level("WARNING", logger="so101.hardware.cameras"):
        manager = CameraManager()  # SO101_CAMERAS=0 is configured, as in .env
        again = CameraManager()
    assert manager.cam_ids == ["front", "wrist_cam"] == manager.streams == again.streams
    assert manager.feature_names == {"front": "front", "wrist_cam": "wrist_cam"}
    frames = manager.read()
    assert list(frames) == ["front", "wrist_cam"] and _SimFrames.opened == ["front", "wrist_cam"]
    manager.close()
    ignored = [record for record in caplog.records if "ignoring real camera" in record.message]
    assert len(ignored) == 1 and "entries 0;" in ignored[0].message  # one line, not per manager
    assert sim_backend.attempts == []


def test_sim_backend_ignores_real_entries_passed_in(
    sim_backend: _ImportRecorder, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="so101.hardware.cameras"):
        manager = CameraManager(["0", "cam=rtsp://user:s3cret@h/s", "realsense+depth"])
    assert manager.streams == [] and manager.read() == {}
    assert "s3cret" not in caplog.text and "cam=rtsp://h/s" in caplog.text
    assert sim_backend.attempts == []


def test_sim_backend_refuses_a_real_entry_that_reaches_open(
    sim_backend: _ImportRecorder,
) -> None:
    manager = CameraManager([])
    for entry in ("0", "realsense:1234+depth", "cam=rtsp://user:s3cret@h/s"):
        with pytest.raises(CameraUnavailableError) as caught:
            manager._open_one(entry)
        assert caught.value.detail.startswith("SO101_BACKEND=sim: ")
        assert "is not a simulated camera" in caught.value.detail
        assert "no real camera was opened" in caught.value.detail
        assert "s3cret" not in str(caught.value)
    assert sim_backend.attempts == []


def test_configured_streams_follow_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, SO101_CAMERAS="0,w=realsense:1234+depth")
    assert cameras_module.configured_streams() == [
        "0",
        "w=realsense:1234+depth",
        "w=realsense:1234+depth#depth",
    ]
    _settings(monkeypatch, SO101_MOCK="0", SO101_BACKEND="sim", SO101_SIM_CAMERAS="front")
    assert cameras_module.configured_streams() == ["front"]


def test_sim_camera_without_mujoco_is_a_camera_error(
    sim_backend: _ImportRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "so101.simulation.devices", None)
    with pytest.raises(CameraUnavailableError, match="sim camera 'front' needs MuJoCo") as caught:
        CameraManager(["front"]).open()
    assert "so101[sim]" in caught.value.remediation


def test_mock_backend_is_unaffected_by_the_sim_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, SO101_BACKEND="mock")
    assert CameraManager(["realsense+depth"]).read()["realsense+depth#depth"].dtype == np.uint16


def test_real_backend_still_reports_a_missing_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, SO101_MOCK="0")
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(CameraUnavailableError, match="no mock stream was substituted"):
        CameraManager(["front=0"]).read()


class _FakeCapture:
    """cv2.VideoCapture stand-in: opens indices 0 and 1 and the path /dev/video9."""

    def __init__(self, source: int | str) -> None:
        self.source = source
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - cv2 spelling
        return self.source in (0, 1, "/dev/video9")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.source == 1:
            return False, None
        frame = np.zeros((4, 6, 3), np.uint8)
        frame[:, :, 0] = 255  # blue in BGR
        return True, frame

    def release(self) -> None:
        self.released = True


def test_opencv_cameras_open_their_source_and_return_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, SO101_MOCK="0")
    cv2 = type(sys)("cv2")
    cv2.VideoCapture = _FakeCapture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    with CameraManager(["0", "side=/dev/video9"]) as manager:
        frames = manager.read()
        assert frames["0"][0, 0].tolist() == [0, 0, 255]  # converted to RGB
        assert manager._cams["side=/dev/video9"]._cap.source == "/dev/video9"  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="failed to read camera '1'"):
        CameraManager(["1"]).read()
    with pytest.raises(CameraUnavailableError, match="cannot open camera '7'"):
        CameraManager(["7"]).read()


def test_opencv_requests_bandwidth_mode_and_reports_actual_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(
        monkeypatch,
        SO101_MOCK="0",
        SO101_CAMERA_WIDTH="640",
        SO101_CAMERA_HEIGHT="480",
        SO101_CAMERA_FPS="15",
        SO101_CAMERA_FOURCC="MJPG",
    )
    calls: list[tuple[int, int]] = []

    class Capture(_FakeCapture):
        def set(self, prop: int, value: int) -> bool:
            calls.append((prop, value))
            return True

        def get(self, prop: int) -> float:
            # The driver may choose a different rate from the request.
            return {1: 640, 2: 480, 3: 30, 4: 1196444237}[prop]

    cv2 = type(sys)("cv2")
    cv2.VideoCapture = Capture  # type: ignore[attr-defined]
    cv2.CAP_PROP_FRAME_WIDTH = 1  # type: ignore[attr-defined]
    cv2.CAP_PROP_FRAME_HEIGHT = 2  # type: ignore[attr-defined]
    cv2.CAP_PROP_FPS = 3  # type: ignore[attr-defined]
    cv2.CAP_PROP_FOURCC = 4  # type: ignore[attr-defined]
    cv2.VideoWriter_fourcc = lambda *letters: 1196444237  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    with CameraManager(["front=0"]) as manager:
        assert manager.read()["front=0"].shape == (4, 6, 3)
        actual = manager.describe()["front=0"]["device"]["streams"]["color"]
    assert calls == [(4, 1196444237), (1, 640), (2, 480), (3, 15)]
    assert actual == {"width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "format": "rgb8"}
