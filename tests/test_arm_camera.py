"""Local relay refuses stale images and exposes one shared frame to clients."""

import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "arm_camera", Path(__file__).resolve().parents[1] / "scripts/arm_camera.py"
)
assert spec and spec.loader
camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera)


@pytest.mark.parametrize(("preview_fps", "minimum_frames"), [(15, 30), (30, 59)])
def test_frame_schedule_does_not_drop_half_the_nominal_frames(
    monkeypatch, preview_fps, minimum_frames
):
    import numpy as np

    stop = threading.Event()
    clock = SimpleNamespace(value=0.0)

    class Capture:
        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def get(self, *args):
            return 30.0

        def read(self):
            clock.value += 1 / 30
            if clock.value >= 2:
                stop.set()
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *args: Capture())
    monkeypatch.setattr(
        camera,
        "time",
        SimpleNamespace(monotonic=lambda: clock.value, time_ns=lambda: round(clock.value * 10**9)),
    )
    frames = camera.Frames()
    camera.producer(frames, stop, "fake", fps=30, preview_fps=preview_fps)
    assert frames.sequence >= minimum_frames


def test_requested_fps_cannot_hide_slow_camera_or_stale_stream():
    frames = camera.Frames()
    now = time.monotonic()
    frames.jpeg = b"jpeg"
    frames.updated = now
    frames.requested_capture_fps = frames.requested_preview_fps = 30
    frames.device_reported_fps = 30
    frames.frame_times.extend([now - 0.4, now - 0.2, now])
    frames.capture_times.extend(frames.frame_times)
    status = frames.status()
    assert status["actual_capture_fps"] == pytest.approx(5)
    assert status["actual_preview_fps"] == pytest.approx(5)
    assert status["device_reported_fps"] == 30
    assert not status["fps_target_met"]
    frames.updated = now - 3
    assert frames.status()["actual_preview_fps"] == 0


def test_disconnect_cleanup_failure_does_not_kill_reconnect_loop(monkeypatch):
    import numpy as np

    stop = threading.Event()
    attempts = []
    monkeypatch.setattr(stop, "wait", lambda timeout: None)

    class Capture:
        def __init__(self):
            attempts.append(self)
            self.disconnected = len(attempts) == 1

        def isOpened(self):
            return True

        def set(self, *args):
            return False

        def get(self, *args):
            return 30

        def read(self):
            if self.disconnected:
                raise RuntimeError("Device disconnected")
            stop.set()
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            if self.disconnected:
                raise RuntimeError("stop() cannot be called before start()")

    monkeypatch.setattr(camera, "RealSenseColorCapture", lambda *args: Capture())
    frames = camera.Frames()
    camera.producer(frames, stop, "realsense", preview_fps=30)
    assert len(attempts) == 2
    assert frames.sequence == 1
    assert frames.error is None


@pytest.mark.parametrize(
    ("serials", "selected", "expected"),
    [
        (["device-a"], None, "device-a"),
        (["device-b", "device-a"], "device-a", "device-a"),
        (["device-a", "device-b"], "device-b", "device-b"),
    ],
)
def test_realsense_relay_selects_one_device_color_only_and_bounds_frame_wait(
    monkeypatch, serials, selected, expected
):
    import sys

    import numpy as np

    calls = []
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    pipeline = SimpleNamespace(
        start=lambda config: calls.append(("start",)),
        wait_for_frames=lambda timeout: (
            calls.append(("wait", timeout))
            or SimpleNamespace(get_color_frame=lambda: SimpleNamespace(get_data=lambda: image))
        ),
        stop=lambda: calls.append(("stop",)),
    )
    config = SimpleNamespace(
        enable_device=lambda serial: calls.append(("device", serial)),
        enable_stream=lambda *args: calls.append(("stream", *args)),
    )
    monkeypatch.setitem(
        sys.modules,
        "pyrealsense2",
        SimpleNamespace(
            context=lambda: SimpleNamespace(
                query_devices=lambda: [
                    SimpleNamespace(get_info=lambda key, serial=serial: serial)
                    for serial in serials
                ]
            ),
            pipeline=lambda context: pipeline,
            config=lambda: config,
            camera_info=SimpleNamespace(serial_number="serial"),
            stream=SimpleNamespace(color="color"),
            format=SimpleNamespace(bgr8="bgr8"),
        ),
    )
    capture = camera.RealSenseColorCapture(640, 480, 30, selected)
    assert capture.read()[1] is image
    assert capture.read()[0]
    capture.release()
    assert ("device", expected) in calls
    assert ("stream", "color", 640, 480, "bgr8", 30) in calls
    assert [("wait", 10000), ("wait", 2000)] == [item for item in calls if item[0] == "wait"]
    assert calls[-1] == ("stop",)


@pytest.mark.parametrize("selected", [None, "missing", ""])
def test_realsense_selection_never_falls_back_to_another_camera(monkeypatch, selected):
    import sys

    def forbidden_pipeline(*args):
        raise AssertionError("Ambiguous/missing selection must not start a pipeline")

    monkeypatch.setitem(
        sys.modules,
        "pyrealsense2",
        SimpleNamespace(
            context=lambda: SimpleNamespace(
                query_devices=lambda: [
                    SimpleNamespace(get_info=lambda key: "device-a"),
                    SimpleNamespace(get_info=lambda key: "device-b"),
                ]
            ),
            pipeline=forbidden_pipeline,
            camera_info=SimpleNamespace(serial_number="serial"),
        ),
    )
    with pytest.raises((ValueError, RuntimeError), match="serial|selection"):
        camera.RealSenseColorCapture(640, 480, 30, selected)


@pytest.fixture
def relay():
    frames = camera.Frames()
    stop = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), camera.handler_for(frames, stop))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield frames, f"http://127.0.0.1:{server.server_port}"
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_initial_and_stale_frames_return_unavailable(relay):
    frames, base = relay
    for stale in (False, True):
        if stale:
            frames.jpeg = b"old frame"
            frames.updated = time.monotonic() - 3
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/snapshot.jpg", timeout=2)
        assert exc.value.code == 503
        assert not json.load(urllib.request.urlopen(base + "/status.json", timeout=2))["live"]


def test_fresh_frame_shared_between_snapshot_and_stream(relay):
    frames, base = relay
    frames.jpeg = b"test-jpeg-data"
    frames.updated = time.monotonic()
    frames.sequence = 1
    frames.captured_at_ns = 123456789
    response = urllib.request.urlopen(base + "/snapshot.jpg", timeout=2)
    assert response.read() == frames.jpeg
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-site"
    assert response.headers["X-SO101-Sequence"] == "1"
    assert response.headers["X-SO101-Captured-At-Ns"] == "123456789"
    response = urllib.request.urlopen(base + "/stream.mjpg", timeout=2)
    try:
        assert response.readline() == b"--frame\r\n"
        assert response.readline() == b"Content-Type: image/jpeg\r\n"
        assert response.readline() == b"Content-Length: 14\r\n"
        assert response.readline() == b"\r\n"
        assert response.read(14) == frames.jpeg
    finally:
        response.close()


def test_preview_and_unknown_paths(relay):
    _, base = relay
    page = urllib.request.urlopen(base, timeout=2).read()
    assert b"/status.json" in page and b"/stream.mjpg" in page
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base + "/missing", timeout=2)
    assert exc.value.code == 404
