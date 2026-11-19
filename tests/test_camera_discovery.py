"""`make find-cameras`: the discovery report and its text, with and without the packages."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from fakes import fake_pyrealsense2 as fake
from so101.hardware import realsense as rsmod
from so101.hardware.camera_discovery import (
    OPENCV_HINT,
    UVC_WARNING,
    discover_cameras,
    render_report,
)


class _Capture:
    """cv2.VideoCapture stand-in: indices in OPEN open; 3 opens but never delivers."""

    OPEN = {0: (480, 640), 2: (720, 1280), 3: None}

    def __init__(self, index: int) -> None:
        self.index = index

    def isOpened(self) -> bool:  # noqa: N802 - cv2 spelling
        return self.index in self.OPEN

    def read(self) -> tuple[bool, np.ndarray | None]:
        shape = self.OPEN.get(self.index)
        return (True, np.zeros((*shape, 3), np.uint8)) if shape else (False, None)

    def release(self) -> None:
        pass


def _cv2() -> ModuleType:
    module = ModuleType("cv2")
    module.VideoCapture = _Capture  # type: ignore[attr-defined]
    return module


def _v4l2(root: Path, names: dict[int, str]) -> Path:
    for index, name in names.items():
        (root / f"video{index}").mkdir(parents=True)
        (root / f"video{index}" / "name").write_text(name + "\n", encoding="utf-8")
    return root


def _usb(root: Path, port: str, product: str, speed: str) -> Path:
    device = root / port
    device.mkdir(parents=True)
    for name, value in (("idVendor", "8086"), ("idProduct", product), ("speed", speed)):
        (device / name).write_text(value + "\n", encoding="utf-8")
    return root


@pytest.fixture
def no_usb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "no-usb"
    monkeypatch.setattr(rsmod, "USB_SYSFS", path)
    return path


def test_missing_packages_print_hints_not_tracebacks(
    monkeypatch: pytest.MonkeyPatch, no_usb: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    report = discover_cameras(platform="linux")
    text = render_report(report)
    assert "pyrealsense2 is not importable" in text and 'pip install "so101[realsense]"' in text
    assert OPENCV_HINT in text and "Traceback" not in text
    assert text.endswith("no cameras found") and report.suggestion is None
    json.dumps(report.to_dict())


def test_devices_entries_and_warnings_off_linux(
    monkeypatch: pytest.MonkeyPatch, no_usb: Path
) -> None:
    world = fake.World()
    world.add(firmware="5.12.7.100")
    world.add(serial="999988887777", usb_type="2.1", name="Intel RealSense D405")
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.make_module(world))
    monkeypatch.setitem(sys.modules, "cv2", _cv2())
    report = discover_cameras(platform="darwin", max_index=4)
    text = render_report(report)
    assert (
        "Intel RealSense:\n  Intel RealSense D435  serial 123622270882  firmware 5.12.7.100" in text
    )
    assert "    color:          realsense:123622270882\n" in text
    assert "    color + depth:  realsense:123622270882+depth\n" in text
    assert "OpenCV indices 0..3:\n  camera 0: OK  640x480\n  camera 2: OK  1280x720\n" in text
    assert "firmware 5.12.7.100 is older than the recommended 5.16.0.1" in text
    assert (
        "Intel RealSense D405 (serial 999988887777): The camera is on a USB 2 link (USB 2.1)"
        in (text)
    )
    assert f"! {UVC_WARNING}" in text
    # Off Linux the RealSense's own UVC indices cannot be told apart: suggest RealSense only.
    assert text.endswith("Set:  SO101_CAMERAS=realsense:123622270882,realsense:999988887777")
    assert report.to_dict()["realsense"][1]["usb2"] is True


def test_linux_names_v4l2_nodes_and_reads_the_usb_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = fake.World()
    world.add()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.make_module(world))
    monkeypatch.setitem(sys.modules, "cv2", _cv2())
    v4l2 = _v4l2(tmp_path / "v4l2", {0: "Intel(R) RealSense(TM) Depth Ca", 2: "HD Webcam"})
    usb = _usb(tmp_path / "usb", "2-1", "0b07", "480")
    report = discover_cameras(platform="linux", max_index=4, sysfs=v4l2, usb_sysfs=usb)
    text = render_report(report)
    assert "  camera 0: OK  640x480  (Intel(R) RealSense(TM) Depth Ca)" in text
    assert (
        "USB bus (Linux sysfs):\n  Intel RealSense (USB 8086:0b07) at USB port 2-1, 480 Mbit/s"
        in (text)
    )
    assert "USB port 2-1, 480 Mbit/s: The camera is on a USB 2 link (480 Mbit/s)" in text
    assert report.bus_problem is None
    assert report.opencv[0].is_realsense and not report.opencv[1].is_realsense
    assert report.suggestion == "realsense:123622270882,2"
    json.dumps(report.to_dict())


def test_linux_bus_shows_a_camera_librealsense_cannot_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.make_module(fake.World()))
    monkeypatch.setitem(sys.modules, "cv2", None)
    usb = _usb(tmp_path / "usb", "2-1", "0b07", "5000")
    report = discover_cameras(platform="linux", usb_sysfs=usb, sysfs=tmp_path / "none")
    text = render_report(report)
    assert "  no RealSense found" in text
    assert "! A RealSense is on the USB bus but not opened by librealsense" in text
    assert "install/install.sh --realsense" in text and "USB 2 link" not in text


def test_enumeration_and_probe_errors_become_notes(
    monkeypatch: pytest.MonkeyPatch, no_usb: Path
) -> None:
    module = fake.make_module(fake.World())

    class Broken:
        def query_devices(self) -> None:
            raise RuntimeError("usb subsystem busy")

    module.context = Broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyrealsense2", module)
    cv2 = ModuleType("cv2")

    def explode(index: int) -> None:
        raise OSError("v4l2 went away")

    cv2.VideoCapture = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    report = discover_cameras(platform="linux")
    assert report.realsense_error == "could not list devices (RuntimeError: usb subsystem busy)"
    assert report.opencv_error == "probing OpenCV indices failed (OSError: v4l2 went away)"
    assert "no cameras found" in render_report(report)


def test_a_found_camera_without_a_serial_has_no_entries(
    monkeypatch: pytest.MonkeyPatch, no_usb: Path
) -> None:
    world = fake.World()
    world.add(unsupported=("serial_number", "usb_type_descriptor", "firmware_version"))
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.make_module(world))
    monkeypatch.setitem(sys.modules, "cv2", None)
    text = render_report(discover_cameras(platform="win32"))
    assert "serial unknown  firmware ? (recommended 5.16.0.1)  USB ?" in text
    assert "color:" not in text and text.endswith("no cameras found")
