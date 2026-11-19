"""Intel RealSense capture against a fake pyrealsense2 (no camera needed)."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from fakes import fake_pyrealsense2 as fake
from so101.config import get_settings
from so101.hardware import realsense as rsmod
from so101.hardware.camera_spec import parse_camera_entry
from so101.hardware.cameras import CameraManager, CameraUnavailableError
from so101.hardware.realsense import (
    RealSenseCamera,
    RealSenseDevice,
    StreamMode,
    choose_color_mode,
    choose_depth_mode,
    depth_to_mm,
    firmware_older,
    install_hint,
    list_realsense_devices,
    not_found_remediation,
    open_realsense,
    remediation_for,
    udev_rules_path,
)

SERIAL = "123622270882"


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[fake.World]:
    """A fake pyrealsense2 with fast timings; tests add devices and failures."""
    world = fake.World()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.make_module(world))
    # Hermetic: never read this machine's USB bus (it may have a real camera).
    monkeypatch.setattr(rsmod, "USB_SYSFS", tmp_path / "no-usb-bus")
    monkeypatch.setattr(rsmod, "WARMUP_S", 0.0)
    monkeypatch.setattr(rsmod, "FRAME_WAIT_MS", 20)
    monkeypatch.setattr(rsmod, "FIRST_FRAME_TIMEOUT_S", 0.5)
    monkeypatch.setattr(rsmod, "RESET_SETTLE_S", 0.0)
    monkeypatch.setattr(rsmod, "RESET_POLL_S", 0.0)
    yield world
    for pipe in world.pipelines:
        assert not pipe.started or pipe.stopped, "a pipeline was left running"


def _camera(entry: str) -> RealSenseCamera:
    camera = RealSenseCamera(parse_camera_entry(entry))
    camera.connect()
    return camera


# --- the fake against the real package ------------------------------------------------------------


def test_every_name_the_fake_models_exists_in_pyrealsense2() -> None:
    try:
        import pyrealsense2 as real
    except ImportError:
        pytest.skip("pyrealsense2 is not installed (the realsense extra)")
    missing = [
        f"{name}.{attr}" if attr else name
        for name, attrs in fake.MODELED.items()
        for attr in ([""] if not hasattr(real, name) else attrs)
        if not hasattr(real, name) or (attr and not hasattr(getattr(real, name), attr))
    ]
    assert missing == []
    for path, keyword in fake.MODELED_KEYWORDS:
        owner, attr = path.split(".")
        assert keyword in (getattr(getattr(real, owner), attr).__doc__ or ""), path


# --- import and install hints ---------------------------------------------------------------------


def test_missing_pyrealsense2_raises_with_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    with pytest.raises(CameraUnavailableError) as caught:
        list_realsense_devices()
    assert "pyrealsense2 is not importable" in caught.value.detail
    assert 'pip install "so101[realsense]"' in caught.value.remediation
    assert caught.value.remediation in str(caught.value)


def test_a_pyrealsense2_without_pipeline_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyrealsense2", ModuleType("pyrealsense2"))
    with pytest.raises(CameraUnavailableError, match="has no pipeline"):
        rsmod.import_pyrealsense2()


def test_install_hints_name_the_distribution_for_each_os() -> None:
    assert "pyrealsense2-macosx" in install_hint("darwin") and "sudo" in install_hint("darwin")
    linux = install_hint("linux")
    assert "(it brings pyrealsense2)" in linux and str(udev_rules_path()) in linux
    assert "udevadm control --reload-rules" in linux
    windows = install_hint("win32")
    assert "(it brings pyrealsense2)" in windows and "udev" not in windows
    assert rsmod.distribution_name("darwin") == "pyrealsense2-macosx"
    assert rsmod.distribution_name("win32") == "pyrealsense2"


def test_vendored_udev_rules_keep_the_version_marker_first() -> None:
    path = udev_rules_path()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert path.name == "99-realsense-libusb.rules"
    assert lines[0] == "##Version=1.1##"
    assert any("v2.56.5" in line and "Source:" in line for line in lines[:12])
    assert any('ATTRS{idVendor}=="8086"' in line for line in lines)


# --- remediation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "platform", "usb", "expected", "absent"),
    [
        ("RS2_USB_STATUS_ACCESS", "linux", None, "install/install.sh --realsense", "USB 2"),
        ("failed to set power state", "linux", None, "udevadm trigger", "sudo cp x"),
        ("failed to set power state", "linux", None, "only one program at a time", "USB 2"),
        ("Permission denied", "linux", "2.1", "USB 2 link (USB 2.1)", "macOS"),
        ("Couldn't resolve requests", "linux", None, "cannot stream the requested modes", "udev"),
        ("Device or resource busy", "linux", None, "only one program at a time", "udev"),
        ("anything", "win32", None, "close RealSense Viewer", "udev"),
        ("anything", "darwin", None, "run so101 with sudo", "udev"),
        ("anything", "darwin", None, "OpenCV camera", "only one program"),
        ("something odd", "linux", None, "Replug the camera", "udev"),
    ],
)
def test_remediation_matches_the_os_and_the_error(
    error: str, platform: str, usb: str | None, expected: str, absent: str
) -> None:
    text = remediation_for(error, platform=platform, usb_type=usb)
    assert expected in text and absent not in text


def test_not_found_remediation_per_os() -> None:
    assert "lsusb" in not_found_remediation("linux")
    assert "sudo" in not_found_remediation("darwin")
    assert not_found_remediation("win32").startswith("Check that it is plugged in")


# --- enumeration ---------------------------------------------------------------------------------


def test_list_devices_reads_supported_camera_info_only(world: fake.World) -> None:
    world.add()
    world.add(serial="f0245826", name="Intel RealSense L515", unsupported=("usb_type_descriptor",))
    first, second = list_realsense_devices()
    assert first == RealSenseDevice(
        name="Intel RealSense D435",
        serial=SERIAL,
        firmware="5.16.0.1",
        recommended_firmware="5.16.0.1",
        usb_type="3.2",
        product_line="D400",
        product_id="0B07",
        physical_port=f"/fake/usb/{SERIAL}",
    )
    assert second.usb_type is None and second.serial == "f0245826"
    assert second.usb2 is False and first.firmware_outdated is False
    assert first.entries == [f"realsense:{SERIAL}", f"realsense:{SERIAL}+depth"]
    assert first.to_dict()["usb2"] is False and first.label.endswith(f"(serial {SERIAL})")


def test_a_failing_camera_info_field_is_left_out(world: fake.World) -> None:
    world.add()
    world.info_error = True
    (device,) = list_realsense_devices()
    assert device.physical_port is None and device.serial == SERIAL


def test_device_flags() -> None:
    old = RealSenseDevice(serial="1234", firmware="5.12.7.100", recommended_firmware="5.16.0.1")
    assert old.firmware_outdated and RealSenseDevice(usb_type="2.1").usb2
    assert (
        RealSenseDevice().entries == [] and RealSenseDevice().label == "RealSense (serial unknown)"
    )


@pytest.mark.parametrize(
    ("firmware", "recommended", "older"),
    [
        ("5.12.7.100", "5.16.0.1", True),
        ("5.16.0.1", "5.16.0.1", False),
        ("5.17.0.10", "5.16.0.1", False),
        ("5.16", "5.16.0.1", True),
        ("unknown", "5.16.0.1", False),
        (None, "5.16.0.1", False),
        ("5.12.7.100", None, False),
    ],
)
def test_firmware_comparison(firmware: str | None, recommended: str | None, older: bool) -> None:
    assert firmware_older(firmware, recommended) is older


# --- selection -----------------------------------------------------------------------------------


def test_no_device_attached(world: fake.World) -> None:
    with pytest.raises(CameraUnavailableError) as caught:
        open_realsense(parse_camera_entry("realsense"))
    assert caught.value.detail == "no RealSense camera is attached"
    assert "make find-cameras" in caught.value.remediation


def test_the_only_device_is_used_without_a_serial(world: fake.World) -> None:
    world.add()
    camera = open_realsense(parse_camera_entry("realsense"))
    try:
        assert camera.read().shape == (480, 640, 3)
        assert world.pipelines[-1].config.serial == SERIAL  # enable_device pins it
    finally:
        camera.release()


def test_several_devices_need_a_serial(world: fake.World) -> None:
    world.add()
    world.add(serial="999999999999", usb_type="2.1")
    with pytest.raises(CameraUnavailableError) as caught:
        open_realsense(parse_camera_entry("realsense+depth"))
    detail = caught.value.detail
    assert "2 RealSense cameras are attached" in detail
    assert (
        f"realsense:{SERIAL} or realsense:{SERIAL}+depth (Intel RealSense D435, USB 3.2)" in detail
    )
    assert "realsense:999999999999 or realsense:999999999999+depth" in detail
    assert world.starts == 0


def test_serial_selection_is_case_insensitive_and_lists_what_is_attached(world: fake.World) -> None:
    world.add(serial="F0245826")
    camera = _camera("realsense:f0245826")
    camera.release()
    with pytest.raises(CameraUnavailableError) as caught:
        _camera("realsense:111122223333")
    assert "no RealSense with serial 111122223333 is attached (attached: realsense:F0245826" in (
        caught.value.detail
    )


# --- stream modes --------------------------------------------------------------------------------

MODES = [StreamMode(*mode) for mode in sorted(fake.COLOR_MODES)]


def test_color_mode_exact_and_explicit_errors() -> None:
    assert choose_color_mode(parse_camera_entry("realsense:1280x720"), MODES) == StreamMode(
        1280, 720, 30
    )
    with pytest.raises(CameraUnavailableError) as caught:
        choose_color_mode(parse_camera_entry("realsense:1920x1080@30"), MODES)
    assert "no rgb8 color mode 1920x1080@30; it offers 424x240@60, 640x480@15" in (
        caught.value.detail
    )
    with pytest.raises(CameraUnavailableError, match="offers no rgb8 color stream"):
        choose_color_mode(parse_camera_entry("realsense"), [])


def test_default_color_mode_falls_back_to_the_nearest(caplog: pytest.LogCaptureFixture) -> None:
    default = parse_camera_entry("realsense")
    at_30 = [StreamMode(1280, 720, 30), StreamMode(848, 480, 30), StreamMode(640, 480, 15)]
    with caplog.at_level("INFO"):
        assert choose_color_mode(default, at_30) == StreamMode(848, 480, 30)
    assert "using 848x480@30" in caplog.text
    any_fps = [StreamMode(1280, 720, 15), StreamMode(640, 480, 6), StreamMode(640, 480, 15)]
    assert choose_color_mode(default, any_fps) == StreamMode(640, 480, 15)


def test_depth_mode_preference() -> None:
    depth = [StreamMode(*mode) for mode in fake.DEPTH_MODES]
    assert choose_depth_mode(StreamMode(1280, 720, 30), depth) == StreamMode(1280, 720, 30)
    assert choose_depth_mode(StreamMode(424, 240, 30), depth) == StreamMode(640, 480, 30)
    no_vga = [StreamMode(848, 480, 30), StreamMode(1280, 720, 30)]
    assert choose_depth_mode(StreamMode(424, 240, 30), no_vga) == StreamMode(848, 480, 30)
    odd = [StreamMode(256, 144, 90), StreamMode(1280, 720, 90)]
    assert choose_depth_mode(StreamMode(424, 240, 90), odd) == StreamMode(256, 144, 90)
    with pytest.raises(CameraUnavailableError, match="no z16 depth mode at 15 fps"):
        choose_depth_mode(StreamMode(640, 480, 15), depth)


# --- depth units ---------------------------------------------------------------------------------


def test_depth_scale_0_001_is_the_identity() -> None:
    raw = np.array([[0, 1, 1234, 65535]], dtype=np.uint16)
    mm = depth_to_mm(raw, 0.001)
    assert mm.shape == (1, 4, 1) and mm.dtype == np.uint16
    assert mm[:, :, 0].tolist() == [[0, 1, 1234, 65535]]


def test_depth_scale_0_0001_divides_by_ten_and_rounds() -> None:
    raw = np.array([[0, 4, 5, 15, 10000, 65535]], dtype=np.uint16)
    assert depth_to_mm(raw, 0.0001)[:, :, 0].tolist() == [[0, 0, 0, 2, 1000, 6554]]


def test_coarser_depth_units_clip_at_65535() -> None:
    raw = np.array([[0, 100, 6553, 6554, 65535]], dtype=np.uint16)
    assert depth_to_mm(raw, 0.01)[:, :, 0].tolist() == [[0, 1000, 65530, 65535, 65535]]
    assert depth_to_mm(raw[:, :, None], 0.00025)[:, :, 0].tolist() == [[0, 25, 1638, 1638, 16384]]


# --- streaming -----------------------------------------------------------------------------------


def test_color_and_aligned_depth_from_one_frameset(world: fake.World) -> None:
    world.add(depth_scale=0.0001)  # a D405 reports 0.1 mm units
    camera = _camera(f"realsense:{SERIAL}:1280x720+depth")
    try:
        color, depth = camera.read_rgbd()
        assert color.shape == (720, 1280, 3) and color.dtype == np.uint8
        assert depth.shape == (720, 1280, 1) and depth.dtype == np.uint16
        assert int(depth[0, 0, 0]) == 100  # 1000 raw units at 0.1 mm
        assert camera._align is not None and camera._align.calls > 0
        cfg = world.pipelines[-1].config
        assert cfg.streams == {"color": (1280, 720, "rgb8", 30), "depth": (1280, 720, "z16", 30)}
    finally:
        camera.release()


def test_color_and_depth_come_from_the_same_frameset(world: fake.World) -> None:
    world.add(depth_scale=0.001)
    world.depth_tracks_index = True  # depth = frameset index (mm), color = index % 256
    camera = _camera(f"realsense:{SERIAL}+depth")
    try:
        for _ in range(5):
            color, depth = camera.read_rgbd()
            assert int(depth[0, 0, 0]) % 256 == int(color[0, 0, 0])
            time.sleep(0.003)
    finally:
        camera.release()


def test_depth_is_resampled_onto_the_color_grid(world: fake.World) -> None:
    world.add()
    world.raw_depth = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    camera = _camera(f"realsense:{SERIAL}:424x240@60+depth")
    try:
        _, depth = camera.read_rgbd()
        # Color is 424x240@60; depth falls back to a 60 fps mode and is aligned to color.
        assert world.pipelines[-1].config.streams["depth"] == (480, 270, "z16", 60)
        assert depth.shape == (240, 424, 1)
        assert depth[0, 0, 0] == 100 and depth[-1, -1, 0] == 400
    finally:
        camera.release()


def test_info_reports_the_device_and_the_streams_opened(world: fake.World) -> None:
    world.add(depth_scale=0.0001, usb_type="2.1", firmware="5.12.7.100")
    camera = _camera(f"realsense:{SERIAL}+depth")
    try:
        info = camera.info()
        assert info["serial"] == SERIAL and info["usb2"] is True
        assert info["firmware_outdated"] is True and info["depth"] is True
        assert info["depth_scale"] == 0.0001
        assert info["streams"]["color"] == {
            "width": 640,
            "height": 480,
            "fps": 30,
            "format": "rgb8",
        }
        assert info["streams"]["depth"]["aligned_to"] == "color"
        assert info["streams"]["depth"]["format"] == "z16"
        thread = camera._thread
        assert thread is not None and thread.daemon and thread.name == f"so101-realsense-{SERIAL}"
    finally:
        camera.release()


def test_color_only_has_no_depth(world: fake.World) -> None:
    world.add()
    camera = _camera("realsense")
    try:
        assert camera.info()["depth_scale"] is None and "depth" not in camera.info()["streams"]
        with pytest.raises(CameraUnavailableError, match="add \\+depth"):
            camera.read_rgbd()
    finally:
        camera.release()


def test_stale_frames_are_refused(world: fake.World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rsmod, "STALE_AFTER_S", 0.05)
    world.add()
    camera = _camera("realsense")
    try:
        camera.read()
        world.stall = True
        time.sleep(0.2)
        with pytest.raises(CameraUnavailableError) as caught:
            camera.read()
        assert "s old (no frameset within 20 ms)" in caught.value.detail
    finally:
        camera.release()


def test_a_dead_capture_thread_is_reported_with_its_error(world: fake.World) -> None:
    world.add(usb_type="2.0")
    camera = _camera("realsense")
    try:
        world.wait_error = RuntimeError("Frame didn't arrive within 5000")
        thread = camera._thread
        assert thread is not None
        thread.join(2)
        with pytest.raises(CameraUnavailableError) as caught:
            camera.read()
        assert "capture thread stopped (RuntimeError: Frame didn't arrive" in caught.value.detail
        assert "USB 2 link (USB 2.0)" in caught.value.remediation
    finally:
        camera.release()


def test_no_first_frame_after_warm_up(world: fake.World) -> None:
    world.add()
    world.stall_after = 1  # warm-up sees one frameset, the capture thread none
    camera = _camera("realsense")
    try:
        with pytest.raises(CameraUnavailableError, match="no frame within 0.5 s"):
            camera.read()
    finally:
        camera.release()


def test_thread_death_before_the_first_frame(world: fake.World) -> None:
    world.add()
    world.wait_error = RuntimeError("usb unplugged")
    world.frames_before_error = 1
    camera = _camera("realsense")
    try:
        with pytest.raises(CameraUnavailableError, match="capture thread stopped"):
            camera.read()
    finally:
        camera.release()


def test_incomplete_framesets_are_skipped(world: fake.World) -> None:
    world.add()
    world.drop_depth = True
    camera = _camera(f"realsense:{SERIAL}+depth")
    try:
        with pytest.raises(CameraUnavailableError, match="without a depth frame"):
            camera.read_rgbd()
    finally:
        camera.release()


# --- connect, reset, release ---------------------------------------------------------------------


def test_the_last_attempt_follows_a_hardware_reset(world: fake.World) -> None:
    device = world.add()
    world.start_errors = ["xioctl failed", "Device or resource busy"]
    world.reset_hides_for = 2  # gone for two enumerations, then back
    camera = _camera(f"realsense:{SERIAL}")
    try:
        assert device.resets == 1 and world.starts == 3
        assert camera.read().shape == (480, 640, 3)
    finally:
        camera.release()


def test_no_reset_when_an_earlier_attempt_works(world: fake.World) -> None:
    device = world.add()
    world.start_errors = ["xioctl failed"]
    camera = _camera(f"realsense:{SERIAL}")
    camera.release()
    assert device.resets == 0 and world.starts == 2


def test_a_device_that_never_returns_after_reset(
    world: fake.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rsmod, "RESET_REENUMERATE_TIMEOUT_S", 0.05)
    world.add()
    world.start_errors = ["a", "b"]
    world.reset_hides_for = None
    with pytest.raises(CameraUnavailableError, match="did not come back within 0.05 s"):
        open_realsense(parse_camera_entry("realsense"))


def test_every_attempt_failing_gives_os_specific_remediation(
    world: fake.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    world.add()
    world.start_errors = ["e1", "e2", "failed to open usb interface: RS2_USB_STATUS_ACCESS"]
    with pytest.raises(CameraUnavailableError) as caught:
        open_realsense(parse_camera_entry(f"realsense:{SERIAL}"))
    assert "did not start after 3 attempts (RuntimeError: failed to open usb" in (
        caught.value.detail
    )
    assert "Install the RealSense udev rules" in caught.value.remediation


def test_warm_up_without_frames_fails_every_attempt(world: fake.World) -> None:
    world.add()
    world.stall = True
    with pytest.raises(CameraUnavailableError, match="TimeoutError: no frameset within 0.5 s"):
        open_realsense(parse_camera_entry("realsense"))
    assert world.starts == 3


def test_an_invalid_depth_scale_is_refused(world: fake.World) -> None:
    world.add(depth_scale=0.0)
    with pytest.raises(CameraUnavailableError, match="depth scale of 0.0"):
        open_realsense(parse_camera_entry("realsense+depth"))


def test_hardware_reset_errors_and_enumeration_errors_are_tolerated(
    world: fake.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = world.add()
    world.start_errors = ["a", "b"]
    calls = {"n": 0}
    original = fake.context.query_devices

    def flaky(self: fake.context) -> fake.device_list:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("bus busy")
        return original(self)

    monkeypatch.setattr(fake.context, "query_devices", flaky)
    monkeypatch.setattr(fake.device, "hardware_reset", lambda self: (_ for _ in ()).throw(OSError))
    camera = _camera("realsense")
    camera.release()
    assert device.resets == 0 and calls["n"] >= 3


def test_release_is_idempotent_and_stops_thread_then_pipeline(world: fake.World) -> None:
    world.add()
    camera = _camera("realsense")
    thread = camera._thread
    pipe = world.pipelines[-1]
    camera.release()
    camera.release()
    assert thread is not None and not thread.is_alive() and pipe.stopped
    with pytest.raises(CameraUnavailableError, match="is not connected"):
        camera.read()
    never = RealSenseCamera(parse_camera_entry("realsense"))
    never.release()  # never connected: nothing to stop


def test_release_from_the_capture_thread_does_not_deadlock(world: fake.World) -> None:
    world.add()
    camera = _camera("realsense")
    capture = camera._thread
    camera._thread = threading.current_thread()  # as if release() ran inside the thread
    camera.release()  # must not join itself
    assert capture is not None
    capture.join(2)
    assert not capture.is_alive() and world.pipelines[-1].stopped


# --- through CameraManager -----------------------------------------------------------------------


@pytest.fixture
def real_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_camera_manager_returns_color_then_depth(world: fake.World, real_mode: None) -> None:
    world.add()
    entry = f"wrist=realsense:{SERIAL}+depth"
    with CameraManager([entry]) as cameras:
        frames = cameras.read()
        assert list(frames) == [entry, entry + "#depth"] == cameras.streams
        assert frames[entry + "#depth"].shape == (480, 640, 1)
        info = cameras.info(entry + "#depth")
        assert info["kind"] == "realsense" and info["device"]["serial"] == SERIAL


def test_camera_manager_wraps_realsense_failures(world: fake.World, real_mode: None) -> None:
    cameras = CameraManager(["realsense"])
    with pytest.raises(CameraUnavailableError) as caught:
        cameras.open()
    assert caught.value.detail == (
        "real camera 'realsense' is unavailable: no RealSense camera is attached; "
        "no mock stream was substituted"
    )
    assert "make find-cameras" in caught.value.remediation


def test_camera_manager_reports_a_missing_pyrealsense2(
    monkeypatch: pytest.MonkeyPatch, real_mode: None
) -> None:
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    with pytest.raises(CameraUnavailableError, match="pyrealsense2 is not importable") as caught:
        CameraManager(["realsense:1234"]).read()
    assert "so101[realsense]" in caught.value.remediation


def test_a_device_without_a_readable_serial(world: fake.World) -> None:
    world.add(unsupported=("serial_number",))
    camera = _camera("realsense")
    try:
        assert world.pipelines[-1].config.serial is None
        assert camera._thread is not None and camera._thread.name == "so101-realsense-device"
    finally:
        camera.release()
    world.add(serial="5555")
    with pytest.raises(CameraUnavailableError, match="without a readable serial"):
        _camera("realsense")


def test_warm_up_drops_the_first_frames(world: fake.World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rsmod, "WARMUP_S", 0.03)
    world.add()
    camera = _camera("realsense")
    try:
        assert world.pipelines[-1].frames_served > 2
    finally:
        camera.release()


def test_reset_waits_for_the_right_serial(world: fake.World) -> None:
    world.add(serial="999988887777")
    target = world.add()
    world.start_errors = ["a", "b"]
    camera = _camera(f"realsense:{SERIAL}")
    camera.release()
    assert target.resets == 1


def test_a_frameset_without_color_is_skipped(world: fake.World) -> None:
    world.add()
    camera = _camera("realsense")
    try:
        assert camera._convert(fake.composite_frame(None, None)) is None
        assert camera._last_error == "a frameset arrived without a color frame"
    finally:
        camera.release()


def test_a_failing_pipeline_stop_is_tolerated(
    world: fake.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add()
    camera = _camera("realsense")
    monkeypatch.setattr(fake.pipeline, "stop", lambda self: (_ for _ in ()).throw(RuntimeError))
    camera.release()
    world.pipelines.clear()  # nothing could be stopped; skip the fixture's check


# --- Linux: the USB bus in sysfs -----------------------------------------------------------------


def _usb_device(root: Path, port: str, vendor: str, product: str, speed: str, name: str) -> None:
    device = root / port
    device.mkdir(parents=True)
    for file_name, value in (
        ("idVendor", vendor),
        ("idProduct", product),
        ("speed", speed),
        ("product", name),
    ):
        if value:
            (device / file_name).write_text(value + "\n", encoding="utf-8")


@pytest.fixture
def usb_bus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty /sys/bus/usb/devices; tests plug devices in with _usb_device()."""
    root = tmp_path / "usb"
    root.mkdir()
    (root / "usb1").mkdir()  # a root hub without idVendor
    _usb_device(root, "1-2", "046d", "0825", "480", "Webcam C270")
    monkeypatch.setattr(rsmod, "USB_SYSFS", root)
    return root


def test_usb_ids_come_from_the_vendored_rules() -> None:
    ids = rsmod.realsense_usb_ids()
    assert ids["0b07"] is False and ids["0b3a"] is False  # D435, D435i
    assert ids["0adb"] is True  # a recovery (DFU) id
    assert "2150" not in ids  # Movidius vendor 03e7 is not 8086


def test_scan_reads_link_speed_and_ignores_other_devices(usb_bus: Path) -> None:
    _usb_device(usb_bus, "2-1", "8086", "0b07", "5000", "Intel(R) RealSense(TM) Depth Camera 435")
    _usb_device(usb_bus, "1-4", "8086", "0B3A", "480", "")
    _usb_device(usb_bus, "1-5", "8086", "1234", "480", "Some Intel thing")
    _usb_device(usb_bus, "1-6", "8086", "0b5c", "weird", "D455")
    found = rsmod.scan_usb_realsense()
    assert [(d.port, d.product_id, d.speed_mbps, d.usb2) for d in found] == [
        ("1-4", "0b3a", 480.0, True),
        ("1-6", "0b5c", None, False),
        ("2-1", "0b07", 5000.0, False),
    ]
    assert found[2].describe() == (
        "Intel(R) RealSense(TM) Depth Camera 435 (USB 8086:0b07) at USB port 2-1, 5000 Mbit/s"
    )
    assert found[0].describe().startswith("Intel RealSense (USB 8086:0b3a)")
    assert found[1].link == "unknown speed"
    assert rsmod.usb_link_mbps() is None  # three cameras: ambiguous


def test_no_sysfs_means_no_bus_information(tmp_path: Path) -> None:
    assert rsmod.scan_usb_realsense(tmp_path / "missing") == []
    assert rsmod.bus_diagnosis(0, tmp_path / "missing") is None


def test_a_camera_librealsense_cannot_see_points_at_the_udev_rules(
    world: fake.World, usb_bus: Path
) -> None:
    _usb_device(usb_bus, "2-1", "8086", "0b07", "5000", "Intel(R) RealSense(TM) Depth Camera 435")
    with pytest.raises(CameraUnavailableError) as caught:
        open_realsense(parse_camera_entry("realsense"))
    assert caught.value.detail == (
        "no RealSense camera is attached; a RealSense is on the USB bus but not opened by "
        "librealsense: Intel(R) RealSense(TM) Depth Camera 435 (USB 8086:0b07) at USB port 2-1, "
        "5000 Mbit/s"
    )
    assert caught.value.remediation.startswith("Install the RealSense udev rules")
    assert "install/install.sh --realsense" in caught.value.remediation
    assert str(udev_rules_path()) in caught.value.remediation


def test_a_camera_in_recovery_mode_needs_firmware(world: fake.World, usb_bus: Path) -> None:
    _usb_device(usb_bus, "2-1", "8086", "0adb", "480", "Intel(R) RealSense(TM) DFU")
    with pytest.raises(CameraUnavailableError) as caught:
        open_realsense(parse_camera_entry("realsense:1234"))
    assert "in recovery (DFU) mode on the USB bus" in caught.value.detail
    assert "firmware" in caught.value.remediation


def test_a_visible_camera_needs_no_bus_diagnosis(world: fake.World, usb_bus: Path) -> None:
    world.add()
    _usb_device(usb_bus, "2-1", "8086", "0b07", "480", "D435")
    camera = _camera("realsense")
    try:
        info = camera.info()
        # librealsense says 3.2 in this fake; the negotiated 480 Mbit/s link wins.
        assert info["usb_link_mbps"] == 480.0 and info["usb2"] is True
    finally:
        camera.release()
    with pytest.raises(CameraUnavailableError) as caught:
        _camera("realsense:111122223333")
    assert "USB bus" not in caught.value.detail  # one on the bus, one seen: no udev problem


def test_start_failures_on_a_usb2_link_say_so(world: fake.World, usb_bus: Path) -> None:
    world.add()
    _usb_device(usb_bus, "2-1", "8086", "0b07", "480", "D435")
    world.start_errors = ["Couldn't resolve requests"] * 3
    with pytest.raises(CameraUnavailableError) as caught:
        open_realsense(parse_camera_entry("realsense+depth"))
    assert "USB 2 link (480 Mbit/s)" in caught.value.remediation
    assert "cannot stream the requested modes" in caught.value.remediation
