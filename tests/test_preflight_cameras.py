"""The preflight cameras check with RealSense facts, depth stats and the sim backend."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fakes import fake_pyrealsense2 as fake
from so101.bringup import preflight as pf
from so101.bringup.lerobot_surface import LeRobotSurface
from so101.bringup.ownership import ServerProbe
from so101.config import Settings, get_settings
from so101.hardware import lerobot_paths
from so101.hardware import realsense as rsmod
from so101.hardware.safety import DEFAULT_LIMITS


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    monkeypatch.setattr(rsmod, "USB_SYSFS", tmp_path / "no-usb-bus")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def _ctx(settings: Any, cameras: pf.CameraFacts, backend: str = "real") -> pf.PreflightContext:
    # check_cameras reads only the camera facts and the backend.
    unused: Any = None
    return pf.PreflightContext(
        settings=settings,
        origin="cli",
        backend=backend,  # type: ignore[arg-type]
        arms=("follower",),
        host=unused,
        lerobot=unused,
        ports={},
        serial_ports=unused,
        bus_access=pf.BusAccess("opened"),
        buses={},
        calibration={},
        rest_pose=None,
        cameras=cameras,
        disks=[],
        estop=pf.EstopFacts(known=False, source="none"),
    )


def _check(settings: Any, *probes: pf.CameraProbe, backend: str = "real") -> Any:
    facts = pf.CameraFacts(
        configured=[probe.camera for probe in probes], source="host", probes=list(probes)
    )
    return pf.check_cameras(_ctx(settings, facts, backend))


def _realsense(**device: Any) -> pf.CameraProbe:
    facts = {"usb2": False, "firmware_outdated": False, **device}
    streams = {"color": {"shape": [480, 640, 3]}}
    if "valid_fraction" in device:
        streams["depth"] = {"shape": [480, 640, 1], "valid_fraction": device["valid_fraction"]}
    return pf.CameraProbe(
        camera="realsense:1234+depth",
        ok=True,
        mean=90.0,
        shape=[480, 640, 3],
        kind="realsense",
        device=facts,
        streams=streams,
    )


def test_a_healthy_realsense_passes(settings: Settings) -> None:
    result = _check(settings, _realsense(valid_fraction=0.8, usb_type="3.2"))
    assert result.status == "pass" and result.message.startswith("opened and read 1 camera(s)")
    assert result.measured["depth_min_valid_fraction"] == pf.DEPTH_MIN_VALID_FRACTION


@pytest.mark.parametrize(
    ("device", "message", "remediation"),
    [
        (
            {"usb2": True, "usb_type": "2.1"},
            "realsense:1234+depth: USB 2 link (USB 2.1)",
            "USB 3 port",
        ),
        (
            {"usb2": True, "usb_type": "3.2", "usb_link_mbps": 480.0},
            "USB 2 link (480 Mbit/s)",
            "USB 3 cable",
        ),
        (
            {
                "firmware_outdated": True,
                "firmware": "5.12.7.100",
                "recommended_firmware": "5.16.0.1",
            },
            "firmware 5.12.7.100 is older than the recommended 5.16.0.1",
            "docs/realsense.md, Firmware",
        ),
        ({"valid_fraction": 0.04}, "only 4% of depth pixels are valid", "depth range"),
    ],
)
def test_realsense_warnings(
    settings: Settings, device: dict[str, Any], message: str, remediation: str
) -> None:
    result = _check(settings, _realsense(**device))
    assert result.status == "warn" and message in result.message
    assert remediation in result.remediation


def test_warnings_combine_with_dark_frames_and_keep_the_old_text_alone(
    settings: Settings,
) -> None:
    dark = pf.CameraProbe(camera="0", ok=True, mean=1.0, shape=[480, 640, 3], kind="opencv")
    only_dark = _check(settings, dark)
    assert only_dark.message == "black frames from 0"
    assert (
        only_dark.remediation == "Remove the lens cap, check lighting, or check the camera index."
    )
    both = _check(settings, dark, _realsense(valid_fraction=0.01))
    assert both.message.startswith("black frames from 0; realsense:1234+depth: only 1%")
    assert both.remediation.startswith("Remove the lens cap") and "depth range" in both.remediation


def test_failures_carry_the_realsense_remediation(settings: Settings) -> None:
    failed = pf.CameraProbe(
        camera="realsense",
        ok=False,
        error="CameraUnavailableError: no RealSense camera is attached",
        kind="realsense",
        remediation="Install the RealSense udev rules ...",
    )
    result = _check(settings, failed)
    assert result.status == "fail"
    assert (
        result.message
        == "camera realsense: CameraUnavailableError: no RealSense camera is attached"
    )
    assert result.remediation == "Install the RealSense udev rules ..."
    plain = pf.CameraProbe(camera="0", ok=False, error="CameraUnavailableError: no", kind="opencv")
    assert _check(settings, plain).remediation == (
        "Reconnect the camera or remove it from SO101_CAMERAS."
    )


def test_server_status_with_depth_streams_counts_cameras_by_color(settings: Settings) -> None:
    entry = "w=realsense:1234+depth"
    live = pf.cameras_from_status(
        [entry, "0"], {"live_cameras": [entry, "0", entry + "#depth"]}, source="server"
    )
    assert pf.check_cameras(_ctx(settings, live)).status == "pass"
    depth_only = pf.cameras_from_status(
        [entry], {"live_cameras": [entry + "#depth"]}, source="server"
    )
    skipped = pf.check_cameras(_ctx(settings, depth_only))
    assert skipped.status == "skip" and skipped.message.endswith(
        f"no live frame from {entry}; start a camera-using mode to check them"
    )


def test_the_sim_backend_skips_the_camera_check(settings: Settings) -> None:
    for facts in (
        pf.CameraFacts(configured=[], source="sim"),
        pf.CameraFacts(configured=["0"], source="host", probes=[pf.CameraProbe("0", ok=False)]),
    ):
        result = pf.check_cameras(_ctx(settings, facts, backend="sim"))
        assert result.status == "skip" and result.blocking is False
        assert result.message == pf.SIM_CAMERAS_SKIP and result.source == "sim"


# --- probing ---------------------------------------------------------------------------------


def test_mock_probe_reports_kind_device_and_depth_stats(settings: Settings) -> None:
    entry = "realsense:1234:320x240+depth"
    (probe, plain) = pf.probe_cameras_locally([entry, "0"], frames=1)
    assert probe.ok and probe.kind == "realsense" and probe.shape == [240, 320, 3]
    assert probe.device is not None and probe.device["synthetic"] is True
    depth = probe.streams["depth"] if probe.streams else {}
    assert depth["shape"] == [240, 320, 1] and 0.5 < depth["valid_fraction"] < 1.0
    assert 300 <= depth["min_mm"] <= depth["median_mm"] <= depth["max_mm"] <= 1500
    assert (
        plain.ok and plain.kind == "opencv" and plain.streams == {"color": {"shape": [480, 640, 3]}}
    )


def test_probe_failures_split_detail_and_remediation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.make_module(fake.World()))
    (probe,) = pf.probe_cameras_locally(["realsense"], frames=1)
    assert not probe.ok and probe.kind == "realsense"
    assert probe.error == (
        "CameraUnavailableError: real camera 'realsense' is unavailable: no RealSense camera is "
        "attached; no mock stream was substituted"
    )
    assert probe.remediation and "make find-cameras" in probe.remediation
    result = pf.check_cameras(
        _ctx(settings, pf.CameraFacts(configured=["realsense"], source="host", probes=[probe]))
    )
    assert result.status == "fail" and result.remediation == probe.remediation


def test_a_malformed_entry_is_a_probe_failure_not_a_crash(settings: Settings) -> None:
    (probe,) = pf.probe_cameras_locally(["realsense:12"], frames=1)
    assert not probe.ok and probe.kind is None and probe.error.startswith("ValueError:")


def test_a_realsense_depth_probe_through_the_fake(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    world = fake.World()
    world.add(usb_type="2.1")
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake.make_module(world))
    monkeypatch.setattr(rsmod, "WARMUP_S", 0.0)
    (probe,) = pf.probe_cameras_locally(["realsense:123622270882+depth"], frames=2)
    assert probe.ok and probe.device is not None and probe.device["usb2"] is True
    assert probe.streams is not None and probe.streams["depth"]["valid_fraction"] == 1.0
    assert probe.streams["depth"]["median_mm"] == 1000
    assert all(pipe.stopped for pipe in world.pipelines if pipe.started)
    result = pf.check_cameras(
        _ctx(settings, pf.CameraFacts(configured=[probe.camera], source="host", probes=[probe]))
    )
    assert result.status == "warn" and "USB 2 link (USB 2.1)" in result.message


# --- runners under SO101_BACKEND=sim -------------------------------------------------------------


def _no_server(settings: Any) -> ServerProbe:
    return ServerProbe(running=False, url="http://127.0.0.1:8816/api/robot/status", detail="none")


def test_cli_preflight_under_sim_never_probes_cameras(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("SO101_BACKEND", "sim")
    get_settings.cache_clear()
    sim = get_settings()

    def prober(ids: list[str]) -> list[pf.CameraProbe]:
        raise AssertionError("a real camera was probed under the sim backend")

    report = pf.run_cli_preflight(
        sim,
        server_probe=_no_server,
        camera_prober=prober,
        lerobot_prober=LeRobotSurface,
        sleep=lambda _: None,
        write_report=False,
    )
    by_key = {check.key: check for check in report.checks}
    assert by_key["cameras"].status == "skip" and by_key["cameras"].message == pf.SIM_CAMERAS_SKIP
    assert [check.id for check in report.checks] == [
        check.id
        for check in pf.run_cli_preflight(
            settings,
            server_probe=_no_server,
            camera_prober=lambda ids: [],
            lerobot_prober=LeRobotSurface,
            sleep=lambda _: None,
            write_report=False,
        ).checks
    ]


def test_api_preflight_under_sim_skips_cameras(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("SO101_BACKEND", "sim")
    get_settings.cache_clear()
    sim = get_settings()
    manager = SimpleNamespace(
        settings=sim,
        guard=SimpleNamespace(limits=dict(DEFAULT_LIMITS)),
        fault=None,
        status=lambda: {"mode": "idle", "estopped": False, "live_cameras": ["front"]},
    )
    report = pf.run_api_preflight(manager, lerobot_prober=LeRobotSurface, sleep=lambda _: None)
    cameras = {check.key: check for check in report.checks}["cameras"]
    assert cameras.status == "skip" and cameras.message == pf.SIM_CAMERAS_SKIP
