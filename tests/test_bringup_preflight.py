"""Preflight: every check's pass/warn/fail/skip, report rules, runners, rehearsal labels."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from so101 import JOINT_NAMES
from so101.bringup import preflight as pf
from so101.bringup.buses import (
    BusFacts,
    FakeFeetechBus,
    collect_bus_facts,
    synthetic_calibration_for,
)
from so101.bringup.lerobot_surface import LeRobotSurface
from so101.bringup.ownership import BusLock, ServerProbe, bus_lock_path
from so101.bringup.schemas import ARMS, CheckResult
from so101.bringup.serial_ports import PYSERIAL_REMEDIATION, PortListing, SerialPortInfo
from so101.config import Settings, get_settings
from so101.hardware import lerobot_paths
from so101.hardware import rest_pose as rp
from so101.hardware.calibration import CalibrationResult, JointCalibration
from so101.hardware.safety import DEFAULT_LIMITS
from so101.simulation.rest_pose import MODEL_REST_POSE

GIB = 1024**3
POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -95.0,
    "elbow_flex": 90.0,
    "wrist_flex": 70.0,
    "wrist_roll": 0.0,
    "gripper": 2.0,
}
SUITE_STATUS = {
    "mode": "idle",
    "estopped": False,
    "live_cameras": ["0"],
    "follower_connected": False,
    "leader_connected": False,
}


def no_server(settings: Any) -> ServerProbe:
    return ServerProbe(running=False, url="http://127.0.0.1:8816/api/robot/status", detail="none")


def suite_server(status: dict[str, Any] | None = None) -> Callable[[Any], ServerProbe]:
    def probe(settings: Any) -> ServerProbe:
        return ServerProbe(
            running=True,
            url="http://127.0.0.1:8816/api/robot/status",
            detail="HTTP 200",
            http_status=200,
            status=dict(SUITE_STATUS if status is None else status),
        )

    return probe


def ok_surface() -> LeRobotSurface:
    return LeRobotSurface(
        installed=True, version="0.6.1", importable=True, sdk_importable=True, tables_checked=True
    )


def no_sleep(_: float) -> None:
    return None


def bus_facts(arm: str = "follower", **fake: Any) -> BusFacts:
    bus = FakeFeetechBus(arm=arm, **fake)
    return collect_bus_facts(bus, arm=arm, port=f"/dev/fake-{arm}", source="mock", sleep=no_sleep)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    # Never touch the real LeRobot calibration directory.
    monkeypatch.setattr(
        lerobot_paths, "lerobot_calibration_root", lambda: (tmp_path / "calibration", "test")
    )
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def adapter(device: str, serial: str) -> SerialPortInfo:
    return SerialPortInfo(device=device, vid=0x1A86, pid=0x55D3, serial_number=serial)


#: What comports() lists in these tests: one USB adapter per arm, on Linux.
LISTING = PortListing(
    ports=[adapter("/dev/ttyACM0", "F0LLOWER"), adapter("/dev/ttyACM1", "LEADER01")],
    platform="linux",
)


def usable_port(arm: str, path: str, listing: PortListing | None = None) -> pf.PortFacts:
    """Port inspector for tests: every configured path is a usable serial device."""
    return pf.PortFacts(
        arm=arm,
        path=path,
        exists=True,
        listed=True,
        adapter=adapter(path, arm.upper()).to_dict(),
        char_device=True,
        readable=True,
        writable=True,
        realpath=path,
    )


@pytest.fixture(autouse=True)
def _fixed_port_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hermetic: never enumerate this machine's serial ports.
    monkeypatch.setattr(pf, "list_serial_ports", lambda: LISTING)


def real_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("SO101_MOCK", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


def calibration_facts(settings: Any, arm: str, **changes: Any) -> pf.CalibrationFacts:
    facts = pf.CalibrationFacts(
        arm=arm,
        device_id=pf.device_id_for(settings, arm),
        lerobot_path=f"/cal/{arm}.json",
        lerobot_exists=True,
        lerobot_payload=synthetic_calibration_for(arm),
        suite_path=f"/data/calibration_{arm}.json",
        suite_exists=True,
        suite_device_id=pf.device_id_for(settings, arm),
    )
    for key, value in changes.items():
        setattr(facts, key, value)
    return facts


def make_ctx(settings: Any, *, backend: str = "real", **overrides: Any) -> pf.PreflightContext:
    arms = overrides.pop("arms", ARMS)
    values: dict[str, Any] = {
        "settings": settings,
        "origin": "cli",
        "backend": backend,
        "arms": arms,
        "host": pf.HostFacts(
            python=(3, 12, 4),
            platform="test-os",
            so101_version="0.1.0",
            environ={},
            calibration_root="/cal",
            calibration_root_how="test",
        ),
        "lerobot": ok_surface(),
        "ports": {
            arm: pf.PortFacts(
                arm=arm,
                path=f"/dev/serial/by-id/usb-{arm}",
                exists=True,
                listed=True,
                adapter=LISTING.ports[index].to_dict(),
                char_device=True,
                readable=True,
                writable=True,
                realpath=f"/dev/ttyACM{index}",
            )
            for index, arm in enumerate(ARMS)
        },
        "serial_ports": LISTING,
        "bus_access": pf.BusAccess("opened" if backend == "real" else "synthetic", "test"),
        "buses": {arm: bus_facts(arm) for arm in arms},
        "calibration": {arm: calibration_facts(settings, arm) for arm in arms},
        "rest_pose": None,
        "cameras": pf.CameraFacts(configured=[], source="host"),
        "disks": [
            pf.DirectoryFacts(
                role="data_dir",
                path="/data",
                exists=True,
                writable=True,
                free_bytes=50 * GIB,
                total_bytes=100 * GIB,
            )
        ],
        "estop": pf.EstopFacts(known=False, source="none", detail=pf.NO_SERVER_ESTOP_DETAIL),
    }
    values.update(overrides)
    return pf.PreflightContext(**values)


def deferred() -> pf.BusAccess:
    return pf.BusAccess("deferred", pf.WAVE2_BUS_MESSAGE, "stop the server", blocking=True)


# --- environment ---------------------------------------------------------------------------------


def test_environment_passes_on_a_clean_host(settings: Settings) -> None:
    result = pf.check_environment(make_ctx(settings))
    assert result.status == "pass"
    assert result.key == "environment" and result.arm is None
    assert result.measured["baudrate"] == 1_000_000
    assert result.rehearsal is False


def test_environment_fails_on_old_python_even_in_a_rehearsal(settings: Settings) -> None:
    host = make_ctx(settings).host
    host.python = (3, 11, 9)
    result = pf.check_environment(make_ctx(settings, backend="mock", host=host))
    assert result.status == "fail"
    assert "3.11.9 is older than 3.12" in result.message
    assert result.rehearsal is True


def test_environment_fails_on_a_python_newer_than_313(settings: Settings) -> None:
    host = make_ctx(settings).host
    host.python = (3, 14, 0)
    result = pf.check_environment(make_ctx(settings, host=host))
    assert result.status == "fail"
    assert "3.14.0 is newer than 3.13" in result.message
    assert "requires-python >=3.12,<3.14" in result.remediation
    host.python = (3, 13, 7)
    assert pf.check_environment(make_ctx(settings, host=host)).status == "pass"


def test_environment_lerobot_home_fails_real_and_warns_in_rehearsal(settings: Settings) -> None:
    host = make_ctx(settings).host
    host.environ = {"LEROBOT_HOME": "/x"}
    real = pf.check_environment(make_ctx(settings, host=host))
    mock = pf.check_environment(make_ctx(settings, backend="mock", host=host))
    assert real.status == "fail" and "constants.py:61-65" in real.message
    assert mock.status == "warn" and "would fail on the real backend" in mock.message
    assert "HF_LEROBOT_HOME" in real.remediation


def test_environment_wrong_baudrate_fails_real(settings: Settings) -> None:
    slow = settings.model_copy(update={"baudrate": 115_200})
    result = pf.check_environment(make_ctx(slow))
    assert result.status == "fail"
    assert "1,000,000 baud" in result.message and "SO101_BAUDRATE" in result.remediation
    assert pf.check_environment(make_ctx(slow, backend="mock")).status == "warn"


# --- lerobot -------------------------------------------------------------------------------------


def test_lerobot_passes_with_the_pinned_surface(settings: Settings) -> None:
    result = pf.check_lerobot(make_ctx(settings))
    assert result.status == "pass"
    assert result.measured["pinned_version"] == "0.6.1"


def test_lerobot_missing_fails_real_and_warns_in_rehearsal(settings: Settings) -> None:
    surface = LeRobotSurface()
    real = pf.check_lerobot(make_ctx(settings, lerobot=surface))
    mock = pf.check_lerobot(make_ctx(settings, backend="mock", lerobot=surface))
    assert real.status == "fail" and "LeRobot is not installed" in real.message
    assert mock.status == "warn" and "synthetic bus" in mock.message
    # Double quotes work in cmd, PowerShell and POSIX shells; single quotes fail in cmd.
    assert 'pip install -e ".[robot]"' in real.remediation


def test_lerobot_version_and_table_drift_are_reported(settings: Settings) -> None:
    surface = ok_surface()
    surface.version = "0.7.0"
    surface.table_problems = ["register 'Torque_Enable' is (41, 1), pinned (40, 1)"]
    result = pf.check_lerobot(make_ctx(settings, lerobot=surface))
    assert result.status == "fail"
    assert "pins 0.6.1" in result.message and "Torque_Enable" in result.message


# --- serial ports --------------------------------------------------------------------------------


def test_serial_ports_skip_in_rehearsal_without_blocking(settings: Settings) -> None:
    result = pf.check_serial_ports(make_ctx(settings, backend="mock"))
    assert result.status == "skip" and result.blocking is False


def test_serial_ports_pass_with_stable_paths(settings: Settings) -> None:
    result = pf.check_serial_ports(make_ctx(settings))
    assert result.status == "pass"
    assert "serial F0LLOWER, VID:PID 1A86:55D3" in result.message
    assert result.measured["listed"][0]["vid_pid"] == "1A86:55D3"
    assert result.measured["platform"] == "linux"


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"exists": False}, "follower arm not found: no serial port"),
        ({"char_device": False}, "not a character device"),
        ({"error": "OSError: bad"}, "cannot be inspected"),
    ],
)
def test_serial_ports_fail_when_the_port_is_missing_or_wrong(
    settings: Settings, changes: dict[str, Any], expected: str
) -> None:
    ctx = make_ctx(settings)
    for key, value in changes.items():
        setattr(ctx.ports["follower"], key, value)
    result = pf.check_serial_ports(ctx)
    assert result.status == "fail" and expected in result.message
    assert "lerobot-find-port" in result.remediation
    assert "/dev/ttyACM0 (serial F0LLOWER, VID:PID 1A86:55D3)" in result.remediation


def test_serial_ports_report_permission_apart_from_not_found(settings: Settings) -> None:
    ctx = make_ctx(settings)
    ctx.ports["follower"].writable = False
    result = pf.check_serial_ports(ctx)
    assert result.status == "fail"
    assert "may not open it" in result.message and "not found" not in result.message
    assert "dialout" in result.remediation and "sudo usermod -aG dialout" in result.remediation
    assert "lerobot-find-port" not in result.remediation

    mac = make_ctx(settings, serial_ports=PortListing(platform="darwin"))
    mac.ports["follower"].readable = False
    assert "ls -l /dev/serial/by-id/usb-follower" in pf.check_serial_ports(mac).remediation


def test_serial_ports_fail_without_pyserial(settings: Settings) -> None:
    listing = PortListing(error="pyserial is not installed (No module named 'serial')")
    result = pf.check_serial_ports(make_ctx(settings, serial_ports=listing))
    assert result.status == "fail" and "cannot be listed" in result.message
    assert PYSERIAL_REMEDIATION in result.remediation

    # On Windows a COM port is only a listing entry, so an unlisted port says
    # nothing about the arm: no "arm not found" on top of the listing failure.
    windows = PortListing(error=listing.error, platform="win32")
    ctx = make_ctx(
        settings,
        serial_ports=windows,
        ports={arm: pf.collect_port_facts(arm, "COM3", windows) for arm in ("follower", "leader")},
    )
    failed = pf.check_serial_ports(ctx)
    assert failed.status == "fail" and "cannot be listed" in failed.message
    assert "not found" not in failed.message and failed.remediation == PYSERIAL_REMEDIATION


def test_serial_ports_on_windows_use_the_com_listing(settings: Settings) -> None:
    listing = PortListing(
        ports=[adapter("COM3", "F0LLOWER"), adapter("COM4", "LEADER01")], platform="win32"
    )
    ctx = make_ctx(
        settings,
        serial_ports=listing,
        ports={
            "follower": pf.collect_port_facts("follower", "com3", listing),
            "leader": pf.collect_port_facts("leader", "\\\\.\\COM4", listing),
        },
    )
    assert ctx.ports["follower"].exists and ctx.ports["follower"].char_device is None
    result = pf.check_serial_ports(ctx)
    assert result.status == "pass", result.message
    assert "follower com3 (serial F0LLOWER" in result.message

    missing = pf.collect_port_facts("follower", "COM7", listing)
    assert missing.exists is False and missing.listed is False
    ctx.ports["follower"] = missing
    failed = pf.check_serial_ports(ctx)
    assert failed.status == "fail" and "follower arm not found" in failed.message
    assert "COM3" in failed.remediation and "Device Manager" in failed.remediation


def test_serial_ports_fail_when_both_arms_share_a_device(settings: Settings) -> None:
    ctx = make_ctx(settings)
    ctx.ports["leader"].realpath = ctx.ports["follower"].realpath
    result = pf.check_serial_ports(ctx)
    assert result.status == "fail" and "both resolve to" in result.message


def test_serial_ports_warn_when_adapters_share_a_serial_number(settings: Settings) -> None:
    ctx = make_ctx(settings)
    assert ctx.ports["leader"].adapter is not None
    ctx.ports["leader"].adapter["serial_number"] = "F0LLOWER"
    result = pf.check_serial_ports(ctx)
    assert result.status == "warn" and "USB serial number F0LLOWER" in result.message


def test_serial_ports_warn_on_unlisted_and_renumberable_ports(settings: Settings) -> None:
    ctx = make_ctx(settings)
    ctx.ports["follower"].listed = False
    ctx.ports["leader"].path = "/dev/ttyUSB0"
    result = pf.check_serial_ports(ctx)
    assert result.status == "warn"
    assert "pyserial does not list it" in result.message
    assert "/dev/serial/by-id/" in result.message
    assert (
        "prefer /dev/so101-leader (install/install.sh --follower-serial/--leader-serial) or its "
        "/dev/serial/by-id/ link" in result.message
    )
    mac = make_ctx(settings, serial_ports=PortListing(platform="darwin"))
    mac.ports["leader"].path = "/dev/ttyUSB0"
    mac.ports["follower"].listed = mac.ports["leader"].listed = True
    assert pf.check_serial_ports(mac).status == "pass"  # the Linux rule stays on Linux


def test_collect_port_facts_on_posix_paths(tmp_path: Path) -> None:
    listing = PortListing(platform="linux")
    missing = pf.collect_port_facts("follower", str(tmp_path / "nope"), listing)
    assert missing.exists is False and missing.error is None and missing.readable is None
    regular = tmp_path / "file"
    regular.write_text("x", encoding="utf-8")
    facts = pf.collect_port_facts("follower", str(regular), listing)
    assert facts.exists and facts.char_device is False and facts.readable
    assert facts.listed is False and facts.adapter is None
    assert pf.collect_port_facts("follower", str(regular)).exists  # no listing at all


def test_a_by_id_link_matches_its_listed_tty(tmp_path: Path) -> None:
    tty = tmp_path / "ttyACM0"
    tty.write_text("", encoding="utf-8")
    link = tmp_path / "usb-Feetech-if00"
    try:
        link.symlink_to(tty)
    except OSError:  # Windows without symlink rights
        pytest.skip("cannot create symlinks here")
    # This machine's own listing, so the link is resolved (as on a Linux host).
    listing = PortListing(ports=[adapter(str(tty), "F0LLOWER")], platform=sys.platform)
    facts = pf.collect_port_facts("follower", str(link), listing)
    assert facts.listed and facts.adapter is not None
    assert facts.adapter["serial_number"] == "F0LLOWER"


# --- motors --------------------------------------------------------------------------------------


def test_motors_pass(settings: Settings) -> None:
    result = pf.check_motors(make_ctx(settings), "follower")
    assert result.status == "pass"
    assert result.key == "motors.follower" and result.title == "Motors (follower)"
    assert "model 777" in result.message


def test_motors_fail_when_nothing_answers(settings: Settings) -> None:
    buses = {"follower": bus_facts(missing_ids=range(1, 7)), "leader": bus_facts("leader")}
    result = pf.check_motors(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail"
    assert "no motor answered" in result.message and "lerobot-setup-motors" in result.remediation


def test_motors_fail_names_the_missing_ids(settings: Settings) -> None:
    buses = {"follower": bus_facts(missing_ids=[3]), "leader": bus_facts("leader")}
    result = pf.check_motors(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail" and "elbow_flex (ID 3)" in result.message


def test_motors_fail_on_wrong_model_number(settings: Settings) -> None:
    buses = {"follower": bus_facts(model_numbers={"gripper": 2825}), "leader": bus_facts("leader")}
    result = pf.check_motors(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail" and "gripper (ID 6) reports 2825" in result.message


def test_motors_warn_on_status_bits(settings: Settings) -> None:
    buses = {
        "follower": bus_facts(registers={"Status": {"wrist_flex": 0x20}}),
        "leader": bus_facts("leader"),
    }
    result = pf.check_motors(make_ctx(settings, buses=buses), "follower")
    assert result.status == "warn"
    assert "0x20" in result.message and "unverified" in result.message
    assert result.measured["status_flags_verified"] is False


def test_motors_fail_when_the_port_will_not_open(settings: Settings) -> None:
    broken = BusFacts(
        arm="follower",
        port="/dev/x",
        source="real",
        attempted=True,
        opened=False,
        open_error="ConnectionError: could not open port",
    )
    buses = {"follower": broken, "leader": bus_facts("leader")}
    result = pf.check_motors(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail" and "could not open the follower bus" in result.message


def test_bus_checks_skip_with_the_wave2_message_when_deferred(settings: Settings) -> None:
    ctx = make_ctx(settings, bus_access=deferred(), buses={})
    for check in (
        pf.check_motors,
        pf.check_firmware,
        pf.check_supply_voltage,
        pf.check_temperature,
        pf.check_torque_off_stability,
    ):
        result = check(ctx, "leader")
        assert result.status == "skip", check
        assert result.message == pf.WAVE2_BUS_MESSAGE
        assert result.blocking is True
        assert result.source == "none"


def test_bus_checks_skip_after_an_open_failure_or_silence(settings: Settings) -> None:
    broken = BusFacts(arm="follower", port="/dev/x", source="real", attempted=True, opened=False)
    silent = bus_facts("leader", missing_ids=range(1, 7))
    ctx = make_ctx(settings, buses={"follower": broken, "leader": silent})
    assert "see motors.follower" in pf.check_firmware(ctx, "follower").message
    assert "no leader motor answered" in pf.check_firmware(ctx, "leader").message
    assert pf.check_temperature(ctx, "leader").status == "skip"


# --- firmware ------------------------------------------------------------------------------------


def test_firmware_pass_notes_no_minimum_is_verified(settings: Settings) -> None:
    result = pf.check_firmware(make_ctx(settings), "follower")
    assert result.status == "pass"
    assert "no minimum version is verified" in result.message
    assert result.measured["minimum_version"] is None


def test_firmware_fail_on_mixed_versions(settings: Settings) -> None:
    buses = {"follower": bus_facts(firmware={"gripper": (3, 10)}), "leader": bus_facts("leader")}
    result = pf.check_firmware(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail"
    assert "gripper 3.10" in result.message and "feetech.py:145-153" in result.message


def test_firmware_fail_when_unreadable(settings: Settings) -> None:
    buses = {
        "follower": bus_facts(register_errors={"Firmware_Major_Version": ["gripper"]}),
        "leader": bus_facts("leader"),
    }
    result = pf.check_firmware(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail" and "gripper (ID 6)" in result.message


def test_firmware_warn_when_some_motors_are_missing(settings: Settings) -> None:
    buses = {"follower": bus_facts(missing_ids=[6]), "leader": bus_facts("leader")}
    result = pf.check_firmware(make_ctx(settings, buses=buses), "follower")
    assert result.status == "warn" and "5 of 6 motors" in result.message


# --- supply voltage ------------------------------------------------------------------------------


def _with_voltage(raw: int, **extra: Any) -> dict[str, BusFacts]:
    registers = {"Present_Voltage": dict.fromkeys(JOINT_NAMES, raw)}
    registers.update(extra.pop("registers", {}))
    return {"follower": bus_facts(registers=registers, **extra), "leader": bus_facts("leader")}


def test_supply_voltage_pass(settings: Settings) -> None:
    result = pf.check_supply_voltage(make_ctx(settings), "follower")
    assert result.status == "pass"
    assert result.measured["profile"] == "7.4v"
    assert result.measured["scale"] == "0.1 V/LSB (unverified)"
    assert "supply 5 V on 6 motors" in result.message


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (40, "fail"),  # 4.0 V == critical_low (inclusive)
        (41, "warn"),
        (45, "warn"),  # 4.5 V == warn_low (inclusive)
        (46, "pass"),
        (79, "pass"),
        (80, "warn"),  # 8.0 V == warn_high
        (85, "fail"),  # 8.5 V == critical_high
    ],
)
def test_supply_voltage_band_boundaries(settings: Settings, raw: int, status: str) -> None:
    # Servo limits are widened so only the supply band decides.
    limits = {
        "Min_Voltage_Limit": dict.fromkeys(JOINT_NAMES, 30),
        "Max_Voltage_Limit": dict.fromkeys(JOINT_NAMES, 90),
    }
    buses = _with_voltage(raw, registers=limits)
    assert pf.check_supply_voltage(make_ctx(settings, buses=buses), "follower").status == status


def test_supply_voltage_12v_servos_on_a_7v4_profile_fail(settings: Settings) -> None:
    buses = {"follower": bus_facts(supply_profile="12v"), "leader": bus_facts("leader")}
    result = pf.check_supply_voltage(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail"
    assert "12 V" in result.message and "SO101_HEALTH_FOLLOWER_SUPPLY" in result.remediation


def test_supply_voltage_12v_profile_passes_12v_servos(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_HEALTH_FOLLOWER_SUPPLY", "12v")
    get_settings.cache_clear()
    twelve = get_settings()
    buses = {"follower": bus_facts(supply_profile="12v"), "leader": bus_facts("leader")}
    result = pf.check_supply_voltage(make_ctx(twelve, buses=buses), "follower")
    assert result.status == "pass" and result.measured["profile"] == "12v"


def test_supply_voltage_warns_outside_the_servo_limits(settings: Settings) -> None:
    buses = _with_voltage(50, registers={"Max_Voltage_Limit": {"shoulder_pan": 48}})
    result = pf.check_supply_voltage(make_ctx(settings, buses=buses), "follower")
    assert result.status == "warn" and "Min/Max_Voltage_Limit" in result.message


def test_supply_voltage_read_error_fails(settings: Settings) -> None:
    buses = {
        "follower": bus_facts(register_errors={"Present_Voltage": ["gripper"]}),
        "leader": bus_facts("leader"),
    }
    result = pf.check_supply_voltage(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail" and "could not read Present_Voltage" in result.message


# --- temperature ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("celsius", "status"),
    [(30, "pass"), (54, "pass"), (55, "warn"), (64, "warn"), (65, "fail")],
)
def test_temperature_boundaries_follow_the_servo_limit(
    settings: Settings, celsius: int, status: str
) -> None:
    # Max_Temperature_Limit 70 -> warn at 55, critical at 65 (suite margins 15/5).
    buses = {
        "follower": bus_facts(registers={"Present_Temperature": {"elbow_flex": celsius}}),
        "leader": bus_facts("leader"),
    }
    result = pf.check_temperature(make_ctx(settings, buses=buses), "follower")
    assert result.status == status
    assert result.measured["motors"]["elbow_flex"]["warn_c"] == 55.0


def test_temperature_read_error_fails(settings: Settings) -> None:
    buses = {
        "follower": bus_facts(register_errors={"Present_Temperature": ["wrist_roll"]}),
        "leader": bus_facts("leader"),
    }
    result = pf.check_temperature(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail"


# --- calibration ---------------------------------------------------------------------------------


def test_calibration_pass_when_files_match_registers(settings: Settings) -> None:
    result = pf.check_calibration(make_ctx(settings), "follower")
    assert result.status == "pass"
    assert result.measured["identity"]["mismatches"] == []
    assert len(result.measured["identity"]["compared"]) == 6


def test_calibration_missing_file_fails_and_mentions_the_legacy_location(
    settings: Settings,
) -> None:
    ctx = make_ctx(settings)
    ctx.calibration["follower"] = calibration_facts(
        settings,
        "follower",
        lerobot_exists=False,
        lerobot_payload=None,
        legacy_exists=True,
        legacy_path="/cal/robots/so101_follower/x.json",
    )
    result = pf.check_calibration(ctx, "follower")
    assert result.status == "fail"
    assert "no LeRobot calibration" in result.message and "pre-0.6 location" in result.message
    assert "so101 calibrate follower" in result.remediation


def test_calibration_unreadable_file_fails(settings: Settings) -> None:
    ctx = make_ctx(settings)
    ctx.calibration["leader"] = calibration_facts(
        settings, "leader", lerobot_payload=None, lerobot_error="ValueError: bad json"
    )
    assert pf.check_calibration(ctx, "leader").status == "fail"


@pytest.mark.parametrize(
    ("joint", "entry", "status", "text"),
    [
        ("shoulder_pan", {"id": 9}, "fail", "id 9, expected 1"),
        ("elbow_flex", {"range_min": 3000, "range_max": 2000}, "fail", "empty or outside"),
        ("elbow_flex", {"range_min": 0, "range_max": 5000}, "fail", "empty or outside"),
        ("wrist_flex", {"range_min": 2000, "range_max": 2099}, "fail", "100-tick minimum"),
        ("wrist_flex", {"range_min": 2000, "range_max": 2100}, "warn", "small"),
        ("wrist_flex", {"range_min": 2000, "range_max": 2400}, "pass", ""),
        ("wrist_roll", {"range_min": 100, "range_max": 4000}, "warn", "so_follower.py:142-143"),
    ],
)
def test_calibration_structural_rules(
    settings: Settings, joint: str, entry: dict[str, int], status: str, text: str
) -> None:
    payload = synthetic_calibration_for("follower")
    payload[joint].update(entry)
    registers = {
        "Min_Position_Limit": {joint: payload[joint]["range_min"]},
        "Max_Position_Limit": {joint: payload[joint]["range_max"]},
    }
    ctx = make_ctx(
        settings,
        buses={"follower": bus_facts(registers=registers), "leader": bus_facts("leader")},
    )
    ctx.calibration["follower"] = calibration_facts(settings, "follower", lerobot_payload=payload)
    result = pf.check_calibration(ctx, "follower")
    assert result.status == status, result.message
    assert text in result.message


def test_calibration_fails_on_wrong_names(settings: Settings) -> None:
    payload = synthetic_calibration_for("follower")
    payload["extra_joint"] = dict(payload["gripper"])
    ctx = make_ctx(settings)
    ctx.calibration["follower"] = calibration_facts(settings, "follower", lerobot_payload=payload)
    assert "motor names" in pf.check_calibration(ctx, "follower").message


def test_calibration_fails_when_registers_differ_from_the_file(settings: Settings) -> None:
    buses = {
        "follower": bus_facts(registers={"Homing_Offset": {"elbow_flex": 12}}),
        "leader": bus_facts("leader"),
    }
    result = pf.check_calibration(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail"
    assert "elbow_flex.homing_offset: motor 12, file 0" in result.message
    assert "SO101_FOLLOWER_ID" in result.remediation


def test_calibration_fails_on_another_devices_suite_file(settings: Settings) -> None:
    ctx = make_ctx(settings)
    ctx.calibration["follower"] = calibration_facts(
        settings, "follower", suite_device_id="someone-else"
    )
    result = pf.check_calibration(ctx, "follower")
    assert result.status == "fail" and "belongs to someone-else" in result.message


def test_calibration_warns_without_a_suite_file_on_real(settings: Settings) -> None:
    ctx = make_ctx(settings)
    ctx.calibration["follower"] = calibration_facts(settings, "follower", suite_exists=False)
    result = pf.check_calibration(ctx, "follower")
    assert result.status == "warn" and "SafetyGuard" in result.message


def test_calibration_skips_register_comparison_when_deferred(settings: Settings) -> None:
    result = pf.check_calibration(make_ctx(settings, bus_access=deferred(), buses={}), "follower")
    assert result.status == "skip" and result.blocking is True
    assert pf.WAVE2_BUS_MESSAGE in result.message
    assert result.source == "host"


def test_calibration_warns_when_a_joint_rests_outside_its_range(settings: Settings) -> None:
    ctx = make_ctx(settings)
    ctx.buses["follower"].position_samples[-1]["shoulder_pan"] = 500  # range 1024..3072
    result = pf.check_calibration(ctx, "follower")
    assert result.status == "warn" and "shoulder_pan at 500" in result.message


def test_calibration_synthetic_payload_passes_as_a_rehearsal(settings: Settings) -> None:
    ctx = make_ctx(settings, backend="mock")
    ctx.calibration["follower"] = calibration_facts(
        settings, "follower", synthetic=True, suite_exists=False
    )
    result = pf.check_calibration(ctx, "follower")
    assert result.status == "pass" and result.rehearsal is True
    assert result.message.startswith("synthetic calibration")
    assert result.source == "mock"


def test_collect_calibration_facts_reads_both_files(settings: Settings) -> None:
    path = lerobot_paths.lerobot_calibration_file("follower", settings.follower_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(synthetic_calibration_for("follower")), encoding="utf-8")
    CalibrationResult(
        arm="follower",
        joints={name: JointCalibration(min=-90, max=90, offset=0) for name in JOINT_NAMES},
        device_id=settings.follower_id,
        backend="lerobot",
    ).save(Path(settings.data_dir) / "calibration_follower.json")
    facts = pf.collect_calibration_facts(settings, "follower")
    assert facts.lerobot_payload == synthetic_calibration_for("follower")
    assert facts.lerobot_sha256 == lerobot_paths.file_sha256(path)
    assert facts.suite_device_id == settings.follower_id
    path.write_text("[1, 2]", encoding="utf-8")
    assert "per-motor" in (pf.collect_calibration_facts(settings, "follower").lerobot_error or "")


# --- rest pose -----------------------------------------------------------------------------------


def _write_follower_calibration(settings: Settings) -> str:
    path = lerobot_paths.lerobot_calibration_file("follower", settings.follower_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(synthetic_calibration_for("follower")), encoding="utf-8")
    return lerobot_paths.file_sha256(path)


def _rest_ctx(settings: Any, backend: str = "real", limits: Any = None) -> pf.PreflightContext:
    facts = pf.collect_rest_pose_facts(
        settings, backend, limits=limits or dict(DEFAULT_LIMITS), limits_source="test"
    )
    return make_ctx(settings, backend=backend, rest_pose=facts)


def test_real_rest_pose_is_incomplete_when_none_is_recorded(settings: Settings) -> None:
    result = pf.check_rest_pose(_rest_ctx(settings), "follower")
    assert result.status == "skip" and result.blocking is True
    assert "no rest pose has been recorded" in result.message
    assert "model-derived default pose" in result.message and "placeholder" not in result.message
    assert result.measured["joints"] == MODEL_REST_POSE
    assert "MJCF" in result.measured["limits_source"]
    assert "so101 rest-pose record" in result.remediation
    assert result.key == "rest_pose.follower"


def test_rest_pose_mentions_the_setting_fallback(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SO101_REST_POSE", "0,-95,90,70,0,2")
    get_settings.cache_clear()
    result = pf.check_rest_pose(_rest_ctx(get_settings()), "follower")
    assert result.status == "skip" and result.blocking and "SO101_REST_POSE" in result.message


def test_rest_pose_passes_when_recorded_for_this_calibration(settings: Settings) -> None:
    digest = _write_follower_calibration(settings)
    rp.save_rest_pose(
        settings, POSE, follower_id=settings.follower_id, calibration_digest=digest, backend="real"
    )
    result = pf.check_rest_pose(_rest_ctx(settings), "follower")
    assert result.status == "pass", result.message
    assert result.measured["source"] == "recorded"


def test_real_rest_pose_is_incomplete_when_the_calibration_changed(settings: Settings) -> None:
    _write_follower_calibration(settings)
    rp.save_rest_pose(
        settings,
        POSE,
        follower_id=settings.follower_id,
        calibration_digest="0" * 64,
        backend="real",
    )
    result = pf.check_rest_pose(_rest_ctx(settings), "follower")
    assert result.status == "skip" and result.blocking
    assert "calibration changed" in result.message


def test_real_rest_pose_is_incomplete_when_a_recorded_pose_was_clamped(settings: Settings) -> None:
    digest = _write_follower_calibration(settings)
    rp.save_rest_pose(
        settings, POSE, follower_id=settings.follower_id, calibration_digest=digest, backend="real"
    )
    limits = dict(DEFAULT_LIMITS)
    limits["shoulder_lift"] = (-60.0, 60.0)
    result = pf.check_rest_pose(_rest_ctx(settings, limits=limits), "follower")
    assert result.status == "skip" and result.blocking and "clamped" in result.message


def test_rest_pose_reports_a_corrupt_file(settings: Settings) -> None:
    rp.rest_pose_path(settings).parent.mkdir(parents=True, exist_ok=True)
    rp.rest_pose_path(settings).write_text("{not json", encoding="utf-8")
    result = pf.check_rest_pose(_rest_ctx(settings), "follower")
    assert result.status == "skip" and result.blocking
    assert "ignored the recorded rest pose" in result.message


def test_rest_pose_skips_nonblocking_on_sim_and_blocking_on_real_errors(
    settings: Settings,
) -> None:
    sim = pf.check_rest_pose(make_ctx(settings, backend="sim"), "follower")
    assert sim.status == "skip" and sim.blocking is False
    broken = make_ctx(settings, rest_pose=pf.RestPoseFacts(error="RuntimeError: x"))
    assert pf.check_rest_pose(broken, "follower").status == "skip"


def test_rest_pose_mock_follows_the_same_rule(settings: Settings) -> None:
    result = pf.check_rest_pose(_rest_ctx(settings, backend="mock"), "follower")
    assert result.status == "warn" and result.rehearsal is True


# --- torque off and steady -----------------------------------------------------------------------


def test_torque_off_stability_pass(settings: Settings) -> None:
    result = pf.check_torque_off_stability(make_ctx(settings), "follower")
    assert result.status == "pass"
    assert result.measured["samples"] == 10
    assert result.measured["degrees_per_tick"] == round(360 / 4095, 6)


def test_torque_on_fails_with_drop_warning(settings: Settings) -> None:
    buses = {
        "follower": bus_facts(registers={"Torque_Enable": {"gripper": 1}}),
        "leader": bus_facts("leader"),
    }
    result = pf.check_torque_off_stability(make_ctx(settings, buses=buses), "follower")
    assert result.status == "fail"
    assert "torque is ON for gripper (ID 6)" in result.message
    assert "Support the arm" in result.remediation


@pytest.mark.parametrize(
    ("fake", "status", "text"),
    [
        ({"register_errors": {"Torque_Enable": ["gripper"]}}, "fail", "Torque_Enable"),
        ({"position_read_errors": 1}, "warn", "1 read retried"),
        ({"position_read_errors": 2}, "fail", "2 of 10 position reads failed"),
        ({"position_read_errors": 10}, "fail", "10 of 10"),
        ({"position_noise": {"elbow_flex": [0, 3]}}, "pass", "max spread 3 ticks"),
        ({"position_noise": {"elbow_flex": [0, 4]}}, "warn", "elbow_flex 4 ticks"),
        ({"position_noise": {"gripper": [5000]}}, "fail", "outside 0..4095"),
    ],
)
def test_torque_off_stability_rules(
    settings: Settings, fake: dict[str, Any], status: str, text: str
) -> None:
    buses = {"follower": bus_facts(**fake), "leader": bus_facts("leader")}
    result = pf.check_torque_off_stability(make_ctx(settings, buses=buses), "follower")
    assert result.status == status, result.message
    assert text in result.message


# --- cameras -------------------------------------------------------------------------------------


def test_cameras_skip_when_none_are_configured(settings: Settings) -> None:
    result = pf.check_cameras(make_ctx(settings))
    assert result.status == "skip" and result.blocking is False


@pytest.mark.parametrize(
    ("probe", "status"),
    [
        (pf.CameraProbe(camera="0", ok=True, mean=90.0, shape=[480, 640, 3]), "pass"),
        (pf.CameraProbe(camera="0", ok=True, mean=1.0, shape=[480, 640, 3]), "warn"),
        (pf.CameraProbe(camera="0", ok=False, error="CameraUnavailableError: no"), "fail"),
    ],
)
def test_cameras_local_probe(settings: Settings, probe: pf.CameraProbe, status: str) -> None:
    cameras = pf.CameraFacts(configured=["0"], source="host", probes=[probe])
    assert pf.check_cameras(make_ctx(settings, cameras=cameras)).status == status


def test_cameras_from_server_status(settings: Settings) -> None:
    live = pf.cameras_from_status(["0"], {"live_cameras": ["0"]}, source="server")
    idle = pf.cameras_from_status(["0", "1"], {"live_cameras": ["0"]}, source="server")
    unknown = pf.cameras_from_status(["0"], None, source="server")
    assert pf.check_cameras(make_ctx(settings, cameras=live)).status == "pass"
    skipped = pf.check_cameras(make_ctx(settings, cameras=idle))
    assert skipped.status == "skip" and skipped.blocking is False and "1" in skipped.message
    assert pf.check_cameras(make_ctx(settings, cameras=unknown)).status == "skip"


def test_probe_cameras_locally_in_mock_mode(settings: Settings) -> None:
    probes = pf.probe_cameras_locally(["0"], frames=2)
    assert probes[0].ok and probes[0].shape is not None and probes[0].shape[-1] == 3


# --- disk space ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("free", "writable", "status"),
    [
        (50 * GIB, True, "pass"),
        (10 * GIB, True, "pass"),
        (10 * GIB - 1, True, "warn"),
        (2 * GIB, True, "warn"),
        (2 * GIB - 1, True, "fail"),
        (50 * GIB, False, "fail"),
    ],
)
def test_disk_space_thresholds(settings: Settings, free: int, writable: bool, status: str) -> None:
    disks = [
        pf.DirectoryFacts(
            role="data_dir",
            path="/d",
            exists=True,
            writable=writable,
            free_bytes=free,
            total_bytes=100 * GIB,
        )
    ]
    assert pf.check_disk_space(make_ctx(settings, disks=disks)).status == status


def test_disk_space_error_fails_and_real_dirs_are_measured(settings: Settings) -> None:
    broken = [pf.DirectoryFacts("data_dir", "/d", False, False, None, None, error="OSError: x")]
    assert pf.check_disk_space(make_ctx(settings, disks=broken)).status == "fail"
    facts = pf.collect_disk_facts(settings)
    assert [fact.role for fact in facts] == ["data_dir", "datasets_dir", "outputs_dir"]
    assert all(fact.free_bytes is not None and fact.writable for fact in facts)


# --- E-stop --------------------------------------------------------------------------------------


def test_estop_skips_without_a_server(settings: Settings) -> None:
    result = pf.check_estop(make_ctx(settings))
    assert result.status == "skip" and result.blocking is False
    assert "power switch is the only immediate stop" in result.message
    assert "persisted E-stop, fault and unconfirmed-torque latches" in result.message
    assert "restored by a server restart" in result.message
    assert "restart clears" not in result.message


def test_estop_states(settings: Settings) -> None:
    latched = pf.EstopFacts(known=True, source="server", estopped=True)
    fault = pf.EstopFacts(known=True, source="server", estopped=False, fault="x", fault_known=True)
    clear_api = pf.EstopFacts(known=True, source="server", estopped=False, fault_known=True)
    clear_cli = pf.EstopFacts(known=True, source="server", estopped=False)
    body = '{"operator_id": "<your name>", "confirm": true}'  # the C2 reset body
    result = pf.check_estop(make_ctx(settings, estop=latched))
    assert result.status == "warn" and "/api/robot/estop/reset" in result.remediation
    assert body in result.remediation
    result = pf.check_estop(make_ctx(settings, estop=fault))
    assert result.status == "warn" and "/api/robot/fault/reset" in result.remediation
    assert body in result.remediation and '"power_cut_acknowledged": true' in result.remediation
    assert "no runtime fault" in pf.check_estop(make_ctx(settings, estop=clear_api)).message
    assert "not part of" in pf.check_estop(make_ctx(settings, estop=clear_cli)).message


def test_estop_from_server_probe() -> None:
    assert pf.estop_from_server(no_server(None)).known is False
    other = ServerProbe(running=True, url="u", detail="HTTP 404", http_status=404)
    assert pf.estop_from_server(other).known is False
    facts = pf.estop_from_server(suite_server({"mode": "idle", "estopped": True})(None))
    assert facts.known and facts.estopped and facts.mode == "idle"


# --- registry, report and exit codes -------------------------------------------------------------


def test_registry_order_and_keys(settings: Settings) -> None:
    keys = [result.key for result in pf.run_checks(make_ctx(settings))]
    assert keys == [
        "environment",
        "lerobot",
        "serial_ports",
        "motors.follower",
        "motors.leader",
        "firmware.follower",
        "firmware.leader",
        "supply_voltage.follower",
        "supply_voltage.leader",
        "temperature.follower",
        "temperature.leader",
        "calibration.follower",
        "calibration.leader",
        "rest_pose.follower",
        "torque_off_stability.follower",
        "torque_off_stability.leader",
        "cameras",
        "disk_space",
        "estop",
    ]
    catalog = pf.check_catalog()
    assert [info.id for info in catalog] == [spec.id for spec in pf.CHECKS]
    assert len(catalog) == 13
    assert {info.id for info in catalog if info.needs_bus} == {
        "motors",
        "firmware",
        "supply_voltage",
        "temperature",
        "calibration",
        "torque_off_stability",
    }


def test_leader_only_run_has_no_rest_pose_check(settings: Settings) -> None:
    keys = [result.key for result in pf.run_checks(make_ctx(settings, arms=("leader",)))]
    assert "rest_pose.follower" not in keys and "motors.leader" in keys


def test_a_crashing_check_becomes_a_fail(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ctx: Any, arm: Any) -> CheckResult:
        raise RuntimeError("kaput")

    spec = pf.CHECKS_BY_ID["disk_space"]
    result = pf._run_one(pf.CheckSpec(spec.id, spec.title, "", boom), make_ctx(settings), None)
    assert result.status == "fail" and "the check crashed: RuntimeError: kaput" in result.message


def _r(
    status: str,
    blocking: bool = True,
    *,
    check_id: str = "x",
    key: str | None = None,
) -> CheckResult:
    return CheckResult(
        id=check_id,
        key=key or check_id,
        title=check_id,
        status=status,
        message="",
        blocking=blocking,
        rehearsal=False,
    )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("statuses", "overall"),
    [
        ([_r("pass")], "pass"),
        ([_r("pass"), _r("skip", blocking=False)], "pass"),
        ([_r("pass"), _r("warn")], "warn"),
        ([_r("warn"), _r("skip")], "incomplete"),
        ([_r("skip"), _r("fail")], "fail"),
    ],
)
def test_overall_status(statuses: list[CheckResult], overall: str) -> None:
    assert pf.overall_status(statuses) == overall


@pytest.mark.parametrize(
    ("overall", "rehearsal", "require", "code"),
    [
        ("pass", False, False, 0),
        ("warn", False, True, 0),
        ("fail", False, False, 1),
        ("incomplete", False, False, 3),
        ("pass", True, False, 0),
        ("pass", True, True, 4),
        ("fail", True, True, 4),
    ],
)
def test_exit_codes(overall: Any, rehearsal: bool, require: bool, code: int) -> None:
    assert pf.exit_code_for(overall, rehearsal=rehearsal, require_hardware=require) == code


def test_report_labels_and_readiness(settings: Settings) -> None:
    real_ctx = make_ctx(settings)
    real = pf.build_report(
        real_ctx,
        [
            _r("pass", check_id="rest_pose", key="rest_pose.follower"),
            _r("warn"),
        ],
    )
    assert real.overall == "warn" and real.ready_for_hardware and not real.rehearsal
    assert real.counts == {"pass": 1, "warn": 1, "fail": 0, "skip": 0}

    no_validated_pose = pf.build_report(real_ctx, [_r("pass"), _r("warn")])
    assert no_validated_pose.overall == "incomplete"
    assert no_validated_pose.ready_for_hardware is False
    assert no_validated_pose.exit_code == pf.EXIT_INCOMPLETE

    mock = pf.build_report(make_ctx(settings, backend="mock"), [_r("pass")])
    assert mock.rehearsal and not mock.ready_for_hardware
    assert mock.notes[0].startswith("REHEARSAL")


# --- runners -------------------------------------------------------------------------------------


def test_cli_rehearsal_labels_everything_and_never_writes(settings: Settings) -> None:
    fakes: list[FakeFeetechBus] = []

    def factory(arm: str) -> FakeFeetechBus:
        fakes.append(FakeFeetechBus(arm=arm))
        return fakes[-1]

    report = pf.run_cli_preflight(
        settings,
        server_probe=no_server,
        fake_bus_factory=factory,
        lerobot_prober=LeRobotSurface,
        sleep=no_sleep,
    )
    assert report.rehearsal is True and report.backend == "mock"
    assert all(check.rehearsal for check in report.checks)
    assert report.bus_access == "synthetic" and not report.ready_for_hardware
    assert report.overall == "warn" and report.exit_code == 0  # lerobot + rest pose warn
    assert [fake.arm for fake in fakes] == ["follower", "leader"]
    for fake in fakes:
        assert fake.calls[-1] == ("disconnect", False)
        assert {name for name, _ in fake.calls} <= {"ping", "read", "sync_read", "disconnect"}
    latest = pf.load_latest_report(settings)
    assert latest is not None and latest.created_at == report.created_at
    (saved,) = pf.reports_dir(settings).glob("*-cli-mock.json")
    assert not set('<>:"|?*') & set(saved.name)  # a legal Windows file name


def test_cli_require_hardware_rejects_a_rehearsal(settings: Settings) -> None:
    report = pf.run_cli_preflight(
        settings,
        require_hardware=True,
        server_probe=no_server,
        lerobot_prober=ok_surface,
        sleep=no_sleep,
        write_report=False,
    )
    assert report.exit_code == pf.EXIT_REHEARSAL_REJECTED
    assert not pf.reports_dir(settings).exists()


def test_cli_asks_a_running_server_for_the_bus_facts(settings: Settings) -> None:
    def factory(arm: str) -> FakeFeetechBus:
        raise AssertionError("must not open a bus while a server runs")

    asked: list[tuple[Any, ...]] = []

    def server_bus_facts(probe: ServerProbe, arms: Any, backend: str) -> Any:
        asked.append((probe.url, tuple(arms), backend))
        access = pf.BusAccess("synthetic", "lent by the server's RobotManager")
        return access, {arm: bus_facts(arm) for arm in arms}

    report = pf.run_cli_preflight(
        settings,
        server_probe=suite_server({**SUITE_STATUS, "estopped": True}),
        fake_bus_factory=factory,
        lerobot_prober=ok_surface,
        sleep=no_sleep,
        write_report=False,
        server_bus_facts=server_bus_facts,
    )
    assert asked == [("http://127.0.0.1:8816/api/robot/status", ARMS, "mock")]
    assert report.bus_access == "synthetic" and "RobotManager" in report.bus_access_detail
    by_key = {check.key: check for check in report.checks}
    assert by_key["motors.follower"].status == "pass" and by_key["motors.leader"].status == "pass"
    assert by_key["estop"].status == "warn"  # read from the server's status
    assert by_key["cameras"].status == "pass"  # live_cameras from the server


def test_cli_reports_a_running_servers_refusal(settings: Settings) -> None:
    def server_bus_facts(probe: ServerProbe, arms: Any, backend: str) -> Any:
        refusal = "RobotManager refused idle bus access: manual control is armed"
        return pf.BusAccess("refused", refusal, pf.REFUSED_REMEDIATION), {}

    report = pf.run_cli_preflight(
        settings,
        server_probe=suite_server(),
        lerobot_prober=ok_surface,
        sleep=no_sleep,
        write_report=False,
        server_bus_facts=server_bus_facts,
    )
    assert report.bus_access == "refused"
    motors = {check.key: check for check in report.checks}["motors.follower"]
    assert motors.status == "skip" and motors.blocking and "armed" in motors.message
    assert "disarm" in motors.remediation
    assert report.overall == "incomplete" and report.exit_code == pf.EXIT_INCOMPLETE


def test_cli_refuses_the_bus_while_another_program_answers(settings: Settings) -> None:
    def not_suite(settings: Any) -> ServerProbe:
        return ServerProbe(running=True, url="http://127.0.0.1:8816/api/robot/status", detail="?")

    def server_bus_facts(*args: Any) -> Any:
        raise AssertionError("only a Suite server lends its buses")

    report = pf.run_cli_preflight(
        settings,
        server_probe=not_suite,
        lerobot_prober=ok_surface,
        sleep=no_sleep,
        write_report=False,
        server_bus_facts=server_bus_facts,
    )
    assert report.bus_access == "refused"
    motors = {check.key: check for check in report.checks}["motors.follower"]
    assert motors.status == "skip" and "owns the serial buses" in motors.message
    assert report.exit_code == pf.EXIT_INCOMPLETE


def test_cli_refuses_the_bus_while_another_command_holds_the_lock(settings: Settings) -> None:
    with BusLock(bus_lock_path(settings), purpose="so101 rest-pose record"):
        report = pf.run_cli_preflight(
            settings,
            server_probe=no_server,
            lerobot_prober=ok_surface,
            sleep=no_sleep,
            write_report=False,
        )
    assert report.bus_access == "refused"
    assert "so101 rest-pose record" in report.bus_access_detail
    assert report.exit_code == pf.EXIT_INCOMPLETE


def _write_real_calibrations(settings: Settings) -> None:
    for arm in ARMS:
        device_id = pf.device_id_for(settings, arm)
        path = lerobot_paths.lerobot_calibration_file(arm, device_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(synthetic_calibration_for(arm)), encoding="utf-8")
        CalibrationResult(
            arm=arm,
            joints={name: JointCalibration(min=-90, max=90, offset=0) for name in JOINT_NAMES},
            device_id=device_id,
            backend="lerobot",
        ).save(Path(settings.data_dir) / f"calibration_{arm}.json")


def test_cli_real_path_reads_through_the_opener_and_closes_without_torque(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = real_settings(
        monkeypatch, SO101_FOLLOWER_PORT="/dev/fake-f", SO101_LEADER_PORT="/dev/fake-l"
    )
    _write_real_calibrations(real)
    opened: dict[str, FakeFeetechBus] = {}

    def opener(port: str) -> FakeFeetechBus:
        arm = {"/dev/fake-f": "follower", "/dev/fake-l": "leader"}[port]
        opened[port] = FakeFeetechBus(arm=arm)
        return opened[port]

    report = pf.run_cli_preflight(
        real,
        server_probe=no_server,
        bus_opener=opener,
        lerobot_prober=ok_surface,
        port_inspector=usable_port,
        port_lister=lambda: LISTING,
        camera_prober=lambda ids: [pf.CameraProbe(camera=i, ok=True, mean=80.0) for i in ids],
        environ={},
        sleep=no_sleep,
    )
    by_key = {check.key: check for check in report.checks}
    assert report.backend == "real" and report.rehearsal is False
    assert not any(check.rehearsal for check in report.checks)
    assert report.bus_access == "opened"
    assert by_key["serial_ports"].status == "pass"
    assert by_key["calibration.follower"].status == "pass", by_key["calibration.follower"].message
    assert by_key["rest_pose.follower"].status == "skip"  # nothing recorded yet
    assert report.overall == "incomplete"
    assert report.ready_for_hardware is False
    assert report.exit_code == pf.EXIT_INCOMPLETE
    assert sorted(opened) == ["/dev/fake-f", "/dev/fake-l"]
    for fake in opened.values():
        assert fake.calls[-1] == ("disconnect", False)
    probe = BusLock(bus_lock_path(real), purpose="t").acquire()
    assert probe.held
    probe.release()


def test_cli_real_path_fails_closed_on_missing_ports(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = real_settings(
        monkeypatch,
        SO101_FOLLOWER_PORT=str(tmp_path / "no-follower"),
        SO101_LEADER_PORT=str(tmp_path / "no-leader"),
    )

    def opener(port: str) -> FakeFeetechBus:
        raise AssertionError("missing ports must not be opened")

    report = pf.run_cli_preflight(
        real,
        server_probe=no_server,
        bus_opener=opener,
        lerobot_prober=ok_surface,
        camera_prober=lambda ids: [],
        environ={},
        sleep=no_sleep,
        write_report=False,
    )
    by_key = {check.key: check for check in report.checks}
    assert by_key["serial_ports"].status == "fail"
    assert by_key["motors.follower"].status == "fail"
    assert "does not exist" in by_key["motors.follower"].message
    assert report.exit_code == pf.EXIT_FAIL and not report.ready_for_hardware


def test_cli_real_path_without_lerobot_never_opens_a_bus(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = real_settings(monkeypatch)

    def opener(port: str) -> FakeFeetechBus:
        raise AssertionError("must not open a bus without LeRobot")

    report = pf.run_cli_preflight(
        real,
        server_probe=no_server,
        bus_opener=opener,
        lerobot_prober=LeRobotSurface,
        camera_prober=lambda ids: [],
        environ={},
        sleep=no_sleep,
        write_report=False,
    )
    assert report.bus_access == "unavailable"
    assert {check.key: check.status for check in report.checks}["lerobot"] == "fail"
    assert report.exit_code == pf.EXIT_FAIL


class StubManager:
    def __init__(
        self,
        settings: Any,
        *,
        estopped: bool = False,
        fault: str | None = None,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.guard = SimpleNamespace(limits=dict(DEFAULT_LIMITS))
        self.fault = fault
        self._status = {**SUITE_STATUS, "estopped": estopped}
        if runner is not None:
            self.run_idle_bus_diagnostic = runner

    def status(self) -> dict[str, Any]:
        return dict(self._status)


def test_api_rehearsal_uses_the_synthetic_bus_and_writes_nothing(settings: Settings) -> None:
    report = pf.run_api_preflight(
        StubManager(settings, fault="overheat"), lerobot_prober=LeRobotSurface, sleep=no_sleep
    )
    assert report.origin == "api" and report.rehearsal and report.bus_access == "synthetic"
    assert all(check.rehearsal for check in report.checks)
    by_key = {check.key: check for check in report.checks}
    assert by_key["motors.follower"].status == "pass"
    assert by_key["estop"].status == "warn" and "overheat" in by_key["estop"].message
    assert not pf.reports_dir(settings).exists()


def test_api_real_without_wave2_defers_the_bus_checks(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = real_settings(monkeypatch)
    report = pf.run_api_preflight(StubManager(real), lerobot_prober=ok_surface, sleep=no_sleep)
    assert report.bus_access == "deferred" and report.bus_access_detail == pf.WAVE2_BUS_MESSAGE
    for check in report.checks:
        if pf.CHECKS_BY_ID[check.id].needs_bus and check.id != "calibration":
            assert check.status == "skip" and check.message == pf.WAVE2_BUS_MESSAGE, check.key
            assert "so101 preflight" in check.remediation
    assert report.exit_code in (pf.EXIT_FAIL, pf.EXIT_INCOMPLETE)
    assert not report.ready_for_hardware


def test_api_real_runner_lends_each_arm_through_the_manager(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = real_settings(
        monkeypatch, SO101_FOLLOWER_PORT="/dev/fake-f", SO101_LEADER_PORT="/dev/fake-l"
    )
    opened: list[str] = []
    calls: list[tuple[str, float]] = []

    def opener(port: str) -> FakeFeetechBus:
        opened.append(port)
        return FakeFeetechBus(arm="leader" if port.endswith("-l") else "follower")

    def runner(target: str, fn: Callable[[Any], Any], *, timeout_s: float, opener: Any) -> Any:
        calls.append((target, timeout_s))
        bus = opener(pf.port_for(real, target))
        try:
            return fn(bus)
        finally:
            bus.disconnect(disable_torque=False)

    report = pf.run_api_preflight(
        StubManager(real, runner=runner),
        bus_opener=opener,
        lerobot_prober=ok_surface,
        port_inspector=usable_port,
        sleep=no_sleep,
    )
    assert report.bus_access == "opened" and "RobotManager" in report.bus_access_detail
    assert calls == [("follower", pf.API_BUS_WAIT_S), ("leader", pf.API_BUS_WAIT_S)]
    assert opened == ["/dev/fake-f", "/dev/fake-l"]
    by_key = {check.key: check.status for check in report.checks}
    assert by_key["motors.follower"] == "pass" and by_key["motors.leader"] == "pass"


def test_api_real_runner_reports_a_port_that_does_not_open(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from so101.bringup.buses import BusDiagnosticOpenError

    real = real_settings(monkeypatch)

    def runner(target: str, fn: Any, *, timeout_s: float, opener: Any) -> Any:
        raise BusDiagnosticOpenError("ConnectionError: could not open the port")

    report = pf.run_api_preflight(
        StubManager(real, runner=runner),
        lerobot_prober=ok_surface,
        port_inspector=usable_port,
        sleep=no_sleep,
    )
    assert report.bus_access == "opened"
    motors = {check.key: check for check in report.checks}["motors.follower"]
    assert motors.status == "fail" and "could not open the port" in motors.message


def test_api_real_runner_skips_a_missing_port_without_asking(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = real_settings(monkeypatch)

    def runner(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a port that does not exist is not lent")

    def missing(arm: str, path: str, listing: Any = None) -> pf.PortFacts:
        return pf.PortFacts(arm=arm, path=path, exists=False)

    report = pf.run_api_preflight(
        StubManager(real, runner=runner),
        lerobot_prober=ok_surface,
        port_inspector=missing,
        sleep=no_sleep,
    )
    motors = {check.key: check for check in report.checks}["motors.follower"]
    assert motors.status == "fail" and "does not exist" in motors.message


def test_api_real_runner_refusal_is_reported(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = real_settings(monkeypatch)

    def runner(target: str, fn: Any, *, timeout_s: float, opener: Any) -> Any:
        raise RuntimeError("bus diagnostics need an idle robot: manual control is armed")

    report = pf.run_api_preflight(
        StubManager(real, runner=runner),
        lerobot_prober=ok_surface,
        port_inspector=usable_port,
        sleep=no_sleep,
    )
    assert report.bus_access == "refused" and "idle robot" in report.bus_access_detail
    motors = {check.key: check for check in report.checks}["motors.follower"]
    assert motors.status == "skip" and motors.remediation == pf.REFUSED_REMEDIATION


def test_api_preflight_works_with_the_real_robot_manager(settings: Settings) -> None:
    from so101.services.robot_manager import get_manager

    report = pf.run_api_preflight(get_manager(), lerobot_prober=LeRobotSurface, sleep=no_sleep)
    assert report.rehearsal and report.bus_access == "synthetic"
    assert {check.key: check.status for check in report.checks}["estop"] == "pass"
