"""Read-only bus diagnostics: the real opener's calls, fact collection and the fake bus."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from so101 import JOINT_NAMES
from so101.bringup import buses, ownership
from so101.bringup.buses import FakeFeetechBus


def no_sleep(_: float) -> None:
    return None


def test_collect_bus_facts_reads_everything_without_writing() -> None:
    bus = FakeFeetechBus(arm="follower")
    facts = buses.collect_bus_facts(
        bus, arm="follower", port="/dev/x", source="mock", sleep=no_sleep
    )
    assert facts.responding == list(JOINT_NAMES) and facts.missing == []
    assert set(facts.firmware.values()) == {"0.0"}
    assert set(facts.registers) == set(buses.SNAPSHOT_REGISTERS)
    assert len(facts.position_samples) == buses.POSITION_SAMPLES
    assert facts.synthetic_calibration == buses.synthetic_calibration_for("follower")
    assert {name for name, _ in bus.calls} == {"ping", "read", "sync_read"}
    raw_reads = [args for name, args in bus.calls if name in ("read", "sync_read")]
    assert all(args[-1] is False for args in raw_reads)  # normalize=False everywhere


def test_register_errors_are_localized_to_the_failing_motor() -> None:
    bus = FakeFeetechBus(register_errors={"Present_Temperature": ["gripper"]})
    facts = buses.collect_bus_facts(
        bus, arm="follower", port="/dev/x", source="mock", sleep=no_sleep
    )
    assert set(facts.register_errors["Present_Temperature"]) == {"gripper"}
    assert "gripper" not in facts.register("Present_Temperature")
    assert len(facts.register("Present_Temperature")) == 5


def test_missing_motors_are_not_read() -> None:
    bus = FakeFeetechBus(missing_ids=[2, 4])
    facts = buses.collect_bus_facts(
        bus, arm="follower", port="/dev/x", source="mock", sleep=no_sleep
    )
    assert facts.missing == ["shoulder_lift", "wrist_flex"]
    assert set(facts.register("Present_Voltage")) == set(facts.responding)
    assert facts.register_errors == {}


def test_position_sampling_spaces_reads() -> None:
    slept: list[float] = []
    bus = FakeFeetechBus()
    buses.collect_bus_facts(
        bus, arm="f", port="/dev/x", source="mock", samples=4, sleep=slept.append
    )
    assert slept == [buses.POSITION_INTERVAL_S] * 3


def test_diagnose_port_reports_open_failures() -> None:
    def opener(port: str) -> Any:
        raise ConnectionError(f"Could not connect on port '{port}'")

    facts = buses.diagnose_port("/dev/x", arm="leader", source="real", opener=opener)
    assert facts.attempted and not facts.opened
    assert facts.open_error is not None and "ConnectionError" in facts.open_error


def test_diagnose_port_always_closes_without_disabling_torque() -> None:
    bus = FakeFeetechBus(arm="leader")
    facts = buses.diagnose_port(
        "/dev/x", arm="leader", source="real", opener=lambda port: bus, sleep=no_sleep
    )
    assert facts.opened and bus.calls[-1] == ("disconnect", False)


def test_fake_bus_has_no_write_surface() -> None:
    bus = FakeFeetechBus()
    for name in (
        "write",
        "sync_write",
        "enable_torque",
        "disable_torque",
        "write_calibration",
        "set_baudrate",
        "configure_motors",
    ):
        assert not hasattr(bus, name), name
    with pytest.raises(ValueError):
        bus.sync_read("Present_Position")
    bus.disconnect(disable_torque=False)
    with pytest.raises(RuntimeError):
        bus.ping("gripper")


class _FakeLeRobotBus:
    instances: list[_FakeLeRobotBus] = []
    fail_baudrate = False

    def __init__(self, port: str, motors: dict[str, Any]) -> None:
        self.port = port
        self.motors = motors
        self.calls: list[tuple[str, Any]] = []
        _FakeLeRobotBus.instances.append(self)

    def connect(self, handshake: bool = True) -> None:
        self.calls.append(("connect", handshake))

    def set_baudrate(self, baudrate: int) -> None:
        self.calls.append(("set_baudrate", baudrate))
        if self.fail_baudrate:
            raise RuntimeError("Failed to write bus baud rate")

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))


@pytest.fixture
def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLeRobotBus]:
    monkeypatch.setattr(ownership, "require_device_lock", lambda port: port)
    motors = types.ModuleType("lerobot.motors")
    motors.Motor = lambda id, model, norm_mode: (id, model, norm_mode)  # type: ignore[attr-defined]
    motors.MotorNormMode = types.SimpleNamespace(  # type: ignore[attr-defined]
        DEGREES="degrees", RANGE_0_100="range_0_100"
    )
    feetech = types.ModuleType("lerobot.motors.feetech")
    feetech.FeetechMotorsBus = _FakeLeRobotBus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.motors", motors)
    monkeypatch.setitem(sys.modules, "lerobot.motors.feetech", feetech)
    _FakeLeRobotBus.instances = []
    _FakeLeRobotBus.fail_baudrate = False
    return _FakeLeRobotBus


def test_open_feetech_bus_skips_the_handshake(fake_lerobot: type[_FakeLeRobotBus]) -> None:
    bus = buses.open_feetech_bus("/dev/ttyACM0")
    created = fake_lerobot.instances[-1]
    assert bus is created
    assert created.calls == [("connect", False), ("set_baudrate", 1_000_000)]
    assert created.motors["shoulder_pan"] == (1, "sts3215", "degrees")
    assert created.motors["gripper"] == (6, "sts3215", "range_0_100")


def test_raw_diagnostic_open_requires_device_lock_before_driver_import() -> None:
    with pytest.raises(ownership.BusOwnershipError, match="no adopted DeviceBusLock"):
        buses.open_feetech_bus("/dev/ttyACM0")


def test_open_feetech_bus_closes_on_failure(fake_lerobot: type[_FakeLeRobotBus]) -> None:
    fake_lerobot.fail_baudrate = True
    with pytest.raises(RuntimeError):
        buses.open_feetech_bus("/dev/ttyACM0")
    assert fake_lerobot.instances[-1].calls[-1] == ("disconnect", False)
