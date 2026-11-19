"""FeetechBus torque-off, disconnect and connect errors (safety-B2 and the bring-up findings).

docs/audit/v0.1/safety.md safety-B2: a torque-off that cannot be confirmed must
never read as "torque off". The bring-up stream checked FeetechBus against the
LeRobot 0.6.1 source and found five gaps, all fixed here:

1. ``set_torque(False)`` used LeRobot's ``disable_torque()``, which writes motor
   by motor and stops at the first servo that does not answer. One dropped
   packet left every later motor energized.
2. ``disconnect()`` skipped ``closePort()`` when LeRobot's torque-off raised,
   so the port handle leaked (on Windows the COM port stays locked).
3. ``LEROBOT_HOME`` makes ``import lerobot`` raise ValueError.
4. The install hint quoted ``'.[robot]'``, which cmd.exe keeps literally.
5. A failed motor handshake did not point at ``so101 preflight``.

The RC2 torque-truth audit found two residual gaps: failed-connect cleanup
discarded the motors it could not confirm off, and successful connect trusted a
torque-disable acknowledgement without reading ``Torque_Enable`` back.

``FakeServoBus`` keeps the LeRobot 0.6.1 ``FeetechMotorsBus`` semantics these
paths depend on (motors_bus.py and feetech/feetech.py):

- ``write``, ``read`` and ``disable_torque`` wait for a status packet and
  raise ConnectionError when a servo does not answer.
- ``disable_torque`` walks the motors in order and stops at the first failure.
- ``sync_write`` is one broadcast with no status packets; it can be lost.
- ``connect`` opens the port first; a failed handshake raises with it open.
- ``disconnect(disable_torque=True)`` skips ``closePort()`` when that raises.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from so101 import JOINT_NAMES
from so101.bringup import ownership
from so101.config import get_settings
from so101.hardware.follower import SO101Follower
from so101.hardware.motors import (
    TORQUE_OFF_RETRIES,
    FeetechBus,
    HardwareUnavailableError,
    MockBus,
    TorqueOffUnconfirmed,
    torque_off_all,
)

SILENT = "shoulder_lift"  # motor 2 of 6
LATER_MOTORS = JOINT_NAMES[2:]  # motors 3-6


class FakeServoBus:
    """One LeRobot 0.6.1 FeetechMotorsBus handle on six STS3215 servos."""

    def __init__(self, *, open_fails: bool = False, handshake_fails: bool = False) -> None:
        self.motors = {
            joint: SimpleNamespace(id=index + 1) for index, joint in enumerate(JOINT_NAMES)
        }
        self.torque = dict.fromkeys(JOINT_NAMES, 1)  # left energized by an earlier session
        self.silent: set[str] = set()
        # These motors acknowledge Torque_Enable=0 but leave the register at 1.
        self.stuck_torque_on: set[str] = set()
        self.broadcast_lost = False
        self.open_fails = open_fails
        self.handshake_fails = handshake_fails
        self.is_connected = False  # port_handler.is_open
        self.writes: list[tuple[str, str, int]] = []  # (register, motor, num_retry)
        self.closed_with: list[bool] = []  # disable_torque argument of each closePort

    def _require_port(self) -> None:
        if not self.is_connected:
            raise ConnectionError("DeviceNotConnectedError: the port is not open")

    def _motor_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        return [motors] if isinstance(motors, str) else list(motors)

    def connect(self, handshake: bool = True) -> None:
        if self.open_fails:
            raise ConnectionError(
                "Could not connect on port 'COM7'. Make sure you are using the correct port."
            )
        self.is_connected = True  # openPort() succeeded
        if handshake and self.handshake_fails:
            raise RuntimeError("FeetechMotorsBus motor check failed: some motors were not found")

    def configure_motors(self) -> None:
        self._require_port()

    def write(
        self, register: str, motor: str, value: Any, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        self._require_port()
        self.writes.append((register, motor, num_retry))
        if motor in self.silent:
            raise ConnectionError(
                f"Failed to write '{register}' on {motor} after {num_retry + 1} tries. "
                "[TxRxResult] There is no status packet!"
            )
        if register == "Torque_Enable":
            if not (int(value) == 0 and motor in self.stuck_torque_on):
                self.torque[motor] = int(value)

    def read(self, register: str, motor: str, *, normalize: bool = True, num_retry: int = 0) -> int:
        self._require_port()
        if motor in self.silent:
            raise ConnectionError(
                f"Failed to read '{register}' on {motor} after {num_retry + 1} tries."
            )
        assert register == "Torque_Enable"
        return self.torque[motor]

    def sync_write(
        self, register: str, values: Any, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        self._require_port()
        if self.broadcast_lost:
            return  # no status packet: a lost broadcast is silent
        for motor in self.motors:
            if motor not in self.silent and register == "Torque_Enable":
                if not (int(values) == 0 and motor in self.stuck_torque_on):
                    self.torque[motor] = int(values)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._motor_list(motors):
            self.write("Torque_Enable", motor, 1, num_retry=num_retry)
            self.write("Lock", motor, 1, num_retry=num_retry)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._motor_list(motors):
            self.write("Torque_Enable", motor, 0, num_retry=num_retry)
            self.write("Lock", motor, 0, num_retry=num_retry)

    def disconnect(self, disable_torque: bool = True) -> None:
        self._require_port()
        if disable_torque:
            self.disable_torque(num_retry=5)  # raises -> closePort() never runs
        self.closed_with.append(disable_torque)
        self.is_connected = False


class FakeSO101Follower:
    """The parts of LeRobot's SO101Follower that FeetechBus touches."""

    is_calibrated = True
    config = SimpleNamespace(
        position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32
    )

    def __init__(self, servos: FakeServoBus) -> None:
        self.bus = servos
        self.cameras: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def disconnect(self) -> None:
        self.bus.disconnect(True)


def _feetech(monkeypatch: pytest.MonkeyPatch, servos: FakeServoBus) -> FeetechBus:
    monkeypatch.setattr(ownership, "require_device_lock", lambda port: port)
    bus = FeetechBus("COM7", 1_000_000, "follower")
    monkeypatch.setattr(bus, "_build_modern_device", lambda: FakeSO101Follower(servos))
    monkeypatch.setattr(bus, "_position_mode_value", lambda: 0)
    return bus


def _energized(monkeypatch: pytest.MonkeyPatch) -> tuple[FeetechBus, FakeServoBus]:
    servos = FakeServoBus()
    bus = _feetech(monkeypatch, servos)
    bus.connect()
    assert servos.torque == dict.fromkeys(JOINT_NAMES, 0), "connect leaves torque off"
    assert bus.torque_state == "off"
    monkeypatch.setattr(ownership, "require_enable_intent", lambda _port: None)
    bus.set_torque(True)
    assert bus.torque_state == "on"
    servos.writes.clear()
    return bus, servos


def _torque_writes(servos: FakeServoBus) -> list[tuple[str, int]]:
    return [
        (motor, retries)
        for register, motor, retries in servos.writes
        if register == "Torque_Enable"
    ]


def test_raw_real_torque_enable_requires_device_lock_and_global_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servos = FakeServoBus()
    bus = _feetech(monkeypatch, servos)
    bus.connect()
    servos.writes.clear()

    with pytest.raises(ownership.BusOwnershipError, match="no adopted DeviceBusLock"):
        bus.set_torque(True)

    assert bus.torque_state == "off"
    assert servos.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert _torque_writes(servos) == []


def test_raw_real_connect_requires_device_lock_before_any_driver_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FeetechBus("COM7", 1_000_000, "follower")
    reached_driver: list[bool] = []
    monkeypatch.setattr(bus, "_build", lambda: reached_driver.append(True))

    with pytest.raises(ownership.BusOwnershipError, match="no adopted DeviceBusLock"):
        bus.connect()

    assert reached_driver == []


def test_raw_real_follower_connect_cannot_bypass_device_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("SO101_BACKEND", "real")
    get_settings.cache_clear()
    follower = SO101Follower(port="/dev/raw-unlocked")

    with pytest.raises(ownership.BusOwnershipError, match="no adopted DeviceBusLock"):
        follower.connect()


def test_endpoint_replacement_during_driver_open_closes_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servos = FakeServoBus()
    bus = _feetech(monkeypatch, servos)
    resolved = iter(["COM7", "COM7", "COM8"])
    monkeypatch.setattr(ownership, "require_device_lock", lambda _port: next(resolved))

    def refuse_disk(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("post-open mismatch must not write disk")

    monkeypatch.setattr(ownership, "_atomic_create_marker", refuse_disk)

    with pytest.raises(RuntimeError, match="changed during driver open"):
        bus.connect()

    assert servos.is_connected is False
    assert servos.closed_with[-1] is False
    assert bus.torque_state == "unconfirmed"


# --- finding 1: torque-off reaches every motor ---------------------------------------


def test_torque_off_reaches_the_motors_after_a_silent_one(monkeypatch: pytest.MonkeyPatch):
    """The coordinator's case: motor 2 of 6 does not answer and the broadcast is
    lost, so every motor needs its own acknowledged write. LeRobot's
    ``disable_torque()`` stopped at motor 2; motors 3-6 must still get theirs."""

    bus, servos = _energized(monkeypatch)
    servos.silent.add(SILENT)
    servos.broadcast_lost = True

    with pytest.raises(TorqueOffUnconfirmed) as caught:
        bus.set_torque(False)

    writes = _torque_writes(servos)
    assert [motor for motor, _ in writes] == JOINT_NAMES, "motors 3-6 still got the write"
    assert all(retries >= TORQUE_OFF_RETRIES >= 2 for _, retries in writes)
    assert all(servos.torque[motor] == 0 for motor in LATER_MOTORS + ["shoulder_pan"])
    assert servos.torque[SILENT] == 1
    assert caught.value.motors == [SILENT], "only the motor still energized is named"
    assert SILENT in str(caught.value)
    assert bus.torque_state == "unconfirmed"
    assert bus.connected, "a torque-off never closes the port (Halt keeps it open)"


def test_a_received_broadcast_still_names_the_silent_motor(monkeypatch: pytest.MonkeyPatch):
    bus, servos = _energized(monkeypatch)
    servos.silent.add(SILENT)

    with pytest.raises(TorqueOffUnconfirmed) as caught:
        bus.set_torque(False)

    assert caught.value.motors == [SILENT]
    assert {motor for motor, level in servos.torque.items() if level} == {SILENT}
    # Motors already read back at 0 after the broadcast need no second write.
    assert [motor for motor, _ in _torque_writes(servos)] == [SILENT]
    assert bus.torque_state == "unconfirmed"


def test_torque_off_all_reports_instead_of_raising(monkeypatch: pytest.MonkeyPatch):
    bus, servos = _energized(monkeypatch)
    servos.silent.add(SILENT)
    servos.broadcast_lost = True

    assert bus.torque_off_all() is False
    assert torque_off_all(bus) is False
    assert bus.torque_state == "unconfirmed"

    servos.silent.clear()
    assert bus.torque_off_all() is True
    assert servos.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert bus.torque_state == "off"


def test_a_healthy_torque_off_is_one_broadcast_and_a_read_back(monkeypatch: pytest.MonkeyPatch):
    bus, servos = _energized(monkeypatch)

    bus.set_torque(False)

    assert servos.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert _torque_writes(servos) == [], "every motor confirmed off by read-back"
    assert bus.torque_state == "off"


# --- finding 2: disconnect always closes the port ---------------------------------------


def test_disconnect_closes_the_port_when_torque_off_is_unconfirmed(monkeypatch: pytest.MonkeyPatch):
    bus, servos = _energized(monkeypatch)
    servos.silent.add(SILENT)

    with pytest.raises(TorqueOffUnconfirmed):
        bus.disconnect()

    assert servos.closed_with == [False], "closePort ran, after our own torque-off"
    assert servos.is_connected is False, "no leaked handle (a locked COM port on Windows)"
    assert bus.connected is False
    assert bus.torque_state == "unconfirmed", "a closed port is not torque off"
    assert {motor for motor, level in servos.torque.items() if level} == {SILENT}


def test_a_healthy_disconnect_turns_torque_off_and_closes(monkeypatch: pytest.MonkeyPatch):
    bus, servos = _energized(monkeypatch)

    bus.disconnect()

    assert servos.closed_with == [False]
    assert servos.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert bus.connected is False
    assert bus.torque_state == "off"


def test_halt_disconnect_closes_the_host_port_without_a_torque_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus, servos = _energized(monkeypatch)

    bus.disconnect_holding()

    assert servos.closed_with == [False]
    assert _torque_writes(servos) == []
    assert servos.torque == dict.fromkeys(JOINT_NAMES, 1)
    assert bus.connected is False
    assert bus.torque_state == "on"


def test_reconnecting_confirms_torque_off_through_a_fresh_port(monkeypatch: pytest.MonkeyPatch):
    bus, servos = _energized(monkeypatch)
    servos.silent.add(SILENT)
    with pytest.raises(TorqueOffUnconfirmed):
        bus.disconnect()

    servos.silent.clear()  # the loose cable is reseated
    bus.connect()

    assert servos.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert bus.torque_state == "off"


# --- findings 3-5: connect errors name the fix ------------------------------------------


@pytest.mark.parametrize("failure", ["open_fails", "handshake_fails"])
def test_a_failed_connect_names_preflight_and_never_leaks_the_port(
    monkeypatch: pytest.MonkeyPatch, failure: str
):
    servos = FakeServoBus(**{failure: True})
    bus = _feetech(monkeypatch, servos)

    with pytest.raises(HardwareUnavailableError) as caught:
        bus.connect()

    message = str(caught.value)
    assert "run `so101 preflight` (motors/firmware checks)" in message
    assert "COM7" in message
    assert servos.is_connected is False, "the port the handshake opened is closed again"
    assert bus.connected is False


def test_a_failed_handshake_marks_an_unreachable_motor_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
):
    servos = FakeServoBus(handshake_fails=True)
    servos.silent.add("gripper")  # the motor the handshake could not find
    bus = _feetech(monkeypatch, servos)

    with pytest.raises(HardwareUnavailableError):
        bus.connect()

    assert {motor for motor, level in servos.torque.items() if level} == {"gripper"}
    assert servos.is_connected is False
    assert servos.closed_with == [False]
    # This connection did not enable torque, but a fresh wrapper cannot know
    # whether an earlier session left the silent motor energized.
    assert bus.torque_state == "unconfirmed"


def test_connect_reads_back_torque_off_before_claiming_it(monkeypatch: pytest.MonkeyPatch):
    servos = FakeServoBus()
    servos.stuck_torque_on.add("gripper")
    bus = _feetech(monkeypatch, servos)

    with pytest.raises(TorqueOffUnconfirmed) as caught:
        bus.connect()

    assert caught.value.motors == ["gripper"]
    assert {motor for motor, level in servos.torque.items() if level} == {"gripper"}
    assert servos.is_connected is False
    assert servos.closed_with == [False]
    assert bus.connected is False
    assert bus.torque_state == "unconfirmed"


def test_connect_without_readback_never_claims_acknowledged_torque_off(
    monkeypatch: pytest.MonkeyPatch,
):
    servos = FakeServoBus()
    servos.stuck_torque_on.add("gripper")
    servos.read = None  # type: ignore[method-assign]
    bus = _feetech(monkeypatch, servos)

    with pytest.raises(TorqueOffUnconfirmed) as caught:
        bus.connect()

    assert caught.value.motors == JOINT_NAMES
    assert {motor for motor, level in servos.torque.items() if level} == {"gripper"}
    assert servos.is_connected is False
    assert servos.closed_with == [False]
    assert bus.connected is False
    assert bus.torque_state == "unconfirmed"


def test_failed_handshake_without_motor_metadata_stays_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
):
    servos = FakeServoBus(handshake_fails=True)
    servos.motors = {}
    bus = _feetech(monkeypatch, servos)

    with pytest.raises(HardwareUnavailableError):
        bus.connect()

    assert {motor for motor, level in servos.torque.items() if level} == set(JOINT_NAMES)
    assert servos.is_connected is False
    assert servos.closed_with == [False]
    assert bus.connected is False
    assert bus.torque_state == "unconfirmed"


def test_a_keyboard_interrupt_during_connect_stays_raw(monkeypatch: pytest.MonkeyPatch):
    servos = FakeServoBus()
    bus = _feetech(monkeypatch, servos)

    def interrupted(handshake: bool = True) -> None:
        servos.is_connected = True
        raise KeyboardInterrupt

    monkeypatch.setattr(servos, "connect", interrupted)
    with pytest.raises(KeyboardInterrupt):
        bus.connect()
    assert servos.is_connected is False


def test_lerobot_home_fails_before_lerobot_is_imported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ownership, "require_device_lock", lambda port: port)
    monkeypatch.setenv("LEROBOT_HOME", "elsewhere")
    bus = FeetechBus("COM7", 1_000_000, "follower")
    built: list[str] = []
    monkeypatch.setattr(bus, "_build_modern_device", lambda: built.append("import"))

    with pytest.raises(HardwareUnavailableError) as caught:
        bus.connect()

    assert "LEROBOT_HOME is no longer supported by LeRobot 0.6.1" in str(caught.value)
    assert "set HF_LEROBOT_HOME" in str(caught.value)
    assert built == []


def test_the_install_hint_survives_cmd_exe_quoting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ownership, "require_device_lock", lambda port: port)
    bus = FeetechBus("COM7", 1_000_000, "follower")

    def missing() -> None:
        raise ModuleNotFoundError("No module named 'lerobot'")

    monkeypatch.setattr(bus, "_build_modern_device", missing)
    with pytest.raises(HardwareUnavailableError) as caught:
        bus.connect()

    assert 'pip install -e ".[robot]"' in str(caught.value)
    assert "'.[robot]'" not in str(caught.value)


# --- MockBus and SO101Follower keep the same semantics ----------------------------------


def test_mock_bus_disconnect_always_closes_and_keeps_the_torque_truth():
    bus = MockBus("follower")
    bus.connect()
    bus.set_torque(True)
    assert bus.torque_state == "on"
    bus.fail_torque_off = True

    with pytest.raises(TorqueOffUnconfirmed):
        bus.disconnect()

    assert bus.connected is False
    assert bus.torque_enabled is True, "the servos still hold torque"
    assert bus.torque_state == "unconfirmed"
    bus.fail_torque_off = False
    bus.connect()
    assert bus.torque_state == "unconfirmed", "opening a port proves nothing"
    assert torque_off_all(bus) is True
    assert bus.torque_enabled is False
    assert bus.torque_state == "off"


def test_mock_bus_halt_disconnect_retains_torque_on_truth():
    bus = MockBus("follower")
    bus.connect()
    bus.set_torque(True)

    bus.disconnect_holding()

    assert bus.connected is False
    assert bus.torque_enabled is True
    assert bus.torque_state == "on"
    assert bus.torque_off_attempts == 0


def test_follower_disconnect_runs_one_torque_off_pass():
    follower = SO101Follower()
    follower.connect()
    bus = follower.bus
    assert isinstance(bus, MockBus) and bus.torque_enabled is True
    attempts = bus.torque_off_attempts

    follower.disconnect()

    assert bus.torque_off_attempts == attempts + 1
    assert bus.torque_enabled is False
    assert follower.connected is False


def test_follower_disconnect_raises_with_the_port_closed_when_unconfirmed():
    follower = SO101Follower()
    follower.connect()
    follower.bus.fail_torque_off = True  # type: ignore[attr-defined]

    with pytest.raises(TorqueOffUnconfirmed):
        follower.disconnect()

    assert follower.connected is False
    assert follower.bus.torque_state == "unconfirmed"  # type: ignore[attr-defined]
