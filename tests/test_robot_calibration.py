"""Calibration through the manager, and its torque-off pass.

- ``POST /api/calibration/start`` runs the sweep through
  ``RobotManager.run_calibration``: the manager's own bus, under its bus lock,
  with the C2 three-state torque truth. An unconfirmed torque-off afterwards is
  the job's error ("cut physical power") and latches a fault, never error None.
- ``FeetechBus.calibrate_timed`` turns torque off with the suite's pass
  (broadcast, per-motor retries, read-back), not LeRobot's ``disable_torque``,
  which stops at the first silent servo (docs/control-plane-safety.md
  "Torque-off on the bus").
- A failed calibration connect whose cleanup closes unconfirmed is latched even
  though the final disconnect has no live handle left to close.
"""

from __future__ import annotations

import dataclasses
import enum
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101 import JOINT_NAMES  # noqa: E402
from so101.bringup import ownership  # noqa: E402
from so101.hardware.calibration import CalibrationResult  # noqa: E402
from so101.hardware.motors import FeetechBus, TorqueOffUnconfirmed  # noqa: E402
from so101.services import robot_manager as rm  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    CalibrationTorqueOffUnconfirmed,
    ControlBusy,
    ControlConflict,
    ControlRejected,
    get_manager,
)
from so101.services.store import get_store  # noqa: E402


@pytest.fixture(autouse=True)
def _adopt_low_level_real_bus_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ownership, "require_device_lock", lambda port: port)


RESET = {"operator_id": "test-operator", "confirm": True}


def _events(kind: str) -> list[dict[str, Any]]:
    return [event for event in get_store().list_events(limit=500) if event["type"] == kind]


def _finished(client: TestClient, arm: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = client.get(f"/api/calibration/status/{arm}").json()
        if not status["in_progress"]:
            return status  # type: ignore[no-any-return]
        time.sleep(0.01)
    raise AssertionError("calibration did not finish")


# --- the route, through the manager ----------------------------------------------------


def test_calibration_runs_on_the_managers_bus_under_its_lock(monkeypatch):
    manager = get_manager()
    seen: dict[str, Any] = {}
    original = rm.calibrate

    def spy(bus: Any, arm: str, **kwargs: Any) -> CalibrationResult:
        seen["locked"] = manager._io_lock.locked()
        seen["bus"] = bus
        seen["connected"] = bus.connected
        return original(bus, arm, **kwargs)

    monkeypatch.setattr(rm, "calibrate", spy)
    with TestClient(create_app()) as client:
        response = client.post("/api/calibration/start", json={"arm": "follower"})
        assert response.status_code == 200, response.text
        status = _finished(client, "follower")
    assert status["message"] == "saved" and status["has_saved"] is True
    assert seen == {"locked": True, "bus": manager.follower().bus, "connected": True}
    assert manager.follower().bus.connected is False  # closed by the torque-off pass
    assert manager.status()["torque_state"] == "off" and manager.fault is None
    assert _events("calibration.saved")


def test_leader_calibration_uses_the_managers_leader_bus(monkeypatch):
    manager = get_manager()
    buses: list[Any] = []
    original = rm.calibrate

    def spy(bus: Any, arm: str, **kwargs: Any) -> CalibrationResult:
        buses.append(bus)
        return original(bus, arm, **kwargs)

    monkeypatch.setattr(rm, "calibrate", spy)
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "leader"}).status_code == 200
        assert _finished(client, "leader")["message"] == "saved"
    assert buses == [manager.leader().bus]


def test_an_unconfirmed_torque_off_is_the_jobs_error_and_latches_a_fault():
    manager = get_manager()
    manager.follower().bus.fail_torque_off = True  # type: ignore[attr-defined]
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "follower"}).status_code == 200
        status = _finished(client, "follower")
        assert status["message"].startswith("failed: calibration of the follower arm saved")
        assert "cut physical power to the follower arm" in status["message"]
        robot = client.get("/api/robot/status").json()
        assert robot["torque_unconfirmed"] is True and robot["torque_state"] == "unconfirmed"
        assert client.get("/api/control/state").json()["safety_state"] == "fault"
        (failed,) = _events("calibration.failed")
        assert "cut physical power" in failed["message"]
        assert _events("safety.force_deenergize_failed")
        # The file was saved before the torque-off, so the guard follows it.
        assert status["has_saved"] is True

        manager.follower().bus.fail_torque_off = False  # type: ignore[attr-defined]
        response = client.post("/api/robot/fault/reset", json=RESET)
        assert response.status_code == 200, response.text
        assert client.get("/api/robot/status").json()["torque_unconfirmed"] is False


def test_a_calibration_that_fails_still_turns_torque_off(monkeypatch):
    manager = get_manager()

    def broken(bus: Any, arm: str, **kwargs: Any) -> CalibrationResult:
        raise ValueError("calibration observed less than 100 encoder ticks")

    monkeypatch.setattr(rm, "calibrate", broken)
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "follower"}).status_code == 200
        status = _finished(client, "follower")
    assert status["message"] == "failed: calibration observed less than 100 encoder ticks"
    assert manager.follower().bus.connected is False
    assert manager.status()["torque_state"] == "off" and manager.fault is None


def test_a_failed_calibration_and_an_unconfirmed_torque_off_are_both_named(monkeypatch):
    manager = get_manager()

    def broken(bus: Any, arm: str, **kwargs: Any) -> CalibrationResult:
        raise KeyError("x")

    monkeypatch.setattr(rm, "calibrate", broken)
    manager.follower().bus.fail_torque_off = True  # type: ignore[attr-defined]
    assert manager.claim_calibration("follower") is None
    with pytest.raises(CalibrationTorqueOffUnconfirmed) as raised:
        manager.run_calibration("follower", 8.0)
    assert raised.value.saved is False
    assert "ended and torque-off was not confirmed" in str(raised.value)
    assert "the calibration itself failed: 'x'" in str(raised.value)
    manager.finish_calibration("follower", str(raised.value), saved=False)
    manager.follower().bus.fail_torque_off = False  # type: ignore[attr-defined]
    manager.clear_fault(operator_id="test-operator")


def test_run_calibration_needs_the_reservation():
    with pytest.raises(ControlConflict, match="no follower calibration reservation"):
        get_manager().run_calibration("follower", 8.0)


def test_run_calibration_refuses_the_sim_backend(monkeypatch):
    monkeypatch.setattr(rm, "current_backend", lambda settings: "sim")
    with pytest.raises(ControlRejected, match="sim backend"):
        get_manager().run_calibration("follower", 8.0)


def test_run_calibration_refuses_an_unknown_arm():
    with pytest.raises(ValueError, match="follower or leader"):
        get_manager().run_calibration("gripper", 8.0)


def test_run_calibration_answers_busy_when_the_bus_is_held():
    manager = get_manager()
    manager.claim_calibration("follower")
    manager.request_io_timeout_s = 0.01
    assert manager._io_lock.acquire(timeout=1)
    try:
        with pytest.raises(ControlBusy, match="calibration not sent"):
            manager.run_calibration("follower", 8.0)
    finally:
        manager._io_lock.release()
        manager.finish_calibration("follower", "busy")
    assert manager.follower().bus.connected is False


def test_a_cancelled_calibration_never_opens_the_bus():
    manager = get_manager()
    manager.claim_calibration("follower")
    manager.calibration_cancel_event.set()
    with pytest.raises(RuntimeError, match="cancelled before it started"):
        manager.run_calibration("follower", 8.0)
    manager.finish_calibration("follower", "cancelled")
    assert manager.follower().bus.connected is False


def test_a_port_that_does_not_open_is_the_jobs_error():
    manager = get_manager()
    manager.follower().bus.fail_connect = True  # type: ignore[attr-defined]
    with TestClient(create_app()) as client:
        assert client.post("/api/calibration/start", json={"arm": "follower"}).status_code == 200
        status = _finished(client, "follower")
    assert "could not open the port" in status["message"]
    assert manager.fault is None


# --- calibrate_timed: the suite's torque-off pass -----------------------------------------


@pytest.fixture
def fake_lerobot(monkeypatch):
    """The two LeRobot 0.6.1 names calibrate_timed imports, without LeRobot."""

    @dataclasses.dataclass
    class MotorCalibration:
        id: int
        drive_mode: int
        homing_offset: int
        range_min: int
        range_max: int

    class OperatingMode(enum.Enum):
        POSITION = 0

    motors = types.ModuleType("lerobot.motors")
    motors.MotorCalibration = MotorCalibration  # type: ignore[attr-defined]
    feetech = types.ModuleType("lerobot.motors.feetech")
    feetech.OperatingMode = OperatingMode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.motors", motors)
    monkeypatch.setitem(sys.modules, "lerobot.motors.feetech", feetech)


class CalibrationMotorsBus:
    """A LeRobot 0.6.1 FeetechMotorsBus with energized servos, some maybe silent."""

    def __init__(
        self, silent: set[str] | None = None, stuck_torque_on: set[str] | None = None
    ) -> None:
        self.motors = {
            name: SimpleNamespace(id=index + 1) for index, name in enumerate(JOINT_NAMES)
        }
        self.torque = dict.fromkeys(JOINT_NAMES, 1)
        self.silent = silent or set()
        self.stuck_torque_on = stuck_torque_on or set()
        self.calls: list[tuple[Any, ...]] = []
        self.samples = 0
        self.homed = False
        self.is_connected = False
        self.opens = 0

    def _answer(self, motor: str) -> None:
        if motor in self.silent:
            raise ConnectionError(f"Failed on {motor!r}: [TxRxResult] There is no status packet!")

    def connect(self, handshake: bool = True) -> None:
        self.opens += 1
        self.is_connected = True
        if handshake:
            for motor in self.motors:
                self._answer(motor)

    def configure_motors(self) -> None:
        return None

    def disconnect(self, disable_torque: bool = True) -> None:
        assert disable_torque is False
        self.is_connected = False

    def sync_write(self, register: str, value: Any, *, normalize: bool = True) -> None:
        self.calls.append(("sync_write", register, value))
        if register == "Torque_Enable":  # a broadcast: silent servos do not act on it
            for motor in self.motors:
                if motor not in self.silent and motor not in self.stuck_torque_on:
                    self.torque[motor] = int(value)

    def read(self, register: str, motor: str, *, normalize: bool = True, num_retry: int = 0) -> int:
        self.calls.append(("read", register, motor))
        self._answer(motor)
        return self.torque[motor]

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.calls.append(("disable_torque", motors))
        targets = (
            list(self.motors) if motors is None else [motors] if isinstance(motors, str) else motors
        )
        for motor in targets:  # LeRobot: stops at the first silent servo
            self._answer(motor)
            if motor not in self.stuck_torque_on:
                self.torque[motor] = 0

    def read_calibration(self) -> dict[str, Any]:
        self.calls.append(("read_calibration",))
        if self.homed:
            return {
                motor: SimpleNamespace(homing_offset=0, range_min=0, range_max=4095)
                for motor in self.motors
            }
        return {"previous": True}

    def write(self, register: str, motor: str, value: Any, **kwargs: Any) -> None:
        self.calls.append(("write", register, motor))
        self._answer(motor)
        if register == "Torque_Enable" and motor not in self.stuck_torque_on:
            self.torque[motor] = int(value)

    def set_half_turn_homings(self) -> dict[str, int]:
        self.calls.append(("set_half_turn_homings",))
        self.homed = True
        return dict.fromkeys(self.motors, 0)

    def sync_read(self, register: str, motors: Any, **kwargs: Any) -> dict[str, int]:
        self.samples += 1
        return {motor: 2048 + (300 if self.samples % 2 else -300) for motor in motors}

    def write_calibration(self, calibration: Any) -> None:
        self.calls.append(("write_calibration",))

    def heal(self) -> None:
        self.silent.clear()
        self.stuck_torque_on.clear()


class CalibrationConnectDevice:
    is_calibrated = True
    cameras: dict[str, Any] = {}
    config = SimpleNamespace(
        position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32
    )

    def __init__(self, bus: CalibrationMotorsBus) -> None:
        self.bus = bus

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected


@pytest.mark.parametrize("failure", ["silent", "stuck"])
def test_calibration_connect_failure_latches_closed_unconfirmed_bus(
    monkeypatch: pytest.MonkeyPatch, failure: str
):
    manager = get_manager()
    servos = CalibrationMotorsBus(
        silent={"gripper"} if failure == "silent" else None,
        stuck_torque_on={"gripper"} if failure == "stuck" else None,
    )
    bus = FeetechBus("/dev/so101-follower", 1_000_000, "follower")
    device = CalibrationConnectDevice(servos)
    monkeypatch.setattr(bus, "_build_modern_device", lambda: device)
    monkeypatch.setattr(bus, "_position_mode_value", lambda: 0)
    manager.follower().bus = bus
    manager.claim_calibration("follower")

    with pytest.raises(CalibrationTorqueOffUnconfirmed) as raised:
        manager.run_calibration("follower", 8.0)
    manager.finish_calibration("follower", str(raised.value), saved=False)

    assert raised.value.saved is False
    assert servos.is_connected is False
    assert bus.torque_state == "unconfirmed"
    assert {motor for motor, enabled in servos.torque.items() if enabled} == {"gripper"}
    status = manager.status()
    assert status["torque_unconfirmed"] is True and status["torque_state"] == "unconfirmed"
    assert manager.fault is not None
    saved = get_store().load_latches()
    assert saved is not None and saved["torque_unconfirmed"] is True

    opens = servos.opens
    with pytest.raises(ControlConflict, match="still unconfirmed"):
        manager.clear_fault(operator_id="test-operator")
    assert servos.opens > opens
    assert manager.status()["torque_unconfirmed"] is True

    servos.heal()
    manager.clear_fault(operator_id="test-operator")
    assert servos.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert servos.is_connected is False
    assert manager.fault is None
    status = manager.status()
    assert status["torque_unconfirmed"] is False and status["torque_state"] == "off"
    saved = get_store().load_latches()
    assert saved is not None and saved["torque_unconfirmed"] is False


def _feetech_bus(tmp_path: Path, motors_bus: CalibrationMotorsBus) -> FeetechBus:
    bus = FeetechBus("/dev/so101-follower", 1_000_000, "follower")
    path = tmp_path / "follower.json"
    bus._backend = "modern"
    bus._bus = motors_bus
    bus._device = SimpleNamespace(
        calibration={},
        calibration_fpath=path,
        is_connected=True,
        _save_calibration=lambda: path.write_text("{}", encoding="utf-8"),
    )
    return bus


def test_calibrate_timed_turns_torque_off_with_the_suite_pass(fake_lerobot, tmp_path):
    motors_bus = CalibrationMotorsBus()
    bus = _feetech_bus(tmp_path, motors_bus)
    ranges = bus.calibrate_timed(0.05)
    assert motors_bus.calls[0] == ("sync_write", "Torque_Enable", 0)  # the broadcast first
    assert ("disable_torque", None) not in motors_bus.calls  # never LeRobot's stop-at-first pass
    assert motors_bus.torque == dict.fromkeys(JOINT_NAMES, 0) and bus.torque_state == "off"
    assert ranges["gripper"] == (0.0, 100.0) and ("write_calibration",) in motors_bus.calls


def test_calibrate_timed_announces_sweep_only_after_midpoint_capture(fake_lerobot, tmp_path):
    motors_bus = CalibrationMotorsBus()
    bus = _feetech_bus(tmp_path, motors_bus)

    def progress(fraction: float, message: str) -> None:
        motors_bus.calls.append(("progress", fraction, message))

    bus.calibrate_timed(0.05, progress=progress)

    homing = motors_bus.calls.index(("set_half_turn_homings",))
    ready = next(
        index
        for index, call in enumerate(motors_bus.calls)
        if call[0] == "progress" and "now sweep" in call[2]
    )
    assert homing < ready
    assert "midpoint captured" in motors_bus.calls[ready][2]


def test_calibrate_timed_stops_before_writing_when_a_servo_stays_energized(fake_lerobot, tmp_path):
    motors_bus = CalibrationMotorsBus(silent={"elbow_flex"})
    bus = _feetech_bus(tmp_path, motors_bus)
    with pytest.raises(TorqueOffUnconfirmed, match="elbow_flex") as raised:
        bus.calibrate_timed(0.05)
    assert raised.value.motors == ["elbow_flex"]
    # Every servo after the silent one is off; LeRobot's pass would have left them on.
    assert {motor for motor, on in motors_bus.torque.items() if on} == {"elbow_flex"}
    assert bus.torque_state == "unconfirmed"
    assert not any(call[0] in {"write", "write_calibration"} for call in motors_bus.calls)


def test_calibrate_timed_rolls_back_with_the_suite_pass(fake_lerobot, tmp_path):
    motors_bus = CalibrationMotorsBus()
    bus = _feetech_bus(tmp_path, motors_bus)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled before commit"):
        bus.calibrate_timed(0.05, cancel_event=cancel)
    broadcasts = [call for call in motors_bus.calls if call == ("sync_write", "Torque_Enable", 0)]
    assert len(broadcasts) == 2  # at the start and again in the rollback
    assert ("disable_torque", None) not in motors_bus.calls
    assert ("write_calibration",) in motors_bus.calls  # the previous calibration restored


@pytest.mark.parametrize(("low", "high"), [(-156, 2048), (2048, 4300)])
def test_calibrate_timed_rejects_invalid_encoder_ranges_before_commit(
    fake_lerobot, tmp_path, monkeypatch, low, high
):
    from so101.hardware import motors

    class InvalidRangeBus(CalibrationMotorsBus):
        def sync_read(self, register, selected, **kwargs):
            positions = super().sync_read(register, selected, **kwargs)
            positions["elbow_flex"] = low if self.samples % 2 else high
            return positions

    motor_bus = InvalidRangeBus()
    bus = _feetech_bus(tmp_path, motor_bus)
    path = Path(bus._device.calibration_fpath)
    previous_file = b'{"preserve": true}\n'
    path.write_bytes(previous_file)
    bus._calibrated = True
    # One sweep tick with a controlled clock; no physical hardware or real sleep.
    ticks = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        motors, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    )
    written = []
    monkeypatch.setattr(motor_bus, "write_calibration", lambda value: written.append(value))

    with pytest.raises(ValueError, match="elbow_flex.*restart.*middle"):
        bus.calibrate_timed(0.05)

    assert written == [{"previous": True}]  # rollback only, never the invalid limits
    assert path.read_bytes() == previous_file
    assert bus._calibrated is True
    assert bus.torque_state == "off"
    assert motor_bus.torque == dict.fromkeys(JOINT_NAMES, 0)


@pytest.mark.parametrize(
    ("field", "actual"), [("homing_offset", 123), ("range_min", 100), ("range_max", 4000)]
)
def test_calibrate_timed_rejects_homing_readback_mismatch_before_sweep(
    fake_lerobot, tmp_path, monkeypatch, field, actual
):
    motor_bus = CalibrationMotorsBus()
    bus = _feetech_bus(tmp_path, motor_bus)
    read_calibration = motor_bus.read_calibration

    def wrong_readback():
        calibration = read_calibration()
        if motor_bus.homed:
            setattr(calibration["elbow_flex"], field, actual)
        return calibration

    monkeypatch.setattr(motor_bus, "read_calibration", wrong_readback)
    written = []
    monkeypatch.setattr(motor_bus, "write_calibration", lambda value: written.append(value))
    progress = []
    with pytest.raises(ValueError, match=f"homing read-back.*elbow_flex.*{field}"):
        bus.calibrate_timed(0.05, progress=lambda *values: progress.append(values))
    assert motor_bus.samples == 0
    assert not any("now sweep" in message for _, message in progress)
    assert written == [{"previous": True}]
    assert motor_bus.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert bus.torque_state == "off"


def test_invalid_range_report_includes_the_initial_homed_position(
    fake_lerobot, tmp_path, monkeypatch
):
    from so101.hardware import motors

    class OneSidedBus(CalibrationMotorsBus):
        def sync_read(self, register, selected, **kwargs):
            positions = super().sync_read(register, selected, **kwargs)
            positions["shoulder_pan"] = 2047 if self.samples == 1 else -542
            return positions

    bus = _feetech_bus(tmp_path, OneSidedBus())
    ticks = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        motors, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    )
    with pytest.raises(ValueError, match="post-homing starting readings.*shoulder_pan.*2047"):
        bus.calibrate_timed(0.05)


def test_live_encoder_frames_include_running_extrema_and_freeze_failure(
    tmp_path, fake_lerobot, monkeypatch
):
    from so101.hardware import motors

    class BadElbow(CalibrationMotorsBus):
        def sync_read(self, register, selected, **kwargs):
            values = super().sync_read(register, selected, **kwargs)
            values["elbow_flex"] = 2047 if self.samples == 1 else -197
            return values

    native = BadElbow()
    bus = _feetech_bus(tmp_path, native)
    ticks = iter((0.0, 0.01, 1.0))
    monkeypatch.setattr(
        motors, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    )
    frames = []
    with pytest.raises(ValueError, match="elbow_flex"):
        bus.calibrate_timed(0.05, telemetry=frames.append)
    assert [frame["phase"] for frame in frames] == [
        "homing",
        "sampling",
        "sampling",
        "validating",
        "failed",
    ]
    first, second = frames[1:3]
    assert first["range_mins"]["elbow_flex"] == 2047  # earlier frame stays immutable
    assert second["positions"]["elbow_flex"] == -197
    assert second["range_mins"]["elbow_flex"] == -197
    assert second["range_maxes"]["elbow_flex"] == 2047
    assert second["sample_count"] == 2 and second["elapsed_seconds"] == 0.01
    assert "wrist_roll" not in second["positions"]  # fixed range, not a fake sample
    assert frames[-1]["affected_motors"] == ["elbow_flex"]
    assert frames[-1]["error"].startswith("calibration encoder ranges outside")
    assert frames[-1] == bus.calibration_diagnostics["live"]
    assert native.torque == dict.fromkeys(JOINT_NAMES, 0)


def test_homing_encoding_error_names_motor_and_restores_upstream_writer(tmp_path, fake_lerobot):
    class BadHoming(CalibrationMotorsBus):
        def write(self, register, motor, value, **kwargs):
            if register == "Homing_Offset":
                raise ValueError("Magnitude 2495 exceeds 2047 (max for sign_bit_index=11)")
            return super().write(register, motor, value, **kwargs)

        def set_half_turn_homings(self):
            self.write("Homing_Offset", "shoulder_lift", -2495)

    native = BadHoming()
    original = native.write
    bus = _feetech_bus(tmp_path, native)
    frames = []
    with pytest.raises(RuntimeError, match="shoulder_lift.*Homing_Offset=-2495.*Magnitude"):
        bus.calibrate_timed(0.05, telemetry=frames.append)
    assert native.write == original and "write" not in vars(native)
    assert native.samples == 0
    assert frames[-1]["phase"] == "failed"
    assert frames[-1]["affected_motors"] == ["shoulder_lift"]
    assert frames[-1]["failed_write"]["value"] == -2495
    assert ("write_calibration",) in native.calls  # prior calibration restored


class JointBaselineBus(CalibrationMotorsBus):
    """One selected motor can be homed; unrelated motors are never recalibrated."""

    def __init__(self):
        super().__init__()
        self.previous = {
            name: SimpleNamespace(
                id=index + 1, drive_mode=0, homing_offset=17, range_min=300, range_max=3700
            )
            for index, name in enumerate(JOINT_NAMES)
        }
        self.calibration = dict(self.previous)
        self.registers = dict(self.previous)
        self.selected = None
        self.calibration_writes = []
        self.sample_targets = []

    def read_calibration(self):
        return dict(self.registers)

    def set_half_turn_homings(self, selected):
        assert len(selected) == 1
        self.selected = selected[0]
        self.registers[self.selected] = SimpleNamespace(
            homing_offset=0, range_min=0, range_max=4095
        )
        return {self.selected: 0}

    def sync_read(self, register, selected, **kwargs):
        assert selected == [self.selected]
        self.sample_targets.append(list(selected))
        return super().sync_read(register, selected, **kwargs)

    def write_calibration(self, values):
        self.calibration_writes.append(dict(values))
        self.registers.update(values)
        self.calibration = dict(values)


@pytest.mark.parametrize("joint", JOINT_NAMES)
def test_joint_baseline_targets_one_joint_and_restores_prior_calibration(
    tmp_path, fake_lerobot, monkeypatch, joint
):
    from so101.hardware import motors

    native = JointBaselineBus()
    bus = _feetech_bus(tmp_path, native)
    path = Path(bus._device.calibration_fpath)
    path.write_bytes(b'{"preserve": "native file"}\n')
    initial_file = path.read_bytes()
    initial_calibrated = bus._calibrated
    ticks = iter((0.0, 0.01, 1.0))
    monkeypatch.setattr(
        motors, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    )
    frames = []
    ranges = bus.calibrate_timed(0.05, joint=joint, telemetry=frames.append)
    assert set(ranges) == {joint}
    assert native.selected == joint
    assert native.sample_targets == [[joint], [joint]]
    assert native.calibration_writes == [{joint: native.previous[joint]}]
    assert native.registers == native.previous
    assert native.calibration == native.previous
    assert path.read_bytes() == initial_file
    assert bus._calibrated == initial_calibrated
    assert bus.calibration_diagnostics["range_mins"] == {joint: 1748}
    assert bus.calibration_diagnostics["range_maxes"] == {joint: 2348}
    assert bus.calibration_diagnostics["restored"] is True
    assert frames[-1]["phase"] == "complete" and frames[-1]["baseline_only"]
    assert frames[-1]["motor_ids"] == {joint: JOINT_NAMES.index(joint) + 1}
    assert native.torque == dict.fromkeys(JOINT_NAMES, 0)
    assert all(call[2] == joint for call in native.calls if call[:2] == ("write", "Operating_Mode"))


def test_failed_joint_baseline_rolls_back_only_the_selected_motor(
    tmp_path, fake_lerobot, monkeypatch
):
    from so101.hardware import motors

    class Invalid(JointBaselineBus):
        def sync_read(self, register, selected, **kwargs):
            values = super().sync_read(register, selected, **kwargs)
            values["gripper"] = -100 if self.samples > 1 else 2047
            return values

    native = Invalid()
    bus = _feetech_bus(tmp_path, native)
    ticks = iter((0.0, 0.01, 1.0))
    monkeypatch.setattr(
        motors, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    )
    with pytest.raises(ValueError, match="gripper"):
        bus.calibrate_timed(0.05, joint="gripper")
    assert native.calibration_writes == [{"gripper": native.previous["gripper"]}]
    assert native.registers == native.previous
    assert bus.calibration_diagnostics["live"]["phase"] == "failed"


def test_joint_baseline_rejects_unconfirmed_register_restore(tmp_path, fake_lerobot, monkeypatch):
    from so101.hardware import motors

    class NoRestore(JointBaselineBus):
        def write_calibration(self, values):
            pass

    native = NoRestore()
    bus = _feetech_bus(tmp_path, native)
    ticks = iter((0.0, 0.01, 1.0))
    monkeypatch.setattr(
        motors, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    )
    with pytest.raises(RuntimeError, match="restore read-back failed for gripper"):
        bus.calibrate_timed(0.05, joint="gripper")
    assert not bus.calibration_diagnostics.get("restored")


def test_measured_joint_baseline_retains_raw_evidence_in_its_own_file(
    tmp_path, fake_lerobot, monkeypatch
):
    from so101.hardware import motors
    from so101.hardware.calibration import calibrate

    native = JointBaselineBus()
    bus = _feetech_bus(tmp_path, native)
    ticks = iter((0.0, 0.01, 1.0))
    monkeypatch.setattr(
        motors, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    )
    result = calibrate(bus, "follower", joint="gripper", sample_seconds=0.05)
    path = result.save()
    assert path.name == "calibration_joint_follower_gripper.json"
    loaded = CalibrationResult.load_joint("follower", "gripper")
    assert loaded is not None and loaded.joint == "gripper"
    assert loaded.backend == "lerobot" and loaded.device_id == bus.device_id
    assert loaded.diagnostics["range_mins"] == {"gripper": 1748}
    assert loaded.diagnostics["range_maxes"] == {"gripper": 2348}
    assert loaded.diagnostics["homing_offsets_requested"] == {"gripper": 0}
    assert loaded.diagnostics["motor_ids"] == {"gripper": 6}
    assert loaded.diagnostics["restored"] is True
    assert CalibrationResult.load("follower") is None
