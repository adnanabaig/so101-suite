"""Fault injection at the real driver seam (safety report §2 item 7).

Real mode (SO101_MOCK=0): RobotManager, SO101Follower and FeetechBus are the
production classes. Only LeRobot's SOFollower device is faked
(docs/audit/v0.1/safety.md §2 item 7; this converts safety repro 04). The
fake keeps the LeRobot 0.6.1 semantics the safety code depends on:

- Per-motor ``read``, ``write``, ``enable_torque`` and ``disable_torque``
  wait for a status packet and raise when a servo does not answer.
  ``disable_torque`` walks the motors in order and stops at the first failure.
- ``sync_write`` is one broadcast with no replies: every servo that hears it
  acts, and a silent one cannot hold up the rest. ``sync_read`` needs every
  servo to answer.
- ``send_action`` reads Present_Position first (``max_relative_target``).
- ``disconnect`` disables torque first and skips ``closePort()`` when that
  raises, so the port stays open and the servos keep their torque. FeetechBus
  therefore runs its own torque-off and closes with ``disable_torque=False``.
- ``connect`` opens the port, then pings every servo.

``Arm`` is the physical side. Each servo's torque is the truth the software
must report. ``Arm`` also injects the faults:

- silent servos (a brown-out, a loose daisy-chain cable), from the start or
  in the middle of an operation;
- an unplugged adapter, whose open handle stays dead after a replug;
- a wedged transfer that blocks its caller (a USB stall);
- servo health registers, plausible by default (5.0 V, 30 degC, idle load),
  that a test can override per servo (``Arm.health``).

Every transfer must run while the manager's bus lock is held, because the
manager is the only bus owner. The ``arm`` fixture fails any test that broke
that rule. Every test runs on the injected manager clock with manual
supervision (devex-B4), so the deadman and the watchdog act only when a test
says so.
"""

from __future__ import annotations

import errno
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101 import JOINT_NAMES, config  # noqa: E402
from so101.bringup import ownership  # noqa: E402
from so101.data.recorder import RecordingConfig  # noqa: E402
from so101.hardware.motors import FeetechBus, TorqueOffUnconfirmed  # noqa: E402
from so101.policy.server import MockPolicy  # noqa: E402
from so101.services import robot_manager as robot_manager_module  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    ControlConflict,
    ControlRejected,
    Mode,
    get_manager,
    reset_manager,
)

OPERATOR = "test-operator"
RESET = {"operator_id": OPERATOR, "confirm": True}
ALL = set(JOINT_NAMES)
TARGET = {"shoulder_pan": 2.0}


#: Raw health and limit registers each servo reports unless ``Arm.health``
#: overrides them: 5.0 V (the SO-101 kit's supply), 30 degC, no load, 13 mA,
#: Status 0, and an STS3215's EEPROM limits (70 degC, 8.0 V, 4.0 V).
HEALTH_RAW = {
    "Present_Voltage": 50,
    "Present_Temperature": 30,
    "Present_Load": 0,
    "Present_Current": 2,
    "Status": 0,
    "Max_Temperature_Limit": 70,
    "Max_Voltage_Limit": 80,
    "Min_Voltage_Limit": 40,
}


class DeviceNotConnectedError(ConnectionError):
    """LeRobot's error for I/O on a closed port."""


class Arm:
    """The physical SO-101: what each servo does, whatever the host believes."""

    def __init__(self) -> None:
        self.torque = dict.fromkeys(JOINT_NAMES, 0)
        self.position = dict.fromkeys(JOINT_NAMES, 0.0)
        self.position["gripper"] = 50.0
        self.silent: set[str] = set()  # servos that stopped answering
        self.unplugged = False  # the USB adapter is gone
        self.generation = 0  # a replug bumps it; older handles stay dead
        self.transfers = 0  # bus transactions attempted through any handle
        self.opens = 0  # port opens attempted through any handle
        self.blocked = threading.Event()  # a transfer is stuck in the driver
        self._flowing = threading.Event()
        self._flowing.set()
        self._brown_out: tuple[str, set[str]] | None = None
        # The manager's bus lock. Every transfer must happen while it is held:
        # the manager is the only bus owner, and two threads on one serial
        # port garble each other's packets.
        self.bus_lock: Any = None
        self.unlocked_transfers = 0
        self.health: dict[str, dict[str, int]] = {}  # register -> {servo: raw value}

    def register_value(self, register: str, motor: str) -> float:
        if register == "Torque_Enable":
            return self.torque[motor]
        if register in HEALTH_RAW:
            return self.health.get(register, {}).get(motor, HEALTH_RAW[register])
        return self.position[motor]

    @property
    def energized(self) -> set[str]:
        return {joint for joint, enabled in self.torque.items() if enabled}

    def wedge(self) -> None:
        """From now on every transfer blocks in the driver until release()."""
        self.blocked.clear()
        self._flowing.clear()

    def release(self) -> None:
        self._flowing.set()

    def replug(self) -> None:
        """The adapter is back under a new device node; open handles stay dead."""
        self.unplugged = False
        self.generation += 1

    def power_cycle(self) -> None:
        """Physical power was cut and restored: every servo boots limp."""
        self.torque = dict.fromkeys(JOINT_NAMES, 0)
        self.silent.clear()

    def heal(self) -> None:
        self.silent.clear()
        self.unplugged = False
        self._brown_out = None
        self.release()

    def brown_out_at(self, register: str, motors: Iterable[str]) -> None:
        """``motors`` go silent the first time a transfer names ``register``."""
        self._brown_out = (register, set(motors))

    def transfer(self, generation: int, addressed: Iterable[str], register: str) -> list[str]:
        """One bus transaction; returns the addressed servos that can answer."""
        self.transfers += 1
        if self.bus_lock is not None and not self.bus_lock.locked():
            self.unlocked_transfers += 1
        if self._brown_out is not None and self._brown_out[0] == register:
            self.silent.update(self._brown_out[1])
            self._brown_out = None
        if not self._flowing.is_set():
            self.blocked.set()
            if not self._flowing.wait(timeout=10):
                raise TimeoutError("the test never released the wedged transfer")
        if self.unplugged or generation != self.generation:
            # pyserial on a vanished adapter
            raise OSError(errno.EIO, "Input/output error")
        return [motor for motor in addressed if motor not in self.silent]


class FakeMotorsBus:
    """One LeRobot 0.6.1 FeetechMotorsBus handle on the arm."""

    def __init__(self, arm: Arm) -> None:
        self.arm = arm
        self.generation = arm.generation
        self.motors = {joint: None for joint in JOINT_NAMES}
        self.is_connected = False  # port_handler.is_open

    def _transfer(self, addressed: list[str], register: str) -> list[str]:
        if not self.is_connected:
            raise DeviceNotConnectedError("the port is not open")
        return self.arm.transfer(self.generation, addressed, register)

    def _require_replies(self, addressed: list[str], register: str) -> None:
        answered = self._transfer(addressed, register)
        silent = [motor for motor in addressed if motor not in answered]
        if silent:
            raise ConnectionError(
                f"Failed on {register!r} for {silent}: [TxRxResult] There is no status packet!"
            )

    def _motor_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        return [motors] if isinstance(motors, str) else list(motors)

    def connect(self, handshake: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("already connected")
        self.arm.opens += 1
        if self.arm.unplugged or self.generation != self.arm.generation:
            raise ConnectionError("could not open the port")
        self.is_connected = True
        if handshake:
            self._require_replies(list(self.motors), "ping")

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("the port is not open")
        if disable_torque:
            self.disable_torque(num_retry=5)  # raises -> closePort() never runs
        self.is_connected = False

    def configure_motors(self, *args: Any, **kwargs: Any) -> None:
        self._require_replies(list(self.motors), "configure")

    def write(
        self, register: str, motor: str, value: Any, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        self._require_replies([motor], register)
        if register == "Torque_Enable":
            self.arm.torque[motor] = int(value)

    def read(
        self, register: str, motor: str, *, normalize: bool = True, num_retry: int = 0
    ) -> float:
        self._require_replies([motor], register)
        return self.arm.register_value(register, motor)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._motor_list(motors):
            self.write("Torque_Enable", motor, 1, num_retry=num_retry)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._motor_list(motors):
            self.write("Torque_Enable", motor, 0, num_retry=num_retry)

    def sync_write(
        self, register: str, values: Any, *, normalize: bool = True, num_retry: int = 0
    ) -> None:
        targets = values if isinstance(values, dict) else dict.fromkeys(self.motors, values)
        for motor in self._transfer(list(targets), register):  # a broadcast: no replies
            if register == "Torque_Enable":
                self.arm.torque[motor] = int(targets[motor])
            elif register == "Goal_Position" and self.arm.torque[motor]:
                self.arm.position[motor] = float(targets[motor])

    def sync_read(
        self,
        register: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, float]:
        names = self._motor_list(motors)
        self._require_replies(names, register)
        return {motor: self.arm.register_value(register, motor) for motor in names}


class FakeSOFollower:
    """LeRobot 0.6.1 ``SO101Follower`` as FeetechBus builds it."""

    is_calibrated = True
    config = SimpleNamespace(
        position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32
    )

    def __init__(self, arm: Arm) -> None:
        self.bus = FakeMotorsBus(arm)
        self.cameras: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def get_observation(self) -> dict[str, float]:
        present = self.bus.sync_read("Present_Position")
        return {f"{motor}.pos": value for motor, value in present.items()}

    def get_action(self) -> dict[str, float]:
        return self.get_observation()

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        goal = {
            key.removesuffix(".pos"): float(value)
            for key, value in action.items()
            if key.endswith(".pos")
        }
        present = self.bus.sync_read("Present_Position")  # max_relative_target=5.0
        goal = {
            motor: min(max(value, present[motor] - 5.0), present[motor] + 5.0)
            for motor, value in goal.items()
        }
        self.bus.sync_write("Goal_Position", goal)
        return {f"{motor}.pos": value for motor, value in goal.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("not connected")
        self.bus.disconnect(disable_torque=True)


@pytest.fixture
def arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Arm]:
    """Real mode, with the fake SOFollower behind the follower's FeetechBus."""

    monkeypatch.setenv("SO101_MOCK", "0")
    monkeypatch.setenv("SO101_CAMERAS", "")
    follower_port = tmp_path / "follower-port"
    leader_port = tmp_path / "leader-port"
    follower_port.touch()
    leader_port.touch()
    monkeypatch.setenv("SO101_FOLLOWER_PORT", str(follower_port))
    monkeypatch.setenv("SO101_LEADER_PORT", str(leader_port))
    config.get_settings.cache_clear()
    reset_manager()
    physical = Arm()
    manager = get_manager()
    physical.bus_lock = manager._io_lock
    passive = Arm()
    passive.bus_lock = manager._io_lock
    bus = manager.follower().bus
    assert isinstance(bus, FeetechBus)

    def build_device(device_bus):
        if device_bus.port == str(follower_port):
            return FakeSOFollower(physical)
        if device_bus.port == str(leader_port):
            return FakeSOFollower(passive)
        raise AssertionError(f"Unexpected hardware path in fault harness: {device_bus.port}")

    monkeypatch.setattr(FeetechBus, "_build_modern_device", build_device)
    monkeypatch.setattr(FeetechBus, "_position_mode_value", lambda self: 0)
    yield physical
    physical.heal()  # never leave a thread blocked in the fake driver
    assert physical.unlocked_transfers == 0, "bus I/O ran outside the manager's bus lock"
    assert passive.unlocked_transfers == 0, "leader I/O ran outside the manager's bus lock"


@pytest.fixture
def client(arm: Arm, manager_clock, supervise) -> Iterator[TestClient]:
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client
        arm.heal()  # before the app's shutdown touches the bus


def _lease(client: TestClient) -> str:
    response = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 30})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _arm(client: TestClient, arm: Arm) -> str:
    lease_id = _lease(client)
    response = client.post("/api/control/arm", json={"lease_id": lease_id})
    assert response.status_code == 200, response.text
    assert arm.energized == ALL
    return lease_id


def _status(client: TestClient) -> dict[str, Any]:
    return client.get("/api/robot/status").json()


def _safety_state(client: TestClient) -> str:
    return client.get("/api/control/state").json()["safety_state"]


def _events(client: TestClient, kind: str) -> list[dict[str, Any]]:
    events = client.get("/api/events").json()["events"]
    return [event for event in events if event["type"] == kind]


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _recover(client: TestClient, arm: Arm) -> None:
    """Heal the bus, then take the deliberate C2 resets back to a clean idle arm."""

    arm.heal()
    if _status(client)["estopped"]:
        response = client.post("/api/robot/estop/reset", json=RESET)
        assert response.status_code == 200, response.text
    response = client.post("/api/robot/fault/reset", json=RESET)
    assert response.status_code == 200, response.text
    status = _status(client)
    assert status["torque_unconfirmed"] is False
    assert status["follower_connected"] is False
    assert _safety_state(client) == "disarmed"
    assert arm.energized == set()
    assert ownership.read_global_energized_intent(config.get_settings()) is None


class _Call(threading.Thread):
    """Run ``func`` on its own thread, the way a second request would."""

    def __init__(self, func: Callable[..., Any], *args: Any) -> None:
        super().__init__(daemon=True)
        self._func = func
        self._args = args
        self.error: BaseException | None = None
        self.start()

    def run(self) -> None:
        try:
            self._func(*self._args)
        except BaseException as exc:  # handed to the test
            self.error = exc


def _start_rollout(arm: Arm) -> None:
    manager = get_manager()
    manager.start_rollout("models/fault-injection", 30, operator_id=OPERATOR, policy=MockPolicy())
    assert _wait_for(lambda: arm.energized == ALL and manager.telemetry()["sample_seq"] > 0)


# --- failed torque-off ---------------------------------------------------------------


def test_server_write_ahead_precedes_enable_halt_retains_and_estop_clears(
    client: TestClient,
    arm: Arm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes_seen: list[dict[str, Any]] = []
    original = FeetechBus.set_torque

    def observed_set_torque(self: FeetechBus, enabled: bool) -> None:
        if enabled:
            intent = ownership.read_global_energized_intent(config.get_settings())
            assert intent is not None, "torque-enable ran before global energized intent"
            writes_seen.append(intent)
        original(self, enabled)

    monkeypatch.setattr(FeetechBus, "set_torque", observed_set_torque)
    _arm(client, arm)

    assert writes_seen
    halted = client.post("/api/robot/halt", json={"reason": "test hold"})
    assert halted.status_code == 200, halted.text
    assert ownership.read_global_energized_intent(config.get_settings()) is not None

    estopped = client.post("/api/robot/estop")
    assert estopped.status_code == 200, estopped.text
    assert arm.energized == set()
    assert ownership.read_global_energized_intent(config.get_settings()) is not None
    assert client.post("/api/robot/estop/reset", json=RESET).status_code == 200
    assert ownership.read_global_energized_intent(config.get_settings()) is None


def test_server_marker_write_failure_prevents_enable_and_latches_fault(
    client: TestClient,
    arm: Arm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ownership.GlobalEnergizedIntentError("injected marker write failure")

    monkeypatch.setattr(ownership, "write_global_energized_intent", refuse)
    response = client.post("/api/control/arm", json={"lease_id": _lease(client)})

    assert response.status_code == 409, response.text
    assert arm.energized == set()
    status = _status(client)
    assert status["fault"] is not None
    assert "refusing to enable follower torque" in status["fault"]


def test_blocked_intent_persistence_does_not_hold_io_lock_or_delay_estop_cut(
    client: TestClient,
    arm: Arm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    _arm(client, arm)
    entered = threading.Event()
    release = threading.Event()
    save_latches = manager.store.save_latches

    def blocked_save(latches: dict[str, Any], *, version: int) -> bool:
        entered.set()
        assert release.wait(timeout=5)
        return save_latches(latches, version=version)

    monkeypatch.setattr(manager.store, "save_latches", blocked_save)
    monkeypatch.setattr(ownership, "enable_intent_is_adopted", lambda *a, **k: False)
    prepare = _Call(manager._prepare_global_energized_intent, "blocked re-arm intent")
    assert entered.wait(timeout=3)
    estop = _Call(lambda: client.post("/api/robot/estop"))
    try:
        assert _wait_for(lambda: arm.energized == set()), (
            "E-stop torque cut waited behind blocked latch persistence"
        )
    finally:
        release.set()
        prepare.join(timeout=5)
        estop.join(timeout=5)
    assert not prepare.is_alive() and not estop.is_alive()
    assert prepare.error is None and estop.error is None


@pytest.mark.parametrize("mode", ["teleop", "recording", "rollout"])
def test_server_loop_enable_paths_write_and_retain_global_intent_through_halt(
    mode: str,
    client: TestClient,
    arm: Arm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = get_manager()
    if mode == "teleop":
        manager.start_teleop(30, operator_id=OPERATOR)
    elif mode == "recording":
        manager.start_recording(
            RecordingConfig(repo_id="test/global-intent", fps=30, camera_keys=[]),
            operator_id=OPERATOR,
        )
    else:
        _start_rollout(arm)
    assert _wait_for(lambda: arm.energized == ALL)

    intent = ownership.read_global_energized_intent(config.get_settings())
    assert intent is not None
    assert "RobotManager" in intent["purpose"] and mode in intent["purpose"]

    assert manager.soft_stop(f"test {mode} Halt") is True
    assert ownership.read_global_energized_intent(config.get_settings()) is not None
    assert client.post("/api/robot/estop").status_code == 200
    assert ownership.read_global_energized_intent(config.get_settings()) is not None
    assert client.post("/api/robot/estop/reset", json=RESET).status_code == 200
    assert ownership.read_global_energized_intent(config.get_settings()) is None


def test_safety_b2_repro_04_estop_never_reports_an_unconfirmed_torque_off_as_success(
    client: TestClient, arm: Arm
):
    """Converts safety repro 04. In v0.1 the first E-stop dropped the handle
    when LeRobot's disconnect raised. The second one then saw "not connected",
    did no bus I/O, and returned success with every servo still energized."""

    _arm(client, arm)
    arm.silent.update(JOINT_NAMES)  # a brown-out: no servo answers

    for attempt in (1, 2):
        transfers = arm.transfers
        response = client.post("/api/robot/estop")
        assert response.status_code == 503, response.text
        assert "cut physical power" in response.json()["detail"]
        assert arm.transfers > transfers, f"E-stop {attempt} must try the bus again"

    status = _status(client)
    assert status["estopped"] is True
    assert status["torque_unconfirmed"] is True
    # The port closes anyway (no leaked handle); each retry opens a fresh one.
    assert status["follower_connected"] is False
    assert arm.energized == ALL, "the report matches what the servos still do"

    # Resetting the E-stop leaves the fault and the unconfirmed torque latched.
    assert client.post("/api/robot/estop/reset", json=RESET).status_code == 200
    assert _safety_state(client) == "fault"
    assert _status(client)["torque_unconfirmed"] is True
    rearm = client.post("/api/control/arm", json={"lease_id": _lease(client)})
    assert rearm.status_code == 409, rearm.text

    # The fault reset retries a real torque-off, and stays 409 while it fails.
    transfers = arm.transfers
    response = client.post("/api/robot/fault/reset", json=RESET)
    assert response.status_code == 409, response.text
    assert arm.transfers > transfers
    assert arm.energized == ALL

    _recover(client, arm)


def test_safety_b2_one_silent_servo_cannot_keep_the_others_energized(client: TestClient, arm: Arm):
    """LeRobot's per-motor disable stops at the first servo that does not
    answer. The E-stop's broadcast reaches every servo that can hear it; the
    one it cannot confirm keeps torque_unconfirmed latched."""

    _arm(client, arm)
    arm.silent.add("shoulder_pan")  # first in LeRobot's per-motor order

    response = client.post("/api/robot/estop")

    assert response.status_code == 503, response.text
    assert arm.energized == {"shoulder_pan"}
    assert _status(client)["torque_unconfirmed"] is True
    _recover(client, arm)


def test_safety_b2_owner_disarm_reports_an_unconfirmed_torque_off(client: TestClient, arm: Arm):
    lease_id = _arm(client, arm)
    arm.silent.update(JOINT_NAMES)

    response = client.post("/api/control/disarm", json={"lease_id": lease_id})

    assert response.status_code == 409, response.text
    assert "cut physical power" in response.json()["detail"]
    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "fault"
    assert _status(client)["torque_unconfirmed"] is True
    assert arm.energized == ALL
    with pytest.raises(ControlRejected, match="disarmed"):
        get_manager().execute_approved_targets(lease_id, TARGET)
    _recover(client, arm)


def test_safety_b2_lease_release_reports_an_unconfirmed_torque_off(client: TestClient, arm: Arm):
    lease_id = _arm(client, arm)
    arm.silent.update(JOINT_NAMES)

    response = client.delete(f"/api/control/lease/{lease_id}")

    assert response.status_code == 409, response.text
    assert "cut physical power" in response.json()["detail"]
    state = client.get("/api/control/state").json()
    assert state["armed"] is False and state["lease"] is None
    assert state["safety_state"] == "fault"
    assert _status(client)["torque_unconfirmed"] is True
    assert arm.energized == ALL
    assert _events(client, "control.lease_released")
    _recover(client, arm)


def test_safety_b2_a_failed_command_cuts_torque_on_every_servo_that_answers(
    client: TestClient, arm: Arm
):
    """A command that dies on the bus gets the same torque-off as a stop. The
    old cleanup was LeRobot's per-motor disable, which stopped at the first
    silent servo. The healthy servos kept holding torque, and the status
    said torque_unconfirmed false."""

    lease_id = _arm(client, arm)
    arm.silent.add("shoulder_pan")

    with pytest.raises(ControlRejected, match="command failed"):
        get_manager().execute_approved_targets(lease_id, TARGET)

    assert arm.energized == {"shoulder_pan"}
    assert _status(client)["torque_unconfirmed"] is True
    assert _safety_state(client) == "fault"
    _recover(client, arm)


def test_bus_failure_cuts_torque_before_fault_persistence(
    client: TestClient, arm: Arm, monkeypatch: pytest.MonkeyPatch
):
    """A stalled SQLite write cannot leave an I/O-failed arm energized."""

    manager = get_manager()
    lease_id = _arm(client, arm)
    entered = threading.Event()
    release = threading.Event()
    original_save = manager.store.save_latches

    def blocked_save(latches: dict[str, Any], *, version: int) -> bool:
        entered.set()
        assert release.wait(timeout=5), "the test never released latch persistence"
        return original_save(latches, version=version)

    def failed_goal(_goal: dict[str, float]) -> dict[str, float]:
        raise ConnectionError("injected command write failure")

    monkeypatch.setattr(manager.store, "save_latches", blocked_save)
    monkeypatch.setattr(manager.follower(), "send_goal", failed_goal)
    call = _Call(manager.execute_approved_targets, lease_id, TARGET)
    try:
        assert entered.wait(timeout=3), "the failed command never reached latch persistence"
        assert arm.energized == set(), "category-0 torque removal waited behind SQLite"
    finally:
        release.set()
        call.join(timeout=5)
    assert not call.is_alive()
    assert isinstance(call.error, ControlRejected)
    _recover(client, arm)


def test_loop_failure_cuts_torque_before_blocking_error_log(
    client: TestClient, arm: Arm, monkeypatch: pytest.MonkeyPatch
):
    manager = get_manager()
    entered = threading.Event()
    release = threading.Event()
    original_error = robot_manager_module.log.error

    def blocked_log(message: str, *args: Any, **kwargs: Any) -> None:
        if "control loop failed" in message:
            entered.set()
            assert release.wait(timeout=5), "the test never released the error log"
        original_error(message, *args, **kwargs)

    monkeypatch.setattr(robot_manager_module.log, "error", blocked_log)
    monkeypatch.setattr(robot_manager_module.log, "exception", blocked_log)

    def failed_loop() -> None:
        with manager._io_lock:
            manager.follower().connect()
        assert arm.energized == ALL
        raise ConnectionError("injected loop failure")

    manager._start_thread(Mode.TELEOP, failed_loop, OPERATOR)
    worker = manager._thread
    assert worker is not None
    try:
        assert entered.wait(timeout=3), "the failed loop never reached error logging"
        assert arm.energized == set(), "category-0 torque removal waited behind logging"
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    _recover(client, arm)


def test_fault_reset_preserves_new_fault_during_retry(
    client: TestClient, arm: Arm, monkeypatch: pytest.MonkeyPatch
):
    manager = get_manager()
    _arm(client, arm)
    manager._set_fault("old fault", source="test")
    entered = threading.Event()
    release = threading.Event()
    original_retry = manager._retry_torque_off

    def gated_retry(reason: str, *, io_timeout: float = 0.5) -> bool:
        entered.set()
        assert release.wait(timeout=5), "the test never released the fault-reset retry"
        return original_retry(reason, io_timeout=io_timeout)

    monkeypatch.setattr(manager, "_retry_torque_off", gated_retry)
    reset = _Call(lambda: manager.clear_fault(operator_id=OPERATOR))
    assert entered.wait(timeout=3), "fault reset did not start its torque-off retry"
    manager._set_fault("newer encoder fault", source="test.concurrent")
    release.set()
    reset.join(timeout=5)

    assert not reset.is_alive()
    assert isinstance(reset.error, ControlConflict)
    assert manager.fault == "newer encoder fault"
    assert manager.status()["torque_unconfirmed"] is True
    assert ownership.read_global_energized_intent(config.get_settings()) is not None
    _recover(client, arm)


def test_safety_b2_a_failed_arm_cleans_up_under_the_bus_lock(client: TestClient, arm: Arm):
    """A servo browns out while arming reads the pose. The old cleanup
    disconnected outside the manager's bus lock, racing any stop on the same
    serial port. It also left the torque-off reported as confirmed when it
    was not. Now arming gets the same bounded, locked torque-off with
    read-back as every stop."""

    arm.brown_out_at("Present_Position", {"wrist_flex"})

    response = client.post("/api/control/arm", json={"lease_id": _lease(client)})

    assert response.status_code == 409, response.text
    assert "could not arm" in response.json()["detail"]
    assert arm.unlocked_transfers == 0
    assert _safety_state(client) == "fault"
    assert _status(client)["torque_unconfirmed"] is True, "wrist_flex cannot be read back"
    assert arm.energized == set()
    _recover(client, arm)


def test_safety_b2_a_failed_observe_cleans_up_under_the_bus_lock(client: TestClient, arm: Arm):
    arm.brown_out_at("Present_Position", {"wrist_flex"})

    response = client.post("/api/control/observe", json={"lease_id": _lease(client)})

    assert response.status_code == 409, response.text
    assert "could not read follower pose" in response.json()["detail"]
    assert arm.unlocked_transfers == 0
    assert _safety_state(client) == "fault"
    assert _status(client)["torque_unconfirmed"] is True
    assert arm.energized == set()
    _recover(client, arm)


def test_safety_b2_a_dead_adapter_is_released_after_an_acknowledged_power_cut(
    client: TestClient, arm: Arm
):
    """An unplugged adapter cannot carry a torque-off. Every stop still tries:
    the first through the open handle, which then closes, and each later one
    by reopening the port. Once the operator cuts power and acknowledges it,
    the fault reset releases the bus, and after a replug the next arm opens a
    fresh port instead of needing a service restart."""

    _arm(client, arm)
    arm.unplugged = True

    for _ in range(2):
        before = arm.transfers + arm.opens
        assert client.post("/api/robot/estop").status_code == 503
        assert arm.transfers + arm.opens > before, "every E-stop tries the bus again"
    assert client.post("/api/robot/estop/reset", json=RESET).status_code == 200
    response = client.post("/api/robot/fault/reset", json=RESET)
    assert response.status_code == 409, response.text
    assert "power_cut_acknowledged" in response.json()["detail"]

    arm.power_cycle()  # what the 409 asks for: the servos go limp
    response = client.post("/api/robot/fault/reset", json={**RESET, "power_cut_acknowledged": True})

    assert response.status_code == 200, response.text
    status = _status(client)
    assert status["torque_unconfirmed"] is False
    assert status["follower_connected"] is False
    released = _events(client, "safety.power_cut_acknowledged")
    assert released and released[0]["detail"]["ports_released"] == ["follower"]
    arm.replug()
    lease_id = _arm(client, arm)  # a fresh port: usable again without a restart
    assert client.post("/api/control/disarm", json={"lease_id": lease_id}).status_code == 200
    # The owner disarm is a Collapse: it lowers the arm, then turns torque off
    # (stop_state leaves "collapsing" once its done event is written).
    assert _wait_for(lambda: _status(client)["stop_state"] == "none")
    assert arm.energized == set()


def test_safety_b2_a_replugged_adapter_confirms_torque_off_without_an_acknowledgement(
    client: TestClient, arm: Arm
):
    """The handle open when the adapter vanished stays dead after a replug,
    but it was closed by the failed torque-off. The fault reset's retry opens
    a fresh port, confirms torque-off by read-back, and clears the fault with
    no power-cut acknowledgement."""

    _arm(client, arm)
    arm.unplugged = True
    assert client.post("/api/robot/estop").status_code == 503
    assert arm.energized == ALL
    arm.replug()
    assert client.post("/api/robot/estop/reset", json=RESET).status_code == 200

    response = client.post("/api/robot/fault/reset", json=RESET)

    assert response.status_code == 200, response.text
    assert arm.energized == set(), "the fresh port turned torque off on every servo"
    status = _status(client)
    assert status["torque_unconfirmed"] is False
    assert status["follower_connected"] is False
    assert _events(client, "safety.power_cut_acknowledged") == []


def test_safety_b2_the_port_release_waits_for_the_bus_and_keeps_the_fault_latched(
    client: TestClient, arm: Arm
):
    """While a driver call holds the bus, the retry cannot reopen the port and
    the unconfirmed bus cannot be released. The acknowledged reset then
    answers 503 with the fault still latched, rather than clearing it over a
    bus nobody confirmed. Once the bus is free, the replugged adapter's fresh
    port confirms torque-off on its own."""

    _arm(client, arm)
    arm.unplugged = True
    assert client.post("/api/robot/estop").status_code == 503
    arm.replug()
    assert client.post("/api/robot/estop/reset", json=RESET).status_code == 200
    arm.power_cycle()
    acknowledged = {**RESET, "power_cut_acknowledged": True}
    bus_lock = get_manager()._io_lock

    assert bus_lock.acquire(timeout=1)  # a wedged driver call holds the bus
    try:
        response = client.post("/api/robot/fault/reset", json=acknowledged)
    finally:
        bus_lock.release()

    assert response.status_code == 503, response.text
    assert "port release" in response.json()["detail"]
    assert _safety_state(client) == "fault"
    assert _status(client)["torque_unconfirmed"] is True
    response = client.post("/api/robot/fault/reset", json=acknowledged)
    assert response.status_code == 200, response.text
    assert _status(client)["follower_connected"] is False


# --- servo health on the real bus ------------------------------------------------------


def test_servo_health_critical_on_the_real_bus_collapses_and_latches(
    client: TestClient, arm: Arm, health_poll
):
    """FeetechBus.read_health -> monitor -> Collapse + fault, all under the bus lock."""

    _arm(client, arm)
    health_poll()
    health = client.get("/api/robot/health").json()["arms"]["follower"]
    assert health["active"] and health["level"] == "ok" and health["source"] == "real"
    assert health["motors"]["elbow_flex"]["sample"]["voltage_v"] == 5.0

    arm.health["Present_Voltage"] = {"elbow_flex": 38}  # 3.8 V: critical on the 7.4v band
    get_manager()._health["follower"].next_due = float("-inf")
    health_poll()

    events = _events(client, "safety.servo_health_critical")
    assert len(events) == 1 and "elbow_flex: supply 3.8 V" in events[0]["message"]
    assert _wait_for(lambda: _status(client)["stop_state"] == "none")
    assert arm.energized == set(), "the Collapse ended with torque off on every servo"
    assert _safety_state(client) == "fault"
    assert arm.unlocked_transfers == 0
    arm.health.clear()
    _recover(client, arm)


def test_a_silent_servo_fails_the_health_reads_until_critical(
    client: TestClient, arm: Arm, manager_clock, health_poll
):
    _arm(client, arm)
    arm.silent.add("gripper")  # sync_read needs every servo to answer
    limit = config.get_settings().health_max_consecutive_failures
    for poll in range(1, limit + 1):
        health_poll()
        manager_clock.advance(1.0)
        if poll < limit:
            view = client.get("/api/robot/health").json()["arms"]["follower"]
            assert view["level"] == "warn" and view["consecutive_failures"] == poll
            assert _safety_state(client) == "armed"
    assert _wait_for(lambda: _status(client)["stop_state"] == "none")
    status = _status(client)
    assert _safety_state(client) == "fault"
    # The Collapse could not read the arm either: torque was cut; the silent
    # gripper stays unconfirmed and the report says so.
    assert status["torque_unconfirmed"] is True and arm.energized == {"gripper"}
    _recover(client, arm)


# --- hung I/O ------------------------------------------------------------------------


def test_safety_b9_estop_answers_while_a_driver_call_is_wedged(client: TestClient, arm: Arm):
    """A command stuck in the driver holds the bus. The E-stop latches at once
    and answers within its bound as unconfirmed, never as success. When the
    late write finally lands, a confirmed torque-off follows it."""

    lease_id = _arm(client, arm)
    arm.wedge()
    command = _Call(get_manager().execute_approved_targets, lease_id, TARGET)
    assert arm.blocked.wait(timeout=5), "the command never reached the driver"

    started = time.monotonic()
    response = client.post("/api/robot/estop")
    elapsed = time.monotonic() - started

    assert response.status_code == 503, response.text
    assert "cut physical power" in response.json()["detail"]
    assert elapsed < 5.0
    status = _status(client)
    assert status["estopped"] is True
    assert status["torque_unconfirmed"] is True
    assert arm.energized == ALL

    arm.release()
    command.join(timeout=5)
    assert not command.is_alive()
    assert isinstance(command.error, ControlRejected)
    assert "revoked" in str(command.error)
    assert arm.energized == set(), "a confirmed torque-off follows the late write"
    assert _status(client)["torque_unconfirmed"] is False
    _recover(client, arm)


def test_safety_b2_a_watchdog_trip_during_a_wedged_call_is_unconfirmed(
    client: TestClient, arm: Arm, manager_clock, supervise
):
    """The deadman fires while a driver call holds the bus. The watchdog cannot
    reach the servos, so it latches the trip as unconfirmed instead of
    reporting the arm de-energized."""

    manager = get_manager()
    lease_id = _arm(client, arm)
    arm.wedge()
    command = _Call(manager.execute_approved_targets, lease_id, TARGET)
    assert arm.blocked.wait(timeout=5)

    manager_clock.advance(manager.watchdog_ms / 1000 + 0.1)
    supervise()

    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "fault"
    assert _status(client)["torque_unconfirmed"] is True
    trips = _events(client, "control.watchdog_timeout")
    assert trips and trips[0]["level"] == "critical"
    assert arm.energized == ALL

    arm.release()
    command.join(timeout=5)
    assert isinstance(command.error, ControlRejected)
    assert arm.energized == set()
    _recover(client, arm)


# --- loops ---------------------------------------------------------------------------


def test_safety_b2_a_rollout_that_loses_the_bus_races_an_estop_to_one_end(
    client: TestClient, arm: Arm
):
    """Whichever wins, the loop's own bus failure or the E-stop, the result is
    one idle manager with the E-stop latched and torque-off unconfirmed."""

    _start_rollout(arm)
    arm.silent.update(JOINT_NAMES)

    response = client.post("/api/robot/estop")

    assert response.status_code == 503, response.text
    manager = get_manager()
    assert _wait_for(lambda: manager.mode is Mode.IDLE)
    status = _status(client)
    assert status["estopped"] is True
    assert status["torque_unconfirmed"] is True
    assert arm.energized == ALL
    _recover(client, arm)


def test_safety_b2_a_rollout_wedged_in_the_driver(client: TestClient, arm: Arm):
    """A rollout stuck in a driver call: the E-stop answers unconfirmed within
    its bound. Once the call returns the loop exits, and the fault reset's
    retry reopens the port to confirm torque-off by read-back."""

    _start_rollout(arm)
    arm.wedge()
    assert arm.blocked.wait(timeout=5)

    response = client.post("/api/robot/estop")

    assert response.status_code == 503, response.text
    assert "cut physical power" in response.json()["detail"]
    assert _status(client)["torque_unconfirmed"] is True
    manager = get_manager()
    assert manager.mode is Mode.ROLLOUT, "the loop thread is still stuck in the driver"

    arm.release()
    assert _wait_for(lambda: manager.mode is Mode.IDLE)
    assert _status(client)["follower_connected"] is False
    _recover(client, arm)


# --- service shutdown ----------------------------------------------------------------


def test_safety_b2_shutdown_after_an_unconfirmed_stop_stays_under_the_bus_lock(
    client: TestClient, arm: Arm
):
    """Shutdown closed the ports without the bus lock. After an unconfirmed
    stop it put torque writes on a serial port the manager no longer
    serialized. Every teardown takes the lock with a timed acquire."""

    _arm(client, arm)
    arm.silent.update(JOINT_NAMES)
    assert client.post("/api/robot/estop").status_code == 503

    get_manager().shutdown()

    assert arm.unlocked_transfers == 0
    assert ownership.read_global_energized_intent(config.get_settings()) is not None


def test_safety_b2_shutdown_does_not_hang_behind_a_wedged_driver_call(client: TestClient, arm: Arm):
    lease_id = _arm(client, arm)
    arm.wedge()
    command = _Call(get_manager().execute_approved_targets, lease_id, TARGET)
    assert arm.blocked.wait(timeout=5)

    started = time.monotonic()
    get_manager().shutdown()
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"shutdown waited {elapsed:.1f} s on the wedged bus"
    assert _status(client)["torque_unconfirmed"] is True
    arm.release()
    command.join(timeout=5)
    assert isinstance(command.error, ControlRejected)
    assert arm.energized == set(), "the late write is followed by a confirmed torque-off"


def test_shutdown_late_disconnect_failure_republishes_global_intent(
    client: TestClient,
    arm: Arm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _arm(client, arm)
    manager = get_manager()
    follower = manager.follower()
    disconnect = follower.disconnect
    calls = 0

    def fail_second_disconnect() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            disconnect()  # Collapse's confirmed aggregate cut clears intent.
            return
        follower.bus.torque_state = "unconfirmed"  # type: ignore[attr-defined]
        raise TorqueOffUnconfirmed("follower", ["shoulder_pan"])

    monkeypatch.setattr(follower, "disconnect", fail_second_disconnect)

    manager.shutdown()

    assert calls >= 2
    assert manager.status()["torque_unconfirmed"] is True
    assert ownership.read_global_energized_intent(config.get_settings()) is not None
