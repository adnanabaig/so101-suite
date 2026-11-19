"""Hardware abstraction in mock mode."""

from __future__ import annotations

import math

import pytest

from so101 import JOINT_NAMES
from so101.bringup import ownership
from so101.hardware.calibration import CalibrationResult, JointCalibration, calibrate
from so101.hardware.cameras import CameraManager, CameraUnavailableError
from so101.hardware.follower import SO101Follower
from so101.hardware.leader import SO101Leader
from so101.hardware.motors import FeetechBus, HardwareUnavailableError, MockBus, make_bus
from so101.hardware.safety import SafetyGuard, SafetyViolation
from so101.teleop.controller import TeleopController


@pytest.fixture(autouse=True)
def _adopt_low_level_real_bus_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module tests the adapter seam; ownership has dedicated regressions."""

    monkeypatch.setattr(ownership, "require_device_lock", lambda port: port)


def test_make_bus_is_mock():
    bus = make_bus("/dev/null", name="t")
    assert isinstance(bus, MockBus)


def test_follower_safety_clamps_goal():
    f = SO101Follower(guard=SafetyGuard())
    f.connect()
    cmd = f.send_goal({"shoulder_pan": 9999})
    # connect() seeds the limiter at the measured position, so the first frame
    # is both position-clamped and rate-limited.
    assert 0.0 < cmd["shoulder_pan"] <= 15.0
    f.disconnect()


def test_estop_blocks_commands():
    f = SO101Follower(guard=SafetyGuard())
    f.connect()
    f.guard.estop()
    try:
        f.send_goal({"gripper": 50})
        raise AssertionError("e-stop did not block")
    except SafetyViolation:
        pass
    f.disconnect()


def test_connect_does_not_clear_latched_estop():
    guard = SafetyGuard()
    guard.estop()
    follower = SO101Follower(guard=guard)
    with pytest.raises(SafetyViolation, match="E-STOP"):
        follower.connect()
    assert follower.guard.estopped
    assert follower.connected is False


def test_follower_primes_measured_hold_before_torque_enable():
    events = []

    class Bus:
        connected = False

        def connect(self):
            self.connected = True

        def disconnect(self):
            self.connected = False

        def read_positions(self):
            return {joint: float(index + 10) for index, joint in enumerate(JOINT_NAMES)}

        def write_positions(self, goal):
            events.append(("goal", dict(goal)))
            return dict(goal)

        def set_torque(self, enabled):
            events.append(("torque", enabled))

    follower = SO101Follower(guard=SafetyGuard())
    follower.bus = Bus()
    follower.connect()
    assert events[0][0] == "goal"
    assert events[1] == ("torque", True)


def test_standalone_halt_writes_fresh_present_and_closes_without_torque_off():
    events = []

    class Bus:
        connected = True
        torque_state = "on"

        def read_positions(self):
            return {
                "shoulder_pan": 12.0,
                "shoulder_lift": -20.0,
                "elbow_flex": 30.0,
                "wrist_flex": -10.0,
                "wrist_roll": 4.0,
                "gripper": 50.0,
            }

        def write_positions(self, goal):
            events.append(("goal", dict(goal)))
            return dict(goal)

        def disconnect_holding(self):
            events.append(("close", self.torque_state))
            self.connected = False

    follower = SO101Follower(guard=SafetyGuard())
    follower.bus = Bus()
    follower.last_goal["gripper"] = 10.0

    commanded = follower.halt_and_disconnect()

    assert commanded["shoulder_pan"] == 12.0
    assert commanded["gripper"] == 45.0
    assert events == [("goal", commanded), ("close", "on")]
    assert follower.connected is False
    assert follower.torque_state == "on"


def test_follower_connect_cleans_up_on_keyboard_interrupt():
    class Bus:
        connected = False

        def connect(self):
            self.connected = True

        def disconnect(self):
            self.connected = False

        def read_positions(self):
            return {joint: 0.0 for joint in JOINT_NAMES}

        def write_positions(self, goal):
            return dict(goal)

        def set_torque(self, _enabled):
            raise KeyboardInterrupt("interrupted during torque enable")

    follower = SO101Follower(guard=SafetyGuard())
    follower.bus = Bus()
    with pytest.raises(KeyboardInterrupt):
        follower.connect()
    assert follower.connected is False


def test_leader_action_has_all_joints():
    leader = SO101Leader()
    leader.connect()
    action = leader.action()
    assert set(action) == {f"{j}.pos" for j in JOINT_NAMES}
    leader.disconnect()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_safety_rejects_non_finite_commands(bad):
    guard = SafetyGuard()
    with pytest.raises(SafetyViolation, match="non-finite"):
        guard.clamp({"shoulder_pan": bad})


def test_safety_rate_limit_uses_dt_and_caps_stall_time():
    guard = SafetyGuard(max_step_deg=100.0, max_velocity_deg_s=10.0, max_dt_s=0.25)
    guard.reset(reference={"shoulder_pan": 0.0})

    first = guard.clamp({"shoulder_pan": 100.0}, dt=0.2)
    assert first["shoulder_pan"] == pytest.approx(2.0)

    # A one-second stall only authorizes max_dt_s worth of movement.
    second = guard.clamp({"shoulder_pan": 100.0}, dt=1.0)
    assert second["shoulder_pan"] == pytest.approx(4.5)


def test_safety_rate_limit_tracks_driver_accepted_command():
    guard = SafetyGuard(max_step_deg=100.0, max_velocity_deg_s=10.0)
    guard.seed({"shoulder_pan": 0.0})
    assert guard.clamp({"shoulder_pan": 100.0}, dt=0.1)["shoulder_pan"] == 1.0
    guard.record_actual_command({"shoulder_pan": 0.2})
    assert guard.clamp({"shoulder_pan": 100.0}, dt=0.1)["shoulder_pan"] == 1.2


def test_real_mode_builds_real_bus_without_mock_fallback(monkeypatch):
    from so101.config import get_settings

    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    bus = make_bus("/dev/not-present", name="follower")
    assert isinstance(bus, FeetechBus)
    assert not isinstance(bus, MockBus)


def test_real_camera_failure_never_substitutes_synthetic_frames(monkeypatch):
    from so101.config import get_settings

    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    manager = CameraManager(["unavailable"])
    monkeypatch.setattr(
        "so101.hardware.cameras._CvCamera",
        lambda _camera_id: (_ for _ in ()).throw(RuntimeError("not connected")),
    )

    with pytest.raises(CameraUnavailableError, match="no mock stream was substituted"):
        manager.open()


def test_missing_real_driver_fails_closed(monkeypatch):
    bus = FeetechBus("/dev/not-present", 1_000_000, "follower")

    def missing():
        raise ModuleNotFoundError("modern")

    monkeypatch.setattr(bus, "_build_modern_device", missing)
    with pytest.raises(HardwareUnavailableError, match="no legacy or mock"):
        bus.connect()


def test_modern_lerobot_adapter_preserves_order_and_units(monkeypatch):
    class FakeMotorBus:
        def __init__(self):
            self.torque = False
            self.connected = False
            self.motors = {joint: object() for joint in JOINT_NAMES}
            self.writes = []

        def connect(self):
            self.connected = True

        def configure_motors(self):
            pass

        def write(self, register, motor, value):
            self.writes.append((register, motor, value))

        def enable_torque(self):
            self.torque = True

        def disable_torque(self):
            self.torque = False

    class FakeFollower:
        def __init__(self):
            from types import SimpleNamespace

            self.bus = FakeMotorBus()
            self.is_calibrated = True
            self.sent = None
            self.cameras = {}
            self.config = SimpleNamespace(
                position_p_coefficient=16,
                position_i_coefficient=0,
                position_d_coefficient=32,
            )

        @property
        def is_connected(self):
            return self.bus.connected

        def disconnect(self):
            self.bus.connected = False

        def get_observation(self):
            # Deliberately reverse insertion order; the adapter must canonicalize.
            return {
                f"{joint}.pos": float(index)
                for index, joint in reversed(list(enumerate(JOINT_NAMES)))
            }

        def send_action(self, action):
            self.sent = action
            return {**action, "shoulder_pan.pos": 7.0}

    fake = FakeFollower()
    monkeypatch.setattr(ownership, "require_device_lock", lambda port: port)
    bus = FeetechBus("/dev/fake", 1_000_000, "follower")
    monkeypatch.setattr(bus, "_build_modern_device", lambda: fake)
    monkeypatch.setattr(bus, "_position_mode_value", lambda: 3)

    bus.connect()
    assert fake.bus.torque is False
    assert list(bus.read_positions()) == JOINT_NAMES
    assert bus.read_positions()["gripper"] == 5.0
    commanded = bus.write_positions({"gripper": 75.0, "shoulder_pan": 12.0})
    assert list(fake.sent) == ["shoulder_pan.pos", "gripper.pos"]
    assert fake.sent["gripper.pos"] == 75.0
    assert commanded == {"shoulder_pan": 7.0, "gripper": 75.0}
    monkeypatch.setattr(ownership, "require_enable_intent", lambda _port: None)
    bus.set_torque(True)
    assert fake.bus.torque
    bus.disconnect()


def test_pinned_leader_shape_connects_without_cameras_attribute(monkeypatch):
    class LeaderBus:
        def __init__(self):
            self.connected = False
            self.torque = False
            self.motors = {joint: object() for joint in JOINT_NAMES}

        def connect(self):
            self.connected = True

        def configure_motors(self):
            pass

        def write(self, *_args):
            pass

        def disable_torque(self):
            self.torque = False

        def enable_torque(self):
            self.torque = True

    class Leader:
        def __init__(self):
            self.bus = LeaderBus()
            self.is_calibrated = True

        @property
        def is_connected(self):
            return self.bus.connected

        def get_action(self):
            return {f"{joint}.pos": 0.0 for joint in JOINT_NAMES}

        def disconnect(self):
            self.bus.connected = False

    leader = Leader()
    bus = FeetechBus("/dev/fake", 1_000_000, "leader")
    monkeypatch.setattr(bus, "_build_modern_device", lambda: leader)
    monkeypatch.setattr(bus, "_position_mode_value", lambda: 3)
    bus.connect()
    assert bus.connected
    assert set(bus.read_positions()) == set(JOINT_NAMES)
    assert leader.bus.torque is False
    bus.disconnect()


def test_modern_bus_cleans_partial_serial_connect_on_keyboard_interrupt(monkeypatch):
    class RawBus:
        connected = False
        torque = True

        def connect(self):
            self.connected = True
            raise KeyboardInterrupt("serial handshake interrupted")

        def disable_torque(self):
            self.torque = False

        def disconnect(self, disable_torque=True):
            if disable_torque:
                self.torque = False
            self.connected = False

    class Device:
        def __init__(self):
            self.bus = RawBus()
            self.cameras = {}

        @property
        def is_connected(self):
            return self.bus.connected

        def disconnect(self):
            self.bus.disconnect()

    device = Device()
    bus = FeetechBus("/dev/fake", 1_000_000, "follower")
    monkeypatch.setattr(bus, "_build_modern_device", lambda: device)
    with pytest.raises(KeyboardInterrupt):
        bus.connect()
    assert device.bus.connected is False
    assert device.bus.torque is False


def test_teleop_step_separates_desired_and_commanded_action():
    class FixedLeader:
        def connect(self):
            pass

        def disconnect(self):
            pass

        def action(self):
            return {
                f"{joint}.pos": (100.0 if joint == "shoulder_pan" else 0.0) for joint in JOINT_NAMES
            }

    follower = SO101Follower(guard=SafetyGuard(max_step_deg=5.0))
    with TeleopController(FixedLeader(), follower) as controller:
        step = controller.step()

    assert step.desired_action["shoulder_pan.pos"] == 100.0
    assert step.action["shoulder_pan.pos"] == pytest.approx(5.0)


def test_teleop_connect_cleans_both_arms_on_base_exception():
    events = []

    class Leader:
        def connect(self):
            events.append("leader.connect")

        def disconnect(self):
            events.append("leader.disconnect")

    class Follower:
        def connect(self):
            events.append("follower.connect")
            raise KeyboardInterrupt("caller window")

        def disconnect(self):
            events.append("follower.disconnect")

    controller = TeleopController(Leader(), Follower())
    with pytest.raises(KeyboardInterrupt):
        controller.connect()
    assert events == [
        "leader.connect",
        "follower.connect",
        "follower.disconnect",
        "leader.disconnect",
    ]


@pytest.mark.parametrize("fps", [0, -1, 241, True, 30.0])
def test_teleop_rejects_an_unsafe_control_rate(fps):
    with pytest.raises(ValueError, match="integer between 1 and 240"):
        TeleopController(object(), object(), fps=fps)


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -0.01])
def test_teleop_rejects_an_unsafe_duration(duration):
    controller = TeleopController(object(), object())
    with pytest.raises(ValueError, match="finite and non-negative"):
        list(controller.run(duration))


def test_real_calibration_delegates_to_lerobot_format(monkeypatch):
    bus = FeetechBus("/dev/fake", 1_000_000, "follower")
    calls = []

    def calibrate_timed(seconds, progress=None, cancel_event=None):
        calls.append(seconds)
        return {
            joint: ((0.0, 100.0) if joint == "gripper" else (-45.0, 45.0)) for joint in JOINT_NAMES
        }

    monkeypatch.setattr(bus, "calibrate_timed", calibrate_timed)
    result = calibrate(bus, "follower", sample_seconds=3.0)

    assert calls == [3.0]
    assert result.schema_version == 2
    assert result.device_id == "so101-follower"
    assert result.port == "/dev/fake"
    assert result.backend == "lerobot"
    assert result.joints["shoulder_pan"].offset == 0.0
    assert result.joints["gripper"].offset == 50.0


def test_follower_safety_intersects_saved_calibration():
    CalibrationResult(
        arm="follower",
        joints={"shoulder_pan": JointCalibration(min=-20.0, max=30.0, offset=5.0)},
    ).save()

    follower = SO101Follower()
    assert follower.guard.limits["shoulder_pan"] == (-20.0, 30.0)
    # An omitted joint retains the conservative built-in limit.
    assert follower.guard.limits["elbow_flex"] == (-100.0, 99.0)


def test_follower_ignores_calibration_for_another_device():
    CalibrationResult(
        arm="follower",
        device_id="different-arm",
        joints={"shoulder_pan": JointCalibration(min=-2.0, max=3.0, offset=0.5)},
    ).save()

    follower = SO101Follower()
    assert follower.guard.limits["shoulder_pan"] == (-110.0, 110.0)
