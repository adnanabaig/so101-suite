from types import SimpleNamespace

import pytest

from so101 import JOINT_NAMES
from so101.hardware.follower import SO101Follower
from so101.hardware.motors import FeetechBus
from so101.hardware.safety import SafetyGuard


def test_native_target_window_is_scoped_and_restored_after_failure():
    bus = FeetechBus("fake", 1000000, "follower")
    bus._device = SimpleNamespace(config=SimpleNamespace(max_relative_target=5.0))
    with pytest.raises(RuntimeError, match="test failure"):
        with bus.leader_teleop_targets():
            assert bus._relative_target_limit is None
            assert bus._device.config.max_relative_target is None
            raise RuntimeError("test failure")
    assert bus._relative_target_limit == 5.0
    assert bus._device.config.max_relative_target == 5.0


def test_window_restored_when_native_device_is_created_inside_profile():
    bus = FeetechBus("fake", 1000000, "follower")
    with bus.leader_teleop_targets():
        bus._device = SimpleNamespace(config=SimpleNamespace(max_relative_target=None))
    assert bus._device.config.max_relative_target == 5.0


def test_teleop_slew_preserves_stricter_joint_limits_and_restores_on_exit():
    follower = SO101Follower(guard=SafetyGuard(max_velocity_deg_s={"shoulder_lift": 10.0}))
    previous = follower.guard.max_velocity_deg_s
    with pytest.raises(RuntimeError, match="test failure"):
        with follower.leader_teleop_profile():
            assert follower.guard.max_velocity_deg_s["shoulder_lift"] == 10.0
            assert follower.guard.max_velocity_deg_s["elbow_flex"] == 60.0
            follower.guard.seed(dict.fromkeys(JOINT_NAMES, 0.0))
            safe = follower.guard.clamp({"shoulder_lift": 100, "elbow_flex": 99}, dt=1 / 30)
            assert safe == {"shoulder_lift": pytest.approx(1 / 3), "elbow_flex": 2.0}
            raise RuntimeError("test failure")
    assert follower.guard.max_velocity_deg_s is previous


def test_feedback_relative_window_cannot_pin_native_target_under_static_load():
    follower = SO101Follower(guard=SafetyGuard())
    follower.bus = FeetechBus("fake", 1000000, "follower")
    config = SimpleNamespace(max_relative_target=5.0)
    follower.bus._device = SimpleNamespace(config=config)
    follower.guard.seed(dict.fromkeys(JOINT_NAMES, 0.0))
    with follower.leader_teleop_profile():
        commands = []
        for _ in range(5):
            command = follower.guard.clamp({"shoulder_lift": 30}, dt=1 / 30)
            if config.max_relative_target is not None:
                command["shoulder_lift"] = min(command["shoulder_lift"], config.max_relative_target)
            follower.guard.record_actual_command(command)
            commands.append(command["shoulder_lift"])
        assert commands == [2, 4, 6, 8, 10]
        assert all(
            later - earlier <= 2 for earlier, later in zip(commands, commands[1:], strict=False)
        )
    assert config.max_relative_target == 5.0
