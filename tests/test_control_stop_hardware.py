"""The hardware layer under Halt and Collapse (design doc §5).

- ``bus_torque_state``: C2 torque truth for any bus, never assuming "off".
- ``SO101Follower.hold`` / ``resume_hold``: the Halt's hold write, and taking
  over a hold without a torque or goal write.
- ``resolve_collapse_rest_pose``: the bring-up resolver with the model-derived
  default, clamped to SafetyGuard ∩ MJCF limits.
"""

from __future__ import annotations

import pytest

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.hardware import rest_pose as rp
from so101.hardware.follower import SO101Follower
from so101.hardware.motors import bus_torque_state
from so101.hardware.safety import DEFAULT_LIMITS, SafetyViolation
from so101.simulation.rest_pose import MODEL_REST_POSE


def test_a_bus_that_cannot_say_counts_as_unconfirmed():

    class Silent:
        connected = True

    class Raising:
        @property
        def torque_state(self) -> str:
            raise OSError("gone")

    class Flag:
        torque_enabled = True

    assert bus_torque_state(Silent()) == "unconfirmed"
    assert bus_torque_state(Raising()) == "unconfirmed"
    assert bus_torque_state(Flag()) == "on"


# --- follower helpers ------------------------------------------------------------


def test_hold_writes_the_goal_straight_and_records_it():
    follower = SO101Follower()
    follower.connect()
    try:
        far = {"shoulder_pan": 40.0}  # beyond one guard step: a hold is not rate-limited
        commanded = follower.hold(far)
        assert commanded == far
        assert follower.last_goal["shoulder_pan"] == 40.0
        assert follower.bus._goal["shoulder_pan"] == 40.0
    finally:
        follower.disconnect()


def test_resume_hold_takes_over_without_writing():
    follower = SO101Follower()
    follower.connect()
    try:
        goal_before = dict(follower.bus._goal)
        measured = follower.resume_hold()
        assert set(measured) == set(JOINT_NAMES)
        assert follower.bus._goal == goal_before
    finally:
        follower.disconnect()


def test_resume_hold_refuses_an_arm_that_is_not_holding_or_an_estop():
    follower = SO101Follower()
    follower.connect(enable_torque=False)
    try:
        with pytest.raises(RuntimeError, match="not holding"):
            follower.resume_hold()
        follower.guard.estop()
        with pytest.raises(SafetyViolation):
            follower.resume_hold()
    finally:
        follower.disconnect()


def test_the_follower_reports_its_bus_torque_state():
    follower = SO101Follower()
    assert follower.torque_state == "off"
    follower.connect()
    assert follower.torque_state == "on"
    follower.disconnect()
    assert follower.torque_state == "off"


# --- the collapse rest pose ---------------------------------------------------------


def test_collapse_uses_the_model_pose_when_nothing_is_set():
    resolution = rp.resolve_collapse_rest_pose(get_settings())

    assert resolution.source == "default" and resolution.placeholder is False
    assert resolution.joints == MODEL_REST_POSE
    # The bring-up views resolve exactly as Collapse does.
    assert rp.resolve_for_settings(get_settings()) == resolution


def test_collapse_clamps_every_source_to_safety_and_model_limits(monkeypatch):
    monkeypatch.setenv("SO101_REST_POSE", "0,-95,99,-85,0,5")  # elbow past the model's 96.83
    get_settings.cache_clear()
    model = rp.model_joint_limits()

    resolution = rp.resolve_collapse_rest_pose(get_settings())

    assert resolution.source == "setting"
    assert resolution.joints["elbow_flex"] == model["elbow_flex"][1]
    tight = rp.resolve_collapse_rest_pose(get_settings(), limits={"shoulder_lift": (-50.0, 50.0)})
    assert tight.joints["shoulder_lift"] == -50.0


def test_a_recorded_pose_wins(tmp_path):
    settings = get_settings()
    recorded = {**MODEL_REST_POSE, "wrist_flex": -80.0}
    rp.save_rest_pose(
        settings,
        recorded,
        follower_id=settings.follower_id,
        calibration_digest=None,
        backend="mock",
    )

    resolution = rp.resolve_collapse_rest_pose(settings)

    assert resolution.source == "recorded" and resolution.joints["wrist_flex"] == -80.0


def test_without_model_limits_the_clamp_is_the_guard_alone(monkeypatch):
    def unreadable():
        raise OSError("vendored model missing")

    monkeypatch.setattr(rp, "model_joint_limits", unreadable)

    resolution = rp.resolve_collapse_rest_pose(get_settings())

    assert resolution.joints == MODEL_REST_POSE
    assert any("model joint limits unavailable" in w for w in resolution.warnings)


def test_ranges_that_do_not_overlap_cannot_resolve(monkeypatch):
    monkeypatch.setattr(rp, "model_joint_limits", lambda: {"elbow_flex": (150.0, 160.0)})

    with pytest.raises(rp.RestPoseError, match="do not overlap"):
        rp.resolve_collapse_rest_pose(get_settings())


def test_intersect_limits_keeps_unmentioned_joints():
    limits = rp.intersect_limits(DEFAULT_LIMITS, {"gripper": (10.0, 90.0)})
    assert limits["gripper"] == (10.0, 90.0)
    assert limits["shoulder_pan"] == DEFAULT_LIMITS["shoulder_pan"]
