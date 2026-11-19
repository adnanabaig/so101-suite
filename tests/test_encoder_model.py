"""Encoder views use the shared MJCF frame, including gripper mapping."""

import pytest

from so101 import JOINT_NAMES
from so101.services.encoder_model import encoder_model
from so101.simulation.kinematics import forward_kinematics


def test_model_uses_cached_encoder_positions_without_clamping():
    joints = dict(zip(JOINT_NAMES, [20, -40, 55, -15, 35, 80], strict=True))
    result = encoder_model(
        {
            "observation": {f"{name}.pos": value for name, value in joints.items()},
            "observation_live": True,
            "sample_age_ms": 25,
        }
    )
    assert result["live"] and result["complete"]
    assert not result["physical_alignment_verified"]
    frames = {frame["name"]: frame["position_m"] for frame in forward_kinematics(joints)["frames"]}
    bodies = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
    for motor, body in zip(result["motors"], bodies, strict=True):
        assert motor["position_m"] == pytest.approx(frames[body], abs=1e-8)
    assert [motor["id"] for motor in result["motors"]] == list(range(1, 7))
    assert result["models"] and all(item["rgba"][3] == 1 for item in result["models"])


@pytest.mark.parametrize(
    "observation", [{}, {"shoulder_pan.pos": 2}, {name: float("nan") for name in JOINT_NAMES}]
)
def test_incomplete_data_never_draws_an_invented_encoder_pose(observation):
    result = encoder_model(
        {"observation": observation, "observation_live": True, "sample_age_ms": 0}
    )
    assert not result["live"] and not result["complete"]
    assert result["models"] == []


def test_old_and_idle_samples_are_labeled_as_last_known():
    observation = dict.fromkeys(JOINT_NAMES, 0)
    for live, age in ((False, 0), (True, 501), (True, -1)):
        result = encoder_model(
            {"observation": observation, "observation_live": live, "sample_age_ms": age}
        )
        assert not result["live"] and result["complete"]
        assert all(item["rgba"][3] == 0.25 for item in result["models"])
