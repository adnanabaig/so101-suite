"""Pose visualization reads the installed reference without motor writes."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.services.app import create_app
from so101.services.calibration_pose import read_calibration_pose
from so101.services.routers import calibration as router


def reference():
    return {
        joint: {"id": index, "homing_offset": 20, "range_min": 100, "range_max": 3900}
        for index, joint in enumerate(JOINT_NAMES, 1)
    }


class ReadBus:
    def __init__(self):
        self.overrides = {}
        self.reads = []

    def read(self, register, joint, **kwargs):
        self.reads.append((register, joint))
        return self.overrides.get(
            (register, joint),
            {
                "Torque_Enable": 0,
                "Homing_Offset": 20,
                "Min_Position_Limit": 100,
                "Max_Position_Limit": 3900,
                "Present_Position": 2000,
            }[register],
        )


def test_all_six_joint_values_come_from_installed_encoder_reference():
    bus = ReadBus()
    pose = read_calibration_pose(bus, reference())
    assert pose["observation"]["shoulder_lift.pos"] == 0
    assert pose["observation"]["gripper.pos"] == 50
    assert len(pose["joints"]) == 6
    assert all(not item["issues"] for item in pose["joints"].values())
    assert all(
        register
        in {
            "Torque_Enable",
            "Present_Position",
            "Homing_Offset",
            "Min_Position_Limit",
            "Max_Position_Limit",
        }
        for register, _ in bus.reads
    )


@pytest.mark.parametrize(
    "register,value",
    [
        ("Present_Position", -20),
        ("Present_Position", float("nan")),
        ("Homing_Offset", 0),
        ("Max_Position_Limit", 4095),
    ],
)
def test_bad_reference_or_sample_never_becomes_a_valid_visual_angle(register, value):
    bus = ReadBus()
    bus.overrides[register, "wrist_roll"] = value
    result = read_calibration_pose(bus, reference())
    assert "wrist_roll.pos" not in result["observation"]
    assert result["joints"]["wrist_roll"]["issues"]


def test_energized_arm_is_refused_without_changing_torque():
    bus = ReadBus()
    bus.overrides["Torque_Enable", "shoulder_pan"] = 1
    with pytest.raises(ValueError, match="already be OFF"):
        read_calibration_pose(bus, reference())
    assert bus.reads == [("Torque_Enable", "shoulder_pan")]


def test_foxglove_uses_only_the_latest_follower_read_and_marks_old_reads_stale(monkeypatch):
    import time

    from so101.services import calibration_pose
    from so101.services.routers import mcap

    monkeypatch.setattr(calibration_pose, "_poses", {})
    manager = SimpleNamespace(telemetry=lambda: {"t": 0, "observation": {}})
    monkeypatch.setattr(mcap, "get_manager", lambda: manager)
    sample = {
        "observation": dict.fromkeys(JOINT_NAMES, 0.0),
        "complete": True,
        "captured_at": time.time(),
        "calibration_sha256": "reference",
    }
    calibration_pose.remember_pose("leader", sample)
    assert not mcap.model_snapshot()["complete"]
    calibration_pose.remember_pose("follower", sample)
    assert mcap.model_snapshot()["complete"]
    assert mcap.model_snapshot()["source"] == "read_only_pose"
    sample["captured_at"] -= 10
    calibration_pose.remember_pose("follower", sample)
    assert not mcap.model_snapshot()["live"]
    assert mcap.model_snapshot()["models"]


def test_pose_read_requires_explicit_confirmation_and_unknown_arm_is_refused(monkeypatch):
    monkeypatch.setattr(router, "get_manager", lambda: SimpleNamespace(settings=get_settings()))
    client = TestClient(create_app())
    assert client.post("/api/calibration/pose/follower", json={}).status_code == 422
    assert (
        client.post(
            "/api/calibration/pose/follower", json={"confirm_supported_torque_off": False}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/calibration/pose/unknown", json={"confirm_supported_torque_off": True}
        ).status_code
        == 404
    )
    response = client.post(
        "/api/calibration/pose/leader", json={"confirm_supported_torque_off": True}
    )
    assert response.status_code == 200
    assert response.json()["backend"] == "mock"
    assert all(item["issues"] for item in response.json()["joints"].values())
