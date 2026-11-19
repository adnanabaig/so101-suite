"""MODEL_REST_POSE: pure data, inside the limits, and physically a rest pose."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from so101 import JOINT_NAMES
from so101.simulation.rest_pose import (
    LIMIT_MARGIN,
    MODEL_REST_POSE,
    SETTLE_LIMIT,
    STOP_CLEARANCE,
    check_rest_pose,
)
from so101.simulation.units import effective_limits


def test_rest_pose_module_never_imports_mujoco():
    code = (
        "import sys; sys.modules['mujoco'] = None; "
        "from so101.simulation.rest_pose import MODEL_REST_POSE; print(len(MODEL_REST_POSE))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6"


def test_rest_pose_is_complete_and_inside_the_limits():
    assert list(MODEL_REST_POSE) == list(JOINT_NAMES)
    limits = effective_limits()
    for joint, value in MODEL_REST_POSE.items():
        low, high = limits[joint]
        margin = min(value - low, high - value)
        # Only the elbow uses the "short of the stop it settles onto" rule.
        assert margin >= (STOP_CLEARANCE if joint == "elbow_flex" else LIMIT_MARGIN), joint
    assert MODEL_REST_POSE["gripper"] <= 10.0


def test_model_rest_pose_passes_the_physics_check():
    pytest.importorskip("mujoco")
    report = check_rest_pose()
    assert report.ok, report.checks
    assert report.commanded_contacts == ()
    assert "lower_arm|shoulder" in report.settled_contacts
    elbow = report.joints["elbow_flex"]
    assert elbow.on_stop and STOP_CLEARANCE <= elbow.settle < SETTLE_LIMIT
    assert abs(elbow.settled - elbow.target) >= STOP_CLEARANCE
    for joint in ("shoulder_pan", "wrist_flex", "wrist_roll", "gripper"):
        assert report.joints[joint].settle < 0.2, joint
    assert report.joints["shoulder_lift"].settle < 1.0
    assert report.reach_m < 0.2 and report.height_m < 0.35
    robust = report.robustness
    assert robust["cases"] == 27 and robust["commanded_collisions"] == 0
    # Only the elbow's 1-degree window is sensitive to a 2-degree error.
    assert robust["worst_settle"]["shoulder_lift"] < 2.0
    assert robust["worst_settle"]["wrist_flex"] < 0.5
    assert robust["worst_settle"]["elbow_flex"] < 5.5
    assert report.as_dict()["ok"] is True


def test_the_former_bringup_example_is_not_a_valid_rest_pose_in_this_model():
    pytest.importorskip("mujoco")
    pose = dict(zip(JOINT_NAMES, (0.0, -95.0, 90.0, 70.0, 0.0, 0.0), strict=True))
    report = check_rest_pose(pose, perturbation_deg=0.0)
    assert not report.checks["collision_free"]
    assert not report.checks["margins"]  # gripper 0 sits on its closed stop
    assert report.robustness == {"perturbation_deg": 0.0, "cases": 0}


def test_check_rest_pose_needs_every_joint():
    with pytest.raises(ValueError, match="exactly the joints"):
        check_rest_pose({"shoulder_pan": 0.0})
