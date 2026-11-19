"""PhysicsWorld: the compiled scene, suite-unit I/O, torque semantics, objects."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from so101 import JOINT_NAMES  # noqa: E402
from so101.simulation.units import UnitsError, effective_limits  # noqa: E402
from so101.simulation.world import (  # noqa: E402
    CAMERAS,
    MarkerSpec,
    ObjectSpec,
    PhysicsWorld,
    SceneSpec,
    WorldError,
    compiled_scene,
    quat_to_rpy_deg,
    rpy_deg_to_quat,
)

CUBE = ObjectSpec("cube", "box", (0.04,), mass_kg=0.03)
SCENE = SceneSpec(
    objects=(CUBE,), markers=(MarkerSpec("zone", "box", (0.1, 0.1, 0.002), (0, 1, 0, 0.3)),)
)


def test_scene_compiles_once_with_both_cameras_and_the_suite_joint_order():
    first, again = (
        compiled_scene(SCENE),
        compiled_scene(SceneSpec(objects=(CUBE,), markers=SCENE.markers)),
    )
    assert first is again  # cached per equal spec
    assert set(CAMERAS) <= set(first.index.cameras)
    model = first.model
    assert [model.joint(i).name for i in range(6)] == list(JOINT_NAMES)
    assert model.opt.timestep == pytest.approx(1.0 / 240.0)
    assert first.index.object_ids == ("cube",)


def test_scene_spec_validation():
    with pytest.raises(WorldError, match="unique"):
        SceneSpec(objects=(CUBE,), markers=(MarkerSpec("cube", "sphere", (0.01,), (1, 0, 0, 1)),))
    with pytest.raises(WorldError, match="timestep"):
        SceneSpec(timestep_s=0.05)


def test_reset_is_deterministic_and_holds_the_pose():
    world = PhysicsWorld(SCENE)
    pose = {"shoulder_pan": 15.0, "elbow_flex": 20.0, "gripper": 30.0}
    world.reset(7, joints=pose)
    first = world.observe()
    world.step(240)
    world.reset(7, joints=pose)
    assert world.observe()["joints"] == first["joints"]
    assert world.rng.integers(1 << 30) == np.random.default_rng(7).integers(1 << 30)
    assert first["joints"]["shoulder_pan"] == pytest.approx(15.0)
    assert first["torque_enabled"] is True
    world.step(480)  # holding: the arm barely moves under gravity
    assert max(abs(world.joint_array() - np.array(list(first["joints"].values())))) < 0.1


def test_targets_are_suite_units_clipped_to_the_effective_limits():
    world = PhysicsWorld()
    applied = world.set_targets({"shoulder_pan": 500.0, "gripper": -20.0})
    limits = effective_limits()
    # Limits are rounded to 0.01; ctrlrange (radians) is the final clip.
    assert applied["shoulder_pan"] == pytest.approx(limits["shoulder_pan"][1], abs=0.01)
    assert applied["gripper"] == pytest.approx(limits["gripper"][0], abs=1e-6)
    assert applied["elbow_flex"] == pytest.approx(0.0, abs=1e-6)  # partial: the rest is kept
    with pytest.raises(UnitsError, match="unknown"):
        world.set_targets({"elbow": 3.0})
    with pytest.raises(UnitsError, match="finite"):
        world.set_targets([0, 0, 0, 0, float("nan"), 0])


def test_observation_has_the_documented_shape():
    world = PhysicsWorld(SCENE)
    world.set_object_pose("cube", (0.2, 0.0, 0.02))
    obs = world.observe()
    assert set(obs) == {
        "time_s",
        "torque_enabled",
        "joints",
        "velocities",
        "targets",
        "effort_pct",
        "effort_nm",
        "gripper_site",
        "objects",
        "markers",
        "contacts",
    }
    for key in ("joints", "velocities", "targets", "effort_pct", "effort_nm"):
        assert list(obs[key]) == list(JOINT_NAMES)
    assert set(obs["objects"]["cube"]) == {
        "position_m",
        "quat_wxyz",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
    }
    assert set(obs["contacts"]) == {
        "total",
        "arm_self",
        "arm_table",
        "max_penetration_m",
        "objects",
    }


def test_torque_off_sags_and_torque_on_holds_without_a_jump():
    world = PhysicsWorld()
    start = world.joint_array()
    world.set_torque(False)
    assert world.torque_enabled is False
    world.step(world.steps_for(0.5))
    sag = world.joint_array() - start
    assert sag[JOINT_NAMES.index("shoulder_lift")] > 5.0  # the extended arm falls
    assert sag[JOINT_NAMES.index("elbow_flex")] > 5.0
    assert abs(sag[JOINT_NAMES.index("shoulder_pan")]) < 0.1  # vertical axis: no gravity
    velocities = world.velocity_array()
    assert velocities.shape == (6,)  # deg/s
    world.step(world.steps_for(1.5))
    resting = world.joint_array()
    world.set_torque(True)
    assert world.target_array() == pytest.approx(resting, abs=1e-6)  # targets := present
    world.step(world.steps_for(0.5))
    assert np.max(np.abs(world.joint_array() - resting)) < 0.1


def test_fork_is_independent_of_its_parent():
    world = PhysicsWorld(SCENE)
    twin = world.fork()
    twin.set_torque(False)
    twin.step(240)
    assert world.torque_enabled is True
    assert world.time_s == 0.0
    assert not np.allclose(twin.joint_array(), world.joint_array())


def test_a_dropped_cube_comes_to_rest_on_the_table():
    world = PhysicsWorld(SCENE)
    world.set_object_pose("cube", (0.22, 0.05, 0.05), rpy_deg_to_quat(0.0, 0.0, 30.0))
    world.step(world.steps_for(1.0))
    position, quat = world.object_pose("cube")
    assert position[2] == pytest.approx(0.02, abs=1.5e-3)  # resting on its face
    assert quat_to_rpy_deg(quat)[2] == pytest.approx(30.0, abs=2.0)
    summary = world.contacts()
    assert summary["objects"]["cube"]["table"] >= 1
    assert summary["objects"]["cube"]["grasped"] is False
    assert summary["max_penetration_m"] < 2e-3
    with pytest.raises(WorldError):
        world.object_pose("ball")
    world.set_marker("zone", (0.1, 0.1, 0.001))
    assert world.marker_position("zone") == pytest.approx([0.1, 0.1, 0.001])
