"""Physics tasks: seeded layouts, env-state layout and the success tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from so101 import JOINT_NAMES  # noqa: E402
from so101.simulation.tasks import (  # noqa: E402
    LIFT_HEIGHT_M,
    OBJECT_REGION,
    READY_POSE,
    TASK_NAMES,
    ZONE_SEPARATION_M,
    make_task,
)
from so101.simulation.units import effective_limits  # noqa: E402

DT = 1.0 / 30.0


def test_registry_and_unknown_task():
    assert TASK_NAMES == ("reach", "pick_lift", "pick_place")
    with pytest.raises(ValueError, match="unknown sim task"):
        make_task("stack")


@pytest.mark.parametrize("name", TASK_NAMES)
def test_reset_is_deterministic_and_env_state_matches_its_names(name):
    task = make_task(name)
    world = task.make_world()
    first = task.reset(world, seed=7)
    state = task.env_state(world)
    assert state.dtype == np.float32 and state.shape == (task.env_state_dim,)
    assert first["env_state_names"] == list(task.env_state_names)
    again = task.reset(world, seed=7)
    assert again == first
    assert np.array_equal(task.env_state(world), state)
    other = task.reset(world, seed=8)
    assert other["layout"] != first["layout"]
    assert task.instruction and task.episode_steps() == round(task.episode_s * 30)


def test_ready_pose_is_inside_the_limits_and_clear_of_the_table():
    limits = effective_limits()
    for joint in JOINT_NAMES:
        low, high = limits[joint]
        assert low + 5.0 <= READY_POSE[joint] <= high - 5.0, joint
    task = make_task("pick_lift")
    world = task.make_world()
    task.reset(world, seed=0)
    site, _ = world.site_pose()
    assert site[2] > 0.12  # 8 cm over the cube's top face
    assert world.contacts()["arm_table"] == 0


def test_cube_layouts_stay_in_the_sampled_region():
    task = make_task("pick_place")
    world = task.make_world()
    for seed in range(20):
        info = task.reset(world, seed=seed)
        cube = np.array(info["layout"]["cube_m"])
        zone = np.array(info["layout"]["zone_m"])
        radius = float(np.hypot(*cube))
        assert OBJECT_REGION.radius_m[0] - 1e-3 <= radius <= OBJECT_REGION.radius_m[1] + 1e-3
        assert np.linalg.norm(cube - zone) >= ZONE_SEPARATION_M - 1e-3
        assert -45.0 <= info["layout"]["cube_yaw_deg"] <= 45.0


def test_pick_lift_needs_the_cube_held_up_for_half_a_second():
    task = make_task("pick_lift")
    world = task.make_world()
    task.reset(world, seed=1)
    position, quat = world.object_pose("cube")
    assert task.update(world, DT)["lift_m"] == pytest.approx(0.0, abs=2e-3)
    assert not task.success()
    # Hold the cube up without stepping physics: the hold timer is all that runs.
    world.set_object_pose("cube", position + [0.0, 0.0, LIFT_HEIGHT_M + 0.01], quat)
    world.forward()
    for step in range(15):
        metrics = task.update(world, DT)
        assert task.success() == (step == 14)
    assert metrics["lift_m"] == pytest.approx(LIFT_HEIGHT_M + 0.01, abs=1e-3)
    assert task.first_success_s == pytest.approx(16 * DT, abs=1e-4)
    summary = task.summary()
    assert summary["max_lift_m"] >= LIFT_HEIGHT_M


def test_pick_place_needs_a_lift_the_zone_the_table_and_a_release():
    task = make_task("pick_place")
    world = task.make_world()
    task.reset(world, seed=2)
    _, quat = world.object_pose("cube")
    zone = task.zone.copy()
    half = task.cube_half_height
    on_zone = np.array([zone[0], zone[1], half])

    # Pushed into the zone without ever being lifted: not placed.
    world.set_object_pose("cube", on_zone, quat)
    world.settle(0.2, rewind_clock=True)
    for _ in range(12):
        task.update(world, DT)
    assert not task.success()

    # Lifted once, then set down in the zone and let go: placed after 0.3 s.
    world.set_object_pose("cube", on_zone + [0.0, 0.0, 0.05], quat)
    world.forward()
    task.update(world, DT)
    world.set_object_pose("cube", on_zone, quat)
    world.settle(0.2, rewind_clock=True)
    results = [task.update(world, DT) for _ in range(10)]
    assert results[-1]["in_zone"] == 1.0 and results[-1]["released"] == 1.0
    assert task.success()
    assert results[7]["success"] == 0.0 and results[8]["success"] == 1.0


def test_reach_needs_a_dwell_inside_two_centimetres():
    task = make_task("reach")
    world = task.make_world()
    task.reset(world, seed=3)
    site, _ = world.site_pose()
    assert task.update(world, DT)["distance_m"] == pytest.approx(
        float(np.linalg.norm(site - task.target)), abs=1e-6
    )
    # Move the target onto the tool: success once 0.3 s have passed inside 2 cm.
    task.target = site.copy()
    outcomes = [task.update(world, DT)["success"] for _ in range(10)]
    assert outcomes.index(1.0) == 8
