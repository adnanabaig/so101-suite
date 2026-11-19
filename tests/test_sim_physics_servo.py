"""STS3215 servo profile and the stop behaviours wave 2 builds on (design section 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so101 import JOINT_NAMES
from so101.simulation.servo import (
    STS3215_ACCELERATION_DEG_S2,
    STS3215_MAX_SPEED_DEG_S,
    ServoProfile,
)

DT = 1.0 / 240.0


def _run(profile: ServoProfile, limit: int = 10_000) -> tuple[int, float]:
    steps, peak = 0, 0.0
    while not profile.settled() and steps < limit:
        profile.step(DT)
        peak = max(peak, float(np.max(np.abs(profile.velocity))))
        steps += 1
    return steps, math.degrees(peak)


def test_register_maths():
    assert STS3215_ACCELERATION_DEG_S2 == pytest.approx(2232.4, abs=0.1)
    assert STS3215_MAX_SPEED_DEG_S == pytest.approx(270.3, abs=0.1)


def test_small_step_is_a_short_triangle():
    profile = ServoProfile(1)
    profile.reset([0.0])
    profile.set_goal([math.radians(0.5)])
    steps, peak = _run(profile)
    assert steps <= 6  # 25 ms at 240 Hz
    assert peak == pytest.approx(
        math.degrees(math.sqrt(math.radians(0.5) * math.radians(2232.4))), rel=0.1
    )
    assert profile.reference[0] == math.radians(0.5)  # lands exactly


def test_long_move_cruises_at_top_speed():
    profile = ServoProfile(1)
    profile.reset([0.0])
    profile.set_goal([math.radians(90.0)])
    steps, peak = _run(profile)
    assert peak == pytest.approx(STS3215_MAX_SPEED_DEG_S, rel=0.01)
    ideal = 90.0 / STS3215_MAX_SPEED_DEG_S + STS3215_MAX_SPEED_DEG_S / STS3215_ACCELERATION_DEG_S2
    assert steps * DT == pytest.approx(ideal, abs=0.04)


def test_goal_change_mid_move_brakes_and_returns():
    profile = ServoProfile(1)
    profile.reset([0.0])
    profile.set_goal([math.radians(90.0)])
    for _ in range(30):
        profile.step(DT)
    halted = profile.reference.copy()
    speed = profile.velocity[0]
    profile.set_goal(halted)
    furthest = halted[0]
    for _ in range(1000):
        if profile.settled():
            break
        profile.step(DT)
        furthest = max(furthest, profile.reference[0])
    assert profile.settled()
    overshoot = math.degrees(furthest - halted[0])
    assert overshoot == pytest.approx(
        math.degrees(speed**2 / (2 * math.radians(STS3215_ACCELERATION_DEG_S2))), rel=0.1
    )
    assert profile.reference[0] == halted[0]


def test_infinite_acceleration_is_a_plain_step():
    profile = ServoProfile(2, acceleration=math.inf)
    profile.reset([0.0, 0.0])
    profile.set_goal([1.0, -1.0])
    assert profile.step(DT) == pytest.approx([1.0, -1.0])
    with pytest.raises(ValueError):
        ServoProfile(1, acceleration=0.0)


# -- with physics -----------------------------------------------------------------
@pytest.fixture
def driver():
    pytest.importorskip("mujoco")
    from so101.simulation.servo import ServoDriver
    from so101.simulation.world import PhysicsWorld

    return ServoDriver(PhysicsWorld())


def test_estop_sags_from_an_extended_pose(driver):
    world = driver.world
    start = world.joint_array()
    driver.set_torque(False)
    driver.advance(0.5)
    sag = world.joint_array() - start
    assert sag[JOINT_NAMES.index("shoulder_lift")] > 15.0
    assert sag[JOINT_NAMES.index("elbow_flex")] > 10.0


def test_halt_holds_within_a_hundredth_of_the_threshold(driver):
    world = driver.world
    world.reset(joints={"shoulder_lift": -20.0, "elbow_flex": 30.0, "wrist_flex": 60.0})
    driver.sync()
    driver.hold()
    start = world.joint_array()
    driver.advance(10.0)
    assert np.max(np.abs(world.joint_array() - start)) < 0.05  # design: < 1 deg


def test_torque_off_tracks_the_arm_so_torque_on_cannot_jump(driver):
    world = driver.world
    driver.set_torque(False)
    driver.advance(1.0)
    sagged = world.joint_array()
    driver.set_torque(True)
    assert driver.goal_array() == pytest.approx(sagged, abs=1e-6)
    driver.advance(0.5)
    assert np.max(np.abs(world.joint_array() - sagged)) < 0.1


def test_collapse_stream_reaches_rest_and_respects_the_cap_on_average(driver):
    from so101.simulation.rest_pose import MODEL_REST_POSE

    world = driver.world
    rest = np.array([MODEL_REST_POSE[joint] for joint in JOINT_NAMES])
    goal = world.joint_array()
    cap, hz = 15.0, 30.0
    track = [world.joint_array()]
    for _ in range(int(9.0 * hz)):
        goal = goal + np.clip(rest - goal, -cap / hz, cap / hz)
        driver.command(goal)
        driver.advance(1.0 / hz)
        track.append(world.joint_array())
    positions = np.array(track)
    window = int(0.2 * hz)
    windowed = np.abs(positions[window:] - positions[:-window]).max() * hz / window
    assert windowed < cap * 1.1
    assert np.max(np.abs(world.joint_array() - rest)) < 0.3
    driver.set_torque(False)
    before = world.joint_array()
    driver.advance(2.0)
    assert np.max(np.abs(world.joint_array() - before)) < 3.0
