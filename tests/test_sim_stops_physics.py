"""The three stops under gravity in MuJoCo, through RobotManager (design doc §5 "Staging").

- E-stop sags: actuation is cut and the arm falls.
- Halt drifts less than 1° over 10 s, from a still arm and once it has caught
  a moving one (the catch's brief excursion is reported and bounded).
- Collapse stays within its speed cap and ends at the rest pose with actuation off.

``SO101_BACKEND=sim`` routes the follower through ``make_bus`` to the simulated
STS3215 arm (``so101.simulation.devices``); the production RobotManager,
SO101Follower and SafetyGuard run unchanged. Lockstep physics
(``SO101_SIM_REALTIME=0``): every torque-on goal write simulates one 1/30 s
control period, and ``SimRuntime.advance`` moves time by hand. The manager runs
on the injected clock, so the 30 Hz descent paces itself in fake time and the
whole module takes about a second. Seeded: task ``reach``, seed 7, whose start
pose is the task's READY pose (shoulder_lift -58, elbow 32, wrist_flex 83).

The speed cap is checked twice. The commanded setpoints move every joint at
most ``speed / 30`` per tick, exactly. The measured motion is checked the way
the bring-up HIL probe checks a real arm: joint speed over 0.25 s windows,
within 10 % of the cap. Within one control period the servo model closes each
goal step along its STS3215 acceleration profile, so instantaneous joint speed
peaks near 3x the cap (reported, not asserted; docs/control-plane-safety.md).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from so101 import JOINT_NAMES, config  # noqa: E402
from so101.bringup.hil import SPEED_TOLERANCE, SPEED_WINDOW_S  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    RobotManager,
    StopKind,
    get_manager,
    reset_manager,
)
from so101.services.store import get_store  # noqa: E402
from so101.simulation import devices  # noqa: E402
from so101.simulation.rest_pose import MODEL_REST_POSE, SETTLE_LIMIT  # noqa: E402

OPERATOR = "physics-operator"
SEED = 7
SPEED = 15.0
RATE = 30.0
ARM = [joint for joint in JOINT_NAMES if joint != "gripper"]
ARM_SLOTS = [JOINT_NAMES.index(joint) for joint in ARM]
REST = np.array([MODEL_REST_POSE[joint] for joint in JOINT_NAMES])

# What the tests measured, for the record (printed with ``pytest -s``).
MEASURED: dict[str, Any] = {}


@pytest.fixture
def sim_env(monkeypatch):
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", "sim")
    monkeypatch.setenv("SO101_SIM_REALTIME", "0")
    monkeypatch.setenv("SO101_SIM_TASK", "reach")
    monkeypatch.setenv("SO101_SIM_SEED", str(SEED))
    monkeypatch.setenv("SO101_COLLAPSE_SPEED_DEG_S", str(SPEED))
    monkeypatch.setattr(RobotManager, "collapse_rate_hz", RATE)
    config.get_settings.cache_clear()
    reset_manager()
    devices.reset_sim_runtime()
    yield
    reset_manager()
    devices.reset_sim_runtime()


class Rig:
    def __init__(self, manager: RobotManager, runtime: devices.SimRuntime, clock: Any) -> None:
        self.manager = manager
        self.runtime = runtime
        self.clock = clock
        self.lease_id = ""

    def pose(self) -> np.ndarray:
        return self.runtime.world.joint_array().copy()


@pytest.fixture
def rig(sim_env, manager_clock, supervise) -> Rig:
    manager = get_manager()
    runtime = devices.get_sim_runtime()
    runtime.on_episode_start(SEED)  # the task's start pose, actuation off
    rig = Rig(manager, runtime, manager_clock)
    manager.operator_heartbeat(OPERATOR)
    lease = manager.acquire_lease("physics", 60)
    rig.lease_id = lease["id"]
    manager.arm_control(rig.lease_id)
    assert runtime.torque_enabled is True
    runtime.advance(0.5)  # holding the start pose under gravity
    return rig


def _events(kind: str) -> list[dict[str, Any]]:
    return [event for event in get_store().list_events(limit=500) if event["type"] == kind]


def _wait_for(predicate, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_the_follower_is_the_simulated_arm(rig: Rig):
    assert isinstance(rig.manager.follower().bus, devices.MujocoBus)
    assert rig.manager.status()["torque_state"] == "on"


def test_estop_cuts_actuation_and_the_arm_sags(rig: Rig):
    before = rig.pose()

    rig.manager.estop()
    assert rig.runtime.torque_enabled is False
    rig.runtime.advance(2.0)

    sag = rig.pose() - before
    MEASURED["estop_sag_2s_deg"] = dict(zip(JOINT_NAMES, np.round(sag, 2).tolist(), strict=True))
    print("E-stop sag over 2 s:", MEASURED["estop_sag_2s_deg"])
    # Category 0: nothing holds the arm, so gravity takes it.
    assert np.max(np.abs(sag[ARM_SLOTS])) > 10.0
    assert rig.manager.status()["torque_state"] == "off"


def _drift(runtime: devices.SimRuntime, reference: np.ndarray, seconds: float) -> np.ndarray:
    worst = np.zeros(len(JOINT_NAMES))
    for _ in range(round(seconds / 0.1)):  # sampled every 0.1 s of simulated time
        runtime.advance(0.1)
        worst = np.maximum(worst, np.abs(runtime.world.joint_array() - reference))
    return worst


def test_halt_holds_a_still_arm_within_1_deg_over_10_s(rig: Rig):
    result = rig.manager.request_stop(StopKind.HALT, reason="physics halt")

    assert result.holding
    held = rig.pose()
    worst = _drift(rig.runtime, held, 10.0)
    MEASURED["halt_drift_10s_deg"] = dict(
        zip(JOINT_NAMES, np.round(worst, 3).tolist(), strict=True)
    )
    print("Halt drift over 10 s (still arm):", MEASURED["halt_drift_10s_deg"])
    assert np.max(worst[ARM_SLOTS]) < 1.0
    assert rig.runtime.torque_enabled is True
    status = rig.manager.status()
    assert status["stop_state"] == "halted" and status["torque_state"] == "on"


def test_halt_catches_a_moving_arm_then_drifts_under_1_deg_over_10_s(rig: Rig, monkeypatch):
    """Goal := Present stops a moving arm where the halt read it.

    The servo brakes onto that goal from its speed (a short excursion past it
    and back, reported), then holds: drift over the next 10 s stays under 1°.
    """

    manager, runtime = rig.manager, rig.runtime
    # The guard's rate limit runs on the same fake clock as the manager.
    monkeypatch.setattr(manager.guard, "_clock", rig.clock)
    start = runtime.read_positions()
    gripper_before = start["gripper"]
    for step in range(1, 13):  # pan and elbow commanded at 30 deg/s for 0.4 s
        rig.clock.advance(1 / RATE)
        manager.send_external_command(
            lease_id=rig.lease_id,
            source="physics",
            sequence=step,
            sent_at=datetime.now(timezone.utc),
            ttl_ms=500,
            deadman=True,
            targets={
                "shoulder_pan": start["shoulder_pan"] + step,
                "elbow_flex": start["elbow_flex"] + step,
            },
        )
    speed_at_halt = np.degrees(runtime.world.data.qvel[runtime.world.index.dof])

    result = manager.request_stop(StopKind.HALT, reason="physics halt")

    assert result.holding
    halted = rig.pose()
    excursion = _drift(runtime, halted, 0.5)  # the servo brakes onto the hold goal
    caught = rig.pose()
    worst = _drift(runtime, caught, 10.0)
    MEASURED["halt_while_moving"] = {
        "speed_at_halt_deg_s": round(float(np.max(np.abs(speed_at_halt[ARM_SLOTS]))), 1),
        "catch_excursion_deg": round(float(np.max(excursion[ARM_SLOTS])), 3),
        "drift_10s_deg": dict(zip(JOINT_NAMES, np.round(worst, 3).tolist(), strict=True)),
    }
    print("Halt while moving:", MEASURED["halt_while_moving"])
    assert np.max(excursion[ARM_SLOTS]) < 3.0
    assert np.max(worst[ARM_SLOTS]) < 1.0
    assert runtime.torque_enabled is True
    assert manager.status()["stop_state"] == "halted"
    assert runtime.read_positions()["gripper"] <= gripper_before + 0.5  # never opened


def test_collapse_stays_within_its_cap_and_ends_at_rest_with_actuation_off(rig: Rig, monkeypatch):
    manager, runtime = rig.manager, rig.runtime
    world = runtime.world
    track: list[tuple[float, np.ndarray]] = []
    peak = np.zeros(len(JOINT_NAMES))
    write_goal, step = runtime.write_goal, world.step

    def recording_write(goal):
        applied = write_goal(goal)
        track.append((world.time_s, world.joint_array().copy()))
        return applied

    def recording_step(n: int = 1) -> None:
        nonlocal peak
        step(n)
        peak = np.maximum(peak, np.abs(np.degrees(world.data.qvel[world.index.dof])))

    monkeypatch.setattr(runtime, "write_goal", recording_write)
    monkeypatch.setattr(world, "step", recording_step)
    follower = manager.follower()
    send_goal = follower.send_goal
    setpoints: list[dict[str, float]] = []

    def spy(goal: dict[str, float], *, dt: float | None = None) -> dict[str, float]:
        setpoints.append(dict(goal))
        return send_goal(goal, dt=dt)

    monkeypatch.setattr(follower, "send_goal", spy)

    result = manager.request_stop(StopKind.COLLAPSE, reason="physics collapse")

    assert result.collapsing
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    (done,) = _events("safety.collapse.done")
    assert _events("safety.collapse.fallback") == []
    # Commanded: every joint at most speed / rate per tick.
    commanded = max(
        abs(after[joint] - before[joint]) * RATE
        for before, after in zip(setpoints, setpoints[1:], strict=False)
        for joint in ARM
    )
    assert commanded <= SPEED + 1e-6
    # Measured: joint speed over 0.25 s windows, as the HIL probe checks it.
    times = np.array([t for t, _ in track])
    poses = np.array([pose for _, pose in track])
    window = round(SPEED_WINDOW_S * RATE)
    windowed = np.max(
        np.abs(poses[window:] - poses[:-window]) / (times[window:] - times[:-window])[:, None],
        axis=0,
    )
    per_tick = np.max(np.abs(np.diff(poses, axis=0)), axis=0) * RATE
    assert np.max(windowed[ARM_SLOTS]) <= SPEED * SPEED_TOLERANCE
    # It ends at rest, then actuation is off and stays off.
    assert done["detail"]["final_error_deg"] <= 1.0
    assert done["detail"]["torque_off"] is True
    assert runtime.torque_enabled is False
    assert manager.status()["torque_state"] == "off"
    at_rest = rig.pose()
    runtime.advance(2.0)
    settled = rig.pose()
    assert np.max(np.abs(settled - at_rest)[ARM_SLOTS]) < SETTLE_LIMIT
    assert np.max(np.abs(settled - REST)[ARM_SLOTS]) < SETTLE_LIMIT
    MEASURED["collapse"] = {
        "duration_s": done["detail"]["duration_s"],
        "commanded_peak_deg_s": round(commanded, 3),
        "windowed_0.25s_peak_deg_s": dict(
            zip(JOINT_NAMES, np.round(windowed, 2).tolist(), strict=True)
        ),
        "per_tick_30hz_peak_deg_s": dict(
            zip(JOINT_NAMES, np.round(per_tick, 2).tolist(), strict=True)
        ),
        "instantaneous_peak_deg_s": dict(zip(JOINT_NAMES, np.round(peak, 1).tolist(), strict=True)),
        "final_error_at_torque_off_deg": done["detail"]["final_error_deg"],
        "error_to_rest_after_2s_off_deg": dict(
            zip(JOINT_NAMES, np.round(settled - REST, 2).tolist(), strict=True)
        ),
    }
    print("Collapse:", MEASURED["collapse"])
