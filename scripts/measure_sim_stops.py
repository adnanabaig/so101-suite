#!/usr/bin/env python3
"""Measure the section 5 stop behaviours on the simulated SO-101.

Dev-only; needs the ``[sim]`` extra. Prints, for both servo modes
(``sts3215`` acceleration profile and ``step``):

- **E-stop**: sag per joint after actuation is cut, from three poses;
- **Halt**: goal := present. Steady drift over 10 s while holding, and the
  overshoot when halting a joint that is moving;
- **Collapse**: a 30 Hz goal stream towards the rest pose, capped per joint at
  ``--speed`` deg/s, then actuation off. Peak instantaneous joint speed vs the
  speed seen by 30 Hz position differences, final error, settle at rest.

    PYTHONPATH=src python scripts/measure_sim_stops.py [--speed 15]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from so101 import JOINT_NAMES
from so101.simulation.rest_pose import MODEL_REST_POSE
from so101.simulation.servo import SERVO_MODES, ServoDriver
from so101.simulation.world import PhysicsWorld

ARM = [joint for joint in JOINT_NAMES if joint != "gripper"]
POSES = {
    "zero": dict.fromkeys(ARM, 0.0) | {"gripper": 50.0},
    "work": {
        "shoulder_pan": 20.0,
        "shoulder_lift": -20.0,
        "elbow_flex": 30.0,
        "wrist_flex": 60.0,
        "wrist_roll": 0.0,
        "gripper": 30.0,
    },
    "rest": dict(MODEL_REST_POSE),
}
CONTROL_HZ = 30.0


def _round(values: np.ndarray, digits: int = 2) -> dict[str, float]:
    return {joint: round(float(v), digits) for joint, v in zip(JOINT_NAMES, values, strict=True)}


def estop(world: PhysicsWorld, driver: ServoDriver) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, pose in POSES.items():
        world.reset(joints=pose)
        driver.sync()
        start = world.joint_array()
        driver.set_torque(False)
        sag = {}
        for horizon in (0.25, 0.5, 1.0, 2.0):
            driver.advance(horizon - world.time_s)
            sag[f"{horizon}s"] = _round(world.joint_array() - start, 1)
        contacts = world.contacts()
        out[name] = {
            "sag_deg": sag,
            "arm_self_contacts": contacts["arm_self"],
            "arm_table_contacts": contacts["arm_table"],
        }
    return out


def halt(world: PhysicsWorld, driver: ServoDriver) -> dict[str, object]:
    steady = {}
    for name, pose in POSES.items():
        world.reset(joints=pose)
        driver.sync()
        driver.advance(0.5)
        driver.hold()
        start = world.joint_array()
        worst = np.zeros(len(JOINT_NAMES))
        for _ in range(100):
            driver.advance(0.1)
            worst = np.maximum(worst, np.abs(world.joint_array() - start))
        steady[name] = round(float(worst.max()), 4)
    moving = {}
    for speed in (30.0, 90.0, 180.0):
        # A smooth pan + lift-up move (goal updated every substep), halted mid-move.
        world.reset(joints=POSES["work"])
        driver.sync()
        goal = world.joint_array()
        for _ in range(world.steps_for(0.4)):
            goal[:2] += np.array([1.0, -1.0]) * speed * world.timestep_s
            driver.command(goal)
            driver.step(1)
        velocity = world.velocity_array()[:2].copy()
        driver.hold()
        halted = world.joint_array()
        worst = np.zeros(len(JOINT_NAMES))
        for _ in range(world.steps_for(10.0)):
            driver.step(1)
            worst = np.maximum(worst, np.abs(world.joint_array() - halted))
        moving[f"{speed:g}deg_s"] = {
            "speed_at_halt": [round(float(v), 1) for v in velocity],
            "max_excursion_deg": round(float(worst.max()), 2),
            "final_error_deg": round(float(np.abs(world.joint_array() - halted).max()), 3),
        }
    return {"steady_10s_max_drift_deg": steady, "halt_while_moving": moving}


def collapse(world: PhysicsWorld, driver: ServoDriver, speed: float) -> dict[str, object]:
    out: dict[str, object] = {}
    rest = np.array([MODEL_REST_POSE[joint] for joint in JOINT_NAMES])
    for name in ("zero", "work"):
        world.reset(joints=POSES[name])
        driver.sync()
        goal = world.joint_array()
        step = speed / CONTROL_HZ
        positions = [world.joint_array()]
        peak = np.zeros(len(JOINT_NAMES))
        ticks = 0
        timeout = min(12.0, float(np.max(np.abs(rest - goal))) / speed + 2.0)
        while ticks < timeout * CONTROL_HZ:
            goal = goal + np.clip(rest - goal, -step, step)
            driver.command(goal)
            for _ in range(world.steps_for(1.0 / CONTROL_HZ)):
                driver.step(1)
                peak = np.maximum(peak, np.abs(world.velocity_array()))
            positions.append(world.joint_array())
            ticks += 1
            if np.allclose(goal, rest) and np.max(np.abs(world.joint_array() - rest)) < 0.5:
                break
        track = np.array(positions)
        finite = np.abs(np.diff(track, axis=0)).max(axis=0) * CONTROL_HZ
        window = int(0.2 * CONTROL_HZ)
        windowed = np.abs(track[window:] - track[:-window]).max(axis=0) * CONTROL_HZ / window
        final_error = world.joint_array() - rest
        driver.set_torque(False)
        before = world.joint_array()
        driver.advance(2.0)
        out[name] = {
            "duration_s": round(ticks / CONTROL_HZ, 2),
            "timeout_s": round(timeout, 2),
            "peak_qvel_deg_s": _round(peak, 1),
            "peak_30hz_diff_deg_s": _round(finite, 1),
            "peak_0.2s_window_deg_s": _round(windowed, 1),
            "final_error_deg": _round(final_error, 2),
            "settle_after_torque_off_deg": _round(world.joint_array() - before, 2),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--speed", type=float, default=15.0, help="collapse cap, deg/s")
    args = parser.parse_args(argv)
    world = PhysicsWorld()
    report = {}
    for mode in SERVO_MODES:
        driver = ServoDriver(world, mode=mode)  # type: ignore[arg-type]
        report[mode] = {
            "estop": estop(world, driver),
            "halt": halt(world, driver),
            "collapse": collapse(world, driver, args.speed),
        }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
