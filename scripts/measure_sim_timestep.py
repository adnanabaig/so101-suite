"""Compare physics timesteps: 1/240 s (the suite default) vs the MJCF's native 5 ms.

    PYTHONPATH=src python scripts/measure_sim_timestep.py
    PYTHONPATH=src python scripts/measure_sim_timestep.py --episodes 100 --out timestep.json

For each timestep (default 5 ms, 1/240 s and a 1/480 s reference) it measures

- control alignment: physics steps per 30 Hz control step and 60 Hz engine
  tick, and how far the realised period is from the nominal one;
- arm hold: steady-state error and drift holding the ready pose for 10 s;
- step response: overshoot and 2 % settling time of a raw 30 degree step
  (``step`` servo mode, no motion profile: the stiffest thing the arm sees);
- cube at rest: penetration into the table, creep and residual speed;
- grasping: scripted-expert success on pick_lift and pick_place, the slip of
  the cube in the jaws over 1.5 s of holding, and the jaw penetration;
- speed: physics steps/s in a contact-free hold and in the expert episodes.

The numbers in docs/simulation.md ("Timestep") come from this script.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from so101 import JOINT_NAMES
from so101.simulation.expert import ScriptedExpert
from so101.simulation.policy import wilson_interval
from so101.simulation.rollout import EpisodeRunner, StepRecord
from so101.simulation.servo import ServoDriver
from so101.simulation.tasks import READY_POSE, Task, make_task
from so101.simulation.world import PhysicsWorld, SceneSpec

DEFAULT_TIMESTEPS = (0.005, 1.0 / 240.0, 1.0 / 480.0)


def task_at(name: str, timestep: float) -> Task:
    task = make_task(name)
    task.scene = dataclasses.replace(task.scene, timestep_s=timestep)
    return task


def alignment(timestep: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, hz in (("control_30hz", 30.0), ("engine_60hz", 60.0)):
        steps = max(1, round(1.0 / (hz * timestep)))
        period = steps * timestep
        out[label] = {
            "substeps": steps,
            "period_ms": round(period * 1e3, 3),
            "error_pct": round(100.0 * (period * hz - 1.0), 2),
        }
    return out


def arm_hold(timestep: float) -> dict[str, Any]:
    world = PhysicsWorld(SceneSpec(timestep_s=timestep))
    world.reset(joints=READY_POSE)
    driver = ServoDriver(world)
    target = np.array([READY_POSE[joint] for joint in JOINT_NAMES])
    driver.command(target)
    driver.advance(1.0)
    after_1s = world.joint_array()
    started = time.perf_counter()
    steps = world.steps_for(9.0)
    driver.step(steps)
    wall = time.perf_counter() - started
    final = world.joint_array()
    speeds = []
    for _ in range(world.steps_for(1.0)):
        driver.step(1)
        speeds.append(np.abs(world.velocity_array()).max())
    return {
        "steady_error_deg": round(float(np.abs(final - target).max()), 4),
        "drift_1s_to_10s_deg": round(float(np.abs(final - after_1s).max()), 5),
        "residual_speed_deg_s": round(float(max(speeds)), 5),
        "free_steps_per_s": round(steps / wall),
    }


def step_response(timestep: float, joint: str = "shoulder_pan", size: float = 30.0) -> dict:
    world = PhysicsWorld(SceneSpec(timestep_s=timestep))
    world.reset(joints=READY_POSE)
    driver = ServoDriver(world, mode="step")
    index = JOINT_NAMES.index(joint)
    start = float(world.joint_array()[index])
    goal = start + size
    driver.command({joint: goal})
    trace = []
    for _ in range(world.steps_for(1.5)):
        driver.step(1)
        trace.append((world.time_s, float(world.joint_array()[index])))
    times = np.array([t for t, _ in trace])
    values = np.array([v for _, v in trace])
    overshoot = float(values.max() - goal)
    outside = np.nonzero(np.abs(values - goal) > 0.02 * size)[0]
    settle = float(times[outside[-1] + 1]) if outside.size and outside[-1] + 1 < len(times) else 0.0
    return {
        "joint": joint,
        "step_deg": size,
        "overshoot_deg": round(overshoot, 3),
        "settle_2pct_s": round(settle, 3),
        "final_error_deg": round(float(abs(values[-1] - goal)), 4),
    }


def cube_rest(timestep: float, seeds: int = 5) -> dict[str, Any]:
    task = task_at("pick_lift", timestep)
    world = task.make_world()
    penetration, creep, speed = [], [], []
    for seed in range(seeds):
        task.reset(world, seed)
        world.settle(1.0, rewind_clock=False)
        start, _ = world.object_pose("cube")
        top = 0.0
        for _ in range(world.steps_for(4.0)):
            world.step(1)
            top = max(top, float(world.contacts()["max_penetration_m"]))
        end, _ = world.object_pose("cube")
        linear, _ = world.object_velocity("cube")
        penetration.append(top)
        creep.append(float(np.linalg.norm(end - start)))
        speed.append(float(np.linalg.norm(linear)))
    return {
        "max_penetration_mm": round(1e3 * max(penetration), 4),
        "creep_4s_mm": round(1e3 * max(creep), 5),
        "residual_speed_mm_s": round(1e3 * max(speed), 5),
    }


def grasping(name: str, timestep: float, episodes: int) -> dict[str, Any]:
    task = task_at(name, timestep)
    runner = EpisodeRunner(task)
    expert = ScriptedExpert(task)
    successes, slips, jaw_penetration, times = 0, [], [], []
    sim_s = wall_s = 0.0
    steps = 0
    for seed in range(episodes):
        gaps: list[float] = []
        penetration: list[float] = []

        def record(step: StepRecord, gaps: list[float] = gaps, pen: list[float] = penetration):
            site, _ = runner.world.site_pose()
            cube, _ = runner.world.object_pose("cube")
            gaps.append(float(site[2] - cube[2]))
            contacts = runner.world.contacts()
            pen.append(float(contacts["max_penetration_m"]) if task.success() else 0.0)

        started = time.perf_counter()
        result = runner.run(
            expert,
            seed,
            on_step=record,
            tail_s=1.5 if name == "pick_lift" else 0.0,
            policy_reset=functools.partial(expert.reset, seed),
        )
        wall_s += time.perf_counter() - started
        sim_s += result.steps * runner.world.steps_for(1 / 30) * timestep
        steps += result.steps * runner.substeps
        if result.success:
            successes += 1
            times.append(float(result.first_success_s or 0.0))
            if name == "pick_lift":
                at = round(float(result.first_success_s or 0.0) * 30) - 1
                slips.append(gaps[-1] - gaps[max(0, at)])
                jaw_penetration.append(max(penetration))
    out: dict[str, Any] = {
        "episodes": episodes,
        "successes": successes,
        "wilson95": [round(v, 3) for v in wilson_interval(successes, episodes)],
        "mean_first_success_s": round(float(np.mean(times)), 3) if times else None,
        "steps_per_s": round(steps / wall_s) if wall_s else None,
        "real_time_factor": round(sim_s / wall_s, 1) if wall_s else None,
    }
    if slips:
        out["hold_slip_1p5s_mm"] = {
            "mean": round(1e3 * float(np.mean(slips)), 3),
            "max": round(1e3 * float(np.max(slips)), 3),
        }
        out["jaw_penetration_mm_max"] = round(1e3 * max(jaw_penetration), 3)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", nargs="+", type=float, default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    results = []
    for timestep in args.timesteps:
        entry = {
            "timestep_s": round(timestep, 7),
            "physics_hz": round(1.0 / timestep, 2),
            "alignment": alignment(timestep),
            "arm_hold": arm_hold(timestep),
            "step_response": step_response(timestep),
            "cube_rest": cube_rest(timestep),
            "pick_lift": grasping("pick_lift", timestep, args.episodes),
            "pick_place": grasping("pick_place", timestep, args.episodes),
        }
        results.append(entry)
        print(json.dumps(entry), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
