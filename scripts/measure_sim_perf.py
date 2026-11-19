"""Measure simulation speed: physics, rendering, /api/simulation and the simulated robot.

    PYTHONPATH=src python scripts/measure_sim_perf.py
    PYTHONPATH=src python scripts/measure_sim_perf.py --sections physics render --out perf.json

Each section prints one JSON object; ``--out`` writes them all to a file.

- ``physics``: the first world's load and compile time (cold, as this section
  runs first), then physics steps/s and the real-time factor at the default
  timestep, holding the ready pose and sweeping every joint (no contacts), and
  holding in the pick_place scene (objects resting on the table);
- ``render``: frames/s per size and camera through the shared render thread,
  the GL start-up and first-frame latency, and PNG/JPEG encoding time. When
  this process cannot render, the section says why and the rest still runs;
- ``engine``: the MuJoCo ``/api/simulation`` engine: construction, ticks/s
  while the arm moves and once it has settled, ``state()``, a 1 s trajectory
  preview and one camera frame;
- ``lockstep``: teleop episodes on a lockstep runtime through the real
  TeleopController, SO101Follower and SafetyGuard with the expert leader, per
  task, then once more reading both cameras every control period;
- ``realtime``: the real-time physics thread holding still, then under 30 Hz
  teleop: simulated time over wall time, dropped steps and CPU use.

Process RSS is added to every section when psutil is installed. The numbers in
docs/simulation.md ("Performance", "Rendering") come from this script. Every
number is wall-clock time, so close other heavy jobs first.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from so101 import JOINT_NAMES
from so101.hardware.follower import SO101Follower
from so101.hardware.leader import SO101Leader
from so101.hardware.safety import SafetyGuard
from so101.simulation.devices import (
    CONTROL_DT_S,
    MujocoBus,
    SimCamera,
    SimExpertLeader,
    SimRuntime,
    make_sim_camera,
)
from so101.simulation.physics_engine import PhysicsSimulationEngine
from so101.simulation.render import encode_frame, jpeg_supported, shared_renderer
from so101.simulation.servo import ServoDriver
from so101.simulation.tasks import READY_POSE, TASK_NAMES, make_task
from so101.simulation.world import CAMERAS, PhysicsWorld, SceneSpec
from so101.teleop.controller import TeleopController

SECTIONS = ("physics", "render", "engine", "lockstep", "realtime")
SWEEP_PERIOD_S = 0.5  # the sweeps switch target this often, so the arm never settles
# The second pose of each sweep, suite units. Every joint moves 30-60 deg.
SWEEP_POSE = {
    "shoulder_pan": 40.0,
    "shoulder_lift": -30.0,
    "elbow_flex": 60.0,
    "wrist_flex": 40.0,
    "wrist_roll": 60.0,
    "gripper": 70.0,
}
ENGINE_SWEEP_POSE = {
    "shoulder_pan": 40.0,
    "shoulder_lift": -30.0,
    "elbow_flex": 40.0,
    "wrist_flex": 30.0,
    "wrist_roll": 60.0,
    "gripper": 80.0,
}


def ms(seconds: float) -> float:
    return round(1e3 * seconds, 2)


def rss_mb() -> float | None:
    """Resident set size of this process, or None without psutil."""

    try:
        psutil = importlib.import_module("psutil")
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / 2**20, 1)


def machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def stepping_rate(run: Callable[[], object], steps: int, timestep_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    run()
    wall = time.perf_counter() - started
    return {
        "steps_per_s": round(steps / wall),
        "real_time_factor": round(steps * timestep_s / wall, 1),
    }


# -- physics -----------------------------------------------------------------------------
def physics(sim_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    world = PhysicsWorld(SceneSpec())  # the first in the process: loads and compiles the MJCF
    first_world = time.perf_counter() - started
    world.reset(joints=READY_POSE)
    driver = ServoDriver(world)
    ready = np.array([READY_POSE[joint] for joint in JOINT_NAMES])
    sweep = np.array([SWEEP_POSE[joint] for joint in JOINT_NAMES])
    driver.command(ready)
    driver.advance(1.0)
    steps = world.steps_for(sim_s)
    period = world.steps_for(SWEEP_PERIOD_S)
    timestep = world.timestep_s

    def sweeping() -> None:
        for count, done in enumerate(range(0, steps, period)):
            driver.command(sweep if count % 2 == 0 else ready)
            driver.step(min(period, steps - done))

    hold = stepping_rate(lambda: driver.step(steps), steps, timestep)
    moving = stepping_rate(sweeping, steps, timestep)

    task = make_task("pick_place")
    scene_world = task.make_world()
    task.reset(scene_world, 0)
    scene_driver = ServoDriver(scene_world)
    scene_driver.command(scene_world.joint_array())
    scene_driver.advance(0.5)
    scene = stepping_rate(lambda: scene_driver.step(steps), steps, timestep)
    return {
        "first_world_ms": ms(first_world),
        "timestep_s": round(timestep, 7),
        "sim_s": sim_s,
        "hold": hold,
        "sweep": moving,
        "pick_place_scene_hold": scene,
    }


# -- rendering -----------------------------------------------------------------------------
def render(sizes: Sequence[tuple[int, int]], frames: int) -> dict[str, Any]:
    renderer = shared_renderer()
    started = time.perf_counter()
    available = renderer.start()
    out: dict[str, Any] = {"available": available, "gl_start_ms": ms(time.perf_counter() - started)}
    if not available:
        out["error"] = renderer.error
        return out
    task = make_task("pick_place")
    world = task.make_world()
    task.reset(world, 0)
    out["sizes"] = {}
    for width, height in sizes:
        entry: dict[str, Any] = {"fps": {}}
        for camera in CAMERAS:
            started = time.perf_counter()
            frame = renderer.render(world.snapshot(), camera, width, height)
            entry.setdefault("first_frame_ms", ms(time.perf_counter() - started))
            started = time.perf_counter()
            for _ in range(frames):
                frame = renderer.render(world.snapshot(), camera, width, height)
            entry["fps"][camera] = round(frames / (time.perf_counter() - started), 1)
        entry["encode_ms"] = encoding(frame)
        out["sizes"][f"{width}x{height}"] = entry
    return out


def encoding(frame: np.ndarray, repeats: int = 5) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for fmt in ("png", "jpeg"):
        if fmt == "jpeg" and not jpeg_supported():
            out[fmt] = None  # encode_frame would fall back to PNG
            continue
        started = time.perf_counter()
        for _ in range(repeats):
            encode_frame(frame, fmt)
        out[fmt] = ms((time.perf_counter() - started) / repeats)
    return out


# -- /api/simulation engine ------------------------------------------------------------------
def engine(ticks: int) -> dict[str, Any]:
    started = time.perf_counter()
    sim = PhysicsSimulationEngine()
    build = time.perf_counter() - started
    home = dict(sim.library.engine.initial_joint_positions)
    chunk = max(1, round(SWEEP_PERIOD_S / sim.fixed_dt_s))

    def moving() -> None:
        for count, done in enumerate(range(0, ticks, chunk)):
            target = ENGINE_SWEEP_POSE if count % 2 == 0 else home
            sim.step(min(chunk, ticks - done), targets=target)

    def settled() -> None:
        for done in range(0, ticks, sim.MAX_MANUAL_STEPS):
            sim.step(min(sim.MAX_MANUAL_STEPS, ticks - done))

    moving_rate = stepping_rate(moving, ticks, sim.fixed_dt_s)
    sim.step(round(2.0 / sim.fixed_dt_s))  # let the arm come to rest on its last target
    settled_rate = stepping_rate(settled, ticks, sim.fixed_dt_s)

    calls = 200
    started = time.perf_counter()
    for _ in range(calls):
        sim.state()
    state_s = (time.perf_counter() - started) / calls

    previews = []
    for _ in range(6):
        started = time.perf_counter()
        sim.preview_trajectory(target=ENGINE_SWEEP_POSE, duration_s=1.0)
        previews.append(time.perf_counter() - started)

    frame_ms = None
    if shared_renderer().available:
        sim.render_frame("front", 320, 240)  # the scene's GL context
        started = time.perf_counter()
        for _ in range(30):
            sim.render_frame("front", 320, 240)
        frame_ms = ms((time.perf_counter() - started) / 30)
    return {
        "construct_ms": ms(build),
        "substeps_per_tick": sim.substeps_per_tick,
        "moving": {
            "ticks_per_s": moving_rate["steps_per_s"],
            "real_time_factor": moving_rate["real_time_factor"],
        },
        "settled": {
            "ticks_per_s": settled_rate["steps_per_s"],
            "real_time_factor": settled_rate["real_time_factor"],
        },
        "state_ms": ms(state_s),
        "preview_1s_ms": {"first": ms(previews[0]), "mean_rest": ms(float(np.mean(previews[1:])))},
        "render_frame_320x240_ms": frame_ms,
    }


# -- simulated robot devices ----------------------------------------------------------------
class StepClock:
    """The SafetyGuard's clock in lockstep: one nominal control period per step."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def teleop_rig(
    runtime: SimRuntime, guard_clock: Callable[[], float] | None = None
) -> tuple[TeleopController, SO101Follower]:
    """The real follower, leader and teleop loop, with sim buses swapped in by hand."""

    guard = SafetyGuard() if guard_clock is None else SafetyGuard(_clock=guard_clock)
    follower = SO101Follower(guard=guard)
    follower.bus = MujocoBus(runtime=runtime)
    leader = SO101Leader()
    leader.bus = SimExpertLeader(runtime=runtime)
    return TeleopController(leader, follower), follower


def lockstep_episodes(
    runtime: SimRuntime,
    episodes: int,
    cameras: Iterable[SimCamera] = (),
) -> dict[str, Any]:
    clock = StepClock()
    controller, follower = teleop_rig(runtime, clock)
    cameras = tuple(cameras)
    controller.connect()
    periods, wall, first = 0, 0.0, []
    try:
        for seed in range(episodes):
            runtime.on_episode_start(seed)
            follower.guard.seed(follower.read_positions())  # after the start-pose move
            steps = runtime.task.episode_steps()
            started = time.perf_counter()
            for _ in range(steps):
                clock.t += CONTROL_DT_S
                controller.step()
                for camera in cameras:
                    camera.read()
            wall += time.perf_counter() - started
            periods += steps
            outcome = runtime.on_episode_end()
            if outcome["success"]:
                first.append(float(outcome["metrics"]["first_success_s"]))
    finally:
        controller.disconnect()
    return {
        "episodes": episodes,
        "successes": len(first),
        "mean_first_success_s": round(float(np.mean(first)), 3) if first else None,
        "periods_per_s": round(periods / wall),
        "real_time_factor": round(periods * CONTROL_DT_S / wall, 1),
    }


def lockstep(
    tasks: Sequence[str], episodes: int, camera_size: tuple[int, int], camera_task: str
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in tasks:
        started = time.perf_counter()
        runtime = SimRuntime(name, realtime=False, seed=0)
        build = time.perf_counter() - started
        out[name] = {"build_ms": ms(build), **lockstep_episodes(runtime, episodes)}
    renderer = shared_renderer()
    if not renderer.available:
        out["with_cameras"] = {"available": False, "error": renderer.error}
        return out
    runtime = SimRuntime(camera_task, realtime=False, seed=0)
    width, height = camera_size
    cameras = [make_sim_camera(cam, runtime=runtime, width=width, height=height) for cam in CAMERAS]
    try:
        for camera in cameras:
            camera.read()  # the scene's GL context
        result = lockstep_episodes(runtime, 1, cameras)
    finally:
        for camera in cameras:
            camera.release()
    out["with_cameras"] = {
        "task": camera_task,
        "cameras": list(CAMERAS),
        "size": f"{width}x{height}",
        **result,
        "frames_per_s": round(result["periods_per_s"] * len(cameras)),
    }
    return out


def drain(steps: Iterable[object]) -> None:
    for _ in steps:
        pass


def caught_up(runtime: SimRuntime, timeout_s: float = 1.0) -> tuple[int, float]:
    """Wait for the physics thread's next catch-up; its step count and when it was seen.

    Right after a catch-up the thread is level with the wall clock (to within
    one physics step), so windows measured between two of them are not skewed
    by whatever backlog the thread happened to have at the edges.
    """

    first = runtime.status()["physics_steps"]
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        steps = runtime.status()["physics_steps"]
        if steps != first:
            return steps, time.perf_counter()
        time.sleep(0.0005)
    raise RuntimeError(f"the physics thread stopped stepping: {runtime.status()}")


def realtime_window(runtime: SimRuntime, body: Callable[[], object]) -> dict[str, Any]:
    """Run ``body`` against the live physics thread and see how well it kept up.

    Simulated time is counted in the thread's physics steps rather than read
    off the world clock, which an episode start rewinds.
    """

    steps, started = caught_up(runtime)
    dropped = runtime.status()["dropped_steps"]
    cpu = time.process_time()
    body()
    end_steps, ended = caught_up(runtime)
    cpu = time.process_time() - cpu
    wall = ended - started
    simulated = (end_steps - steps) * runtime.world.timestep_s
    return {
        "wall_s": round(wall, 3),
        "sim_s": round(simulated, 4),
        "real_time_factor": round(simulated / wall, 4),
        "dropped_steps": runtime.status()["dropped_steps"] - dropped,
        "cpu_pct_of_one_core": round(100.0 * cpu / wall, 1),
    }


def realtime(duration_s: float, task: str) -> dict[str, Any]:
    runtime = SimRuntime(task, realtime=True, seed=0)
    controller, follower = teleop_rig(runtime)  # the guard runs on the real clock
    out: dict[str, Any] = {"task": task}

    def teleop() -> None:
        runtime.on_episode_start(0)
        follower.guard.seed(follower.read_positions())  # after the start-pose move
        drain(controller.run(duration_s))

    follower.connect()  # starts the physics thread; torque on, holding the folded pose
    try:
        out["hold"] = realtime_window(runtime, lambda: time.sleep(duration_s))
        controller.leader.connect()
        out["teleop_30hz"] = realtime_window(runtime, teleop)
        outcome = runtime.on_episode_end()
        out["teleop_30hz"]["success"] = outcome["success"]
        out["teleop_30hz"]["first_success_s"] = outcome["metrics"]["first_success_s"]
    finally:
        controller.disconnect()
        runtime.stop()
    return out


# -- summary ---------------------------------------------------------------------------------
def rate(entry: dict[str, Any], key: str = "steps_per_s") -> str:
    return f"{entry[key]:,} ({entry['real_time_factor']}x)"


def summary(results: dict[str, Any]) -> list[str]:
    lines = []
    if "physics" in results:
        p = results["physics"]
        lines.append(
            f"physics   steps/s: hold {rate(p['hold'])}, sweep {rate(p['sweep'])}, "
            f"pick_place scene {rate(p['pick_place_scene_hold'])}"
        )
    if "render" in results:
        r = results["render"]
        if not r["available"]:
            lines.append(f"render    unavailable: {r.get('error')}")
        for size, entry in r.get("sizes", {}).items():
            fps = ", ".join(f"{cam} {value}" for cam, value in entry["fps"].items())
            lines.append(
                f"render    {size}: {fps} frames/s, first frame {entry['first_frame_ms']} ms, "
                f"png {entry['encode_ms']['png']} ms"
            )
    if "engine" in results:
        e = results["engine"]
        lines.append(
            f"engine    ticks/s: moving {rate(e['moving'], 'ticks_per_s')}, settled "
            f"{rate(e['settled'], 'ticks_per_s')}; state {e['state_ms']} ms, "
            f"1 s preview {e['preview_1s_ms']['mean_rest']} ms"
        )
    for name, entry in results.get("lockstep", {}).items():
        if name == "rss_mb":
            continue
        if "periods_per_s" not in entry:
            lines.append(f"lockstep  {name}: unavailable: {entry.get('error')}")
            continue
        lines.append(
            f"lockstep  {name}: {rate(entry, 'periods_per_s')} control periods/s, "
            f"{entry['successes']}/{entry['episodes']} succeeded"
        )
    if "realtime" in results:
        for phase in ("hold", "teleop_30hz"):
            entry = results["realtime"][phase]
            lines.append(
                f"realtime  {phase}: {entry['real_time_factor']}x the wall clock, "
                f"{entry['dropped_steps']} dropped steps, "
                f"{entry['cpu_pct_of_one_core']} % of one core"
            )
    return lines


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from exc
    return width, height


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--physics-s", type=float, default=20.0, help="simulated s per case")
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(640, 480), (320, 240)])
    parser.add_argument("--frames", type=int, default=120, help="frames per size and camera")
    parser.add_argument("--engine-ticks", type=int, default=1200)
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--episodes", type=int, default=3, help="lockstep episodes per task")
    parser.add_argument("--camera-size", type=parse_size, default=(320, 240))
    parser.add_argument("--camera-task", choices=TASK_NAMES, default="pick_lift")
    parser.add_argument("--realtime-s", type=float, default=5.0, help="seconds per phase")
    parser.add_argument("--realtime-task", choices=TASK_NAMES, default="reach")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    results: dict[str, Any] = {"machine": machine()}
    print(json.dumps({"section": "machine", **results["machine"]}), flush=True)
    runs: dict[str, Callable[[], dict[str, Any]]] = {
        "physics": lambda: physics(args.physics_s),
        "render": lambda: render(args.sizes, args.frames),
        "engine": lambda: engine(args.engine_ticks),
        "lockstep": lambda: lockstep(args.tasks, args.episodes, args.camera_size, args.camera_task),
        "realtime": lambda: realtime(args.realtime_s, args.realtime_task),
    }
    for section in SECTIONS:
        if section not in args.sections:
            continue
        data = runs[section]()
        data["rss_mb"] = rss_mb()
        results[section] = data
        print(json.dumps({"section": section, **data}), flush=True)
    for line in summary(results):
        print(line)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
