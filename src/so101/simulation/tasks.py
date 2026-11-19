"""Seeded physics tasks for demos and evaluation: ``reach``, ``pick_lift``, ``pick_place``.

Each task owns a :class:`~so101.simulation.world.SceneSpec`, a seeded reset, a
success test, per-step metrics, a natural-language instruction and a fixed
``observation.environment_state`` layout (:attr:`Task.env_state_names`).

Positions are metres in the world frame: the arm base is the origin, +x points
forward from the base, +z up, and the table top is z = 0. Quaternions are
``(w, x, y, z)``.

Usage (one control step at 30 Hz; ``update`` must run once per step because
the hold timers live in the task)::

    task = make_task("pick_lift")
    world = task.make_world()
    info = task.reset(world, seed=3)
    ...advance physics by 1/30 s...
    metrics = task.update(world)
    done = task.success(world)

Needs the ``[sim]`` extra.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .. import JOINT_NAMES
from .world import (
    MarkerSpec,
    ObjectSpec,
    PhysicsWorld,
    SceneSpec,
    resting_half_height,
    rpy_deg_to_quat,
)

CONTROL_HZ = 30.0
# Every episode starts near this pose: gripperframe about 15 cm above the
# table and 20 cm in front of the base, pointing 55 deg below horizontal (the
# wrist_flex limit rules out straight down at that height), jaws half open.
# It is clear of the table and of every sampled object. ``START_JITTER_DEG``
# is added per arm joint from the episode seed.
READY_POSE: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -60.0,
    "elbow_flex": 30.0,
    "wrist_flex": 85.0,
    "wrist_roll": 0.0,
    "gripper": 30.0,
}
START_JITTER_DEG = 3.0
SETTLE_S = 0.3  # objects come to rest before t = 0

CUBE_SIZE_M = 0.04
CUBE_RGBA = (1.0, 0.72, 0.3, 1.0)  # the scenario cube colour, #ffb84d
TARGET_RADIUS_M = 0.012
TARGET_RGBA = (0.9, 0.15, 0.15, 0.7)
ZONE_SIZE_M = 0.10
ZONE_RGBA = (0.15, 0.75, 0.3, 0.5)

REACH_TOLERANCE_M = 0.02
REACH_DWELL_S = 0.3
LIFT_HEIGHT_M = 0.05
LIFT_HOLD_S = 0.5
PLACE_HOLD_S = 0.3
PLACE_ON_TABLE_M = 0.005  # cube bottom within 5 mm of the table
PLACE_LIFTED_M = 0.02  # pushing the cube into the zone is not placing it
ZONE_SEPARATION_M = 0.12


@dataclass(frozen=True)
class Region:
    """An annular sector of the table: radius from the base axis, azimuth from +x."""

    radius_m: tuple[float, float]
    azimuth_deg: tuple[float, float]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        radius = rng.uniform(*self.radius_m)
        azimuth = math.radians(rng.uniform(*self.azimuth_deg))
        return np.array([radius * math.cos(azimuth), radius * math.sin(azimuth)])


REACH_REGION = Region((0.16, 0.26), (-40.0, 40.0))
REACH_HEIGHT_M = (0.04, 0.16)
OBJECT_REGION = Region((0.16, 0.25), (-35.0, 35.0))
ZONE_REGION = Region((0.15, 0.25), (-40.0, 40.0))


def yaw_from_quat(quat: np.ndarray) -> float:
    """Heading (rad) of a body's x axis projected on the table."""

    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Task(ABC):
    """One task family. Instances are stateful per episode; ``reset`` starts one."""

    name: ClassVar[str]
    env_state_names: ClassVar[tuple[str, ...]]
    episode_s: ClassVar[float]
    instruction: ClassVar[str]

    def __init__(self) -> None:
        self.scene = self.build_scene()
        self.seed: int | None = None
        self.layout: dict[str, Any] = {}
        self.time_s = 0.0
        self.first_success_s: float | None = None
        self._success = False

    # -- definition -------------------------------------------------------------
    @abstractmethod
    def build_scene(self) -> SceneSpec: ...

    @abstractmethod
    def _sample(self, world: PhysicsWorld, rng: np.random.Generator) -> dict[str, Any]:
        """Place the task's objects and markers; return the layout for reports."""

    @abstractmethod
    def _step(self, world: PhysicsWorld, dt: float) -> tuple[dict[str, float], bool]:
        """Per-step metrics and whether the success condition holds now."""

    @abstractmethod
    def env_state(self, world: PhysicsWorld) -> np.ndarray:
        """``observation.environment_state``: float32, layout ``env_state_names``."""

    # -- lifecycle -----------------------------------------------------------------
    @property
    def env_state_dim(self) -> int:
        return len(self.env_state_names)

    def episode_steps(self, control_hz: float = CONTROL_HZ) -> int:
        return round(self.episode_s * control_hz)

    def make_world(self) -> PhysicsWorld:
        return PhysicsWorld(self.scene)

    def reset(self, world: PhysicsWorld, seed: int | None = None) -> dict[str, Any]:
        """Deterministic episode start for ``seed``. Returns the episode info."""

        if world.scene != self.scene:
            raise ValueError(f"world was not built from the {self.name} scene")
        rng = np.random.default_rng(seed)
        start = dict(READY_POSE)
        for joint in JOINT_NAMES[:-1]:
            start[joint] += float(rng.uniform(-START_JITTER_DEG, START_JITTER_DEG))
        world.reset(seed, joints=start)
        self.seed = seed
        self.layout = self._sample(world, rng)
        world.settle(SETTLE_S)
        self.time_s = 0.0
        self.first_success_s = None
        self._success = False
        self._reset_timers()
        return {
            "task": self.name,
            "seed": seed,
            "instruction": self.instruction,
            "start_pose": {joint: round(value, 3) for joint, value in start.items()},
            "layout": self.layout,
            "env_state_names": list(self.env_state_names),
        }

    @abstractmethod
    def _reset_timers(self) -> None:
        """Clear the per-episode hold timers and running metrics."""

    def update(self, world: PhysicsWorld, dt: float = 1.0 / CONTROL_HZ) -> dict[str, float]:
        """Advance the task clock by one control step; return its metrics."""

        self.time_s += dt
        metrics, holding = self._step(world, dt)
        self._success = holding
        if holding and self.first_success_s is None:
            self.first_success_s = round(self.time_s, 4)
        metrics["success"] = float(holding)
        return metrics

    def success(self, world: PhysicsWorld | None = None) -> bool:
        """Whether the success condition held at the last :meth:`update`."""

        return self._success

    def summary(self) -> dict[str, Any]:
        """Episode-level metrics (subclasses add theirs)."""

        return {"time_s": round(self.time_s, 4), "first_success_s": self.first_success_s}


# -- reach --------------------------------------------------------------------------
class ReachTask(Task):
    """Bring ``gripperframe`` within 2 cm of a red ball and stay there for 0.3 s."""

    name = "reach"
    env_state_names = ("target_x", "target_y", "target_z")
    episode_s = 6.0
    instruction = "move the gripper to the red ball"

    def build_scene(self) -> SceneSpec:
        return SceneSpec(markers=(MarkerSpec("target", "sphere", (TARGET_RADIUS_M,), TARGET_RGBA),))

    def _sample(self, world: PhysicsWorld, rng: np.random.Generator) -> dict[str, Any]:
        xy = REACH_REGION.sample(rng)
        target = np.array([xy[0], xy[1], rng.uniform(*REACH_HEIGHT_M)])
        world.set_marker("target", target)
        self.target = target
        return {"target_m": [round(float(v), 4) for v in target]}

    def _reset_timers(self) -> None:
        self.dwell_s = 0.0
        self.min_distance_m = math.inf

    def _step(self, world: PhysicsWorld, dt: float) -> tuple[dict[str, float], bool]:
        distance = float(np.linalg.norm(world.site_pose()[0] - self.target))
        within = distance <= REACH_TOLERANCE_M
        self.dwell_s = self.dwell_s + dt if within else 0.0
        self.min_distance_m = min(self.min_distance_m, distance)
        metrics = {"distance_m": distance, "within": float(within), "dwell_s": self.dwell_s}
        return metrics, self.dwell_s >= REACH_DWELL_S - 1e-9

    def env_state(self, world: PhysicsWorld) -> np.ndarray:
        return world.marker_position("target").astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return super().summary() | {"min_distance_m": round(self.min_distance_m, 4)}


# -- pick and lift ------------------------------------------------------------------------
class _CubeTask(Task):
    cube_half_height = resting_half_height("box", (CUBE_SIZE_M,))

    def _cube_scene(self, markers: tuple[MarkerSpec, ...] = ()) -> SceneSpec:
        cube = ObjectSpec("cube", "box", (CUBE_SIZE_M,), rgba=CUBE_RGBA)
        return SceneSpec(objects=(cube,), markers=markers)

    def _place_cube(self, xy: np.ndarray, yaw_deg: float, world: PhysicsWorld) -> dict[str, Any]:
        position = np.array([xy[0], xy[1], self.cube_half_height + 5e-4])
        world.set_object_pose("cube", position, rpy_deg_to_quat(0.0, 0.0, yaw_deg))
        return {
            "cube_m": [round(float(v), 4) for v in position[:2]],
            "cube_yaw_deg": round(yaw_deg, 3),
        }

    def _cube_lift(self, world: PhysicsWorld) -> float:
        return float(world.object_pose("cube")[0][2]) - self.cube_half_height

    def _cube_state(self, world: PhysicsWorld) -> np.ndarray:
        position, quat = world.object_pose("cube")
        return np.concatenate([position, quat])


class PickLiftTask(_CubeTask):
    """Lift the cube at least 5 cm off the table and hold it there for 0.5 s."""

    name = "pick_lift"
    env_state_names = ("cube_x", "cube_y", "cube_z", "cube_qw", "cube_qx", "cube_qy", "cube_qz")
    episode_s = 10.0
    instruction = "pick up the orange cube and lift it"

    def build_scene(self) -> SceneSpec:
        return self._cube_scene()

    def _sample(self, world: PhysicsWorld, rng: np.random.Generator) -> dict[str, Any]:
        xy = OBJECT_REGION.sample(rng)
        return self._place_cube(xy, float(rng.uniform(-45.0, 45.0)), world)

    def _reset_timers(self) -> None:
        self.lifted_s = 0.0
        self.max_lift_m = 0.0
        self.grasp_steps = 0

    def _step(self, world: PhysicsWorld, dt: float) -> tuple[dict[str, float], bool]:
        lift = self._cube_lift(world)
        grasped = bool(world.contacts()["objects"]["cube"]["grasped"])
        self.lifted_s = self.lifted_s + dt if lift >= LIFT_HEIGHT_M else 0.0
        self.max_lift_m = max(self.max_lift_m, lift)
        self.grasp_steps += grasped
        tool = float(np.linalg.norm(world.site_pose()[0] - world.object_pose("cube")[0]))
        metrics = {
            "lift_m": lift,
            "grasped": float(grasped),
            "lifted_s": self.lifted_s,
            "tool_to_cube_m": tool,
        }
        return metrics, self.lifted_s >= LIFT_HOLD_S - 1e-9

    def env_state(self, world: PhysicsWorld) -> np.ndarray:
        return self._cube_state(world).astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return super().summary() | {
            "max_lift_m": round(self.max_lift_m, 4),
            "grasp_s": round(self.grasp_steps / CONTROL_HZ, 4),
        }


# -- pick and place ---------------------------------------------------------------------------
class PickPlaceTask(_CubeTask):
    """Carry the cube into the green square, put it down and let go (0.3 s)."""

    name = "pick_place"
    env_state_names = PickLiftTask.env_state_names + ("zone_x", "zone_y", "zone_z")
    episode_s = 16.0
    instruction = "put the orange cube in the green square"

    def build_scene(self) -> SceneSpec:
        zone = MarkerSpec("zone", "box", (ZONE_SIZE_M, ZONE_SIZE_M, 0.002), ZONE_RGBA)
        return self._cube_scene((zone,))

    def _sample(self, world: PhysicsWorld, rng: np.random.Generator) -> dict[str, Any]:
        cube = OBJECT_REGION.sample(rng)
        zone = ZONE_REGION.sample(rng)
        for _ in range(1000):  # rejection sampling stays deterministic per seed
            if np.linalg.norm(zone - cube) >= ZONE_SEPARATION_M:
                break
            zone = ZONE_REGION.sample(rng)
        else:  # pragma: no cover - the regions make this vanishingly unlikely
            zone = np.array([cube[0], -cube[1] or ZONE_SEPARATION_M])
        layout = self._place_cube(cube, float(rng.uniform(-45.0, 45.0)), world)
        self.zone = np.array([zone[0], zone[1], 0.001])
        world.set_marker("zone", self.zone)
        return layout | {"zone_m": [round(float(v), 4) for v in self.zone[:2]]}

    def _reset_timers(self) -> None:
        self.placed_s = 0.0
        self.max_lift_m = 0.0
        self.lifted = False

    def _step(self, world: PhysicsWorld, dt: float) -> tuple[dict[str, float], bool]:
        position = world.object_pose("cube")[0]
        lift = self._cube_lift(world)
        self.max_lift_m = max(self.max_lift_m, lift)
        self.lifted = self.lifted or lift >= PLACE_LIFTED_M
        offset = position[:2] - self.zone[:2]
        in_zone = bool(np.all(np.abs(offset) <= ZONE_SIZE_M / 2.0))
        contacts = world.contacts()["objects"]["cube"]
        released = not (contacts["fixed_jaw"] or contacts["moving_jaw"] or contacts["arm"])
        on_table = lift <= PLACE_ON_TABLE_M and contacts["table"] > 0
        placed = self.lifted and in_zone and on_table and released
        self.placed_s = self.placed_s + dt if placed else 0.0
        metrics = {
            "zone_distance_m": float(np.linalg.norm(offset)),
            "in_zone": float(in_zone),
            "lift_m": lift,
            "released": float(released),
            "placed_s": self.placed_s,
        }
        return metrics, self.placed_s >= PLACE_HOLD_S - 1e-9

    def env_state(self, world: PhysicsWorld) -> np.ndarray:
        state = np.concatenate([self._cube_state(world), world.marker_position("zone")])
        return state.astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return super().summary() | {"max_lift_m": round(self.max_lift_m, 4)}


TASKS: dict[str, type[Task]] = {
    ReachTask.name: ReachTask,
    PickLiftTask.name: PickLiftTask,
    PickPlaceTask.name: PickPlaceTask,
}
TASK_NAMES: tuple[str, ...] = tuple(TASKS)


def make_task(name: str) -> Task:
    try:
        return TASKS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown sim task {name!r}; choose from {list(TASK_NAMES)}") from exc


__all__ = [
    "CONTROL_HZ",
    "READY_POSE",
    "TASKS",
    "TASK_NAMES",
    "PickLiftTask",
    "PickPlaceTask",
    "ReachTask",
    "Region",
    "Task",
    "make_task",
    "yaw_from_quat",
]
