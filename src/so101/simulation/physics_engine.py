"""MuJoCo engine behind ``/api/simulation``, with the same contract as the kinematic engine.

:class:`PhysicsSimulationEngine` shares its request semantics with
:class:`~so101.simulation.engine.SimulationEngine`:

- fixed 60 Hz ticks, advanced by the same request-synchronized monotonic
  scheduler;
- the same target validation (rejected, never clamped);
- the same preview planner and step/play/pause/reset rules.

Only the truth model differs. Every tick is 4 MuJoCo substeps of 1/240 s of
the vendored Menagerie SO-101, with gravity, contacts, friction and the
STS3215 servo profile. Scene objects are free rigid bodies; target zones are
visual-only markers.

- ``telemetry.measured`` and ``telemetry.velocity`` are the simulated joint
  angles and rates, including gravity sag, contact and servo lag.
  ``telemetry.target`` is the exact commanded value.
- ``scene.objects[].pose`` of a rigid body is its live pose.
- ``physics_state`` carries contacts, effort, object velocities, the servo
  configuration and dropped ticks.
- Trajectory preview runs the same ticks on a :meth:`PhysicsWorld.fork`, so
  it predicts manual steps and play exactly and never touches live state.
- :meth:`PhysicsSimulationEngine.render_frame` renders a camera from a
  snapshot copied under the lock, on the shared render thread.

Play never runs a background thread here. If nobody observes the simulator
for a long time, the next request simulates at most ``MAX_CATCHUP_WALL_S`` of
wall time (times the playback rate). The rest is skipped and counted in
``physics_state.dropped_ticks``, and ``sim_time_s`` keeps counting only what
was simulated. With a constant target the scene is at rest by then anyway.

Needs the ``[sim]`` extra. Import this module only once
:func:`so101.simulation.selection.mujoco_status` reports MuJoCo usable.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import mujoco
import numpy as np

from .. import JOINT_NAMES
from .engine import (
    SimulationConflict,
    SimulationValidationError,
    plan_preview,
    validate_targets,
)
from .kinematics import JOINT_LIMITS, MODEL_ID, forward_kinematics
from .render import check_size, shared_renderer
from .scenarios import ScenarioLibrary
from .servo import (
    SERVO_MODES,
    STS3215_ACCELERATION_DEG_S2,
    STS3215_MAX_SPEED_DEG_S,
    ServoDriver,
    ServoMode,
)
from .units import mjcf_to_suite_array, velocity_mjcf_to_suite, velocity_suite_to_mjcf
from .world import (
    DEFAULT_TIMESTEP_S,
    FRONT_CAMERA,
    TABLE_HEIGHT_M,
    TABLE_SIZE_M,
    MarkerSpec,
    ObjectSpec,
    PhysicsWorld,
    SceneSpec,
    hex_to_rgba,
    quat_to_rpy_deg,
    rpy_deg_to_quat,
)

TARGET_ZONE_ROLE = "target_zone"
MARKER_ALPHA = 0.6
RESET_SETTLE_S = 0.25  # contacts settle before tick 0; the clock restarts at 0
PHYSICS_SCENE_DETAIL = (
    "MuJoCo rigid-body scene for the isolated physics backend: gravity, contacts and "
    "STS3215 servo dynamics. Target zones are visual-only."
)


def _r(value: float, digits: int = 9) -> float:
    return round(float(value), digits) + 0.0  # ``+ 0.0`` folds -0.0 into 0.0


@dataclass(frozen=True)
class Placement:
    """Where reset puts one scene object (world frame, table top at z = 0)."""

    object_id: str
    marker: bool
    position_m: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


def scene_spec_for(
    scene: Mapping[str, Any], *, timestep_s: float = DEFAULT_TIMESTEP_S
) -> tuple[SceneSpec, tuple[Placement, ...]]:
    """The physics scene for an instantiated scenario, plus its reset poses.

    ``target_zone`` objects become visual-only markers. Every other object is
    a free rigid body with the manifest's shape, size and colour. A scenario
    ``pose`` is ``[x, y, z, roll, pitch, yaw]`` in metres and degrees, with
    ``z`` the centre height above the table, which is the world frame here.
    """

    objects: list[ObjectSpec] = []
    markers: list[MarkerSpec] = []
    placements: list[Placement] = []
    for item in scene.get("objects", ()):
        object_id = str(item["id"])
        shape = item["shape"]
        size = tuple(float(value) for value in item["size_m"])
        marker = item.get("role") == TARGET_ZONE_ROLE
        if marker:
            markers.append(
                MarkerSpec(object_id, shape, size, hex_to_rgba(item.get("color"), MARKER_ALPHA))
            )
        else:
            objects.append(ObjectSpec(object_id, shape, size, hex_to_rgba(item.get("color"))))
        pose = [float(value) for value in item["pose"]]
        quat = rpy_deg_to_quat(pose[3], pose[4], pose[5])
        placements.append(
            Placement(
                object_id,
                marker,
                (pose[0], pose[1], pose[2]),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
        )
    surface = scene.get("surface") or {}
    size_m = surface.get("size_m", TABLE_SIZE_M)
    spec = SceneSpec(
        objects=tuple(objects),
        markers=tuple(markers),
        table_size_m=(float(size_m[0]), float(size_m[1]), float(size_m[2])),
        table_height_m=float(surface.get("height_m", TABLE_HEIGHT_M)),
        timestep_s=timestep_s,
    )
    return spec, tuple(placements)


def servo_speed_limits(max_velocity: Mapping[str, float]) -> dict[str, float]:
    """Per-joint servo top speed in MJCF deg/s.

    The scenario manifest's commanded-reference slew limit (suite units/s,
    shared with the kinematic engine), capped by the STS3215's 270 deg/s.
    """

    limits: dict[str, float] = {}
    for joint in JOINT_NAMES:
        mjcf_deg_s = math.degrees(velocity_suite_to_mjcf(joint, float(max_velocity[joint])))
        limits[joint] = min(mjcf_deg_s, STS3215_MAX_SPEED_DEG_S)
    return limits


class PhysicsSimulationEngine:
    """MuJoCo runtime with the ``SimulationEngine`` request contract. Thread-safe."""

    MAX_MANUAL_STEPS = 10_000
    MAX_CATCHUP_WALL_S = 2.0
    # ``final_reached`` tolerance, suite units (deg; gripper 0..100). Gravity
    # and contacts leave a small steady-state error, so exact equality never holds.
    REACHED_TOLERANCE = 0.5
    BACKEND = "mujoco"

    def __init__(
        self,
        library: ScenarioLibrary | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        servo_mode: ServoMode = "sts3215",
    ) -> None:
        if servo_mode not in SERVO_MODES:
            raise ValueError(f"servo mode must be one of {SERVO_MODES}, got {servo_mode!r}")
        self.library = library or ScenarioLibrary()
        self._clock = clock
        self._lock = threading.RLock()
        engine = self.library.engine
        self._fixed_hz = float(engine.fixed_hz)
        self._fixed_dt_s = 1.0 / self._fixed_hz
        self._substeps = max(1, round(self._fixed_dt_s / DEFAULT_TIMESTEP_S))
        self._timestep_s = self._fixed_dt_s / self._substeps
        self._servo_mode: ServoMode = servo_mode
        self._speed_limits = servo_speed_limits(engine.max_velocity_deg_s)
        self._initial = dict(engine.initial_joint_positions)
        scene = self.library.instantiate(
            next(iter(item["id"] for item in self.library.catalog())),
            seed=engine.default_seed,
            randomize=True,
        )
        self._world, self._driver = self._build(scene)
        self._scene = scene
        self._target = dict(self._initial)
        self._step_count = 0
        self._dropped_ticks = 0
        self._playing = False
        self._playback_rate = 1.0
        self._clock_anchor = self._clock()
        self._generation = 0
        self._updated_at = datetime.now(timezone.utc)

    # -- configuration -----------------------------------------------------------
    @property
    def fixed_dt_s(self) -> float:
        return self._fixed_dt_s

    @property
    def substeps_per_tick(self) -> int:
        return self._substeps

    @property
    def timestep_s(self) -> float:
        return self._timestep_s

    @property
    def servo_mode(self) -> str:
        return self._servo_mode

    @property
    def cameras(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._world.index.cameras)

    def catalog(self) -> list[dict[str, Any]]:
        """Scenario entries labelled with this engine (``/api/simulation/scenarios``)."""

        return self.library.catalog(backend=self.BACKEND, physics=True, detail=PHYSICS_SCENE_DETAIL)

    def truth_model(self) -> str:
        return (
            "MuJoCo rigid-body physics of the vendored Menagerie SO-101 MJCF: gravity, "
            "contacts and friction, STS3215 position servos (kp 998 N m/rad, 2.94 N m; "
            "gripper 1.47 N m) behind a trapezoidal acceleration and speed profile, "
            f"{self._fixed_hz:g} Hz ticks of {self._substeps} x {self._timestep_s:.6g} s "
            "implicitfast substeps. Scene objects are free rigid bodies; target zones "
            "are visual-only."
        )

    # -- building ------------------------------------------------------------------
    def _build(self, scene: Mapping[str, Any]) -> tuple[PhysicsWorld, ServoDriver]:
        spec, placements = scene_spec_for(scene, timestep_s=self._timestep_s)
        world = PhysicsWorld(spec)
        world.reset(int(scene.get("seed") or 0), joints=self._initial)
        for placement in placements:
            if placement.marker:
                world.set_marker(placement.object_id, placement.position_m, placement.quat_wxyz)
            else:
                world.set_object_pose(
                    placement.object_id, placement.position_m, placement.quat_wxyz
                )
        world.settle(RESET_SETTLE_S, rewind_clock=True)
        driver = ServoDriver(
            world,
            mode=self._servo_mode,
            max_speed_deg_s=[self._speed_limits[joint] for joint in JOINT_NAMES],
        )
        return world, driver

    # -- internals ---------------------------------------------------------------------
    def _touch_locked(self) -> None:
        self._updated_at = datetime.now(timezone.utc)

    def _set_targets_locked(self, targets: Mapping[str, Any]) -> None:
        validated = validate_targets(targets)
        self._target.update(validated)
        # Always the full validated map, so the goal is bit-identical however
        # the same target was reached (manual step, play or preview).
        self._driver.command(self._target)
        self._touch_locked()

    def _advance_locked(self, ticks: int) -> None:
        if ticks <= 0:
            return
        self._driver.step(ticks * self._substeps)
        self._step_count += ticks
        self._touch_locked()

    def _sync_playback_locked(self) -> None:
        if not self._playing:
            return
        now = self._clock()
        elapsed = max(0.0, now - self._clock_anchor) * self._playback_rate
        ticks = int((elapsed + 1e-12) / self._fixed_dt_s)
        if ticks <= 0:
            return
        limit = max(1, math.ceil(self.MAX_CATCHUP_WALL_S * self._playback_rate * self._fixed_hz))
        simulated = min(ticks, limit)
        self._advance_locked(simulated)
        self._dropped_ticks += ticks - simulated
        # Retain sub-tick elapsed time rather than resetting to ``now``.
        self._clock_anchor += (ticks * self._fixed_dt_s) / self._playback_rate

    def _physics_state_locked(self, contacts: Mapping[str, Any]) -> dict[str, Any]:
        world = self._world
        objects: dict[str, Any] = {}
        for object_id in world.index.object_ids:
            position, quat = world.object_pose(object_id)
            linear, _ = world.object_velocity(object_id)
            touching = contacts["objects"][object_id]
            objects[object_id] = {
                "position_m": [_r(value) for value in position],
                "quat_wxyz": [_r(value) for value in quat],
                "speed_m_s": _r(float(np.linalg.norm(linear))),
                "grasped": bool(touching["grasped"]),
                "contacts": {
                    key: int(touching[key])
                    for key in ("table", "fixed_jaw", "moving_jaw", "arm", "objects")
                },
            }
        reference = mjcf_to_suite_array(self._driver.profile.reference)
        return {
            "engine": self.BACKEND,
            "mujoco_version": str(mujoco.__version__),
            "model": MODEL_ID,
            "timestep_s": self._timestep_s,
            "substeps_per_tick": self._substeps,
            "physics_steps": self._step_count * self._substeps,
            "physics_time_s": _r(world.time_s),
            "dropped_ticks": self._dropped_ticks,
            "torque_enabled": bool(world.torque_enabled),
            "servo": {
                "mode": self._servo_mode,
                "acceleration_deg_s2": (
                    None if self._servo_mode == "step" else _r(STS3215_ACCELERATION_DEG_S2, 3)
                ),
                "max_speed": {
                    joint: _r(velocity_mjcf_to_suite(joint, math.radians(value)), 3)
                    for joint, value in self._speed_limits.items()
                },
                "speed_units": "suite_units_per_second",
                "reference": {
                    joint: _r(reference[index], 6) for index, joint in enumerate(JOINT_NAMES)
                },
            },
            "effort_pct": {
                joint: _r(value, 3)
                for joint, value in zip(JOINT_NAMES, world.effort_pct(), strict=True)
            },
            "contacts": {
                "total": int(contacts["total"]),
                "arm_self": int(contacts["arm_self"]),
                "arm_table": int(contacts["arm_table"]),
                "max_penetration_m": _r(contacts["max_penetration_m"]),
            },
            "objects": objects,
            "markers": {
                marker_id: [_r(value) for value in world.marker_position(marker_id)]
                for marker_id in world.index.marker_mocap
            },
        }

    def _live_scene_locked(self) -> dict[str, Any]:
        scene = copy.deepcopy(self._scene)
        bodies = set(self._world.index.object_ids)
        for item in scene.get("objects", ()):
            if item["id"] not in bodies:
                continue
            position, quat = self._world.object_pose(item["id"])
            roll, pitch, yaw = quat_to_rpy_deg(quat)
            item["pose"] = [
                _r(position[0], 6),
                _r(position[1], 6),
                _r(position[2], 6),
                _r(roll, 4),
                _r(pitch, 4),
                _r(yaw, 4),
            ]
        return scene

    def _snapshot_locked(self) -> dict[str, Any]:
        world = self._world
        joints = world.joint_array()
        rates = world.velocity_array()
        measured = {joint: _r(joints[index]) for index, joint in enumerate(JOINT_NAMES)}
        target = {joint: _r(self._target[joint]) for joint in JOINT_NAMES}
        velocity = {joint: _r(rates[index]) for index, joint in enumerate(JOINT_NAMES)}
        error = {joint: _r(target[joint] - measured[joint]) for joint in JOINT_NAMES}
        contacts = world.contacts()
        return {
            "backend": self.BACKEND,
            "backend_kind": "physics",
            "physics": True,
            "truth_model": self.truth_model(),
            "hardware_isolated": True,
            "writes_motor_commands": False,
            "status": "playing" if self._playing else "paused",
            "scheduler": "request_synchronized_monotonic_fixed_step",
            "playback_rate": self._playback_rate,
            "fixed_dt_s": self._fixed_dt_s,
            "fixed_hz": self.library.engine.fixed_hz,
            "step_count": self._step_count,
            "sim_time_s": round(self._step_count * self._fixed_dt_s, 12),
            "generation": self._generation,
            "updated_at": self._updated_at,
            "joint_order": list(JOINT_NAMES),
            "joint_limits_deg": {
                joint: [JOINT_LIMITS[joint][0], JOINT_LIMITS[joint][1]] for joint in JOINT_NAMES
            },
            "telemetry": {
                "source": "simulation",
                "position_units": "degrees",
                "velocity_units": "degrees_per_second",
                "measured": measured,
                "target": target,
                "error": error,
                "velocity": velocity,
            },
            "kinematics": forward_kinematics(measured),
            "scene": self._live_scene_locked(),
            "physics_state": self._physics_state_locked(contacts),
        }

    # -- the SimulationEngine contract ----------------------------------------------------
    def state(self) -> dict[str, Any]:
        with self._lock:
            self._sync_playback_locked()
            return self._snapshot_locked()

    def telemetry(self) -> dict[str, Any]:
        return self.state()["telemetry"]

    def set_targets(self, targets: dict[str, float]) -> dict[str, Any]:
        with self._lock:
            self._sync_playback_locked()
            self._set_targets_locked(targets)
            return self._snapshot_locked()

    def step(
        self,
        steps: int = 1,
        *,
        targets: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if not 1 <= steps <= self.MAX_MANUAL_STEPS:
            raise SimulationValidationError(f"steps must be between 1 and {self.MAX_MANUAL_STEPS}")
        with self._lock:
            if self._playing:
                raise SimulationConflict("pause simulation before using manual step")
            if targets is not None:
                self._set_targets_locked(targets)
            self._advance_locked(steps)
            return self._snapshot_locked()

    def play(self, rate: float = 1.0) -> dict[str, Any]:
        rate = float(rate)
        if not math.isfinite(rate) or not 0.1 <= rate <= 4.0:
            raise SimulationValidationError("playback rate must be between 0.1 and 4.0")
        with self._lock:
            self._sync_playback_locked()
            self._playback_rate = rate
            self._clock_anchor = self._clock()
            self._playing = True
            self._touch_locked()
            return self._snapshot_locked()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._sync_playback_locked()
            self._playing = False
            self._clock_anchor = self._clock()
            self._touch_locked()
            return self._snapshot_locked()

    def reset(
        self,
        scenario_id: str,
        *,
        seed: int | None = None,
        randomize: bool = True,
    ) -> dict[str, Any]:
        # Instantiate and build first: a bad scenario, manifest or model leaves
        # the current state intact, and the lock is not held while compiling.
        scene = self.library.instantiate(scenario_id, seed=seed, randomize=randomize)
        world, driver = self._build(scene)
        with self._lock:
            self._world, self._driver = world, driver
            self._scene = scene
            self._playing = False
            self._playback_rate = 1.0
            self._clock_anchor = self._clock()
            self._target = dict(self._initial)
            self._step_count = 0
            self._dropped_ticks = 0
            self._generation += 1
            self._touch_locked()
            return self._snapshot_locked()

    def preview_trajectory(
        self,
        *,
        target: dict[str, float] | None = None,
        duration_s: float | None = None,
        waypoints: list[dict[str, Any]] | None = None,
        sample_hz: float = 20.0,
    ) -> dict[str, Any]:
        """Simulate the request on a fork of the live world. Never mutates state.

        The fork gets the same commands at the same ticks as ``set_targets``
        plus ``step`` (see :class:`~so101.simulation.engine.PreviewPlan`), so
        every sampled point equals what manual steps or play would produce.
        """

        with self._lock:
            self._sync_playback_locked()
            world = self._world.fork()
            driver = self._driver.fork(world)
            state_target = dict(self._target)
            source_step = self._step_count
        plan = plan_preview(
            fixed_hz=self._fixed_hz,
            current_target=state_target,
            target=target,
            duration_s=duration_s,
            waypoints=waypoints,
            sample_hz=sample_hz,
        )
        points = [self._point(0, world.joint_array(), plan.target_at(0))]
        commanded: dict[str, float] | None = None
        for tick in range(1, plan.total_steps + 1):
            wanted = plan.target_at(tick)
            if wanted is not commanded:
                driver.command(wanted)
                commanded = wanted
            driver.step(self._substeps)
            if plan.sampled(tick):
                points.append(self._point(tick, world.joint_array(), wanted))
        final_target = plan.segments[-1][1]
        final = world.joint_array()
        reached = all(
            abs(final_target[joint] - float(final[index])) <= self.REACHED_TOLERANCE
            for index, joint in enumerate(JOINT_NAMES)
        )
        return {
            "backend": self.BACKEND,
            "physics": True,
            "mutates_state": False,
            "semantics": "step_targets_like_play",
            "source_step_count": source_step,
            "fixed_dt_s": self._fixed_dt_s,
            "requested_duration_s": plan.duration_s,
            "quantized_duration_s": round(plan.total_steps * self._fixed_dt_s, 12),
            "requested_sample_hz": plan.requested_sample_hz,
            "sample_hz": plan.sample_hz,
            "points": points,
            "final_reached": reached,
            "reached_tolerance": self.REACHED_TOLERANCE,
            "joint_limits_deg": {
                joint: [JOINT_LIMITS[joint][0], JOINT_LIMITS[joint][1]] for joint in JOINT_NAMES
            },
            "starting_target": state_target,
        }

    def _point(
        self, step: int, joints: np.ndarray, commanded: Mapping[str, float]
    ) -> dict[str, Any]:
        measured = {joint: _r(joints[index]) for index, joint in enumerate(JOINT_NAMES)}
        target = {joint: _r(commanded[joint]) for joint in JOINT_NAMES}
        return {
            "step": step,
            "time_s": round(step * self._fixed_dt_s, 12),
            "measured": measured,
            "target": target,
            "error": {joint: _r(target[joint] - measured[joint]) for joint in JOINT_NAMES},
            "end_effector": forward_kinematics(measured),
        }

    # -- rendering ----------------------------------------------------------------------------
    def render_frame(
        self,
        camera: str = FRONT_CAMERA,
        width: int = 320,
        height: int = 240,
        *,
        timeout: float = 10.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """An HxWx3 uint8 frame of the live scene, plus what it shows.

        Raises :class:`SimulationValidationError` for a bad camera or size,
        :class:`~so101.simulation.render.RenderUnavailable` without a GL
        context, and ``TimeoutError`` if the render thread is stuck.
        """

        try:
            width, height = check_size(width, height)
        except ValueError as exc:
            raise SimulationValidationError(str(exc)) from exc
        with self._lock:
            self._sync_playback_locked()
            cameras = tuple(self._world.index.cameras)
            if camera not in cameras:
                raise SimulationValidationError(
                    f"unknown camera {camera!r}; available: {', '.join(cameras)}"
                )
            snapshot = self._world.snapshot()
            meta = {
                "camera": camera,
                "width": width,
                "height": height,
                "step_count": self._step_count,
                "generation": self._generation,
                "sim_time_s": round(self._step_count * self._fixed_dt_s, 12),
            }
        frame = shared_renderer().render(snapshot, camera, width, height, timeout=timeout)
        return frame, meta


__all__ = [
    "PHYSICS_SCENE_DETAIL",
    "PhysicsSimulationEngine",
    "Placement",
    "scene_spec_for",
    "servo_speed_limits",
]
