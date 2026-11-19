"""STS3215-like goal tracking for the MuJoCo position actuators.

A Feetech STS3215 does not jump to a new ``Goal_Position``. It runs a
trapezoidal move limited by its ``Acceleration`` register and its top speed.
LeRobot writes ``Acceleration = 254`` on every SO-101 motor
(``configure_motors``): 254 x 100 steps/s^2 x 360/4096 deg/step = 2232 deg/s^2.
The 7.4 V STS3215 turns at about 0.222 s per 60 deg no-load, so 270 deg/s.

:class:`ServoProfile` reproduces that move in MJCF radians, once per physics
substep, and the result becomes the actuator ``ctrl``. Without it, MuJoCo's
stiff position actuator (kp 998 N m/rad, saturating at 2.94 N m, so above
about 0.17 deg of error) turns every goal change into a full-torque step.

This module is pure numpy and does not import MuJoCo.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .. import JOINT_NAMES
from .units import effective_limit_arrays, mjcf_to_suite_array, suite_to_mjcf_array

STS3215_ACCELERATION_REGISTER = 254
STS3215_STEPS_PER_REV = 4096
STS3215_ACCELERATION_DEG_S2 = STS3215_ACCELERATION_REGISTER * 100 * 360.0 / STS3215_STEPS_PER_REV
STS3215_MAX_SPEED_DEG_S = 60.0 / 0.222  # 7.4 V no-load speed, about 270 deg/s

PerJoint = float | Sequence[float] | np.ndarray  # a scalar broadcasts to every joint


class ServoProfile:
    """Per-joint acceleration- and speed-limited tracking of a goal, in radians.

    ``reference`` is what the servo is currently commanding (the actuator
    ``ctrl``); ``goal`` is the last written goal. ``step(dt)`` moves the
    reference towards the goal along a time-optimal trapezoid: accelerate at
    ``acceleration``, cruise at ``max_speed``, brake to stop on the goal. A goal
    change in mid-move is re-planned from the current reference speed, as the
    servo does.

    ``acceleration=math.inf`` disables the profile (the reference jumps to the
    goal), which is the ``step`` servo mode.
    """

    def __init__(
        self,
        size: int,
        *,
        acceleration: PerJoint = math.radians(STS3215_ACCELERATION_DEG_S2),
        max_speed: PerJoint = math.radians(STS3215_MAX_SPEED_DEG_S),
    ) -> None:
        self.size = int(size)
        self.acceleration = np.broadcast_to(
            np.asarray(acceleration, dtype=float), (self.size,)
        ).copy()
        self.max_speed = np.broadcast_to(np.asarray(max_speed, dtype=float), (self.size,)).copy()
        if np.any(self.acceleration <= 0.0) or np.any(self.max_speed <= 0.0):
            raise ValueError("acceleration and max_speed must be positive")
        self.reference = np.zeros(self.size)
        self.velocity = np.zeros(self.size)
        self.goal = np.zeros(self.size)

    @property
    def enabled(self) -> bool:
        return bool(np.all(np.isfinite(self.acceleration)))

    def reset(self, position: Sequence[float] | np.ndarray) -> None:
        """Stand still at ``position``: reference = goal = position, speed 0."""

        values = np.asarray(position, dtype=float).reshape(self.size)
        self.reference = values.copy()
        self.goal = values.copy()
        self.velocity = np.zeros(self.size)

    def set_goal(self, goal: Sequence[float] | np.ndarray) -> None:
        self.goal = np.asarray(goal, dtype=float).reshape(self.size).copy()

    def settled(self, tolerance: float = 1e-9) -> bool:
        return bool(
            np.all(np.abs(self.goal - self.reference) <= tolerance) and np.all(self.velocity == 0.0)
        )

    def step(self, dt: float) -> np.ndarray:
        """Advance the reference by ``dt`` seconds and return it."""

        if dt <= 0.0:
            return self.reference
        error = self.goal - self.reference
        if not self.enabled:
            self.reference = self.goal.copy()
            self.velocity = np.zeros(self.size)
            return self.reference
        direction = np.sign(error)
        # Fastest speed from which the servo can still brake onto the goal.
        braking = np.sqrt(2.0 * self.acceleration * np.abs(error))
        desired = direction * np.minimum(self.max_speed, braking)
        limit = self.acceleration * dt
        velocity = self.velocity + np.clip(desired - self.velocity, -limit, limit)
        reference = self.reference + velocity * dt
        # Crossing the goal slowly is arriving: snap onto it and stop. A fast
        # crossing (the goal moved under a moving reference) overshoots and
        # comes back, as the servo does.
        crossed = (direction != 0.0) & (np.sign(self.goal - reference) != direction)
        arrived = crossed & (np.abs(velocity) <= 2.0 * limit)
        reference[arrived] = self.goal[arrived]
        velocity[arrived] = 0.0
        self.reference, self.velocity = reference, velocity
        return self.reference


ServoMode = Literal["sts3215", "step"]
SERVO_MODES: tuple[str, ...] = ("sts3215", "step")


class ServoDriver:
    """Drives a :class:`~so101.simulation.world.PhysicsWorld` arm through a profile.

    ``command`` sets the goal (suite units, clipped to the effective limits);
    ``step``/``advance`` run physics substeps, updating the actuator ``ctrl``
    from the profile before each one. With the profile at rest the substeps
    are batched into one ``mj_step`` call, so holding still costs nothing.

    ``mode="step"`` writes the goal straight into ``ctrl`` (no profile), which
    is what :meth:`PhysicsWorld.set_targets` does.
    """

    def __init__(
        self,
        world: Any,
        *,
        mode: ServoMode = "sts3215",
        acceleration_deg_s2: float = STS3215_ACCELERATION_DEG_S2,
        max_speed_deg_s: PerJoint = STS3215_MAX_SPEED_DEG_S,
    ) -> None:
        if mode not in SERVO_MODES:
            raise ValueError(f"servo mode must be one of {SERVO_MODES}, got {mode!r}")
        self.world = world
        self.mode = mode
        acceleration = math.inf if mode == "step" else math.radians(acceleration_deg_s2)
        speeds = np.radians(np.broadcast_to(np.asarray(max_speed_deg_s, dtype=float), (6,)))
        self.profile = ServoProfile(len(JOINT_NAMES), acceleration=acceleration, max_speed=speeds)
        self._low, self._high = effective_limit_arrays()
        self.sync()

    def fork(self, world: Any) -> ServoDriver:
        """An independent driver for ``world``, a :meth:`PhysicsWorld.fork` of this one's.

        The profile (reference, speed, goal) is copied, so the fork continues a
        move exactly where this driver is: same commands, same result.
        """

        twin = ServoDriver.__new__(ServoDriver)
        twin.world = world
        twin.mode = self.mode
        twin.profile = copy.deepcopy(self.profile)
        twin._low, twin._high = self._low, self._high
        return twin

    def sync(self) -> None:
        """Adopt the world's current ``ctrl`` as reference and goal (after reset/torque-on)."""

        self.profile.reset(self.world.data.ctrl[self.world.index.actuators])

    def _resolve(
        self, targets: Mapping[str, float] | Sequence[float] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The goal as actuator radians and in suite units.

        Values inside both the effective limits and the actuator range come
        back exactly as given. Only clipped joints carry the round-trip
        rounding (about 1e-14), so a value clamped by the SafetyGuard never
        reads back as just outside the guard's range.
        """

        suite = np.clip(self.world.joint_vector(targets, self.goal_array()), self._low, self._high)
        unclipped = suite_to_mjcf_array(suite)
        radians = np.clip(unclipped, self.world.index.ctrl_low, self.world.index.ctrl_high)
        return radians, np.where(radians == unclipped, suite, mjcf_to_suite_array(radians))

    def command(self, targets: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
        """Set the goal. A mapping may be partial. Returns the applied goal, suite units."""

        radians, suite = self._resolve(targets)
        self.profile.set_goal(radians)
        return suite

    def clip_goal(self, targets: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
        """The goal :meth:`command` would apply (suite units), without applying it."""

        return self._resolve(targets)[1]

    def goal_array(self) -> np.ndarray:
        return mjcf_to_suite_array(self.profile.goal)

    def hold(self) -> np.ndarray:
        """Goal := present position; the reference brakes onto it from its current speed."""

        return self.command(self.world.joint_array())

    def set_torque(self, enabled: bool) -> None:
        self.world.set_torque(enabled)
        if enabled:
            self.sync()

    def step(self, n: int = 1) -> None:
        world = self.world
        actuators = world.index.actuators
        dt = world.timestep_s
        remaining = int(n)
        while remaining > 0:
            if not world.torque_enabled:
                # The servo is off: track the arm so torque-on cannot jump.
                world.step(remaining)
                self.profile.reset(suite_to_mjcf_array(world.joint_array()))
                return
            if self.profile.settled():
                world.data.ctrl[actuators] = self.profile.reference
                world.step(remaining)
                return
            world.data.ctrl[actuators] = self.profile.step(dt)
            world.step(1)
            remaining -= 1

    def advance(self, duration_s: float) -> None:
        self.step(self.world.steps_for(duration_s))
