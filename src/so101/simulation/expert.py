"""Scripted expert for the physics tasks: DLS inverse kinematics plus a phase machine.

:class:`DlsIk` solves for the five arm joints that put ``gripperframe`` at a
position (and optionally an orientation) by damped least squares on the site
Jacobian (``mj_jacSite``), on a private ``MjData``.

:class:`ScriptedExpert` implements the policy protocol (``reset()`` and
``select_action(observation)``). It reads only the observation:
``observation.state`` (suite units) and ``observation.environment_state``
(the task layout), and returns absolute suite-unit targets at the control
rate. The caller sends them through a ``SafetyGuard`` like any other policy.

Phases: ``approach`` (joint-space move to a hover pose 3 cm above the cube,
jaws open and aligned to its faces) -> ``pre_grasp`` (re-centre on the cube
as it lies now) -> ``descend`` -> ``close`` -> ``lift``; ``pick_place``
continues with ``transport`` -> ``place`` -> ``release`` -> ``retreat``.
``reach`` is ``approach`` -> ``hold``. Every phase ends when the commanded
segment is done and the arm has converged (or a timeout).

Workspace: the wrist_flex limit (95 deg) only allows a straight-down gripper
up to 4-9 cm above the table (9 cm at 20-23 cm from the base). The expert asks
for straight down with a strong weight and, where that cannot be met, keeps
the position and lets the orientation give (:meth:`ScriptedExpert._solve`).

Gripper geometry (vendored model): the jaws close along the site's +z axis,
the fixed jaw's inner face sits 20 mm behind the site, and the jaw tips are
about 3 mm below it. Gripper 45 opens the tips to 69 mm, 25 is 40 mm.

Needs the ``[sim]`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .. import JOINT_NAMES
from .tasks import CONTROL_HZ, CUBE_SIZE_M, Task, yaw_from_quat
from .units import effective_limit_arrays, mjcf_to_suite_array, suite_to_mjcf_array
from .world import compiled_scene

ARM = slice(0, 5)  # the five arm joints; the gripper is index 5
GRIPPER = 5
DOWN = np.array([0.0, 0.0, -1.0])

GRIPPER_OPEN = 45.0  # 69 mm between the tips
GRIPPER_CLOSED = 0.0  # squeeze: the cube stops the jaw near 24
FIXED_JAW_OFFSET_M = 0.020  # fixed-jaw inner face to gripperframe, along site +z
JAW_CLEARANCE_M = 0.006  # fixed jaw clears the cube face by this much on the way down
TIP_BELOW_SITE_M = 0.003
GRASP_TIP_HEIGHT_M = 0.009  # jaw tips this far above the table at the grasp
HOVER_ABOVE_M = 0.03  # site above the cube top before the descent
LIFT_BY_M = 0.065  # the task needs the cube 5 cm up; this leaves 1.5 cm of slip
PLACE_GAP_M = 0.003  # cube bottom above the table when it is let go
RETREAT_BY_M = 0.04
IK_TOLERANCE_M = 2e-3  # beyond this the orientation is relaxed
STRAIGHT_DOWN_WEIGHT = 0.3  # metres per radian: orientation nearly as firm as position
RELAXED_WEIGHT = 0.02  # position first, orientation as close as it can be

JOINT_SPEED_DEG_S = 75.0  # average speed of joint-space moves
SPEED_APPROACH_M_S = 0.15
SPEED_DESCEND_M_S = 0.06
SPEED_LIFT_M_S = 0.08
SPEED_TRANSPORT_M_S = 0.12
CONVERGED_DEG = 1.0
# Fine phases near contact get a quarter of the configured joint noise, as a
# teleoperator slows down and steadies the hand there.
FINE_PHASES = frozenset({"descend", "close", "place", "release"})
FINE_NOISE_SCALE = 0.25
SETTLE_TIMEOUT_S = 1.0


def wrap_quarter_turn(angle: float) -> float:
    """``angle`` modulo 90 deg, in [-45 deg, 45 deg) (a cube looks the same)."""

    quarter = math.pi / 2.0
    return (angle + quarter / 2.0) % quarter - quarter / 2.0


def grasp_rotation(closing_heading: float) -> np.ndarray:
    """Site frame pointing down with the jaws closing along ``closing_heading``.

    Columns are the site axes in world coordinates: x (approach) is -z, z (the
    closing direction, fixed jaw -> moving jaw) is horizontal.
    """

    z_axis = np.array([math.cos(closing_heading), math.sin(closing_heading), 0.0])
    y_axis = np.cross(z_axis, DOWN)
    return np.column_stack([DOWN, y_axis, z_axis])


def smoothstep(s: float) -> float:
    """Minimum-jerk time scaling on [0, 1]."""

    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


# -- inverse kinematics ---------------------------------------------------------------
@dataclass
class IkResult:
    joints: np.ndarray  # suite units, all six joints (gripper copied from the seed)
    position_error_m: float
    rotation_error_rad: float
    iterations: int


class DlsIk:
    """Damped-least-squares IK for ``gripperframe`` over the five arm joints."""

    def __init__(
        self,
        model: Any,
        *,
        damping: float = 0.03,
        max_iterations: int = 100,
        position_tolerance_m: float = 2e-4,
        rotation_tolerance_rad: float = 2e-3,
        max_step_rad: float = 0.25,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        index = compiled_index(model)
        self._qpos = index["qpos"][ARM]
        self._dof = index["dof"][ARM]
        self._site = index["site"]
        low, high = effective_limit_arrays()
        self._low = suite_to_mjcf_array(low)[ARM]
        self._high = suite_to_mjcf_array(high)[ARM]
        self.damping = damping
        self.max_iterations = max_iterations
        self.position_tolerance_m = position_tolerance_m
        self.rotation_tolerance_rad = rotation_tolerance_rad
        self.max_step_rad = max_step_rad
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site position and rotation for suite-unit ``joints``."""

        self.data.qpos[self._qpos] = suite_to_mjcf_array(joints)[ARM]
        mujoco.mj_kinematics(self.model, self.data)
        site = self._site
        return self.data.site_xpos[site].copy(), self.data.site_xmat[site].reshape(3, 3).copy()

    def solve(
        self,
        seed: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        *,
        rotation_weight: float = 0.3,
        max_iterations: int | None = None,
    ) -> IkResult:
        """Joints (suite units) reaching ``position`` [and ``rotation``], from ``seed``.

        ``rotation_weight`` (metres per radian) trades orientation against
        position error when both cannot be met.
        """

        model, data = self.model, self.data
        seed = np.asarray(seed, dtype=float)
        q = suite_to_mjcf_array(seed)[ARM].copy()
        target = np.asarray(position, dtype=float)
        weight = rotation_weight if rotation is not None else 0.0
        rows = 6 if weight > 0.0 else 3
        position_error = rotation_error = math.inf
        iterations = 0
        for iterations in range(1, (max_iterations or self.max_iterations) + 1):  # noqa: B007
            data.qpos[self._qpos] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            current = data.site_xpos[self._site]
            error = np.zeros(rows)
            error[:3] = target - current
            position_error = float(np.linalg.norm(error[:3]))
            rotation_error = 0.0
            if rows == 6 and rotation is not None:
                frame = data.site_xmat[self._site].reshape(3, 3)
                orientation = 0.5 * np.cross(frame.T, rotation.T).sum(axis=0)
                error[3:] = weight * orientation
                rotation_error = float(np.linalg.norm(orientation))
            if (
                position_error <= self.position_tolerance_m
                and rotation_error <= self.rotation_tolerance_rad
            ):
                break
            mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self._site)
            jacobian = self._jacp[:, self._dof]
            if rows == 6:
                jacobian = np.vstack([jacobian, weight * self._jacr[:, self._dof]])
            regularized = jacobian @ jacobian.T + (self.damping**2) * np.eye(rows)
            step = jacobian.T @ np.linalg.solve(regularized, error)
            largest = float(np.max(np.abs(step)))
            if largest > self.max_step_rad:
                step *= self.max_step_rad / largest
            q = np.clip(q + step, self._low, self._high)
        joints = seed.copy()
        full = suite_to_mjcf_array(seed)
        full[ARM] = q
        joints[ARM] = mjcf_to_suite_array(full)[ARM]
        return IkResult(joints, position_error, rotation_error, iterations)


def compiled_index(model: Any) -> dict[str, Any]:
    """qpos/dof addresses and the tool site for ``model`` (any compiled scene)."""

    joint_ids = [model.joint(name).id for name in JOINT_NAMES]
    return {
        "qpos": np.array([model.jnt_qposadr[j] for j in joint_ids]),
        "dof": np.array([model.jnt_dofadr[j] for j in joint_ids]),
        "site": model.site("gripperframe").id,
    }


# -- expert ----------------------------------------------------------------------------------
@dataclass
class ExpertConfig:
    """Demo diversity. Zero noise is the deterministic expert."""

    noise_deg: float = 0.0  # per-step joint-target noise (AR(1), about 0.3 s correlation)
    # Per-episode offset of the grasp and place points: across the jaws freely,
    # along the closing axis only where the fixed jaw still clears the cube.
    waypoint_noise_m: float = 0.0
    noise_correlation: float = 0.9
    seed: int | None = None


@dataclass
class _Move:
    """One commanded segment: joint-space, Cartesian or gripper-only."""

    name: str
    kind: str  # "joint" | "cartesian" | "gripper" | "wait"
    duration_s: float
    start: np.ndarray  # joints (suite)
    goal: np.ndarray | None = None  # joints for "joint"/"gripper"
    start_position: np.ndarray | None = None
    goal_position: np.ndarray | None = None
    rotation: np.ndarray | None = None
    settle: bool = True  # wait for the arm to converge at the end
    elapsed_s: float = 0.0
    waited_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class ScriptedExpert:
    """Privileged scripted policy for :mod:`so101.simulation.tasks`."""

    def __init__(self, task: Task, config: ExpertConfig | None = None) -> None:
        self.task = task
        self.config = config or ExpertConfig()
        self.ik = DlsIk(copy_model(task))
        self.control_dt = 1.0 / CONTROL_HZ
        self._episode = 0
        self.reset()

    # -- policy protocol --------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Start a new episode. ``seed`` (or config seed + episode count) drives the noise."""

        base = self.config.seed if seed is None else seed
        noise_seed = None if base is None else [int(base), self._episode if seed is None else 0]
        self._episode += 1
        self.rng = np.random.default_rng(noise_seed)
        self.noise: np.ndarray = np.zeros(5)
        self.local_offset = np.zeros(2)  # (along closing, across the jaws)
        if self.config.waypoint_noise_m > 0.0:
            spread = self.config.waypoint_noise_m
            along, across = self.rng.normal(0.0, spread, 2)
            self.local_offset = np.array(
                [
                    np.clip(along, -2 * spread, JAW_CLEARANCE_M / 2.0),
                    np.clip(across, -2 * spread, 2 * spread),
                ]
            )
        self.sent: np.ndarray | None = None
        self.phase = "start"
        self.command: np.ndarray | None = None
        self.move: _Move | None = None
        self.closing_heading = 0.0
        self.grasp_z = 0.0
        self.history: list[str] = []

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=float).reshape(6)
        env = np.asarray(observation["observation.environment_state"], dtype=float)
        if self.command is None:
            self.command = state.copy()
            self._next(state, env)
        assert self.move is not None
        target = self._advance(state, env)
        self.command = target
        action = target.copy()
        if self.config.noise_deg > 0.0:
            rho = self.config.noise_correlation
            scale = FINE_NOISE_SCALE if self.phase in FINE_PHASES else 1.0
            kick = self.rng.normal(0.0, self.config.noise_deg, 5)
            self.noise = rho * self.noise + math.sqrt(1.0 - rho * rho) * kick
            action[ARM] += scale * self.noise
        self.sent = action
        return action.astype(np.float32)

    # -- planning ---------------------------------------------------------------------------
    def _begin(self, move: _Move) -> None:
        self.move = move
        self.phase = move.name
        self.history.append(move.name)

    def _cube(self, env: np.ndarray) -> tuple[np.ndarray, float]:
        return env[:3], yaw_from_quat(env[3:7])

    def offset(self) -> np.ndarray:
        """This episode's waypoint offset in the world frame (metres, xy)."""

        along, across = self.local_offset
        c, s = math.cos(self.closing_heading), math.sin(self.closing_heading)
        return np.array([along * c - across * s, along * s + across * c])

    def _grasp_point(self, env: np.ndarray) -> np.ndarray:
        """Site target at the grasp: the fixed jaw just clear of the cube's near face."""

        cube, _ = self._cube(env)
        closing = np.array([math.cos(self.closing_heading), math.sin(self.closing_heading)])
        # The fixed jaw's inner face is FIXED_JAW_OFFSET_M behind the site (-closing).
        back = CUBE_SIZE_M / 2.0 + JAW_CLEARANCE_M - FIXED_JAW_OFFSET_M
        xy = cube[:2] - back * closing + self.offset()
        return np.array([xy[0], xy[1], self.grasp_z])

    def _solve(
        self, seed: np.ndarray, position: np.ndarray, rotation: np.ndarray | None
    ) -> np.ndarray:
        """Joints for ``position``: straight down if possible, else position first."""

        result = self.ik.solve(seed, position, rotation, rotation_weight=STRAIGHT_DOWN_WEIGHT)
        if rotation is not None and result.position_error_m > IK_TOLERANCE_M:
            result = self.ik.solve(
                result.joints, position, rotation, rotation_weight=RELAXED_WEIGHT
            )
            if result.position_error_m > IK_TOLERANCE_M:
                result = self.ik.solve(result.joints, position)
        return result.joints

    def _joint_move(self, name: str, start: np.ndarray, goal: np.ndarray, **kwargs: Any) -> _Move:
        span = float(np.max(np.abs(goal[ARM] - start[ARM])))
        duration = max(0.4, 1.5 * span / JOINT_SPEED_DEG_S)
        return _Move(name, "joint", duration, start.copy(), goal=goal.copy(), **kwargs)

    def _line(
        self, name: str, start: np.ndarray, goal_position: np.ndarray, speed: float, **kwargs: Any
    ) -> _Move:
        position, _ = self.ik.forward(start)
        rotation = kwargs.pop("rotation", None)
        distance = float(np.linalg.norm(goal_position - position))
        duration = max(0.3, 1.5 * distance / speed)
        return _Move(
            name,
            "cartesian",
            duration,
            start.copy(),
            start_position=position,
            goal_position=np.asarray(goal_position, dtype=float),
            rotation=rotation,
            **kwargs,
        )

    def _gripper(self, name: str, start: np.ndarray, value: float, duration: float) -> _Move:
        goal = start.copy()
        goal[GRIPPER] = value
        return _Move(name, "gripper", duration, start.copy(), goal=goal)

    def _hold(self, command: np.ndarray) -> _Move:
        return _Move("hold", "wait", math.inf, command.copy(), settle=False)

    def _next(self, state: np.ndarray, env: np.ndarray) -> None:
        """Plan the move after the current one (or the first)."""

        assert self.command is not None
        command = self.command
        name, phase = self.task.name, self.phase
        if name == "reach":
            if phase == "start":
                target = env[:3]
                down = grasp_rotation(math.atan2(target[1], target[0]))
                goal = self._solve(command, target, down)
                goal[GRIPPER] = command[GRIPPER]
                self._begin(self._joint_move("approach", command, goal))
            else:
                self._begin(self._hold(command))
            return

        cube, yaw = self._cube(env)
        down = grasp_rotation(self.closing_heading)
        if phase == "start":
            heading = math.atan2(cube[1], cube[0])
            # Close across the pair of faces most nearly facing the base.
            self.closing_heading = heading + wrap_quarter_turn(yaw - heading)
            bottom = cube[2] - CUBE_SIZE_M / 2.0
            self.grasp_z = max(bottom + GRASP_TIP_HEIGHT_M + TIP_BELOW_SITE_M, 0.008)
            hover = self._grasp_point(env)
            hover[2] = cube[2] + CUBE_SIZE_M / 2.0 + HOVER_ABOVE_M
            goal = self._solve(command, hover, grasp_rotation(self.closing_heading))
            goal[GRIPPER] = GRIPPER_OPEN
            self._begin(self._joint_move("approach", command, goal))
        elif phase == "approach":
            hover = self._grasp_point(env)
            hover[2] = cube[2] + CUBE_SIZE_M / 2.0 + HOVER_ABOVE_M
            move = self._line("pre_grasp", command, hover, SPEED_APPROACH_M_S, rotation=down)
            move.extra["track"] = True
            self._begin(move)
        elif phase == "pre_grasp":
            grasp = self._grasp_point(env)
            move = self._line("descend", command, grasp, SPEED_DESCEND_M_S, rotation=down)
            move.extra["track"] = True
            self._begin(move)
        elif phase == "descend":
            move = self._gripper("close", command, GRIPPER_CLOSED, 0.35)
            move.extra["hold_s"] = 0.45
            self._begin(move)
        elif phase == "close":
            position, _ = self.ik.forward(command)
            goal = position + np.array([0.0, 0.0, LIFT_BY_M])
            self._begin(self._line("lift", command, goal, SPEED_LIFT_M_S, rotation=down))
        elif name == "pick_lift":
            self._begin(self._hold(command))
        elif phase == "lift":
            zone = env[7:10]
            position, _ = self.ik.forward(command)
            # Keep the wrist roll: the cube turns with the pan, which no place test minds.
            turn = math.atan2(zone[1], zone[0]) - math.atan2(position[1], position[0])
            self.closing_heading += turn
            offset = self.offset()
            goal = np.array([zone[0] + offset[0], zone[1] + offset[1], position[2]])
            carry = grasp_rotation(self.closing_heading)
            self._begin(self._line("transport", command, goal, SPEED_TRANSPORT_M_S, rotation=carry))
        elif phase == "transport":
            position, _ = self.ik.forward(command)
            drop = max((cube[2] - CUBE_SIZE_M / 2.0) - PLACE_GAP_M, 0.0)
            goal = position - np.array([0.0, 0.0, drop])
            self._begin(self._line("place", command, goal, SPEED_DESCEND_M_S, rotation=down))
        elif phase == "place":
            move = self._gripper("release", command, GRIPPER_OPEN, 0.3)
            move.extra["hold_s"] = 0.3
            self._begin(move)
        elif phase == "release":
            position, _ = self.ik.forward(command)
            goal = position + np.array([0.0, 0.0, RETREAT_BY_M])
            self._begin(self._line("retreat", command, goal, SPEED_LIFT_M_S, rotation=down))
        else:
            self._begin(self._hold(command))

    # -- execution ----------------------------------------------------------------------------
    def _advance(self, state: np.ndarray, env: np.ndarray) -> np.ndarray:
        move = self.move
        assert move is not None and self.command is not None
        dt = self.control_dt
        if move.elapsed_s < move.duration_s:
            move.elapsed_s = min(move.elapsed_s + dt, move.duration_s)
            s = smoothstep(move.elapsed_s / move.duration_s)
            if move.kind in ("joint", "gripper"):
                assert move.goal is not None
                return move.start + s * (move.goal - move.start)
            if move.kind == "cartesian":
                assert move.start_position is not None and move.goal_position is not None
                if move.extra.get("track"):
                    # Follow the cube while it is still on the table.
                    goal = self._grasp_point(env)
                    goal[2] = move.goal_position[2]
                    move.goal_position = goal
                position = move.start_position + s * (move.goal_position - move.start_position)
                joints = self._solve(self.command, position, move.rotation)
                joints[GRIPPER] = move.start[GRIPPER]
                return joints
            return self.command
        # The commanded segment is done: hold it until the arm converges.
        move.waited_s += dt
        hold = float(move.extra.get("hold_s", 0.0))
        reference = self.command if self.sent is None else self.sent
        error = float(np.max(np.abs(state[ARM] - reference[ARM])))
        converged = (not move.settle or error <= CONVERGED_DEG) and move.waited_s >= hold
        if converged or move.waited_s >= max(SETTLE_TIMEOUT_S, hold):
            if math.isfinite(move.duration_s):
                self._next(state, env)
        return self.command


def copy_model(task: Task) -> Any:
    """A private model for kinematics (never the cached template)."""

    import copy

    return copy.copy(compiled_scene(task.scene).model)


__all__ = [
    "DlsIk",
    "ExpertConfig",
    "IkResult",
    "ScriptedExpert",
    "grasp_rotation",
    "wrap_quarter_turn",
]
