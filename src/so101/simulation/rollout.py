"""Closed-loop episodes: observation -> policy -> SafetyGuard -> servo -> physics.

One :class:`EpisodeRunner` owns a world built from a task's scene and runs
episodes as fast as the CPU allows (no wall-clock pacing), so a run is a pure
function of (task, policy, seed, settings). Every command goes through a real
:class:`~so101.hardware.safety.SafetyGuard` with an explicit ``dt``, and the
arm is driven by the STS3215 servo profile unless ``servo_mode="step"``.

Observation keys (the policy interface):

- ``observation.state``: float32 (6,), suite units, ``JOINT_NAMES`` order;
- ``observation.environment_state``: float32 (k,), the task layout
  (:attr:`so101.simulation.tasks.Task.env_state_names`);
- ``observation.images.<camera>``: HxWx3 uint8, only for requested cameras.

The action is float32 (6,): absolute suite-unit joint targets.

Needs the ``[sim]`` extra.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .. import JOINT_NAMES
from ..hardware.safety import SafetyGuard
from .servo import ServoDriver, ServoMode
from .tasks import CONTROL_HZ, Task
from .world import PhysicsWorld

STATE_KEY = "observation.state"
ENV_STATE_KEY = "observation.environment_state"
IMAGE_KEY_PREFIX = "observation.images."

# ``(world, camera, width, height) -> HxWx3 uint8``; see ``render.frame_source``.
FrameSource = Callable[[PhysicsWorld, str, int, int], np.ndarray]


@runtime_checkable
class Policy(Protocol):
    """What :class:`EpisodeRunner` drives: LeRobot-style ``reset``/``select_action``."""

    def reset(self) -> None: ...

    def select_action(self, observation: Mapping[str, Any]) -> Any:
        """Six absolute suite-unit targets (array, ``JOINT_NAMES`` order, or a dict)."""


@dataclass
class StepRecord:
    index: int
    time_s: float
    observation: dict[str, Any]
    action: np.ndarray  # what the policy asked for
    command: np.ndarray  # what the SafetyGuard let through (sent to the servos)
    metrics: dict[str, float]


@dataclass
class EpisodeResult:
    task: str
    seed: int | None
    success: bool
    steps: int
    sim_time_s: float
    wall_time_s: float
    first_success_s: float | None
    instruction: str
    layout: dict[str, Any]
    summary: dict[str, Any]
    final_metrics: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def action_array(action: Any) -> np.ndarray:
    """A policy's action as a (6,) float array.

    Takes the pinned form, an array of six suite-unit targets in
    ``JOINT_NAMES`` order, and also the ``{joint: value}`` dict that
    :class:`so101.policy.server.Policy` implementations return.
    """

    if isinstance(action, Mapping):
        missing = [joint for joint in JOINT_NAMES if joint not in action]
        if missing:
            raise ValueError(f"policy action is missing joints {missing}")
        action = [action[joint] for joint in JOINT_NAMES]
    array = np.asarray(action, dtype=float).reshape(-1)
    if array.shape != (len(JOINT_NAMES),):
        raise ValueError(f"policy returned shape {array.shape}, expected ({len(JOINT_NAMES)},)")
    if not np.all(np.isfinite(array)):
        raise ValueError("policy returned a non-finite action")
    return array


class EpisodeRunner:
    """Runs seeded episodes of one task. Not thread-safe; one runner per thread."""

    def __init__(
        self,
        task: Task,
        *,
        servo_mode: ServoMode = "sts3215",
        control_hz: float = CONTROL_HZ,
        cameras: Sequence[str] = (),
        image_size: tuple[int, int] = (320, 240),
        frames: FrameSource | None = None,
    ) -> None:
        if cameras and frames is None:
            raise ValueError("cameras were requested but no frame source was given")
        self.task = task
        self.world = task.make_world()
        self.driver = ServoDriver(self.world, mode=servo_mode)
        self.control_hz = float(control_hz)
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.frames = frames
        self.guard = SafetyGuard()
        self.substeps = self.world.steps_for(1.0 / self.control_hz)
        if self.substeps < 1:
            raise ValueError("control rate is faster than the physics timestep")

    def observe(self) -> dict[str, Any]:
        observation: dict[str, Any] = {
            STATE_KEY: self.world.joint_array().astype(np.float32),
            ENV_STATE_KEY: self.task.env_state(self.world),
        }
        if self.frames is not None:
            width, height = self.image_size
            for camera in self.cameras:
                observation[IMAGE_KEY_PREFIX + camera] = self.frames(
                    self.world, camera, width, height
                )
        return observation

    def reset(self, seed: int | None) -> dict[str, Any]:
        info = self.task.reset(self.world, seed)
        self.driver.sync()
        self.guard.reset(dict(zip(JOINT_NAMES, self.world.joint_array().tolist(), strict=True)))
        return info

    def run(
        self,
        policy: Policy,
        seed: int | None,
        *,
        on_step: Callable[[StepRecord], None] | None = None,
        stop_on_success: bool = True,
        tail_s: float = 0.0,
        max_steps: int | None = None,
        policy_reset: Callable[[], None] | None = None,
    ) -> EpisodeResult:
        """One episode. Stops ``tail_s`` after success, or at the task's time limit."""

        started = time.perf_counter()
        info = self.reset(seed)
        (policy_reset or policy.reset)()
        limit = self.task.episode_steps(self.control_hz) if max_steps is None else max_steps
        dt = 1.0 / self.control_hz
        tail_steps = round(tail_s * self.control_hz)
        succeeded_at: int | None = None
        metrics: dict[str, float] = {}
        steps = 0
        for index in range(limit):
            observation = self.observe()
            action = action_array(policy.select_action(observation))
            safe = self.guard.clamp(dict(zip(JOINT_NAMES, action.tolist(), strict=True)), dt=dt)
            applied = self.driver.command(safe)
            self.guard.record_actual_command(dict(zip(JOINT_NAMES, applied.tolist(), strict=True)))
            self.driver.step(self.substeps)
            metrics = self.task.update(self.world, dt)
            steps = index + 1
            if on_step is not None:
                on_step(
                    StepRecord(
                        index=index,
                        time_s=round(index * dt, 6),
                        observation=observation,
                        action=action.astype(np.float32),
                        command=applied.astype(np.float32),
                        metrics=metrics,
                    )
                )
            if succeeded_at is None and self.task.success(self.world):
                succeeded_at = index
            if stop_on_success and succeeded_at is not None and index - succeeded_at >= tail_steps:
                break
        return EpisodeResult(
            task=self.task.name,
            seed=seed,
            success=succeeded_at is not None,
            steps=steps,
            sim_time_s=round(steps * dt, 6),
            wall_time_s=round(time.perf_counter() - started, 6),
            first_success_s=self.task.first_success_s,
            instruction=self.task.instruction,
            layout=info["layout"],
            summary=self.task.summary(),
            final_metrics={key: round(float(value), 6) for key, value in metrics.items()},
            info={"start_pose": info["start_pose"]},
        )


__all__ = [
    "ENV_STATE_KEY",
    "IMAGE_KEY_PREFIX",
    "STATE_KEY",
    "EpisodeResult",
    "EpisodeRunner",
    "FrameSource",
    "Policy",
    "StepRecord",
    "action_array",
]
