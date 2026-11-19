"""Policies for the physics simulator and closed-loop evaluation.

A policy is anything with ``reset()`` and ``select_action(observation)``
(:class:`so101.simulation.rollout.Policy`). The observation keys are pinned:

- ``observation.state``: float32 (6,), suite units, ``JOINT_NAMES`` order;
- ``observation.environment_state``: float32 (k,), the task layout
  (:attr:`so101.simulation.tasks.Task.env_state_names`);
- ``observation.images.<camera>``: HxWx3 uint8, only for requested cameras.

The action is six absolute suite-unit targets: a float32 (6,) array in
``JOINT_NAMES`` order (a ``{joint: value}`` dict is accepted too).

:func:`make_policy` resolves a policy spec: ``expert`` (the scripted expert),
``hold`` (hold the start pose) or a model path. Model paths go through a small
registry so other packages can add formats without editing the CLI::

    from so101.simulation.policy import register_policy_loader
    register_policy_loader(is_baseline_checkpoint, load_baseline_policy)

``predicate(path) -> bool`` says whether a loader recognises ``path``;
``loader(path) -> Policy`` builds the policy. Later registrations are tried
first. A path nobody recognises raises :class:`UnsupportedPolicyError`.

Needs the ``[sim]`` extra.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .expert import ExpertConfig, ScriptedExpert
from .rollout import STATE_KEY, EpisodeResult, EpisodeRunner, FrameSource, Policy
from .servo import ServoMode
from .tasks import Task, make_task

BUILTIN_POLICIES: tuple[str, ...] = ("expert", "hold")
Z_95 = 1.959963984540054

PathPredicate = Callable[[Path], bool]
PolicyLoader = Callable[[Path], Policy]
_LOADERS: list[tuple[PathPredicate, PolicyLoader]] = []


class UnsupportedPolicyError(ValueError):
    """The policy spec is neither a built-in nor a path a registered loader takes."""


def register_policy_loader(predicate: PathPredicate, loader: PolicyLoader) -> None:
    """Teach :func:`make_policy` a model format. Later registrations win."""

    _LOADERS.append((predicate, loader))


def registered_policy_loaders() -> tuple[tuple[PathPredicate, PolicyLoader], ...]:
    return tuple(_LOADERS)


class HoldPolicy:
    """Holds the first observed joint state for the whole episode (a null baseline)."""

    def __init__(self) -> None:
        self._hold: np.ndarray | None = None

    def reset(self) -> None:
        self._hold = None

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._hold is None:
            self._hold = np.asarray(observation[STATE_KEY], dtype=np.float32).reshape(6).copy()
        return self._hold.copy()


def make_policy(spec: str, task: Task, *, expert: ExpertConfig | None = None) -> Policy:
    """``expert``, ``hold`` or a model path a registered loader recognises."""

    if spec == "expert":
        return ScriptedExpert(task, expert)
    if spec == "hold":
        return HoldPolicy()
    path = Path(spec).expanduser()
    if not path.exists():
        raise UnsupportedPolicyError(
            f"unsupported model path {spec!r}: it does not exist "
            f"(built-in policies: {', '.join(BUILTIN_POLICIES)})"
        )
    for predicate, loader in reversed(_LOADERS):
        if predicate(path):
            return loader(path)
    raise UnsupportedPolicyError(
        f"unsupported model path {spec!r}: no registered policy loader recognises it "
        f"({len(_LOADERS)} registered; built-in policies: {', '.join(BUILTIN_POLICIES)}). "
        "Register one with so101.simulation.policy.register_policy_loader(predicate, loader)."
    )


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""

    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def episode_seeds(seed: int, episodes: int) -> list[int]:
    """Episode ``i`` of a run uses seed ``seed + i``: any episode can be rerun alone."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    return [seed + index for index in range(episodes)]


@dataclass
class EvalReport:
    task: str
    policy: str
    episodes: int
    successes: int
    success_rate: float
    wilson95: tuple[float, float]
    seeds: list[int]
    outcomes: list[dict[str, Any]]
    mean_first_success_s: float | None
    sim_time_s: float
    wall_time_s: float
    physics_steps: int
    servo_mode: str
    instruction: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def real_time_factor(self) -> float:
        return self.sim_time_s / self.wall_time_s if self.wall_time_s > 0 else math.inf

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["wilson95"] = [round(v, 4) for v in self.wilson95]
        data["real_time_factor"] = round(self.real_time_factor, 1)
        data["physics_steps_per_s"] = (
            round(self.physics_steps / self.wall_time_s) if self.wall_time_s > 0 else None
        )
        return data


def evaluate(
    task: str | Task,
    policy: str | Policy,
    *,
    episodes: int,
    seed: int = 0,
    servo_mode: ServoMode = "sts3215",
    expert: ExpertConfig | None = None,
    on_episode: Callable[[EpisodeResult], None] | None = None,
    seeds: Sequence[int] | None = None,
    cameras: Sequence[str] = (),
    image_size: tuple[int, int] = (320, 240),
    frames: FrameSource | None = None,
) -> EvalReport:
    """Run ``episodes`` seeded episodes (seed + i) and summarise the outcomes.

    ``cameras`` adds ``observation.images.<camera>`` (``image_size`` = W, H) to
    every observation, rendered by ``frames`` (default: the shared renderer).
    """

    task_obj = make_task(task) if isinstance(task, str) else task
    spec = policy if isinstance(policy, str) else type(policy).__name__
    policy_obj = make_policy(policy, task_obj, expert=expert) if isinstance(policy, str) else policy
    if cameras and frames is None:
        from .render import frame_source

        frames = frame_source()
    runner = EpisodeRunner(
        task_obj, servo_mode=servo_mode, cameras=cameras, image_size=image_size, frames=frames
    )
    run_seeds = list(seeds) if seeds is not None else episode_seeds(seed, episodes)
    outcomes: list[dict[str, Any]] = []
    started = time.perf_counter()
    sim_time = 0.0
    steps = 0
    for episode_seed in run_seeds:
        reset = _seeded_reset(policy_obj, episode_seed)
        result = runner.run(policy_obj, episode_seed, policy_reset=reset)
        sim_time += result.sim_time_s
        steps += result.steps
        outcomes.append(
            {
                "seed": episode_seed,
                "success": result.success,
                "first_success_s": result.first_success_s,
                "steps": result.steps,
                "layout": result.layout,
                "summary": result.summary,
            }
        )
        if on_episode is not None:
            on_episode(result)
    successes = sum(1 for outcome in outcomes if outcome["success"])
    times = [o["first_success_s"] for o in outcomes if o["first_success_s"] is not None]
    return EvalReport(
        task=task_obj.name,
        policy=spec,
        episodes=len(run_seeds),
        successes=successes,
        success_rate=successes / len(run_seeds),
        wilson95=wilson_interval(successes, len(run_seeds)),
        seeds=run_seeds,
        outcomes=outcomes,
        mean_first_success_s=round(float(np.mean(times)), 4) if times else None,
        sim_time_s=round(sim_time, 4),
        wall_time_s=round(time.perf_counter() - started, 4),
        physics_steps=steps * runner.substeps,
        servo_mode=servo_mode,
        instruction=task_obj.instruction,
        extra={"cameras": list(cameras), "image_size": list(image_size)} if cameras else {},
    )


def _seeded_reset(policy: Policy, seed: int) -> Callable[[], None]:
    """The scripted expert takes the episode seed for its noise; others reset plainly."""

    if isinstance(policy, ScriptedExpert):
        return lambda: policy.reset(seed)
    return policy.reset


__all__ = [
    "BUILTIN_POLICIES",
    "EvalReport",
    "HoldPolicy",
    "Policy",
    "UnsupportedPolicyError",
    "episode_seeds",
    "evaluate",
    "make_policy",
    "register_policy_loader",
    "registered_policy_loaders",
    "wilson_interval",
]
