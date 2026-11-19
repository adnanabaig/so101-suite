"""Toy reach: a numpy-only synthetic task for testing and measuring the BC baseline.

This is not a physics simulation (MuJoCo lives in :mod:`so101.simulation`).
It is a small kinematic stand-in that mirrors the simulator's reach task, so
the record -> train -> evaluate loop can be exercised with the core install
in well under a second:

* The pan, lift and elbow joints place the gripper through a two-link arm
  (13 cm upper arm, 15 cm forearm, shoulder 12 cm above the table) rotated
  by the pan joint. The wrist, roll and gripper joints do not move the
  gripper point.
* Every episode starts at the simulator's ready pose plus up to 3 degrees of
  jitter per arm joint. A target is drawn from the simulator's reach region
  (16-26 cm from the base, +-40 degrees of azimuth, 4-16 cm above the table).
* The joint state follows the command with a first-order lag: each frame it
  moves 35% of the way to the command, like the mock arm.
* The scripted expert commands at most 6 degrees ahead of the state toward
  the inverse-kinematics solution for the target (plus small command noise),
  and returns the wrist, roll and gripper joints to the ready pose.
* Success, as in the simulator: the gripper point stays within 2 cm of the
  target for 0.3 s.

Episodes carry the target as ``observation.environment_state``
(``target_x``, ``target_y``, ``target_z``), so a baseline trained on them is
a privileged-state baseline, exactly like one trained on simulator data.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .. import JOINT_NAMES
from ..data.loader import EpisodeData, TrainingData
from ..hardware.safety import DEFAULT_LIMITS

TASK = "move the gripper to the red ball"
ENV_STATE_NAMES = ("target_x", "target_y", "target_z")
FPS = 30
EPISODE_S = 3.0
READY_POSE = np.asarray([0.0, -60.0, 30.0, 85.0, 0.0, 30.0])
START_JITTER_DEG = 3.0
REACH_RADIUS_M = (0.16, 0.26)
REACH_AZIMUTH_DEG = (-40.0, 40.0)
REACH_HEIGHT_M = (0.04, 0.16)
REACH_TOLERANCE_M = 0.02
REACH_DWELL_S = 0.3
TRACKING = 0.35
MAX_LEAD_DEG = 6.0
COMMAND_NOISE_DEG = 0.3
SHOULDER_HEIGHT_M = 0.12
UPPER_ARM_M = 0.13
FOREARM_M = 0.15
OUTCOME_SOURCE = "auto:toy_reach"

_LOW = np.asarray([DEFAULT_LIMITS[joint][0] for joint in JOINT_NAMES])
_HIGH = np.asarray([DEFAULT_LIMITS[joint][1] for joint in JOINT_NAMES])
# The ready pose's lift and elbow joints put the gripper point here (r, z).
_HOME_M = (0.12, 0.22)


def _planar_ik(radius: float, height: float) -> tuple[float, float]:
    """Upper-arm angle from vertical and elbow bend (degrees) reaching (radius, height)."""

    dz = height - SHOULDER_HEIGHT_M
    distance = math.hypot(radius, dz)
    reach = UPPER_ARM_M + FOREARM_M
    distance = min(max(distance, abs(UPPER_ARM_M - FOREARM_M) + 1e-6), reach - 1e-6)
    cos_elbow = (distance**2 - UPPER_ARM_M**2 - FOREARM_M**2) / (2 * UPPER_ARM_M * FOREARM_M)
    elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))
    direction = math.atan2(radius, dz)
    inner = math.atan2(FOREARM_M * math.sin(elbow), UPPER_ARM_M + FOREARM_M * math.cos(elbow))
    return math.degrees(direction - inner), math.degrees(elbow)


_HOME_UPPER, _HOME_ELBOW = _planar_ik(*_HOME_M)
LIFT_OFFSET_DEG = _HOME_UPPER - READY_POSE[1]
ELBOW_OFFSET_DEG = _HOME_ELBOW - READY_POSE[2]


def gripper_position(state: np.ndarray) -> np.ndarray:
    """Forward kinematics: the gripper point (x, y, z in metres) for joint states (..., 6)."""

    state = np.asarray(state, dtype=np.float64)
    pan = np.radians(state[..., 0])
    upper = np.radians(state[..., 1] + LIFT_OFFSET_DEG)
    elbow = np.radians(state[..., 2] + ELBOW_OFFSET_DEG)
    radius = UPPER_ARM_M * np.sin(upper) + FOREARM_M * np.sin(upper + elbow)
    height = SHOULDER_HEIGHT_M + UPPER_ARM_M * np.cos(upper) + FOREARM_M * np.cos(upper + elbow)
    return np.stack([radius * np.cos(pan), radius * np.sin(pan), height], axis=-1)


def goal_joints(target: Sequence[float] | np.ndarray) -> np.ndarray:
    """The joint configuration that puts the gripper point on ``target`` (x, y, z)."""

    x, y, z = (float(value) for value in target)
    upper, elbow = _planar_ik(math.hypot(x, y), z)
    goal = READY_POSE.copy()
    goal[0] = math.degrees(math.atan2(y, x))
    goal[1] = upper - LIFT_OFFSET_DEG
    goal[2] = elbow - ELBOW_OFFSET_DEG
    return np.clip(goal, _LOW, _HIGH)


def sample_target(rng: np.random.Generator) -> np.ndarray:
    radius = rng.uniform(*REACH_RADIUS_M)
    azimuth = math.radians(rng.uniform(*REACH_AZIMUTH_DEG))
    height = rng.uniform(*REACH_HEIGHT_M)
    return np.asarray([radius * math.cos(azimuth), radius * math.sin(azimuth), height])


def sample_start(rng: np.random.Generator) -> np.ndarray:
    start = READY_POSE.copy()
    start[:5] += rng.uniform(-START_JITTER_DEG, START_JITTER_DEG, 5)
    return start


def track(state: np.ndarray, command: np.ndarray) -> np.ndarray:
    """One frame of the toy arm: the state moves 35% of the way to the command."""

    return state + TRACKING * (np.clip(command, _LOW, _HIGH) - state)


def expert_action(
    state: np.ndarray, goal: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    command = state + np.clip(goal - state, -MAX_LEAD_DEG, MAX_LEAD_DEG)
    if rng is not None:
        command = command + rng.normal(0.0, COMMAND_NOISE_DEG, command.shape)
    return np.clip(command, _LOW, _HIGH)


class ReachSuccess:
    """Tracks the dwell: within 2 cm of the target for 0.3 s (as in the simulator)."""

    def __init__(self, target: np.ndarray, fps: int = FPS) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        self.needed = max(1, round(REACH_DWELL_S * fps))
        self.dwell = 0
        self.succeeded = False
        self.min_distance = math.inf

    def update(self, state: np.ndarray) -> bool:
        distance = float(np.linalg.norm(gripper_position(state) - self.target))
        self.min_distance = min(self.min_distance, distance)
        self.dwell = self.dwell + 1 if distance <= REACH_TOLERANCE_M else 0
        if self.dwell >= self.needed:
            self.succeeded = True
        return self.succeeded


@dataclass
class ToyEpisode:
    state: np.ndarray  # (T, 6)
    action: np.ndarray  # (T, 6)
    target: np.ndarray  # (3,)
    success: bool
    distracted: bool
    min_distance_m: float
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))


def generate_episode(
    rng: np.random.Generator,
    *,
    episode_s: float = EPISODE_S,
    distracted: bool = False,
    noise: bool = True,
) -> ToyEpisode:
    """One expert demonstration. A distracted expert reaches for the wrong spot."""

    target = sample_target(rng)
    aim = sample_target(rng) if distracted else target
    goal = goal_joints(aim)
    state = sample_start(rng)
    frames = max(2, round(episode_s * FPS))
    states = np.empty((frames, 6))
    actions = np.empty((frames, 6))
    success = ReachSuccess(target)
    for i in range(frames):
        action = expert_action(state, goal, rng if noise else None)
        states[i] = state
        actions[i] = action
        state = track(state, action)
        success.update(state)
    return ToyEpisode(
        state=states,
        action=actions,
        target=target,
        success=success.succeeded,
        distracted=distracted,
        min_distance_m=success.min_distance,
        t=np.arange(frames) / FPS,
    )


def generate_toy_reach_episodes(
    count: int, *, seed: int = 0, failure_fraction: float = 0.0, episode_s: float = EPISODE_S
) -> list[ToyEpisode]:
    """``count`` demonstrations; about ``failure_fraction`` of them are distracted."""

    rng = np.random.default_rng(seed)
    return [
        generate_episode(rng, episode_s=episode_s, distracted=bool(rng.random() < failure_fraction))
        for _ in range(count)
    ]


def _outcome(episode: ToyEpisode) -> str:
    return "success" if episode.success else "failure"


def toy_training_data(
    episodes: Sequence[ToyEpisode],
    *,
    repo_id: str = "toy/reach",
    selection: str = "kept",
    privileged: bool = True,
) -> TrainingData:
    """In-memory :class:`TrainingData` (what the loader would return for the dataset).

    ``selection`` follows the loader: ``kept`` and ``success`` keep the
    successful demonstrations, ``all`` keeps every one.
    """

    chosen = [
        (index, episode)
        for index, episode in enumerate(episodes)
        if selection == "all" or episode.success
    ]
    return TrainingData(
        repo_id=repo_id,
        format="so101-fallback-v2",
        fps=FPS,
        selection=selection,
        episodes=[
            EpisodeData(
                episode=index,
                state=episode.state.astype(np.float32),
                action=episode.action.astype(np.float32),
                t=episode.t.astype(np.float64),
                env_state=(
                    np.tile(episode.target.astype(np.float32), (episode.state.shape[0], 1))
                    if privileged
                    else None
                ),
                outcome=_outcome(episode),
                outcome_source=OUTCOME_SOURCE,
            )
            for index, episode in chosen
        ],
        env_state_names=list(ENV_STATE_NAMES) if privileged else [],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_toy_reach_dataset(
    repo_id: str,
    episodes: Sequence[ToyEpisode],
    *,
    privileged: bool = True,
    label: bool = True,
) -> Path:
    """Write ``episodes`` as an SO-101 fallback dataset under the datasets directory.

    Episodes are labeled ``success``/``failure`` with source
    ``auto:toy_reach`` (``label=False`` leaves them ``unlabeled``). With
    ``privileged`` the frames carry ``observation.environment_state``.
    Raises ``FileExistsError`` if the dataset exists.
    """

    from ..data.dataset import (
        DatasetBusyError,
        dataset_busy_message,
        dataset_operation_target,
        refuse_case_alias,
    )

    root, operation_lock = dataset_operation_target(repo_id)
    if not operation_lock.acquire(blocking=False):
        raise DatasetBusyError(dataset_busy_message(repo_id))
    try:
        # Every case variant shares this lock. Recheck admission under it so a
        # concurrent or journal-hidden lowercase dataset cannot race an
        # uppercase publication into a second root.
        refuse_case_alias(repo_id)
        return _write_toy_reach_dataset_locked(
            root,
            repo_id,
            episodes,
            privileged=privileged,
            label=label,
        )
    finally:
        operation_lock.release()


def _write_toy_reach_dataset_locked(
    root: Path,
    repo_id: str,
    episodes: Sequence[ToyEpisode],
    *,
    privileged: bool,
    label: bool,
) -> Path:
    """Publish a toy dataset while the caller owns its operation lock."""

    if root.exists():
        raise FileExistsError(f"dataset {repo_id!r} already exists at {root}")
    root.mkdir(parents=True)
    started = time.time()
    total = 0
    for index, episode in enumerate(episodes):
        directory = root / f"episode_{index:06d}"
        directory.mkdir()
        lines = []
        for i in range(episode.state.shape[0]):
            frame: dict[str, Any] = {
                "t": round(started + float(episode.t[i]), 6),
                "observation.state": [round(float(v), 5) for v in episode.state[i]],
                "action": [round(float(v), 5) for v in episode.action[i]],
                "desired_action": [round(float(v), 5) for v in episode.action[i]],
            }
            if privileged:
                frame["observation.environment_state"] = [
                    round(float(v), 6) for v in episode.target
                ]
            lines.append(json.dumps(frame))
        (directory / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        record: dict[str, Any] = {
            "schema_version": 1,
            "episode": index,
            "frames": int(episode.state.shape[0]),
            "task": TASK,
            "outcome": _outcome(episode) if label else "unlabeled",
            "notes": "toy reach (distracted expert)" if episode.distracted else "toy reach",
            "camera_keys": [],
        }
        if label:
            record["outcome_source"] = OUTCOME_SOURCE
            record["outcome_detail"] = {"min_distance_m": round(episode.min_distance_m, 4)}
        _write_json(directory / "episode.json", record)
        total += int(episode.state.shape[0])
    meta: dict[str, Any] = {
        "schema_version": 1,
        "repo_id": repo_id,
        "fps": FPS,
        "task": TASK,
        "camera_keys": [],
        "episodes": len(episodes),
        "total_frames": total,
        "format": "so101-fallback-v2",
        "image_storage": "chunked-npy",
        "camera_storage": {},
        "generator": "so101.training.toy_reach",
    }
    if privileged:
        meta["environment_state_names"] = list(ENV_STATE_NAMES)
    tmp = root / f".meta.json.{os.getpid()}"
    _write_json(tmp, meta)
    os.replace(tmp, root / "meta.json")
    return root


class _Policy(Protocol):
    def reset(self) -> None: ...

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class ToyEvalResult:
    successes: int
    trials: int
    ci95: tuple[float, float]
    mean_min_distance_m: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


def evaluate_toy_reach(
    policy: _Policy,
    trials: int = 50,
    *,
    seed: int = 1000,
    episode_s: float = EPISODE_S,
    provide_env_state: bool = True,
) -> ToyEvalResult:
    """Closed-loop success of ``policy`` on fresh targets (Wilson 95% interval).

    The policy is called once per frame with ``observation.state`` and, when
    ``provide_env_state``, ``observation.environment_state``; its action is
    applied through the toy arm's lag. Seeds differ from the training data's
    defaults, so targets are new.
    """

    from ..eval.stats import wilson_interval

    rng = np.random.default_rng(seed)
    frames = max(2, round(episode_s * FPS))
    successes = 0
    distances = []
    for _ in range(trials):
        target = sample_target(rng)
        state = sample_start(rng)
        tracker = ReachSuccess(target)
        policy.reset()
        for _frame in range(frames):
            observation = {"observation.state": state.astype(np.float32)}
            if provide_env_state:
                observation["observation.environment_state"] = target.astype(np.float32)
            action = np.asarray(policy.select_action(observation), dtype=np.float64)
            state = track(state, action)
            tracker.update(state)
        successes += int(tracker.succeeded)
        distances.append(tracker.min_distance)
    return ToyEvalResult(
        successes=successes,
        trials=trials,
        ci95=wilson_interval(successes, trials),
        mean_min_distance_m=float(np.mean(distances)) if distances else math.nan,
    )


class ExpertPolicy:
    """The scripted expert as a privileged policy (noise-free): an upper bound."""

    def reset(self) -> None:
        return None

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float64)
        goal = goal_joints(observation["observation.environment_state"])
        return expert_action(state, goal).astype(np.float32)


__all__ = [
    "ENV_STATE_NAMES",
    "FPS",
    "TASK",
    "ExpertPolicy",
    "ReachSuccess",
    "ToyEpisode",
    "ToyEvalResult",
    "evaluate_toy_reach",
    "generate_episode",
    "generate_toy_reach_episodes",
    "goal_joints",
    "gripper_position",
    "toy_training_data",
    "write_toy_reach_dataset",
]
