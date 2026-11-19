"""Replay a recorded episode's actions back onto the follower arm.

Replay runs at the rate the episode was recorded at unless ``fps`` overrides
it (audit B18), on a deadline schedule so slow I/O does not stretch the
episode. LeRobot datasets are read locally only, never via a Hub download
(audit B27).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .. import JOINT_NAMES
from ..bringup.standalone import refuse_real_motion, standalone_access
from ..hardware.follower import SO101Follower
from ..logging import get_logger
from .dataset import require_dataset, validate_fallback_episode_tree

log = get_logger(__name__)

MAX_REPLAY_FPS = 240


def _recorded_fps(value: Any) -> float:
    """Validate a dataset-derived replay rate before any arm is opened."""

    if isinstance(value, bool):
        raise ValueError(f"recorded fps must be between 1 and {MAX_REPLAY_FPS}")
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"recorded fps must be between 1 and {MAX_REPLAY_FPS}") from None
    if not math.isfinite(rate) or not 1 <= rate <= MAX_REPLAY_FPS:
        raise ValueError(f"recorded fps must be between 1 and {MAX_REPLAY_FPS}")
    return rate


def _load_fallback_actions(root: Path, episode: int) -> list[list[float]]:
    directory = root / f"episode_{episode:06d}"
    fj = directory / "frames.jsonl"
    if episode < 0 or (not directory.exists() and not directory.is_symlink()):
        saved = []
        for candidate in root.glob("episode_*"):
            index = candidate.name.removeprefix("episode_")
            if not index.isdigit():
                continue
            validate_fallback_episode_tree(candidate)
            if (candidate / "frames.jsonl").is_file():
                saved.append(int(index))
        saved.sort()
        available = f"episodes {saved[0]}..{saved[-1]}" if saved else "no saved episodes"
        raise FileNotFoundError(f"episode {episode} not found in {root.name} ({available})")
    validate_fallback_episode_tree(directory)
    if not fj.is_file():
        raise FileNotFoundError(f"episode {episode} has no frames.jsonl in {root.name}")
    return [json.loads(line)["action"] for line in fj.read_text().splitlines() if line.strip()]


def _load_lerobot_actions(
    root: Path,
    repo_id: str,
    episode: int,
) -> tuple[list[list[float]], int | float]:
    """Load one local LeRobot episode in canonical SO-101 joint order."""

    from .lerobot_local import open_local_lerobot_dataset

    dataset = open_local_lerobot_dataset(repo_id, root, episodes=[episode])
    feature = dataset.features.get("action")
    if not feature or not feature.get("names"):
        raise ValueError("LeRobot dataset action feature has no named joints")
    names = [str(name) for name in feature["names"]]
    canonical = [f"{joint}.pos" for joint in JOINT_NAMES]
    if set(names) != set(canonical):
        raise ValueError(f"dataset action joints do not match SO-101: {names} != {canonical}")
    actions_column = dataset.select_columns("action")
    actions: list[list[float]] = []
    for index in range(dataset.num_frames):
        raw = actions_column[index]["action"]
        by_name = {name: float(raw[position]) for position, name in enumerate(names)}
        actions.append([by_name[name] for name in canonical])
    # Keep the raw value until replay_episode has either selected an explicit
    # override or validated the dataset-derived rate. Converting here would
    # make NaN/Infinity fail with an incidental conversion error even when a
    # valid explicit override was supplied.
    return actions, dataset.fps


def replay_episode(
    repo_id: str,
    episode: int,
    follower: SO101Follower | None = None,
    fps: int | None = None,
) -> int:
    """Replay ``episode`` of ``repo_id`` onto the follower. Returns frame count.

    ``fps`` defaults to the dataset's recorded rate. Supports both fallback
    and local LeRobot v3 layouts. Every target still passes through the
    follower's calibration-derived safety guard.

    Without ``follower`` this call opens the arm itself, so it first passes
    the standalone gate (:mod:`so101.bringup.standalone`): it raises
    ``StandaloneRefused`` before connecting while a safety latch is set, the
    latch store is unreadable, or (real backend) a server owns the bus. A
    ``follower`` passed in belongs to the caller and is used as it is on mock
    or simulation.

    Owned real replay is refused; use RobotManager through the Suite API.
    Passing a caller-owned follower does not bypass that rule. Mock/simulation
    replay may still own or receive a follower directly.
    """
    override_fps = _recorded_fps(fps) if fps is not None else None
    if follower is not None:
        refuse_real_motion("replay_episode with a caller-owned follower")
    ds = require_dataset(repo_id)

    source_fps: int | float
    if ds.format.startswith("so101-fallback-"):
        actions = _load_fallback_actions(Path(ds.path), episode)
        source_fps = ds.fps
    elif ds.format == "lerobot":
        actions, source_fps = _load_lerobot_actions(Path(ds.path), ds.repo_id, episode)
    else:
        raise ValueError(f"unsupported dataset format: {ds.format}")
    playback_fps = override_fps if override_fps is not None else _recorded_fps(source_fps)
    own = follower is None
    access = standalone_access("replay_episode", motion=True) if own else None
    completed = False
    opened = False
    try:
        follower = follower or SO101Follower()
        if own:
            assert access is not None
            access.connect_follower(follower)
            opened = True
        period = 1.0 / playback_fps
        deadline = time.monotonic()
        for vec in actions:
            goal = {j: vec[i] for i, j in enumerate(JOINT_NAMES)}
            follower.send_goal(goal)
            # Deadline pacing: the send time is part of the period, so the
            # episode plays at its recorded speed. After a stall, resume from
            # now rather than bursting to catch up.
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
        completed = True
    finally:
        try:
            if own and opened and follower is not None:
                assert access is not None
                if completed:
                    access.halt_follower(follower)
                else:
                    access.disconnect_follower(follower)
        finally:
            if access is not None:
                access.release()
    log.info("replayed %d frames of %s ep %d", len(actions), repo_id, episode)
    return len(actions)
