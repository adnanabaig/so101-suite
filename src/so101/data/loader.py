"""Load recorded episodes as numpy arrays for in-process training.

The torch-free baseline trainer reads datasets through this module:

* The SO-101 fallback layout (``so101-fallback-v2``: ``meta.json`` plus one
  ``frames.jsonl`` per episode) is read directly with the core install.
* LeRobot datasets are read through the ``lerobot`` package when it is
  installed; without it the loader raises a clear :class:`TrainingDataError`.

Episode selection respects the operator's outcome labels and per-episode
quality checks:

``kept`` (default)
    Episodes labeled ``success`` plus ``unlabeled`` ones (the recorder labels
    every new episode ``unlabeled``; operators drop an episode by marking it
    ``failure``, ``partial`` or ``discarded``). Episodes that fail a quality
    check (out-of-limit commands, non-increasing timestamps) are skipped.
``success``
    Only episodes explicitly labeled ``success``, with the same quality checks.
``all``
    Every outcome, and quality failures that do not make the data unusable.

Unreadable or non-finite state, action or environment-state frames are skipped
in every mode. Missing/non-finite timestamps fail strict selections; ``all``
reconstructs them from the dataset fps, matching the fallback reader. Each
skipped episode is reported with a reason so the training manifest can record
it.

Labels come from :func:`~so101.data.inspection.inspect_dataset`: each
episode's ``episode.json`` in the fallback format, the dataset's
``so101_outcomes.json`` sidecar for LeRobot datasets. When stored label
metadata is unreadable or semantically invalid, affected episodes would read
as unlabeled, so ``kept`` and ``success`` are refused rather than silently
training on failed episodes.
:func:`selected_episodes` applies a selection by label alone, for trainers
that read the dataset themselves (``lerobot-train --dataset.episodes``).
LeRobot state and action vectors are reordered from their required named-joint
metadata into :data:`so101.JOINT_NAMES` order. Images are not loaded: the
baseline consumes joint state (and, for sim datasets, the privileged
``observation.environment_state``) only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .. import JOINT_NAMES
from ..hardware.safety import DEFAULT_LIMITS
from .dataset import (
    DatasetInfo,
    dataset_read_hold,
    require_dataset,
    validate_fallback_episode_tree,
)
from .inspection import (
    DatasetInspection,
    EpisodeManifest,
    inspect_dataset,
    outcome_labels_problem,
)

STATE_KEY = "observation.state"
ACTION_KEY = "action"
ENV_STATE_KEY = "observation.environment_state"

EpisodeSelection = Literal["kept", "success", "all"]
EnvStateMode = Literal["auto", "require", "ignore"]
EPISODE_SELECTIONS: dict[str, frozenset[str] | None] = {
    "kept": frozenset({"success", "unlabeled"}),
    "success": frozenset({"success"}),
    "all": None,
}
# meta.json keys a writer may use to name the env-state dimensions.
_ENV_NAME_KEYS = ("environment_state_names", "env_state_names")

_LOW = np.asarray([DEFAULT_LIMITS[joint][0] for joint in JOINT_NAMES], dtype=np.float64)
_HIGH = np.asarray([DEFAULT_LIMITS[joint][1] for joint in JOINT_NAMES], dtype=np.float64)


class TrainingDataError(ValueError):
    """The dataset cannot provide any usable training episodes."""


@dataclass
class EpisodeData:
    """One episode as aligned per-frame arrays (frame ``i`` pairs obs and action)."""

    episode: int
    state: np.ndarray  # (T, 6) float32, suite units
    action: np.ndarray  # (T, 6) float32, the command sent at each frame
    t: np.ndarray  # (T,) float64 seconds since the episode's first frame
    env_state: np.ndarray | None  # (T, k) float32 sim-only privileged state
    outcome: str = "unlabeled"
    outcome_source: str = ""

    @property
    def frames(self) -> int:
        return int(self.state.shape[0])


@dataclass
class SkippedEpisode:
    episode: int
    reason: str
    # The outcome label, when the episode was skipped for it (None for a
    # quality or data problem).
    outcome: str | None = None


@dataclass
class TrainingData:
    repo_id: str
    format: str
    fps: int
    selection: str
    episodes: list[EpisodeData]
    skipped: list[SkippedEpisode] = field(default_factory=list)
    env_state_names: list[str] = field(default_factory=list)

    @property
    def env_state_dim(self) -> int:
        return len(self.env_state_names)

    @property
    def total_frames(self) -> int:
        return sum(episode.frames for episode in self.episodes)

    def describe(self) -> str:
        env = f", env-state {self.env_state_dim} dims (sim-only)" if self.env_state_dim else ""
        skipped = f", skipped {len(self.skipped)}" if self.skipped else ""
        return (
            f"{len(self.episodes)} episode(s) / {self.total_frames} frames from "
            f"{self.repo_id} [{self.selection}]{env}{skipped}"
        )


class _EpisodeRejected(Exception):
    def __init__(self, reason: str, *, unusable: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.unusable = unusable


def _vector(raw: Any, key: str, dim: int | None, location: str) -> np.ndarray:
    value = raw.get(key)
    if not isinstance(value, list):
        raise _EpisodeRejected(f"{location}: {key} is missing or not a list", unusable=True)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise _EpisodeRejected(f"{location}: {key} is not numeric", unusable=True) from None
    if array.ndim != 1 or (dim is not None and array.shape[0] != dim):
        expected = f"{dim} values" if dim is not None else "a flat list"
        raise _EpisodeRejected(f"{location}: {key} must contain {expected}", unusable=True)
    if not np.isfinite(array).all():
        raise _EpisodeRejected(f"{location}: {key} contains a non-finite value", unusable=True)
    return array


def _read_fallback_frames(
    path: Path,
    *,
    fps: int,
    env_dim: int | None,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    try:
        validate_fallback_episode_tree(path.parent)
    except (OSError, ValueError) as exc:
        raise _EpisodeRejected(str(exc), unusable=True) from exc
    if not path.is_file():
        raise _EpisodeRejected("frames.jsonl is missing", unusable=True)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    times: list[float | None] = []
    envs: list[np.ndarray] = []
    saw_env: bool | None = None
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            location = f"frame {index}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _EpisodeRejected(
                    f"{location}: invalid JSON ({exc.msg})", unusable=True
                ) from None
            if not isinstance(raw, dict):
                raise _EpisodeRejected(f"{location}: not a JSON object", unusable=True)
            states.append(_vector(raw, STATE_KEY, len(JOINT_NAMES), location))
            action = _vector(raw, ACTION_KEY, len(JOINT_NAMES), location)
            if strict and ((action < _LOW).any() or (action > _HIGH).any()):
                joint = JOINT_NAMES[int(np.argmax((action < _LOW) | (action > _HIGH)))]
                raise _EpisodeRejected(
                    f"{location}: commanded {joint} outside the SO-101 limits", unusable=False
                )
            actions.append(action)
            has_env = raw.get(ENV_STATE_KEY) is not None
            if saw_env is None:
                saw_env = has_env
            elif saw_env != has_env:
                raise _EpisodeRejected(
                    f"{location}: {ENV_STATE_KEY} is present in some frames only", unusable=True
                )
            if has_env:
                envs.append(_vector(raw, ENV_STATE_KEY, env_dim, location))
                env_dim = envs[-1].shape[0]
            raw_t = raw.get("t")
            try:
                timestamp = float(raw_t) if raw_t is not None else math.nan
            except (TypeError, ValueError):
                timestamp = math.nan
            times.append(timestamp if math.isfinite(timestamp) else None)
    if not states:
        raise _EpisodeRejected("episode has no frames", unusable=True)
    if any(value is None for value in times):
        if strict:
            raise _EpisodeRejected("a frame timestamp is missing or non-finite", unusable=False)
        t = np.arange(len(states), dtype=np.float64) / max(fps, 1)
    else:
        t = np.asarray(times, dtype=np.float64)
        if strict and len(t) > 1 and (np.diff(t) <= 0).any():
            frame = int(np.argmax(np.diff(t) <= 0)) + 1
            raise _EpisodeRejected(f"frame {frame}: timestamp did not increase", unusable=False)
        t = t - t[0]
    env = np.asarray(envs, dtype=np.float32) if envs else None
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        t,
        env,
    )


def _env_state_names(meta: dict[str, Any], dim: int) -> list[str]:
    for key in _ENV_NAME_KEYS:
        names = meta.get(key)
        if isinstance(names, list) and len(names) == dim:
            return [str(name) for name in names]
    features = meta.get("features")
    if isinstance(features, dict):
        spec = features.get(ENV_STATE_KEY)
        if isinstance(spec, dict):
            names = spec.get("names")
            if isinstance(names, list) and len(names) == dim:
                return [str(name) for name in names]
    return [f"env_{index}" for index in range(dim)]


def _select(
    manifests: Sequence[EpisodeManifest], selection: str
) -> tuple[list[EpisodeManifest], list[SkippedEpisode]]:
    if selection not in EPISODE_SELECTIONS:
        raise ValueError(f"episode selection must be one of {sorted(EPISODE_SELECTIONS)}")
    allowed = EPISODE_SELECTIONS[selection]
    chosen: list[EpisodeManifest] = []
    skipped: list[SkippedEpisode] = []
    for manifest in manifests:
        if allowed is not None and manifest.outcome not in allowed:
            skipped.append(
                SkippedEpisode(manifest.episode, f"outcome is {manifest.outcome}", manifest.outcome)
            )
        else:
            chosen.append(manifest)
    return chosen, skipped


def _refuse_untrusted_labels(inspection: DatasetInspection, selection: str) -> None:
    problem = outcome_labels_problem(inspection)
    if problem is not None and selection != "all":
        raise TrainingDataError(
            f"cannot select {selection!r} episodes of {inspection.summary.repo_id!r}: its "
            f"outcome labels are unreadable ({problem}). Repair or restore the labels file, "
            "or choose --episodes all."
        )


def selected_episodes(
    repo_id: str, selection: str = "kept"
) -> tuple[list[int], list[SkippedEpisode]]:
    """The episode indices ``selection`` keeps by outcome label, and the ones it drops.

    No frame is read, so the per-episode quality checks of
    :func:`load_training_data` do not apply. Meant for trainers that read the
    dataset themselves, e.g. ``lerobot-train --dataset.episodes=[...]``.

    Raises :class:`~so101.data.dataset.DatasetNotFoundError`, ``ValueError``
    for an unknown selection, or :class:`TrainingDataError` when the labels
    are unreadable or nothing is selected.
    """

    with dataset_read_hold(repo_id):
        require_dataset(repo_id)
        inspection = inspect_dataset(repo_id)
        _refuse_untrusted_labels(inspection, selection)
        chosen, skipped = _select(inspection.episodes, selection)
        if not chosen:
            raise TrainingDataError(
                f"no episodes in {repo_id!r} for selection {selection!r} "
                f"({_outcome_summary(inspection)}). Label episodes as success in the Datasets "
                "page, or choose --episodes all."
            )
        return [manifest.episode for manifest in chosen], skipped


def _outcome_summary(inspection: DatasetInspection) -> str:
    counts: dict[str, int] = {}
    for item in inspection.episodes:
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
    summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
    return f"dataset has {summary or 'no episodes'}"


def _apply_env_mode(
    episodes: list[EpisodeData], skipped: list[SkippedEpisode], env_state: str
) -> list[EpisodeData]:
    if env_state not in ("auto", "require", "ignore"):
        raise ValueError("env_state must be 'auto', 'require' or 'ignore'")
    use_env = env_state == "require" or (
        env_state == "auto" and any(episode.env_state is not None for episode in episodes)
    )
    if not use_env:
        for episode in episodes:
            episode.env_state = None
        return episodes
    dims = [episode.env_state.shape[1] for episode in episodes if episode.env_state is not None]
    expected = max(set(dims), key=dims.count) if dims else None
    kept: list[EpisodeData] = []
    for episode in episodes:
        if episode.env_state is None:
            skipped.append(SkippedEpisode(episode.episode, f"no {ENV_STATE_KEY}"))
        elif episode.env_state.shape[1] != expected:
            skipped.append(
                SkippedEpisode(
                    episode.episode,
                    f"{ENV_STATE_KEY} has {episode.env_state.shape[1]} dims, expected {expected}",
                )
            )
        else:
            kept.append(episode)
    return kept


def _load_fallback(
    info: DatasetInfo, manifests: list[EpisodeManifest], selection: str
) -> tuple[list[EpisodeData], list[SkippedEpisode]]:
    root = Path(info.path)
    episodes: list[EpisodeData] = []
    skipped: list[SkippedEpisode] = []
    strict = selection != "all"
    for manifest in manifests:
        try:
            state, action, t, env = _read_fallback_frames(
                root / manifest.name / "frames.jsonl", fps=info.fps, env_dim=None, strict=strict
            )
        except _EpisodeRejected as exc:
            kind = "unusable" if exc.unusable else "failed quality check"
            skipped.append(SkippedEpisode(manifest.episode, f"{kind}: {exc.reason}"))
            continue
        except OSError as exc:
            skipped.append(SkippedEpisode(manifest.episode, f"unreadable: {exc}"))
            continue
        episodes.append(
            EpisodeData(
                episode=manifest.episode,
                state=state,
                action=action,
                t=t,
                env_state=env,
                outcome=manifest.outcome,
                outcome_source=manifest.outcome_source,
            )
        )
    return episodes, skipped


def _lerobot_dataset_class() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-not-found]
            LeRobotDataset,
        )
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[import-not-found]
                LeRobotDataset,
            )
        except ImportError as exc:
            raise TrainingDataError(
                "this is a LeRobot dataset; reading it for the numpy baseline needs the "
                "lerobot package (install the [robot] or [train] extra), or record with "
                "the SO-101 fallback writer"
            ) from exc
    return LeRobotDataset


def lerobot_reader_available() -> bool:
    try:
        _lerobot_dataset_class()
    except TrainingDataError:
        return False
    return True


def _column(dataset: Any, key: str) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float64) for value in dataset[key]])


def _lerobot_joint_order(dataset: Any, key: str) -> list[int]:
    """Column positions that put a named LeRobot joint vector in suite order."""

    features = getattr(dataset, "features", None)
    feature = features.get(key) if isinstance(features, Mapping) else None
    raw_names = feature.get("names") if isinstance(feature, Mapping) else None
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise TrainingDataError(f"LeRobot dataset {key} feature has no named joints")
    names = [str(name) for name in raw_names]
    canonical = [f"{joint}.pos" for joint in JOINT_NAMES]
    if len(names) != len(canonical) or set(names) != set(canonical):
        raise TrainingDataError(
            f"LeRobot dataset {key} named joints do not match SO-101: {names} != {canonical}"
        )
    return [names.index(name) for name in canonical]


def _load_lerobot(
    info: DatasetInfo, manifests: list[EpisodeManifest], selection: str
) -> tuple[list[EpisodeData], list[SkippedEpisode]]:
    from .lerobot_local import open_local_lerobot_dataset

    dataset_cls = _lerobot_dataset_class()
    try:
        # Local data only: never let LeRobot download into the dataset root.
        dataset = open_local_lerobot_dataset(info.repo_id, Path(info.path), dataset_cls=dataset_cls)
    except (FileNotFoundError, ValueError) as exc:
        raise TrainingDataError(str(exc)) from exc
    table = dataset.hf_dataset
    columns = set(table.column_names)
    missing = [key for key in (STATE_KEY, ACTION_KEY, "episode_index") if key not in columns]
    if missing:
        raise TrainingDataError(f"LeRobot dataset {info.repo_id!r} has no {missing} columns")
    state_order = _lerobot_joint_order(dataset, STATE_KEY)
    action_order = _lerobot_joint_order(dataset, ACTION_KEY)
    episode_index = np.asarray(table["episode_index"], dtype=np.int64).reshape(-1)
    state = _column(table, STATE_KEY)
    action = _column(table, ACTION_KEY)
    if state.shape[1:] != (len(JOINT_NAMES),) or action.shape[1:] != (len(JOINT_NAMES),):
        raise TrainingDataError("LeRobot dataset state/action are not 6 SO-101 joints")
    state = state[:, state_order]
    action = action[:, action_order]
    timestamps = (
        np.asarray(table["timestamp"], dtype=object).reshape(-1) if "timestamp" in columns else None
    )
    env = _column(table, ENV_STATE_KEY) if ENV_STATE_KEY in columns else None
    by_episode = {manifest.episode: manifest for manifest in manifests}
    episodes: list[EpisodeData] = []
    skipped: list[SkippedEpisode] = []
    strict = selection != "all"
    for episode in sorted(by_episode):
        rows = np.flatnonzero(episode_index == episode)
        manifest = by_episode[episode]
        if rows.size == 0:
            skipped.append(SkippedEpisode(episode, "unusable: no frames"))
            continue
        ep_state, ep_action = state[rows], action[rows]
        values = [ep_state, ep_action] + ([env[rows]] if env is not None else [])
        if not all(np.isfinite(value).all() for value in values):
            skipped.append(SkippedEpisode(episode, "unusable: non-finite values"))
            continue
        if strict and ((ep_action < _LOW).any() or (ep_action > _HIGH).any()):
            skipped.append(
                SkippedEpisode(episode, "failed quality check: command outside SO-101 limits")
            )
            continue
        fallback_t = np.arange(rows.size, dtype=np.float64) / max(info.fps, 1)
        try:
            episode_timestamps = (
                None
                if timestamps is None
                else np.asarray(timestamps[rows], dtype=np.float64).reshape(-1)
            )
        except (TypeError, ValueError, OverflowError):
            episode_timestamps = None
        if episode_timestamps is None or not np.isfinite(episode_timestamps).all():
            if strict:
                skipped.append(
                    SkippedEpisode(
                        episode,
                        "failed quality check: a frame timestamp is missing or non-finite",
                    )
                )
                continue
            t = fallback_t
        else:
            deltas = np.diff(episode_timestamps)
            if strict and (deltas <= 0).any():
                frame = int(np.argmax(deltas <= 0)) + 1
                skipped.append(
                    SkippedEpisode(
                        episode,
                        f"failed quality check: frame {frame} timestamp did not increase",
                    )
                )
                continue
            t = episode_timestamps - episode_timestamps[0]
        episodes.append(
            EpisodeData(
                episode=episode,
                state=ep_state.astype(np.float32),
                action=ep_action.astype(np.float32),
                t=t,
                env_state=env[rows].astype(np.float32) if env is not None else None,
                outcome=manifest.outcome,
                outcome_source=manifest.outcome_source,
            )
        )
    return episodes, skipped


def load_training_data(
    repo_id: str,
    *,
    selection: str = "kept",
    env_state: str = "auto",
) -> TrainingData:
    """Read the selected episodes of a local dataset as aligned numpy arrays.

    ``env_state``: ``auto`` uses ``observation.environment_state`` when the
    selected episodes carry it (the result is then privileged sim data),
    ``require`` skips episodes without it, ``ignore`` drops it (state-only).

    Raises :class:`~so101.data.dataset.DatasetNotFoundError`, or
    :class:`TrainingDataError` when labels or LeRobot joint metadata cannot be
    trusted, or when nothing usable remains.
    """

    info = require_dataset(repo_id)
    inspection = inspect_dataset(repo_id)
    _refuse_untrusted_labels(inspection, selection)
    chosen, skipped = _select(inspection.episodes, selection)
    if info.format.startswith("so101-fallback-"):
        episodes, rejected = _load_fallback(info, chosen, selection)
    elif info.format == "lerobot":
        episodes, rejected = _load_lerobot(info, chosen, selection)
    else:
        raise TrainingDataError(f"unsupported dataset format {info.format!r}")
    skipped.extend(rejected)
    episodes = _apply_env_mode(episodes, skipped, env_state)
    skipped.sort(key=lambda item: item.episode)
    if not episodes:
        reasons = "; ".join(f"episode {s.episode}: {s.reason}" for s in skipped[:3])
        raise TrainingDataError(
            f"no usable episodes in {repo_id!r} for selection {selection!r} "
            f"({_outcome_summary(inspection)}; {reasons or 'nothing selected'}). "
            "Label episodes as success in the Datasets page, or choose --episodes all."
        )
    names: list[str] = []
    first_env = episodes[0].env_state
    if first_env is not None:
        meta: dict[str, Any] = {}
        root = Path(info.path)
        meta_path = root / "meta.json" if info.format != "lerobot" else root / "meta" / "info.json"
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                meta = {}
        names = _env_state_names(meta, int(first_env.shape[1]))
    return TrainingData(
        repo_id=repo_id,
        format=info.format,
        fps=info.fps,
        selection=selection,
        episodes=episodes,
        skipped=skipped,
        env_state_names=names,
    )
