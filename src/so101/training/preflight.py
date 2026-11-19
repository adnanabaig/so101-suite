"""Validate a training request before any trainer, thread or subprocess starts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import get_settings
from ..data.dataset import (
    DatasetInfo,
    DatasetNotFoundError,
    dataset_read_hold,
    dataset_root,
    get_dataset,
)
from .configs import POLICY_PRESETS, PolicyPreset
from .jobs import TrainingJob

if TYPE_CHECKING:
    from ..data.loader import SkippedEpisode


class UnknownPolicyError(KeyError):
    """The requested policy is not a registered training preset."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else "unknown policy"


def resolve_preset(policy: str) -> PolicyPreset:
    """Accept a preset name or its ``policy_type`` (what the web UI posts)."""

    if policy in POLICY_PRESETS:
        return POLICY_PRESETS[policy]
    for preset in POLICY_PRESETS.values():
        if preset.policy_type == policy:
            return preset
    raise UnknownPolicyError(f"unknown policy {policy!r}; options: {sorted(POLICY_PRESETS)}")


def validate_training_dataset(
    repo_id: str,
    preset: PolicyPreset,
    *,
    remote_dataset: bool = False,
) -> DatasetInfo | None:
    """Return the local dataset to train on, or ``None`` for an explicit Hub dataset.

    Mock training stays synthetic but must still name a real local dataset;
    the numpy baseline always reads local data. Only real LeRobot training may
    opt in to a Hub dataset that ``lerobot-train`` downloads itself.

    Raises ``ValueError`` (bad id, empty dataset, wrong format) or
    :class:`DatasetNotFoundError`.
    """

    dataset_root(repo_id)  # validates the logical repo id
    settings = get_settings()
    hub_allowed = preset.trainer == "lerobot" and not settings.mock
    if remote_dataset and not hub_allowed:
        where = "mock mode" if settings.mock else f"the {preset.name} trainer"
        raise ValueError(f"Hub datasets need real LeRobot training; {where} trains on local data")
    info = get_dataset(repo_id)
    if info is None:
        if remote_dataset:
            return None
        hint = " (pass --hub-dataset to train on a Hugging Face Hub dataset)" if hub_allowed else ""
        raise DatasetNotFoundError(
            f"dataset {repo_id!r} not found under {settings.datasets_dir}; record it first "
            f"or check `so101 datasets ls`{hint}"
        )
    if info.num_episodes <= 0:
        raise ValueError(f"dataset {repo_id!r} has no saved episodes to train on")
    if preset.trainer == "lerobot" and not settings.mock and info.format != "lerobot":
        raise ValueError(
            f"dataset {repo_id!r} uses {info.format}; convert and validate it as a LeRobot "
            "dataset before real training"
        )
    if preset.trainer == "numpy_in_process" and info.format == "lerobot":
        from ..data.loader import lerobot_reader_available

        if not lerobot_reader_available():
            raise ValueError(
                f"dataset {repo_id!r} is a LeRobot dataset; the {preset.name} trainer reads it "
                "through the lerobot package (install the [robot] or [train] extra), or "
                "record with the SO-101 fallback writer"
            )
    return info


class InvalidTrainingArgsError(ValueError):
    """An ``extra_args`` entry is not an allowed trainer override."""


_ARG_KEY = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")
_MAX_ARG_VALUE = 512
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# lerobot-train overrides a request may set. Everything that decides where
# outputs go, what is uploaded, which data is read or how long the job runs is
# owned by so101 (and appended after these, so draccus keeps so101's value).
_LEROBOT_TOP_LEVEL = frozenset(
    {
        "seed",
        "num_workers",
        "log_freq",
        "save_freq",
        "save_checkpoint",
        "use_policy_training_preset",
        "cudnn_deterministic",
    }
)
_LEROBOT_PREFIXES = ("policy.", "optimizer.", "scheduler.")
_LEROBOT_DENIED = frozenset(
    {
        "policy.type",
        "policy.path",
        "policy.push_to_hub",
        "policy.repo_id",
        "policy.private",
        "policy.tags",
        "policy.license",
    }
)
_HUB_BASE_MODEL = re.compile(r"^lerobot/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def validate_extra_args(preset: PolicyPreset, extra_args: Mapping[str, object]) -> dict[str, str]:
    """Return ``extra_args`` if every entry is an allowed override for ``preset``.

    LeRobot presets accept ``policy.*``, ``optimizer.*`` and ``scheduler.*``
    keys plus a few training-loop knobs (seed, workers, logging and checkpoint
    cadence). Output location, dataset, step budget, Hub upload, W&B and
    resume flags are rejected, and ``policy.pretrained_path`` must name a
    ``lerobot/...`` base model or an existing local checkpoint directory. The
    numpy baseline has its own small schema.

    Raises :class:`InvalidTrainingArgsError` (a ``ValueError``).
    """

    if not isinstance(extra_args, Mapping):
        raise InvalidTrainingArgsError("extra_args must be an object of string values")
    cleaned: dict[str, str] = {}
    for key, value in extra_args.items():
        if not isinstance(key, str) or not _ARG_KEY.fullmatch(key):
            raise InvalidTrainingArgsError(
                f"invalid extra_args key {key!r}: use dotted lower-case names "
                "like policy.n_obs_steps"
            )
        if not isinstance(value, str) or not value or len(value) > _MAX_ARG_VALUE:
            raise InvalidTrainingArgsError(
                f"extra_args[{key!r}] must be a non-empty string of at most "
                f"{_MAX_ARG_VALUE} characters"
            )
        if _CONTROL_CHARS.search(value):
            raise InvalidTrainingArgsError(f"extra_args[{key!r}] contains control characters")
        cleaned[key] = value
    if preset.trainer == "numpy_in_process":
        from .bc_baseline import validate_bc_args

        validate_bc_args(cleaned)
        return cleaned
    for key, value in cleaned.items():
        allowed = key in _LEROBOT_TOP_LEVEL or key.startswith(_LEROBOT_PREFIXES)
        if not allowed or key in _LEROBOT_DENIED:
            raise InvalidTrainingArgsError(
                f"extra_args[{key!r}] is not an allowed lerobot-train override for {preset.name}; "
                "so101 owns the output dir, dataset, steps, batch size, Hub upload and W&B "
                "settings (allowed: policy.*, optimizer.*, scheduler.*, "
                f"{', '.join(sorted(_LEROBOT_TOP_LEVEL))})"
            )
        if key == "policy.pretrained_path" and not (
            _HUB_BASE_MODEL.fullmatch(value) or Path(value).expanduser().is_dir()
        ):
            raise InvalidTrainingArgsError(
                "policy.pretrained_path must be a lerobot/<model> base model or an existing "
                f"local checkpoint directory, not {value!r}"
            )
    return cleaned


# --- episode selection ------------------------------------------------------------

DEFAULT_EPISODE_SELECTION = "kept"


def _episode_selections() -> tuple[str, ...]:
    from ..data.loader import EPISODE_SELECTIONS

    return tuple(EPISODE_SELECTIONS)


def _parse_selection(value: object, where: str) -> str:
    choices = _episode_selections()
    if not isinstance(value, str) or value.strip().lower() not in choices:
        raise InvalidTrainingArgsError(
            f"{where} must be one of {', '.join(choices)}, not {value!r}"
        )
    return value.strip().lower()


def resolve_episode_selection(
    requested: str | None, extra_args: Mapping[str, object] | None = None
) -> str:
    """The episode selection of a training request: ``kept`` (default), ``success`` or ``all``.

    ``requested`` is the request's ``episodes`` (``None`` when not given). The
    bc_baseline preset has always read the selection from
    ``extra_args["episodes"]`` as well; that still works, and when both are
    given they must agree. LeRobot presets refuse that key
    (:func:`validate_extra_args`).

    Raises :class:`InvalidTrainingArgsError`.
    """

    chosen = None if requested is None else _parse_selection(requested, "episodes")
    legacy = (extra_args or {}).get("episodes")
    if legacy is None:
        return chosen or DEFAULT_EPISODE_SELECTION
    from_args = _parse_selection(legacy, "extra_args['episodes']")
    if chosen is not None and chosen != from_args:
        raise InvalidTrainingArgsError(
            f"episodes {chosen!r} conflicts with extra_args['episodes'] {from_args!r} "
            "(--arg episodes=...); give one of them, or the same value"
        )
    return from_args


def job_episode_selection(job: TrainingJob) -> str:
    """The selection ``job`` trains with: its ``episodes``, or, when that is left at
    the default, a bc_baseline job's ``extra_args["episodes"]``."""

    requested = None if job.episodes == DEFAULT_EPISODE_SELECTION else job.episodes
    return resolve_episode_selection(requested, job.extra_args)


@dataclass(frozen=True)
class EpisodePlan:
    """The episodes a training run learns from, chosen by outcome label before it starts."""

    selection: str
    # The episode indices the trainer is given; None when it is given every
    # episode (selection "all") or chooses them itself (bc_baseline, whose
    # checkpoint manifest lists them).
    episodes: tuple[int, ...] | None = None
    # Episodes left out for their outcome label; None when the trainer
    # decides (bc_baseline).
    skipped: tuple[SkippedEpisode, ...] | None = ()

    def lerobot_flag(self) -> str | None:
        """``--dataset.episodes=[0,2,5]`` for lerobot-train; None when it keeps every episode.

        Draccus 0.11 (LeRobot 0.6.1's pin) reads a ``list[int]`` field from one
        ``[a,b,c]`` token; ``0,2,5`` and ``(0,2,5)`` are rejected.
        """

        if self.episodes is None or not self.skipped:
            return None
        return "--dataset.episodes=[" + ",".join(str(index) for index in self.episodes) + "]"

    def to_manifest(self) -> dict[str, Any]:
        """The run manifest's ``episode_selection`` entry."""

        return {
            "selection": self.selection,
            "episodes": None if self.episodes is None else list(self.episodes),
            "skipped": None
            if self.skipped is None
            else [
                {"episode": item.episode, "outcome": item.outcome, "reason": item.reason}
                for item in self.skipped
            ],
        }


def plan_episodes(repo_id: str, selection: str, info: DatasetInfo | None) -> EpisodePlan:
    """Choose the episodes a run trains on, by outcome label, before anything starts.

    ``info`` is the job's local dataset, or ``None`` for a Hub dataset that
    ``lerobot-train`` downloads itself. ``all`` reads no labels. Any other
    selection reads a local dataset's labels with
    :func:`~so101.data.loader.selected_episodes`. A Hub dataset has no labels
    here (the suite keeps them in the local dataset folder), so it may only
    train with ``all``: the default ``kept`` would otherwise quietly include
    episodes someone labeled ``failure`` on the machine that recorded them.

    Raises :class:`InvalidTrainingArgsError` for an unknown selection,
    :class:`~so101.data.loader.TrainingDataError` when the labels are
    unreadable, nothing is selected or the dataset is not local, and
    :class:`DatasetNotFoundError`.
    """

    from ..data.inspection import OUTCOMES_SIDECAR
    from ..data.loader import TrainingDataError, selected_episodes

    selection = _parse_selection(selection, "episodes")
    if selection == "all":
        # Explicit ``all`` still reads no labels, but it must not select a
        # just-published episode in the recorder's save→auto-label transaction.
        if info is not None:
            with dataset_read_hold(repo_id):
                pass
        return EpisodePlan("all")
    if info is None:
        raise TrainingDataError(
            f"episode selection {selection!r} needs the outcome labels of {repo_id!r}, and a "
            f"Hugging Face Hub dataset has none here (the suite keeps them in {OUTCOMES_SIDECAR} "
            "in a local dataset's folder). Train on a local copy to learn from labeled "
            "episodes only, or choose --episodes all to train on every episode, failures "
            "included."
        )
    chosen, skipped = selected_episodes(repo_id, selection)
    return EpisodePlan(selection, tuple(chosen), tuple(skipped))
