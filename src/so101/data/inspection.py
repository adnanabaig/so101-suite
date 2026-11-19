"""Read-only dataset inspection, quality control, and replay previews.

The inspection path is deliberately separate from :mod:`so101.data.replay`:
nothing in this module opens a motor bus or sends a command.  Fallback
datasets can be inspected with the core install.  LeRobot datasets expose
their manifest information without LeRobot and report a capability state for
features that need the optional runtime.

Outcome labels live in each episode's ``episode.json`` in the fallback
format. LeRobot's metadata has no outcome field, so for LeRobot datasets the
suite keeps them in a sidecar, ``<dataset>/so101_outcomes.json``
(:data:`OUTCOMES_SIDECAR`), with the same per-episode fields and rules.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .. import JOINT_NAMES
from ..atomic_io import atomic_write_json
from ..hardware.safety import DEFAULT_LIMITS
from .dataset import (
    DatasetBusyError,
    DatasetInfo,
    dataset_busy_message,
    dataset_operation_lock,
    get_dataset,
    validate_fallback_episode_tree,
)

JOINT_FEATURE_NAMES = [f"{joint}.pos" for joint in JOINT_NAMES]
_OUTCOMES = {"unlabeled", "success", "failure", "partial", "discarded"}
# Outcome provenance. Labels written without a source (older datasets, the
# Datasets page) are treated as human labels, which automation never overrides.
HUMAN_OUTCOME_SOURCE = "human"
SIM_OUTCOME_SOURCE = "auto:sim_predicate"
_AUTO_OUTCOME_SOURCE = re.compile(r"^auto:[a-z0-9][a-z0-9_.:-]{0,63}$")
_OUTCOME_HISTORY_LIMIT = 20
_OUTCOME_DETAIL_MAX_BYTES = 4096
# The suite's outcome labels for a LeRobot dataset (LeRobot has no such field).
OUTCOMES_SIDECAR = "so101_outcomes.json"
OUTCOMES_SIDECAR_SCHEMA = 1
OUTCOME_LABELS_CAPABILITY = "outcome_labels"


class OutcomeLabelsUnreadableError(RuntimeError):
    """A dataset's outcome metadata cannot be trusted, so it is left unchanged."""


class LabelNotStoredError(RuntimeError):
    """A saved episode's predicate label could not be stored.

    The episode deliberately remains published: deleting a successfully saved
    recording would lose data. The message tells the operator that the episode
    currently reads as ``unlabeled`` and how to repair it before training.
    """

    def __init__(
        self,
        repo_id: str,
        episode: int,
        outcome: str | None,
        cause: BaseException,
        *,
        grading: bool = False,
    ) -> None:
        self.repo_id, self.episode, self.outcome, self.cause = repo_id, episode, outcome, cause
        if grading:
            problem = f"its automatic outcome could not be graded ({cause})"
        else:
            problem = f"its {outcome} label could not be stored ({cause})"
        super().__init__(
            f"episode {episode} of {repo_id} was saved, but {problem}. It reads as unlabeled, "
            "which the default 'kept' selection trains on: label it in the Datasets page "
            f"(or POST /api/datasets/{repo_id}/episodes/{episode}/outcome) before training"
        )

    @classmethod
    def grading_failed(
        cls, repo_id: str, episode: int, cause: BaseException
    ) -> LabelNotStoredError:
        return cls(repo_id, episode, None, cause, grading=True)


def persist_sim_outcome(
    repo_id: str,
    episode: int,
    success: bool,
    detail: dict[str, Any],
    *,
    operation_lock_held: bool = False,
) -> tuple[str, str]:
    """Persist a sim predicate verdict, preserving a human label when present."""

    outcome = "success" if success else "failure"
    try:
        if operation_lock_held:
            written = _set_episode_outcome_locked(
                repo_id,
                episode,
                outcome,
                source=SIM_OUTCOME_SOURCE,
                detail=detail,
            )
        else:
            written = set_episode_outcome(
                repo_id,
                episode,
                outcome,
                source=SIM_OUTCOME_SOURCE,
                detail=detail,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise LabelNotStoredError(repo_id, episode, outcome, exc) from exc
    return outcome, "auto-labeled" if written else "kept the human label"


@dataclass
class InspectionCapability:
    id: str
    available: bool
    level: Literal["full", "partial", "unavailable"]
    detail: str


@dataclass
class EpisodeManifest:
    episode: int
    name: str
    frames: int | None
    duration_seconds: float | None
    started_at: float | None = None
    ended_at: float | None = None
    outcome: str = "unlabeled"
    notes: str = ""
    complete: bool = True
    camera_chunks: dict[str, int] = field(default_factory=dict)
    camera_frames: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    # "human", "auto:<predicate>", or "" while the episode is unlabeled.
    outcome_source: str = ""


@dataclass
class IncompleteCapture:
    id: str
    frames: int
    duration_seconds: float | None
    reason: str
    captured_at: float | None
    camera_chunks: dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetInspection:
    summary: DatasetInfo
    duration_seconds: float
    episodes: list[EpisodeManifest]
    incomplete_captures: list[IncompleteCapture]
    outcome_counts: dict[str, int]
    capabilities: list[InspectionCapability]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.summary),
            "duration_seconds": self.duration_seconds,
            "episodes": [asdict(episode) for episode in self.episodes],
            "incomplete_captures": [asdict(item) for item in self.incomplete_captures],
            "outcome_counts": self.outcome_counts,
            "capabilities": [asdict(item) for item in self.capabilities],
        }


@dataclass
class TimelinePoint:
    frame_index: int
    t: float | None
    relative_time_seconds: float
    observation: list[float]
    action: list[float]
    desired_action: list[float]


@dataclass
class EpisodeTimeline:
    repo_id: str
    episode: int
    fps: int
    total_frames: int | None
    offset: int
    stride: int
    returned_frames: int
    has_more: bool
    joint_names: list[str]
    actuates_hardware: bool
    capability: InspectionCapability
    points: list[TimelinePoint]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityCheck:
    id: str
    label: str
    status: Literal["pass", "warn", "fail", "skip"]
    detail: str
    remediation: str = ""
    affected: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    repo_id: str
    generated_at: float
    overall: Literal["pass", "warn", "fail"]
    checked_frames: int
    checks: list[QualityCheck]
    capabilities: list[InspectionCapability]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}, f"{path.name} must contain a JSON object"
        return value, None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"cannot read {path.name}: {exc}"


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def _timestamps(path: Path) -> tuple[float | None, float | None]:
    first: float | None = None
    last: float | None = None
    if not path.is_file():
        return first, last
    try:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                value = float(raw["t"])
                if not math.isfinite(value):
                    continue
                if first is None:
                    first = value
                last = value
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return first, last


def _duration(frames: int | None, fps: int, start: float | None, end: float | None) -> float | None:
    if start is not None and end is not None and end >= start:
        return round(end - start, 9)
    if frames is None:
        return None
    return frames / fps if fps > 0 else None


def _camera_storage(meta: dict[str, Any]) -> dict[str, str]:
    cameras = [str(value) for value in meta.get("camera_keys", [])]
    configured = meta.get("camera_storage")
    if isinstance(configured, dict):
        return {camera: str(configured.get(camera, camera)) for camera in cameras}
    return {camera: camera for camera in cameras}


def _camera_counts(
    directory: Path,
    storage: dict[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    chunk_counts: dict[str, int] = {}
    frame_counts: dict[str, int] = {}
    for camera, storage_name in storage.items():
        chunks = sorted((directory / "images" / storage_name).glob("chunk_*.npy"))
        chunk_counts[camera] = len(chunks)
        frames = 0
        for chunk in chunks:
            try:
                array = np.load(chunk, mmap_mode="r", allow_pickle=False)
                if array.ndim >= 1:
                    frames += int(array.shape[0])
            except (OSError, ValueError):
                continue
        frame_counts[camera] = frames
    return chunk_counts, frame_counts


def _read_episode_manifest(directory: Path) -> tuple[dict[str, Any], str | None]:
    try:
        validate_fallback_episode_tree(directory)
    except (OSError, ValueError) as exc:
        return {}, str(exc)
    for name in ("episode.json", "manifest.json", "outcome.json"):
        path = directory / name
        if path.is_file():
            value, error = _safe_json(path)
            if error is not None:
                return {}, error
            problem = _outcome_record_problem(value, location=path.name, require_outcome=False)
            return ({}, problem) if problem is not None else (value, None)
    return {}, None


def _normalize_outcome(value: Any) -> str:
    if isinstance(value, bool):
        return "success" if value else "failure"
    normalized = str(value or "unlabeled").strip().lower()
    aliases = {"succeeded": "success", "failed": "failure", "unknown": "unlabeled"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in _OUTCOMES else "unlabeled"


def _is_stored_outcome(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in _OUTCOMES or normalized in {"succeeded", "failed", "unknown"}


def _outcome_record_problem(
    record: dict[str, Any], *, location: str, require_outcome: bool
) -> str | None:
    """Return why a stored outcome record is semantically unsafe to use."""

    if "outcome" not in record:
        if require_outcome:
            return f"{location} has no outcome"
    else:
        raw = record["outcome"]
        if not _is_stored_outcome(raw):
            return f"{location} has invalid outcome {raw!r}; expected one of {sorted(_OUTCOMES)}"
    if "outcome_source" in record:
        raw_source = record["outcome_source"]
        if not isinstance(raw_source, str):
            return f"{location} has a non-string outcome_source"
        source = raw_source.strip()
        if source and source != HUMAN_OUTCOME_SOURCE and not _AUTO_OUTCOME_SOURCE.fullmatch(source):
            return f"{location} has invalid outcome_source {raw_source!r}"
    return None


def _outcome_source(record: dict[str, Any], outcome: str) -> str:
    raw = record.get("outcome_source")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return HUMAN_OUTCOME_SOURCE if outcome != "unlabeled" else ""


def _fallback_episodes(
    dataset: DatasetInfo,
) -> tuple[list[EpisodeManifest], list[IncompleteCapture], str | None]:
    root = Path(dataset.path)
    meta, _ = _safe_json(root / "meta.json")
    if not meta:
        manifests = sorted(root.glob("incomplete_*/INCOMPLETE.json"))
        if manifests:
            meta, _ = _safe_json(manifests[-1])
    # Every recorded stream has image chunks: color cameras and depth streams.
    meta.setdefault("camera_keys", dataset.camera_keys + dataset.depth_keys)
    storage = _camera_storage(meta)
    episodes: list[EpisodeManifest] = []
    label_problems: list[str] = []
    for directory in sorted(root.glob("episode_*")):
        try:
            episode_index = int(directory.name.removeprefix("episode_"))
        except ValueError:
            continue
        try:
            validate_fallback_episode_tree(directory)
        except (OSError, ValueError) as exc:
            label_problems.append(f"episode {episode_index}: {exc}")
            continue
        frames_path = directory / "frames.jsonl"
        frames = _count_lines(frames_path)
        started_at, ended_at = _timestamps(frames_path)
        manifest, manifest_problem = _read_episode_manifest(directory)
        chunks, camera_frames = _camera_counts(directory, storage)
        issues: list[str] = []
        if not frames_path.is_file():
            issues.append("frames.jsonl is missing")
        if manifest_problem is not None:
            problem = f"episode {episode_index}: {manifest_problem}"
            label_problems.append(problem)
            issues.append(f"outcome labels unreadable: {manifest_problem}")
        outcome = _normalize_outcome(manifest.get("outcome"))
        episodes.append(
            EpisodeManifest(
                episode=episode_index,
                name=directory.name,
                frames=frames,
                duration_seconds=_duration(frames, dataset.fps, started_at, ended_at),
                started_at=started_at,
                ended_at=ended_at,
                outcome=outcome,
                notes=str(manifest.get("notes", "")),
                complete=True,
                camera_chunks=chunks,
                camera_frames=camera_frames,
                issues=issues,
                outcome_source=_outcome_source(manifest, outcome),
            )
        )

    incomplete: list[IncompleteCapture] = []
    candidates = sorted(root.glob("incomplete_*")) + sorted(root.glob(".episode-staging-*"))
    for directory in candidates:
        if not directory.is_dir():
            continue
        metadata, _ = _safe_json(directory / "INCOMPLETE.json")
        frames_path = directory / "frames.jsonl"
        try:
            frames = int(metadata.get("frames", _count_lines(frames_path)))
        except (TypeError, ValueError):
            frames = _count_lines(frames_path)
        started_at, ended_at = _timestamps(frames_path)
        chunks, _ = _camera_counts(directory, storage)
        incomplete.append(
            IncompleteCapture(
                id=directory.name,
                frames=frames,
                duration_seconds=_duration(frames, dataset.fps, started_at, ended_at),
                reason=str(
                    metadata.get(
                        "reason",
                        "recording was interrupted before the episode was finalized",
                    )
                ),
                captured_at=_optional_float(metadata.get("aborted_at")) or _path_mtime(directory),
                camera_chunks=chunks,
            )
        )
    labels_error = "; ".join(label_problems) if label_problems else None
    return episodes, incomplete, labels_error


def _optional_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _path_mtime(path: Path) -> float | None:
    try:
        return _optional_float(path.stat().st_mtime)
    except OSError:
        return None


def _lerobot_episode_records(root: Path) -> list[dict[str, Any]]:
    """Read v2 JSONL episode metadata when present.

    LeRobot v3 stores this table as parquet.  The core install intentionally
    does not pull in a parquet stack, so v3 gets synthetic manifest rows and a
    clear partial-capability state.
    """

    candidates = (root / "meta" / "episodes.jsonl", root / "meta" / "episodes" / "episodes.jsonl")
    for path in candidates:
        if not path.is_file():
            continue
        records: list[dict[str, Any]] = []
        try:
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict):
                            records.append(item)
            return records
        except (OSError, json.JSONDecodeError):
            return []
    return []


def read_outcome_sidecar(root: Path) -> tuple[dict[int, dict[str, Any]], str | None]:
    """A LeRobot dataset's labels by episode index, and why they are unreadable.

    No sidecar means no labels yet: ``({}, None)``. A sidecar that cannot be
    read, or does not have the expected shape, returns ``({}, reason)``:
    every episode then reads as unlabeled, and callers must not mistake that
    for "nothing was labeled".
    """

    path = Path(root) / OUTCOMES_SIDECAR
    if not path.exists():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"cannot read {OUTCOMES_SIDECAR}: {exc}"
    if not isinstance(value, dict) or not isinstance(value.get("episodes"), dict):
        return {}, f"{OUTCOMES_SIDECAR} must be a JSON object with an 'episodes' object"
    schema = value.get("schema_version", OUTCOMES_SIDECAR_SCHEMA)
    if schema != OUTCOMES_SIDECAR_SCHEMA:
        return {}, (
            f"{OUTCOMES_SIDECAR} has schema_version {schema!r}; this so101 reads "
            f"version {OUTCOMES_SIDECAR_SCHEMA}"
        )
    records: dict[int, dict[str, Any]] = {}
    for key, record in value["episodes"].items():
        if not (isinstance(key, str) and key.isdecimal() and isinstance(record, dict)):
            return {}, f"{OUTCOMES_SIDECAR}: episode entry {key!r} is not an object keyed by index"
        problem = _outcome_record_problem(
            record,
            location=f"{OUTCOMES_SIDECAR} episode {key}",
            require_outcome=True,
        )
        if problem is not None:
            return {}, problem
        records[int(key)] = record
    return records, None


def _write_outcome_sidecar(root: Path, repo_id: str, records: dict[int, dict[str, Any]]) -> None:
    atomic_write_json(
        Path(root) / OUTCOMES_SIDECAR,
        {
            "schema_version": OUTCOMES_SIDECAR_SCHEMA,
            "repo_id": repo_id,
            "updated_at": time.time(),
            "episodes": {str(index): records[index] for index in sorted(records)},
        },
    )


def outcome_labels_problem(inspection: DatasetInspection) -> str | None:
    """Why the dataset's outcome labels cannot be trusted, or ``None``."""

    for capability in inspection.capabilities:
        if capability.id == OUTCOME_LABELS_CAPABILITY and not capability.available:
            return capability.detail
    return None


def lerobot_runtime_capability() -> InspectionCapability:
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-not-found] # noqa: F401

        return InspectionCapability(
            id="lerobot_runtime",
            available=True,
            level="full",
            detail="LeRobot runtime is installed; action previews and deep checks are available.",
        )
    except (ImportError, ModuleNotFoundError):
        return InspectionCapability(
            id="lerobot_runtime",
            available=False,
            level="unavailable",
            detail=(
                "LeRobot runtime is not installed. Manifest inspection remains available; "
                "install the 'robot' extra for action previews and deep validation."
            ),
        )


def _lerobot_episodes(dataset: DatasetInfo) -> tuple[list[EpisodeManifest], str | None]:
    root = Path(dataset.path)
    labels, labels_error = read_outcome_sidecar(root)
    records = _lerobot_episode_records(root)
    by_index: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(records):
        try:
            index = int(record.get("episode_index", position))
        except (TypeError, ValueError):
            continue
        by_index[index] = record

    episodes: list[EpisodeManifest] = []
    for index in range(dataset.num_episodes):
        record = by_index.get(index, {})
        raw_frames = record.get("length", record.get("num_frames"))
        try:
            frames = int(raw_frames) if raw_frames is not None else None
        except (TypeError, ValueError):
            frames = None
        # The suite's sidecar label wins over any outcome a v2 table carries.
        label = labels.get(index, record)
        outcome = _normalize_outcome(label.get("outcome", label.get("success")))
        episodes.append(
            EpisodeManifest(
                episode=index,
                name=f"episode_{index:06d}",
                frames=frames,
                duration_seconds=_duration(frames, dataset.fps, None, None),
                outcome=outcome,
                notes=str(label.get("notes", record.get("notes", ""))),
                issues=[f"outcome labels unreadable: {labels_error}"] if labels_error else [],
                outcome_source=_outcome_source(label, outcome),
            )
        )
    return episodes, labels_error


def inspect_dataset(repo_id: str) -> DatasetInspection:
    dataset = get_dataset(repo_id)
    if dataset is None:
        raise FileNotFoundError(f"dataset {repo_id!r} not found")

    if dataset.format.startswith("so101-fallback-"):
        episodes, incomplete, labels_error = _fallback_episodes(dataset)
        capabilities = [
            InspectionCapability(
                id="manifest_inspection",
                available=True,
                level="full" if labels_error is None else "partial",
                detail=(
                    "Episode manifests, timestamps, outcomes, and camera chunks are readable."
                    if labels_error is None
                    else "Episode files are present, but outcome metadata is unreadable."
                ),
            ),
            InspectionCapability(
                id="replay_preview",
                available=True,
                level="full",
                detail="Read-only action preview is available and never opens the robot.",
            ),
            InspectionCapability(
                id="quality_control",
                available=True,
                level="full",
                detail=(
                    "Frame schema, finite values, timing, limits, and camera chunks are checked."
                ),
            ),
            InspectionCapability(
                id=OUTCOME_LABELS_CAPABILITY,
                available=labels_error is None,
                level="full" if labels_error is None else "unavailable",
                detail=(
                    "Outcome labels are kept in each episode's episode.json."
                    if labels_error is None
                    else (
                        f"{labels_error}. Every affected episode reads as unlabeled, so training "
                        "selections other than 'all' are refused until the file is repaired or "
                        "restored; labels cannot be edited meanwhile."
                    )
                ),
            ),
        ]
    elif dataset.format == "lerobot":
        episodes, labels_error = _lerobot_episodes(dataset)
        incomplete = []
        runtime = lerobot_runtime_capability()
        manifest_level: Literal["full", "partial"] = (
            "full" if _lerobot_episode_records(Path(dataset.path)) else "partial"
        )
        capabilities = [
            InspectionCapability(
                id="manifest_inspection",
                available=True,
                level=manifest_level,
                detail=(
                    "LeRobot episode metadata is readable."
                    if manifest_level == "full"
                    else (
                        "Dataset totals are readable; per-episode v3 parquet metadata needs "
                        "LeRobot."
                    )
                ),
            ),
            InspectionCapability(
                id="replay_preview",
                available=runtime.available,
                level="partial" if runtime.available else "unavailable",
                detail=(
                    "Action-only read-only preview is available through LeRobot."
                    if runtime.available
                    else runtime.detail
                ),
            ),
            InspectionCapability(
                id="quality_control",
                available=True,
                level="partial",
                detail=(
                    "Manifest and action checks are available; camera/timestamp checks depend "
                    "on the installed LeRobot decoder."
                ),
            ),
            runtime,
            InspectionCapability(
                id=OUTCOME_LABELS_CAPABILITY,
                available=labels_error is None,
                level="full" if labels_error is None else "unavailable",
                detail=(
                    f"Outcome labels are kept in {OUTCOMES_SIDECAR} in the dataset folder "
                    "(LeRobot's metadata has no outcome field)."
                    if labels_error is None
                    else (
                        f"{labels_error}. Every episode reads as unlabeled, so training "
                        "selections other than 'all' are refused until the file is repaired "
                        "or restored; labels cannot be edited meanwhile."
                    )
                ),
            ),
        ]
    else:
        episodes, incomplete = [], []
        capabilities = [
            InspectionCapability(
                id="manifest_inspection",
                available=False,
                level="unavailable",
                detail=f"Unsupported dataset format: {dataset.format}",
            )
        ]

    outcome_counts = {outcome: 0 for outcome in sorted(_OUTCOMES)}
    for episode in episodes:
        outcome_counts[episode.outcome] = outcome_counts.get(episode.outcome, 0) + 1
    duration = sum(episode.duration_seconds or 0.0 for episode in episodes)
    return DatasetInspection(
        summary=dataset,
        duration_seconds=duration,
        episodes=episodes,
        incomplete_captures=incomplete,
        outcome_counts=outcome_counts,
        capabilities=capabilities,
    )


def _vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} joint values")
    return [float(item) for item in value]


def episode_timeline(
    repo_id: str,
    episode: int,
    *,
    offset: int = 0,
    limit: int = 300,
    stride: int = 1,
) -> EpisodeTimeline:
    if offset < 0 or limit < 1 or limit > 1000 or stride < 1 or stride > 10_000:
        raise ValueError("offset must be non-negative, limit 1..1000, and stride 1..10000")
    inspection = inspect_dataset(repo_id)
    manifest = next((item for item in inspection.episodes if item.episode == episode), None)
    if manifest is None:
        raise FileNotFoundError(f"episode {episode} not found in {repo_id!r}")
    dataset = inspection.summary

    if dataset.format.startswith("so101-fallback-"):
        capability = InspectionCapability(
            id="replay_preview",
            available=True,
            level="full",
            detail="Read-only preview; no hardware connection or command path is used.",
        )
        points: list[TimelinePoint] = []
        path = Path(dataset.path) / manifest.name / "frames.jsonl"
        first_t = manifest.started_at
        last_selected_index = -1
        with path.open() as handle:
            for frame_index, line in enumerate(line for line in handle if line.strip()):
                if frame_index < offset or (frame_index - offset) % stride:
                    continue
                if len(points) >= limit:
                    break
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"frame {frame_index} must contain a JSON object")
                timestamp = _optional_float(raw.get("t"))
                if first_t is None and timestamp is not None:
                    first_t = timestamp
                relative = (
                    timestamp - first_t
                    if timestamp is not None and first_t is not None
                    else frame_index / max(dataset.fps, 1)
                )
                action = _vector(raw.get("action"))
                desired_raw = raw.get("desired_action", action)
                points.append(
                    TimelinePoint(
                        frame_index=frame_index,
                        t=timestamp,
                        relative_time_seconds=max(0.0, float(relative)),
                        observation=_vector(raw.get("observation.state")),
                        action=action,
                        desired_action=_vector(desired_raw),
                    )
                )
                last_selected_index = frame_index
        total = manifest.frames
        has_more = total is not None and last_selected_index + stride < total
    else:
        runtime = lerobot_runtime_capability()
        capability = InspectionCapability(
            id="replay_preview",
            available=runtime.available,
            level="partial" if runtime.available else "unavailable",
            detail=(
                "Action-only read-only preview; LeRobot observations are not decoded in core mode."
                if runtime.available
                else runtime.detail
            ),
        )
        points = []
        total = manifest.frames
        has_more = False
        if runtime.available:
            from .replay import _load_lerobot_actions

            actions, source_fps = _load_lerobot_actions(
                Path(dataset.path), dataset.repo_id, episode
            )
            total = len(actions)
            for frame_index in range(offset, len(actions), stride):
                if len(points) >= limit:
                    break
                action = [float(value) for value in actions[frame_index]]
                points.append(
                    TimelinePoint(
                        frame_index=frame_index,
                        t=None,
                        relative_time_seconds=frame_index / max(source_fps, 1),
                        observation=[],
                        action=action,
                        desired_action=action,
                    )
                )
            has_more = bool(points) and points[-1].frame_index + stride < len(actions)

    return EpisodeTimeline(
        repo_id=repo_id,
        episode=episode,
        fps=dataset.fps,
        total_frames=total,
        offset=offset,
        stride=stride,
        returned_frames=len(points),
        has_more=has_more,
        joint_names=list(JOINT_NAMES),
        actuates_hardware=False,
        capability=capability,
        points=points,
    )


def replay_preview(repo_id: str, episode: int, *, max_points: int = 240) -> EpisodeTimeline:
    if max_points < 10 or max_points > 1000:
        raise ValueError("max_points must be between 10 and 1000")
    inspection = inspect_dataset(repo_id)
    manifest = next((item for item in inspection.episodes if item.episode == episode), None)
    if manifest is None:
        raise FileNotFoundError(f"episode {episode} not found in {repo_id!r}")
    stride = max(1, math.ceil((manifest.frames or max_points) / max_points))
    return episode_timeline(repo_id, episode, limit=max_points, stride=stride)


def _check(
    check_id: str,
    label: str,
    status: Literal["pass", "warn", "fail", "skip"],
    detail: str,
    *,
    remediation: str = "",
    affected: int = 0,
    examples: list[str] | None = None,
) -> QualityCheck:
    return QualityCheck(
        id=check_id,
        label=label,
        status=status,
        detail=detail,
        remediation=remediation,
        affected=affected,
        examples=(examples or [])[:8],
    )


def _fallback_qc(inspection: DatasetInspection) -> tuple[int, list[QualityCheck]]:
    dataset = inspection.summary
    root = Path(dataset.path)
    meta, meta_error = _safe_json(root / "meta.json")
    manifest_problems: list[str] = []
    if meta_error:
        manifest_problems.append(meta_error)
    try:
        manifest_fps = int(meta.get("fps", 0) or 0)
    except (TypeError, ValueError):
        manifest_fps = 0
    if manifest_fps <= 0:
        manifest_problems.append("fps must be positive")
    if str(meta.get("repo_id", "")) != dataset.repo_id:
        manifest_problems.append("repo_id does not match the discovered dataset")
    try:
        declared_episodes = int(meta.get("episodes", 0))
    except (TypeError, ValueError):
        declared_episodes = -1
    if declared_episodes != len(inspection.episodes):
        manifest_problems.append(
            f"manifest declares {declared_episodes} episodes, found {len(inspection.episodes)}"
        )
    if "total_frames" in meta:
        try:
            declared_frames = int(meta["total_frames"])
        except (TypeError, ValueError):
            declared_frames = -1
        discovered_frames = sum(episode.frames or 0 for episode in inspection.episodes)
        if declared_frames != discovered_frames:
            manifest_problems.append(
                f"manifest declares {declared_frames} frames, found {discovered_frames}"
            )
    checks = [
        _check(
            "manifest",
            "Dataset manifest",
            "fail" if manifest_problems else "pass",
            (
                f"{len(manifest_problems)} manifest problem(s) found."
                if manifest_problems
                else "Dataset manifest is readable and internally consistent."
            ),
            remediation="Repair meta.json or recover it from a known-good copy.",
            affected=len(manifest_problems),
            examples=manifest_problems,
        )
    ]

    schema_errors: list[str] = []
    finite_errors: list[str] = []
    timestamp_errors: list[str] = []
    timestamp_gaps: list[str] = []
    limit_errors: list[str] = []
    checked_frames = 0
    for episode in inspection.episodes:
        path = root / episode.name / "frames.jsonl"
        previous_t: float | None = None
        if not path.is_file():
            schema_errors.append(f"episode {episode.episode}: frames.jsonl is missing")
            continue
        with path.open() as handle:
            for frame_index, line in enumerate(handle):
                if not line.strip():
                    continue
                checked_frames += 1
                location = f"episode {episode.episode}, frame {frame_index}"
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    schema_errors.append(f"{location}: invalid JSON ({exc.msg})")
                    continue
                if not isinstance(raw, dict):
                    schema_errors.append(f"{location}: frame must contain a JSON object")
                    continue
                vectors: dict[str, list[float]] = {}
                for key in ("observation.state", "action"):
                    try:
                        vectors[key] = _vector(raw.get(key))
                    except (TypeError, ValueError) as exc:
                        schema_errors.append(f"{location}: {key} {exc}")
                if "desired_action" in raw:
                    try:
                        vectors["desired_action"] = _vector(raw["desired_action"])
                    except (TypeError, ValueError) as exc:
                        schema_errors.append(f"{location}: desired_action {exc}")
                for key, vector in vectors.items():
                    if not all(math.isfinite(value) for value in vector):
                        finite_errors.append(f"{location}: {key} contains a non-finite value")
                action = vectors.get("action", [])
                for position, value in enumerate(action):
                    low, high = DEFAULT_LIMITS[JOINT_NAMES[position]]
                    if math.isfinite(value) and not low <= value <= high:
                        limit_errors.append(
                            f"{location}: {JOINT_NAMES[position]}={value:g} outside "
                            f"[{low:g}, {high:g}]"
                        )
                timestamp = _optional_float(raw.get("t"))
                if timestamp is None:
                    timestamp_errors.append(f"{location}: timestamp is missing or non-finite")
                elif previous_t is not None:
                    delta = timestamp - previous_t
                    if delta <= 0:
                        timestamp_errors.append(f"{location}: timestamp did not increase")
                    elif delta > max(0.5, 5.0 / max(dataset.fps, 1)):
                        timestamp_gaps.append(f"{location}: {delta:.3f}s gap")
                if timestamp is not None:
                    previous_t = timestamp

    checks.extend(
        [
            _check(
                "schema",
                "Frame schema",
                "fail" if schema_errors else "pass",
                f"Checked {checked_frames} frames; {len(schema_errors)} schema issue(s).",
                remediation="Remove or repair malformed frames before training.",
                affected=len(schema_errors),
                examples=schema_errors,
            ),
            _check(
                "finite",
                "Finite joint values",
                "fail" if finite_errors else "pass",
                f"{len(finite_errors)} non-finite vector(s) found.",
                remediation="Discard affected episodes; NaN/Inf values are unsafe training inputs.",
                affected=len(finite_errors),
                examples=finite_errors,
            ),
            _check(
                "timestamps",
                "Timestamp continuity",
                "fail" if timestamp_errors else ("warn" if timestamp_gaps else "pass"),
                (
                    f"{len(timestamp_errors)} invalid ordering issue(s), "
                    f"{len(timestamp_gaps)} large gap(s)."
                ),
                remediation=(
                    "Inspect capture load and camera latency; re-record discontinuous episodes."
                ),
                affected=len(timestamp_errors) + len(timestamp_gaps),
                examples=timestamp_errors + timestamp_gaps,
            ),
            _check(
                "action_limits",
                "SO-101 action limits",
                "fail" if limit_errors else "pass",
                f"{len(limit_errors)} commanded joint value(s) exceed conservative limits.",
                remediation=(
                    "Do not replay affected episodes; verify calibration and command units."
                ),
                affected=len(limit_errors),
                examples=limit_errors,
            ),
        ]
    )

    storage = _camera_storage(meta)
    camera_errors: list[str] = []
    camera_warnings: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    for episode in inspection.episodes:
        expected = episode.frames or 0
        for camera, storage_name in storage.items():
            directory = root / episode.name / "images" / storage_name
            chunks = sorted(directory.glob("chunk_*.npy"))
            indexes: list[int] = []
            decoded_frames = 0
            for chunk in chunks:
                try:
                    indexes.append(int(chunk.stem.removeprefix("chunk_")))
                    array = np.load(chunk, mmap_mode="r", allow_pickle=False)
                    if array.ndim != 4 or array.shape[-1] not in (1, 3, 4):
                        camera_errors.append(
                            f"episode {episode.episode}, {camera}: {chunk.name} has shape "
                            f"{array.shape}"
                        )
                        continue
                    decoded_frames += int(array.shape[0])
                    shape = tuple(int(value) for value in array.shape[1:])
                    if camera in shapes and shapes[camera] != shape:
                        camera_errors.append(
                            f"episode {episode.episode}, {camera}: shape {shape} != "
                            f"{shapes[camera]}"
                        )
                    shapes.setdefault(camera, shape)
                except (OSError, ValueError) as exc:
                    camera_errors.append(
                        f"episode {episode.episode}, {camera}: cannot read {chunk.name} ({exc})"
                    )
            if indexes and indexes != list(range(len(indexes))):
                camera_errors.append(
                    f"episode {episode.episode}, {camera}: chunk indexes have gaps"
                )
            if decoded_frames != expected:
                camera_errors.append(
                    f"episode {episode.episode}, {camera}: {decoded_frames} image frames for "
                    f"{expected} records"
                )
            if not chunks and expected:
                camera_warnings.append(f"episode {episode.episode}, {camera}: no camera chunks")

    if not storage:
        camera_detail = "No camera streams are configured for this dataset."
        camera_status: Literal["pass", "warn", "fail", "skip"] = "pass"
    elif camera_errors:
        camera_detail = f"{len(camera_errors)} camera chunk integrity issue(s)."
        camera_status = "fail"
    elif camera_warnings:
        camera_detail = f"{len(camera_warnings)} camera stream warning(s)."
        camera_status = "warn"
    else:
        camera_detail = "Camera chunks are readable, contiguous, and aligned with frame records."
        camera_status = "pass"
    checks.append(
        _check(
            "camera_chunks",
            "Camera chunk integrity",
            camera_status,
            camera_detail,
            remediation="Recover or re-record episodes with missing/corrupt camera chunks.",
            affected=len(camera_errors) + len(camera_warnings),
            examples=camera_errors + camera_warnings,
        )
    )

    incomplete_count = len(inspection.incomplete_captures)
    checks.append(
        _check(
            "incomplete_captures",
            "Incomplete captures",
            "warn" if incomplete_count else "pass",
            (
                f"{incomplete_count} interrupted capture(s) are preserved for recovery."
                if incomplete_count
                else "No interrupted captures found."
            ),
            remediation="Review preserved captures, then recover or discard them deliberately.",
            affected=incomplete_count,
            examples=[item.id for item in inspection.incomplete_captures],
        )
    )
    return checked_frames, checks


def _lerobot_qc(inspection: DatasetInspection) -> tuple[int, list[QualityCheck]]:
    dataset = inspection.summary
    root = Path(dataset.path)
    info, info_error = _safe_json(root / "meta" / "info.json")
    manifest_errors: list[str] = []
    if info_error:
        manifest_errors.append(info_error)
    try:
        manifest_fps = int(info.get("fps", 0) or 0)
    except (TypeError, ValueError):
        manifest_fps = 0
    if manifest_fps <= 0:
        manifest_errors.append("fps must be positive")
    checks = [
        _check(
            "manifest",
            "LeRobot manifest",
            "fail" if manifest_errors else "pass",
            (
                f"{len(manifest_errors)} manifest problem(s) found."
                if manifest_errors
                else "meta/info.json is readable and has a valid frame rate."
            ),
            remediation="Regenerate or repair the LeRobot metadata.",
            affected=len(manifest_errors),
            examples=manifest_errors,
        )
    ]
    runtime = lerobot_runtime_capability()
    if not runtime.available:
        for check_id, label in (
            ("schema", "Frame schema and finite values"),
            ("timestamps", "Timestamp continuity"),
            ("action_limits", "SO-101 action limits"),
            ("camera_chunks", "Camera media integrity"),
        ):
            checks.append(
                _check(
                    check_id,
                    label,
                    "skip",
                    "Deep validation needs the optional LeRobot runtime.",
                    remediation="Install the 'robot' extra and run QC again.",
                )
            )
        return 0, checks

    from .replay import _load_lerobot_actions

    action_errors: list[str] = []
    finite_errors: list[str] = []
    checked = 0
    for episode in inspection.episodes:
        try:
            actions, _ = _load_lerobot_actions(root, dataset.repo_id, episode.episode)
        except Exception as exc:  # e.g. datasets' DatasetGenerationError (B22)
            # A corrupt episode is exactly what QC must report, as a failed
            # check rather than an HTTP 500.
            action_errors.append(
                f"episode {episode.episode}: cannot decode actions ({type(exc).__name__}: {exc})"
            )
            continue
        for frame_index, action in enumerate(actions):
            checked += 1
            if len(action) != len(JOINT_NAMES):
                action_errors.append(
                    f"episode {episode.episode}, frame {frame_index}: expected 6 action values"
                )
                continue
            for position, value in enumerate(action):
                if not math.isfinite(float(value)):
                    finite_errors.append(
                        f"episode {episode.episode}, frame {frame_index}: non-finite action"
                    )
                    continue
                low, high = DEFAULT_LIMITS[JOINT_NAMES[position]]
                if not low <= float(value) <= high:
                    action_errors.append(
                        f"episode {episode.episode}, frame {frame_index}: "
                        f"{JOINT_NAMES[position]}={float(value):g} outside [{low:g}, {high:g}]"
                    )
    checks.extend(
        [
            _check(
                "schema",
                "Action schema and finite values",
                "fail" if finite_errors else "pass",
                f"Checked {checked} actions; {len(finite_errors)} non-finite value(s).",
                remediation="Remove or repair malformed rows before training.",
                affected=len(finite_errors),
                examples=finite_errors,
            ),
            _check(
                "action_limits",
                "SO-101 action limits",
                "fail" if action_errors else "pass",
                f"{len(action_errors)} decode or limit issue(s).",
                remediation="Verify action names, units, calibration, and damaged episodes.",
                affected=len(action_errors),
                examples=action_errors,
            ),
            _check(
                "timestamps",
                "Timestamp continuity",
                "skip",
                "Timestamp decoding is not exposed by the lightweight LeRobot adapter.",
            ),
            _check(
                "camera_chunks",
                "Camera media integrity",
                "skip",
                "Video decoding is delegated to LeRobot training/visualization tooling.",
            ),
        ]
    )
    return checked, checks


def run_quality_control(repo_id: str) -> QualityReport:
    inspection = inspect_dataset(repo_id)
    if inspection.summary.format.startswith("so101-fallback-"):
        checked, checks = _fallback_qc(inspection)
    elif inspection.summary.format == "lerobot":
        checked, checks = _lerobot_qc(inspection)
    else:
        checked, checks = (
            0,
            [
                _check(
                    "format",
                    "Dataset format",
                    "fail",
                    f"Unsupported dataset format: {inspection.summary.format}",
                )
            ],
        )
    statuses = {check.status for check in checks}
    overall: Literal["pass", "warn", "fail"] = (
        "fail" if "fail" in statuses else ("warn" if statuses & {"warn", "skip"} else "pass")
    )
    return QualityReport(
        repo_id=repo_id,
        generated_at=time.time(),
        overall=overall,
        checked_frames=checked,
        checks=checks,
        capabilities=inspection.capabilities,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _validated_outcome_source(source: str) -> str:
    value = str(source or "").strip()
    if value == HUMAN_OUTCOME_SOURCE or _AUTO_OUTCOME_SOURCE.match(value):
        return value
    raise ValueError(
        "outcome source must be 'human' or 'auto:<predicate>' "
        "(lowercase letters, digits, '_', '.', ':' or '-')"
    )


def _validated_outcome_detail(detail: Any) -> Any:
    if detail is None:
        return None
    try:
        encoded = json.dumps(detail, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome detail must be JSON-serializable with finite numbers") from exc
    if len(encoded.encode()) > _OUTCOME_DETAIL_MAX_BYTES:
        raise ValueError(f"outcome detail exceeds {_OUTCOME_DETAIL_MAX_BYTES} bytes")
    return json.loads(encoded)


def _apply_outcome(
    record: dict[str, Any],
    outcome: str,
    notes: str,
    source: str,
    detail: Any,
) -> bool:
    """Apply one label to an episode's label record in place, by the rules below."""

    current_outcome = _normalize_outcome(record.get("outcome"))
    current_source = _outcome_source(record, current_outcome)
    if (
        source != HUMAN_OUTCOME_SOURCE
        and current_source == HUMAN_OUTCOME_SOURCE
        and current_outcome != "unlabeled"
    ):
        return False
    history = record.get("outcome_history")
    history = list(history) if isinstance(history, list) else []
    if current_outcome != "unlabeled" or "outcome_source" in record:
        previous: dict[str, Any] = {
            "outcome": current_outcome,
            "source": current_source,
            "labeled_at": record.get("labeled_at"),
        }
        if record.get("outcome_detail") is not None:
            previous["detail"] = record["outcome_detail"]
        history.append(previous)
    record.update(
        outcome=outcome,
        # An automatic label keeps the operator's notes unless it brings its own.
        notes=notes if (notes or source == HUMAN_OUTCOME_SOURCE) else record.get("notes", ""),
        labeled_at=time.time(),
        outcome_source=source,
        outcome_history=history[-_OUTCOME_HISTORY_LIMIT:],
    )
    if detail is None:
        record.pop("outcome_detail", None)
    else:
        record["outcome_detail"] = detail
    return True


def set_episode_outcome(
    repo_id: str,
    episode: int,
    outcome: str,
    notes: str = "",
    *,
    source: str = HUMAN_OUTCOME_SOURCE,
    detail: Any = None,
) -> bool:
    """Label one episode's outcome, recording who or what decided it.

    ``source`` is ``"human"`` (the default, used by the Datasets page) or
    ``"auto:<predicate>"`` for automatic labels such as a simulation task
    predicate (``"auto:sim_predicate"``). ``detail`` is optional JSON-serializable
    evidence stored with the label (for example, the predicate's measurements).

    Human labels always win: an automatic label never replaces a human
    ``success``/``failure``/``partial``/``discarded`` label (labels saved
    without a source count as human), while a human label replaces anything.
    ``unlabeled`` is not a label, so automation may label an episode a human
    reset to ``unlabeled``. The previous label is kept in ``outcome_history``
    (the last 20).

    Fallback datasets keep the label in the episode's ``episode.json``;
    LeRobot datasets keep it in the dataset's :data:`OUTCOMES_SIDECAR`, which is
    written atomically (temp file, fsync, rename). Neither store is overwritten
    while unreadable or semantically invalid
    (:class:`OutcomeLabelsUnreadableError`).

    Returns ``True`` when the label was written and ``False`` when an automatic
    label was skipped because a human label exists.
    """

    operation_lock = dataset_operation_lock(repo_id)
    if not operation_lock.acquire(blocking=False):
        raise DatasetBusyError(dataset_busy_message(repo_id))
    try:
        return _set_episode_outcome_locked(
            repo_id,
            episode,
            outcome,
            notes,
            source=source,
            detail=detail,
        )
    finally:
        operation_lock.release()


def _set_episode_outcome_locked(
    repo_id: str,
    episode: int,
    outcome: str,
    notes: str = "",
    *,
    source: str = HUMAN_OUTCOME_SOURCE,
    detail: Any = None,
) -> bool:
    """Set an outcome while the caller owns the dataset operation lock."""

    normalized = _normalize_outcome(outcome)
    if normalized != outcome or outcome not in _OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(_OUTCOMES)}")
    source = _validated_outcome_source(source)
    detail = _validated_outcome_detail(detail)
    dataset = get_dataset(repo_id)
    if dataset is None:
        raise FileNotFoundError(f"dataset {repo_id!r} not found")
    if dataset.format == "lerobot":
        return _set_lerobot_outcome(dataset, episode, outcome, notes, source, detail)
    if not dataset.format.startswith("so101-fallback-"):
        raise NotImplementedError(
            f"outcome labels support SO-101 fallback and LeRobot datasets, not {dataset.format}"
        )
    directory = Path(dataset.path) / f"episode_{episode:06d}"
    if directory.is_symlink():
        raise OutcomeLabelsUnreadableError(
            f"episode {episode} of {repo_id!r} is a symlink; labels were not changed"
        )
    if not directory.is_dir():
        raise FileNotFoundError(f"episode {episode} not found in {repo_id!r}")
    payload, problem = _read_episode_manifest(directory)
    if problem is not None:
        raise OutcomeLabelsUnreadableError(
            f"{problem}; labels were not changed. Repair the episode metadata in {directory} "
            "or restore it from a copy, then label again"
        )
    if not _apply_outcome(payload, outcome, notes, source, detail):
        return False
    payload.update(schema_version=1, episode=episode)
    _atomic_json(directory / "episode.json", payload)
    return True


def _set_lerobot_outcome(
    dataset: DatasetInfo,
    episode: int,
    outcome: str,
    notes: str,
    source: str,
    detail: Any,
) -> bool:
    if not 0 <= episode < dataset.num_episodes:
        raise FileNotFoundError(f"episode {episode} not found in {dataset.repo_id!r}")
    root = Path(dataset.path)
    records, problem = read_outcome_sidecar(root)
    if problem is not None:
        raise OutcomeLabelsUnreadableError(
            f"{problem}; labels were not changed. Repair {root / OUTCOMES_SIDECAR} "
            "or restore it from a copy, then label again"
        )
    record = dict(records.get(episode, {}))
    if not _apply_outcome(record, outcome, notes, source, detail):
        return False
    records[episode] = record
    _write_outcome_sidecar(root, dataset.repo_id, records)
    return True
