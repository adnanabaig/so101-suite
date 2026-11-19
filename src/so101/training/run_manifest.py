"""Run manifest (``so101_run.json``): what a training run was trained on.

Every so101 training run writes ``<run>/so101_run.json`` beside its
``checkpoints/`` directory: the dataset (repo id, local root, format, fps,
task strings, camera keys), the preset and trainer, the step budget, the
validated overrides, the episode selection, the backend and the git revision.
Rollout and eval read it to default the task (VLA policies need one) and the
control rate (a policy trained on 15 fps data must run at 15 fps), and the
model registry shows it.

``episode_selection`` (manifests written before it existed lack it) records the
outcome-label selection (``kept``, ``success`` or ``all``), ``episodes``, the
indices the trainer was given, and ``skipped``, the episodes left out with
their ``outcome`` and ``reason``. ``episodes`` is null when the trainer got
every episode (``all``) or chose them itself (``bc_baseline``, whose
checkpoint manifest lists them; ``skipped`` is null then too).

The manifest is written with fsync (file and directory) before the rename. A
manifest that exists but can't be read is an error (``InvalidCheckpointError``),
never a silent fallback: running a policy at a guessed rate or with a guessed
instruction is worse than refusing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..atomic_io import atomic_write_json
from ..data.dataset import DatasetInfo, get_dataset
from ..logging import get_logger
from .configs import PolicyPreset
from .jobs import TrainingJob

if TYPE_CHECKING:
    from .preflight import EpisodePlan

log = get_logger(__name__)

RUN_MANIFEST_NAME = "so101_run.json"
RUN_MANIFEST_VERSION = 1
# outputs/<run>/checkpoints/<label>/pretrained_model -> outputs/<run>
_MAX_LEVELS_UP = 4
_SHA_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")  # SHA-1 or SHA-256


def dataset_tasks(info: DatasetInfo) -> list[str]:
    """Return the task strings of a local dataset (best effort, never raises)."""

    tasks: list[str] = []
    if info.task:
        tasks.append(info.task)
    root = Path(info.path)
    legacy = root / "meta" / "tasks.jsonl"  # LeRobot v2.1
    if legacy.is_file():
        try:
            for line in legacy.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    task = json.loads(line).get("task")
                    if isinstance(task, str) and task:
                        tasks.append(task)
        except (OSError, ValueError, AttributeError) as exc:
            log.warning("could not read %s: %s", legacy, exc)
    current = root / "meta" / "tasks.parquet"  # LeRobot v3 (needs pyarrow)
    if current.is_file():
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]

            table = pq.read_table(current)
            if "task" in table.column_names:
                column = table.column("task").to_pylist()
            else:  # v3 stores the task string as the index
                index = table.schema.pandas_metadata or {}
                names = [name for name in index.get("index_columns", []) if isinstance(name, str)]
                column = table.column(names[0]).to_pylist() if names else []
            tasks.extend(str(task) for task in column if task)
        except ImportError:
            pass
        except Exception as exc:  # corrupt metadata must not block training
            log.warning("could not read %s: %s", current, exc)
    return list(dict.fromkeys(tasks))


def _git_sha(start: Path | None = None) -> str | None:
    """The commit checked out in the so101 source tree, or None if there is none.

    Read from the repository files rather than by running ``git``, so it needs
    no git on PATH. Covers worktrees (``.git`` is a ``gitdir:`` file), loose
    and packed refs, and a detached HEAD; anything else is None.
    """

    here = (start or Path(__file__)).resolve()
    dot_git = next((d / ".git" for d in here.parents if (d / ".git").exists()), None)
    if dot_git is None:
        return None
    try:
        git_dir = dot_git
        if dot_git.is_file():
            pointer = dot_git.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = (dot_git.parent / pointer.removeprefix("gitdir:").strip()).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head if _SHA_RE.fullmatch(head) else None
        ref = head.removeprefix("ref:").strip()
        common = git_dir
        if (git_dir / "commondir").is_file():
            shared = (git_dir / "commondir").read_text(encoding="utf-8").strip()
            common = (git_dir / shared).resolve()
        for base in (git_dir, common):  # a loose ref wins over packed-refs
            if (base / ref).is_file():
                sha = (base / ref).read_text(encoding="utf-8").strip()
                return sha if _SHA_RE.fullmatch(sha) else None
        if (common / "packed-refs").is_file():
            for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
                sha, _, name = line.partition(" ")
                if name.strip() == ref and _SHA_RE.fullmatch(sha):
                    return sha
    except OSError:
        return None
    return None


def build_run_manifest(
    job: TrainingJob,
    preset: PolicyPreset,
    *,
    trainer: str,
    info: DatasetInfo | None = None,
    episodes: EpisodePlan | None = None,
) -> dict[str, Any]:
    """Describe a run; ``info`` defaults to the job's local dataset (if any).

    ``episodes`` is the run's episode plan; without one only the job's
    selection is recorded.
    """

    from ..policy.checkpoint import active_backend
    from .preflight import EpisodePlan

    if episodes is None:
        episodes = EpisodePlan(job.episodes, episodes=None, skipped=None)
    info = info if info is not None else get_dataset(job.repo_id)
    tasks = dataset_tasks(info) if info is not None else []
    dataset: dict[str, Any] = {"repo_id": job.repo_id, "remote": info is None}
    if info is not None:
        dataset.update(
            root=info.path,
            format=info.format,
            fps=info.fps,
            tasks=tasks,
            camera_keys=list(info.camera_keys),
            episodes=info.num_episodes,
            frames=info.num_frames,
        )
    return {
        "schema_version": RUN_MANIFEST_VERSION,
        "job_id": job.id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trainer": trainer,
        "preset": preset.name,
        "policy_type": preset.policy_type,
        "dataset": dataset,
        "fps": info.fps if info is not None else None,
        # A single-task dataset defines the rollout task; a multi-task one
        # needs an explicit --task.
        "task": tasks[0] if len(tasks) == 1 else None,
        "steps": job.steps,
        "batch_size": job.batch_size,
        "extra_args": {**preset.extra_args, **job.extra_args},
        "episode_selection": episodes.to_manifest(),
        "backend": active_backend(),
        "so101_version": __version__,
        "git_sha": _git_sha(),
    }


def write_run_manifest(
    job: TrainingJob,
    preset: PolicyPreset,
    *,
    trainer: str,
    info: DatasetInfo | None = None,
    episodes: EpisodePlan | None = None,
) -> Path | None:
    """Write ``<job.output_dir>/so101_run.json``; a failure is logged, never raised."""

    path = Path(job.output_dir) / RUN_MANIFEST_NAME
    try:
        manifest = build_run_manifest(job, preset, trainer=trainer, info=info, episodes=episodes)
        atomic_write_json(path, manifest)
    except Exception as exc:  # the manifest must never fail a training run
        log.warning("could not write run manifest %s: %s", path, exc)
        return None
    return path


def find_run_manifest(path: str | Path) -> Path | None:
    """Find the run manifest for a checkpoint (or run) directory."""

    current = Path(path).expanduser()
    for _ in range(_MAX_LEVELS_UP + 1):
        candidate = current / RUN_MANIFEST_NAME
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def read_run_manifest(path: str | Path) -> dict[str, Any] | None:
    """Return the run manifest for a checkpoint or run directory, if one exists.

    Raises :class:`~so101.policy.checkpoint.InvalidCheckpointError` when the
    manifest exists but can't be read or isn't a JSON object (for example a
    file truncated by a power cut): callers must not fall back to defaults.
    """

    found = find_run_manifest(path)
    if found is None:
        return None
    try:
        payload = json.loads(found.read_text(encoding="utf-8"))
        problem = None if isinstance(payload, dict) else "it is not a JSON object"
    except (OSError, ValueError) as exc:
        problem = str(exc)
    if problem is None:
        return payload
    from ..policy.checkpoint import InvalidCheckpointError

    raise InvalidCheckpointError(
        f"the run manifest {found} can't be read ({problem}). It records the task and "
        "frame rate this policy was trained with, so so101 won't guess them. Restore the "
        "file, or delete it and give the task and fps yourself (--task and --fps)."
    )


@dataclass(frozen=True)
class CheckpointDefaults:
    """What a checkpoint was trained on: the defaults for rollout and eval."""

    task: str | None = None
    fps: int | None = None
    policy_type: str | None = None
    dataset: str | None = None


def _control_fps(value: object, *, source: str) -> int | None:
    if value is None:
        return None
    try:
        # One shared bound/whole-number rule for CLI, API and checkpoint data.
        from ..policy.checkpoint import resolve_control_fps

        return resolve_control_fps(value)  # type: ignore[arg-type]
    except ValueError as exc:
        from ..policy.checkpoint import InvalidCheckpointError

        raise InvalidCheckpointError(f"{source} records an invalid fps ({value!r}): {exc}") from exc


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def checkpoint_defaults(path: str | Path) -> CheckpointDefaults:
    """Task, fps, policy type and dataset a checkpoint was trained on.

    Sources, first match wins: the so101 run manifest; the numpy baseline's
    ``policy_manifest.json``; LeRobot's ``train_config.json`` (its dataset, if
    present locally, supplies the task and fps). A run manifest that exists but
    can't be read raises ``InvalidCheckpointError`` (see :func:`read_run_manifest`).
    """

    path = Path(path).expanduser()
    task: str | None = None
    fps: int | None = None
    policy_type: str | None = None
    dataset: str | None = None

    run = read_run_manifest(path) or {}
    task = _text(run.get("task"))
    fps = _control_fps(run.get("fps"), source="the run manifest")
    policy_type = _text(run.get("policy_type"))
    run_dataset = run.get("dataset")
    if isinstance(run_dataset, dict):
        dataset = _text(run_dataset.get("repo_id"))

    for name in ("policy_manifest.json", "train_config.json"):
        candidate = path / name
        if not candidate.is_file():
            continue
        try:
            meta = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        data = _mapping(meta.get("dataset"))
        if name == "policy_manifest.json":
            task = task or _text(meta.get("task")) or _text(data.get("task"))
            if fps is None:
                fps = _control_fps(data.get("fps"), source=f"the checkpoint metadata {candidate}")
            policy_type = policy_type or _text(meta.get("type"))
        else:
            policy_type = policy_type or _text(_mapping(meta.get("policy")).get("type"))
        dataset = dataset or _text(data.get("repo_id"))

    if dataset and (task is None or fps is None):
        try:
            info = get_dataset(dataset)
        except ValueError:
            info = None
        if info is not None:
            if fps is None:
                fps = _control_fps(info.fps, source=f"dataset {dataset!r}")
            if task is None:
                tasks = dataset_tasks(info)
                task = tasks[0] if len(tasks) == 1 else None
    return CheckpointDefaults(task=task, fps=fps, policy_type=policy_type, dataset=dataset)
