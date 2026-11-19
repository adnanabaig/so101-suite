"""Record teleop episodes into a dataset.

Whenever LeRobot's dataset support is installed (the default install, real
hardware and the simulation backend alike) the recorder writes a
``LeRobotDataset``, so the data trains directly with LeRobot. On real hardware
it refuses to start without it, rather than silently recording a format real
training cannot use. In mock mode, and on the simulation backend without
LeRobot, it writes a self-describing fallback layout (``meta.json`` +
per-episode ``frames.jsonl`` + ``images/``) that the rest of the platform
(browser, replay, stats, the numpy baseline trainer) reads directly.

Outcome labels are not part of either layout's recording: new episodes are
``unlabeled`` until :func:`so101.data.inspection.set_episode_outcome` labels
them (fallback: ``episode.json``; LeRobot: ``so101_outcomes.json``).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import JOINT_NAMES
from ..atomic_io import durable_makedirs, fsync_directory
from ..config import get_settings
from ..hardware.camera_spec import is_depth_key
from ..hardware.cameras import stream_feature_names
from ..logging import get_logger
from ..teleop.controller import TeleopStep
from .dataset import (
    FALLBACK_PUBLISH_TRANSACTION,
    DatasetBusyError,
    dataset_busy_message,
    dataset_operation_target,
    refuse_case_alias,
    validate_fallback_episode_tree,
)

log = get_logger(__name__)
_FALLBACK_FORMATS = (None, "so101-fallback-v1", "so101-fallback-v2")
_LEROBOT_GENERATED_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
_LEROBOT_GENERATED_VIDEO_INFO = {
    "video.height",
    "video.width",
    "video.channels",
    "video.fps",
    "video.codec",
    "video.pix_fmt",
    "video.is_depth_map",
    "has_audio",
}


def _exact_int(value: Any, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{label} must be an exact {qualifier} integer, got {value!r}")
    return value


@dataclass
class RecordingConfig:
    repo_id: str  # e.g. "user/so101_pick_place"
    task: str = "Perform the task."
    fps: int = 30
    camera_keys: list[str] = field(default_factory=list)
    use_videos: bool = True
    # Fallback recordings stream bounded chunks instead of retaining an entire
    # episode's RGB frames in RAM.
    image_chunk_frames: int = 30
    # Optional ``observation.environment_state`` (float32, one value per name),
    # e.g. the simulator's object poses. Empty: the feature is not recorded.
    environment_state_names: list[str] = field(default_factory=list)
    # Optional ``observation.images.<name>`` per camera key; cameras not listed
    # keep ``stream_feature_names(camera_keys)[camera]`` (a depth stream
    # ``<entry>#depth`` records as its camera's name plus ``_depth``).
    camera_feature_names: dict[str, str] = field(default_factory=dict)


_PROCESS_DATASET_LOCK = threading.RLock()


@contextmanager
def _video_encoder_errors() -> Iterator[None]:
    """Turn LeRobot's encoder lookup failure (``resolve_vcodec``) into a clear error.

    LeRobot 0.6.1 builds its color (libsvtav1) and depth (HEVC, gray12le)
    encoder configs whenever it opens a dataset for writing, and raises a bare
    ``ValueError`` when PyAV lacks one.
    """
    try:
        yield
    except ValueError as exc:
        text = str(exc)
        if "video codec" not in text.lower() and "vcodec" not in text.lower():
            raise
        raise RuntimeError(
            f"LeRobot could not set up its video encoders ({text}). LeRobot 0.6.1 encodes "
            "color with libsvtav1 and depth with HEVC (libx265, gray12le) through PyAV; "
            "install the PyAV it pins (av>=15,<16, whose Windows and Linux wheels bundle "
            "libx265) in this environment."
        ) from exc


@contextmanager
def _dataset_lock(root: Path) -> Iterator[None]:
    """Serialize episode allocation across threads and, on Unix, processes."""

    with _PROCESS_DATASET_LOCK:
        if root.is_symlink():
            raise ValueError(f"dataset root {root} is a symlink; refusing to follow it")
        durable_makedirs(root)
        if root.is_symlink():
            raise ValueError(f"dataset root {root} became a symlink; refusing to follow it")
        lock_path = root / ".dataset.lock"
        with lock_path.open("a+") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows fallback
                fcntl = None  # type: ignore[assignment]
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _fsync_dir(path: Path) -> None:
    """Persist ``path``'s directory entries (new files, renames into or out of it).

    Unsupported directory handles/fsync are ignored by ``fsync_directory``;
    real storage failures (EIO, ENOSPC, ...) propagate so publication remains
    journalled and fail closed.
    """

    fsync_directory(path)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:  # Windows flushes only a writable handle
        os.fsync(handle.fileno())


def _sync_stage(stage: Path) -> None:
    """Make a staged episode durable before it is renamed into the dataset.

    Image chunks are written during capture without fsync (an fsync per chunk
    would stall the control loop), so each one is flushed here, then every
    directory of the stage, so a power cut cannot publish an episode whose
    chunks are empty or missing.
    """

    validate_fallback_episode_tree(stage)
    images = stage / "images"
    directories = sorted(path for path in images.glob("*") if path.is_dir())
    for directory in directories:
        for chunk in sorted(directory.glob("chunk_*.npy")):
            _fsync_file(chunk)
        _fsync_dir(directory)
    if images.is_dir():
        _fsync_dir(images)
    _fsync_dir(stage)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a small JSON manifest without exposing a partial file."""

    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    _fsync_dir(path.parent)  # persist the rename


def _read_json_object(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {path} must be a JSON object")
    return payload


def _fallback_transaction(
    root: Path, *, expected_repo_id: str | None = None
) -> dict[str, Any] | None:
    """Read and validate an interrupted fallback publication marker.

    The marker itself is atomically written, and every field is checked before
    recovery mutates either published path.
    """

    path = root / FALLBACK_PUBLISH_TRANSACTION
    if path.is_symlink():
        raise ValueError(f"fallback publication transaction {path} is a symlink")
    payload = _read_json_object(path, "fallback publication transaction")
    if payload is None:
        return None
    expected_keys = {
        "schema_version",
        "episode",
        "stage",
        "destination",
        "previous_metadata",
        "next_metadata",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != 1:
        raise ValueError(f"fallback publication transaction {path} has an invalid schema")
    episode = _exact_int(payload.get("episode"), "transaction episode", minimum=0)
    stage_name = payload.get("stage")
    destination = payload.get("destination")
    if (
        not isinstance(stage_name, str)
        or Path(stage_name).name != stage_name
        or not stage_name.startswith(".episode-staging-")
    ):
        raise ValueError(f"fallback publication transaction {path} has an invalid stage")
    expected_destination = f"episode_{episode:06d}"
    if destination != expected_destination:
        raise ValueError(
            f"fallback publication transaction {path} destination must be {expected_destination!r}"
        )

    previous = payload.get("previous_metadata")
    following = payload.get("next_metadata")
    if previous is not None and not isinstance(previous, dict):
        raise ValueError(f"fallback publication transaction {path} previous metadata is invalid")
    if not isinstance(following, dict):
        raise ValueError(f"fallback publication transaction {path} next metadata is invalid")
    previous_episodes = (
        0
        if previous is None
        else _exact_int(previous.get("episodes", 0), "previous episode count", minimum=0)
    )
    if previous_episodes != episode:
        raise ValueError(
            f"fallback publication transaction {path} starts at episode "
            f"{previous_episodes}, expected {episode}"
        )
    if _exact_int(following.get("episodes"), "next episode count", minimum=0) != episode + 1:
        raise ValueError(
            f"fallback publication transaction {path} does not publish exactly one episode"
        )
    next_repo_id = following.get("repo_id")
    if not isinstance(next_repo_id, str) or not next_repo_id:
        raise ValueError(f"fallback publication transaction {path} has no dataset id")
    if following.get("format") != "so101-fallback-v2":
        raise ValueError(f"fallback publication transaction {path} has an invalid format")
    _exact_int(following.get("fps"), "transaction FPS", minimum=1)
    camera_keys = following.get("camera_keys")
    if (
        not isinstance(following.get("task"), str)
        or not isinstance(camera_keys, list)
        or not all(isinstance(camera, str) for camera in camera_keys)
        or len(camera_keys) != len(set(camera_keys))
    ):
        raise ValueError(f"fallback publication transaction {path} has invalid metadata")
    if expected_repo_id is not None and next_repo_id != expected_repo_id:
        raise ValueError(
            f"fallback publication transaction {path} belongs to {next_repo_id!r}, "
            f"not {expected_repo_id!r}"
        )
    if previous is not None and next_repo_id != previous.get("repo_id"):
        raise ValueError(f"fallback publication transaction {path} changes the dataset id")
    return payload


def _transaction_previous_frame_count(root: Path, episodes: int) -> int:
    """Validate and count only the episodes preceding a journalled publication."""

    total = 0
    for index in range(episodes):
        directory = root / f"episode_{index:06d}"
        validate_fallback_episode_tree(directory)
        manifest = _read_json_object(directory / "episode.json", "fallback episode metadata")
        if manifest is None:
            raise ValueError(f"fallback transaction is missing prior episode {index}")
        stored_index = _exact_int(manifest.get("episode"), "prior episode index", minimum=0)
        frames = _exact_int(manifest.get("frames"), "prior episode frame count", minimum=1)
        if stored_index != index:
            raise ValueError(
                f"fallback transaction prior episode {directory} stores index {stored_index}"
            )
        try:
            with (directory / "frames.jsonl").open(encoding="utf-8") as handle:
                actual = sum(1 for line in handle if line.strip())
        except OSError as exc:
            raise ValueError(
                f"fallback transaction prior episode has no readable frames: {directory}"
            ) from exc
        if actual != frames:
            raise ValueError(
                f"fallback transaction prior episode {directory} stores {frames} frames, "
                f"but frames.jsonl has {actual}"
            )
        total += frames
    return total


def _validate_transaction_episode(
    directory: Path,
    *,
    episode: int,
    previous_metadata: dict[str, Any] | None,
    next_metadata: dict[str, Any],
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"fallback transaction episode is not a directory: {directory}")
    validate_fallback_episode_tree(directory)
    manifest = _read_json_object(directory / "episode.json", "fallback episode metadata")
    if manifest is None:
        raise ValueError(f"fallback transaction episode has no episode.json: {directory}")
    stored_episode = _exact_int(manifest.get("episode"), "transaction episode index", minimum=0)
    frames = _exact_int(manifest.get("frames"), "transaction frame count", minimum=1)
    if stored_episode != episode:
        raise ValueError(
            f"fallback transaction episode {directory} stores index {stored_episode}, "
            f"expected {episode}"
        )
    if manifest.get("task") != next_metadata.get("task") or manifest.get(
        "camera_keys"
    ) != next_metadata.get("camera_keys"):
        raise ValueError(
            f"fallback transaction episode {directory} conflicts with dataset metadata"
        )
    try:
        with (directory / "frames.jsonl").open(encoding="utf-8") as handle:
            actual_frames = sum(1 for line in handle if line.strip())
    except OSError as exc:
        raise ValueError(
            f"fallback transaction episode has no readable frames: {directory}"
        ) from exc
    if actual_frames != frames:
        raise ValueError(
            f"fallback transaction episode {directory} stores {frames} frames, "
            f"but frames.jsonl has {actual_frames}"
        )

    previous_frames = _transaction_previous_frame_count(directory.parent, episode)
    if previous_metadata is not None and "total_frames" in previous_metadata:
        stored_previous_frames = _exact_int(
            previous_metadata["total_frames"], "previous total frame count", minimum=0
        )
        if stored_previous_frames != previous_frames:
            raise ValueError(
                f"fallback publication transaction previous total frame count is "
                f"{stored_previous_frames}, expected {previous_frames}"
            )
    next_frames = _exact_int(next_metadata.get("total_frames"), "next total frame count", minimum=0)
    if next_frames != previous_frames + frames:
        raise ValueError(
            f"fallback publication transaction total frame count is {next_frames}, "
            f"expected {previous_frames + frames}"
        )


def _remove_fallback_transaction(root: Path, transaction: dict[str, Any]) -> None:
    """Durably remove a completed marker, restoring it if unlink fsync fails."""

    journal = root / FALLBACK_PUBLISH_TRANSACTION
    journal.unlink()
    try:
        _fsync_dir(root)
    except OSError as sync_error:
        restore_error: Exception | None = None
        try:
            _atomic_json(journal, transaction)
        except Exception as exc:
            # atomic JSON may have replaced the marker and then failed the same
            # directory fsync. Visibility is enough to keep readers closed.
            restore_error = exc
        if not journal.exists():
            raise RuntimeError(
                f"could not durably remove or restore fallback publication marker {journal}: "
                f"{restore_error or sync_error}"
            ) from sync_error
        log.error(
            "could not durably remove fallback publication marker %s; restored it for "
            "fail-closed recovery",
            journal,
        )
        raise


def _recover_fallback_transaction(root: Path, *, expected_repo_id: str | None = None) -> None:
    """Finish a journalled fallback publication, or leave it visibly failed closed."""

    if root.is_symlink():
        raise ValueError(f"dataset root {root} is a symlink; refusing to recover through it")
    transaction = _fallback_transaction(root, expected_repo_id=expected_repo_id)
    if transaction is None:
        return
    journal = root / FALLBACK_PUBLISH_TRANSACTION
    meta_path = root / "meta.json"
    episode = int(transaction["episode"])
    previous = transaction["previous_metadata"]
    following = transaction["next_metadata"]
    stage = root / str(transaction["stage"])
    destination = root / str(transaction["destination"])
    current = _read_json_object(meta_path, "fallback dataset metadata")

    if current == previous:
        if not stage.exists() or destination.exists():
            raise ValueError(
                f"cannot recover fallback publication {journal}: its previous metadata is "
                "current but its staged episode is ambiguous"
            )
        _validate_transaction_episode(
            stage,
            episode=episode,
            previous_metadata=previous,
            next_metadata=following,
        )
        _atomic_json(meta_path, following)
        current = following
    if current != following:
        raise ValueError(
            f"cannot recover fallback publication {journal}: meta.json matches neither "
            "journalled snapshot"
        )

    if stage.exists() and not destination.exists():
        _validate_transaction_episode(
            stage,
            episode=episode,
            previous_metadata=previous,
            next_metadata=following,
        )
        os.replace(stage, destination)
    elif destination.exists() and not stage.exists():
        _validate_transaction_episode(
            destination,
            episode=episode,
            previous_metadata=previous,
            next_metadata=following,
        )
    else:
        raise ValueError(
            f"cannot recover fallback publication {journal}: staged and published paths "
            "are ambiguous"
        )
    # This may be a second recovery after the rename itself succeeded but its
    # directory fsync failed. Prove both meta.json and the destination entry
    # durable before removing the fail-closed marker.
    _fsync_dir(root)
    _remove_fallback_transaction(root, transaction)


def _rollback_fallback_transaction(root: Path, transaction: dict[str, Any]) -> None:
    """Restore the previous manifest after a caught pre-publication failure."""

    journal = root / FALLBACK_PUBLISH_TRANSACTION
    meta_path = root / "meta.json"
    destination = root / str(transaction["destination"])
    if destination.exists():
        raise RuntimeError(
            f"cannot roll back {journal}: the episode destination already exists; "
            "the recovery marker was retained"
        )
    previous = transaction["previous_metadata"]
    following = transaction["next_metadata"]
    current = _read_json_object(meta_path, "fallback dataset metadata")
    if current == following:
        if previous is None:
            meta_path.unlink()
            _fsync_dir(root)
        else:
            _atomic_json(meta_path, previous)
    elif current != previous:
        raise RuntimeError(
            f"cannot roll back {journal}: meta.json matches neither journalled snapshot; "
            "the recovery marker was retained"
        )
    _remove_fallback_transaction(root, transaction)


def prune_unreferenced_lerobot_files(root: Path) -> list[str]:
    """Delete parquet/mp4 files of a LeRobot v3 dataset that no saved episode references.

    A save that fails after LeRobot wrote the episode's parquet rows (for
    example during video encoding), or a crash in that window, leaves files
    the metadata never references. A footerless orphan parquet makes the whole
    dataset unloadable. Only files under ``data/`` and ``videos/`` are
    considered, and nothing is deleted unless the episode metadata is complete
    and readable (it must list exactly ``total_episodes`` episodes). Returns the
    removed paths relative to ``root``.
    """

    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return []
    info = json.loads(info_path.read_text(encoding="utf-8"))
    data_template = info.get("data_path") if isinstance(info, dict) else None
    if not isinstance(data_template, str) or not data_template:
        return []
    video_template = info.get("video_path")
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    video_keys = [
        key
        for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    ]
    total_episodes = int(info.get("total_episodes", 0))

    referenced: set[Path] = set()
    episode_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    rows = 0
    if episode_files:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        columns = ["data/chunk_index", "data/file_index"]
        for key in video_keys:
            columns += [f"videos/{key}/chunk_index", f"videos/{key}/file_index"]
        for path in episode_files:
            table = pq.read_table(path, columns=columns).to_pydict()
            count = len(table["data/chunk_index"])
            rows += count
            for index in range(count):
                referenced.add(
                    root
                    / data_template.format(
                        chunk_index=table["data/chunk_index"][index],
                        file_index=table["data/file_index"][index],
                    )
                )
                if isinstance(video_template, str):
                    for key in video_keys:
                        referenced.add(
                            root
                            / video_template.format(
                                video_key=key,
                                chunk_index=table[f"videos/{key}/chunk_index"][index],
                                file_index=table[f"videos/{key}/file_index"][index],
                            )
                        )
    if rows != total_episodes:
        log.warning(
            "not pruning %s: metadata lists %d episodes but info.json says %d",
            root,
            rows,
            total_episodes,
        )
        return []
    candidates = sorted((root / "data").rglob("*.parquet")) + sorted(
        (root / "videos").rglob("*.mp4")
    )
    removed: list[str] = []
    for path in candidates:
        if path not in referenced:
            path.unlink()
            removed.append(str(path.relative_to(root)))
    return removed


class EpisodeRecorder:
    """Accumulates frames for one episode and persists them on ``save``."""

    def __init__(self, cfg: RecordingConfig) -> None:
        _exact_int(cfg.fps, "recording FPS", minimum=1)
        if cfg.image_chunk_frames <= 0:
            raise ValueError("image_chunk_frames must be positive")
        self.cfg = cfg
        self._lock = threading.RLock()
        self.settings = get_settings()
        self.root, self._operation_lock = dataset_operation_target(cfg.repo_id)
        if not self._operation_lock.acquire(blocking=False):
            raise DatasetBusyError(dataset_busy_message(cfg.repo_id))
        self._operation_lock_held = True
        try:
            if (self.root / FALLBACK_PUBLISH_TRANSACTION).exists():
                with _dataset_lock(self.root):
                    _recover_fallback_transaction(self.root, expected_repo_id=cfg.repo_id)
            # Under the lock, which every case variant of the id shares.
            refuse_case_alias(cfg.repo_id)
        except Exception:
            self._release_operation_lock()
            raise
        # Kept for compatibility with callers that inspect these fields. The
        # fallback writer no longer grows them with every frame.
        self._frames: list[dict] = []
        self._images: list[dict[str, np.ndarray]] = []
        try:
            self._lerobot_cls = self._find_lerobot_dataset()
        except Exception:
            self._release_operation_lock()
            raise
        self._lerobot_ds = None
        self._backend_initialized = False
        self._backend_failed = False
        self._num_frames = 0
        self._last_saved_frames = 0
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._camera_shapes: dict[str, tuple[int, ...]] = {}
        self._camera_dtypes: dict[str, str] = {}
        self._camera_depth_keys: list[str] = []
        self._stage_dir: Path | None = None
        self._image_buffers: dict[str, list[np.ndarray]] = {
            camera: [] for camera in cfg.camera_keys
        }
        self._image_chunk_indexes: dict[str, int] = {camera: 0 for camera in cfg.camera_keys}
        self._camera_storage_names = {
            camera: f"{index:02d}_{self._safe_component(camera)}"
            for index, camera in enumerate(cfg.camera_keys)
        }
        try:
            default_names = stream_feature_names(cfg.camera_keys)
        except ValueError:  # a depth stream without its color stream
            self._release_operation_lock()
            raise
        self._camera_feature_names = {
            camera: cfg.camera_feature_names.get(camera) or default_names[camera]
            for camera in cfg.camera_keys
        }
        features = list(self._camera_feature_names.values())
        overrides = [cfg.camera_feature_names.get(camera, "") for camera in cfg.camera_keys]
        if len(set(features)) != len(features) or not all(
            re.fullmatch(r"[A-Za-z0-9_-]{0,80}", name) for name in overrides
        ):
            self._release_operation_lock()
            raise ValueError(f"camera feature names must be unique and [A-Za-z0-9_-]: {features}")
        # Backend selection is already known, and these layout checks need no
        # camera frame. Refuse an incompatible root before a fallback writer
        # can even create its per-episode staging directory.
        if self._lerobot_cls is None:
            try:
                self._validate_fallback_target(check_camera_layout=False)
            except Exception:
                self._release_operation_lock()
                raise
        # No camera shape needs to be inferred, so initialize eagerly.
        if not self.cfg.camera_keys:
            try:
                self._initialize_backend({})
            except Exception:
                self._release_operation_lock()
                raise

    # --- backend selection ---------------------------------------------
    def _find_lerobot_dataset(self):
        if self.settings.mock:
            return None
        try:
            from lerobot.datasets import LeRobotDataset  # type: ignore[import-not-found]

            return LeRobotDataset
        except ImportError as exc:
            primary = exc
        try:
            from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[import-not-found]
                LeRobotDataset,
            )

            return LeRobotDataset
        except ImportError:
            pass
        if str(getattr(self.settings, "backend", "") or "").strip().lower() == "sim":
            # Simulated demonstrations train with the numpy baseline, which
            # reads the fallback layout directly.
            log.warning(
                "LeRobot datasets unavailable (%s); simulation records in the fallback format",
                primary,
            )
            return None
        # Fail closed on real hardware: a fallback recording looks successful
        # but real (LeRobot) training refuses it, and no converter exists.
        raise RuntimeError(
            f"LeRobot dataset support is unavailable ({primary}). Real recordings must be "
            "LeRobot datasets so they train directly; install the robot extra "
            "(pip install 'so101[robot]', which pins lerobot[dataset,feetech]==0.6.1) "
            "or set SO101_MOCK=1 for a mock session."
        ) from primary

    def _initialize_backend(self, images: dict[str, np.ndarray]) -> None:
        if self._backend_initialized:
            return
        if self._backend_failed:
            raise RuntimeError("this recorder cannot be reused after backend initialization failed")
        try:
            features = self._features(images)
            if self._lerobot_cls is None:
                # _features populated this take's camera layout without writing;
                # compare it with existing chunks before _write_fallback_frame
                # creates a staging directory.
                self._validate_fallback_target(check_camera_layout=True)
                self._backend_initialized = True
                return

            artifacts = self._partial_fallback_artifacts()
            if artifacts:
                raise RuntimeError(
                    f"{self.root} contains partial fallback recording artifacts "
                    f"({', '.join(path.name for path in artifacts)}); refusing to create or "
                    "resume a LeRobot dataset there"
                )
            root_already_exists = self.root.exists()
            if root_already_exists:
                # A previous create may have made this entry and then failed
                # its parent-directory fsync, with or without reaching complete
                # metadata. Re-prove it before resume or another create attempt.
                _fsync_dir(self.root.parent)
            existing_lerobot = (self.root / "meta" / "info.json").is_file()
            if existing_lerobot:
                resume = getattr(self._lerobot_cls, "resume", None)
                if resume is None:
                    raise RuntimeError(
                        "installed LeRobot cannot resume an existing dataset; "
                        "use the pinned robot extra"
                    )
                with _video_encoder_errors():
                    self._lerobot_ds = resume(
                        repo_id=self.cfg.repo_id,
                        root=self.root,
                        image_writer_threads=4 * len(self.cfg.camera_keys),
                    )
                self._validate_resumed_features(features)
                # A crash, SIGKILL or power loss mid-episode leaves that take's
                # temporary PNGs (and possibly unreferenced parquet/mp4 files)
                # behind; the next episode reuses the index and would encode the
                # dead take's frames into its own video.
                self._cleanup_interrupted_lerobot_episode()
            else:
                root_was_absent = not root_already_exists
                with _video_encoder_errors():
                    try:
                        self._lerobot_ds = self._lerobot_cls.create(
                            repo_id=self.cfg.repo_id,
                            fps=self.cfg.fps,
                            features=features,
                            use_videos=self.cfg.use_videos,
                            root=self.root,
                            robot_type="so_follower",
                            image_writer_threads=4 * len(self.cfg.camera_keys),
                        )
                    except TypeError:
                        # Compatibility with older LeRobot versions that did not accept
                        # robot_type/image-writer configuration yet.
                        self._lerobot_ds = self._lerobot_cls.create(
                            repo_id=self.cfg.repo_id,
                            fps=self.cfg.fps,
                            features=features,
                            use_videos=self.cfg.use_videos,
                            root=self.root,
                        )
                if root_was_absent and self.root.exists():
                    if self.root.is_symlink() or not self.root.is_dir():
                        raise RuntimeError(
                            f"LeRobot created a non-directory dataset root: {self.root}"
                        )
                    # LeRobot owns its internal files, but the suite owns the
                    # parent directory and must make the new root entry survive
                    # power loss before this recorder can report success.
                    _fsync_dir(self.root.parent)
            # Dataset creation failures can leave a partial on-disk tree. Do not
            # silently switch formats in that directory.
            self._backend_initialized = True
        except BaseException:
            self._backend_failed = True
            self._close_failed_backend()
            self._release_operation_lock()
            raise

    def _partial_fallback_artifacts(self) -> list[Path]:
        artifacts: list[Path] = []
        metadata = self.root / "meta.json"
        if metadata.exists():
            artifacts.append(metadata)
        for pattern in ("episode_*", "incomplete_*", ".episode-staging-*"):
            artifacts.extend(sorted(self.root.glob(pattern)))
        return artifacts

    def _close_failed_backend(self) -> None:
        dataset, self._lerobot_ds = self._lerobot_ds, None
        if dataset is None:
            return
        for method_name in ("finalize", "close"):
            method = getattr(dataset, method_name, None)
            if method is None:
                continue
            try:
                method()
            except Exception as exc:
                log.warning("could not %s failed LeRobot dataset: %s", method_name, exc)
            break

    def _read_fallback_metadata(self) -> dict[str, Any] | None:
        """Read an existing fallback manifest, refusing any LeRobot tree.

        A mock session or a simulation without the optional LeRobot package
        must not write ``meta.json`` and ``episode_*`` beside LeRobot's
        ``meta/info.json``. A ``meta`` directory also counts as a partial
        LeRobot tree: after an interrupted LeRobot create, switching formats
        would make recovery ambiguous.
        """

        lerobot_artifacts = [
            path
            for path in (self.root / "meta", self.root / "data", self.root / "videos")
            if path.exists()
        ]
        if lerobot_artifacts:
            raise RuntimeError(
                f"{self.root} is already a LeRobot dataset (or a partial LeRobot create); "
                "refusing to mix fallback files into it "
                f"({', '.join(p.name for p in lerobot_artifacts)})"
            )
        path = self.root / "meta.json"
        if not path.exists():
            artifacts = self._fallback_data_artifacts()
            if artifacts:
                if all(item.name.startswith("incomplete_") for item in artifacts):
                    self._validate_incomplete_only(artifacts)
                else:
                    raise ValueError(
                        f"{self.root} has orphan published or staging fallback artifacts but no "
                        "meta.json; repair or remove them before recording"
                    )
            return None
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot read existing fallback dataset metadata {path}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"existing fallback dataset metadata {path} must be a JSON object")
        return metadata

    def _validate_incomplete_only(self, directories: list[Path]) -> None:
        """Allow retry beside complete, compatible aborted-take diagnostics."""

        for directory in directories:
            manifest_path = directory / "INCOMPLETE.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"incomplete fallback capture {directory} has no readable INCOMPLETE.json"
                ) from exc
            if not isinstance(manifest, dict):
                raise ValueError(f"incomplete fallback capture {directory} metadata is invalid")
            if manifest.get("format") != "so101-fallback-v2":
                raise ValueError(
                    f"incomplete fallback capture {directory} has unsupported format "
                    f"{manifest.get('format')!r}"
                )
            incomplete_fps = _exact_int(manifest.get("fps"), "incomplete recording FPS", minimum=1)
            incomplete_frames = _exact_int(
                manifest.get("frames"), "incomplete recording frame count", minimum=1
            )
            expected = {
                "repo_id": self.cfg.repo_id,
                "task": self.cfg.task,
                "camera_keys": self.cfg.camera_keys,
                "camera_storage": self._camera_storage_names,
            }
            for key, value in expected.items():
                if manifest.get(key) != value:
                    raise ValueError(
                        f"incomplete fallback capture {directory} {key} is "
                        f"{manifest.get(key)!r}, requested {value!r}"
                    )
            if incomplete_fps != self.cfg.fps:
                raise ValueError(
                    f"incomplete fallback capture {directory} FPS is {incomplete_fps}, "
                    f"requested {self.cfg.fps}"
                )
            frames_path = directory / "frames.jsonl"
            try:
                with frames_path.open(encoding="utf-8") as handle:
                    actual_frames = sum(1 for line in handle if line.strip())
            except OSError as exc:
                raise ValueError(
                    f"incomplete fallback capture {directory} has no readable frames.jsonl"
                ) from exc
            if actual_frames != incomplete_frames:
                raise ValueError(
                    f"incomplete fallback capture {directory} stores {incomplete_frames} "
                    f"frames, but frames.jsonl has {actual_frames}"
                )

    def _fallback_data_artifacts(self) -> list[Path]:
        artifacts: list[Path] = []
        for pattern in ("episode_*", "incomplete_*", ".episode-staging-*"):
            for path in sorted(self.root.glob(pattern)):
                if self._stage_dir is not None and path == self._stage_dir:
                    continue
                artifacts.append(path)
        return artifacts

    def _published_fallback_frame_count(self, episodes: int) -> int:
        directories = sorted(self.root.glob("episode_*"))
        indexes: list[int] = []
        total = 0
        for directory in directories:
            match = re.fullmatch(r"episode_(\d{6})", directory.name)
            if not directory.is_dir() or match is None:
                raise ValueError(
                    f"existing fallback dataset has an invalid episode path: {directory.name}"
                )
            validate_fallback_episode_tree(directory)
            index = int(match.group(1))
            indexes.append(index)
            manifest_path = directory / "episode.json"
            frames_path = directory / "frames.jsonl"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"existing fallback episode {directory.name} has unreadable metadata"
                ) from exc
            if not isinstance(manifest, dict):
                raise ValueError(
                    f"existing fallback episode {directory.name} metadata must be an object"
                )
            stored_index = _exact_int(
                manifest.get("episode"), f"{directory.name} episode index", minimum=0
            )
            stored_frames = _exact_int(
                manifest.get("frames"), f"{directory.name} frame count", minimum=1
            )
            if stored_index != index:
                raise ValueError(
                    f"existing fallback episode {directory.name} stores index {stored_index}"
                )
            try:
                with frames_path.open(encoding="utf-8") as handle:
                    actual_frames = sum(1 for line in handle if line.strip())
            except OSError as exc:
                raise ValueError(
                    f"existing fallback episode {directory.name} has no readable frames.jsonl"
                ) from exc
            if actual_frames != stored_frames:
                raise ValueError(
                    f"existing fallback episode {directory.name} frame count is "
                    f"{stored_frames}, but frames.jsonl has {actual_frames}"
                )
            total += actual_frames
        expected_indexes = list(range(episodes))
        if indexes != expected_indexes:
            raise ValueError(
                f"existing fallback episode indexes are {indexes}, expected {expected_indexes}"
            )
        return total

    @staticmethod
    def _metadata_string_list(metadata: dict[str, Any], key: str) -> list[str]:
        raw = metadata.get(key, [])
        if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
            raise ValueError(f"existing fallback dataset {key!r} must be a list of strings")
        return list(raw)

    @staticmethod
    def _metadata_map(
        metadata: dict[str, Any], key: str, cameras: list[str]
    ) -> dict[str, Any] | None:
        raw = metadata.get(key)
        if raw is None:
            return None
        if not isinstance(raw, dict) or set(raw) != set(cameras):
            raise ValueError(
                f"existing fallback dataset {key!r} must map exactly these cameras: {cameras}"
            )
        return raw

    def _recorded_camera_layout(
        self, metadata: dict[str, Any], *, populated: bool
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, str], list[str]]:
        """Camera layout proven by all published chunks and any stored schema."""

        stored_shapes = self._metadata_map(metadata, "camera_shapes", self.cfg.camera_keys)
        stored_dtypes = self._metadata_map(metadata, "camera_dtypes", self.cfg.camera_keys)
        stored_depth = metadata.get("camera_depth_keys")
        if stored_depth is not None:
            if (
                not isinstance(stored_depth, list)
                or not all(isinstance(key, str) for key in stored_depth)
                or len(stored_depth) != len(set(stored_depth))
                or any(key not in self.cfg.camera_keys for key in stored_depth)
            ):
                raise ValueError("existing fallback dataset camera depth keys are invalid")

        chunk_shapes: dict[str, tuple[int, ...]] = {}
        chunk_dtypes: dict[str, str] = {}
        chunk_depth: list[str] = []
        episode_dirs = sorted(self.root.glob("episode_*"))
        for episode_dir in episode_dirs:
            validate_fallback_episode_tree(episode_dir)
        for camera in self.cfg.camera_keys:
            storage = self._camera_storage_names[camera]
            chunks: list[Path] = []
            for episode_dir in episode_dirs:
                episode_chunks = sorted((episode_dir / "images" / storage).glob("chunk_*.npy"))
                if populated and not episode_chunks:
                    raise ValueError(
                        f"existing fallback dataset camera chunks are missing for {camera!r} "
                        f"in {episode_dir.name}"
                    )
                chunks.extend(episode_chunks)
            for chunk in chunks:
                try:
                    array = np.load(chunk, mmap_mode="r", allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        f"existing fallback dataset camera chunks are unreadable: {chunk}: {exc}"
                    ) from exc
                if array.ndim != 4:
                    raise ValueError(
                        f"existing fallback dataset camera chunks have invalid shape: "
                        f"{chunk}: {array.shape}"
                    )
                shape = tuple(int(value) for value in array.shape[1:])
                dtype = np.dtype(array.dtype)
                if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
                    raise ValueError(
                        f"existing fallback dataset camera chunks use unsupported dtype "
                        f"{dtype.name}: {chunk}"
                    )
                dtype_name = dtype.name
                previous = (chunk_shapes.get(camera), chunk_dtypes.get(camera))
                if previous[0] is not None and previous != (shape, dtype_name):
                    raise ValueError(
                        f"existing fallback dataset camera chunks conflict for {camera!r}: "
                        f"{previous} != {(shape, dtype_name)}"
                    )
                chunk_shapes[camera], chunk_dtypes[camera] = shape, dtype_name
            if chunks:
                is_depth = chunk_shapes[camera][-1:] == (1,) and chunk_dtypes[camera] == "uint16"
                if is_depth:
                    chunk_depth.append(camera)
                if is_depth != is_depth_key(camera):
                    raise ValueError(
                        f"existing fallback dataset camera chunks disagree with depth key "
                        f"{camera!r}"
                    )
            elif populated:
                raise ValueError(
                    f"existing fallback dataset has no camera chunks for {camera!r}; its "
                    "layout cannot be verified"
                )

        shapes: dict[str, tuple[int, ...]] = {}
        dtypes: dict[str, str] = {}
        for camera in self.cfg.camera_keys:
            if stored_shapes is not None:
                raw_shape = stored_shapes[camera]
                if (
                    not isinstance(raw_shape, list)
                    or len(raw_shape) != 3
                    or any(type(value) is not int or value <= 0 for value in raw_shape)
                ):
                    raise ValueError(
                        f"existing fallback dataset camera shape for {camera!r} is invalid: "
                        f"{raw_shape!r}"
                    )
                shapes[camera] = tuple(raw_shape)
            elif camera in chunk_shapes:
                shapes[camera] = chunk_shapes[camera]
            if stored_dtypes is not None:
                raw_dtype = stored_dtypes[camera]
                if not isinstance(raw_dtype, str) or raw_dtype not in {"uint8", "uint16"}:
                    raise ValueError(
                        f"existing fallback dataset camera dtype for {camera!r} is invalid: "
                        f"{raw_dtype!r}"
                    )
                dtypes[camera] = raw_dtype
            elif camera in chunk_dtypes:
                dtypes[camera] = chunk_dtypes[camera]
            if camera in chunk_shapes and shapes.get(camera) != chunk_shapes[camera]:
                raise ValueError(
                    f"existing fallback dataset camera chunks conflict with stored shape for "
                    f"{camera!r}"
                )
            if camera in chunk_dtypes and dtypes.get(camera) != chunk_dtypes[camera]:
                raise ValueError(
                    f"existing fallback dataset camera chunks conflict with stored dtype for "
                    f"{camera!r}"
                )
        depth = list(stored_depth) if stored_depth is not None else chunk_depth
        if chunk_shapes and depth != chunk_depth:
            raise ValueError(
                "existing fallback dataset camera chunks conflict with stored depth keys"
            )
        return shapes, dtypes, depth

    def _validate_fallback_target(self, *, check_camera_layout: bool) -> None:
        """Require one self-consistent schema before appending fallback data.

        Fallback discovery exposes one repo id, rate and task, while its frame
        reader assumes one environment and camera layout. Appending a take
        with different values would make the new frames be interpreted using
        the old metadata, so every invariant is exact (including camera order,
        which determines the on-disk storage names).
        """

        metadata = self._read_fallback_metadata()
        if metadata is None:
            return
        format_name = metadata.get("format")
        if format_name not in _FALLBACK_FORMATS:
            raise ValueError(
                f"unsupported fallback dataset format {format_name!r}; use a converter or a "
                "new repo id"
            )
        if metadata.get("repo_id") != self.cfg.repo_id:
            raise ValueError(
                f"existing fallback dataset repo_id is {metadata.get('repo_id')!r}, "
                f"requested {self.cfg.repo_id!r}"
            )
        episodes = _exact_int(metadata.get("episodes", 0), "episode count", minimum=0)
        total_frames = _exact_int(metadata.get("total_frames", 0), "total frame count", minimum=0)
        artifacts = self._fallback_data_artifacts()
        incomplete = [path for path in artifacts if not path.name.startswith("episode_")]
        if episodes == 0 and (total_frames != 0 or artifacts):
            raise ValueError(
                "existing fallback dataset claims zero episodes but has frames or artifacts"
            )
        if episodes > 0:
            published_frames = self._published_fallback_frame_count(episodes)
            if "total_frames" in metadata and total_frames != published_frames:
                raise ValueError(
                    f"existing fallback dataset total frame count is {total_frames}, but "
                    f"published episodes contain {published_frames}"
                )
        if incomplete and episodes > 0:
            staging = [path for path in incomplete if path.name.startswith(".episode-staging-")]
            if staging:
                raise ValueError(
                    "existing fallback dataset has an unfinished staging directory; recover or "
                    "remove it before recording"
                )
        empty = episodes == 0

        def require_match(key: str, expected: Any, label: str) -> None:
            if key not in metadata:
                if not empty:
                    raise ValueError(f"existing fallback dataset is missing {label}")
                return
            if metadata[key] != expected:
                raise ValueError(
                    f"existing fallback dataset {label} is {metadata[key]!r}, "
                    f"requested {expected!r}"
                )

        if "fps" in metadata:
            recorded_fps = _exact_int(metadata["fps"], "FPS", minimum=1)
            if recorded_fps != self.cfg.fps:
                raise ValueError(
                    f"existing fallback dataset FPS is {recorded_fps}, requested {self.cfg.fps}"
                )
        elif not empty:
            raise ValueError("existing fallback dataset is missing FPS")
        require_match("task", self.cfg.task, "task")

        recorded_cameras = list(self.cfg.camera_keys)
        if "camera_keys" in metadata:
            recorded_cameras = self._metadata_string_list(metadata, "camera_keys")
            if recorded_cameras != self.cfg.camera_keys:
                raise ValueError(
                    f"existing fallback dataset camera keys are {recorded_cameras}, "
                    f"requested {self.cfg.camera_keys}"
                )
        elif not empty:
            raise ValueError("existing fallback dataset is missing camera keys")
        if "image_storage" in metadata and metadata["image_storage"] != "chunked-npy":
            raise ValueError(
                f"existing fallback dataset image storage is {metadata['image_storage']!r}, "
                "expected 'chunked-npy'"
            )
        recorded_storage = metadata.get("camera_storage")
        if recorded_storage is not None and (
            not isinstance(recorded_storage, dict) or recorded_storage != self._camera_storage_names
        ):
            raise ValueError(
                f"existing fallback dataset camera storage is {recorded_storage!r}, "
                f"requested {self._camera_storage_names!r}"
            )

        recorded_features = metadata.get("camera_feature_names")
        if recorded_features is None:
            try:
                recorded_features = stream_feature_names(recorded_cameras)
            except ValueError as exc:
                raise ValueError(
                    f"existing fallback dataset camera feature names are invalid: {exc}"
                ) from exc
        if (
            not isinstance(recorded_features, dict)
            or not all(
                isinstance(camera, str) and isinstance(name, str)
                for camera, name in recorded_features.items()
            )
            or recorded_features != self._camera_feature_names
        ):
            raise ValueError(
                f"existing fallback dataset camera feature names are {recorded_features!r}, "
                f"requested {self._camera_feature_names!r}"
            )

        canonical_environment = metadata.get("environment_state_names")
        legacy_environment = metadata.get("env_state_names")
        if canonical_environment is not None and legacy_environment is not None:
            if canonical_environment != legacy_environment:
                raise ValueError(
                    "existing fallback dataset environment-state name aliases conflict"
                )
        environment_key = (
            "environment_state_names" if canonical_environment is not None else "env_state_names"
        )
        recorded_environment = self._metadata_string_list(metadata, environment_key)
        if recorded_environment != self.cfg.environment_state_names:
            raise ValueError(
                f"existing fallback dataset environment state names are {recorded_environment}, "
                f"requested {self.cfg.environment_state_names}"
            )
        if not check_camera_layout or not self.cfg.camera_keys:
            return
        shapes, dtypes, depth = self._recorded_camera_layout(metadata, populated=not empty)
        for camera, expected in self._camera_shapes.items():
            actual_shape = shapes.get(camera)
            actual_dtype = dtypes.get(camera)
            if actual_shape is None and not empty:
                raise ValueError(
                    f"existing fallback dataset camera {camera!r} shape cannot be verified"
                )
            if actual_shape is not None and actual_shape != expected:
                raise ValueError(
                    f"existing fallback dataset camera {camera!r} shape is {actual_shape}, "
                    f"requested {expected}"
                )
            if actual_dtype is None and not empty:
                raise ValueError(
                    f"existing fallback dataset camera {camera!r} dtype cannot be verified"
                )
            if actual_dtype is not None and actual_dtype != self._camera_dtypes[camera]:
                raise ValueError(
                    f"existing fallback dataset camera {camera!r} dtype is {actual_dtype}, "
                    f"requested {self._camera_dtypes[camera]}"
                )
        if ("camera_depth_keys" in metadata or not empty) and depth != self._camera_depth_keys:
            raise ValueError(
                f"existing fallback dataset camera depth keys are {depth}, requested "
                f"{self._camera_depth_keys}"
            )

    def _validate_resumed_features(self, expected: dict[str, dict]) -> None:
        assert self._lerobot_ds is not None
        actual_fps = _exact_int(self._lerobot_ds.fps, "dataset FPS", minimum=1)
        if actual_fps != self.cfg.fps:
            raise ValueError(f"dataset FPS is {actual_fps}, requested {self.cfg.fps}")
        actual = self._lerobot_ds.features
        if not isinstance(actual, dict):
            raise ValueError("existing dataset features must be an object")
        controlled_actual = set(actual) - _LEROBOT_GENERATED_FEATURES
        missing = sorted(set(expected) - controlled_actual)
        if missing:
            raise ValueError(f"existing dataset is missing required features: {missing}")
        unexpected = sorted(controlled_actual - set(expected))
        if unexpected:
            raise ValueError(f"existing dataset has unexpected features: {unexpected}")
        for key, specification in expected.items():
            actual_specification = actual[key]
            if not isinstance(actual_specification, dict):
                raise ValueError(f"existing dataset feature {key!r} must be an object")
            unknown = set(actual_specification) - {"dtype", "shape", "names", "info"}
            if unknown:
                raise ValueError(
                    f"existing dataset feature {key!r} has unknown metadata: {sorted(unknown)}"
                )
            actual_dtype = actual_specification.get("dtype")
            expected_dtype = specification.get("dtype")
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"existing dataset feature {key!r} dtype differs: "
                    f"{actual_dtype!r} != {expected_dtype!r}"
                )
            actual_shape_raw = actual_specification.get("shape")
            expected_shape_raw = specification.get("shape")
            if not isinstance(actual_shape_raw, (list, tuple)) or any(
                type(value) is not int or value <= 0 for value in actual_shape_raw
            ):
                raise ValueError(
                    f"existing dataset feature {key!r} shape is invalid: {actual_shape_raw!r}"
                )
            actual_shape = tuple(actual_shape_raw)
            expected_shape = tuple(expected_shape_raw or ())
            if actual_shape != expected_shape:
                raise ValueError(
                    f"existing dataset feature {key!r} shape differs: "
                    f"{actual_shape} != {expected_shape}"
                )
            actual_names_raw = actual_specification.get("names")
            expected_names_raw = specification.get("names")
            if not isinstance(actual_names_raw, (list, tuple)):
                raise ValueError(
                    f"existing dataset feature {key!r} names are invalid: {actual_names_raw!r}"
                )
            actual_names = list(actual_names_raw)
            expected_names = list(expected_names_raw or [])
            if actual_names != expected_names:
                raise ValueError(
                    f"existing dataset feature {key!r} names differ: "
                    f"{actual_names} != {expected_names}"
                )
            actual_info = actual_specification.get("info") or {}
            expected_info = specification.get("info") or {}
            if not isinstance(actual_info, dict):
                raise ValueError(
                    f"existing dataset feature {key!r} info is invalid: {actual_info!r}"
                )
            unknown_info = set(actual_info) - (_LEROBOT_GENERATED_VIDEO_INFO | {"is_depth_map"})
            if unknown_info:
                raise ValueError(
                    f"existing dataset feature {key!r} has unknown metadata in info: "
                    f"{sorted(unknown_info)}"
                )
            depth_flags = [
                bool(actual_info[name])
                for name in ("is_depth_map", "video.is_depth_map")
                if name in actual_info
            ]
            if len(set(depth_flags)) > 1:
                raise ValueError(
                    f"existing dataset feature {key!r} has conflicting depth-map metadata"
                )
            actual_depth = depth_flags[0] if depth_flags else False
            expected_depth = bool(expected_info.get("is_depth_map"))
            if actual_depth != expected_depth:
                raise ValueError(
                    f"existing dataset feature {key!r} depth-map meaning differs: "
                    f"{actual_depth} != {expected_depth}"
                )

    def _features(self, images: dict[str, np.ndarray]) -> dict:
        names = [f"{j}.pos" for j in JOINT_NAMES]
        feats = {
            "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
            "action": {"dtype": "float32", "shape": (len(names),), "names": names},
            "desired_action": {
                "dtype": "float32",
                "shape": (len(names),),
                "names": names,
            },
        }
        if self.cfg.environment_state_names:
            env_names = list(self.cfg.environment_state_names)
            feats["observation.environment_state"] = {
                "dtype": "float32",
                "shape": (len(env_names),),
                "names": env_names,
            }
        for cam in self.cfg.camera_keys:
            if cam not in images:
                raise ValueError(f"configured camera {cam!r} did not provide a frame")
            image = np.asarray(images[cam])
            shape = tuple(image.shape)
            dtype = np.dtype(image.dtype)
            depth = is_depth_key(cam)
            expected_channels = 1 if depth else 3
            expected_dtype = np.dtype(np.uint16 if depth else np.uint8)
            if len(shape) != 3 or shape[2] != expected_channels:
                kind = "depth" if depth else "color"
                raise ValueError(
                    f"{kind} camera {cam!r} frame must be HxWx{expected_channels}, "
                    f"got shape {shape}"
                )
            if dtype != expected_dtype:
                raise ValueError(
                    f"camera {cam!r} frame must use {expected_dtype.name}, got {dtype.name}"
                )
            previous_shape = self._camera_shapes.get(cam)
            if previous_shape is not None and shape != previous_shape:
                raise ValueError(
                    f"camera {cam!r} shape changed: expected {previous_shape}, got {shape}"
                )
            self._camera_shapes[cam] = shape
            self._camera_dtypes[cam] = dtype.name
            if depth and cam not in self._camera_depth_keys:
                self._camera_depth_keys.append(cam)
            feature_name = self._camera_feature_names[cam]
            feature: dict[str, Any] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
            feats[f"observation.images.{feature_name}"] = feature
            if depth:
                # A depth map in millimetres, kept uint16: LeRobot 0.6.1 stores it
                # as 16-bit TIFF (images) or 12-bit lossless HEVC (videos) and
                # infers depth_unit "mm" from the dtype (docs/realsense.md).
                feature["info"] = {"is_depth_map": True}
        return feats

    @staticmethod
    def _safe_component(value: str) -> str:
        component = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
        return component[:80] or "camera"

    def _ensure_stage(self) -> Path:
        if self._stage_dir is not None:
            return self._stage_dir
        if self.root.is_symlink():
            raise ValueError(f"dataset root {self.root} is a symlink; refusing to follow it")
        durable_makedirs(self.root)
        if self.root.is_symlink():
            raise ValueError(f"dataset root {self.root} became a symlink; refusing to follow it")
        self._stage_dir = Path(tempfile.mkdtemp(prefix=".episode-staging-", dir=self.root))
        (self._stage_dir / "frames.jsonl").touch()
        for storage_name in self._camera_storage_names.values():
            (self._stage_dir / "images" / storage_name).mkdir(parents=True, exist_ok=True)
        return self._stage_dir

    def _flush_image_chunk(self, camera: str) -> None:
        frames = self._image_buffers[camera]
        if not frames:
            return
        stage = self._ensure_stage()
        chunk_index = self._image_chunk_indexes[camera]
        storage_name = self._camera_storage_names[camera]
        destination = stage / "images" / storage_name / f"chunk_{chunk_index:06d}.npy"
        np.save(destination, np.stack(frames))
        self._image_chunk_indexes[camera] = chunk_index + 1
        frames.clear()

    def _write_fallback_frame(
        self,
        frame: dict[str, Any],
        images: dict[str, np.ndarray],
    ) -> None:
        stage = self._ensure_stage()
        with (stage / "frames.jsonl").open("a") as handle:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")
        for camera, image in images.items():
            self._image_buffers[camera].append(np.asarray(image).copy())
            if len(self._image_buffers[camera]) >= self.cfg.image_chunk_frames:
                self._flush_image_chunk(camera)

    @staticmethod
    def _ordered_vector(values: dict[str, float], *, label: str) -> np.ndarray:
        names = [f"{joint}.pos" for joint in JOINT_NAMES]
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"{label} is missing canonical SO-101 joints: {missing}")
        vector = np.asarray([values[name] for name in names], dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(f"{label} contains a non-finite joint value")
        return vector

    # --- recording ------------------------------------------------------
    def add(
        self,
        step: TeleopStep,
        images: dict[str, np.ndarray] | None = None,
        environment_state: Any = None,
    ) -> None:
        with self._lock:
            self._add_locked(step, images, environment_state)

    def _add_locked(
        self,
        step: TeleopStep,
        images: dict[str, np.ndarray] | None = None,
        environment_state: Any = None,
    ) -> None:
        images = images or {}
        env_state = self._environment_vector(environment_state)
        self._initialize_backend(images)

        state = self._ordered_vector(step.observation, label="observation")
        # ``action`` is the safety-clamped command actually sent. Older
        # TeleopStep producers may not yet supply desired_action; treating the
        # actual command as their intent preserves compatibility without
        # inventing values.
        action = self._ordered_vector(step.action, label="action")
        desired_values = step.desired_action or step.action
        desired_action = self._ordered_vector(desired_values, label="desired_action")

        configured_images: dict[str, np.ndarray] = {}
        for cam in self.cfg.camera_keys:
            if cam not in images:
                raise ValueError(f"configured camera {cam!r} did not provide a frame")
            frame = np.asarray(images[cam])
            expected_shape = self._camera_shapes[cam]
            if tuple(frame.shape) != expected_shape:
                raise ValueError(
                    f"camera {cam!r} shape changed: expected {expected_shape}, got {frame.shape}"
                )
            expected_dtype = self._camera_dtypes[cam]
            if np.dtype(frame.dtype).name != expected_dtype:
                raise ValueError(
                    f"camera {cam!r} dtype changed: expected {expected_dtype}, "
                    f"got {np.dtype(frame.dtype).name}"
                )
            configured_images[cam] = frame

        timestamp = float(step.t)
        if not np.isfinite(timestamp):
            raise ValueError("frame timestamp must be finite")
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        self._last_timestamp = timestamp

        if self._lerobot_ds is not None:
            frame = {
                "observation.state": state,
                "action": action,
                "desired_action": desired_action,
                "task": self.cfg.task,
            }
            for cam, img in configured_images.items():
                feature_name = self._camera_feature_names[cam]
                frame[f"observation.images.{feature_name}"] = img
            if env_state is not None:
                frame["observation.environment_state"] = env_state
            self._lerobot_ds.add_frame(frame)
        else:
            row: dict[str, Any] = {
                "t": step.t,
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "desired_action": desired_action.tolist(),
            }
            if env_state is not None:
                row["observation.environment_state"] = env_state.tolist()
            self._write_fallback_frame(row, configured_images)
        self._num_frames += 1

    def _environment_vector(self, values: Any) -> np.ndarray | None:
        names = self.cfg.environment_state_names
        if not names:
            if values is not None:
                raise ValueError("environment_state given but environment_state_names is empty")
            return None
        if values is None:
            raise ValueError("this recording needs an environment_state with every frame")
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.shape != (len(names),):
            raise ValueError(
                f"environment_state has shape {vector.shape}, expected ({len(names)},)"
            )
        if not np.isfinite(vector).all():
            raise ValueError("environment_state contains a non-finite value")
        return vector

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def last_saved_frames(self) -> int:
        return self._last_saved_frames

    def save_episode(self, *, retain_operation_lock: bool = False) -> int:
        """Persist one episode, optionally retaining its dataset transaction lock."""
        with self._lock:
            frame_count = self._num_frames
            episode = self._save_episode_locked()
            self._last_saved_frames = frame_count
            if not retain_operation_lock:
                self._release_operation_lock()
            return episode

    def release_operation_lock(self) -> None:
        """End a retained save transaction after suite-owned metadata is published."""

        self._release_operation_lock()

    def _save_episode_locked(self) -> int:
        if self._num_frames == 0:
            raise ValueError("cannot save an empty episode")
        if self._lerobot_ds is not None:
            self._lerobot_ds.save_episode()
            ep = self._lerobot_ds.num_episodes - 1
            finalize = getattr(self._lerobot_ds, "finalize", None)
            if finalize is not None:
                finalize()
            self._num_frames = 0
            self._first_timestamp = None
            self._last_timestamp = None
            log.info("saved LeRobot episode %d (%s)", ep, self.cfg.repo_id)
            return ep
        return self._save_fallback_episode()

    def _save_fallback_episode(self) -> int:
        stage = self._ensure_stage()
        for camera in self.cfg.camera_keys:
            self._flush_image_chunk(camera)
        frames_path = stage / "frames.jsonl"
        with frames_path.open("a") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        meta_path = self.root / "meta.json"
        with _dataset_lock(self.root):
            # Re-check immediately before publication. The operation lock
            # protects cooperating writers; this also fails closed if an
            # external tool changed the manifest during capture.
            self._validate_fallback_target(check_camera_layout=True)
            if meta_path.exists():
                meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
                previous_meta: dict[str, Any] | None = dict(meta)
            else:
                meta = {
                    "repo_id": self.cfg.repo_id,
                    "fps": self.cfg.fps,
                    "task": self.cfg.task,
                    "camera_keys": self.cfg.camera_keys,
                    "episodes": 0,
                }
                previous_meta = None
            ep = _exact_int(meta.get("episodes", 0), "episode count", minimum=0)
            previous_frames = self._published_fallback_frame_count(ep)
            ep_dir = self.root / f"episode_{ep:06d}"
            if ep_dir.exists():  # validation above requires contiguous 0..episodes-1
                raise ValueError(f"next fallback episode path already exists: {ep_dir}")
            _atomic_json(
                stage / "episode.json",
                {
                    "schema_version": 1,
                    "episode": ep,
                    "frames": self._num_frames,
                    "started_at": self._first_timestamp,
                    "ended_at": self._last_timestamp,
                    "duration_seconds": max(
                        0.0,
                        (self._last_timestamp or 0.0) - (self._first_timestamp or 0.0),
                    ),
                    "task": self.cfg.task,
                    "outcome": "unlabeled",
                    "notes": "",
                    "camera_keys": self.cfg.camera_keys,
                    "created_at": time.time(),
                },
            )
            _sync_stage(stage)
            meta.update(
                repo_id=self.cfg.repo_id,
                format="so101-fallback-v2",
                fps=self.cfg.fps,
                task=self.cfg.task,
                camera_keys=list(self.cfg.camera_keys),
                image_storage="chunked-npy",
                camera_storage=dict(self._camera_storage_names),
                episodes=ep + 1,
                total_frames=previous_frames + self._num_frames,
                updated_at=time.time(),
            )
            if self.cfg.environment_state_names:
                meta["environment_state_names"] = list(self.cfg.environment_state_names)
            else:
                meta.pop("environment_state_names", None)
            meta.pop("env_state_names", None)
            if self.cfg.camera_feature_names:
                meta["camera_feature_names"] = dict(self._camera_feature_names)
            if self.cfg.camera_keys:
                meta["camera_shapes"] = {
                    camera: list(self._camera_shapes[camera]) for camera in self.cfg.camera_keys
                }
                meta["camera_dtypes"] = dict(self._camera_dtypes)
                meta["camera_depth_keys"] = list(self._camera_depth_keys)
            else:
                meta.pop("camera_shapes", None)
                meta.pop("camera_dtypes", None)
                meta.pop("camera_depth_keys", None)
            transaction: dict[str, Any] = {
                "schema_version": 1,
                "episode": ep,
                "stage": stage.name,
                "destination": ep_dir.name,
                "previous_metadata": previous_meta,
                "next_metadata": meta,
            }
            journal = self.root / FALLBACK_PUBLISH_TRANSACTION
            if journal.exists():
                raise RuntimeError(
                    f"fallback publication transaction already exists: {journal}; "
                    "refusing to overwrite recovery evidence"
                )
            _atomic_json(journal, transaction)
            try:
                # While the journal exists readers fail closed. Publish the
                # manifest first, then atomically reveal its already-durable
                # episode directory; recovery can finish either crash point.
                _atomic_json(meta_path, meta)
                os.replace(stage, ep_dir)
            except Exception as publish_error:
                try:
                    _rollback_fallback_transaction(self.root, transaction)
                except Exception as rollback_error:
                    raise RuntimeError(
                        "fallback episode publication failed and could not be rolled back; "
                        f"{journal} was retained for fail-closed recovery: {rollback_error}"
                    ) from publish_error
                raise
            _fsync_dir(self.root)
            self._stage_dir = None
            _remove_fallback_transaction(self.root, transaction)

        log.info("saved fallback episode %d -> %s", ep, ep_dir)
        self._frames.clear()
        self._images.clear()
        self._image_buffers = {camera: [] for camera in self.cfg.camera_keys}
        self._image_chunk_indexes = {camera: 0 for camera in self.cfg.camera_keys}
        self._first_timestamp = None
        self._last_timestamp = None
        self._num_frames = 0
        return ep

    def abort(self, reason: str = "aborted") -> Path | None:
        """Finalize an incomplete fallback episode without publishing it as data."""

        with self._lock:
            try:
                return self._abort_locked(reason)
            finally:
                self._release_operation_lock()

    def _release_operation_lock(self) -> None:
        if getattr(self, "_operation_lock_held", False):
            self._operation_lock_held = False
            self._operation_lock.release()

    def __del__(self) -> None:
        try:
            self._release_operation_lock()
        except Exception:
            pass

    def _cleanup_interrupted_lerobot_episode(self) -> None:
        """Remove what an unsaved LeRobot episode left behind (best effort)."""

        dataset = self._lerobot_ds
        if dataset is None:
            return
        writer = getattr(dataset, "writer", None)
        cleanup = getattr(writer, "cleanup_interrupted_episode", None)
        if cleanup is not None:
            try:
                cleanup(int(dataset.num_episodes))
            except Exception as exc:
                log.warning("could not remove interrupted-episode images: %s", exc)
        try:
            removed = prune_unreferenced_lerobot_files(self.root)
        except Exception as exc:
            log.warning("could not check %s for unreferenced episode files: %s", self.root, exc)
            return
        if removed:
            log.warning("removed files of an unsaved episode from %s: %s", self.root, removed)

    def _abort_locked(self, reason: str) -> Path | None:
        if self._lerobot_ds is not None:
            dataset = self._lerobot_ds
            try:
                try:
                    pending = bool(getattr(dataset, "has_pending_frames", lambda: False)())
                except KeyError:
                    # LeRobot 0.6.1's save_episode() pops "size"/"task" from the
                    # buffer before writing parquet and video, so a save that
                    # failed part-way leaves a buffer has_pending_frames() cannot
                    # read. That buffer is pending by definition.
                    pending = True
                if pending:
                    try:
                        dataset.clear_episode_buffer(delete_images=True)
                    except Exception as exc:
                        log.warning("could not clear the LeRobot episode buffer: %s", exc)
            finally:
                # Mirror LeRobot's VideoEncodingManager: always finalize (parquet
                # footers, metadata), then drop the interrupted take's files.
                try:
                    finalize = getattr(dataset, "finalize", None)
                    if finalize is not None:
                        finalize()
                finally:
                    self._cleanup_interrupted_lerobot_episode()
                    self._num_frames = 0
                    self._first_timestamp = None
                    self._last_timestamp = None
            return None
        journal = self.root / FALLBACK_PUBLISH_TRANSACTION
        if journal.exists() or journal.is_symlink():
            # The staged directory belongs to the write-ahead transaction. Moving
            # it to incomplete_* would strand the journal and permanently hide
            # the last valid dataset snapshot. Recovery owns it from this point.
            log.error(
                "leaving fallback publication stage untouched for recovery via %s",
                journal,
            )
            return None
        if self._stage_dir is None:
            return None
        for camera in self.cfg.camera_keys:
            self._flush_image_chunk(camera)
        stage = self._stage_dir
        _atomic_json(
            stage / "INCOMPLETE.json",
            {
                "schema_version": 1,
                "repo_id": self.cfg.repo_id,
                "format": "so101-fallback-v2",
                "fps": self.cfg.fps,
                "task": self.cfg.task,
                "camera_keys": self.cfg.camera_keys,
                "camera_storage": self._camera_storage_names,
                "reason": reason,
                "frames": self._num_frames,
                "aborted_at": time.time(),
            },
        )
        destination = self.root / f"incomplete_{int(time.time() * 1000)}"
        _sync_stage(stage)
        os.replace(stage, destination)
        _fsync_dir(self.root)
        self._stage_dir = None
        self._num_frames = 0
        self._image_buffers = {camera: [] for camera in self.cfg.camera_keys}
        self._first_timestamp = None
        self._last_timestamp = None
        log.warning("preserved incomplete episode -> %s", destination)
        return destination
