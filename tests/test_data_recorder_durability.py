"""Fallback episodes are durable and manifest publication is transactional.

After a power cut an episode must never appear in the dataset with
zero-length image chunks: every chunk and every directory of the staged
episode is fsynced before the rename that publishes it, and the dataset
directory after it. A durable marker makes the episode directory and root
manifest recover as one commit. Directory fsync is POSIX only, so on Windows
the tests check the files alone.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import numpy as np
import pytest

from so101 import JOINT_NAMES, atomic_io
from so101 import config as config_module
from so101.data import dataset as dataset_module
from so101.data import recorder as recorder_module
from so101.data.dataset import DatasetIdConflictError, dataset_root, get_dataset
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.teleop.controller import TeleopStep

POSIX = os.name != "nt"  # Windows cannot fsync a directory
JOINTS = {f"{joint}.pos": 0.0 for joint in JOINT_NAMES}


def _identity(path) -> tuple[int, int]:
    info = os.stat(path)
    return info.st_dev, info.st_ino


@pytest.fixture
def events(monkeypatch):
    """Every fsync (by inode) and rename, in order."""

    log: list[tuple] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        info = os.fstat(fd)
        log.append(("fsync", (info.st_dev, info.st_ino)))
        return real_fsync(fd)

    def replace(src, dst, *args, **kwargs):
        identity = _identity(src)
        real_replace(src, dst, *args, **kwargs)
        log.append(("replace", identity, str(dst)))

    monkeypatch.setattr(recorder_module.os, "fsync", fsync)
    monkeypatch.setattr(recorder_module.os, "replace", replace)
    return log


def _take(repo_id: str, frames: int = 10) -> EpisodeRecorder:
    take = EpisodeRecorder(
        RecordingConfig(repo_id=repo_id, camera_keys=["0", "1"], image_chunk_frames=4)
    )
    for index in range(frames):
        image = np.full((8, 8, 3), index, np.uint8)
        take.add(
            TeleopStep(t=index / 30, observation=JOINTS, action=JOINTS), {"0": image, "1": image}
        )
    return take


def _synced_before(events, identity, replace_index: int) -> bool:
    return ("fsync", identity) in events[:replace_index]


def test_storage_roots_are_durably_created(tmp_path, monkeypatch):
    storage = tmp_path / "new-storage"
    data = storage / "data"
    datasets = storage / "datasets"
    outputs = storage / "outputs"
    monkeypatch.setenv("SO101_DATA_DIR", str(data))
    monkeypatch.setenv("SO101_DATASETS_DIR", str(datasets))
    monkeypatch.setenv("SO101_OUTPUTS_DIR", str(outputs))
    calls: list[Path] = []
    fsync_directory = atomic_io.fsync_directory

    def spy(path: Path) -> None:
        calls.append(Path(path))
        fsync_directory(path)

    monkeypatch.setattr(atomic_io, "fsync_directory", spy)
    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
    finally:
        config_module.get_settings.cache_clear()

    assert (settings.data_dir, settings.datasets_dir, settings.outputs_dir) == (
        data,
        datasets,
        outputs,
    )
    assert tmp_path in calls  # names new-storage in its existing parent
    assert calls.count(storage) == 3  # names data, datasets and outputs


@pytest.mark.parametrize("nested", [False, True])
def test_durable_makedirs_retries_parent_fsync_after_eio(tmp_path, monkeypatch, nested):
    target = (
        tmp_path / "created-parent" / "child"
        if nested
        else tmp_path / "created-before-fsync-failed"
    )
    calls: list[Path] = []
    failed = False

    def fail_once(path: Path) -> None:
        nonlocal failed
        calls.append(Path(path))
        if not failed and Path(path) == tmp_path:
            failed = True
            raise OSError(errno.EIO, "injected parent fsync EIO")

    monkeypatch.setattr(atomic_io, "fsync_directory", fail_once)
    with pytest.raises(OSError, match="parent fsync EIO"):
        atomic_io.durable_makedirs(target)
    assert (target.parent if nested else target).is_dir()

    # The directory now exists, but the next durability attempt must retry the
    # failed parent entry instead of treating existence as success.
    atomic_io.durable_makedirs(target)
    assert target.is_dir()
    assert calls.count(tmp_path) >= 2


def test_every_chunk_and_directory_is_fsynced_before_the_episode_is_published(events):
    episode = _take("u/durable").save_episode()
    root = dataset_root("u/durable")
    published = root / f"episode_{episode:06d}"
    [publish] = [i for i, e in enumerate(events) if e[0] == "replace" and e[2] == str(published)]
    chunks = sorted(published.glob("images/*/chunk_*.npy"))
    assert len(chunks) == 6  # 3 chunks (4 + 4 + 2 frames) per camera
    staged = [*chunks, published / "frames.jsonl", published / "episode.json"]
    if POSIX:
        staged += [*published.glob("images/*"), published / "images", published]
    for path in staged:  # inodes survive the rename
        assert _synced_before(events, _identity(path), publish), path
    if not POSIX:
        return
    # A durable transaction marker hides the dataset first.  The new manifest
    # precedes the episode rename, and each step's directory entry is fsynced.
    [transaction] = [
        i
        for i, event in enumerate(events)
        if event[0] == "replace" and event[2].endswith(".fallback-publish.json")
    ]
    [meta] = [
        i
        for i, event in enumerate(events)
        if event[0] == "replace" and event[2].endswith("meta.json")
    ]
    assert transaction < meta < publish
    # The first save also has to persist the directory entry that names the
    # dataset root; fsyncing only the new root cannot make that parent entry
    # power-loss durable.
    assert ("fsync", _identity(root.parent)) in events[:transaction]
    root_sync = ("fsync", _identity(root))
    assert root_sync in events[transaction + 1 : meta]
    assert root_sync in events[meta + 1 : publish]
    assert root_sync in events[publish + 1 :]


def test_an_aborted_take_is_fsynced_before_it_is_kept_apart(events):
    kept = _take("u/aborted").abort("operator stopped")
    assert kept is not None
    [keep] = [i for i, e in enumerate(events) if e[0] == "replace" and e[2] == str(kept)]
    synced = [*kept.glob("images/*/chunk_*.npy")]
    assert len(synced) == 6
    if POSIX:
        synced += [*kept.glob("images/*"), kept]
    for path in synced:
        assert _synced_before(events, _identity(path), keep), path
    if POSIX:
        assert ("fsync", _identity(dataset_root("u/aborted"))) in events[keep + 1 :]


def test_an_unsupported_directory_fsync_is_not_fatal(monkeypatch, tmp_path):
    stand_in = (tmp_path / "directory-handle").open("w+b")

    def open_stand_in(_path, _flags):
        return os.dup(stand_in.fileno())

    def unsupported(_fd):
        raise OSError(errno.EINVAL, "unsupported")

    with monkeypatch.context() as patch:
        patch.setattr(atomic_io.os, "open", open_stand_in)
        patch.setattr(atomic_io.os, "fsync", unsupported)
        recorder_module._fsync_dir(tmp_path)
    stand_in.close()


@pytest.mark.parametrize("error", [errno.EIO, errno.ENOSPC])
def test_real_directory_fsync_failures_propagate(monkeypatch, tmp_path, error):
    stand_in = (tmp_path / "directory-handle").open("w+b")

    def open_stand_in(_path, _flags):
        return os.dup(stand_in.fileno())

    def fail_fsync(_fd):
        raise OSError(error, "injected directory durability failure")

    with monkeypatch.context() as patch:
        patch.setattr(atomic_io.os, "open", open_stand_in)
        patch.setattr(atomic_io.os, "fsync", fail_fsync)
        with pytest.raises(OSError, match="directory durability failure"):
            recorder_module._fsync_dir(tmp_path)
    stand_in.close()


def test_publication_fsync_failure_retains_the_recovery_marker(monkeypatch):
    repo_id = "u/publication-fsync"
    take = _take(repo_id, frames=2)
    root = dataset_root(repo_id)
    transaction = root / dataset_module.FALLBACK_PUBLISH_TRANSACTION
    fsync_dir = recorder_module._fsync_dir

    def fail_after_marker(path):
        if path == root and transaction.exists():
            raise OSError(errno.EIO, "injected publication fsync failure")
        return fsync_dir(path)

    monkeypatch.setattr(recorder_module, "_fsync_dir", fail_after_marker)
    with pytest.raises(OSError, match="publication fsync failure"):
        take.save_episode()

    assert transaction.is_file()
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    stage = root / payload["stage"]
    assert stage.is_dir()
    assert not (root / "episode_000000").exists()
    assert get_dataset(repo_id) is None

    # RobotManager/CLI call abort after a failed save. Abort must leave the
    # journal-owned stage in place, then a fresh recorder completes recovery.
    assert take.abort("save failed after journal publication") is None
    assert stage.is_dir() and transaction.is_file()
    monkeypatch.setattr(recorder_module, "_fsync_dir", fsync_dir)
    recovered = EpisodeRecorder(
        RecordingConfig(repo_id=repo_id, camera_keys=["0", "1"], image_chunk_frames=4)
    )
    recovered.abort("no new take")
    assert not transaction.exists()
    assert (root / "episode_000000" / "episode.json").is_file()
    assert get_dataset(repo_id) is not None


def test_final_marker_unlink_fsync_failure_restores_fail_closed_marker(monkeypatch):
    repo_id = "u/final-marker-fsync"
    take = _take(repo_id, frames=2)
    root = dataset_root(repo_id)
    transaction = root / dataset_module.FALLBACK_PUBLISH_TRANSACTION
    fsync_dir = recorder_module._fsync_dir
    failed = False

    def fail_after_marker_unlink(path):
        nonlocal failed
        if (
            not failed
            and path == root
            and not transaction.exists()
            and (root / "episode_000000").exists()
        ):
            failed = True
            raise OSError(errno.EIO, "injected marker unlink fsync failure")
        return fsync_dir(path)

    monkeypatch.setattr(recorder_module, "_fsync_dir", fail_after_marker_unlink)
    with pytest.raises(OSError, match="marker unlink fsync failure"):
        take.save_episode()

    assert failed and transaction.is_file()
    assert (root / "episode_000000" / "episode.json").is_file()
    assert get_dataset(repo_id) is None
    assert take.abort("marker cleanup failed") is None

    monkeypatch.setattr(recorder_module, "_fsync_dir", fsync_dir)
    recovered = EpisodeRecorder(
        RecordingConfig(repo_id=repo_id, camera_keys=["0", "1"], image_chunk_frames=4)
    )
    recovered.abort("no new take")
    assert not transaction.exists()
    assert get_dataset(repo_id) is not None


@pytest.mark.parametrize("failure_point", ["manifest", "episode"])
def test_failed_publication_never_exposes_an_orphan_episode(monkeypatch, failure_point):
    repo_id = "u/transaction-failure"
    assert _take(repo_id, frames=2).save_episode() == 0
    root = dataset_root(repo_id)
    original_meta = (root / "meta.json").read_bytes()
    original_episode = (root / "episode_000000" / "frames.jsonl").read_bytes()

    take = _take(repo_id, frames=3)
    atomic_json = recorder_module._atomic_json
    replace = recorder_module.os.replace

    def fail_meta(path, payload):
        if failure_point == "manifest" and path == root / "meta.json":
            raise OSError("injected publication failure")
        return atomic_json(path, payload)

    def fail_episode_rename(src, dst, *args, **kwargs):
        if failure_point == "episode" and os.fspath(dst) == os.fspath(root / "episode_000001"):
            raise OSError("injected publication failure")
        return replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(recorder_module, "_atomic_json", fail_meta)
    monkeypatch.setattr(recorder_module.os, "replace", fail_episode_rename)
    with pytest.raises(OSError, match="publication failure"):
        take.save_episode()

    assert (root / "meta.json").read_bytes() == original_meta
    assert (root / "episode_000000" / "frames.jsonl").read_bytes() == original_episode
    assert not (root / "episode_000001").exists()
    assert not (root / ".fallback-publish.json").exists()

    # The failed take remains private staging data until the caller explicitly
    # aborts it.  That preserves diagnostics without blocking a clean retry.
    monkeypatch.setattr(recorder_module, "_atomic_json", atomic_json)
    monkeypatch.setattr(recorder_module.os, "replace", replace)
    incomplete = take.abort("metadata publication failed")
    assert incomplete is not None and incomplete.name.startswith("incomplete_")
    assert _take(repo_id, frames=1).save_episode() == 1


@pytest.mark.parametrize(
    ("crash_point", "manifest_episodes"),
    [("before_manifest", 1), ("before_episode", 2)],
)
def test_crash_during_publication_recovers_the_transaction(
    monkeypatch, crash_point, manifest_episodes
):
    repo_id = "u/transaction-crash"
    assert _take(repo_id, frames=2).save_episode() == 0
    root = dataset_root(repo_id)
    take = _take(repo_id, frames=3)
    replace = recorder_module.os.replace
    atomic_json = recorder_module._atomic_json

    class SimulatedCrash(BaseException):
        pass

    def crash_before_episode_rename(src, dst, *args, **kwargs):
        if crash_point == "before_episode" and os.fspath(dst) == os.fspath(root / "episode_000001"):
            raise SimulatedCrash("power lost after manifest publication")
        return replace(src, dst, *args, **kwargs)

    def crash_before_manifest(path, payload):
        if crash_point == "before_manifest" and path == root / "meta.json":
            raise SimulatedCrash("power lost before manifest publication")
        return atomic_json(path, payload)

    monkeypatch.setattr(recorder_module.os, "replace", crash_before_episode_rename)
    monkeypatch.setattr(recorder_module, "_atomic_json", crash_before_manifest)
    with pytest.raises(SimulatedCrash, match="power lost"):
        take.save_episode()
    take.release_operation_lock()  # the OS would release this when the process died

    transaction = root / ".fallback-publish.json"
    assert transaction.is_file()
    assert (
        json.loads((root / "meta.json").read_text(encoding="utf-8"))["episodes"]
        == manifest_episodes
    )
    assert not (root / "episode_000001").exists()
    # Readers fail closed while the commit marker says the two files are not a
    # coherent snapshot; they must not consume the new count with missing data.
    assert get_dataset(repo_id) is None
    with pytest.raises(DatasetIdConflictError, match="'u/transaction-crash' already exists"):
        EpisodeRecorder(RecordingConfig(repo_id="U/TRANSACTION-CRASH"))

    monkeypatch.setattr(recorder_module.os, "replace", replace)
    monkeypatch.setattr(recorder_module, "_atomic_json", atomic_json)
    recovered = EpisodeRecorder(
        RecordingConfig(repo_id=repo_id, camera_keys=["0", "1"], image_chunk_frames=4)
    )
    recovered.abort("no new take")

    assert not transaction.exists()
    assert (root / "episode_000001" / "episode.json").is_file()
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert (meta["episodes"], meta["total_frames"]) == (2, 5)
    assert get_dataset(repo_id) is not None


def test_invalid_recovery_journal_fails_closed_without_changing_valid_data():
    repo_id = "u/transaction-corrupt"
    assert _take(repo_id, frames=2).save_episode() == 0
    root = dataset_root(repo_id)
    meta_path = root / "meta.json"
    episode_path = root / "episode_000000" / "frames.jsonl"
    before = (meta_path.read_bytes(), episode_path.read_bytes())
    transaction = root / ".fallback-publish.json"
    transaction.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="publication transaction.*invalid schema"):
        EpisodeRecorder(
            RecordingConfig(repo_id=repo_id, camera_keys=["0", "1"], image_chunk_frames=4)
        )

    assert (meta_path.read_bytes(), episode_path.read_bytes()) == before
    assert transaction.is_file()
    assert get_dataset(repo_id) is None
    with pytest.raises(DatasetIdConflictError, match="'u/transaction-corrupt' already exists"):
        EpisodeRecorder(RecordingConfig(repo_id="U/TRANSACTION-CORRUPT"))


@pytest.mark.parametrize("prior_frames", [0, 2])
def test_recovery_requires_an_exact_total_for_first_and_legacy_datasets(monkeypatch, prior_frames):
    repo_id = f"u/transaction-count-{prior_frames}"
    if prior_frames:
        assert _take(repo_id, frames=prior_frames).save_episode() == 0
        meta_path = dataset_root(repo_id) / "meta.json"
        legacy = json.loads(meta_path.read_text(encoding="utf-8"))
        legacy.pop("total_frames")
        meta_path.write_text(json.dumps(legacy), encoding="utf-8")

    take = _take(repo_id, frames=3)
    root = dataset_root(repo_id)
    meta_path = root / "meta.json"
    atomic_json = recorder_module._atomic_json

    class SimulatedCrash(BaseException):
        pass

    def crash_before_manifest(path, payload):
        if path == meta_path:
            raise SimulatedCrash("power lost before manifest publication")
        return atomic_json(path, payload)

    monkeypatch.setattr(recorder_module, "_atomic_json", crash_before_manifest)
    with pytest.raises(SimulatedCrash):
        take.save_episode()
    take.release_operation_lock()
    monkeypatch.setattr(recorder_module, "_atomic_json", atomic_json)

    transaction = root / dataset_module.FALLBACK_PUBLISH_TRANSACTION
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    payload["next_metadata"]["total_frames"] += 1
    transaction.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="total frame count"):
        EpisodeRecorder(
            RecordingConfig(repo_id=repo_id, camera_keys=["0", "1"], image_chunk_frames=4)
        )
    assert transaction.is_file()
    assert not (root / f"episode_{int(bool(prior_frames)):06d}").exists()
