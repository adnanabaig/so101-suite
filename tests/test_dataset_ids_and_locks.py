"""Dataset ids, discovery and the per-dataset operation lock (data review fixes)."""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.data import dataset as dataset_module
from so101.data import recorder as recorder_module
from so101.data.dataset import (
    DatasetBusyError,
    DatasetIdConflictError,
    DatasetLock,
    DatasetNotFoundError,
    dataset_operation_lock,
    dataset_read_hold,
    dataset_root,
    delete_dataset,
    get_dataset,
    list_datasets,
    list_trashed_datasets,
    require_dataset,
    restore_trashed_dataset,
)
from so101.data.inspection import (
    OUTCOME_LABELS_CAPABILITY,
    OutcomeLabelsUnreadableError,
    inspect_dataset,
    set_episode_outcome,
)
from so101.data.loader import TrainingDataError, load_training_data
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.data.replay import replay_episode
from so101.services.app import create_app
from so101.teleop.controller import TeleopStep

_JOINTS = {f"{joint}.pos": 0.0 for joint in JOINT_NAMES}


def _lerobot_info(root, features: dict, *, episodes: int = 0) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": episodes,
                "total_frames": episodes,
                "features": features,
            }
        )
    )


def test_lerobot_camera_keys_keep_every_dot_after_the_images_prefix():
    # Two rtsp cameras whose names differ only before their last dot used to
    # list as the same key ("5_live").
    root = get_settings().datasets_dir / "v2--lab%2Frtsp"
    _lerobot_info(
        root,
        {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "observation.images.rtsp_10.0.0.5_live": {"dtype": "video", "shape": [4, 4, 3]},
            "observation.images.rtsp_10.0.1.5_live": {"dtype": "video", "shape": [4, 4, 3]},
            "observation.images.rtsp_10.0.0.5_live_depth": {
                "dtype": "video",
                "shape": [4, 4, 1],
                "info": {"is_depth_map": True},
            },
        },
    )
    info = get_dataset("lab/rtsp")
    assert info is not None
    assert info.camera_keys == ["rtsp_10.0.0.5_live", "rtsp_10.0.1.5_live"]
    assert info.depth_keys == ["rtsp_10.0.0.5_live_depth"]
    assert [ds.camera_keys for ds in list_datasets()] == [info.camera_keys]


# -- reserved id components (finding 5) ------------------------------------------------
@pytest.mark.parametrize(
    ("repo_id", "word"),
    [
        ("lab/qc", "qc"),
        ("trash", "trash"),
        ("trash/lab", "trash"),
        ("lab/push", "push"),
        ("lab/episodes/3", "episodes"),
    ],
)
def test_api_route_words_are_reserved_id_components(repo_id, word):
    with pytest.raises(ValueError, match=f"reserved component '{word}'"):
        dataset_root(repo_id)
    with pytest.raises(ValueError, match="dataset API route names"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    assert not list(get_settings().datasets_dir.glob("v2--*"))


@pytest.mark.parametrize("repo_id", ["lab/qcs", "lab/QC", "trashcan", "lab/push_back", "lab/q.c"])
def test_ids_that_only_contain_a_route_word_stay_valid(repo_id):
    assert dataset_root(repo_id).name.startswith("v2--")


def test_recording_api_refuses_a_reserved_id_with_the_reason():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/recording/start", json={"repo_id": "lab/qc", "operator_id": "op"}
        )
    assert response.status_code == 422
    assert "reserved component 'qc'" in json.dumps(response.json())


# -- folder names never alias another id (finding 4) -------------------------------------
def _fallback_folder(name: str, repo_id: str, episodes: int = 0):
    root = get_settings().datasets_dir / name
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps({"repo_id": repo_id, "episodes": episodes, "format": "so101-fallback-v2"})
    )
    return root


def test_a_legacy_flattened_folder_name_is_not_another_repo_id():
    legacy = _fallback_folder("bob__reach", "bob/reach")
    assert get_dataset("bob/reach").path == str(legacy)
    assert get_dataset("bob__reach") is None
    lock = dataset_operation_lock("bob/reach")
    assert lock.acquire(blocking=False)  # e.g. a recording into bob/reach
    try:
        assert delete_dataset("bob__reach") is False
        with pytest.raises(FileNotFoundError):
            set_episode_outcome("bob__reach", 0, "success")
        with pytest.raises(DatasetNotFoundError):
            require_dataset("bob__reach")
        with pytest.raises(DatasetBusyError):
            delete_dataset("bob/reach")
    finally:
        lock.release()
    assert legacy.is_dir() and get_dataset("bob/reach") is not None


def test_every_lock_is_taken_on_the_folder_the_operation_touches():
    # A folder named for one id whose manifest names another: deleting the
    # manifest's id moves the folder the encoded id's operation lock protects.
    _fallback_folder("v2--carol%2Fpick", "carol/place")
    lock = dataset_operation_lock("carol/pick")
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(DatasetBusyError):
            delete_dataset("carol/place")
        with pytest.raises(DatasetBusyError):
            with dataset_read_hold("carol/place"):
                pass
    finally:
        lock.release()
    assert delete_dataset("carol/place") is True


# -- ids that differ only in case (finding 3) ----------------------------------------------
def _record(repo_id: str, frames: int = 3) -> int:
    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    for index in range(frames):
        recorder.add(TeleopStep(t=float(index), observation=_JOINTS, action=_JOINTS))
    return recorder.save_episode()


def test_noncanonical_discovered_dataset_uses_one_root_and_one_lock():
    repo_id = "u/path-alias"
    imported = _fallback_folder("zz-import", repo_id)
    canonical = dataset_root(repo_id)
    take = EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    take.add(TeleopStep(t=0.0, observation=_JOINTS, action=_JOINTS))
    assert take.save_episode(retain_operation_lock=True) == 0
    try:
        matches = [dataset for dataset in list_datasets() if dataset.repo_id == repo_id]
        assert len(matches) == 1
        assert Path(matches[0].path) == imported
        assert not canonical.exists()
        with pytest.raises(DatasetBusyError):
            EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    finally:
        take.release_operation_lock()


def test_ids_that_differ_only_in_case_share_one_lock():
    _record("alice/pick")
    assert dataset_operation_lock("Alice/Pick") is dataset_operation_lock("alice/pick")
    assert dataset_operation_lock("ALICE/PICK") is dataset_operation_lock("alice/pick")
    with dataset_read_hold("alice/pick"):
        with pytest.raises(DatasetBusyError):
            EpisodeRecorder(RecordingConfig(repo_id="Alice/Pick"))


def test_a_new_id_equal_to_an_existing_one_ignoring_case_is_refused():
    _record("alice/pick")
    with pytest.raises(DatasetIdConflictError, match="'alice/pick' already exists"):
        EpisodeRecorder(RecordingConfig(repo_id="Alice/Pick"))
    # Nothing was written, nothing stays locked, and the existing id still works.
    assert [(ds.repo_id, ds.num_episodes) for ds in list_datasets()] == [("alice/pick", 1)]
    assert _record("alice/pick") == 1
    assert get_dataset("Alice/Pick") is None
    with pytest.raises(DatasetNotFoundError, match="did you mean 'alice/pick'"):
        require_dataset("Alice/Pick")


def test_fallback_discovery_rechecks_the_transaction_marker_after_read(monkeypatch):
    root = _fallback_folder("v2--u%2Fread-race", "u/read-race")
    meta_path = root / "meta.json"
    read_snapshot = dataset_module._read_file_snapshot

    def publish_marker_after_read(path):
        snapshot = read_snapshot(path)
        if path == meta_path:
            (root / dataset_module.FALLBACK_PUBLISH_TRANSACTION).write_text("{}")
        return snapshot

    monkeypatch.setattr(dataset_module, "_read_file_snapshot", publish_marker_after_read)
    assert dataset_module._read_fallback(root) is None


def test_fallback_discovery_rejects_marker_aba_after_rollback(monkeypatch):
    root = _fallback_folder("v2--u%2Fread-rollback", "u/read-rollback")
    meta_path = root / "meta.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8"))
    previous.update(fps=30, task="Perform the task.", camera_keys=[], total_frames=0)
    meta_path.write_text(json.dumps(previous), encoding="utf-8")
    following = {**previous, "episodes": 1, "total_frames": 1}
    marker = root / dataset_module.FALLBACK_PUBLISH_TRANSACTION
    read_snapshot = dataset_module._read_file_snapshot
    first = True

    def read_next_while_writer_rolls_back(path):
        nonlocal first
        if path != meta_path or not first:
            return read_snapshot(path)
        first = False
        marker.write_text("{}", encoding="utf-8")
        meta_path.write_text(json.dumps(following), encoding="utf-8")
        captured = read_snapshot(path)
        # An injected episode-rename EIO restores the prior manifest and then
        # removes the marker before the reader performs its final validation.
        meta_path.write_text(json.dumps(previous), encoding="utf-8")
        marker.unlink()
        return captured

    monkeypatch.setattr(dataset_module, "_read_file_snapshot", read_next_while_writer_rolls_back)
    assert dataset_module._read_fallback(root) is None
    assert json.loads(meta_path.read_text(encoding="utf-8"))["episodes"] == 0


def test_incomplete_discovery_rechecks_the_transaction_marker_after_scan(monkeypatch):
    root = get_settings().datasets_dir / "v2--u%2Fincomplete-race"
    incomplete = root / "incomplete_1"
    incomplete.mkdir(parents=True)
    manifest = incomplete / "INCOMPLETE.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "u/incomplete-race",
                "format": "so101-fallback-v2",
                "fps": 30,
                "camera_keys": [],
            }
        )
    )
    read_text = Path.read_text

    def publish_marker_after_read(path, *args, **kwargs):
        text = read_text(path, *args, **kwargs)
        if path == manifest:
            (root / dataset_module.FALLBACK_PUBLISH_TRANSACTION).write_text("{}")
        return text

    monkeypatch.setattr(Path, "read_text", publish_marker_after_read)
    assert dataset_module._read_incomplete_only(root) is None


def test_an_encoded_dataset_root_symlink_is_never_followed(tmp_path, monkeypatch):
    datasets = get_settings().datasets_dir
    datasets.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = datasets / "v2--u%2Flinked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        linked.mkdir()
        is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == linked or is_symlink(path),
        )

    with pytest.raises(ValueError, match="dataset root.*symlink"):
        dataset_root("u/linked")
    with pytest.raises(ValueError, match="dataset root.*symlink"):
        EpisodeRecorder(RecordingConfig(repo_id="u/linked"))
    assert list_datasets() == []
    assert list(outside.iterdir()) == []


def test_episode_directory_symlink_cannot_cross_dataset_locks(monkeypatch):
    source_id = "u/symlink-source"
    alias_id = "u/symlink-alias"
    _record(source_id, frames=1)
    source_root = dataset_root(source_id)
    source_episode = source_root / "episode_000000"
    source_manifest = source_episode / "episode.json"
    before = source_manifest.read_bytes()

    alias_root = dataset_root(alias_id)
    alias_root.mkdir(parents=True)
    metadata = json.loads((source_root / "meta.json").read_text(encoding="utf-8"))
    metadata["repo_id"] = alias_id
    (alias_root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    alias_episode = alias_root / "episode_000000"
    try:
        alias_episode.symlink_to(source_episode.resolve(), target_is_directory=True)
    except OSError:
        # Windows may disallow symlinks without developer mode. Keep the
        # contract test live with a copied stand-in reported as the symlink the
        # production checks must reject.
        shutil.copytree(source_episode, alias_episode)
        is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == alias_episode or is_symlink(path),
        )

    with pytest.raises(ValueError, match="episode path is a symlink"):
        EpisodeRecorder(RecordingConfig(repo_id=alias_id))
    with pytest.raises(ValueError, match="symlink"):
        recorder_module._transaction_previous_frame_count(alias_root, 1)
    with dataset_read_hold(source_id):
        with pytest.raises(OutcomeLabelsUnreadableError, match="symlink"):
            set_episode_outcome(alias_id, 0, "failure")
    assert source_manifest.read_bytes() == before


@pytest.mark.parametrize(
    "artifact",
    ["episode.json", "frames.jsonl", "images/extra/chunk_000000.npy"],
)
def test_fallback_episode_file_symlinks_fail_closed_everywhere(artifact, monkeypatch):
    source_id = f"u/file-source-{Path(artifact).name}"
    alias_id = f"u/file-alias-{Path(artifact).name}"
    _record(source_id, frames=1)
    source_root = dataset_root(source_id)
    source_episode = source_root / "episode_000000"
    source_artifact = source_episode / artifact
    if not source_artifact.exists():
        source_artifact.parent.mkdir(parents=True, exist_ok=True)
        source_artifact.write_bytes(b"source artifact")

    alias_root = dataset_root(alias_id)
    alias_root.mkdir(parents=True)
    metadata = json.loads((source_root / "meta.json").read_text(encoding="utf-8"))
    metadata["repo_id"] = alias_id
    (alias_root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    alias_episode = alias_root / "episode_000000"
    shutil.copytree(source_episode, alias_episode)
    alias_artifact = alias_episode / artifact
    alias_artifact.unlink()
    try:
        alias_artifact.symlink_to(source_artifact.resolve())
    except OSError:
        alias_artifact.write_bytes(source_artifact.read_bytes())
        is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == alias_artifact or is_symlink(path),
        )

    with pytest.raises(ValueError, match="contains a symlink"):
        EpisodeRecorder(RecordingConfig(repo_id=alias_id))
    with pytest.raises(ValueError, match="contains a symlink"):
        recorder_module._transaction_previous_frame_count(alias_root, 1)

    inspection = inspect_dataset(alias_id)
    labels = next(item for item in inspection.capabilities if item.id == OUTCOME_LABELS_CAPABILITY)
    assert not labels.available and "symlink" in labels.detail
    with pytest.raises(TrainingDataError, match="no usable episodes"):
        load_training_data(alias_id, selection="all")

    class Follower:
        def send_goal(self, _goal):
            raise AssertionError("replay reached the arm")

    with pytest.raises(ValueError, match="contains a symlink"):
        replay_episode(alias_id, 0, follower=Follower(), fps=240)


def test_an_unidentifiable_marker_blocks_new_dataset_admission():
    hidden = get_settings().datasets_dir / "legacy-unknown"
    hidden.mkdir(parents=True)
    (hidden / dataset_module.FALLBACK_PUBLISH_TRANSACTION).write_text("{}")

    with pytest.raises(DatasetIdConflictError, match="cannot verify the dataset identity"):
        EpisodeRecorder(RecordingConfig(repo_id="u/new-dataset"))


def test_restore_refuses_an_id_that_differs_only_in_case():
    _record("Bob/Reach")
    assert delete_dataset("Bob/Reach")
    _record("bob/reach")
    [trashed] = list_trashed_datasets()
    with pytest.raises(FileExistsError, match="'bob/reach' already exists"):
        restore_trashed_dataset(trashed.id)
    assert [ds.repo_id for ds in list_datasets()] == ["bob/reach"]


def test_recording_api_names_the_existing_id_on_a_case_conflict():
    _record("alice/pick")
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/recording/start", json={"repo_id": "Alice/Pick", "operator_id": "op"}
        )
    assert response.status_code == 409
    assert "'alice/pick' already exists" in response.json()["detail"]


# -- locks hold across processes (finding 6) ------------------------------------------------
_CHILD = """
import sys
from so101 import JOINT_NAMES
from so101.data.dataset import dataset_read_hold
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.teleop.controller import TeleopStep

mode, repo_id = sys.argv[1], sys.argv[2]
joints = {f"{joint}.pos": 0.0 for joint in JOINT_NAMES}
if mode == "record":  # an API recording, mid-episode
    take = EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    take.add(TeleopStep(t=0.0, observation=joints, action=joints))
    print("held", flush=True)
    sys.stdin.readline()
    take.add(TeleopStep(t=0.1, observation=joints, action=joints))
    print("saved", take.save_episode(), flush=True)
else:  # a training job or Hub push in another process
    with dataset_read_hold(repo_id):
        print("held", flush=True)
        sys.stdin.readline()
    print("released", flush=True)
"""


@contextmanager
def _other_process(mode: str, repo_id: str, cwd) -> Iterator[subprocess.Popen[str]]:
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, mode, repo_id],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=dict(os.environ),
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        yield child
    finally:
        if child.poll() is None:
            child.communicate("\n", timeout=30)


def test_a_recording_in_another_process_blocks_deletion(tmp_path):
    _record("u/shared")
    with _other_process("record", "u/shared", tmp_path) as child:
        with pytest.raises(DatasetBusyError, match="another so101 process"):
            delete_dataset("u/shared")
        with pytest.raises(DatasetBusyError, match="another so101 process"):
            set_episode_outcome("u/shared", 0, "failure")
        with pytest.raises(DatasetBusyError, match="another so101 process"):
            with dataset_read_hold("u/shared"):
                pass
        out, _ = child.communicate("\n", timeout=30)
    assert child.returncode == 0 and out.split() == ["saved", "1"]
    assert get_dataset("u/shared").num_episodes == 2  # the take was not lost


def test_a_read_hold_in_another_process_blocks_writers(tmp_path):
    _record("u/read")
    with _other_process("read", "u/read", tmp_path):
        with pytest.raises(DatasetBusyError, match="another so101 process"):
            delete_dataset("u/read")
        with pytest.raises(DatasetBusyError, match="another so101 process"):
            EpisodeRecorder(RecordingConfig(repo_id="u/read"))
        if os.name != "nt":  # Windows file locks are exclusive only
            with dataset_read_hold("u/read"):  # readers share the dataset
                pass
    assert delete_dataset("u/read") is True


def test_a_writer_here_blocks_a_read_hold_in_another_process(tmp_path):
    _record("u/write")
    lock = dataset_operation_lock("u/write")
    assert lock.acquire(blocking=False)
    try:
        child = subprocess.run(
            [sys.executable, "-c", _CHILD, "read", "u/write"],
            input="\n",
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=dict(os.environ),
            timeout=60,
        )
    finally:
        lock.release()
    assert child.returncode != 0
    assert "DatasetBusyError" in child.stderr and "another so101 process" in child.stderr


def test_the_lock_waits_with_a_timeout_and_is_reentrant_safe():
    lock = dataset_operation_lock("u/timeout")
    other = DatasetLock(lock.path)  # another holder of the same lock file
    assert other.acquire(blocking=False)
    try:
        assert not lock.acquire(timeout=0.1)  # the file lock is held elsewhere
        assert not lock.locked()  # and the thread lock was given back
    finally:
        other.release()
    with lock:
        assert lock.locked()
    assert not lock.locked()
    with pytest.raises(RuntimeError):
        lock.release()  # like threading.Lock: releasing an unheld lock is an error


def test_dataset_lock_open_failure_never_falls_back_to_process_only(monkeypatch, tmp_path):
    lock = DatasetLock(tmp_path / "unwritable" / "dataset.lock")

    def fail_open():
        raise PermissionError(errno.EACCES, "injected lock open failure")

    monkeypatch.setattr(lock, "_open", fail_open)
    with pytest.raises(PermissionError, match="lock open failure"):
        lock.acquire(blocking=False)
    assert not lock.locked()


def test_dataset_lock_enolck_never_falls_back_to_process_only(monkeypatch, tmp_path):
    lock = DatasetLock(tmp_path / "dataset.lock")

    def fail_lock(_fd, *, shared):
        del shared
        raise OSError(errno.ENOLCK, "injected lock service failure")

    monkeypatch.setattr(dataset_module, "_try_file_lock", fail_lock)
    with pytest.raises(OSError, match="lock service failure"):
        lock.acquire(blocking=False)
    assert not lock.locked()


def test_cli_real_training_holds_dataset_for_stream_lifetime(tmp_path, monkeypatch):
    """The direct CLI needs the same lazy-reader hold as the API JobRunner."""

    from typer.testing import CliRunner

    from so101.cli import app
    from so101.training import trainer as trainer_module

    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    repo_id = "u/cli_train_hold"
    _lerobot_info(dataset_root(repo_id), {}, episodes=1)
    writer_access: list[bool] = []

    def writer_can_acquire() -> bool:
        lock = dataset_operation_lock(repo_id)
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        return acquired

    def training_stream(_job, cancel_event=None, on_start=None):
        del cancel_event, on_start
        try:
            writer_access.append(writer_can_acquire())
            yield 1, 0.5, "step:1 loss:0.5"
        finally:
            # Generator exhaustion/close still happens inside the read hold.
            writer_access.append(writer_can_acquire())

    monkeypatch.setattr(trainer_module, "run_training", training_stream)
    result = CliRunner().invoke(
        app,
        [
            "train",
            repo_id,
            "--episodes",
            "all",
            "--steps",
            "1",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert writer_access == [False, False]
    assert writer_can_acquire() is True
