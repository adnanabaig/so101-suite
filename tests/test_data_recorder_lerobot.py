"""Real-hardware recording fails closed and cleans up failed takes (audit B5, B9, B19)."""

from __future__ import annotations

import errno
import json
import sys
from copy import deepcopy

import numpy as np
import pytest

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.data import recorder as recorder_module
from so101.data.dataset import dataset_operation_lock, dataset_root
from so101.data.recorder import EpisodeRecorder, RecordingConfig, prune_unreferenced_lerobot_files
from so101.teleop.controller import TeleopStep

NAMES = [f"{joint}.pos" for joint in JOINT_NAMES]


def _step(value: float) -> TeleopStep:
    joints = dict.fromkeys(NAMES, value)
    return TeleopStep(t=1.0 + value, observation=joints, action=joints)


def _hide_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``import lerobot...`` fail, even after another test imported it.

    Masking only ``lerobot`` is not enough: a submodule already in
    ``sys.modules`` (the LeRobot contract tests load ``lerobot.datasets``)
    would still import.
    """

    for name in [n for n in sys.modules if n == "lerobot" or n.startswith("lerobot.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "lerobot", None)


def _lock_is_free(repo_id: str) -> bool:
    lock = dataset_operation_lock(repo_id)
    if not lock.acquire(blocking=False):
        return False
    lock.release()
    return True


class _Writer:
    def __init__(self) -> None:
        self.cleaned: list[int] = []

    def cleanup_interrupted_episode(self, episode_index: int) -> None:
        self.cleaned.append(episode_index)


class _FailingSaveDataset:
    """Mimics LeRobot 0.6.1: save_episode pops size/task, then video encoding fails."""

    def __init__(self) -> None:
        self.num_episodes = 0
        self.buffer: dict[str, object] = {}
        self.writer = _Writer()
        self.finalized = 0
        self.cleared: list[bool] = []

    @classmethod
    def create(cls, **_kwargs):
        return cls()

    def add_frame(self, _frame) -> None:
        self.buffer.setdefault("size", 0)
        self.buffer["size"] = int(self.buffer["size"]) + 1  # type: ignore[call-overload]
        self.buffer["task"] = ["task"]

    def has_pending_frames(self) -> bool:
        return int(self.buffer["size"]) > 0  # type: ignore[call-overload]  # KeyError after a failed save

    def save_episode(self) -> None:
        self.buffer.pop("size")
        self.buffer.pop("task")
        raise OSError(28, "No space left on device")

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        self.cleared.append(delete_images)
        self.buffer = {}

    def finalize(self) -> None:
        self.finalized += 1


def test_real_recording_refuses_without_lerobot_dataset_support(monkeypatch):
    _hide_lerobot(monkeypatch)
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match=r"lerobot\[dataset,feetech\]") as excinfo:
        EpisodeRecorder(RecordingConfig(repo_id="test/no_lerobot", camera_keys=[]))
    assert "SO101_MOCK=1" in str(excinfo.value)
    assert not dataset_root("test/no_lerobot").exists()
    assert _lock_is_free("test/no_lerobot")

    monkeypatch.setenv("SO101_MOCK", "1")
    get_settings.cache_clear()
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/no_lerobot", camera_keys=[]))
    recorder.add(_step(1.0))
    assert recorder.save_episode() == 0


def test_simulation_backend_records_fallback_without_lerobot(monkeypatch):
    _hide_lerobot(monkeypatch)
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    sim_settings = get_settings().model_copy(update={"backend": "sim"})
    monkeypatch.setattr(recorder_module, "get_settings", lambda: sim_settings)
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/sim_fallback", camera_keys=[]))
    recorder.add(_step(1.0))
    assert recorder.save_episode() == 0
    meta = json.loads((recorder.root / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "so101-fallback-v2"


def test_fallback_refuses_existing_lerobot_root_before_any_write(monkeypatch):
    """Missing optional LeRobot support must never mix a fallback tree into its root."""

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: None)
    root = dataset_root("test/existing_lerobot")
    (root / "meta").mkdir(parents=True)
    info = root / "meta" / "info.json"
    info.write_text(
        json.dumps(
            {
                "repo_id": "test/existing_lerobot",
                "fps": 30,
                "total_episodes": 1,
                "total_frames": 2,
                "features": {},
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    with pytest.raises(RuntimeError, match="already a LeRobot dataset"):
        EpisodeRecorder(RecordingConfig(repo_id="test/existing_lerobot", camera_keys=[]))

    after = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert after == before
    assert not (root / "meta.json").exists()
    assert not list(root.glob("episode_*"))
    assert not list(root.glob(".episode-staging-*"))
    assert _lock_is_free("test/existing_lerobot")


def test_new_lerobot_root_parent_is_fsynced_by_suite(monkeypatch):
    class CreatingDataset:
        @classmethod
        def create(cls, **kwargs):
            kwargs["root"].mkdir(parents=True)
            return cls()

        def finalize(self) -> None:
            pass

    synced = []
    fsync_dir = recorder_module._fsync_dir

    def spy(path):
        synced.append(path)
        fsync_dir(path)

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: CreatingDataset)
    monkeypatch.setattr(recorder_module, "_fsync_dir", spy)
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/new_lerobot_root"))
    try:
        assert recorder.root.parent in synced
    finally:
        recorder.release_operation_lock()


def test_new_lerobot_root_parent_fsync_failure_refuses_recorder(monkeypatch):
    class CreatingDataset:
        @classmethod
        def create(cls, **kwargs):
            kwargs["root"].mkdir(parents=True)
            return cls()

        def finalize(self) -> None:
            pass

    repo_id = "test/new_lerobot_root_eio"
    parent = dataset_root(repo_id).parent

    def fail_parent_fsync(path):
        if path == parent:
            raise OSError(errno.EIO, "injected LeRobot root parent EIO")

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: CreatingDataset)
    monkeypatch.setattr(recorder_module, "_fsync_dir", fail_parent_fsync)

    with pytest.raises(OSError, match="root parent EIO"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    assert _lock_is_free(repo_id)


def test_lerobot_resume_retries_failed_first_root_parent_fsync(monkeypatch):
    class CreatingDataset:
        resumed = 0

        @classmethod
        def create(cls, **kwargs):
            meta = kwargs["root"] / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text("{}", encoding="utf-8")
            return cls()

        @classmethod
        def resume(cls, **_kwargs):
            cls.resumed += 1
            return cls()

        def finalize(self) -> None:
            pass

    repo_id = "test/lerobot_parent_retry"
    parent = dataset_root(repo_id).parent
    syncs = 0

    def fail_once(path):
        nonlocal syncs
        if path == parent:
            syncs += 1
            if syncs == 1:
                raise OSError(errno.EIO, "injected first root parent EIO")

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: CreatingDataset)
    monkeypatch.setattr(EpisodeRecorder, "_validate_resumed_features", lambda *_args: None)
    monkeypatch.setattr(recorder_module, "_fsync_dir", fail_once)

    with pytest.raises(OSError, match="first root parent EIO"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id))

    recovered = EpisodeRecorder(RecordingConfig(repo_id=repo_id))
    try:
        assert syncs == 2
        assert CreatingDataset.resumed == 1
    finally:
        recovered.release_operation_lock()


@pytest.mark.parametrize(
    "artifact",
    [
        "episode_000000/frames.jsonl",
        "incomplete_123/INCOMPLETE.json",
        ".episode-staging-dead/frames.jsonl",
    ],
)
def test_lerobot_create_refuses_partial_fallback_artifacts_and_releases(monkeypatch, artifact):
    class CreatingDataset:
        calls = 0

        @classmethod
        def create(cls, **_kwargs):
            cls.calls += 1
            raise AssertionError("LeRobot create reached a fallback root")

    repo_id = "test/partial_fallback"
    root = dataset_root(repo_id)
    path = root / artifact
    path.parent.mkdir(parents=True)
    path.write_text("partial", encoding="utf-8")
    before = {
        item.relative_to(root): item.read_bytes() for item in root.rglob("*") if item.is_file()
    }
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: CreatingDataset)

    with pytest.raises(RuntimeError, match="partial fallback"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))

    after = {
        item.relative_to(root): item.read_bytes() for item in root.rglob("*") if item.is_file()
    }
    assert after == before and CreatingDataset.calls == 0
    assert not (root / "meta").exists()
    assert _lock_is_free(repo_id)


def _feature(dtype="float32", shape=(6,), names=None, **extra):
    return {
        "dtype": dtype,
        "shape": shape,
        "names": list(NAMES if names is None else names),
        **extra,
    }


def _base_features() -> dict:
    spec = _feature()
    return {
        "observation.state": deepcopy(spec),
        "action": deepcopy(spec),
        "desired_action": deepcopy(spec),
    }


class _ResumedFeatures:
    configured_features: dict = {}
    configured_fps: object = 30
    latest = None

    def __init__(self) -> None:
        self.fps = self.configured_fps
        self.features = deepcopy(self.configured_features)
        self.num_episodes = 1
        self.finalized = 0
        self.writer = _Writer()
        type(self).latest = self

    @classmethod
    def resume(cls, **_kwargs):
        return cls()

    def finalize(self) -> None:
        self.finalized += 1

    def add_frame(self, _frame) -> None:
        pass


def _resume_root(repo_id: str):
    root = dataset_root(repo_id)
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    return root


def test_lerobot_resume_accepts_only_known_generated_features_and_video_info(monkeypatch):
    repo_id = "test/resume_generated"
    _resume_root(repo_id)
    generated = {
        key: _feature(dtype="int64", shape=(1,), names=[])
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    _ResumedFeatures.configured_fps = 30
    _ResumedFeatures.configured_features = {**_base_features(), **generated}
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _ResumedFeatures)

    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))
    assert recorder._lerobot_ds is _ResumedFeatures.latest
    recorder.abort("test")
    assert _lock_is_free(repo_id)


@pytest.mark.parametrize(
    ("key", "feature"),
    [
        (
            "observation.environment_state",
            _feature(shape=(3,), names=("x", "y", "z")),
        ),
        (
            "observation.images.old_rgb",
            _feature(dtype="video", shape=(2, 3, 3), names=("height", "width", "channels")),
        ),
        (
            "observation.images.old_depth",
            _feature(
                dtype="video",
                shape=(2, 3, 1),
                names=("height", "width", "channels"),
                info={"video.is_depth_map": True},
            ),
        ),
    ],
)
def test_lerobot_resume_rejects_omitted_config_features_and_finalizes(monkeypatch, key, feature):
    repo_id = "test/resume_unexpected"
    _resume_root(repo_id)
    _ResumedFeatures.configured_fps = 30
    _ResumedFeatures.configured_features = {**_base_features(), key: feature}
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _ResumedFeatures)

    with pytest.raises(ValueError, match="unexpected features"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))
    assert _ResumedFeatures.latest.finalized == 1
    assert _lock_is_free(repo_id)


@pytest.mark.parametrize("fps", [30.0, "30", True, 30.5, -1])
def test_lerobot_resume_fps_requires_an_exact_positive_integer(monkeypatch, fps):
    repo_id = "test/resume_fps"
    _resume_root(repo_id)
    _ResumedFeatures.configured_fps = fps
    _ResumedFeatures.configured_features = _base_features()
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _ResumedFeatures)

    with pytest.raises(ValueError, match="FPS"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))
    assert _ResumedFeatures.latest.finalized == 1
    assert _lock_is_free(repo_id)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_camera", "missing required features"),
        ("dtype", "dtype differs"),
        ("depth", "depth-map meaning differs"),
        ("unknown_info", "unknown metadata"),
    ],
)
def test_lerobot_resume_validates_full_camera_payload_and_releases(monkeypatch, mutation, message):
    repo_id = "test/resume_camera_payload"
    _resume_root(repo_id)
    camera = "front"
    image_key = "observation.images.camera_00_front"
    image_feature = _feature(
        dtype="video",
        shape=(2, 3, 3),
        names=("height", "width", "channels"),
        info={
            "video.height": 2,
            "video.width": 3,
            "video.channels": 3,
            "video.fps": 30,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    )
    features = {**_base_features(), image_key: image_feature}
    if mutation == "missing_camera":
        features.pop(image_key)
    elif mutation == "dtype":
        features[image_key]["dtype"] = "image"
    elif mutation == "depth":
        features[image_key]["info"]["video.is_depth_map"] = True
    else:
        features[image_key]["info"]["mystery"] = "value"
    _ResumedFeatures.configured_fps = 30
    _ResumedFeatures.configured_features = features
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _ResumedFeatures)
    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[camera]))

    with pytest.raises(ValueError, match=message):
        recorder.add(_step(1.0), {camera: np.zeros((2, 3, 3), dtype=np.uint8)})
    assert _ResumedFeatures.latest.finalized == 1
    assert _lock_is_free(repo_id)


def test_lerobot_resume_accepts_known_generated_video_metadata(monkeypatch):
    repo_id = "test/resume_video_info"
    _resume_root(repo_id)
    camera = "front"
    _ResumedFeatures.configured_fps = 30
    _ResumedFeatures.configured_features = {
        **_base_features(),
        "observation.images.camera_00_front": _feature(
            dtype="video",
            shape=(2, 3, 3),
            names=("height", "width", "channels"),
            info={
                "video.height": 2,
                "video.width": 3,
                "video.channels": 3,
                "video.fps": 30,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        ),
    }
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _ResumedFeatures)
    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[camera]))
    recorder.add(_step(1.0), {camera: np.zeros((2, 3, 3), dtype=np.uint8)})
    recorder.abort("test")
    assert _lock_is_free(repo_id)


def test_failed_resume_detaches_before_a_new_owner_takes_the_dataset(monkeypatch):
    repo_id = "test/resume_owner_race"
    _resume_root(repo_id)
    camera = "front"
    key = "observation.images.camera_00_front"
    bad = _feature(
        dtype="image",
        shape=(2, 3, 3),
        names=("height", "width", "channels"),
    )
    _ResumedFeatures.configured_fps = 30
    _ResumedFeatures.configured_features = {**_base_features(), key: bad}
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _ResumedFeatures)
    failed = EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[camera]))
    with pytest.raises(ValueError, match="dtype differs"):
        failed.add(_step(1.0), {camera: np.zeros((2, 3, 3), dtype=np.uint8)})
    closed = _ResumedFeatures.latest
    assert closed.finalized == 1 and failed._lerobot_ds is None

    # A new recorder now owns the lock. Cleanup on the failed recorder must be
    # a no-op: no second finalize and no release of the new owner's lock.
    new_owner = EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[camera]))
    failed.abort("late caller cleanup")
    assert closed.finalized == 1
    with pytest.raises(RuntimeError, match="currently in use"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[camera]))
    new_owner.abort("test")
    assert _lock_is_free(repo_id)


def test_lerobot_resume_rejects_missing_requested_depth_and_environment(monkeypatch):
    for repo_id, config, images, missing in (
        (
            "test/resume_missing_depth",
            RecordingConfig(
                repo_id="test/resume_missing_depth", camera_keys=["front", "front#depth"]
            ),
            {
                "front": np.zeros((2, 3, 3), dtype=np.uint8),
                "front#depth": np.zeros((2, 3, 1), dtype=np.uint16),
            },
            "camera_00_front_depth",
        ),
        (
            "test/resume_missing_env",
            RecordingConfig(
                repo_id="test/resume_missing_env",
                camera_keys=[],
                environment_state_names=["target_x"],
            ),
            {},
            "observation.environment_state",
        ),
    ):
        _resume_root(repo_id)
        _ResumedFeatures.configured_fps = 30
        _ResumedFeatures.configured_features = _base_features()
        monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _ResumedFeatures)
        if images:
            recorder = EpisodeRecorder(config)
            with pytest.raises(ValueError, match=missing):
                recorder.add(
                    _step(1.0), images, environment_state=[0.0] if "env" in repo_id else None
                )
        else:
            with pytest.raises(ValueError, match=missing):
                EpisodeRecorder(config)
        assert _lock_is_free(repo_id)


def test_failed_lerobot_save_aborts_cleanly(monkeypatch):
    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _FailingSaveDataset)
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/save_fails", camera_keys=[]))
    recorder.add(_step(1.0))
    recorder.add(_step(2.0))
    with pytest.raises(OSError, match="No space left"):
        recorder.save_episode()
    dataset = recorder._lerobot_ds
    assert dataset is not None

    # Before the fix abort() raised KeyError('size') and never finalized, so the
    # next take reused the dead take's PNGs and parquet writer state.
    assert recorder.abort("episode save failed") is None
    assert dataset.cleared == [True]
    assert dataset.finalized == 1
    assert dataset.writer.cleaned == [0]
    assert recorder.num_frames == 0
    assert _lock_is_free("test/save_fails")


def test_resumed_lerobot_dataset_drops_interrupted_take(monkeypatch):
    class _Resumable(_FailingSaveDataset):
        def __init__(self) -> None:
            super().__init__()
            self.num_episodes = 3
            self.fps = 30
            spec = {"dtype": "float32", "shape": (6,), "names": NAMES}
            self.features = {
                "observation.state": spec,
                "action": spec,
                "desired_action": spec,
            }

        @classmethod
        def resume(cls, **_kwargs):
            return cls()

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: _Resumable)
    root = dataset_root("test/resume_cleanup")
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/resume_cleanup", camera_keys=[]))
    assert recorder._lerobot_ds is not None
    assert recorder._lerobot_ds.writer.cleaned == [3]
    recorder.abort()


def _lerobot_tree(root, *, total_episodes: int) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": total_episodes,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                ),
                "features": {"observation.images.top": {"dtype": "video"}},
            }
        ),
        encoding="utf-8",
    )
    for relative in (
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.top/chunk-000/file-000.mp4",
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(b"orphan")


def test_prune_removes_files_of_a_first_take_that_never_saved(tmp_path):
    _lerobot_tree(tmp_path, total_episodes=0)
    removed = prune_unreferenced_lerobot_files(tmp_path)
    assert removed == [
        "data/chunk-000/file-000.parquet",
        "videos/observation.images.top/chunk-000/file-000.mp4",
    ]
    assert not list((tmp_path / "data").rglob("*.parquet"))


def test_prune_refuses_when_episode_metadata_is_incomplete(tmp_path):
    _lerobot_tree(tmp_path, total_episodes=2)  # metadata lists 0 of 2 episodes
    assert prune_unreferenced_lerobot_files(tmp_path) == []
    assert (tmp_path / "data/chunk-000/file-000.parquet").is_file()
    assert prune_unreferenced_lerobot_files(tmp_path / "missing") == []


def test_prune_keeps_referenced_files(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    _lerobot_tree(tmp_path, total_episodes=1)
    for relative in (
        "data/chunk-000/file-001.parquet",
        "videos/observation.images.top/chunk-000/file-001.mp4",
    ):
        (tmp_path / relative).write_bytes(b"orphan")
    episodes = tmp_path / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    table = pa.table(
        {
            "data/chunk_index": [0],
            "data/file_index": [0],
            "videos/observation.images.top/chunk_index": [0],
            "videos/observation.images.top/file_index": [0],
        }
    )
    pq.write_table(table, episodes / "file-000.parquet")
    removed = prune_unreferenced_lerobot_files(tmp_path)
    assert removed == [
        "data/chunk-000/file-001.parquet",
        "videos/observation.images.top/chunk-000/file-001.mp4",
    ]
    assert (tmp_path / "data/chunk-000/file-000.parquet").is_file()
