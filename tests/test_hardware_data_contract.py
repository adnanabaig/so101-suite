"""Safety-critical contracts across teleop, recording, and policy loading."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from so101 import JOINT_NAMES
from so101.data.dataset import dataset_operation_lock, dataset_root
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.policy.server import LeRobotPolicy, load_policy
from so101.teleop.controller import TeleopStep


def _joint_dict(value: float) -> dict[str, float]:
    return {f"{joint}.pos": value for joint in JOINT_NAMES}


def _lock_is_free(repo_id: str) -> bool:
    lock = dataset_operation_lock(repo_id)
    if not lock.acquire(blocking=False):
        return False
    lock.release()
    return True


def test_recorder_persists_actual_and_desired_actions_separately():
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/action_contract", camera_keys=[]))
    recorder.add(
        TeleopStep(
            t=123.0,
            observation=_joint_dict(1.0),
            action=_joint_dict(2.0),
            desired_action=_joint_dict(9.0),
        )
    )
    assert recorder.num_frames == 1
    recorder.save_episode()

    frame_path = recorder.root / "episode_000000" / "frames.jsonl"
    frame = json.loads(frame_path.read_text().strip())
    assert frame["action"] == [2.0] * len(JOINT_NAMES)
    assert frame["desired_action"] == [9.0] * len(JOINT_NAMES)
    assert recorder.num_frames == 0


def test_current_lerobot_dataset_contract_includes_robot_and_desired_action(monkeypatch):
    class FakeDataset:
        create_kwargs = None

        def __init__(self):
            self.frames = []
            self.num_episodes = 0

        @classmethod
        def create(cls, **kwargs):
            cls.create_kwargs = kwargs
            return cls()

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.num_episodes += 1

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: FakeDataset)
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/current_lerobot", camera_keys=[]))
    recorder.add(
        TeleopStep(
            t=1.0,
            observation=_joint_dict(0.0),
            action=_joint_dict(1.0),
            desired_action=_joint_dict(3.0),
        )
    )

    assert FakeDataset.create_kwargs["robot_type"] == "so_follower"
    assert FakeDataset.create_kwargs["features"]["desired_action"]["names"] == [
        f"{joint}.pos" for joint in JOINT_NAMES
    ]
    assert recorder._lerobot_ds.frames[0]["action"].tolist() == [1.0] * len(JOINT_NAMES)
    assert recorder._lerobot_ds.frames[0]["desired_action"].tolist() == [3.0] * len(JOINT_NAMES)


def test_existing_lerobot_dataset_is_resumed_and_finalized(monkeypatch):
    names = [f"{joint}.pos" for joint in JOINT_NAMES]

    class ResumableDataset:
        resume_kwargs = None

        def __init__(self):
            self.fps = 30
            self.features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": names,
                },
                "action": {"dtype": "float32", "shape": (6,), "names": names},
                "desired_action": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": names,
                },
            }
            self.num_episodes = 1
            self.finalized = False

        @classmethod
        def resume(cls, **kwargs):
            cls.resume_kwargs = kwargs
            return cls()

        def add_frame(self, _frame):
            pass

        def save_episode(self):
            self.num_episodes += 1

        def finalize(self):
            self.finalized = True

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: ResumableDataset)
    root = dataset_root("test/resume_lerobot")
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/resume_lerobot", camera_keys=[]))
    recorder.add(
        TeleopStep(
            t=1.0,
            observation=_joint_dict(0.0),
            action=_joint_dict(1.0),
        )
    )

    episode = recorder.save_episode()

    assert ResumableDataset.resume_kwargs["root"] == recorder.root
    assert episode == 1
    assert recorder._lerobot_ds.finalized is True


def test_fallback_images_stream_in_bounded_atomic_chunks():
    config = RecordingConfig(
        repo_id="test/chunked_images",
        camera_keys=["/dev/camera 0"],
        image_chunk_frames=2,
    )
    recorder = EpisodeRecorder(config)
    for frame_index in range(5):
        recorder.add(
            TeleopStep(
                t=float(frame_index),
                observation=_joint_dict(0.0),
                action=_joint_dict(1.0),
            ),
            {"/dev/camera 0": np.full((2, 3, 3), frame_index, dtype=np.uint8)},
        )

    # Only the not-yet-full final chunk is resident, regardless of episode length.
    assert len(recorder._image_buffers["/dev/camera 0"]) == 1
    episode = recorder.save_episode()
    assert episode == 0
    episode_dir = recorder.root / "episode_000000"
    chunks = sorted((episode_dir / "images" / "00_dev_camera_0").glob("*.npy"))
    assert [np.load(chunk).shape[0] for chunk in chunks] == [2, 2, 1]
    metadata = json.loads((recorder.root / "meta.json").read_text())
    assert metadata["format"] == "so101-fallback-v2"
    assert metadata["camera_storage"]["/dev/camera 0"] == "00_dev_camera_0"
    assert not list(recorder.root.glob(".episode-staging-*"))


def _fallback_layout_dataset() -> tuple[RecordingConfig, EpisodeRecorder]:
    config = RecordingConfig(
        repo_id="test/fallback_layout",
        fps=10,
        task="Reach for the target.",
        camera_keys=["front"],
        camera_feature_names={"front": "front_rgb"},
        environment_state_names=["target_x"],
    )
    recorder = EpisodeRecorder(config)
    recorder.add(
        TeleopStep(t=1.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)),
        {"front": np.zeros((2, 3, 3), dtype=np.uint8)},
        environment_state=[0.25],
    )
    recorder.save_episode()
    return config, recorder


@pytest.mark.parametrize(
    ("difference", "message"),
    [
        ("repo_id", "repo_id"),
        ("fps", "FPS"),
        ("task", "task"),
        ("camera_keys", "camera keys"),
        ("camera_features", "camera feature names"),
        ("camera_storage", "camera storage"),
        ("image_storage", "image storage"),
        ("environment_state", "environment state names"),
    ],
)
def test_fallback_resume_rejects_incompatible_config_before_staging(difference, message):
    config, first = _fallback_layout_dataset()
    meta_path = first.root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    resumed = config
    if difference == "repo_id":
        meta["repo_id"] = "somebody/elses_dataset"
    elif difference == "fps":
        resumed = replace(config, fps=30)
    elif difference == "task":
        resumed = replace(config, task="Pick up the target.")
    elif difference == "camera_keys":
        resumed = replace(
            config,
            camera_keys=["wrist_cam"],
            camera_feature_names={"wrist_cam": "wrist_rgb"},
        )
    elif difference == "camera_features":
        resumed = replace(config, camera_feature_names={"front": "another_rgb"})
    elif difference == "camera_storage":
        meta["camera_storage"] = {"front": "99_elsewhere"}
    elif difference == "image_storage":
        meta["image_storage"] = "loose-png"
    elif difference == "environment_state":
        resumed = replace(config, environment_state_names=["cube_x"])
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    before = {
        path.relative_to(first.root): path.read_bytes()
        for path in first.root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match=message):
        candidate = EpisodeRecorder(resumed)
        candidate.add(
            TeleopStep(t=2.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)),
            {camera: np.zeros((2, 3, 3), dtype=np.uint8) for camera in resumed.camera_keys},
            environment_state=[0.5] if resumed.environment_state_names else None,
        )
        candidate.save_episode()

    after = {
        path.relative_to(first.root): path.read_bytes()
        for path in first.root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(first.root.glob(".episode-staging-*"))
    assert not (first.root / "episode_000001").exists()
    assert _lock_is_free(config.repo_id)


def test_fallback_resume_rejects_camera_shape_before_staging():
    config, first = _fallback_layout_dataset()
    resumed = EpisodeRecorder(config)
    with pytest.raises(ValueError, match="camera.*shape"):
        resumed.add(
            TeleopStep(t=2.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)),
            {"front": np.zeros((4, 5, 3), dtype=np.uint8)},
            environment_state=[0.5],
        )
    assert resumed.num_frames == 0
    assert not list(first.root.glob(".episode-staging-*"))
    assert _lock_is_free(config.repo_id)
    resumed.abort("test")


@pytest.mark.parametrize("format_name", [None, "so101-fallback-v1"])
def test_zero_episode_fallback_metadata_is_safely_upgraded_on_first_publish(format_name):
    repo_id = "test/empty_upgrade"
    root = dataset_root(repo_id)
    root.mkdir(parents=True)
    initial = {"repo_id": repo_id, "episodes": 0, "total_frames": 0}
    if format_name is not None:
        initial["format"] = format_name
    (root / "meta.json").write_text(json.dumps(initial), encoding="utf-8")
    config = RecordingConfig(repo_id=repo_id, fps=12, task="Reach.", camera_keys=[])
    recorder = EpisodeRecorder(config)
    recorder.add(TeleopStep(t=1.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)))
    assert recorder.save_episode() == 0

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert meta == {
        **meta,
        "repo_id": repo_id,
        "format": "so101-fallback-v2",
        "fps": 12,
        "task": "Reach.",
        "camera_keys": [],
        "image_storage": "chunked-npy",
        "camera_storage": {},
        "episodes": 1,
        "total_frames": 1,
    }
    assert _lock_is_free(repo_id)


@pytest.mark.parametrize(
    "artifact",
    [
        "episode_000000/frames.jsonl",
        "incomplete_1/INCOMPLETE.json",
        ".episode-staging-dead/frames.jsonl",
    ],
)
def test_zero_episode_metadata_with_data_artifacts_is_not_treated_as_empty(artifact):
    repo_id = "test/false_empty"
    root = dataset_root(repo_id)
    path = root / artifact
    path.parent.mkdir(parents=True)
    path.write_text("partial", encoding="utf-8")
    (root / "meta.json").write_text(
        json.dumps({"repo_id": repo_id, "episodes": 0, "total_frames": 0}),
        encoding="utf-8",
    )
    before = {
        item.relative_to(root): item.read_bytes() for item in root.rglob("*") if item.is_file()
    }

    with pytest.raises(ValueError, match="zero episodes.*artifacts"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))

    after = {
        item.relative_to(root): item.read_bytes() for item in root.rglob("*") if item.is_file()
    }
    assert after == before
    assert _lock_is_free(repo_id)


def test_zero_episode_upgrade_still_rejects_an_explicit_conflict():
    repo_id = "test/empty_conflict"
    root = dataset_root(repo_id)
    root.mkdir(parents=True)
    path = root / "meta.json"
    path.write_text(json.dumps({"repo_id": repo_id, "episodes": 0, "fps": 15}), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="FPS"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=30, camera_keys=[]))
    assert path.read_bytes() == original
    assert _lock_is_free(repo_id)


def test_unknown_future_fallback_format_is_refused_without_mutation():
    repo_id = "test/future_fallback"
    root = dataset_root(repo_id)
    root.mkdir(parents=True)
    path = root / "meta.json"
    path.write_text(
        json.dumps({"repo_id": repo_id, "format": "so101-fallback-v3", "episodes": 0}),
        encoding="utf-8",
    )
    original = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported fallback dataset format"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))
    assert path.read_bytes() == original
    assert _lock_is_free(repo_id)


def test_populated_missing_format_and_camera_layout_metadata_is_inferred_then_upgraded():
    config, first = _fallback_layout_dataset()
    path = first.root / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "format",
        "total_frames",
        "camera_shapes",
        "camera_dtypes",
        "camera_depth_keys",
    ):
        meta.pop(key, None)
    path.write_text(json.dumps(meta), encoding="utf-8")

    recorder = EpisodeRecorder(config)
    recorder.add(
        TeleopStep(t=2.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)),
        {"front": np.ones((2, 3, 3), dtype=np.uint8)},
        environment_state=[0.5],
    )
    assert recorder.save_episode() == 1
    upgraded = json.loads(path.read_text(encoding="utf-8"))
    assert upgraded["format"] == "so101-fallback-v2"
    assert upgraded["total_frames"] == 2
    assert upgraded["camera_shapes"] == {"front": [2, 3, 3]}
    assert upgraded["camera_dtypes"] == {"front": "uint8"}
    assert upgraded["camera_depth_keys"] == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 10.0, "FPS"),
        ("fps", "10", "FPS"),
        ("fps", True, "FPS"),
        ("fps", float("nan"), "FPS"),
        ("fps", float("inf"), "FPS"),
        ("episodes", 1.0, "episode count"),
        ("episodes", "1", "episode count"),
        ("episodes", True, "episode count"),
        ("episodes", -1, "episode count"),
        ("total_frames", 1.0, "total frame count"),
        ("total_frames", "1", "total frame count"),
        ("total_frames", True, "total frame count"),
        ("total_frames", -1, "total frame count"),
    ],
)
def test_fallback_metadata_counters_require_exact_non_boolean_integers(field, value, message):
    config, first = _fallback_layout_dataset()
    path = first.root / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta[field] = value
    path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        EpisodeRecorder(config)
    assert _lock_is_free(config.repo_id)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("gap", "episode indexes"),
        ("index_type", "episode index"),
        ("frame_type", "frame count"),
        ("frame_file", "frames.jsonl has"),
        ("total", "total frame count"),
    ],
)
def test_fallback_append_requires_contiguous_typed_episode_integrity(damage, message):
    config, first = _fallback_layout_dataset()
    root = first.root
    meta_path = root / "meta.json"
    episode = root / "episode_000000"
    manifest_path = episode / "episode.json"
    if damage == "gap":
        renamed = root / "episode_000001"
        episode.rename(renamed)
        moved_manifest = renamed / "episode.json"
        manifest = json.loads(moved_manifest.read_text(encoding="utf-8"))
        manifest["episode"] = 1
        moved_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage in {"index_type", "frame_type"}:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["episode" if damage == "index_type" else "frames"] = "1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "frame_file":
        with (episode / "frames.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["total_frames"] += 1
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        EpisodeRecorder(config)
    assert _lock_is_free(config.repo_id)


def test_fallback_camera_dtype_and_depth_layout_persists_and_legacy_infers():
    color, depth = "front", "front#depth"
    config = RecordingConfig(repo_id="test/fallback_depth", camera_keys=[color, depth])
    images = {
        color: np.zeros((2, 3, 3), dtype=np.uint8),
        depth: np.zeros((2, 3, 1), dtype=np.uint16),
    }
    first = EpisodeRecorder(config)
    first.add(TeleopStep(t=1.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)), images)
    first.save_episode()
    path = first.root / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["camera_dtypes"] == {color: "uint8", depth: "uint16"}
    assert meta["camera_depth_keys"] == [depth]

    # A dataset recorded before these metadata fields existed is inferred from
    # its first published chunks, then upgraded on the next publication.
    meta.pop("camera_dtypes")
    meta.pop("camera_depth_keys")
    path.write_text(json.dumps(meta), encoding="utf-8")
    resumed = EpisodeRecorder(config)
    resumed.add(TeleopStep(t=2.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)), images)
    resumed.save_episode()
    upgraded = json.loads(path.read_text(encoding="utf-8"))
    assert upgraded["camera_dtypes"] == {color: "uint8", depth: "uint16"}
    assert upgraded["camera_depth_keys"] == [depth]


@pytest.mark.parametrize(
    ("camera", "dtype", "message"),
    [
        ("front", np.float32, "uint8"),
        ("front#depth", np.uint8, "uint16"),
    ],
)
def test_fallback_camera_dtype_or_depth_mismatch_refuses_and_releases(camera, dtype, message):
    color, depth = "front", "front#depth"
    config = RecordingConfig(repo_id="test/fallback_dtype_mismatch", camera_keys=[color, depth])
    good = {
        color: np.zeros((2, 3, 3), dtype=np.uint8),
        depth: np.zeros((2, 3, 1), dtype=np.uint16),
    }
    first = EpisodeRecorder(config)
    first.add(TeleopStep(t=1.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)), good)
    first.save_episode()
    bad = dict(good)
    bad[camera] = np.zeros(good[camera].shape, dtype=dtype)

    resumed = EpisodeRecorder(config)
    with pytest.raises(ValueError, match=message):
        resumed.add(TeleopStep(t=2.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)), bad)
    assert not list(first.root.glob(".episode-staging-*"))
    assert _lock_is_free(config.repo_id)


@pytest.mark.parametrize("damage", ["corrupt", "conflicting_shape", "conflicting_dtype"])
def test_legacy_camera_layout_inference_fails_closed_on_bad_or_conflicting_chunks(damage):
    config = RecordingConfig(repo_id="test/fallback_chunk_conflict", camera_keys=["front"])
    image = {"front": np.zeros((2, 3, 3), dtype=np.uint8)}
    for timestamp in (1.0, 2.0):
        recorder = EpisodeRecorder(config)
        recorder.add(
            TeleopStep(
                t=timestamp,
                observation=_joint_dict(0.0),
                action=_joint_dict(1.0),
            ),
            image,
        )
        recorder.save_episode()
    root = dataset_root(config.repo_id)
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("camera_shapes", "camera_dtypes", "camera_depth_keys"):
        meta.pop(key, None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    chunks = sorted(root.glob("episode_*/images/00_front/chunk_*.npy"))
    assert len(chunks) == 2
    if damage == "corrupt":
        chunks[0].write_bytes(b"not numpy")
    elif damage == "conflicting_shape":
        np.save(chunks[1], np.zeros((1, 4, 3, 3), dtype=np.uint8))
    else:
        np.save(chunks[1], np.zeros((1, 2, 3, 3), dtype=np.float32))

    resumed = EpisodeRecorder(config)
    with pytest.raises(ValueError, match="camera chunks"):
        resumed.add(
            TeleopStep(t=3.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)),
            image,
        )
    assert _lock_is_free(config.repo_id)


def test_camera_device_paths_become_lerobot_safe_feature_names():
    camera = "/dev/v4l/by-id/usb-camera"
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/camera_feature", camera_keys=[camera]))
    features = recorder._features({camera: np.zeros((2, 3, 3), dtype=np.uint8)})
    image_keys = [key for key in features if key.startswith("observation.images.")]
    assert image_keys == ["observation.images.camera_00_dev_v4l_by-id_usb-camera"]
    assert "/" not in image_keys[0]


def test_fallback_abort_preserves_incomplete_episode_without_publishing():
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/incomplete", camera_keys=[]))
    recorder.add(
        TeleopStep(
            t=1.0,
            observation=_joint_dict(0.0),
            action=_joint_dict(0.0),
        )
    )
    preserved = recorder.abort("operator stopped")

    assert preserved is not None
    assert (preserved / "frames.jsonl").is_file()
    manifest = json.loads((preserved / "INCOMPLETE.json").read_text())
    assert manifest["reason"] == "operator stopped"
    assert manifest["frames"] == 1
    assert not (recorder.root / "meta.json").exists()


def test_valid_first_aborted_fallback_allows_a_fresh_retry():
    repo_id = "test/retry_incomplete"
    config = RecordingConfig(repo_id=repo_id, fps=10, task="Retry me", camera_keys=[])
    first = EpisodeRecorder(config)
    first.add(TeleopStep(t=1.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)))
    incomplete = first.abort("camera unplugged")
    assert incomplete is not None and not (first.root / "meta.json").exists()

    retry = EpisodeRecorder(config)
    retry.add(TeleopStep(t=2.0, observation=_joint_dict(0.0), action=_joint_dict(1.0)))
    assert retry.save_episode() == 0
    assert incomplete.is_dir()
    assert (first.root / "episode_000000" / "episode.json").is_file()
    assert _lock_is_free(repo_id)


@pytest.mark.parametrize(
    "artifact", ["episode_000000/frames.jsonl", ".episode-staging-x/frames.jsonl"]
)
def test_fallback_retry_refuses_orphan_published_or_staging_artifacts(artifact):
    repo_id = "test/orphan_retry"
    root = dataset_root(repo_id)
    path = root / artifact
    path.parent.mkdir(parents=True)
    path.write_text("orphan", encoding="utf-8")
    with pytest.raises(ValueError, match="orphan published or staging"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))
    assert _lock_is_free(repo_id)


def test_real_mode_refuses_mock_checkpoint(tmp_path, monkeypatch):
    from so101.config import get_settings

    checkpoint = tmp_path / "mock_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"mock": True}))
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError, match="refusing to load.*mock"):
        load_policy(checkpoint)


def test_real_policy_load_error_is_not_converted_to_mock(tmp_path, monkeypatch):
    from so101.config import get_settings

    checkpoint = tmp_path / "real_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"type": "act"}))
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()

    def fail(self):
        raise ModuleNotFoundError("lerobot is unavailable")

    monkeypatch.setattr(LeRobotPolicy, "_ensure", fail)
    with pytest.raises(ModuleNotFoundError, match="lerobot is unavailable"):
        load_policy(checkpoint)
