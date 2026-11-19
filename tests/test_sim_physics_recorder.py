"""Recorder additions used by simulated demos: environment state and feature names.

They are opt-in: a recording that sets neither writes exactly what it did before.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from so101 import JOINT_NAMES
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.teleop.controller import TeleopStep

ENV_NAMES = ["cube_x", "cube_y", "cube_z"]


def _step(t: float, value: float = 0.0) -> TeleopStep:
    joints = {f"{joint}.pos": value for joint in JOINT_NAMES}
    return TeleopStep(t=t, observation=joints, action=joints, desired_action=joints)


def _frames(recorder: EpisodeRecorder, episode: int = 0) -> list[dict]:
    path = recorder.root / f"episode_{episode:06d}" / "frames.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_environment_state_is_recorded_per_frame_and_named_in_meta():
    recorder = EpisodeRecorder(
        RecordingConfig(repo_id="test/env_state", camera_keys=[], environment_state_names=ENV_NAMES)
    )
    for index in range(3):
        recorder.add(_step(index / 30), environment_state=[0.2 + index, -0.1, 0.02])
    recorder.save_episode()

    frames = _frames(recorder)
    assert [len(frame["observation.environment_state"]) for frame in frames] == [3, 3, 3]
    assert frames[2]["observation.environment_state"] == pytest.approx([2.2, -0.1, 0.02])
    meta = json.loads((recorder.root / "meta.json").read_text(encoding="utf-8"))
    assert meta["environment_state_names"] == ENV_NAMES
    assert "camera_feature_names" not in meta


def test_recordings_without_the_additions_are_unchanged():
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/plain", camera_keys=[]))
    recorder.add(_step(0.0))
    recorder.save_episode()

    assert set(_frames(recorder)[0]) == {"t", "observation.state", "action", "desired_action"}
    meta = json.loads((recorder.root / "meta.json").read_text(encoding="utf-8"))
    assert "environment_state_names" not in meta and "camera_feature_names" not in meta


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (None, "needs an environment_state"),
        ([1.0, 2.0], "shape"),
        ([[1.0, 2.0, 3.0, 4.0]], "shape"),
        ([1.0, math.nan, 0.0], "non-finite"),
        ([1.0, math.inf, 0.0], "non-finite"),
    ],
)
def test_bad_environment_state_is_rejected_before_anything_is_written(values, message):
    recorder = EpisodeRecorder(
        RecordingConfig(repo_id="test/env_bad", camera_keys=[], environment_state_names=ENV_NAMES)
    )
    with pytest.raises(ValueError, match=message):
        recorder.add(_step(0.0), environment_state=values)
    assert recorder.num_frames == 0
    recorder.add(_step(0.0), environment_state=np.zeros(3))
    recorder.save_episode()
    assert len(_frames(recorder)) == 1


def test_environment_state_without_names_is_rejected():
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/env_unnamed", camera_keys=[]))
    with pytest.raises(ValueError, match="environment_state_names is empty"):
        recorder.add(_step(0.0), environment_state=[1.0, 2.0, 3.0])
    recorder.abort("test")


def test_lerobot_features_include_environment_state(monkeypatch):
    class FakeDataset:
        create_kwargs: dict = {}

        def __init__(self):
            self.frames = []

        @classmethod
        def create(cls, **kwargs):
            cls.create_kwargs = kwargs
            return cls()

        def add_frame(self, frame):
            self.frames.append(frame)

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: FakeDataset)
    recorder = EpisodeRecorder(
        RecordingConfig(
            repo_id="test/env_lerobot", camera_keys=[], environment_state_names=ENV_NAMES
        )
    )
    recorder.add(_step(0.0), environment_state=[0.1, 0.2, 0.3])

    feature = FakeDataset.create_kwargs["features"]["observation.environment_state"]
    assert feature == {"dtype": "float32", "shape": (3,), "names": ENV_NAMES}
    value = recorder._lerobot_ds.frames[0]["observation.environment_state"]
    assert value.dtype == np.float32 and value.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_camera_feature_names_can_be_set_per_camera():
    cameras = ["front", "rtsp://cam1/stream"]  # an id that is not a safe name
    recorder = EpisodeRecorder(
        RecordingConfig(
            repo_id="test/feature_names",
            camera_keys=cameras,
            camera_feature_names={"front": "front"},
        )
    )
    images = {camera: np.zeros((4, 6, 3), dtype=np.uint8) for camera in cameras}
    keys = [key for key in recorder._features(images) if key.startswith("observation.images.")]
    assert keys == ["observation.images.front", "observation.images.camera_01_rtsp_cam1_stream"]

    recorder.add(_step(0.0), images)
    recorder.save_episode()
    meta = json.loads((recorder.root / "meta.json").read_text(encoding="utf-8"))
    assert meta["camera_feature_names"] == {
        "front": "front",
        "rtsp://cam1/stream": "camera_01_rtsp_cam1_stream",
    }
    assert meta["camera_storage"] == {
        "front": "00_front",
        "rtsp://cam1/stream": "01_rtsp_cam1_stream",
    }


@pytest.mark.parametrize(
    "names",
    [
        {"front": "front.left"},  # not LeRobot-safe
        {"front": "x" * 81},
        {"front": "same", "wrist_cam": "same"},
        {"front": "camera_01_wrist_cam"},  # collides with wrist_cam's default name
    ],
)
def test_bad_camera_feature_names_are_rejected_and_release_the_dataset(names):
    cfg = RecordingConfig(
        repo_id="test/feature_bad", camera_keys=["front", "wrist_cam"], camera_feature_names=names
    )
    with pytest.raises(ValueError, match="camera feature names"):
        EpisodeRecorder(cfg)
    # The dataset lock was released, so a valid recorder can start right away.
    EpisodeRecorder(RecordingConfig(repo_id="test/feature_bad", camera_keys=[])).abort("test")
