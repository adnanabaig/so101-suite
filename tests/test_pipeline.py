"""End-to-end-ish: teleop -> record -> list -> replay -> train -> model."""

from __future__ import annotations

import itertools

import pytest

from so101.data.dataset import list_datasets
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.data.replay import replay_episode
from so101.hardware.cameras import CameraManager
from so101.hardware.follower import SO101Follower
from so101.hardware.leader import SO101Leader
from so101.models import list_models
from so101.policy.runner import PolicyRunner
from so101.teleop.controller import TeleopController
from so101.training.jobs import TrainingJob
from so101.training.trainer import run_training


def test_record_and_list_dataset():
    cams = CameraManager(["0"])
    cams.open()
    cfg = RecordingConfig(repo_id="test/pick", task="pick", fps=30, camera_keys=["0"])
    rec = EpisodeRecorder(cfg)
    with TeleopController(SO101Leader(), SO101Follower(), fps=120) as tc:
        for step in itertools.islice(tc.run(), 10):
            rec.add(step, cams.read())
    idx = rec.save_episode()
    cams.close()
    assert idx == 0
    datasets = list_datasets()
    assert any(d.repo_id == "test/pick" for d in datasets)


def test_replay_fallback_episode():
    cfg = RecordingConfig(repo_id="test/replay", fps=30, camera_keys=[])
    rec = EpisodeRecorder(cfg)
    with TeleopController(SO101Leader(), SO101Follower(), fps=120) as tc:
        for step in itertools.islice(tc.run(), 5):
            rec.add(step)
    rec.save_episode()
    n = replay_episode("test/replay", 0, fps=240)
    assert n == 5


def test_mock_training_produces_model():
    cfg = RecordingConfig(repo_id="test/pick", fps=30, camera_keys=[])
    rec = EpisodeRecorder(cfg)
    with TeleopController(SO101Leader(), SO101Follower(), fps=120) as tc:
        for step in itertools.islice(tc.run(), 5):
            rec.add(step)
    rec.save_episode()
    job = TrainingJob(
        repo_id="test/pick",
        policy="act",
        steps=200,
        batch_size=8,
        output_dir="",
    )
    from so101.config import get_settings

    job.output_dir = str(get_settings().outputs_dir / "act_run")
    final = None
    for step, loss, _ in run_training(job):
        final = (step, loss)
    assert final is not None and final[1] < 2.0  # loss decayed
    models = list_models()
    assert any(m.policy_type == "act" for m in models)


def test_policy_runner_camera_failure_always_deenergizes_follower():
    order: list[str] = []

    class Follower:
        connected = False

        def connect(self):
            self.connected = True
            order.append("follower.connect")

        def disconnect(self):
            self.connected = False
            order.append("follower.disconnect")

    class Cameras:
        def open(self):
            order.append("cameras.open")
            raise KeyboardInterrupt("camera interrupted")

        def close(self):
            order.append("cameras.close")

    class Policy:
        def reset(self):
            order.append("policy.reset")

    runner = PolicyRunner(Policy(), follower=Follower(), cameras=Cameras())
    with pytest.raises(KeyboardInterrupt, match="camera interrupted"):
        runner.connect()
    assert order == [
        "follower.connect",
        "cameras.open",
        "follower.disconnect",
        "cameras.close",
    ]


@pytest.mark.parametrize("fps", [0, -1, 241, True, 30.0])
def test_policy_runner_rejects_an_unsafe_control_rate(fps):
    with pytest.raises(ValueError, match="integer between 1 and 240"):
        PolicyRunner(object(), follower=object(), cameras=object(), fps=fps)


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -0.01])
def test_policy_rollout_rejects_an_unsafe_duration_before_connect(duration):
    class Follower:
        connected = False

    runner = PolicyRunner(object(), follower=Follower(), cameras=object())
    with pytest.raises(ValueError, match="finite and non-negative"):
        runner.rollout(duration)
