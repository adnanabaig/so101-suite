"""Modern LeRobot dataset read/push adapters without the heavy optional stack."""

from __future__ import annotations

import json
import os
import sys
import time
from types import ModuleType

import pytest

from so101 import JOINT_NAMES
from so101.data.dataset import (
    DatasetBusyError,
    dataset_root,
    delete_dataset,
    push_to_hub,
)
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.data.replay import _load_lerobot_actions
from so101.training.jobs import TrainingJob
from so101.training.trainer import build_train_command


class FakeLeRobotDataset:
    pushed: list[bool] = []

    def __init__(self, repo_id, root=None, episodes=None, token=None, revision=None):
        self.repo_id = repo_id
        self.revision = revision
        self.root = root
        self.episodes = episodes
        self.token = token
        self.features = {"action": {"names": [f"{joint}.pos" for joint in reversed(JOINT_NAMES)]}}
        self.num_frames = 2
        self.fps = 30

    def select_columns(self, name):
        assert name == "action"
        return [
            {"action": [float(value) for value in reversed(range(6))]},
            {"action": [float(value + 10) for value in reversed(range(6))]},
        ]

    def push_to_hub(self, private=None):
        assert os.environ["HF_TOKEN"] == "test-token"
        self.pushed.append(bool(private))


def _install_fake_lerobot(monkeypatch, dataset_cls=FakeLeRobotDataset) -> None:
    package = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    datasets.LeRobotDataset = dataset_cls
    package.datasets = datasets
    monkeypatch.setitem(sys.modules, "lerobot", package)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)


def _local_lerobot_metadata(root, episodes: int) -> None:
    """The metadata a local read needs before LeRobot is constructed (audit B27)."""

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"")
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "total_frames": 2 * episodes, "fps": 30}),
        encoding="utf-8",
    )


def test_lerobot_replay_reorders_named_actions(monkeypatch, tmp_path):
    _install_fake_lerobot(monkeypatch)
    _local_lerobot_metadata(tmp_path, episodes=4)
    actions, fps = _load_lerobot_actions(tmp_path, "test/replay", 3)

    assert fps == 30
    assert actions[0] == [float(value) for value in range(6)]
    assert actions[1] == [float(value + 10) for value in range(6)]


def test_explicit_replay_fps_overrides_raw_nan_from_fake_lerobot(monkeypatch):
    from so101.data import replay

    class NanFPSLeRobotDataset(FakeLeRobotDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fps = float("nan")

    class Clock:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

        @classmethod
        def sleep(cls, seconds):
            cls.now += seconds

    class Follower:
        def __init__(self):
            self.sent = []

        def send_goal(self, goal):
            self.sent.append(goal)

    _install_fake_lerobot(monkeypatch, NanFPSLeRobotDataset)
    repo_id = "test/replay_nan_fps"
    _local_lerobot_metadata(dataset_root(repo_id), episodes=1)
    monkeypatch.setattr(replay, "time", Clock)
    follower = Follower()

    assert replay.replay_episode(repo_id, 0, follower=follower, fps=20) == 2
    assert len(follower.sent) == 2


def test_push_uses_local_dataset_token_and_restores_environment(monkeypatch):
    from so101.config import get_settings

    _install_fake_lerobot(monkeypatch)
    settings = get_settings()
    root = settings.datasets_dir / "test__hub"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "repo_id": "test/hub",
                "total_episodes": 1,
                "total_frames": 2,
                "fps": 30,
                "features": {"action": {"names": [f"{j}.pos" for j in JOINT_NAMES]}},
            }
        )
    )
    monkeypatch.setenv("SO101_HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_TOKEN", "previous-token")
    get_settings.cache_clear()
    FakeLeRobotDataset.pushed.clear()

    url = push_to_hub("test/hub", private=True)

    assert url == "https://huggingface.co/datasets/test/hub"
    assert FakeLeRobotDataset.pushed == [True]
    assert os.environ["HF_TOKEN"] == "previous-token"
    get_settings.cache_clear()


def test_training_command_uses_local_lerobot_root():
    from so101.config import get_settings

    root = get_settings().datasets_dir / "test__local_train"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "repo_id": "test/local_train",
                "total_episodes": 1,
                "total_frames": 2,
                "fps": 30,
                "features": {},
            }
        )
    )
    job = TrainingJob(
        repo_id="test/local_train",
        policy="act",
        steps=10,
        batch_size=1,
        output_dir=str(get_settings().outputs_dir / "local_train"),
    )

    command = build_train_command(job)

    assert f"--dataset.root={root}" in command


@pytest.mark.parametrize("repo_id", ["..", "../escape", "org/../escape", "/absolute"])
def test_dataset_repo_id_cannot_escape_storage(repo_id):
    with pytest.raises(ValueError, match="repo_id"):
        EpisodeRecorder(RecordingConfig(repo_id=repo_id))


def test_valid_repo_ids_map_to_collision_free_storage():
    roots = {
        dataset_root("a/b"),
        dataset_root("a__b"),
        dataset_root("a/b__c"),
        dataset_root("a__b/c"),
    }
    assert len(roots) == 4
    assert all(root.parent == dataset_root("single").parent for root in roots)


def test_legacy_flattened_directory_cannot_capture_colliding_repo_id():
    from so101.config import get_settings

    legacy = get_settings().datasets_dir / "a__b"
    legacy.mkdir(parents=True)
    (legacy / "meta.json").write_text(
        json.dumps({"repo_id": "a/b", "format": "so101-fallback-v1", "episodes": 0})
    )
    assert dataset_root("a/b") == legacy
    assert dataset_root("a__b") != legacy
    assert dataset_root("a__b").name.startswith("v2--")


def test_dataset_delete_moves_to_recoverable_trash():
    from so101.config import get_settings

    root = get_settings().datasets_dir / "test__bin"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "repo_id": "test/bin",
                "format": "so101-fallback-v2",
                "episodes": 0,
            }
        )
    )

    assert delete_dataset("test/bin") is True
    assert not root.exists()
    assert len(list((get_settings().datasets_dir / ".trash").glob("test__bin.*"))) == 1


def test_dataset_delete_is_rejected_while_recorder_owns_it():
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/busy", camera_keys=[]))
    with pytest.raises(DatasetBusyError, match="currently in use"):
        delete_dataset("test/busy")
    recorder.abort("test complete")


def test_rejected_training_job_cannot_release_recorders_dataset_lock():
    from so101.config import get_settings
    from so101.services.jobs import JobRunner

    root = get_settings().datasets_dir / "test__busy_train"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "repo_id": "test/busy_train",
                "format": "so101-fallback-v2",
                "fps": 30,
                "task": "Perform the task.",
                "camera_keys": [],
                "image_storage": "chunked-npy",
                "camera_storage": {},
                "episodes": 0,
            }
        )
    )
    recorder = EpisodeRecorder(RecordingConfig(repo_id="test/busy_train", camera_keys=[]))
    runner = JobRunner()
    job = runner.submit("test/busy_train", "act", 1, 1)
    deadline = time.monotonic() + 2
    while job.status.value in {"pending", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.shutdown()
    assert job.status.value == "failed"
    with pytest.raises(DatasetBusyError):
        delete_dataset("test/busy_train")
    recorder.abort("test complete")
