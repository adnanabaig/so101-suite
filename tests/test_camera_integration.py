"""Cameras across the suite: credentials, depth in recordings, training and rollouts."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from so101 import JOINT_NAMES
from so101.config import get_settings
from so101.data import dataset as dataset_module
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.hardware import camera_discovery as discovery_module
from so101.hardware import cameras as cameras_module
from so101.hardware.camera_spec import stream_feature_names
from so101.hardware.cameras import CameraManager, CameraUnavailableError
from so101.policy.runner import _build_observation
from so101.teleop.controller import TeleopStep
from so101.training.jobs import TrainingJob
from so101.training.trainer import build_train_command, policy_inputs_without_depth

SECRET = "s3cretPW"
CREDENTIALED = f"lab=rtsp://admin:{SECRET}@10.0.0.5:554/stream1,rtsp://viewer:{SECRET}@10.0.0.6/s"
NAMES = [f"{joint}.pos" for joint in JOINT_NAMES]


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _step(value: float = 1.0) -> TeleopStep:
    joints = dict.fromkeys(NAMES, value)
    return TeleopStep(t=1.0 + value, observation=joints, action=joints)


# --- credentials ----------------------------------------------------------------------------------


def test_no_credential_reaches_keys_feature_names_or_info(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, SO101_CAMERAS=CREDENTIALED)
    assert SECRET in settings.cameras  # the setting itself is the operator's to keep
    streams = settings.camera_streams
    assert streams == ["lab=rtsp://10.0.0.5:554/stream1", "rtsp://10.0.0.6/s"]
    names = stream_feature_names(streams)
    assert names == {streams[0]: "lab", streams[1]: "camera_01_rtsp_10.0.0.6_s"}
    manager = CameraManager()
    frames = manager.read()
    blob = json.dumps(
        {
            "streams": manager.streams,
            "frames": list(frames),
            "names": manager.feature_names,
            "describe": manager.describe(),
            "specs": [repr(spec) for spec in manager.specs],
        }
    )
    assert SECRET not in blob and "admin" not in blob and "viewer" not in blob
    manager.close()


def test_open_errors_redact_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, SO101_MOCK="0", SO101_CAMERAS=CREDENTIALED)

    class NeverOpens:
        def __init__(self, source: Any) -> None:
            self.source = source

        def isOpened(self) -> bool:  # noqa: N802 - cv2 spelling
            return False

    cv2 = type(json)("cv2")
    cv2.VideoCapture = NeverOpens  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "cv2", cv2)
    with pytest.raises(CameraUnavailableError) as caught:
        CameraManager().open()
    assert SECRET not in str(caught.value) and "rtsp://***@10.0.0.5" in str(caught.value)


def test_settings_errors_do_not_echo_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(Exception) as caught:
        _settings(monkeypatch, SO101_CAMERAS=f"rtsp://admin:{SECRET}@h/s+depth")
    assert SECRET not in str(caught.value) and "rtsp://***@h/s+depth" in str(caught.value)


def test_status_diagnostics_events_and_logs_carry_no_credentials(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from so101.services.app import create_app
    from so101.services.robot_manager import get_manager, reset_manager
    from so101.services.store import get_store

    _settings(monkeypatch, SO101_CAMERAS=CREDENTIALED)
    reset_manager()
    original = CameraManager._open_one

    def real_mode_open(self: CameraManager, cid: str):
        # Mock bus, but the cameras take the real path and fail to open.
        self.force_mock = False
        return original(self, cid)

    monkeypatch.setattr(CameraManager, "_open_one", real_mode_open)
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    try:
        with caplog.at_level(logging.INFO), TestClient(create_app()) as client:
            manager = get_manager()
            manager.operator_heartbeat("op")
            manager.start_teleop(30, operator_id="op")
            deadline = time.monotonic() + 5
            while manager.status()["mode"] != "idle" and time.monotonic() < deadline:
                time.sleep(0.01)
            status = client.get("/api/robot/status").json()
            diagnostics = client.get("/api/diagnostics").text
        events = json.dumps(get_store().list_events(limit=100))
        assert status["cameras"] == ["lab=rtsp://10.0.0.5:554/stream1", "rtsp://10.0.0.6/s"]
        assert "camera" in events  # the failure was recorded...
        for text in (json.dumps(status), diagnostics, events, caplog.text):
            assert SECRET not in text  # ...without the password
    finally:
        reset_manager()


def test_preflight_and_info_carry_no_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from so101 import cli
    from so101.bringup import preflight as pf

    settings = _settings(monkeypatch, SO101_CAMERAS=CREDENTIALED)
    probes = pf.probe_cameras_locally(list(settings.camera_ids), frames=1)
    assert [probe.camera for probe in probes] == settings.camera_streams
    assert SECRET not in json.dumps([probe.__dict__ for probe in probes])
    cli.info()
    assert SECRET not in capsys.readouterr().out


# --- recording -----------------------------------------------------------------------------------


def test_recorder_names_depth_and_keeps_it_uint16(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = "wrist=realsense:1234:64x48+depth"
    manager = CameraManager(["0", entry])
    frames = manager.read()
    recorder = EpisodeRecorder(
        RecordingConfig(repo_id="test/depth_rec", camera_keys=manager.streams)
    )
    features = recorder._features(frames)
    assert features["observation.images.wrist_depth"]["info"] == {"is_depth_map": True}
    assert features["observation.images.wrist_depth"]["shape"] == (48, 64, 1)
    assert "info" not in features["observation.images.wrist"]
    assert "observation.images.camera_00_0" in features
    for _ in range(3):
        recorder.add(_step(), manager.read())
    recorder.save_episode()
    chunks = list((recorder.root / "episode_000000").rglob("*.npy"))
    depth_chunk = [np.load(path) for path in chunks if np.load(path).shape[-1] == 1]
    assert depth_chunk and depth_chunk[0].dtype == np.uint16
    info = dataset_module.get_dataset("test/depth_rec")
    assert info is not None and info.depth_keys == [entry + "#depth"]
    assert entry + "#depth" not in info.camera_keys


def test_a_missing_video_encoder_gets_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoHevc:
        @classmethod
        def create(cls, **_kwargs: Any) -> None:
            raise ValueError("Unsupported video codec: hevc with video backend pyav")

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: NoHevc)
    # With no camera the recorder opens its dataset at once.
    with pytest.raises(RuntimeError, match="could not set up its video encoders") as caught:
        EpisodeRecorder(RecordingConfig(repo_id="test/no_hevc", camera_keys=[]))
    assert "libx265" in str(caught.value) and "av>=15,<16" in str(caught.value)


def test_other_value_errors_from_create_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        @classmethod
        def create(cls, **_kwargs: Any) -> None:
            raise ValueError("features are malformed")

    monkeypatch.setattr(EpisodeRecorder, "_find_lerobot_dataset", lambda self: Broken)
    with pytest.raises(ValueError, match="features are malformed"):
        EpisodeRecorder(RecordingConfig(repo_id="test/broken", camera_keys=[]))


def test_lerobot_dataset_info_separates_depth_maps(tmp_path: Path) -> None:
    root = tmp_path / "lab" / "pick"
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3], "names": []},
        "observation.images.wrist_depth": {
            "dtype": "video",
            "shape": [480, 640, 1],
            "names": [],
            "info": {"is_depth_map": True},
        },
    }
    info = {"repo_id": "lab/pick", "total_episodes": 1, "total_frames": 2, "fps": 30}
    (root / "meta" / "info.json").write_text(json.dumps({**info, "features": features}))
    parsed = dataset_module._read_lerobot(root)
    assert parsed is not None
    assert parsed.camera_keys == ["wrist"] and parsed.depth_keys == ["wrist_depth"]


# --- rollouts ------------------------------------------------------------------------------------


class _Follower:
    def read_positions(self) -> dict[str, float]:
        return dict.fromkeys(JOINT_NAMES, 0.0)


def test_rollout_observation_indexes_depth_by_its_camera() -> None:
    frames = {
        "1": np.zeros((2, 2, 3)),
        "realsense:2222+depth": np.zeros((2, 2, 3)),
        "realsense:2222+depth#depth": np.zeros((2, 2, 1)),
    }
    obs = _build_observation(_Follower(), frames)  # type: ignore[arg-type]
    assert sorted(obs) == [
        "observation.images.camera_00_1",
        "observation.images.camera_01_realsense_2222",
        "observation.images.camera_01_realsense_2222_depth",
        "observation.state",
    ]


def test_rollout_feeds_a_policy_only_what_it_declares() -> None:
    frames = {
        "w=realsense:1111+depth": np.zeros((2, 2, 3)),
        "w=realsense:1111+depth#depth": np.zeros((2, 2, 1)),
    }
    policy = SimpleNamespace(
        input_feature_keys=frozenset({"observation.state", "observation.images.w"})
    )
    obs = _build_observation(_Follower(), frames, policy=policy)  # type: ignore[arg-type]
    assert sorted(obs) == ["observation.images.w", "observation.state"]
    undeclared = SimpleNamespace(input_feature_keys=None)
    assert "observation.images.w_depth" in _build_observation(
        _Follower(), frames, policy=undeclared
    )  # type: ignore[arg-type]
    named = _build_observation(
        _Follower(), {"front": np.zeros((2, 2, 3))}, feature_names={"front": "front"}
    )  # type: ignore[arg-type]
    assert "observation.images.front" in named


def test_lerobot_policy_declares_its_input_features() -> None:
    from so101.policy.server import LeRobotPolicy

    policy = LeRobotPolicy.__new__(LeRobotPolicy)
    policy._policy = SimpleNamespace(
        config=SimpleNamespace(input_features={"observation.state": 1})
    )
    assert policy.input_feature_keys == frozenset({"observation.state"})
    policy._policy = SimpleNamespace(config=SimpleNamespace(input_features={}))
    assert policy.input_feature_keys is None


# --- training ------------------------------------------------------------------------------------


def _lerobot_dataset(root: Path, *, depth: bool) -> Path:
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": NAMES},
        "observation.environment_state": {"dtype": "float32", "shape": [3], "names": None},
        "action": {"dtype": "float32", "shape": [6], "names": NAMES},
        "desired_action": {"dtype": "float32", "shape": [6], "names": NAMES},
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    if depth:
        features["observation.images.wrist_depth"] = {
            "dtype": "video",
            "shape": [480, 640, 1],
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": features}))
    return root


def test_policy_inputs_leave_out_depth_maps(tmp_path: Path) -> None:
    root = _lerobot_dataset(tmp_path / "d", depth=True)
    assert policy_inputs_without_depth(root) == {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.environment_state": {"type": "ENV", "shape": [3]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
    }
    assert policy_inputs_without_depth(_lerobot_dataset(tmp_path / "c", depth=False)) is None
    assert policy_inputs_without_depth(tmp_path / "missing") is None
    (tmp_path / "bad" / "meta").mkdir(parents=True)
    (tmp_path / "bad" / "meta" / "info.json").write_text("[]")
    assert policy_inputs_without_depth(tmp_path / "bad") is None


def test_train_command_drops_depth_unless_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lerobot_dataset(tmp_path / "pick", depth=True)
    from so101.training import trainer

    monkeypatch.setattr(
        trainer, "get_dataset", lambda repo_id: SimpleNamespace(format="lerobot", path=str(root))
    )
    # "all" reads no outcome labels (the stand-in dataset above has none).
    job = TrainingJob(
        repo_id="lab/pick", policy="act", steps=10, batch_size=2, output_dir="/o", episodes="all"
    )
    flag = [arg for arg in build_train_command(job) if arg.startswith("--policy.input_features=")]
    assert len(flag) == 1
    parsed = json.loads(flag[0].split("=", 1)[1])
    assert "observation.images.wrist_depth" not in parsed and "observation.images.wrist" in parsed
    job.include_depth = True
    assert not any("input_features" in arg for arg in build_train_command(job))
    job.include_depth = False
    job.extra_args = {"policy.input_features": "{}"}
    assert sum("input_features" in arg for arg in build_train_command(job)) == 1  # the user's


def test_cli_include_depth_is_for_lerobot_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from so101.cli import app

    result = CliRunner().invoke(
        app, ["train", "demo/x", "--policy", "bc_baseline", "--include-depth"]
    )
    assert result.exit_code == 1 and "--include-depth applies to LeRobot presets" in result.output


# --- CLI ----------------------------------------------------------------------------------------


def test_find_cameras_command_prints_text_or_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from so101.cli import app

    _settings(monkeypatch, SO101_MOCK="0", SO101_BACKEND="real")
    monkeypatch.setitem(__import__("sys").modules, "pyrealsense2", None)
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    monkeypatch.setattr(cameras_module, "get_settings", get_settings)
    text = CliRunner().invoke(app, ["find-cameras"])
    assert text.exit_code == 0 and "OpenCV indices 0..9" in text.output
    as_json = CliRunner().invoke(app, ["find-cameras", "--json"])
    report = json.loads(as_json.output)
    assert as_json.exit_code == 0 and report["suggestion"] is None and "realsense_error" in report


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_find_cameras_refuses_non_real_backends_without_probing(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    from typer.testing import CliRunner

    from so101.cli import EXIT_REFUSED, app

    _settings(monkeypatch, SO101_MOCK="0", SO101_BACKEND=backend)

    def physical_probe_was_called() -> None:
        raise AssertionError("find-cameras probed physical devices")

    monkeypatch.setattr(discovery_module, "discover_cameras", physical_probe_was_called)
    result = CliRunner().invoke(app, ["find-cameras", "--json"])
    assert result.exit_code == EXIT_REFUSED
    assert f"{backend} backend" in result.output
    assert "probes physical USB/video devices" in result.output
