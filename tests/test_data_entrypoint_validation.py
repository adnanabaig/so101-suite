"""Entry points reject missing datasets and models in every mode (audit B17)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from so101 import JOINT_NAMES
from so101.cli import app as cli_app
from so101.config import get_settings
from so101.data.dataset import DatasetNotFoundError
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.services.app import create_app
from so101.teleop.controller import TeleopStep
from so101.training.jobs import TrainingJob
from so101.training.trainer import run_training


def _record(repo_id: str, episodes: int = 1) -> None:
    for _ in range(episodes):
        recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=10, camera_keys=[]))
        for index in range(3):
            joints = {f"{joint}.pos": float(index) for joint in JOINT_NAMES}
            recorder.add(TeleopStep(t=1.0 + index / 10, observation=joints, action=joints))
        recorder.save_episode()


def _mock_checkpoint(tmp_path):
    # Under the outputs dir, so the model registry lists it (API rollouts
    # accept registered models only, audit B16).
    checkpoint = get_settings().outputs_dir / "run" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"policy_type": "act", "mock": True}), encoding="utf-8"
    )
    return checkpoint


def _invoke(*args: str):
    result = CliRunner().invoke(cli_app, list(args))
    return result, " ".join(result.output.split())


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        from so101.services.robot_manager import get_manager

        get_manager().operator_heartbeat("test-operator")
        yield test_client
    from so101.services.robot_manager import get_manager

    get_manager().stop()


def test_cli_run_rejects_missing_model_in_mock_mode():
    result, text = _invoke("run", "/does/not/exist", "--duration", "0.1")
    assert result.exit_code == 1
    assert "model not found" in text
    assert "rollout done" not in text


def test_cli_run_rejects_directory_that_is_not_a_checkpoint(tmp_path):
    result, text = _invoke("run", str(tmp_path), "--duration", "0.1")
    assert result.exit_code == 1
    assert "not a policy checkpoint" in text


def test_cli_run_accepts_real_mock_checkpoint(tmp_path):
    result, text = _invoke("run", str(_mock_checkpoint(tmp_path)), "--duration", "0.05")
    assert result.exit_code == 0, text
    assert "rollout done" in text


def test_cli_eval_rejects_missing_model():
    result, text = _invoke("eval", "/does/not/exist", "--episodes", "1")
    assert result.exit_code == 1
    assert "model not found" in text


def test_cli_train_rejects_missing_dataset_and_unknown_policy(tmp_path):
    out = tmp_path / "never"
    result, text = _invoke(
        "train", "no/such-dataset", "--policy", "act", "--steps", "10", "--output-dir", str(out)
    )
    assert result.exit_code == 1
    assert "not found" in text
    assert not out.exists()

    _record("test/cli_train")
    result, text = _invoke("train", "test/cli_train", "--policy", "nonsense", "--steps", "10")
    assert result.exit_code == 1
    assert "unknown policy" in text


def test_cli_train_mock_runs_against_real_dataset(tmp_path):
    _record("test/cli_mock_train")
    out = tmp_path / "mock_run"
    result, text = _invoke(
        "train", "test/cli_mock_train", "--steps", "20", "--output-dir", str(out)
    )
    assert result.exit_code == 0, text
    assert (out / "checkpoints" / "last" / "pretrained_model" / "config.json").is_file()


def test_cli_hub_dataset_opt_in_is_refused_in_mock_mode():
    result, text = _invoke("train", "lerobot/remote", "--steps", "10", "--hub-dataset")
    assert result.exit_code == 1
    assert "real LeRobot training" in text


def test_cli_replay_missing_episode_and_dataset_are_clean_errors():
    _record("test/cli_replay")
    result, text = _invoke("replay", "test/cli_replay", "--episode", "9")
    assert result.exit_code == 1
    assert "episode 9 not found" in text
    assert "episodes 0..0" in text
    assert "Traceback" not in result.output

    result, text = _invoke("replay", "no/such-dataset")
    assert result.exit_code == 1
    assert "not found" in text


def test_cli_datasets_rm_missing_exits_nonzero():
    result, text = _invoke("datasets", "rm", "no/such-dataset")
    assert result.exit_code == 1
    assert "not found" in text


def test_run_training_refuses_missing_dataset_before_writing(tmp_path):
    out = tmp_path / "out"
    job = TrainingJob(repo_id="no/such", policy="act", steps=10, batch_size=1, output_dir=str(out))
    with pytest.raises(DatasetNotFoundError, match="not found"):
        list(run_training(job))
    assert not out.exists()


def test_training_api_validates_dataset_and_policy(client):
    body = {"repo_id": "no/such", "policy": "act", "steps": 10, "batch_size": 1}
    assert client.post("/api/training/jobs", json=body).status_code == 404

    root = get_settings().datasets_dir / "v2--test%2Fempty"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps({"repo_id": "test/empty", "format": "so101-fallback-v2", "episodes": 0}),
        encoding="utf-8",
    )
    empty = client.post("/api/training/jobs", json={**body, "repo_id": "test/empty"})
    assert empty.status_code == 422
    assert "no saved episodes" in empty.json()["detail"]

    _record("test/api_train")
    unknown = client.post(
        "/api/training/jobs", json={**body, "repo_id": "test/api_train", "policy": "nope"}
    )
    assert unknown.status_code == 400
    bad_steps = client.post(
        "/api/training/jobs", json={**body, "repo_id": "test/api_train", "steps": 0}
    )
    assert bad_steps.status_code == 422


def test_rollout_api_validates_model_before_starting(client, tmp_path):
    body = {"fps": 30, "operator_id": "test-operator"}
    missing = client.post(
        "/api/models/rollout/start", json={**body, "model_path": "/does/not/exist"}
    )
    assert missing.status_code == 404
    unregistered = client.post(
        "/api/models/rollout/start", json={**body, "model_path": str(tmp_path)}
    )
    assert unregistered.status_code == 404
    empty = get_settings().outputs_dir / "empty" / "checkpoints" / "last" / "pretrained_model"
    empty.mkdir(parents=True)
    not_checkpoint = client.post(
        "/api/models/rollout/start", json={**body, "model_path": str(empty)}
    )
    assert not_checkpoint.status_code == 422
    assert "not a policy checkpoint" in not_checkpoint.json()["detail"]
    assert client.get("/api/robot/status").json()["mode"] == "idle"

    started = client.post(
        "/api/models/rollout/start",
        json={**body, "model_path": str(_mock_checkpoint(tmp_path))},
    )
    assert started.status_code == 200, started.text
    assert client.post("/api/models/rollout/stop").status_code == 200


def test_rollout_api_refuses_mock_checkpoint_on_real_backend(client, tmp_path, monkeypatch):
    checkpoint = _mock_checkpoint(tmp_path)
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    try:
        refused = client.post(
            "/api/models/rollout/start",
            json={"model_path": str(checkpoint), "fps": 30, "operator_id": "test-operator"},
        )
    finally:
        monkeypatch.setenv("SO101_MOCK", "1")
        get_settings.cache_clear()
    assert refused.status_code == 422
    assert "mock checkpoint" in refused.json()["detail"]
