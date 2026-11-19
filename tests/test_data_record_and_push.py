"""CLI recording between episodes (audit B28) and the dataset push request (audit B8)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from so101.cli import app as cli_app
from so101.data.dataset import dataset_root, get_dataset
from so101.data.recorder import EpisodeRecorder
from so101.hardware.cameras import CameraManager
from so101.services.app import create_app
from so101.services.routers import datasets as datasets_router
from so101.teleop.controller import TeleopController


def _invoke(*args: str, input: str = ""):
    result = CliRunner().invoke(cli_app, list(args), input=input)
    return result, " ".join(result.output.split())


def _record(repo_id: str, *extra: str, input: str = ""):
    return _invoke(
        "record", repo_id, "--episode-seconds", "0.05", "--reset-seconds", "0", *extra, input=input
    )


def _episodes(repo_id: str) -> int:
    info = get_dataset(repo_id)
    return 0 if info is None else info.num_episodes


# -- so101 record ----------------------------------------------------------------------


def test_record_asks_before_each_episode_after_the_first_and_can_stop():
    result, text = _record("lab/gate", "--episodes", "3", input="y\nn\n")
    assert result.exit_code == 0, text
    assert text.count("Reset the scene, then record episode") == 2
    assert "2 of 3 episodes saved" in text
    assert _episodes("lab/gate") == 2


def test_record_without_prompts_records_every_episode():
    result, text = _record("lab/auto", "--episodes", "2", "--no-prompt")
    assert result.exit_code == 0, text
    assert "Reset the scene" not in text
    assert _episodes("lab/auto") == 2


def test_record_gives_an_unrecorded_reset_window_between_episodes(monkeypatch):
    durations: list[float | None] = []
    added: list[int] = []
    real_run = TeleopController.run
    real_add = EpisodeRecorder.add

    def run(self, duration=None):
        durations.append(duration)
        return real_run(self, duration)

    def add(self, step, images=None):
        added.append(len(durations))
        return real_add(self, step, images)

    monkeypatch.setattr(TeleopController, "run", run)
    monkeypatch.setattr(EpisodeRecorder, "add", add)
    result, text = _invoke(
        "record", "lab/window", "--episodes", "2", "--episode-seconds", "0.05",
        "--reset-seconds", "0.05", "--no-prompt",
    )  # fmt: skip
    assert result.exit_code == 0, text
    assert "get ready" in text
    assert durations == [0.05, 0.05, 0.05]
    # Frames come from the two recording runs (1st and 3rd), never the reset run.
    assert set(added) == {1, 3}


def test_ctrl_c_mid_take_keeps_saved_episodes_and_sets_the_take_aside(monkeypatch):
    closed: list[bool] = []
    real_close = CameraManager.close
    real_add = EpisodeRecorder.add
    calls = {"n": 0}

    def close(self):
        closed.append(True)
        return real_close(self)

    def add(self, step, images=None):
        if self.cfg.repo_id == "lab/ctrlc" and _episodes("lab/ctrlc") == 1:
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt
        return real_add(self, step, images)

    monkeypatch.setattr(CameraManager, "close", close)
    monkeypatch.setattr(EpisodeRecorder, "add", add)
    result, text = _record("lab/ctrlc", "--episodes", "3", "--no-prompt")
    assert result.exit_code == 130, text
    assert "discarded the unsaved take" in text
    assert "1 of 3 episodes saved" in text
    assert closed, "cameras were not closed"
    assert _episodes("lab/ctrlc") == 1
    root = dataset_root("lab/ctrlc")
    assert len(list(root.glob("incomplete_*"))) == 1
    assert not list(root.glob(".episode-staging-*"))


def test_record_rejects_impossible_options():
    result, text = _invoke("record", "lab/bad", "--episodes", "0")
    assert result.exit_code == 1
    assert "must be positive" in text


# -- POST /api/datasets/{repo}/push ----------------------------------------------------


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def pushes(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def fake_push(repo_id: str, private: bool = True) -> str:
        calls.append((repo_id, private))
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr(datasets_router, "push_to_hub", fake_push)
    return calls


def _heartbeat(operator_id: str = "test-operator") -> None:
    from so101.services.robot_manager import get_manager

    get_manager().operator_heartbeat(operator_id)


def test_push_needs_a_json_body(client, pushes):
    _heartbeat()
    response = client.post("/api/datasets/lab/data/push?private=false")
    assert response.status_code == 422
    assert pushes == []


def test_push_needs_a_live_operator(client, pushes):
    response = client.post("/api/datasets/lab/data/push", json={"operator_id": "nobody"})
    assert response.status_code == 403
    assert "heartbeat" in response.json()["detail"]
    assert pushes == []


def test_push_is_private_unless_the_body_says_otherwise(client, pushes):
    _heartbeat()
    first = client.post(
        "/api/datasets/lab/data/push?private=false", json={"operator_id": "test-operator"}
    )
    second = client.post(
        "/api/datasets/lab/data/push", json={"operator_id": "test-operator", "private": False}
    )
    assert (first.status_code, second.status_code) == (200, 200), first.text
    assert first.json()["detail"].endswith("/lab/data")
    assert pushes == [("lab/data", True), ("lab/data", False)]
