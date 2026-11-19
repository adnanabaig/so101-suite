"""Policies and recordings on ``SO101_BACKEND=sim`` see the simulator's object state.

- A privileged-state baseline (trained on ``observation.environment_state``)
  runs on the simulated arm (``so101 run``): the follower hands the policy the
  task's object state.
- Recording through the robot manager (the UI and API path) resets the scene
  for every episode, records the object state with each frame and labels the
  episode with the task's success predicate, as ``so101 sim demos`` does.

Lockstep physics (``SO101_SIM_REALTIME=0``), no simulated cameras.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from so101.config import get_settings  # noqa: E402
from so101.services.robot_manager import reset_manager  # noqa: E402
from so101.simulation import devices  # noqa: E402


@pytest.fixture
def sim(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", "sim")
    monkeypatch.setenv("SO101_SIM_REALTIME", "0")
    monkeypatch.setenv("SO101_SIM_TASK", "reach")
    monkeypatch.setenv("SO101_SIM_SEED", "3")
    monkeypatch.setenv("SO101_SIM_CAMERAS", "")
    get_settings.cache_clear()
    reset_manager()
    devices.reset_sim_runtime()
    yield
    reset_manager()
    devices.reset_sim_runtime()


# --- M9: a privileged-state baseline runs on the simulated arm ----------------------


@pytest.fixture(scope="module")
def privileged_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny privileged-state baseline trained by bc_baseline on the toy reach task."""

    from so101.policy.baseline import save_baseline_checkpoint
    from so101.training import toy_reach
    from so101.training.bc_baseline import BCConfig, fit_bc

    episodes = toy_reach.generate_toy_reach_episodes(8, seed=0)
    config = BCConfig(steps=60, batch_size=64, seed=0, hidden=(16, 16), eval_every=20)
    result = fit_bc(toy_reach.toy_training_data(episodes), config)
    assert result.model.spec.privileged_state
    root = tmp_path_factory.mktemp("bc") / "pretrained_model"
    return save_baseline_checkpoint(root, result.model)


def test_the_sim_follower_reports_the_tasks_object_state(sim):
    from so101.hardware.follower import SO101Follower

    follower = SO101Follower()
    follower.connect(enable_torque=False)
    try:
        state = follower.environment_state()
        expected = devices.get_sim_runtime().observation()["observation.environment_state"]
    finally:
        follower.disconnect()
    assert state.dtype == np.float32 and state.shape == (3,)  # reach: target_x, _y, _z
    np.testing.assert_array_equal(state, expected)


def test_a_mock_follower_has_no_object_state():
    from so101.hardware.follower import SO101Follower
    from so101.policy.runner import _build_observation

    follower = SO101Follower()
    follower.connect(enable_torque=False)
    try:
        assert follower.environment_state() is None
        assert "observation.environment_state" not in _build_observation(follower, {})
    finally:
        follower.disconnect()


def test_so101_run_drives_the_simulated_arm_with_a_privileged_baseline(
    sim, privileged_checkpoint: Path
):
    from typer.testing import CliRunner

    from so101.cli import app

    result = CliRunner().invoke(app, ["run", str(privileged_checkpoint), "--duration", "0.3"])
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output
    assert "privileged-state sim baseline" in result.output
    assert "rollout done" in result.output


# --- M8: recording through the robot manager under sim ------------------------------

OPERATOR = "sim-recorder"
REACH_ENV = ["target_x", "target_y", "target_z"]


def _wait_for(predicate, timeout_s: float = 20.0) -> bool:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _frames(repo_id: str, episode: int) -> list[dict]:
    import json

    from so101.data.dataset import dataset_root

    path = dataset_root(repo_id) / f"episode_{episode:06d}" / "frames.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _episode_manifest(repo_id: str, episode: int) -> dict:
    import json

    from so101.data.dataset import dataset_root

    path = dataset_root(repo_id) / f"episode_{episode:06d}" / "episode.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def recording(sim, supervise):
    """A manager recording ``test/sim_record`` from the simulated expert leader."""

    from so101.data.recorder import RecordingConfig
    from so101.services.robot_manager import Mode, get_manager

    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    manager.start_recording(
        RecordingConfig(repo_id="test/sim_record", fps=30, camera_keys=[]), operator_id=OPERATOR
    )
    assert _wait_for(lambda: manager.follower().connected and manager.mode is Mode.RECORDING)
    yield manager
    manager.soft_stop()


def _record_episode(manager, frames: int = 6) -> dict:
    manager.begin_episode()
    assert _wait_for(lambda: manager._episode_frames >= frames)
    return manager.save_episode()


def test_sim_recording_resets_records_environment_state_and_labels(recording):
    import json

    from so101.data.dataset import dataset_root

    runtime = devices.get_sim_runtime()
    first = _record_episode(recording)
    assert runtime.status()["seed"] == 3 and runtime.status()["episode"] == 1
    first_target = np.asarray(
        _frames("test/sim_record", first["episode"])[0]["observation.environment_state"]
    )
    second = _record_episode(recording)
    # Each episode starts from a fresh layout: the next seed.
    assert runtime.status()["seed"] == 4 and runtime.status()["episode"] == 2

    meta = json.loads((dataset_root("test/sim_record") / "meta.json").read_text("utf-8"))
    assert meta["environment_state_names"] == REACH_ENV
    for result in (first, second):
        frames = _frames("test/sim_record", result["episode"])
        assert len(frames) == result["frames"] > 0
        states = {tuple(frame["observation.environment_state"]) for frame in frames}
        assert len(states) == 1 and len(next(iter(states))) == 3  # the target never moves
        assert result["outcome"] in {"success", "failure"}
        assert result["label"] == "auto-labeled"
        manifest = _episode_manifest("test/sim_record", result["episode"])
        assert manifest["outcome"] == result["outcome"]
        assert manifest["outcome_source"] == "auto:sim_predicate"
    second_target = np.asarray(
        _frames("test/sim_record", second["episode"])[0]["observation.environment_state"]
    )
    assert not np.allclose(first_target, second_target)  # seeds 3 and 4: two layouts


def test_the_episode_reset_reseeds_the_safety_guard(recording, monkeypatch):
    """The reset moves the arm; the guard must rate-limit from the new pose, not the old."""

    follower = recording.follower()
    runtime = devices.get_sim_runtime()
    seeded: list[tuple[dict[str, float], dict[str, float]]] = []
    original = follower.guard.seed

    def spy(reference: dict[str, float] | None = None) -> None:
        assert reference is not None
        seeded.append((dict(reference), runtime.read_positions()))
        original(reference)

    monkeypatch.setattr(follower.guard, "seed", spy)
    recording.begin_episode()
    assert len(seeded) == 1, "begin_episode did not re-seed the guard after the reset"
    reference, pose_then = seeded[0]
    assert reference == pose_then  # the pose measured after the reset
    from so101.simulation.tasks import make_task

    task = make_task("reach")  # its own copy: the runtime's task is in use
    start = task.reset(task.make_world(), 3)["start_pose"]
    assert max(abs(reference[joint] - start[joint]) for joint in start) < 5.0
    assert _wait_for(lambda: recording._episode_frames >= 2)
    recording.save_episode()


def test_a_sim_dataset_with_another_layout_is_refused(sim, supervise):
    from so101 import JOINT_NAMES
    from so101.data.recorder import RecordingConfig
    from so101.services.robot_manager import get_manager
    from so101.teleop.controller import TeleopStep

    joints = {f"{joint}.pos": 0.0 for joint in JOINT_NAMES}
    other = RecordingConfig(repo_id="test/plain", fps=30, camera_keys=[])
    from so101.data.recorder import EpisodeRecorder

    recorder = EpisodeRecorder(other)
    recorder.add(TeleopStep(t=1.0, observation=joints, action=joints))
    recorder.save_episode()

    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    with pytest.raises(RuntimeError, match="already holds episodes with environment state"):
        manager.start_recording(other, operator_id=OPERATOR)


def test_the_manager_labels_with_the_same_source_as_sim_demos():
    from so101 import cli_sim
    from so101.services import robot_manager

    assert robot_manager.SIM_OUTCOME_SOURCE == cli_sim.SIM_OUTCOME_SOURCE == "auto:sim_predicate"
