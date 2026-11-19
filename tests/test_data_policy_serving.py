"""Policy serving: task, device, fps, per-episode reset and registry gating.

Covers audit B6 (VLA task), B11 (reset every episode), B15 (MPS), B16
(registered-only API rollouts), B18 (fps defaults), B21 (registry type),
B22 (QC on corrupt data) and B27 (local-only LeRobot reads).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from so101 import JOINT_NAMES
from so101.cli import app as cli_app
from so101.config import get_settings
from so101.data.lerobot_local import (
    LOCAL_ONLY_REVISION,
    IncompleteLocalDatasetError,
    check_local_lerobot_dataset,
    open_local_lerobot_dataset,
)
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.policy import checkpoint as checkpoint_mod
from so101.policy.checkpoint import (
    InvalidCheckpointError,
    MissingTaskError,
    ModelNotFoundError,
    inspect_checkpoint,
    preflight_policy,
    resolve_control_fps,
    resolve_registered_model,
    uses_language_task,
)
from so101.policy.server import LeRobotPolicy, load_policy, select_torch_device
from so101.services.app import create_app
from so101.teleop.controller import TeleopStep
from so101.training.run_manifest import RUN_MANIFEST_NAME

# -- helpers ---------------------------------------------------------------------------


def _checkpoint(
    run: str = "run",
    *,
    config: dict | None = None,
    manifest: dict | None = None,
    label: str = "last",
) -> Path:
    """A checkpoint under the outputs dir (registered), with an optional run manifest."""

    run_dir = get_settings().outputs_dir / run
    path = run_dir / "checkpoints" / label / "pretrained_model"
    path.mkdir(parents=True)
    if config is not None:
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if manifest is not None:
        (run_dir / RUN_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _mock_checkpoint(run: str = "run", manifest: dict | None = None) -> Path:
    return _checkpoint(run, config={"type": "act", "mock": True}, manifest=manifest)


def _real_mode(monkeypatch) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()
    # Pretend torch + lerobot are importable; nothing here loads weights.
    monkeypatch.setattr(checkpoint_mod, "missing_lerobot_runtime", lambda: [])


def _record(repo_id: str, fps: int = 10, frames: int = 3) -> None:
    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, fps=fps, camera_keys=[]))
    for index in range(frames):
        joints = {f"{joint}.pos": float(index) for joint in JOINT_NAMES}
        recorder.add(TeleopStep(t=1.0 + index / fps, observation=joints, action=joints))
    recorder.save_episode()


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


# -- B15: inference device -------------------------------------------------------------


class _FakeTorch:
    def __init__(self, *, cuda: bool = False, mps: bool | None = False) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        if mps is not None:  # None: a torch build without torch.backends.mps
            self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))

    @staticmethod
    def device(name: str) -> str:
        return f"device:{name}"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "device:cuda"),
        (False, True, "device:mps"),
        (False, False, "device:cpu"),
        (False, None, "device:cpu"),
    ],
)
def test_auto_device_prefers_cuda_then_mps_then_cpu(cuda, mps, expected):
    torch = _FakeTorch(cuda=cuda, mps=mps)
    assert select_torch_device(torch) == expected
    assert select_torch_device(torch, "auto") == expected


def test_explicit_device_wins_and_fails_closed_when_unavailable():
    assert select_torch_device(_FakeTorch(cuda=True, mps=True), "cpu") == "device:cpu"
    assert select_torch_device(_FakeTorch(cuda=True), "CUDA:1") == "device:cuda:1"
    with pytest.raises(RuntimeError, match="not available"):
        select_torch_device(_FakeTorch(cuda=True, mps=False), "mps")
    for bad in ("tpu", "mps:0", "cpu:1"):
        with pytest.raises(ValueError, match="not a device"):
            select_torch_device(_FakeTorch(), bad)


def test_policy_device_setting_reads_the_environment(monkeypatch):
    monkeypatch.setenv("SO101_POLICY_DEVICE", "mps")
    get_settings.cache_clear()
    assert get_settings().policy_device == "mps"


# -- B6: language task -----------------------------------------------------------------


def test_language_policies_are_detected_by_type_or_tokenizer_step(tmp_path):
    assert uses_language_task(tmp_path, "smolvla")
    assert uses_language_task(tmp_path, "pi0")
    assert uses_language_task(tmp_path, "PI05")
    assert not uses_language_task(tmp_path, "act")
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {"name": "policy_preprocessor", "steps": [{"registry_name": "tokenizer_processor"}]}
        ),
        encoding="utf-8",
    )
    assert uses_language_task(tmp_path, "custom_vla")
    (tmp_path / "policy_preprocessor.json").write_text("{not json", encoding="utf-8")
    assert not uses_language_task(tmp_path, "custom_vla")


def test_task_and_fps_default_from_the_run_manifest():
    path = _checkpoint(
        config={"type": "smolvla"},
        manifest={"task": "Pick up the cube.", "fps": 15, "dataset": {"repo_id": "lab/cubes"}},
    )
    info = inspect_checkpoint(path)
    assert info.kind == "lerobot"
    assert info.policy_type == "smolvla"
    assert info.needs_task
    assert (info.task, info.fps, info.dataset) == ("Pick up the cube.", 15, "lab/cubes")
    assert resolve_control_fps(None, info) == 15
    assert resolve_control_fps(20, info) == 20
    assert resolve_control_fps(None, None) == 30
    for bad in (0, 241, 12.5, float("inf")):
        with pytest.raises(ValueError, match="between 1 and 240"):
            resolve_control_fps(bad, info)
        with pytest.raises(ValueError, match="between 1 and 240"):
            resolve_control_fps(None, SimpleNamespace(fps=bad))


@pytest.mark.parametrize("fps", [0, -1, 12.5, 241])
def test_invalid_checkpoint_fps_fails_closed_instead_of_falling_back(fps):
    path = _checkpoint(
        f"bad_fps_{str(fps).replace('.', '_').replace('-', 'neg')}",
        config={"type": "act", "mock": True},
        manifest={"fps": fps},
    )

    with pytest.raises(InvalidCheckpointError, match="invalid fps"):
        inspect_checkpoint(path)


def test_invalid_checkpoint_fps_is_a_client_error_for_every_motion_entrypoint(client, monkeypatch):
    path = _checkpoint(
        "bad_fps_entrypoints",
        config={"type": "act", "mock": True},
        manifest={"fps": 241},
    )

    response = client.post(
        "/api/models/rollout/start",
        json={"model_path": str(path), "operator_id": "test-operator"},
    )
    assert response.status_code == 422
    assert "invalid fps" in response.json()["detail"]

    loaded: list[str] = []
    monkeypatch.setattr(
        "so101.policy.server.load_policy",
        lambda model_path, **kwargs: loaded.append(str(model_path)),
    )

    for command in (
        ("run", str(path), "--duration", "0"),
        ("eval", str(path), "--episodes", "1", "--max-seconds", "0.01"),
    ):
        result, text = _invoke(*command)
        assert result.exit_code == 1, text
        assert "invalid fps" in text
    assert loaded == []


def test_task_defaults_from_train_config_dataset():
    path = _checkpoint(config={"type": "smolvla"})
    _record("lab/train_config_task", fps=12)
    meta_path = get_settings().datasets_dir / "v2--lab%2Ftrain_config_task" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["task"] = "Stack the blocks."
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (path / "train_config.json").write_text(
        json.dumps(
            {"dataset": {"repo_id": "lab/train_config_task"}, "policy": {"type": "smolvla"}}
        ),
        encoding="utf-8",
    )
    info = inspect_checkpoint(path)
    assert (info.task, info.fps, info.dataset) == ("Stack the blocks.", 12, "lab/train_config_task")


def test_vla_without_a_task_fails_closed_off_mock(monkeypatch):
    path = _checkpoint(config={"type": "smolvla"})
    # Mock mode serves MockPolicy, which reads no prompt.
    assert preflight_policy(path).task is None

    _real_mode(monkeypatch)
    with pytest.raises(MissingTaskError, match="needs a task"):
        preflight_policy(path)
    assert preflight_policy(path, task="  Wipe the table.  ").task == "Wipe the table."
    with pytest.raises(MissingTaskError):
        load_policy(path)


def test_act_needs_no_task(monkeypatch):
    path = _checkpoint(config={"type": "act"})
    _real_mode(monkeypatch)
    info = preflight_policy(path)
    assert not info.needs_task
    assert info.task is None


def test_load_policy_serves_the_training_task_unless_overridden(monkeypatch):
    path = _checkpoint(config={"type": "smolvla"}, manifest={"task": "Open the drawer."})
    _real_mode(monkeypatch)
    monkeypatch.setattr(LeRobotPolicy, "_ensure", lambda self: None)

    policy = load_policy(path)
    assert isinstance(policy, LeRobotPolicy)
    assert policy.task == "Open the drawer."
    assert load_policy(path, task="Close the drawer.").task == "Close the drawer."


def test_lerobot_policy_passes_the_task_to_inference(monkeypatch):
    seen: dict = {}

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def squeeze(self, _axis):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeInnerPolicy:
        def select_action(self, batch):
            return FakeTensor(np.arange(6))

    def prepare(obs, device, task=None, robot_type=None):
        seen["task"] = task
        seen["robot_type"] = robot_type
        return obs

    torch = SimpleNamespace(inference_mode=lambda: _NullContext())
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    policy = LeRobotPolicy("/unused", task="Fold the towel.")
    policy._policy = FakeInnerPolicy()
    policy._preprocessor = lambda batch: batch
    policy._postprocessor = lambda action: action
    policy._prepare_observation = prepare
    policy._modern = True
    action = policy.select_action({"observation.state": np.zeros(6, dtype=np.float32)})
    assert seen == {"task": "Fold the towel.", "robot_type": "so_follower"}
    assert action == {joint: float(index) for index, joint in enumerate(JOINT_NAMES)}


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# -- B11: reset every episode ----------------------------------------------------------


class _CountingPolicy:
    def __init__(self) -> None:
        self.steps_since_reset: list[int] = []

    def reset(self) -> None:
        self.steps_since_reset.append(0)

    def select_action(self, observation):
        assert self.steps_since_reset, "select_action before the first reset"
        self.steps_since_reset[-1] += 1
        return {joint: 0.0 for joint in JOINT_NAMES}


def test_evaluate_resets_the_policy_before_every_episode(monkeypatch):
    from so101.eval import evaluator

    policy = _CountingPolicy()
    calls: list[dict] = []

    def fake_load(path, *, task=None):
        calls.append({"path": str(path), "task": task})
        return policy

    monkeypatch.setattr(evaluator, "load_policy", fake_load)
    path = _mock_checkpoint(manifest={"fps": 15})
    report = evaluator.evaluate(
        evaluator.EvalConfig(model_path=str(path), n_episodes=3, max_seconds=0.02, task="Go.")
    )
    assert report.successes == 3
    assert calls == [{"path": str(path), "task": "Go."}]
    # One reset on connect, then one at the start of each of the three
    # episodes; every episode's actions follow its own reset.
    assert len(policy.steps_since_reset) == 4
    assert all(count > 0 for count in policy.steps_since_reset[1:])


def test_evaluate_runs_at_the_checkpoint_fps(monkeypatch):
    from so101.eval import evaluator

    rates: list[int] = []
    real_runner = evaluator.PolicyRunner

    def capture(policy, fps=30, **kwargs):
        rates.append(fps)
        return real_runner(policy, fps=fps, **kwargs)

    monkeypatch.setattr(evaluator, "PolicyRunner", capture)
    path = _mock_checkpoint(manifest={"fps": 15})
    evaluator.evaluate(evaluator.EvalConfig(model_path=str(path), n_episodes=1, max_seconds=0.01))
    evaluator.evaluate(
        evaluator.EvalConfig(model_path=str(path), n_episodes=1, max_seconds=0.01, fps=25)
    )
    assert rates == [15, 25]


# -- CLI: --task / --fps ---------------------------------------------------------------


def test_cli_run_defaults_fps_and_task_from_the_checkpoint():
    path = _mock_checkpoint(manifest={"fps": 15, "task": "Sort the parts."})
    result, text = _invoke("run", str(path), "--duration", "0.02")
    assert result.exit_code == 0, text
    assert "15 fps" in text
    assert "task 'Sort the parts.'" in text

    result, text = _invoke(
        "run", str(path), "--duration", "0.02", "--fps", "20", "--task", "Count the parts."
    )
    assert result.exit_code == 0, text
    assert "20 fps" in text
    assert "task 'Count the parts.'" in text


def test_cli_run_refuses_vla_without_task_on_real_backend(monkeypatch):
    path = _checkpoint(config={"type": "smolvla"})
    _real_mode(monkeypatch)
    result, text = _invoke("run", str(path), "--duration", "0.02")
    assert result.exit_code == 1
    assert "needs a task" in text
    assert "rollout done" not in text


# -- B16 / B18: HTTP rollout -----------------------------------------------------------


def test_rollout_api_accepts_only_registered_models(client, tmp_path):
    outside = tmp_path / "elsewhere" / "pretrained_model"
    outside.mkdir(parents=True)
    (outside / "config.json").write_text(
        json.dumps({"type": "act", "mock": True}), encoding="utf-8"
    )
    refused = client.post(
        "/api/models/rollout/start",
        json={"model_path": str(outside), "operator_id": "test-operator"},
    )
    assert refused.status_code == 404
    assert "not a registered model" in refused.json()["detail"]
    assert client.get("/api/robot/status").json()["mode"] == "idle"

    _mock_checkpoint("registered")
    by_name = client.post(
        "/api/models/rollout/start",
        json={"model_path": "registered@last", "operator_id": "test-operator"},
    )
    assert by_name.status_code == 200, by_name.text
    assert client.post("/api/models/rollout/stop").status_code == 200


def test_registered_paths_resolve_through_symlinks():
    real = _checkpoint("linked", config={"type": "act", "mock": True}, label="000100")
    last = real.parent.parent / "last"
    try:
        last.symlink_to("000100", target_is_directory=True)
    except OSError:  # Windows without Developer Mode cannot create symlinks
        pytest.skip("this platform does not allow unprivileged symlinks")
    assert resolve_registered_model(last / "pretrained_model") == real
    assert resolve_registered_model("linked@000100") == real
    with pytest.raises(ModelNotFoundError):
        resolve_registered_model("linked@last")


def test_rollout_api_defaults_to_checkpoint_fps(client, monkeypatch):
    from so101.services.robot_manager import get_manager

    _mock_checkpoint("fps15", manifest={"fps": 15})
    started: list[dict] = []

    def fake_start(model_path, fps=30, *, operator_id):
        started.append({"model_path": model_path, "fps": fps})

    monkeypatch.setattr(get_manager(), "start_rollout", fake_start)
    ok = client.post(
        "/api/models/rollout/start",
        json={"model_path": "fps15@last", "operator_id": "test-operator"},
    )
    assert ok.status_code == 200, ok.text
    assert "at 15 fps" in ok.json()["detail"]
    explicit = client.post(
        "/api/models/rollout/start",
        json={"model_path": "fps15@last", "operator_id": "test-operator", "fps": 30},
    )
    assert explicit.status_code == 200
    assert [call["fps"] for call in started] == [15, 30]


def test_rollout_api_never_substitutes_a_requested_task(client, monkeypatch):
    from so101.services.robot_manager import get_manager

    _mock_checkpoint("tasked", manifest={"task": "Pick the red cube."})
    body = {"model_path": "tasked@last", "operator_id": "test-operator"}
    manager = get_manager()
    started: list[dict] = []

    def legacy_start(model_path, fps=30, *, operator_id):
        started.append({"task": None})

    monkeypatch.setattr(manager, "start_rollout", legacy_start)
    # This manager cannot take a task: a different prompt must be refused,
    # the checkpoint's own task is fine.
    mismatch = client.post(
        "/api/models/rollout/start", json={**body, "task": "Pick the blue cube."}
    )
    assert mismatch.status_code == 422
    same = client.post("/api/models/rollout/start", json={**body, "task": "Pick the red cube."})
    assert same.status_code == 200
    assert "task 'Pick the red cube.'" in same.json()["detail"]

    def task_aware_start(model_path, fps=30, *, operator_id, task=None):
        started.append({"task": task})

    monkeypatch.setattr(manager, "start_rollout", task_aware_start)
    passed = client.post("/api/models/rollout/start", json={**body, "task": "Pick the blue cube."})
    assert passed.status_code == 200, passed.text
    assert started[-1] == {"task": "Pick the blue cube."}


# -- B21: registry ---------------------------------------------------------------------


def test_registry_reports_lerobot_type_and_training_context(client):
    _checkpoint(
        "act_run",
        config={"type": "act"},
        manifest={"fps": 15, "task": "Push the block.", "dataset": {"repo_id": "lab/push"}},
    )
    _checkpoint("broken")  # a pretrained_model dir without config.json
    custom = _checkpoint("custom_vla", config={"type": "lab_policy"})
    (custom / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": "tokenizer_processor"}]}),
        encoding="utf-8",
    )
    models = {item["name"]: item for item in client.get("/api/models").json()}
    act = models["act_run@last"]
    assert act["policy_type"] == "act"
    assert act["kind"] == "lerobot"
    assert act["label"] == "LeRobot act"
    assert act["needs_task"] is False
    assert (act["fps"], act["task"], act["dataset"]) == (15, "Push the block.", "lab/push")
    assert models["custom_vla@last"]["needs_task"] is True
    broken = models["broken@last"]
    assert broken["policy_type"] == "unknown"
    assert broken["label"].startswith("unreadable checkpoint")


# -- B18: replay pacing ----------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _TimedFollower:
    """Each send takes ``io_s`` seconds of fake time; one send can stall."""

    def __init__(self, clock: _FakeClock, io_s: float, stall_at: int | None = None) -> None:
        self.clock = clock
        self.io_s = io_s
        self.stall_at = stall_at
        self.sent: list[float] = []
        self.connected = True

    def send_goal(self, goal):
        self.sent.append(self.clock.now)
        self.clock.now += 0.5 if len(self.sent) - 1 == self.stall_at else self.io_s
        return goal


def test_replay_defaults_to_recorded_fps_on_a_deadline_schedule(monkeypatch):
    from so101.data import replay

    _record("lab/replay_fps", fps=10, frames=4)
    clock = _FakeClock()
    monkeypatch.setattr(replay, "time", clock)
    follower = _TimedFollower(clock, io_s=0.03)
    assert replay.replay_episode("lab/replay_fps", 0, follower=follower) == 4
    # 10 fps: sends are 0.1 s apart; the 30 ms send is part of the period.
    assert np.allclose(np.diff(follower.sent), 0.1)
    assert np.allclose(clock.sleeps, 0.07)

    clock.sleeps.clear()
    fast = _TimedFollower(clock, io_s=0.03)
    replay.replay_episode("lab/replay_fps", 0, follower=fast, fps=20)
    assert np.allclose(np.diff(fast.sent), 0.05)


def test_replay_resumes_after_a_stall_instead_of_bursting(monkeypatch):
    from so101.data import replay

    _record("lab/replay_stall", fps=10, frames=4)
    clock = _FakeClock()
    monkeypatch.setattr(replay, "time", clock)
    follower = _TimedFollower(clock, io_s=0.01, stall_at=1)
    replay.replay_episode("lab/replay_stall", 0, follower=follower)
    gaps = np.diff(follower.sent)
    assert np.isclose(gaps[0], 0.1)
    assert np.isclose(gaps[1], 0.5)  # the stall itself
    assert np.isclose(gaps[2], 0.1)  # then one period, not a catch-up burst


@pytest.mark.parametrize("fps", [0, -1, 241, float("nan"), float("inf")])
def test_replay_rejects_unsafe_explicit_fps(fps):
    _record("lab/replay_bad_fps")
    from so101.data.replay import replay_episode

    with pytest.raises(ValueError, match="fps must be between 1 and 240"):
        replay_episode("lab/replay_bad_fps", 0, fps=fps)


def test_replay_rejects_episode_symlink_before_opening_arm(tmp_path, monkeypatch):
    from so101.data import replay

    outside = tmp_path / "outside-episode"
    outside.mkdir()
    (outside / "frames.jsonl").write_text(
        json.dumps({"action": [0.0] * len(JOINT_NAMES)}) + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "dataset"
    root.mkdir()
    episode = root / "episode_000000"
    try:
        episode.symlink_to(outside, target_is_directory=True)
    except OSError:
        episode.mkdir()
        is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == episode or is_symlink(path),
        )
    events: list[str] = []

    class Dataset:
        format = "so101-fallback-v2"
        path = str(root)
        fps = 30
        repo_id = "lab/symlink-replay"

    monkeypatch.setattr(replay, "require_dataset", lambda _repo_id: Dataset())
    monkeypatch.setattr(
        replay,
        "standalone_access",
        lambda _purpose, **_kwargs: events.append("access") or None,
    )
    monkeypatch.setattr(
        replay,
        "SO101Follower",
        lambda: events.append("follower") or object(),
    )

    with pytest.raises(ValueError, match="symlink"):
        replay.replay_episode("lab/symlink-replay", 0)
    assert events == []


@pytest.mark.parametrize(
    "recorded_fps",
    [
        float("nan"),
        float("inf"),
        0,
        -1,
        0.5,
        241,
        pytest.param(10**10_000, id="float-overflow"),
    ],
)
def test_replay_rejects_unsafe_recorded_fps_before_opening_the_arm(monkeypatch, recorded_fps):
    from so101.data import replay

    events: list[str] = []

    class Dataset:
        format = "so101-fallback-v2"
        path = "."
        fps = recorded_fps
        repo_id = "lab/replay_recorded_fps"

    class Follower:
        def connect(self) -> None:
            events.append("connect")

        def send_goal(self, goal) -> None:
            events.append("send")

        def disconnect(self) -> None:
            events.append("disconnect")

    class Access:
        def disconnect_follower(self, follower) -> None:
            follower.disconnect()

        def release(self) -> None:
            pass

    monkeypatch.setattr(replay, "require_dataset", lambda repo_id: Dataset())
    monkeypatch.setattr(replay, "_load_fallback_actions", lambda root, episode: [[0.0] * 6])
    monkeypatch.setattr(replay, "SO101Follower", Follower)
    monkeypatch.setattr(replay, "standalone_access", lambda _purpose, **_kwargs: Access())

    with pytest.raises(ValueError, match="recorded fps must be between 1 and 240"):
        replay.replay_episode("lab/replay_recorded_fps", 0)

    assert events == []


def test_explicit_valid_replay_fps_overrides_an_invalid_recorded_rate(monkeypatch):
    from so101.data import replay

    class Dataset:
        format = "so101-fallback-v2"
        path = "."
        fps = float("nan")
        repo_id = "lab/replay_override"

    sent: list[dict[str, float]] = []

    class Follower:
        connected = True

        def send_goal(self, goal):
            sent.append(goal)

    clock = _FakeClock()
    monkeypatch.setattr(replay, "require_dataset", lambda repo_id: Dataset())
    monkeypatch.setattr(replay, "_load_fallback_actions", lambda root, episode: [[0.0] * 6])
    monkeypatch.setattr(replay, "time", clock)

    assert replay.replay_episode("lab/replay_override", 0, follower=Follower(), fps=20) == 1
    assert len(sent) == 1


# -- B22: QC on corrupt data -----------------------------------------------------------


def _lerobot_dataset(repo_id: str, episodes: int = 2) -> Path:
    from so101.data.dataset import dataset_root

    root = dataset_root(repo_id)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"")
    (root / "meta" / "tasks.parquet").write_bytes(b"")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": episodes,
                "total_frames": 4 * episodes,
                "total_tasks": 1,
                "fps": 30,
                "features": {"action": {"names": [f"{j}.pos" for j in JOINT_NAMES]}},
            }
        ),
        encoding="utf-8",
    )
    return root


class DatasetGenerationError(Exception):
    """Stands in for datasets.exceptions.DatasetGenerationError (not an OSError)."""


def test_qc_reports_undecodable_episodes_instead_of_raising(monkeypatch):
    from so101.data import inspection, replay

    _lerobot_dataset("lab/corrupt")
    monkeypatch.setattr(
        inspection,
        "lerobot_runtime_capability",
        lambda: inspection.InspectionCapability(
            id="lerobot_runtime", available=True, level="full", detail="test"
        ),
    )

    def corrupt(root, repo_id, episode):
        raise DatasetGenerationError("parquet footer missing")

    monkeypatch.setattr(replay, "_load_lerobot_actions", corrupt)
    report = inspection.run_quality_control("lab/corrupt")
    assert report.overall == "fail"
    text = json.dumps(report.to_dict())
    assert "cannot decode actions (DatasetGenerationError: parquet footer missing)" in text


def test_qc_route_maps_unexpected_failures_to_409(client, monkeypatch):
    from so101.services.routers import datasets as datasets_router

    _record("lab/qc_route")

    def explode(repo_id):
        raise DatasetGenerationError("unreadable")

    monkeypatch.setattr(datasets_router, "run_quality_control", explode)
    response = client.get("/api/datasets/lab/qc_route/qc")
    assert response.status_code == 409
    assert "DatasetGenerationError: unreadable" in response.json()["detail"]


# -- B27: local-only LeRobot reads -----------------------------------------------------


def test_local_checks_refuse_incomplete_metadata_and_bad_episodes(tmp_path):
    root = tmp_path / "ds"
    with pytest.raises(IncompleteLocalDatasetError, match="info.json is missing"):
        check_local_lerobot_dataset(root, "lab/ds")
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        check_local_lerobot_dataset(root, "lab/ds")
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_tasks": 1, "fps": 30}), encoding="utf-8"
    )
    with pytest.raises(IncompleteLocalDatasetError, match="tasks.parquet"):
        check_local_lerobot_dataset(root, "lab/ds")
    (root / "meta" / "tasks.parquet").write_bytes(b"")
    with pytest.raises(IncompleteLocalDatasetError, match="episode metadata"):
        check_local_lerobot_dataset(root, "lab/ds")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"")
    assert check_local_lerobot_dataset(root, "lab/ds", [0, 1])["total_episodes"] == 2
    with pytest.raises(IncompleteLocalDatasetError, match=r"episode 2 not found .*0\.\.1"):
        check_local_lerobot_dataset(root, "lab/ds", [2])
    with pytest.raises(IncompleteLocalDatasetError, match="episode -1 not found"):
        check_local_lerobot_dataset(root, "lab/ds", [-1])


def test_local_open_never_downloads(tmp_path):
    root = _lerobot_dataset("lab/local_only")
    opened: list[dict] = []

    class FakeLeRobotDataset:
        def __init__(self, repo_id, root=None, episodes=None, revision=None, **kwargs):
            opened.append({"repo_id": repo_id, "episodes": episodes, "revision": revision})
            self.root = root
            # LeRobot calls _download() when its local load fails.
            self._download()

        def _download(self):  # pragma: no cover - replaced by the local-only subclass
            raise AssertionError("downloaded from the Hub")

    with pytest.raises(IncompleteLocalDatasetError, match="never downloads"):
        open_local_lerobot_dataset(
            "lab/local_only", root, episodes=[1], dataset_cls=FakeLeRobotDataset
        )
    assert opened == [
        {"repo_id": "lab/local_only", "episodes": [1], "revision": LOCAL_ONLY_REVISION}
    ]
    # An out-of-range episode is refused before LeRobot is even constructed.
    with pytest.raises(IncompleteLocalDatasetError, match="episode 5 not found"):
        open_local_lerobot_dataset(
            "lab/local_only", root, episodes=[5], dataset_cls=FakeLeRobotDataset
        )
    assert len(opened) == 1


def test_local_open_without_lerobot_is_a_clear_error(monkeypatch):
    import builtins

    root = _lerobot_dataset("lab/no_runtime")
    real_import = builtins.__import__

    def without_lerobot(name, *args, **kwargs):
        if name == "lerobot" or name.startswith("lerobot."):
            raise ModuleNotFoundError("no lerobot")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_lerobot)
    with pytest.raises(RuntimeError, match="not installed"):
        open_local_lerobot_dataset("lab/no_runtime", root)


def test_replay_of_a_lerobot_episode_out_of_range_is_refused_before_loading():
    _lerobot_dataset("lab/replay_range", episodes=1)
    from so101.data.replay import replay_episode

    with pytest.raises(FileNotFoundError, match=r"episode 3 not found in lab/replay_range"):
        replay_episode("lab/replay_range", 3)
