"""The numpy BC baseline wired through so101: ``so101 train --policy bc_baseline``, the
jobs API, the model registry, preflight gating, serving, rollout and eval."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from so101 import JOINT_NAMES
from so101.cli import app as cli_app
from so101.config import get_settings
from so101.eval.evaluator import EvalConfig, EvalRun
from so101.models import list_models
from so101.policy import checkpoint as checkpoint_mod
from so101.policy.baseline import (
    MANIFEST_NAME,
    PRIVILEGED_LABEL,
    STATE_ONLY_LABEL,
    WEIGHTS_NAME,
    read_baseline_manifest,
    save_baseline_checkpoint,
)
from so101.policy.checkpoint import (
    InvalidCheckpointError,
    PolicyBackendError,
    inspect_checkpoint,
    preflight_policy,
)
from so101.policy.runner import PolicyRunner, _build_observation
from so101.policy.server import JointDictPolicy, load_policy
from so101.services.app import create_app
from so101.services.jobs import get_runner
from so101.training.bc_baseline import CHECKPOINT_SUBDIR, BCConfig, fit_bc, manifest_extra
from so101.training.configs import POLICY_PRESETS
from so101.training.preflight import validate_training_dataset
from so101.training.run_manifest import read_run_manifest
from so101.training.toy_reach import (
    ENV_STATE_NAMES,
    TASK,
    ReachSuccess,
    generate_toy_reach_episodes,
    sample_start,
    sample_target,
    toy_training_data,
    track,
    write_toy_reach_dataset,
)

FAST = ("--steps", "300", "--arg", "hidden=32,32", "--arg", "eval_every=50")


def _toy(repo_id: str = "toy/reach", *, count: int = 8, seed: int = 0, **kwargs):
    episodes = generate_toy_reach_episodes(
        count, seed=seed, failure_fraction=kwargs.pop("failure_fraction", 0.0)
    )
    write_toy_reach_dataset(repo_id, episodes, **kwargs)
    return episodes


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(cli_app, list(args))
    return result.exit_code, " ".join(result.output.split())


def _train(repo_id: str, run: str, *extra: str):
    out = get_settings().outputs_dir / run
    code, text = _invoke(
        "train", repo_id, "--policy", "bc_baseline", *FAST, "--output-dir", str(out), *extra
    )
    assert code == 0, text
    return out / CHECKPOINT_SUBDIR, text


def _real_mode(monkeypatch) -> None:
    monkeypatch.setenv("SO101_MOCK", "0")
    get_settings.cache_clear()


def _wait_for(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


@pytest.fixture
def client():
    from so101.services.robot_manager import get_manager

    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat("test-operator")
        yield test_client
    get_manager().stop()


class _ToyArm:
    """A follower that moves like the toy reach task; ``target`` is its object state."""

    def __init__(self, start: np.ndarray, target: np.ndarray) -> None:
        self.state = np.asarray(start, dtype=np.float64)
        self.success = ReachSuccess(target)
        self.target = np.asarray(target, dtype=np.float64)
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_positions(self) -> dict[str, float]:
        return {joint: float(value) for joint, value in zip(JOINT_NAMES, self.state, strict=True)}

    def environment_state(self) -> np.ndarray:
        return self.target.copy()

    def send_goal(self, goal: dict[str, float]) -> dict[str, float]:
        self.state = track(self.state, np.asarray([goal[joint] for joint in JOINT_NAMES]))
        self.success.update(self.state)
        return dict(goal)


class _BlindToyArm(_ToyArm):
    """The same arm without an object-state sensor (what hardware looks like)."""

    environment_state = None  # type: ignore[assignment]


class _NoCameras:
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self) -> dict[str, np.ndarray]:
        return {}


def test_cli_trains_a_privileged_baseline_in_mock_mode_and_registers_it():
    _toy()
    checkpoint, text = _train("toy/reach", "bc_priv")
    assert "[bc] data:" in text and "[bc] held-out action MAE" in text
    assert f"[bc] saved {PRIVILEGED_LABEL}" in text
    assert (checkpoint / MANIFEST_NAME).is_file() and (checkpoint / WEIGHTS_NAME).is_file()
    run = read_run_manifest(checkpoint)
    assert run is not None
    assert (run["trainer"], run["preset"], run["policy_type"]) == (
        "numpy_in_process",
        "bc_baseline",
        "numpy_bc_mlp",
    )
    info = inspect_checkpoint(checkpoint)
    assert (info.kind, info.label, info.privileged_state) == (
        "numpy_bc_mlp",
        PRIVILEGED_LABEL,
        True,
    )
    assert info.env_state_names == ENV_STATE_NAMES
    assert (info.task, info.fps, info.dataset, info.needs_task) == (TASK, 30, "toy/reach", False)
    (model,) = list_models()
    assert (model.name, model.kind, model.label) == (
        "bc_priv@last",
        "numpy_bc_mlp",
        PRIVILEGED_LABEL,
    )
    code, table = _invoke("models", "ls")
    assert code == 0 and "label" in table

    # A finished run is never overwritten.
    again = str(get_settings().outputs_dir / "bc_priv")
    code, text = _invoke("train", "toy/reach", "--policy", "bc_baseline", "--output-dir", again)
    assert code == 1 and "already exists" in text


def test_cli_trains_a_state_only_baseline_in_real_mode_without_lerobot(monkeypatch):
    _toy()
    _real_mode(monkeypatch)
    # No LeRobot trainer importable by this interpreter.
    monkeypatch.setattr("so101.training.trainer.TRAIN_MODULE", "so101_no_such_trainer")
    checkpoint, text = _train("toy/reach", "bc_state", "--state-only", "--seed", "3")
    assert f"[bc] saved {STATE_ONLY_LABEL}" in text
    manifest = read_baseline_manifest(checkpoint)
    assert manifest["label"] == STATE_ONLY_LABEL and manifest["privileged_state"] is False
    assert manifest["training"]["config"]["seed"] == 3
    assert manifest["training"]["config"]["state_only"] is True
    assert manifest["dataset"]["environment_state"] == []
    assert read_run_manifest(checkpoint)["backend"] == "real"
    # A state-only baseline may drive the real arm: preflight and loading pass
    # without touching hardware.
    assert preflight_policy(checkpoint).label == STATE_ONLY_LABEL
    policy = load_policy(checkpoint)
    assert isinstance(policy, JointDictPolicy) and not policy.privileged_state


def test_training_keeps_successful_episodes_by_default_and_all_on_request():
    episodes = _toy("toy/mixed", count=12, seed=0, failure_fraction=0.4)
    failures = [index for index, episode in enumerate(episodes) if not episode.success]
    assert failures and len(failures) < 12
    kept = read_baseline_manifest(_train("toy/mixed", "bc_kept")[0])["dataset"]
    assert kept["selection"] == "kept"
    assert kept["episodes"] == [index for index in range(12) if index not in failures]
    assert [skip["episode"] for skip in kept["skipped"]] == failures
    every = read_baseline_manifest(_train("toy/mixed", "bc_all", "--episodes", "all")[0])
    assert every["dataset"]["selection"] == "all"
    assert every["dataset"]["episodes"] == list(range(12))
    assert every["dataset"]["skipped"] == []


def test_cli_rejects_baseline_options_it_cannot_honour():
    _toy()
    code, text = _invoke("train", "toy/reach", "--policy", "act", "--state-only")
    assert code == 1 and "bc_baseline preset" in text
    # --episodes applies to every preset, but only with a known selection.
    code, text = _invoke("train", "toy/reach", "--policy", "act", "--episodes", "best")
    assert code == 1 and "episodes must be one of" in text
    code, text = _invoke("train", "toy/reach", "--policy", "bc_baseline", "--episodes", "best")
    assert code == 1 and "episodes" in text
    both = ("--episodes", "all", "--arg", "episodes=kept")
    code, text = _invoke("train", "toy/reach", "--policy", "bc_baseline", *both)
    assert code == 1 and "conflicts" in text
    code, text = _invoke("train", "toy/reach", "--policy", "bc_baseline", "--arg", "steps=5")
    assert code == 1 and "not a bc_baseline option" in text
    assert not any(get_settings().outputs_dir.glob("*"))


def test_jobs_api_trains_registers_and_gates_a_privileged_baseline(client):
    _toy()
    presets = {preset["name"]: preset for preset in client.get("/api/training/presets").json()}
    assert presets["bc_baseline"]["policy_type"] == "numpy_bc_mlp"
    assert presets["bc_baseline"]["trainer"] == "numpy_in_process"
    assert presets["bc_baseline"]["requires_torch"] is False
    assert (presets["act"]["trainer"], presets["act"]["requires_torch"]) == ("lerobot", True)

    submitted = client.post(
        "/api/training/jobs",
        json={"repo_id": "toy/reach", "policy": "bc_baseline", "extra_args": {"hidden": "32,32"}},
    )
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["id"]
    # An omitted budget takes the preset's defaults, not ACT's 100k steps.
    preset = POLICY_PRESETS["bc_baseline"]
    job = get_runner().get(job_id)
    assert job is not None
    assert (job.steps, job.batch_size) == (preset.default_steps, preset.default_batch_size)
    terminal = {"succeeded", "failed", "cancelled"}
    _wait_for(lambda: client.get(f"/api/training/jobs/{job_id}").json()["status"] in terminal)
    done = client.get(f"/api/training/jobs/{job_id}").json()
    assert done["status"] == "succeeded", done
    assert done["loss"] is not None and done["progress"].startswith("Training:")
    logs = client.get(f"/api/training/jobs/{job_id}/logs").json()["log_tail"]
    assert any(line.startswith(f"[bc] saved {PRIVILEGED_LABEL}") for line in logs)

    (model,) = [entry for entry in client.get("/api/models").json() if entry["kind"]]
    assert (model["kind"], model["label"], model["privileged_state"]) == (
        "numpy_bc_mlp",
        PRIVILEGED_LABEL,
        True,
    )
    assert (model["task"], model["fps"], model["dataset"]) == (TASK, 30, "toy/reach")
    refused = client.post(
        "/api/models/rollout/start",
        json={"model_path": model["name"], "operator_id": "test-operator"},
    )
    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert PRIVILEGED_LABEL in detail and "refuses the mock backend" in detail
    assert "--state-only" in detail
    assert client.get("/api/robot/status").json()["mode"] == "idle"


def test_rollout_api_runs_a_state_only_baseline_on_the_mock_arm(client):
    from so101.services.robot_manager import get_manager

    _toy()
    checkpoint, _text = _train("toy/reach", "bc_state", "--state-only")
    started = client.post(
        "/api/models/rollout/start",
        json={"model_path": str(checkpoint), "operator_id": "test-operator"},
    )
    assert started.status_code == 200, started.text
    assert STATE_ONLY_LABEL in started.json()["detail"] and "30 fps" in started.json()["detail"]
    manager = get_manager()
    _wait_for(lambda: manager.telemetry()["sample_seq"] >= 5 or manager.fault is not None)
    assert manager.fault is None
    assert client.get("/api/robot/status").json()["mode"] == "rollout"
    assert client.post("/api/models/rollout/stop").status_code == 200


@pytest.mark.parametrize("backend", ["real", "mock"])
def test_preflight_refuses_a_privileged_baseline_off_the_simulator(backend, monkeypatch):
    _toy()
    checkpoint, _text = _train("toy/reach", "bc_priv")
    with pytest.raises(PolicyBackendError, match=f"refuses the {backend} backend") as refused:
        preflight_policy(checkpoint, backend=backend)
    assert PRIVILEGED_LABEL in str(refused.value) and "target_x" in str(refused.value)
    monkeypatch.setenv("SO101_SIM_TASK", "reach")
    get_settings.cache_clear()
    assert preflight_policy(checkpoint, backend="sim").privileged_state


def test_preflight_refuses_privileged_environment_state_for_another_sim_task(monkeypatch):
    _toy()
    checkpoint, _text = _train("toy/reach", "bc_priv")
    monkeypatch.setenv("SO101_BACKEND", "sim")
    monkeypatch.setenv("SO101_SIM_TASK", "pick_lift")
    get_settings.cache_clear()

    with pytest.raises(PolicyBackendError, match="environment state") as refused:
        preflight_policy(checkpoint)

    detail = str(refused.value)
    assert "target_x" in detail
    assert "cube_x" in detail
    assert "pick_lift" in detail


def test_cli_run_and_eval_refuse_a_privileged_baseline_in_mock_mode():
    _toy()
    checkpoint, _text = _train("toy/reach", "bc_priv")
    for command in (
        ("run", str(checkpoint), "--duration", "0.1"),
        ("eval", str(checkpoint), "--episodes", "1"),
    ):
        code, text = _invoke(*command)
        assert code == 1, text
        assert "refuses the mock backend" in text and PRIVILEGED_LABEL in text


def test_load_policy_gates_a_privileged_baseline_on_the_active_backend(monkeypatch):
    _toy()
    checkpoint, _text = _train("toy/reach", "bc_priv")
    # Mock mode never substitutes MockPolicy for a baseline.
    with pytest.raises(PolicyBackendError, match="refuses the mock backend"):
        load_policy(checkpoint)
    _real_mode(monkeypatch)
    with pytest.raises(PolicyBackendError, match="refuses the real backend"):
        load_policy(checkpoint)
    monkeypatch.setattr(checkpoint_mod, "active_backend", lambda: "sim")
    policy = load_policy(checkpoint)
    assert isinstance(policy, JointDictPolicy) and policy.privileged_state
    assert policy.label == PRIVILEGED_LABEL


def test_preflight_verifies_baseline_weights_and_the_registry_survives_a_bad_manifest():
    _toy()
    checkpoint, _text = _train("toy/reach", "bc_state", "--state-only")
    weights = checkpoint / WEIGHTS_NAME
    data = bytearray(weights.read_bytes())
    data[len(data) // 2] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(InvalidCheckpointError, match="does not match the manifest"):
        preflight_policy(checkpoint)
    (checkpoint / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(InvalidCheckpointError):
        inspect_checkpoint(checkpoint)
    with pytest.raises(InvalidCheckpointError):
        load_policy(checkpoint)  # never MockPolicy for a damaged baseline
    (model,) = list_models()
    assert model.label.startswith("unreadable checkpoint")


def test_privileged_baseline_reaches_through_policy_runner_with_simulated_object_state(
    monkeypatch, tmp_path
):
    data = toy_training_data(generate_toy_reach_episodes(30, seed=0))
    result = fit_bc(data, BCConfig(steps=1500, batch_size=128, hidden=(64, 64), eval_every=50))
    checkpoint = save_baseline_checkpoint(
        tmp_path / "ckpt", result.model, extra=manifest_extra(result, tasks=[TASK])
    )
    monkeypatch.setattr(checkpoint_mod, "active_backend", lambda: "sim")
    policy = load_policy(checkpoint)
    rng = np.random.default_rng(7)
    reached = 0
    for _ in range(5):
        arm = _ToyArm(sample_start(rng), sample_target(rng))
        runner = PolicyRunner(policy, follower=arm, cameras=_NoCameras(), fps=30)  # type: ignore[arg-type]
        runner.connect()
        for _step in range(90):
            runner.step()
        runner.disconnect()
        reached += arm.success.succeeded
    assert reached >= 4
    # Without the object-state sensor the policy stops at its first step.
    blind = PolicyRunner(
        policy,
        follower=_BlindToyArm(sample_start(rng), sample_target(rng)),  # type: ignore[arg-type]
        cameras=_NoCameras(),  # type: ignore[arg-type]
        fps=30,
    )
    blind.connect()
    with pytest.raises(ValueError, match=PRIVILEGED_LABEL):
        blind.step()


def test_observation_has_environment_state_only_from_followers_that_provide_it():
    arm = _ToyArm(np.zeros(6), np.asarray([0.2, 0.0, 0.1]))
    obs = _build_observation(arm, {})  # type: ignore[arg-type]
    env = obs["observation.environment_state"]
    assert env.dtype == np.float32 and env.shape == (3,)
    np.testing.assert_allclose(env, [0.2, 0.0, 0.1], rtol=1e-6)
    arm.environment_state = lambda: None  # type: ignore[method-assign]
    assert "observation.environment_state" not in _build_observation(arm, {})  # type: ignore[arg-type]
    blind = _BlindToyArm(np.zeros(6), np.zeros(3))
    assert "observation.environment_state" not in _build_observation(blind, {})  # type: ignore[arg-type]
    mock = MagicMock()
    mock.read_positions.return_value = {joint: 0.0 for joint in JOINT_NAMES}
    assert set(_build_observation(mock, {})) == {"observation.state"}


def test_joint_dict_policy_checks_every_action():
    class Fixed:
        label = "fixed"

        def __init__(self, action):
            self.action = action

        def reset(self) -> None:
            pass

        def select_action(self, observation):
            return self.action

    action = np.arange(6, dtype=np.float32)
    assert JointDictPolicy(Fixed(action)).select_action({}) == {
        joint: float(index) for index, joint in enumerate(JOINT_NAMES)
    }
    assert JointDictPolicy(Fixed(action)).label == "fixed"
    with pytest.raises(ValueError, match="6 actions"):
        JointDictPolicy(Fixed(np.zeros(5))).select_action({})
    with pytest.raises(ValueError, match="non-finite"):
        JointDictPolicy(Fixed(np.full(6, np.nan))).select_action({})


def test_eval_runs_a_state_only_baseline_and_labels_the_report():
    _toy()
    checkpoint, _text = _train("toy/reach", "bc_state", "--state-only")
    run = EvalRun(
        EvalConfig(model_path=str(checkpoint), n_episodes=2, max_seconds=0.1),
        score_fn=lambda episode: episode == 0,
    )
    report = run.run()
    assert (report.label, report.privileged_state) == (STATE_ONLY_LABEL, False)
    assert (report.source, report.successes, report.n, report.fps) == ("mock", 1, 2, 30)
    assert report.task == TASK


def test_baseline_refuses_lerobot_datasets_without_the_lerobot_reader(monkeypatch):
    root = get_settings().datasets_dir / f"v2--{quote('lr/ds', safe='')}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 40, "fps": 30, "features": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("so101.data.loader.lerobot_reader_available", lambda: False)
    with pytest.raises(ValueError, match="lerobot package"):
        validate_training_dataset("lr/ds", POLICY_PRESETS["bc_baseline"])
