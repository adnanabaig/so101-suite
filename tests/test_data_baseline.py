"""The numpy BC baseline: trainer, checkpoint format and policy protocol.

Everything here trains on the in-memory toy reach task (no disk, no torch),
so the module runs in a few seconds.
"""

from __future__ import annotations

import errno
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from so101 import JOINT_NAMES, atomic_io
from so101.hardware.safety import DEFAULT_LIMITS
from so101.policy import baseline as baseline_module
from so101.policy.baseline import (
    BASELINE_TYPE,
    MANIFEST_NAME,
    PRIVILEGED_LABEL,
    STATE_ONLY_LABEL,
    TRAIN_LOG_NAME,
    WEIGHTS_NAME,
    BaselinePolicy,
    FeatureSpec,
    describe_baseline_checkpoint,
    is_baseline_checkpoint,
    load_baseline_model,
    load_baseline_policy,
    manifest_metadata,
    save_baseline_checkpoint,
)
from so101.policy.checkpoint import InvalidCheckpointError, PolicyBackendError
from so101.training import toy_reach
from so101.training.bc_baseline import (
    BCConfig,
    TrainingCancelled,
    fit_bc,
    split_episodes,
    validate_bc_args,
)
from so101.training.preflight import InvalidTrainingArgsError

FAST = BCConfig(steps=1500, batch_size=128, seed=0, hidden=(64, 64), eval_every=50)
LOW = np.asarray([DEFAULT_LIMITS[joint][0] for joint in JOINT_NAMES], dtype=np.float32)
HIGH = np.asarray([DEFAULT_LIMITS[joint][1] for joint in JOINT_NAMES], dtype=np.float32)


@pytest.fixture(scope="module")
def episodes() -> list[toy_reach.ToyEpisode]:
    return toy_reach.generate_toy_reach_episodes(40, seed=0)


@pytest.fixture(scope="module")
def privileged(episodes):
    return fit_bc(toy_reach.toy_training_data(episodes), FAST)


@pytest.fixture(scope="module")
def state_only(episodes):
    return fit_bc(toy_reach.toy_training_data(episodes, privileged=False), FAST)


def _weights_bytes(result) -> bytes:
    return b"".join(array.tobytes() for array in result.model.mlp.arrays().values())


def _observation(state, target=None) -> dict[str, np.ndarray]:
    obs = {"observation.state": np.asarray(state, dtype=np.float32)}
    if target is not None:
        obs["observation.environment_state"] = np.asarray(target, dtype=np.float32)
    return obs


def test_trainer_learns_far_below_the_mean_and_hold_predictors(privileged) -> None:
    metrics = privileged.metrics
    assert metrics["evaluated_on"] == "held_out"
    assert privileged.val_episodes and not set(privileged.val_episodes) & set(
        privileged.train_episodes
    )
    # Held-out action error versus predicting the mean action or "stay put".
    assert metrics["mae"] < 0.1 * metrics["mean_predictor_mae"]
    assert metrics["mae"] < 0.75 * metrics["hold_predictor_mae"]
    log = privileged.train_log
    assert log[0]["train_loss"] > 2 * log[-1]["train_loss"]
    assert all("val_loss" in entry for entry in log)
    assert privileged.label == PRIVILEGED_LABEL


def test_training_is_seeded_and_deterministic(episodes, privileged) -> None:
    again = fit_bc(toy_reach.toy_training_data(episodes), FAST)
    assert _weights_bytes(again) == _weights_bytes(privileged)
    assert again.val_episodes == privileged.val_episodes
    other = fit_bc(toy_reach.toy_training_data(episodes), BCConfig(**{**FAST.to_dict(), "seed": 1}))
    assert _weights_bytes(other) != _weights_bytes(privileged)


def test_checkpoint_round_trips_bit_identically(tmp_path: Path, privileged) -> None:
    first = save_baseline_checkpoint(
        tmp_path / "a" / "pretrained_model",
        privileged.model,
        extra=privileged.extra,
        train_log=privileged.train_log,
    )
    assert sorted(p.name for p in first.iterdir()) == sorted(
        [MANIFEST_NAME, WEIGHTS_NAME, TRAIN_LOG_NAME]
    )
    assert [p.name for p in first.parent.iterdir()] == ["pretrained_model"]  # no staging left
    model, manifest = load_baseline_model(first)
    for name, array in privileged.model.mlp.arrays().items():
        loaded = model.mlp.arrays()[name]
        assert loaded.dtype == np.float32 and loaded.tobytes() == array.tobytes()
    for field in ("input_mean", "input_std", "output_mean", "output_std"):
        original = getattr(privileged.model.normalization, field)
        assert getattr(model.normalization, field).tobytes() == original.tobytes()
    assert model.spec == privileged.model.spec
    # Saving the loaded checkpoint again reproduces every file byte for byte.
    second = save_baseline_checkpoint(
        tmp_path / "b" / "pretrained_model",
        model,
        extra=manifest_metadata(manifest),
        train_log=[
            json.loads(line)
            for line in (first / TRAIN_LOG_NAME).read_text(encoding="utf-8").splitlines()
        ],
    )
    for name in (MANIFEST_NAME, WEIGHTS_NAME, TRAIN_LOG_NAME):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert (
        manifest["weights"]["sha256"]
        == hashlib.sha256((first / WEIGHTS_NAME).read_bytes()).hexdigest()
    )
    assert manifest["type"] == BASELINE_TYPE and manifest["privileged_state"] is True
    assert manifest["label"] == PRIVILEGED_LABEL
    assert manifest["dataset"]["repo_id"] == "toy/reach"
    assert manifest["dataset"]["environment_state"] == list(toy_reach.ENV_STATE_NAMES)
    assert manifest["training"]["config"]["seed"] == 0
    assert set(manifest["metrics"]) >= {"mae", "mean_predictor_mae", "best_step"}
    with pytest.raises(FileExistsError):
        save_baseline_checkpoint(first, model)


def test_checkpoint_fsyncs_staged_directory_before_publish_and_parent_after(
    tmp_path: Path, privileged, monkeypatch
) -> None:
    events: list[tuple] = []
    real_replace = baseline_module.os.replace

    def fsync_directory(path: Path) -> None:
        events.append(("fsync", Path(path)))

    def replace(source: Path, target: Path) -> None:
        events.append(("replace", Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(atomic_io, "fsync_directory", fsync_directory)
    monkeypatch.setattr(baseline_module.os, "replace", replace)

    target = tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"
    saved = save_baseline_checkpoint(target, privileged.model)

    assert saved.is_dir()
    stage = events[-3][1]
    assert events == [
        ("fsync", tmp_path),
        ("fsync", tmp_path / "run"),
        ("fsync", tmp_path / "run" / "checkpoints"),
        ("fsync", stage),
        ("replace", stage, target),
        ("fsync", target.parent),
    ]


def test_checkpoint_cleans_staging_if_staged_directory_fsync_has_a_storage_error(
    tmp_path: Path, privileged, monkeypatch
) -> None:
    target = tmp_path / "pretrained_model"

    def fsync_directory(path: Path) -> None:
        if Path(path).name.startswith(".staging-"):
            raise OSError(errno.EIO, "directory sync failed")

    monkeypatch.setattr(atomic_io, "fsync_directory", fsync_directory)

    with pytest.raises(OSError, match="directory sync failed") as caught:
        save_baseline_checkpoint(target, privileged.model)

    assert caught.value.errno == errno.EIO
    assert not target.exists()
    assert not list(tmp_path.glob(".staging-*"))


def test_checkpoint_parent_fsync_storage_error_keeps_complete_published_checkpoint(
    tmp_path: Path, privileged, monkeypatch
) -> None:
    target = tmp_path / "pretrained_model"

    def fsync_directory(path: Path) -> None:
        if Path(path) == target.parent and target.exists():
            raise OSError(errno.ENOSPC, "parent sync failed")

    monkeypatch.setattr(atomic_io, "fsync_directory", fsync_directory)

    with pytest.raises(OSError, match="parent sync failed") as caught:
        save_baseline_checkpoint(target, privileged.model)

    assert caught.value.errno == errno.ENOSPC
    assert sorted(path.name for path in target.iterdir()) == sorted(
        [MANIFEST_NAME, WEIGHTS_NAME, TRAIN_LOG_NAME]
    )
    assert not list(tmp_path.glob(".staging-*"))


@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOSPC])
def test_directory_fsync_propagates_storage_errors(
    tmp_path: Path, monkeypatch, error_number
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(atomic_io.os, "open", lambda path, flags: 73)

    def fail(fd: int) -> None:
        raise OSError(error_number, "storage failure")

    monkeypatch.setattr(atomic_io.os, "fsync", fail)
    monkeypatch.setattr(atomic_io.os, "close", closed.append)

    with pytest.raises(OSError, match="storage failure") as caught:
        atomic_io.fsync_directory(tmp_path)

    assert caught.value.errno == error_number
    assert closed == [73]


def test_directory_fsync_ignores_unsupported_operation(tmp_path: Path, monkeypatch) -> None:
    closed: list[int] = []
    monkeypatch.setattr(atomic_io.os, "open", lambda path, flags: 73)

    def unsupported(fd: int) -> None:
        raise OSError(errno.EINVAL, "unsupported")

    monkeypatch.setattr(atomic_io.os, "fsync", unsupported)
    monkeypatch.setattr(atomic_io.os, "close", closed.append)

    atomic_io.fsync_directory(tmp_path)

    assert closed == [73]


def test_windows_unsupported_directory_handle_is_ignored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", True)

    def unsupported(path: Path, flags: int) -> int:
        raise PermissionError(errno.EACCES, "unsupported")

    monkeypatch.setattr(atomic_io.os, "open", unsupported)

    atomic_io.fsync_directory(tmp_path)


def test_posix_directory_permission_error_is_not_misclassified(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", False)

    def denied(path: Path, flags: int) -> int:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(atomic_io.os, "open", denied)

    with pytest.raises(PermissionError, match="denied"):
        atomic_io.fsync_directory(tmp_path)


def test_policy_follows_the_protocol(tmp_path: Path, privileged) -> None:
    path = save_baseline_checkpoint(tmp_path / "pretrained_model", privileged.model)
    policy = load_baseline_policy(path)
    assert isinstance(policy, BaselinePolicy)
    assert policy.label == PRIVILEGED_LABEL and policy.fps == toy_reach.FPS
    target = [0.2, 0.05, 0.1]
    state = toy_reach.READY_POSE.copy()
    actions = []
    for _ in range(5):
        action = policy.select_action(_observation(state, target))
        assert action.dtype == np.float32 and action.shape == (6,)
        assert np.all(action >= LOW) and np.all(action <= HIGH)
        actions.append(action)
        state = toy_reach.track(state, action)
    policy.reset()
    replay = toy_reach.READY_POSE.copy()
    for expected in actions:
        action = policy.select_action(_observation(replay, target))
        assert np.array_equal(action, expected)
        replay = toy_reach.track(replay, action)
    # A far-out-of-range reading is clipped to the joint limits, never beyond.
    wild = policy.select_action(_observation([500, -500, 500, -500, 500, 500], target))
    assert np.all(wild >= LOW) and np.all(wild <= HIGH)


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ({"observation.state": np.zeros(6)}, "observation.environment_state"),
        ({"observation.environment_state": np.zeros(3)}, "observation.state"),
        ({"observation.state": np.zeros(5), "observation.environment_state": np.zeros(3)}, "6"),
        (
            {"observation.state": np.zeros(6), "observation.environment_state": np.zeros(2)},
            "3 values",
        ),
        (
            {"observation.state": np.full(6, np.nan), "observation.environment_state": np.zeros(3)},
            "non-finite",
        ),
    ],
)
def test_policy_rejects_bad_observations(privileged, observation, message) -> None:
    policy = BaselinePolicy(privileged.model)
    with pytest.raises(ValueError, match=message):
        policy.select_action(observation)


def test_privileged_policy_without_env_state_names_its_label(privileged) -> None:
    with pytest.raises(ValueError, match=PRIVILEGED_LABEL):
        BaselinePolicy(privileged.model).select_action(_observation(np.zeros(6)))


def test_state_only_policy_ignores_env_state(state_only) -> None:
    policy = BaselinePolicy(state_only.model)
    assert policy.label == STATE_ONLY_LABEL and not policy.privileged_state
    with_env = policy.select_action(_observation(toy_reach.READY_POSE, [0.2, 0.0, 0.1]))
    policy.reset()
    without = policy.select_action(_observation(toy_reach.READY_POSE))
    assert np.array_equal(with_env, without)


def test_step_features_match_training_features() -> None:
    rng = np.random.default_rng(7)
    spec = FeatureSpec(env_state_names=("a", "b"), history=3, time_feature="time", fps=15)
    states = rng.normal(0, 20, (12, 6)).astype(np.float32)
    env = rng.normal(0, 1, (12, 2)).astype(np.float32)
    batch = spec.episode_inputs(states, env)
    assert batch.shape == (12, spec.input_dim) and spec.input_dim == 6 + 12 + 2 + 1
    window = [states[0]] * spec.history
    for i in range(12):
        window = [*window[1:], states[i]] if i else window
        assert np.array_equal(spec.step_inputs(window, env[i], i), batch[i]), i
    assert spec.input_names[-1] == "time_s" and "env.b" in spec.input_names


def test_history_and_time_policy_matches_offline_predictions(episodes) -> None:
    config = BCConfig(**{**FAST.to_dict(), "steps": 200, "history": 3, "time_feature": "time"})
    result = fit_bc(toy_reach.toy_training_data(episodes[:6]), config)
    episode = episodes[0]
    env = np.tile(episode.target.astype(np.float32), (episode.state.shape[0], 1))
    offline = result.model.predict(
        result.model.spec.episode_inputs(episode.state, env), episode.state
    )
    policy = BaselinePolicy(result.model)
    for i in range(episode.state.shape[0]):
        online = policy.select_action(_observation(episode.state[i], episode.target))
        np.testing.assert_allclose(online, offline[i], rtol=0, atol=1e-4)


def test_closed_loop_privileged_reaches_and_state_only_does_not(privileged, state_only) -> None:
    good = toy_reach.evaluate_toy_reach(BaselinePolicy(privileged.model), 40, seed=123)
    blind = toy_reach.evaluate_toy_reach(
        BaselinePolicy(state_only.model), 40, seed=123, provide_env_state=False
    )
    assert good.success_rate >= 0.8, good
    assert blind.success_rate <= 0.3, blind
    assert good.ci95[0] > blind.ci95[1]


def test_damaged_checkpoints_fail_closed(tmp_path: Path, privileged) -> None:
    path = save_baseline_checkpoint(tmp_path / "ok" / "pretrained_model", privileged.model)
    manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))

    def variant(name: str, mutate) -> Path:
        copy = tmp_path / name / "pretrained_model"
        copy.mkdir(parents=True)
        for file in path.iterdir():
            (copy / file.name).write_bytes(file.read_bytes())
        mutate(copy)
        return copy

    def write_manifest(copy: Path, **changes) -> None:
        (copy / MANIFEST_NAME).write_text(json.dumps({**manifest, **changes}), encoding="utf-8")

    def flip_a_weight_byte(copy: Path) -> None:
        data = bytearray((copy / WEIGHTS_NAME).read_bytes())
        data[len(data) // 2] ^= 0xFF
        (copy / WEIGHTS_NAME).write_bytes(bytes(data))

    tampered = variant("tampered", flip_a_weight_byte)
    relabeled = variant("relabeled", lambda c: write_manifest(c, privileged_state=False))
    newer = variant("newer", lambda c: write_manifest(c, version=99))
    garbled = variant(
        "garbled", lambda c: (c / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    )
    missing = variant("missing", lambda c: (c / WEIGHTS_NAME).unlink())
    for broken, message in (
        (tampered, "sha256"),
        (relabeled, "privileged_state"),
        (newer, "newer"),
        (garbled, "unreadable"),
        (missing, "cannot read"),
    ):
        assert is_baseline_checkpoint(broken)
        with pytest.raises(InvalidCheckpointError, match=message):
            load_baseline_policy(broken)
    assert not is_baseline_checkpoint(tmp_path)
    with pytest.raises(InvalidCheckpointError, match="weights.npz is missing"):
        describe_baseline_checkpoint(missing)


def test_privileged_checkpoint_refuses_every_backend_but_sim(tmp_path: Path, privileged) -> None:
    path = save_baseline_checkpoint(tmp_path / "pretrained_model", privileged.model)
    for backend in ("real", "mock"):
        with pytest.raises(PolicyBackendError, match=PRIVILEGED_LABEL) as caught:
            load_baseline_policy(path, backend=backend)
        assert f"refuses the {backend} backend" in str(caught.value)
        assert "--state-only" in str(caught.value)
    assert load_baseline_policy(path, backend="sim").privileged_state
    description = describe_baseline_checkpoint(path)
    assert description.privileged_state and description.label == PRIVILEGED_LABEL
    assert description.env_state_names == toy_reach.ENV_STATE_NAMES


def test_state_only_checkpoint_loads_on_every_backend(tmp_path: Path, state_only) -> None:
    path = save_baseline_checkpoint(tmp_path / "pretrained_model", state_only.model)
    for backend in ("real", "mock", "sim", None):
        assert load_baseline_policy(path, backend=backend).label == STATE_ONLY_LABEL


def test_cancelled_training_returns_nothing(episodes) -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TrainingCancelled):
        fit_bc(toy_reach.toy_training_data(episodes[:4]), FAST, cancel_event=cancel)


def test_episode_split_is_by_episode_and_keeps_one_held_out() -> None:
    rng = np.random.default_rng(0)
    assert split_episodes(1, 0.2, rng) == ([0], [])
    assert split_episodes(5, 0.0, rng) == ([0, 1, 2, 3, 4], [])
    train, held_out = split_episodes(2, 0.2, np.random.default_rng(0))
    assert len(train) == 1 and len(held_out) == 1
    train, held_out = split_episodes(10, 0.2, np.random.default_rng(0))
    assert len(held_out) == 2 and sorted(train + held_out) == list(range(10))


def test_bc_args_are_parsed_and_validated() -> None:
    parsed = validate_bc_args(
        {
            "seed": "7",
            "episodes": "all",
            "state_only": "true",
            "hidden": "32,16",
            "activation": "tanh",
            "lr": "3e-4",
            "val_fraction": "0",
            "history": "2",
            "time_feature": "time",
            "target": "absolute",
        }
    )
    assert parsed["hidden"] == (32, 16) and parsed["state_only"] is True
    assert parsed["episodes"] == "all" and parsed["lr"] == pytest.approx(3e-4)
    for bad in (
        {"steps": "10"},
        {"policy.type": "act"},
        {"seed": "-1"},
        {"hidden": "0"},
        {"hidden": "8,8,8,8,8"},
        {"lr": "nan"},
        {"history": "99"},
        {"episodes": "failures"},
        {"state_only": "maybe"},
    ):
        with pytest.raises(InvalidTrainingArgsError):
            validate_bc_args(bad)
