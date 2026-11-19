"""Manager halves of safety-B1 and data-B13.

- safety-B1 (docs/audit/v0.1/safety.md): every ``start_*`` refuses inside the
  manager while the robot is E-stopped or faulted, so no router or agent path
  can forget the check. Nothing is loaded, connected or energized first.
- data-B13 (docs/audit/v0.1/data.md): ``start_rollout(..., policy=...)`` runs a
  policy the caller already loaded and never reloads it. The rollout clock
  starts at the first action: no sample is published before it, and the first
  frame does not report a rate that includes connect or load time. Gate's eval
  keys its trial clock on that first ``sample_seq`` change.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from so101.data.recorder import RecordingConfig
from so101.policy.server import MockPolicy
from so101.services import robot_manager as robot_manager_module
from so101.services.robot_manager import Mode, RobotManager, get_manager

OPERATOR = "test-operator"


class _GatedPolicy:
    """A policy whose calls can be held open, to pin the loop at one point."""

    def __init__(self, gated_calls: int = 0) -> None:
        self.inner = MockPolicy()
        self.resets = 0
        self.calls = 0
        self.entered = [threading.Event() for _ in range(gated_calls)]
        self.release = [threading.Event() for _ in range(gated_calls)]

    def reset(self) -> None:
        self.resets += 1
        self.inner.reset()

    def select_action(self, observation):
        index = self.calls
        self.calls += 1
        if index < len(self.entered):
            self.entered[index].set()
            assert self.release[index].wait(timeout=5), "test never released the policy"
        return self.inner.select_action(observation)

    def release_all(self) -> None:
        for event in self.release:
            event.set()


def _wait_for(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def manager() -> RobotManager:
    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    return manager


@pytest.fixture
def no_reload(monkeypatch) -> list[str]:
    """Fail loudly if the manager tries to load a checkpoint itself."""

    loads: list[str] = []

    def refuse(model_path, *args, **kwargs):
        loads.append(str(model_path))
        raise AssertionError(f"the manager reloaded {model_path!r}")

    monkeypatch.setattr(robot_manager_module, "load_policy", refuse)
    return loads


def _starters(policy: _GatedPolicy) -> dict[str, Callable[[RobotManager], None]]:
    return {
        "teleop": lambda m: m.start_teleop(30, operator_id=OPERATOR),
        "recording": lambda m: m.start_recording(
            RecordingConfig(repo_id="local/refused"), operator_id=OPERATOR
        ),
        "rollout-load": lambda m: m.start_rollout("models/x", 30, operator_id=OPERATOR),
        "rollout-preloaded": lambda m: m.start_rollout(
            "models/x", 30, operator_id=OPERATOR, policy=policy
        ),
        "calibration": lambda m: m.claim_calibration("follower"),
    }


STARTERS = ["teleop", "recording", "rollout-load", "rollout-preloaded", "calibration"]


def _assert_nothing_started(manager: RobotManager, policy: _GatedPolicy) -> None:
    assert manager.mode is Mode.IDLE
    assert manager._thread is None
    assert manager.cal_in_progress is False
    assert policy.resets == 0 and policy.calls == 0
    follower = manager._follower
    assert follower is None or not follower.connected


@pytest.mark.parametrize("starter", STARTERS)
def test_safety_b1_every_start_refuses_while_estopped(
    manager: RobotManager, no_reload: list[str], starter: str
):
    policy = _GatedPolicy()
    manager.estop("test E-stop")

    with pytest.raises(RuntimeError, match="E-STOP is latched"):
        _starters(policy)[starter](manager)

    _assert_nothing_started(manager, policy)
    assert no_reload == []


@pytest.mark.parametrize("starter", STARTERS)
def test_safety_b1_every_start_refuses_while_faulted(
    manager: RobotManager, no_reload: list[str], starter: str
):
    policy = _GatedPolicy()
    manager._set_fault(RuntimeError("injected bus fault"), source="test")

    with pytest.raises(RuntimeError, match="injected bus fault"):
        _starters(policy)[starter](manager)

    _assert_nothing_started(manager, policy)
    assert no_reload == []


@pytest.mark.parametrize("starter", ["teleop", "recording"])
def test_rejected_data_start_leaves_the_existing_recording_session_untouched(
    manager: RobotManager, starter: str
):
    original = RecordingConfig(repo_id="keep/original", fps=17)
    manager._rec_cfg = original
    manager._episode_active = True
    manager._episode_frames = 23
    manager._set_fault(RuntimeError("reject this start"), source="test")

    with pytest.raises(RuntimeError, match="reject this start"):
        _starters(_GatedPolicy())[starter](manager)

    assert manager._rec_cfg is original
    assert manager._episode_active is True
    assert manager._episode_frames == 23


def test_data_b13_preloaded_policy_runs_without_a_reload(
    manager: RobotManager, no_reload: list[str]
):
    policy = _GatedPolicy()
    seq0 = manager.telemetry()["sample_seq"]

    manager.start_rollout("not/a/checkpoint", 30, operator_id=OPERATOR, policy=policy)
    try:
        assert _wait_for(lambda: manager.telemetry()["sample_seq"] - seq0 >= 3)
        assert manager.mode is Mode.ROLLOUT
    finally:
        assert manager.stop() is True

    assert no_reload == []
    assert manager._fault is None
    # Every rollout starts from a fresh policy state, preloaded or not.
    assert policy.resets == 1
    assert policy.calls >= 3


def test_data_b13_preloaded_policy_is_reset_for_every_rollout(
    manager: RobotManager, no_reload: list[str]
):
    policy = _GatedPolicy()
    for trial in range(1, 3):
        seq0 = manager.telemetry()["sample_seq"]
        manager.start_rollout("not/a/checkpoint", 30, operator_id=OPERATOR, policy=policy)
        try:
            assert _wait_for(lambda seq0=seq0: manager.telemetry()["sample_seq"] > seq0)
        finally:
            assert manager.stop() is True
        assert policy.resets == trial


def test_data_b13_the_rollout_clock_starts_at_the_first_action(
    manager: RobotManager, no_reload: list[str]
):
    policy = _GatedPolicy(gated_calls=2)
    seq0 = manager.telemetry()["sample_seq"]

    manager.start_rollout("not/a/checkpoint", 30, operator_id=OPERATOR, policy=policy)
    try:
        # The loop is connected and waiting on the first inference.
        assert policy.entered[0].wait(timeout=3)
        time.sleep(0.25)
        assert manager.telemetry()["sample_seq"] == seq0, "a sample was published before any action"

        policy.release[0].set()
        assert policy.entered[1].wait(timeout=3)
        first = manager.telemetry()
        assert first["sample_seq"] == seq0 + 1
        assert first["action"], "the first sample must carry the first action"
        # The first frame has no predecessor: no rate that includes the wait.
        assert first["fps"] == 0.0

        policy.release[1].set()
        assert _wait_for(lambda: manager.telemetry()["sample_seq"] >= seq0 + 2)
        assert manager.telemetry()["fps"] > 0.0
    finally:
        policy.release_all()
        assert manager.stop() is True


def test_data_b13_a_preloaded_policy_must_be_a_policy(manager: RobotManager):
    with pytest.raises(TypeError, match="select_action"):
        manager.start_rollout("models/x", 30, operator_id=OPERATOR, policy=object())
    assert manager.mode is Mode.IDLE


# --- task pass-through for gate's eval and language-conditioned policies ----
#
# data-learn binds the instruction at load time (``load_policy(path, task=...)``
# stores it on the policy, which reads ``policy.task`` on every inference). The
# manager hands the task to the loader, or onto a preloaded policy that has a
# ``task`` attribute. ``None`` (or blank) keeps today's calls exactly.


class _TaskPolicy(MockPolicy):
    """A preloaded policy that, like data-learn's LeRobotPolicy, reads ``task``."""

    def __init__(self) -> None:
        super().__init__()
        self.task: str | None = "checkpoint default"
        self.seen: list[str | None] = []

    def select_action(self, observation):
        self.seen.append(self.task)
        return super().select_action(observation)


@pytest.fixture
def loader_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_load_policy(model_path, *, task=None, **kwargs):
        calls.append({"model_path": str(model_path), "kwargs": {"task": task, **kwargs}})
        return MockPolicy()

    monkeypatch.setattr(robot_manager_module, "load_policy", fake_load_policy)
    return calls


def _run_briefly(manager: RobotManager, **kwargs) -> None:
    seq0 = manager.telemetry()["sample_seq"]
    manager.start_rollout("models/x", 30, operator_id=OPERATOR, **kwargs)
    try:
        assert _wait_for(lambda: manager.telemetry()["sample_seq"] > seq0)
    finally:
        assert manager.stop() is True
    assert manager._fault is None


def test_rollout_task_reaches_the_policy_loader(manager: RobotManager, loader_calls):
    _run_briefly(manager, task="put the cube in the bowl")

    assert loader_calls == [
        {"model_path": "models/x", "kwargs": {"task": "put the cube in the bowl"}}
    ]


@pytest.mark.parametrize("task", [None, "", "   "], ids=["none", "empty", "blank"])
def test_rollout_without_a_task_keeps_todays_loader_call(
    manager: RobotManager, monkeypatch, task: str | None
):
    calls: list[tuple] = []

    def legacy_load_policy(model_path):  # the v0.1 signature: no task keyword
        calls.append((str(model_path),))
        return MockPolicy()

    monkeypatch.setattr(robot_manager_module, "load_policy", legacy_load_policy)

    _run_briefly(manager, task=task)

    assert calls == [("models/x",)]


def test_rollout_task_is_bound_to_a_preloaded_policy(manager: RobotManager, no_reload):
    policy = _TaskPolicy()

    _run_briefly(manager, policy=policy, task="stack the blocks")

    assert policy.seen and set(policy.seen) == {"stack the blocks"}
    assert no_reload == []


def test_rollout_without_a_task_leaves_a_preloaded_policy_alone(manager: RobotManager, no_reload):
    policy = _TaskPolicy()

    _run_briefly(manager, policy=policy)

    assert set(policy.seen) == {"checkpoint default"}


def test_rollout_task_is_refused_up_front_when_the_loader_cannot_bind_it(
    manager: RobotManager, monkeypatch
):
    monkeypatch.setattr(robot_manager_module, "load_policy", lambda model_path: MockPolicy())

    with pytest.raises(TypeError, match="cannot bind a task"):
        manager.start_rollout("models/x", 30, operator_id=OPERATOR, task="wave")

    assert manager.mode is Mode.IDLE and manager._thread is None
