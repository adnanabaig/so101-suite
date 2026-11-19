"""A saved simulated episode must never silently become trainable as unlabeled."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from so101 import JOINT_NAMES
from so101.cli_sim import LabelNotStoredError
from so101.data import inspection
from so101.data.dataset import (
    DatasetBusyError,
    dataset_operation_lock,
    dataset_read_hold,
    get_dataset,
)
from so101.data.inspection import SIM_OUTCOME_SOURCE, persist_sim_outcome
from so101.data.loader import selected_episodes
from so101.data.recorder import EpisodeRecorder, RecordingConfig
from so101.services import robot_manager as manager_module
from so101.services.robot_manager import Mode, RobotManager
from so101.teleop.controller import TeleopStep
from so101.training.preflight import plan_episodes


def _recorder(repo_id: str) -> EpisodeRecorder:
    joints = {f"{joint}.pos": 0.0 for joint in JOINT_NAMES}
    recorder = EpisodeRecorder(RecordingConfig(repo_id=repo_id, camera_keys=[]))
    recorder.add(TeleopStep(t=1.0, observation=joints, action=joints))
    return recorder


def _manager(recorder: EpisodeRecorder) -> tuple[RobotManager, list[tuple[str, str]]]:
    manager = object.__new__(RobotManager)
    manager.settings = object()
    manager.mode = Mode.RECORDING
    manager._lock = threading.RLock()
    manager._io_lock = threading.Lock()
    manager._episode_write_condition = threading.Condition(manager._lock)
    manager._episode_writes = 0
    manager._episode_write_error = None
    manager._episode_grade = {"success": False, "metrics": {"distance_m": 0.5}}
    manager._episode_grade_error = None
    manager._episode_active = True
    manager._episode_saving = False
    manager._episode_starting = False
    manager._rec_cfg = recorder.cfg
    manager._recorder = recorder
    events: list[tuple[str, str]] = []
    manager._event = lambda event_type, message, **_kwargs: events.append((event_type, message))
    return manager, events


def _lock_is_free(repo_id: str) -> bool:
    lock = dataset_operation_lock(repo_id)
    if not lock.acquire(blocking=False):
        return False
    lock.release()
    return True


def _threaded_save(
    manager: RobotManager,
) -> tuple[threading.Thread, dict[str, Any], list[Exception]]:
    result: dict[str, Any] = {}
    errors: list[Exception] = []

    def save() -> None:
        try:
            result.update(manager.save_episode())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=save)
    thread.start()
    return thread, result, errors


def _signal_condition_wait(manager: RobotManager, monkeypatch) -> threading.Event:
    wait_entered = threading.Event()
    original_wait = manager._episode_write_condition.wait

    def wait(*args, **kwargs):
        wait_entered.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(manager._episode_write_condition, "wait", wait)
    return wait_entered


def test_shared_sim_label_writer_keeps_a_human_label(monkeypatch):
    calls = []

    def human_wins(repo_id, episode, outcome, notes="", *, source="human", detail=None):
        calls.append((repo_id, episode, outcome, notes, source, detail))
        return False

    monkeypatch.setattr(inspection, "set_episode_outcome", human_wins)
    detail = {"distance_m": 0.01}
    assert persist_sim_outcome("test/human", 2, True, detail) == (
        "success",
        "kept the human label",
    )
    assert calls == [("test/human", 2, "success", "", SIM_OUTCOME_SOURCE, detail)]


def test_sim_save_fails_loudly_when_outcome_cannot_be_persisted(monkeypatch):
    repo_id = "test/sim_label_failure"
    recorder = _recorder(repo_id)
    manager, events = _manager(recorder)

    monkeypatch.setattr(manager_module, "current_backend", lambda _settings: "sim")
    failure = OSError("disk full")

    def cannot_store(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(inspection, "_set_episode_outcome_locked", cannot_store)

    with pytest.raises(LabelNotStoredError) as caught:
        manager.save_episode()

    assert caught.value.cause is failure
    assert "episode 0 of test/sim_label_failure was saved" in str(caught.value)
    assert "reads as unlabeled" in str(caught.value)
    assert (recorder.root / "episode_000000" / "episode.json").is_file()
    assert not list(recorder.root.glob("incomplete_*"))
    assert manager._recorder is None
    assert manager._episode_active is False and manager._episode_saving is False
    assert events[-1][0] == "recording.sim_label_failed"
    assert _lock_is_free(repo_id)


def test_sim_grade_failure_is_recoverable_after_the_episode_is_saved(monkeypatch):
    repo_id = "test/sim_grade_failure"
    recorder = _recorder(repo_id)
    manager, events = _manager(recorder)
    failure = RuntimeError("predicate failed")

    monkeypatch.setattr(manager_module, "current_backend", lambda _settings: "sim")
    manager._episode_grade = None
    manager._episode_grade_error = failure

    with pytest.raises(LabelNotStoredError) as caught:
        manager.save_episode()

    assert caught.value.cause is failure and caught.value.outcome is None
    assert "episode 0 of test/sim_grade_failure was saved" in str(caught.value)
    assert "could not be graded" in str(caught.value)
    assert (recorder.root / "episode_000000" / "episode.json").is_file()
    assert manager._recorder is None
    assert manager._episode_active is False and manager._episode_saving is False
    assert events[-1][0] == "recording.sim_label_failed"
    assert _lock_is_free(repo_id)


def test_sim_save_snapshots_before_io_and_holds_state_and_dataset_through_label(
    monkeypatch,
):
    repo_id = "test/sim_save_transaction"
    recorder = _recorder(repo_id)
    manager, _events = _manager(recorder)
    manager._episode_grade = {"success": True, "metrics": {"marker": 1}}

    def late_grade_is_a_bug():
        raise AssertionError("save re-graded the post-frame simulation world")

    monkeypatch.setattr(manager_module, "current_backend", lambda _settings: "sim")
    monkeypatch.setattr(
        manager_module,
        "_sim_devices",
        lambda: SimpleNamespace(on_episode_end=late_grade_is_a_bug),
    )

    published = threading.Event()
    release_save = threading.Event()
    label_started = threading.Event()
    release_label = threading.Event()
    original_save = recorder.save_episode
    original_persist = manager_module.persist_sim_outcome

    def blocked_save(*, retain_operation_lock: bool = False) -> int:
        episode = original_save(retain_operation_lock=retain_operation_lock)
        published.set()
        assert release_save.wait(5), "test did not release the blocked save"
        return episode

    def blocked_persist(*args, **kwargs):
        label_started.set()
        assert release_label.wait(5), "test did not release the blocked label"
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(recorder, "save_episode", blocked_save)
    monkeypatch.setattr(manager_module, "persist_sim_outcome", blocked_persist)
    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def save() -> None:
        try:
            result.update(manager.save_episode())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=save)
    thread.start()
    assert published.wait(5), "episode was not published"

    # The world can continue after the close, but its later state must not
    # change this episode's already-captured verdict.
    assert manager._episode_saving and manager._recorder is recorder
    with pytest.raises(RuntimeError, match="previous episode is still finalizing"):
        manager.begin_episode()
    with pytest.raises(DatasetBusyError):
        with dataset_read_hold(repo_id):
            pass
    with pytest.raises(DatasetBusyError):
        selected_episodes(repo_id, "all")
    with pytest.raises(DatasetBusyError):
        plan_episodes(repo_id, "all", get_dataset(repo_id))

    release_save.set()
    assert label_started.wait(5), "label persistence did not start"
    assert manager._episode_saving and manager._recorder is recorder
    with pytest.raises(RuntimeError, match="previous episode is still finalizing"):
        manager.begin_episode()
    with pytest.raises(DatasetBusyError):
        inspection.set_episode_outcome(repo_id, 0, "failure")

    release_label.set()
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert result["outcome"] == "success"
    manifest = inspection.inspect_dataset(repo_id).episodes[0]
    assert manifest.outcome == "success"
    stored = json.loads(
        (recorder.root / "episode_000000" / "episode.json").read_text(encoding="utf-8")
    )
    assert stored["outcome_detail"] == {"marker": 1}
    assert manager._recorder is None and manager._episode_saving is False
    assert _lock_is_free(repo_id)


def test_sim_save_waits_for_final_write_and_uses_its_newer_grade(monkeypatch):
    repo_id = "test/sim_pending_grade"
    recorder = _recorder(repo_id)
    manager, _events = _manager(recorder)
    manager._episode_grade = {"success": False, "metrics": {"frame": "older"}}
    manager._episode_writes = 1
    monkeypatch.setattr(manager_module, "current_backend", lambda _settings: "sim")
    wait_entered = _signal_condition_wait(manager, monkeypatch)

    thread, result, errors = _threaded_save(manager)
    assert wait_entered.wait(5), "save did not block for the selected final frame"
    assert manager._episode_saving and not (recorder.root / "meta.json").exists()

    # This is the ordering used by the control loop: install the successful
    # frame's grade, then decrement and notify the waiting save.
    with manager._episode_write_condition:
        manager._episode_grade = {"success": True, "metrics": {"frame": "final"}}
        manager._episode_grade_error = None
        manager._episode_writes = 0
        manager._episode_write_condition.notify_all()

    thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert result["outcome"] == "success"
    stored = json.loads(
        (recorder.root / "episode_000000" / "episode.json").read_text(encoding="utf-8")
    )
    assert stored["outcome_detail"] == {"frame": "final"}
    assert manager._recorder is None and manager._episode_saving is False
    assert _lock_is_free(repo_id)


def test_sim_save_final_write_error_publishes_nothing_and_cleans_up(monkeypatch):
    repo_id = "test/sim_pending_error"
    recorder = _recorder(repo_id)
    manager, _events = _manager(recorder)
    manager._episode_writes = 1
    monkeypatch.setattr(manager_module, "current_backend", lambda _settings: "sim")
    wait_entered = _signal_condition_wait(manager, monkeypatch)

    thread, result, errors = _threaded_save(manager)
    assert wait_entered.wait(5), "save did not block for the selected final frame"
    failure = OSError("final camera frame failed")
    with manager._episode_write_condition:
        manager._episode_write_error = failure
        manager._episode_writes = 0
        manager._episode_write_condition.notify_all()

    thread.join(timeout=5)
    assert not thread.is_alive() and not result
    assert len(errors) == 1 and "final recording frame failed" in str(errors[0])
    assert not (recorder.root / "meta.json").exists()
    assert not list(recorder.root.glob("episode_*"))
    incomplete = list(recorder.root.glob("incomplete_*"))
    assert len(incomplete) == 1 and (incomplete[0] / "INCOMPLETE.json").is_file()
    assert manager._recorder is None
    assert manager._episode_active is False and manager._episode_saving is False
    assert _lock_is_free(repo_id)
