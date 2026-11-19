"""Eval gates (C3, safety-B1) and eval durability (data-B6/B7/B13/B14/B18).

Audit reports: docs/audit/v0.1/safety.md (B1) and docs/audit/v0.1/data.md
(B6, B7, B8, B13, B14, B18).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from so101.config import get_settings
from so101.services import robot_manager as robot_manager_module
from so101.services.app import create_app
from so101.services.robot_manager import RobotManager, get_manager
from so101.services.routers import eval as eval_router
from so101.services.schemas import StartEvalRequest
from so101.services.store import get_store
from so101.training.run_manifest import RUN_MANIFEST_NAME

OPERATOR = "test-operator"
EVIL = {"Origin": "http://evil.example", "Content-Type": "text/plain"}


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _mock_checkpoint(
    run: str = "eval_run",
    *,
    fps: int = 30,
    task: str | None = None,
    policy_type: str = "act",
) -> Path:
    """A registered mock checkpoint with the defaults the eval gate resolves."""

    run_dir = get_settings().outputs_dir / run
    path = run_dir / "checkpoints" / "last" / "pretrained_model"
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"type": policy_type, "mock": True}), encoding="utf-8"
    )
    (run_dir / RUN_MANIFEST_NAME).write_text(
        json.dumps({"fps": fps, "task": task}), encoding="utf-8"
    )
    return path


def _start(client: TestClient, **overrides: Any) -> None:
    body = {
        "model_path": str(_mock_checkpoint()),
        "n_episodes": 1,
        "max_seconds": 0.05,
        "operator_id": OPERATOR,
        **overrides,
    }
    response = client.post("/api/eval/start", json=body)
    assert response.status_code == 200, response.text


def _wait_state(
    client: TestClient, predicate: Callable[[dict], bool], timeout: float = 5.0
) -> dict:
    deadline = time.monotonic() + timeout
    state: dict = {}
    while time.monotonic() < deadline:
        # Keep the supervision heartbeat fresh like an open operator tab.
        get_manager().operator_heartbeat(OPERATOR)
        state = client.get("/api/eval/state").json()
        if predicate(state):
            return state
        time.sleep(0.01)
    raise AssertionError(f"eval state never matched: {state}")


def _phase(name: str, episode: int | None = None) -> Callable[[dict], bool]:
    def matches(state: dict) -> bool:
        return state["phase"] == name and (episode is None or state["episode"] == episode)

    return matches


def _finished(phase: str) -> Callable[[dict], bool]:
    return lambda state: state["phase"] == phase and bool(state["report_path"])


def _score(client: TestClient, success: bool = True, operator: str = OPERATOR):
    return client.post("/api/eval/score", json={"operator_id": operator, "success": success})


def _continue(client: TestClient, operator: str = OPERATOR):
    return client.post("/api/eval/continue", json={"operator_id": operator})


def _report(state: dict) -> dict:
    return json.loads(Path(state["report_path"]).read_text())


def _reports() -> list[Path]:
    return sorted(get_settings().outputs_dir.glob("eval_*.json"))


def _fail_audit_event(monkeypatch: pytest.MonkeyPatch, event_type: str):
    """Inject one SQLite-class failure at the durable event-store seam."""

    store = get_store()
    original = store.append_event

    def append(seen_type: str, *args: Any, **kwargs: Any):
        if seen_type == event_type:
            if event_type != "eval.aborted":
                assert kwargs.get("durable") is True
            else:
                assert kwargs.get("durable") is not True
            raise sqlite3.OperationalError("injected SQLite disk I/O error")
        return original(seen_type, *args, **kwargs)

    monkeypatch.setattr(store, "append_event", append)
    return original


# --- C3 / safety-B1: score and continue are attributed and supervised ------------


def test_safety_b1_data_b8_cross_origin_score_and_continue_rejected(client: TestClient):
    """Repros 02 and 13: a cross-site simple POST must not score or start motion."""

    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    body = json.dumps({"operator_id": OPERATOR, "success": True})
    for path in ("/api/eval/score", "/api/eval/score?success=true"):
        response = client.post(path, content=body, headers=EVIL)
        assert response.status_code == 403, response.text
        assert response.json() == {"detail": "cross-origin request rejected"}
    state = client.get("/api/eval/state").json()
    assert state["phase"] == "awaiting_score"
    assert state["successes"] == 0

    assert _score(client).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    response = client.post(
        "/api/eval/continue", content=json.dumps({"operator_id": OPERATOR}), headers=EVIL
    )
    assert response.status_code == 403, response.text
    time.sleep(0.1)
    assert client.get("/api/eval/state").json()["phase"] == "awaiting_reset"
    assert get_manager().mode.value == "idle"


def test_safety_b1_query_and_bodyless_gates_are_gone(client: TestClient):
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert client.post("/api/eval/score?success=true").status_code == 422
    assert client.post("/api/eval/score", json={"success": True}).status_code == 422
    assert client.post("/api/eval/score", json={"operator_id": OPERATOR}).status_code == 422
    assert _score(client).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    assert client.post("/api/eval/continue").status_code == 422
    assert client.post("/api/eval/continue", json={}).status_code == 422
    assert client.get("/api/eval/state").json()["phase"] == "awaiting_reset"


def test_safety_b1_score_requires_the_session_operator(client: TestClient):
    get_manager().operator_heartbeat("intruder")
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    response = _score(client, operator="intruder")
    assert response.status_code == 409
    assert "operator" in response.json()["detail"]
    state = client.get("/api/eval/state").json()
    assert (state["phase"], state["successes"]) == ("awaiting_score", 0)

    accepted = _score(client)
    assert accepted.status_code == 200, accepted.text
    # Both endpoints answer with the EvalState.
    assert {"active", "phase", "episode", "successes", "success_rate"} <= set(accepted.json())
    _wait_state(client, _finished("done"))


def test_safety_b1_score_and_continue_require_a_fresh_heartbeat(client: TestClient):
    manager = get_manager()
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    manager.operator_timeout_ms = 50
    time.sleep(0.1)
    response = _score(client)
    assert response.status_code == 409
    assert "heartbeat" in response.json()["detail"]
    manager.operator_heartbeat(OPERATOR)
    assert _score(client).status_code == 200

    _wait_state(client, _phase("awaiting_reset"))
    time.sleep(0.1)
    response = _continue(client)
    assert response.status_code == 409
    assert "heartbeat" in response.json()["detail"]
    assert client.get("/api/eval/state").json()["phase"] == "awaiting_reset"
    manager.operator_timeout_ms = 2_000
    manager.operator_heartbeat(OPERATOR)
    continued = _continue(client)
    assert continued.status_code == 200, continued.text
    assert continued.json()["phase"] in {"preparing", "rollout", "awaiting_score"}
    client.post("/api/eval/stop")


def test_safety_b1_continue_requires_the_session_operator(client: TestClient):
    get_manager().operator_heartbeat("intruder")
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    response = _continue(client, operator="intruder")
    assert response.status_code == 409
    assert "operator" in response.json()["detail"]
    time.sleep(0.1)
    assert client.get("/api/eval/state").json()["phase"] == "awaiting_reset"
    assert get_manager().mode.value == "idle"
    assert _continue(client).status_code == 200
    _wait_state(client, _phase("awaiting_score", episode=2))
    client.post("/api/eval/stop")


def test_safety_b1_estop_aborts_eval_awaiting_reset_and_reset_cannot_resume(
    client: TestClient,
):
    """Repro 13: E-stop between episodes ends the session for good."""

    _start(client, n_episodes=3)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client, success=True).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))

    assert client.post("/api/robot/estop").status_code == 200
    state = _wait_state(client, _finished("aborted"))
    assert state["active"] is False
    assert state["error"] == "E-stop engaged"

    get_manager().reset_estop(operator_id=OPERATOR)
    response = _continue(client)
    assert response.status_code == 409
    time.sleep(0.1)
    assert get_manager().mode.value == "idle"
    report = _report(state)
    assert report["phase"] == "aborted"
    assert report["error"] == "E-stop engaged"
    assert (report["requested_episodes"], report["scored"], report["successes"]) == (3, 1, 1)


def test_safety_b1_estop_during_rollout_aborts_and_marks_trial_interrupted(
    client: TestClient,
):
    _start(client, n_episodes=2, max_seconds=10)
    _wait_state(client, lambda s: s["phase"] == "rollout" and get_manager().mode.value == "rollout")
    assert client.post("/api/robot/estop").status_code == 200
    state = _wait_state(client, _finished("aborted"))
    assert state["error"] == "E-stop engaged"
    report = _report(state)
    assert report["interrupted"] == 1
    assert report["scored"] == 0
    assert report["per_episode"][0]["success"] is None
    assert report["per_episode"][0]["interrupted"] == "E-stop engaged"


def test_safety_b1_estop_while_awaiting_score_aborts_and_refuses_scores(
    client: TestClient,
):
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert client.post("/api/robot/estop").status_code == 200
    state = _wait_state(client, _finished("aborted"))
    assert state["error"] == "E-stop engaged"
    assert _score(client).status_code == 409
    report = _report(state)
    assert (report["n_episodes"], report["scored"], report["interrupted"]) == (1, 0, 1)


def test_safety_b1_a_score_after_the_trial_is_decided_unscored_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Gate review: a score that arrived after the session read "no score"
    still answered 200 and logged eval.scored, but the trial was recorded as
    interrupted. The phase now changes in the same locked block as that read.
    """
    late: list[str] = []
    original = eval_router._EvalSession._record

    def record(self: Any, number: int, success: Any, duration: float, **kwargs: Any) -> None:
        # The race window, made deterministic: the trial is decided but not
        # yet recorded. A score that lands here must be refused.
        if kwargs.get("interrupted") and not late:
            get_manager().operator_heartbeat(OPERATOR)
            try:
                self.score(OPERATOR, True)
                late.append("accepted")
            except RuntimeError as exc:
                late.append(str(exc))
        original(self, number, success, duration, **kwargs)

    monkeypatch.setattr(eval_router._EvalSession, "_record", record)
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert client.post("/api/robot/estop").status_code == 200
    state = _wait_state(client, _finished("aborted"))
    assert late == ["not awaiting a score"]
    trial = _report(state)["per_episode"][0]
    assert (trial["success"], trial["interrupted"]) == (None, "E-stop engaged")
    assert [e for e in get_store().list_events(limit=500) if e["type"] == "eval.scored"] == []


def test_safety_b1_runtime_fault_aborts_eval(client: TestClient):
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    get_manager()._set_fault("injected servo fault", source="test")
    state = _wait_state(client, _finished("aborted"))
    assert "injected servo fault" in state["error"]


def test_safety_b1_continue_refuses_while_estopped_or_faulted():
    """The gate itself refuses, even before the session thread notices."""

    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    session = eval_router._EvalSession()
    session.active = True
    session.phase = "awaiting_reset"
    session.operator_id = OPERATOR

    manager.estop()
    with pytest.raises(RuntimeError, match="E-stop engaged"):
        session.continue_next(OPERATOR)
    assert session.phase == "awaiting_reset"
    assert not session._continue.is_set()

    manager.reset_estop(operator_id=OPERATOR)
    manager._set_fault("encoder dropout", source="test")
    with pytest.raises(RuntimeError, match="encoder dropout"):
        session.continue_next(OPERATOR)
    assert not session._continue.is_set()


# --- data-B7: shutdown keeps the scored trials ------------------------------------


def test_data_b7_lifespan_shutdown_saves_partial_report_before_manager_shutdown(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, Any] = {}
    original = RobotManager.shutdown

    def shutdown(self: RobotManager) -> None:
        seen["eval_active"] = eval_router.get_session().active
        seen["reports"] = list(_reports())
        original(self)

    monkeypatch.setattr(RobotManager, "shutdown", shutdown)
    with TestClient(create_app()) as client:
        get_manager().operator_heartbeat(OPERATOR)
        _start(client, n_episodes=4)
        _wait_state(client, _phase("awaiting_score"))
        assert _score(client, success=True).status_code == 200
        _wait_state(client, _phase("awaiting_reset"))
    assert seen["eval_active"] is False
    assert len(seen["reports"]) == 1
    report = json.loads(seen["reports"][0].read_text())
    assert report["phase"] == "aborted"
    assert report["error"] == "server shutdown"
    assert (report["requested_episodes"], report["n_episodes"]) == (4, 1)
    assert (report["scored"], report["successes"], report["success_rate"]) == (1, 1, 1.0)
    journal = seen["reports"][0].with_suffix(".jsonl")
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "trial", "finished"]
    assert events[1]["episode"] == 1 and events[1]["success"] is True


# --- required audit ordering and journaling -----------------------------------------


def test_start_refuses_before_motion_when_the_sqlite_audit_write_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    started: list[str] = []
    manager = get_manager()
    monkeypatch.setattr(manager, "start_rollout", lambda *args, **kwargs: started.append("motion"))
    _fail_audit_event(monkeypatch, "eval.start_requested")

    response = client.post(
        "/api/eval/start",
        json={"model_path": str(_mock_checkpoint()), "operator_id": OPERATOR},
    )

    assert response.status_code == 503
    assert "SQLite disk I/O error" in response.json()["detail"]
    assert started == []
    assert manager.mode.value == "idle"
    state = client.get("/api/eval/state").json()
    assert (state["active"], state["phase"], state["model_path"]) == (False, "idle", "")
    assert list(get_settings().outputs_dir.glob("eval_*.json*")) == []


def test_start_audit_describes_a_request_when_the_final_readiness_check_refuses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    session = eval_router.get_session()
    original = session._check_ready
    checks = 0

    def check_ready(manager: RobotManager, operator_id: str) -> None:
        nonlocal checks
        checks += 1
        if checks == 3:  # after journal + eval.start_requested, before thread.start
            raise RuntimeError("injected authority change after audit")
        original(manager, operator_id)

    monkeypatch.setattr(session, "_check_ready", check_ready)
    response = client.post(
        "/api/eval/start",
        json={"model_path": str(_mock_checkpoint()), "operator_id": OPERATOR},
    )

    assert response.status_code == 409
    events = get_store().list_events(limit=100)
    types = [event["type"] for event in events]
    assert "eval.start_requested" in types
    assert "eval.start_refused" in types
    assert "eval.started" not in types
    assert get_manager().mode.value == "idle"
    assert client.get("/api/eval/state").json()["phase"] == "idle"


@pytest.mark.parametrize("blocked_stage", ["journal", "audit"])
def test_transient_estop_during_blocked_start_persistence_permanently_refuses_motion(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    blocked_stage: str,
):
    entered = threading.Event()
    release = threading.Event()
    response: list[Any] = []
    if blocked_stage == "audit":
        original_event = eval_router._required_event

        def blocked_event(event_type: str, *args: Any, **kwargs: Any) -> None:
            if event_type == "eval.start_requested":
                entered.set()
                assert release.wait(2), "test did not release the start audit"
            original_event(event_type, *args, **kwargs)

        monkeypatch.setattr(eval_router, "_required_event", blocked_event)
    else:
        original_journal = eval_router._Journal.write

        def blocked_journal(journal, event: str, **payload: Any) -> None:
            if event == "started":
                entered.set()
                assert release.wait(2), "test did not release the start journal"
            original_journal(journal, event, **payload)

        monkeypatch.setattr(eval_router._Journal, "write", blocked_journal)
    request = threading.Thread(
        target=lambda: response.append(
            client.post(
                "/api/eval/start",
                json={"model_path": str(_mock_checkpoint()), "operator_id": OPERATOR},
            )
        )
    )
    request.start()
    assert entered.wait(2), f"start {blocked_stage} did not block"
    manager = get_manager()
    manager.estop()
    manager.reset_estop(operator_id=OPERATOR)
    release.set()
    request.join(2)

    assert response and response[0].status_code == 409
    assert manager.mode.value == "idle"
    assert client.get("/api/eval/state").json()["phase"] == "idle"


def test_score_refuses_without_mutation_when_the_sqlite_audit_write_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    original = _fail_audit_event(monkeypatch, "eval.score_requested")

    response = _score(client)

    assert response.status_code == 503
    state = client.get("/api/eval/state").json()
    assert (state["phase"], state["successes"], state["scored"]) == (
        "awaiting_score",
        0,
        0,
    )
    assert not eval_router.get_session()._score.is_set()
    monkeypatch.setattr(get_store(), "append_event", original)
    assert _score(client).status_code == 200
    _wait_state(client, _finished("done"))


@pytest.mark.parametrize("transient", ["estop", "fault"])
def test_transient_safety_latch_during_blocked_score_audit_aborts_without_a_score(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, transient: str
):
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    entered = threading.Event()
    release = threading.Event()
    response: list[Any] = []
    original = eval_router._required_event

    def blocked_event(event_type: str, *args: Any, **kwargs: Any) -> None:
        if event_type == "eval.score_requested":
            entered.set()
            assert release.wait(2), "test did not release the score audit"
        original(event_type, *args, **kwargs)

    monkeypatch.setattr(eval_router, "_required_event", blocked_event)
    scoring = threading.Thread(target=lambda: response.append(_score(client, success=True)))
    scoring.start()
    assert entered.wait(2), "score audit did not block"
    manager = get_manager()
    if transient == "estop":
        manager.estop()
        manager.reset_estop(operator_id=OPERATOR)
    else:
        manager._set_fault("injected transient fault", source="test")
        manager.clear_fault(operator_id=OPERATOR)
    release.set()
    scoring.join(2)

    assert response and response[0].status_code == 409
    state = _wait_state(client, _finished("aborted"))
    report = _report(state)
    assert (report["scored"], report["interrupted"]) == (0, 1)
    assert report["per_episode"][0]["success"] is None


def test_continue_refuses_before_next_motion_when_the_sqlite_audit_write_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    original = _fail_audit_event(monkeypatch, "eval.continue_requested")

    response = _continue(client)

    assert response.status_code == 503
    assert client.get("/api/eval/state").json()["phase"] == "awaiting_reset"
    assert get_manager().mode.value == "idle"
    assert not eval_router.get_session()._continue.is_set()
    monkeypatch.setattr(get_store(), "append_event", original)
    assert _continue(client).status_code == 200
    _wait_state(client, _phase("awaiting_score", episode=2))
    client.post("/api/eval/stop")


def test_transient_estop_during_blocked_continue_audit_prevents_the_second_rollout(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    manager = get_manager()
    starts: list[int] = []
    original_start = manager.start_rollout

    def counted_start(*args: Any, **kwargs: Any) -> None:
        starts.append(len(starts) + 1)
        original_start(*args, **kwargs)

    monkeypatch.setattr(manager, "start_rollout", counted_start)
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    entered = threading.Event()
    release = threading.Event()
    response: list[Any] = []
    original_event = eval_router._required_event

    def blocked_event(event_type: str, *args: Any, **kwargs: Any) -> None:
        if event_type == "eval.continue_requested":
            entered.set()
            assert release.wait(2), "test did not release the continue audit"
        original_event(event_type, *args, **kwargs)

    monkeypatch.setattr(eval_router, "_required_event", blocked_event)
    continuing = threading.Thread(target=lambda: response.append(_continue(client)))
    continuing.start()
    assert entered.wait(2), "continue audit did not block"
    manager.estop()
    manager.reset_estop(operator_id=OPERATOR)
    release.set()
    continuing.join(2)

    assert response and response[0].status_code == 409
    state = _wait_state(client, _finished("aborted"))
    assert starts == [1]
    report = _report(state)
    assert (report["requested_episodes"], report["n_episodes"], report["scored"]) == (2, 1, 1)


def test_stop_remains_safety_first_when_its_sqlite_audit_write_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    _start(client, max_seconds=10)
    _wait_state(
        client,
        lambda state: state["phase"] == "rollout" and get_manager().mode.value == "rollout",
    )
    _fail_audit_event(monkeypatch, "eval.aborted")

    response = client.post("/api/eval/stop")

    assert response.status_code == 200
    assert "WARNING" in response.json()["detail"]
    assert "SQLite disk I/O error" in response.json()["detail"]
    assert get_manager().mode.value == "idle"
    state = client.get("/api/eval/state").json()
    assert state["active"] is False and state["phase"] == "aborted"


def test_stop_reaches_a_live_rollout_while_start_rollout_is_still_returning(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    manager = get_manager()
    original_start = manager.start_rollout
    original_stop = manager.stop
    start_blocked = threading.Event()
    release_start = threading.Event()
    manager_stopped = threading.Event()
    response: list[Any] = []

    def blocked_start(*args: Any, **kwargs: Any) -> None:
        original_start(*args, **kwargs)
        start_blocked.set()
        assert release_start.wait(2), "test did not release start_rollout"

    def observed_stop(*args: Any, **kwargs: Any) -> bool:
        result = original_stop(*args, **kwargs)
        manager_stopped.set()
        return result

    monkeypatch.setattr(manager, "start_rollout", blocked_start)
    monkeypatch.setattr(manager, "stop", observed_stop)
    _start(client, max_seconds=10)
    assert start_blocked.wait(2), "rollout did not become live"
    assert manager.mode.value == "rollout"

    stopping = threading.Thread(target=lambda: response.append(client.post("/api/eval/stop")))
    stopping.start()
    assert manager_stopped.wait(2), "eval stop did not reach RobotManager"
    assert manager.mode.value == "idle"
    release_start.set()
    stopping.join(2)

    assert not stopping.is_alive()
    assert response and response[0].status_code == 200


def test_initial_journal_fsync_failure_refuses_before_motion(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    real_fsync = eval_router.os.fsync
    failed = False

    def fail_once(fd: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected journal fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(eval_router.os, "fsync", fail_once)
    response = client.post(
        "/api/eval/start",
        json={"model_path": str(_mock_checkpoint()), "operator_id": OPERATOR},
    )

    assert response.status_code == 503
    assert "journal fsync failure" in response.json()["detail"]
    assert get_manager().mode.value == "idle"
    assert client.get("/api/eval/state").json()["phase"] == "idle"
    assert list(get_settings().outputs_dir.glob("eval_*.json*")) == []
    start_events = {
        event["type"]
        for event in get_store().list_events()
        if event["type"].startswith("eval.start")
    }
    assert start_events == set()


def test_trial_journal_fsync_failure_aborts_before_next_trial_and_keeps_the_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    real_fsync = eval_router.os.fsync
    failed = False

    def fail_once(fd: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected trial journal fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(eval_router.os, "fsync", fail_once)
    response = _score(client, success=True)
    assert response.status_code == 503
    assert "trial journal fsync failure" in response.json()["detail"]
    state = _wait_state(client, _finished("failed"))

    assert state["episode"] == 1
    assert "trial journal fsync failure" in state["error"]
    report = _report(state)
    assert (report["phase"], report["scored"], report["successes"]) == ("failed", 1, 1)
    assert len(report["per_episode"]) == 1
    assert report["per_episode"][0]["episode"] == 1
    assert report["per_episode"][0]["success"] is True
    assert report["per_episode"][0]["duration"] > 0


@pytest.mark.parametrize(
    ("reset_before_release", "expected_reason"),
    [
        (False, "E-stop engaged"),
        (True, "safety/control authority changed during evaluation"),
    ],
)
def test_final_trial_rechecks_estop_after_its_blocked_journal_fsync(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    reset_before_release: bool,
    expected_reason: str,
):
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    journal = eval_router.get_session()._journal
    assert journal is not None
    original = journal.write
    trial_write = threading.Event()
    release_write = threading.Event()
    score_response: list[Any] = []

    def blocked_write(event: str, **payload: Any) -> None:
        if event == "trial":
            trial_write.set()
            assert release_write.wait(2), "test did not release the trial journal"
        original(event, **payload)

    monkeypatch.setattr(journal, "write", blocked_write)
    scoring = threading.Thread(target=lambda: score_response.append(_score(client, success=True)))
    scoring.start()
    assert trial_write.wait(2), "trial journal write did not start"
    assert client.post("/api/robot/estop").status_code == 200
    if reset_before_release:
        get_manager().reset_estop(operator_id=OPERATOR)
    release_write.set()
    scoring.join(2)

    assert not scoring.is_alive()
    assert score_response and score_response[0].status_code == 409
    state = _wait_state(client, _finished("aborted"))
    assert state["error"] == expected_reason
    report = _report(state)
    assert report["phase"] == "aborted"
    assert (report["scored"], report["interrupted"]) == (0, 1)
    assert report["per_episode"][0]["success"] is None
    assert report["per_episode"][0]["interrupted"] == expected_reason
    journal = Path(state["report_path"]).with_suffix(".jsonl")
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    invalidated = [event for event in events if event["event"] == "trial_invalidated"]
    assert len(invalidated) == 1
    assert invalidated[0]["episode"] == 1
    assert invalidated[0]["interrupted"] == expected_reason


# --- data-B14: self-describing, durable reports ------------------------------------


def test_data_b14_rate_uses_scored_trials_and_report_is_self_describing(client: TestClient):
    _start(client, n_episodes=3, max_seconds=0.05)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client, success=True).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    assert _continue(client).status_code == 200
    _wait_state(
        client,
        lambda s: s["episode"] == 2 and get_manager().mode.value == "rollout",
    )
    assert client.post("/api/robot/stop").status_code == 200
    state = _wait_state(client, _finished("aborted"))
    assert state["success_rate"] == 1.0
    assert (state["scored"], state["interrupted"]) == (1, 1)
    report = _report(state)
    assert report["requested_episodes"] == 3
    assert report["n_episodes"] == 2
    assert (report["scored"], report["interrupted"], report["successes"]) == (1, 1, 1)
    assert report["success_rate"] == 1.0  # interrupted trials are not failures
    assert report["n"] == 1
    assert report["ci95"] == pytest.approx([0.2065, 1.0], abs=5e-5)
    assert report["source"] == "mock"
    assert report["report_version"] == 2
    assert report["label"] == (
        "deterministic MockPolicy stand-in (checkpoint weights were not loaded)"
    )
    assert report["phase"] == "aborted"
    assert report["error"]
    assert [entry["episode"] for entry in report["per_episode"]] == [1, 2]
    assert report["per_episode"][1]["success"] is None
    for key in ("model_path", "task", "fps", "session_id", "created_at", "started_at"):
        assert key in report
    # The operator id authorizes scoring; it never lands in files or events.
    assert OPERATOR not in json.dumps(report)
    assert report["fps"] == 30
    assert report["task"] is None


def test_data_b14_completed_report_counts(client: TestClient):
    _start(client, n_episodes=2)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client, success=False).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    assert _continue(client).status_code == 200
    _wait_state(client, _phase("awaiting_score", episode=2))
    assert _score(client, success=True).status_code == 200
    state = _wait_state(client, _finished("done"))
    report = _report(state)
    assert (report["requested_episodes"], report["n_episodes"], report["scored"]) == (2, 2, 2)
    assert (report["successes"], report["success_rate"], report["interrupted"]) == (1, 0.5, 0)
    assert report["n"] == 2
    assert report["ci95"] == pytest.approx([0.0945, 0.9055], abs=5e-5)
    assert report["source"] == "mock"
    assert (report["phase"], report["error"]) == ("done", "")
    assert [entry["success"] for entry in report["per_episode"]] == [False, True]


def test_terminal_phase_is_not_exposed_until_the_report_write_finishes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    entered = threading.Event()
    release = threading.Event()
    original = eval_router._write_json_atomic

    def blocked_write(path: Path, payload: dict[str, Any]) -> None:
        entered.set()
        assert release.wait(2), "test did not release the report write"
        original(path, payload)

    monkeypatch.setattr(eval_router, "_write_json_atomic", blocked_write)
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    assert entered.wait(2), "report write did not start"

    saving = client.get("/api/eval/state").json()
    assert saving["active"] is True
    assert saving["phase"] == "finalizing"
    assert saving["report_path"] == ""

    release.set()
    done = _wait_state(client, _finished("done"))
    assert done["active"] is False


def test_data_b14_stop_saves_partial_report(client: TestClient):
    _start(client, n_episodes=4)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client, success=True).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    assert client.post("/api/eval/stop").status_code == 200
    state = client.get("/api/eval/state").json()
    assert state["phase"] == "aborted" and state["report_path"]
    report = _report(state)
    assert (report["requested_episodes"], report["n_episodes"], report["scored"]) == (4, 1, 1)
    assert report["error"] == "stopped by operator"


def test_data_b14_report_name_is_millisecond_plus_uuid_and_write_is_atomic(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    replaced: list[tuple[Path, Path]] = []
    original_replace = eval_router.os.replace

    def replace(src: Any, dst: Any) -> None:
        source, target = Path(src), Path(dst)
        assert not target.exists(), "the final report must appear only via os.replace"
        replaced.append((source, target))
        original_replace(src, dst)

    monkeypatch.setattr(eval_router.os, "replace", replace)
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    state = _wait_state(client, _finished("done"))
    path = Path(state["report_path"])
    assert re.fullmatch(r"eval_\d{13}_[0-9a-f]{12}\.json", path.name)
    assert [target for _, target in replaced] == [path]
    assert replaced[0][0].parent == path.parent and replaced[0][0] != path
    leftovers = [p.name for p in path.parent.iterdir() if p.suffix not in {".json", ".jsonl"}]
    assert leftovers == []


def test_data_b14_failed_report_write_leaves_no_partial_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    def replace(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(eval_router.os, "replace", replace)
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    state = _wait_state(client, lambda s: s["phase"] == "done" and "disk full" in s["error"])
    assert state["report_path"] == ""
    assert _reports() == []
    outputs = get_settings().outputs_dir
    # The per-trial journal survives; no truncated or temporary report does.
    assert [p.suffix for p in outputs.iterdir()] == [".jsonl"]
    journal = next(outputs.glob("eval_*.jsonl"))
    final = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    assert final["event"] == "finished"
    assert final["report"] is None
    assert "report not saved: disk full" in final["error"]


def test_report_directory_fsync_failure_is_visible_and_quarantined(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    _start(client)
    _wait_state(client, _phase("awaiting_score"))

    def fail_directory_fsync(path: Path) -> None:
        raise OSError("injected directory fsync EIO")

    monkeypatch.setattr(eval_router, "fsync_directory", fail_directory_fsync)
    assert _score(client).status_code == 200
    state = _wait_state(
        client,
        lambda value: value["phase"] == "done" and "directory fsync EIO" in value["error"],
    )

    assert state["report_path"] == ""
    assert list(get_settings().outputs_dir.glob("eval_*.json")) == []
    assert len(list(get_settings().outputs_dir.glob("eval_*.json.invalid-*"))) == 1


def test_terminal_journal_and_failed_report_rewrite_quarantine_the_false_green_report(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    _start(client)
    _wait_state(client, _phase("awaiting_score"))
    session = eval_router.get_session()
    journal = session._journal
    assert journal is not None
    original_journal_write = journal.write
    original_report_write = eval_router._write_json_atomic
    report_writes = 0

    def journal_write(event: str, **payload: Any) -> None:
        if event == "finished":
            raise eval_router.EvalPersistenceError("injected terminal journal failure")
        original_journal_write(event, **payload)

    def report_write(path: Path, payload: dict[str, Any]) -> None:
        nonlocal report_writes
        report_writes += 1
        if report_writes == 2:
            raise OSError("injected failed-report rewrite failure")
        original_report_write(path, payload)

    monkeypatch.setattr(journal, "write", journal_write)
    monkeypatch.setattr(eval_router, "_write_json_atomic", report_write)
    assert _score(client).status_code == 200
    state = _wait_state(
        client,
        lambda value: value["phase"] == "failed" and value["active"] is False,
    )

    assert state["report_path"] == ""
    assert "terminal journal failure" in state["error"]
    assert "failed-report rewrite failure" in state["error"]
    assert list(get_settings().outputs_dir.glob("eval_*.json")) == []
    quarantined = list(get_settings().outputs_dir.glob("eval_*.json.invalid-*"))
    assert len(quarantined) == 1
    stale = json.loads(quarantined[0].read_text(encoding="utf-8"))
    assert stale["phase"] == "done"  # quarantined, so normal report scanners never consume it


# --- data-B13: the policy is loaded and validated before /start returns ------------


def test_data_b13_eval_accepts_only_registered_models(client: TestClient, tmp_path: Path):
    outside = tmp_path / "outside" / "pretrained_model"
    outside.mkdir(parents=True)
    (outside / "config.json").write_text(
        json.dumps({"type": "act", "mock": True}), encoding="utf-8"
    )

    response = client.post(
        "/api/eval/start",
        json={"model_path": str(outside), "operator_id": OPERATOR},
    )

    assert response.status_code == 404
    assert "not a registered model" in response.json()["detail"]
    assert client.get("/api/eval/state").json()["phase"] == "idle"
    assert get_manager().mode.value == "idle"


def test_data_b13_policy_load_failure_fails_the_start_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    path = _mock_checkpoint("broken")

    def broken(model_path: Any, *, task: str | None = None) -> Any:
        raise RuntimeError("checkpoint has no config.json")

    monkeypatch.setattr(eval_router, "load_policy", broken)
    response = client.post(
        "/api/eval/start",
        json={"model_path": str(path), "n_episodes": 3, "operator_id": OPERATOR},
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "could not load policy" in detail and "checkpoint has no config.json" in detail
    state = client.get("/api/eval/state").json()
    assert (state["active"], state["phase"]) == (False, "idle")
    manager = get_manager()
    assert manager.mode.value == "idle"
    assert manager.fault is None
    assert _reports() == []


def test_data_b13_policy_is_loaded_once_at_start(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    path = _mock_checkpoint()
    calls: list[tuple[str, str | None]] = []
    original = eval_router.load_policy

    def counting(model_path: Any, *, task: str | None = None) -> Any:
        calls.append((str(model_path), task))
        return original(model_path, task=task)

    def reloaded_in_rollout(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "the rollout thread reloaded a policy already loaded by /api/eval/start"
        )

    monkeypatch.setattr(eval_router, "load_policy", counting)
    monkeypatch.setattr(robot_manager_module, "load_policy", reloaded_in_rollout)
    _start(client, model_path=str(path), n_episodes=2)
    assert calls == [(str(path), None)]
    _wait_state(client, _phase("awaiting_score"))
    assert _score(client).status_code == 200
    _wait_state(client, _phase("awaiting_reset"))
    assert _continue(client).status_code == 200
    _wait_state(client, _phase("awaiting_score", episode=2))
    assert calls == [(str(path), None)]
    client.post("/api/eval/stop")


def test_data_b13_trial_clock_starts_at_the_first_action(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    path = _mock_checkpoint("load_before_rollout")
    entered = threading.Event()
    release = threading.Event()
    response: list[Any] = []
    original_load = eval_router.load_policy
    original_start = RobotManager.start_rollout
    order: list[str] = []

    def blocked_load(model_path: Any, *, task: str | None = None) -> Any:
        order.append("load")
        entered.set()
        assert release.wait(2), "test did not release the policy load"
        return original_load(model_path, task=task)

    def start_rollout(
        self: RobotManager,
        model_path: str,
        fps: int = 30,
        *,
        operator_id: str,
        policy: Any | None = None,
        task: str | None = None,
    ) -> None:
        order.append("rollout")
        original_start(
            self,
            model_path,
            fps,
            operator_id=operator_id,
            policy=policy,
            task=task,
        )

    monkeypatch.setattr(eval_router, "load_policy", blocked_load)
    monkeypatch.setattr(RobotManager, "start_rollout", start_rollout)

    request = threading.Thread(
        target=lambda: response.append(
            client.post(
                "/api/eval/start",
                json={
                    "model_path": str(path),
                    "max_seconds": 0.15,
                    "operator_id": OPERATOR,
                },
            )
        )
    )
    request.start()
    assert entered.wait(2), "policy load did not start"
    assert order == ["load"]
    assert eval_router.get_session().phase == "idle"
    release.set()
    request.join(2)
    assert not request.is_alive()
    assert response
    assert response[0].status_code == 200, response[0].text
    _wait_state(client, _phase("awaiting_score"))
    assert order == ["load", "rollout"]
    assert get_manager().telemetry()["sample_seq"] > 0, "the policy never acted"
    assert _score(client).status_code == 200
    report = _report(_wait_state(client, _finished("done")))
    assert 0.1 <= report["per_episode"][0]["duration"] < 0.4


def test_data_b13_no_first_action_times_out_on_an_injected_clock(
    monkeypatch: pytest.MonkeyPatch,
):
    class Clock:
        now = 10.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    class Guard:
        estopped = False

    class Manager:
        guard = Guard()
        fault = None

        @staticmethod
        def telemetry() -> dict[str, Any]:
            return {"sample_seq": 7, "mode": "rollout"}

    clock = Clock()
    monkeypatch.setattr(eval_router, "time", clock)
    monkeypatch.setattr(eval_router, "FIRST_ACTION_TIMEOUT_S", 0.2)
    session = eval_router._EvalSession()

    cause, acted_at = session._watch_rollout(Manager(), 7)  # type: ignore[arg-type]

    assert cause == "the policy produced no action within 0.2 s"
    assert acted_at is None
    assert clock.now == pytest.approx(10.2)


# --- data-B6/B18: task and fps --------------------------------------------------


def _capture_start_rollout(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    original = RobotManager.start_rollout

    def start_rollout(
        self: RobotManager,
        model_path: str,
        fps: int = 30,
        *,
        operator_id: str,
        policy: Any | None = None,
        task: str | None = None,
    ) -> None:
        calls.append(
            {
                "model_path": model_path,
                "fps": fps,
                "task": task,
                "policy": policy,
            }
        )
        original(
            self,
            model_path,
            fps,
            operator_id=operator_id,
            policy=policy,
            task=task,
        )

    monkeypatch.setattr(RobotManager, "start_rollout", start_rollout)
    return calls


def test_data_b18_eval_fps_defaults_from_the_checkpoint_and_accepts_an_override(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    path = _mock_checkpoint("fps12", fps=12)
    calls = _capture_start_rollout(monkeypatch)
    _start(client, model_path=str(path))
    state = _wait_state(client, _phase("awaiting_score"))
    assert calls[0]["model_path"] == str(path)
    assert calls[0]["fps"] == 12
    assert calls[0]["policy"] is not None
    assert state["fps"] == 12
    client.post("/api/eval/stop")

    _start(client, model_path=str(path), fps=15)
    _wait_state(client, _phase("awaiting_score"))
    assert calls[-1]["fps"] == 15
    client.post("/api/eval/stop")
    for fps in (0, 241, 1.5):
        response = client.post(
            "/api/eval/start",
            json={"model_path": "mock/model", "fps": fps, "operator_id": OPERATOR},
        )
        assert response.status_code == 422, (fps, response.text)


def test_data_b18_eval_refuses_an_out_of_range_checkpoint_fps_before_loading(
    client: TestClient,
):
    path = _mock_checkpoint("fps_too_high", fps=241)

    response = client.post(
        "/api/eval/start",
        json={"model_path": str(path), "operator_id": OPERATOR},
    )

    assert response.status_code == 422
    assert "fps must be between 1 and 240" in response.json()["detail"]
    assert get_manager().mode.value == "idle"


def test_data_b6_explicit_task_is_loaded_passed_and_reported(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    path = _mock_checkpoint("task_override", task="pick up the blue cube", policy_type="smolvla")
    loaded: list[tuple[str, str | None]] = []
    original_load = eval_router.load_policy

    def load(model_path: Any, *, task: str | None = None) -> Any:
        loaded.append((str(model_path), task))
        return original_load(model_path, task=task)

    monkeypatch.setattr(eval_router, "load_policy", load)
    calls = _capture_start_rollout(monkeypatch)
    _start(client, model_path=str(path), task="  pick up the red cube ")
    state = _wait_state(client, _phase("awaiting_score"))
    assert loaded == [(str(path), "pick up the red cube")]
    assert calls[0]["task"] == "pick up the red cube"
    assert state["task"] == "pick up the red cube"
    assert _score(client).status_code == 200
    report = _report(_wait_state(client, _finished("done")))
    assert report["task"] == "pick up the red cube"


def test_data_b6_blank_task_uses_the_checkpoint_training_task(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    path = _mock_checkpoint("task_default", task="pick up the training cube", policy_type="smolvla")
    calls = _capture_start_rollout(monkeypatch)
    _start(client, model_path=str(path), task="   ")
    state = _wait_state(client, _phase("awaiting_score"))
    assert calls[0]["task"] == "pick up the training cube"
    assert state["task"] == "pick up the training cube"
    client.post("/api/eval/stop")


# --- admission ---------------------------------------------------------------------


def test_eval_start_refuses_while_estopped_and_concurrently(client: TestClient):
    manager = get_manager()
    manager.estop()
    response = client.post(
        "/api/eval/start", json={"model_path": "mock/model", "operator_id": OPERATOR}
    )
    assert response.status_code == 409
    assert "E-stop engaged" in response.json()["detail"]
    manager.reset_estop(operator_id=OPERATOR)
    _start(client, n_episodes=2, max_seconds=5)
    second = client.post(
        "/api/eval/start", json={"model_path": "mock/model", "operator_id": OPERATOR}
    )
    assert second.status_code == 409
    client.post("/api/eval/stop")


@pytest.mark.parametrize(
    "max_seconds", ["NaN", "nan", "Infinity", "-Infinity", "inf", 0, -1, 3600.5]
)
def test_eval_max_seconds_must_be_a_finite_bounded_number(client: TestClient, max_seconds):
    """Gate review (eval_nan_string): the string "NaN" passed both bound checks.

    NaN compares false, so the trial clock never expired, and the journal and
    report then failed to serialise the NaN.
    """
    response = client.post(
        "/api/eval/start",
        json={"model_path": "mock/model", "operator_id": OPERATOR, "max_seconds": max_seconds},
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/eval/state").json()["active"] is False


def test_eval_start_rechecks_max_seconds_for_unvalidated_requests():
    request = StartEvalRequest.model_construct(
        model_path="mock/model",
        n_episodes=1,
        max_seconds=float("nan"),
        operator_id=OPERATOR,
        fps=None,
        task=None,
    )
    with pytest.raises(RuntimeError, match="max_seconds"):
        eval_router.get_session().start(request)
    assert StartEvalRequest(
        model_path="m", operator_id=OPERATOR, max_seconds=3600
    ).max_seconds == pytest.approx(3600)


def test_eval_stop_without_a_session_leaves_other_modes_alone(client: TestClient):
    manager = get_manager()
    manager.start_teleop(30, operator_id=OPERATOR)
    try:
        assert client.post("/api/eval/stop").status_code == 200
        assert manager.mode.value == "teleop"
    finally:
        manager.stop()
