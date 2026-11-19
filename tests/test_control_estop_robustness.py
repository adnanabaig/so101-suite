"""safety-B9: one wedged driver call can never starve the E-stop.

Contract: docs/audit/v0.1/safety.md B9 and the control-hw brief.

- ``POST /api/robot/estop`` is an ``async def``. It sets the latch inline on
  the event loop and runs the torque-off teardown on a dedicated executor, so
  it never needs a request worker thread.
- Request handlers validate the lease, source, armed state and sequence before
  they wait for the motor bus, and wait at most ``request_io_timeout_s``
  (250 ms). On expiry the request is 503 and nothing was sent.

Converts safety repro 12 (``12_estop_threadpool_starvation.py``).
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import anyio.to_thread
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services import robot_manager as robot_manager_module  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    ControlBusy,
    ControlConflict,
    ControlRejected,
    RobotManager,
    get_manager,
)
from so101.services.routers import robot as robot_router  # noqa: E402

SOURCE = "owner"


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat("test-operator")
        yield test_client


def _pause_background_watchdog(manager: RobotManager) -> None:
    manager._watchdog_shutdown.set()
    thread = manager._watchdog_thread
    if thread is not None:
        thread.join(timeout=2)
        assert not thread.is_alive()


def _lease(client: TestClient) -> str:
    response = client.post("/api/control/lease", json={"source": SOURCE, "ttl_seconds": 30})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _armed(client: TestClient) -> str:
    lease_id = _lease(client)
    armed = client.post("/api/control/arm", json={"lease_id": lease_id})
    assert armed.status_code == 200, armed.text
    return lease_id


def _command(lease_id: str, sequence: int, source: str = SOURCE) -> dict[str, Any]:
    return {
        "lease_id": lease_id,
        "source": source,
        "sequence": sequence,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "ttl_ms": 500,
        "deadman": True,
        "targets": {"shoulder_pan": 1.0},
    }


@contextmanager
def _bus_held(manager: RobotManager) -> Iterator[None]:
    """Hold the motor-bus lock, as a wedged driver call would."""

    assert manager._io_lock.acquire(timeout=2), "the bus lock was not free"
    try:
        yield
    finally:
        manager._io_lock.release()


def _within(seconds: float, call: Callable[[], Any]) -> Any:
    """Run ``call`` on a helper thread; fail, instead of hanging, if it blocks."""

    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = call()
        except BaseException as exc:  # re-raised on the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    if worker.is_alive():
        pytest.fail(f"the call blocked for more than {seconds} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _set_request_workers(client: TestClient, tokens: int) -> None:
    """Shrink AnyIO's request threadpool (40 by default) for this app's loop."""

    async def apply() -> None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = tokens

    assert client.portal is not None
    client.portal.call(apply)


@contextmanager
def _wedged_send_goal(manager: RobotManager, monkeypatch) -> Iterator[threading.Event]:
    """Make the follower's next goal write block, holding the bus lock."""

    follower = manager.follower()
    original = follower.send_goal
    entered = threading.Event()
    release = threading.Event()

    def wedged(goal: dict[str, float]) -> dict[str, float]:
        entered.set()
        release.wait(timeout=10)
        return original(goal)

    monkeypatch.setattr(follower, "send_goal", wedged)
    try:
        yield entered
    finally:
        release.set()


def test_safety_b9_the_estop_endpoint_is_async():
    # A sync handler needs a request worker thread, which a pile of blocked
    # requests can exhaust; an async handler runs on the event loop.
    assert inspect.iscoroutinefunction(robot_router.estop)


def test_safety_b9_repro_12_estop_latches_and_answers_behind_a_wedged_bus(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    _pause_background_watchdog(manager)
    lease_id = _armed(client)
    _set_request_workers(client, 4)
    pool = ThreadPoolExecutor(max_workers=16)
    try:
        with _wedged_send_goal(manager, monkeypatch) as entered:
            wedged = pool.submit(client.post, "/api/control/command", json=_command(lease_id, 1))
            assert entered.wait(timeout=3), "the first command never reached the bus"
            # v0.1: each of these held a worker forever at `with self._io_lock`,
            # before any lease check, until no worker was left for the E-stop.
            junk = [
                pool.submit(client.post, "/api/control/command", json=_command("bogus", 100 + i))
                for i in range(8)
            ]
            for future in junk:
                response = future.result(timeout=3)
                assert response.status_code == 409, response.text
                assert "lease" in response.json()["detail"]

            response = _within(4.0, lambda: client.post("/api/robot/estop"))

            assert manager.guard.estopped is True
            # The wedged call still holds the bus, so torque-off is unconfirmed.
            assert response.status_code == 503, response.text
            assert "power" in response.json()["detail"]
        assert wedged.result(timeout=5).status_code == 409
    finally:
        pool.shutdown(wait=True)


def test_safety_b9_estop_needs_no_request_worker(client: TestClient, monkeypatch):
    manager = get_manager()
    _pause_background_watchdog(manager)
    lease_id = _armed(client)
    # One worker, and the wedged command holds it.
    _set_request_workers(client, 1)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        with _wedged_send_goal(manager, monkeypatch) as entered:
            pool.submit(client.post, "/api/control/command", json=_command(lease_id, 1))
            assert entered.wait(timeout=3)

            response = _within(4.0, lambda: client.post("/api/robot/estop"))

            assert manager.guard.estopped is True
            assert response.status_code == 503, response.text
    finally:
        pool.shutdown(wait=True)


def test_safety_b9_estop_answers_fast_while_every_request_worker_is_blocked(
    client: TestClient, monkeypatch
):
    """A review reproduced this with 45 sync /api/simulation/render calls stuck
    behind a GL probe: they held every AnyIO worker and E-stop took 8.99 s. Here
    blocked status reads fill the shrunken pool, with more queued behind them,
    and nothing holds the bus, so the E-stop must confirm torque-off at once."""

    manager = get_manager()
    _set_request_workers(client, 4)
    entered = threading.Semaphore(0)
    release = threading.Event()
    original_status = manager.status

    def blocked_status() -> dict[str, Any]:
        entered.release()
        release.wait(timeout=10)
        return original_status()

    monkeypatch.setattr(manager, "status", blocked_status)
    pool = ThreadPoolExecutor(max_workers=6)
    try:
        blocked = [pool.submit(client.get, "/api/robot/status") for _ in range(6)]
        for _ in range(4):
            assert entered.acquire(timeout=3), "the request workers never filled up"
        assert not entered.acquire(timeout=0.2), "a request found a free worker"

        started = time.monotonic()
        response = _within(4.0, lambda: client.post("/api/robot/estop"))
        elapsed = time.monotonic() - started

        assert response.status_code == 200, response.text
        assert manager.guard.estopped is True
        assert elapsed < 1.0, f"E-stop took {elapsed:.2f} s behind blocked requests"
    finally:
        release.set()
        pool.shutdown(wait=True)
    assert [future.result(timeout=5).status_code for future in blocked] == [200] * 6


def test_safety_b9_estop_teardown_runs_on_its_own_executor(client: TestClient, monkeypatch):
    manager = get_manager()
    threads: list[str] = []
    original = manager.estop

    def recording_estop(*args: Any, **kwargs: Any) -> None:
        threads.append(threading.current_thread().name)
        original(*args, **kwargs)

    monkeypatch.setattr(manager, "estop", recording_estop)

    response = client.post("/api/robot/estop")

    assert response.status_code == 200, response.text
    assert len(threads) == 1 and threads[0].startswith("so101-estop"), threads


def test_estop_is_durable_before_warning_log_can_block(client: TestClient, monkeypatch):
    manager = get_manager()
    entered = threading.Event()
    release = threading.Event()
    original_warning = robot_manager_module.log.warning

    def blocked_warning(message: str, *args: Any, **kwargs: Any) -> None:
        if "E-STOP engaged" in message:
            entered.set()
            assert release.wait(timeout=5), "the test never released the E-stop warning"
        original_warning(message, *args, **kwargs)

    monkeypatch.setattr(robot_manager_module.log, "warning", blocked_warning)
    outcome: dict[str, BaseException] = {}

    def estop() -> None:
        try:
            manager.estop()
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=estop, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=3), "the E-stop never reached its warning log"
        saved = manager.store.load_latches()
        assert saved is not None and saved["estopped"] is True
        assert manager.guard.estopped is True
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert outcome == {}


def test_safety_b9_estop_answers_within_its_bound_when_teardown_hangs(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    release = threading.Event()
    monkeypatch.setattr(robot_router, "ESTOP_RESPONSE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(manager, "estop", lambda *a, **k: release.wait(timeout=10))
    try:
        response = _within(3.0, lambda: client.post("/api/robot/estop"))

        # The latch is set inline, before the teardown is even scheduled.
        assert manager.guard.estopped is True
        assert response.status_code == 503, response.text
        assert "latched" in response.json()["detail"]
    finally:
        release.set()


def test_safety_b9_latch_estop_revokes_authority_without_bus_io(client: TestClient):
    manager = get_manager()
    _pause_background_watchdog(manager)
    _armed(client)
    epoch = manager._authority_epoch

    with _bus_held(manager):
        assert _within(1.0, lambda: manager.latch_estop("test")) is True

    assert manager.guard.estopped is True
    state = manager.control_state()
    assert state["armed"] is False and state["safety_state"] == "estopped"
    assert state["lease"] is None
    assert manager._authority_epoch > epoch


def test_new_estop_wins_over_reset_in_flight(client: TestClient, monkeypatch):
    manager = get_manager()
    manager.latch_estop("first E-stop")
    entered = threading.Event()
    proceed = threading.Event()
    original_reset = manager.guard.reset

    def paused_reset(reference: dict[str, float] | None = None) -> None:
        entered.set()
        assert proceed.wait(timeout=5), "the test never released the E-stop reset"
        original_reset(reference)

    monkeypatch.setattr(manager.guard, "reset", paused_reset)
    outcome: dict[str, BaseException] = {}

    def reset() -> None:
        try:
            manager.reset_estop(operator_id="test-operator")
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=reset, daemon=True)
    worker.start()
    assert entered.wait(timeout=3), "the reset never reached the guard"
    assert manager.latch_estop("newer E-stop", lock_timeout=0) is False
    proceed.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), ControlConflict)
    assert manager.guard.estopped is True


def test_estop_already_waiting_for_state_wins_over_new_reset(client: TestClient, monkeypatch):
    manager = get_manager()
    manager.latch_estop("first E-stop")
    latch_published = threading.Event()
    reset_snapshotted = threading.Event()
    original_estop = manager.guard.estop
    original_snapshot = manager._estop_reset_snapshot
    estop_calls = 0

    def announced_estop() -> None:
        nonlocal estop_calls
        original_estop()
        estop_calls += 1
        if estop_calls == 2:  # latch_estop publishes, reasserts, then waits for state
            latch_published.set()

    def announced_snapshot() -> tuple[int, bool]:
        snapshot = original_snapshot()
        reset_snapshotted.set()
        return snapshot

    monkeypatch.setattr(manager.guard, "estop", announced_estop)
    monkeypatch.setattr(manager, "_estop_reset_snapshot", announced_snapshot)
    reset_outcome: dict[str, BaseException] = {}

    def reset() -> None:
        try:
            manager.reset_estop(operator_id="test-operator")
        except BaseException as exc:
            reset_outcome["error"] = exc

    latch_outcome: dict[str, bool] = {}

    def latch() -> None:
        latch_outcome["bookkeeping"] = manager.latch_estop(
            "E-stop already in flight", lock_timeout=5
        )

    assert manager._lock.acquire(timeout=2)
    try:
        latch_worker = threading.Thread(target=latch, daemon=True)
        latch_worker.start()
        assert latch_published.wait(timeout=3), "the E-stop never queued on the state lock"
        reset_worker = threading.Thread(target=reset, daemon=True)
        reset_worker.start()
        assert reset_snapshotted.wait(timeout=3), "the reset never queued on the state lock"
    finally:
        manager._lock.release()
    latch_worker.join(timeout=5)
    reset_worker.join(timeout=5)

    assert not latch_worker.is_alive()
    assert not reset_worker.is_alive()
    assert latch_outcome["bookkeeping"] is True
    assert isinstance(reset_outcome.get("error"), ControlConflict)
    assert manager.guard.estopped is True


@pytest.mark.parametrize(
    ("make_request", "detail"),
    [
        (lambda lease_id: _command("bogus", 1), "lease"),
        (lambda lease_id: _command(lease_id, 1, source="someone-else"), "source"),
        (lambda lease_id: {**_command(lease_id, 1), "deadman": False}, "deadman"),
    ],
    ids=["bogus-lease", "wrong-source", "deadman-released"],
)
def test_safety_b9_invalid_commands_are_rejected_without_waiting_for_the_bus(
    client: TestClient, make_request, detail: str
):
    manager = get_manager()
    _pause_background_watchdog(manager)
    lease_id = _armed(client)

    with _bus_held(manager):
        response = _within(
            2.0, lambda: client.post("/api/control/command", json=make_request(lease_id))
        )

    assert response.status_code == 409, response.text
    assert detail in response.json()["detail"]


def test_safety_b9_disarmed_and_stale_sequence_are_rejected_before_the_bus():
    manager = get_manager()
    manager.operator_heartbeat("test-operator")
    _pause_background_watchdog(manager)
    lease = manager.acquire_lease(SOURCE, 30)

    def send(sequence: int) -> Any:
        return manager.send_external_command(
            lease_id=lease["id"],
            source=SOURCE,
            sequence=sequence,
            sent_at=datetime.now(timezone.utc),
            ttl_ms=500,
            deadman=True,
            targets={"shoulder_pan": 1.0},
        )

    with _bus_held(manager), pytest.raises(ControlRejected, match="disarmed") as disarmed:
        _within(2.0, lambda: send(1))
    assert not isinstance(disarmed.value, ControlBusy)

    manager.arm_control(lease["id"])
    send(5)
    with _bus_held(manager), pytest.raises(ControlRejected, match="out-of-order") as stale:
        _within(2.0, lambda: send(5))
    assert not isinstance(stale.value, ControlBusy)


def test_safety_b9_a_valid_command_is_503_when_the_bus_does_not_yield(client: TestClient):
    manager = get_manager()
    _pause_background_watchdog(manager)
    lease_id = _armed(client)
    assert RobotManager.request_io_timeout_s == 0.25

    with _bus_held(manager):
        started = time.monotonic()
        response = _within(
            2.0, lambda: client.post("/api/control/command", json=_command(lease_id, 1))
        )
        waited = time.monotonic() - started

    assert response.status_code == 503, response.text
    assert "busy" in response.json()["detail"]
    assert waited >= 0.2
    # Nothing was sent: the session is intact and the sequence was not used.
    state = client.get("/api/control/state").json()
    assert state["armed"] is True and state["safety_state"] == "armed"
    assert manager._fault is None
    retry = client.post("/api/control/command", json=_command(lease_id, 1))
    assert retry.status_code == 200, retry.text


def test_safety_b9_observe_is_503_when_the_bus_does_not_yield(client: TestClient):
    manager = get_manager()
    lease_id = _lease(client)

    with _bus_held(manager):
        response = _within(
            2.0, lambda: client.post("/api/control/observe", json={"lease_id": lease_id})
        )

    assert response.status_code == 503, response.text
    assert manager._fault is None
    assert client.post("/api/control/observe", json={"lease_id": lease_id}).status_code == 200


def test_safety_b9_arm_is_503_when_the_bus_does_not_yield_and_stays_disarmed(client: TestClient):
    manager = get_manager()
    lease_id = _lease(client)

    with _bus_held(manager):
        response = _within(
            2.0, lambda: client.post("/api/control/arm", json={"lease_id": lease_id})
        )

    assert response.status_code == 503, response.text
    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "disarmed", "a busy bus must not leave the state at arming"
    assert manager._fault is None
    assert client.post("/api/control/arm", json={"lease_id": lease_id}).status_code == 200


def test_safety_b9_tracked_arm_execute_is_503_when_the_bus_does_not_yield(client: TestClient):
    manager = get_manager()
    lease_id = _lease(client)
    body = {
        "source": SOURCE,
        "confidence": 0.95,
        "intent": {"normalized_x": 0.5},
        "execute": True,
        "lease_id": lease_id,
        "sequence": 1,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "ttl_ms": 300,
        "deadman": True,
    }

    with _bus_held(manager):
        response = _within(2.0, lambda: client.post("/api/tracking/arm-execute", json=body))

    assert response.status_code == 503, response.text
    assert client.get("/api/control/state").json()["armed"] is False
    assert manager.fault is not None and "motor bus did not come free" in manager.fault


def test_safety_b9_heartbeat_renews_the_lease_while_the_bus_is_busy(client: TestClient):
    manager = get_manager()
    _pause_background_watchdog(manager)
    lease_id = _armed(client)
    seq = manager.telemetry()["sample_seq"]

    with _bus_held(manager):
        response = _within(2.0, lambda: client.post(f"/api/control/lease/{lease_id}/heartbeat"))

    # The renewal is what the heartbeat is for; the observation refresh is
    # skipped rather than failing the lease or latching a fault.
    assert response.status_code == 200, response.text
    assert manager._fault is None
    assert manager.telemetry()["sample_seq"] == seq
    assert client.get("/api/control/state").json()["armed"] is True
