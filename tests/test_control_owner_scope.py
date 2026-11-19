"""C5 owner-scoped disarm and safety-B6 owner-scoped error cleanup.

Contracts: docs/audit/v0.1/safety.md B6, docs/audit/v0.1/frontend.md F3 and
the C5 text in the control-hw brief. Converts safety repro 05 parts (b) and (c).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import Mode, RobotManager, StopKind, get_manager  # noqa: E402
from so101.services.store import get_store  # noqa: E402


@pytest.fixture
def client():
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat("test-operator")
        yield test_client


def _owner_armed(client: TestClient, source: str = "owner") -> str:
    lease = client.post("/api/control/lease", json={"source": source, "ttl_seconds": 5})
    assert lease.status_code == 201, lease.text
    lease_id = lease.json()["id"]
    armed = client.post("/api/control/arm", json={"lease_id": lease_id})
    assert armed.status_code == 200, armed.text
    return lease_id


def _pause_background_watchdog(manager: RobotManager) -> None:
    manager._watchdog_shutdown.set()
    thread = manager._watchdog_thread
    if thread is not None:
        thread.join(timeout=2)
        assert not thread.is_alive()


def _still_armed(client: TestClient) -> None:
    state = client.get("/api/control/state").json()
    assert state["armed"] is True
    assert state["safety_state"] == "armed"
    assert client.get("/api/robot/status").json()["follower_connected"] is True
    assert get_manager().follower().bus.torque_enabled is True


def test_get_status_and_control_state_do_not_expire_a_lease_or_touch_the_bus(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = get_manager()
    lease_id = _owner_armed(client)
    _pause_background_watchdog(manager)
    follower = manager.follower()
    bus_calls: list[str] = []
    read_positions = follower.read_positions
    hold = follower.hold

    def record_read() -> dict[str, float]:
        bus_calls.append("read")
        return read_positions()

    def record_hold(goal: dict[str, float]) -> dict[str, float]:
        bus_calls.append("write")
        return hold(goal)

    monkeypatch.setattr(follower, "read_positions", record_read)
    monkeypatch.setattr(follower, "hold", record_hold)
    with manager._lock:
        assert manager._lease is not None
        manager._lease.deadline = time.monotonic() - 1

    status = client.get("/api/robot/status")
    state = client.get("/api/control/state")

    assert status.status_code == 200 and state.status_code == 200
    assert bus_calls == []
    assert state.json()["armed"] is True and state.json()["lease"] is not None
    assert manager._lease is not None and manager._lease.id == lease_id

    manager._watchdog_tick()
    assert bus_calls == ["read", "write"]
    assert manager._lease is None
    assert manager.control_state()["safety_state"] == "lease_expired"


def _first_command(lease_id: str, source: str) -> dict:
    return {
        "lease_id": lease_id,
        "source": source,
        "sequence": 1,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "ttl_ms": 500,
        "deadman": True,
        "targets": {"shoulder_pan": 0.0},
    }


@pytest.mark.parametrize(
    "body",
    [None, {}, {"lease_id": ""}, {"lease_id": "not-the-lease"}],
    ids=["no-body", "empty", "blank", "wrong"],
)
def test_c5_disarm_without_the_owners_lease_is_409_and_changes_nothing(
    client: TestClient, body: dict | None
):
    _owner_armed(client)

    if body is None:
        response = client.post("/api/control/disarm")
    else:
        response = client.post("/api/control/disarm", json=body)

    assert response.status_code == 409, response.text
    _still_armed(client)


def test_c5_disarm_with_an_expired_lease_is_409(client: TestClient):
    manager = get_manager()
    lease_id = _owner_armed(client)
    _pause_background_watchdog(manager)
    with manager._lock:
        assert manager._lease is not None
        manager._lease.deadline = time.monotonic() - 0.01

    response = client.post("/api/control/disarm", json={"lease_id": lease_id})

    assert response.status_code == 409
    # The lapsed lease was handled as an expiry HALT, not as the disarm.
    assert client.get("/api/control/state").json()["safety_state"] == "lease_expired"


def test_c5_owner_disarm_keeps_a_watchdog_trip_visible(client: TestClient):
    manager = get_manager()
    lease_id = _owner_armed(client)
    _pause_background_watchdog(manager)
    with manager._lock:
        manager._last_command_monotonic = time.monotonic() - 10
    manager._watchdog_tick()
    assert client.get("/api/control/state").json()["safety_state"] == "watchdog_timeout"

    response = client.post("/api/control/disarm", json={"lease_id": lease_id})

    assert response.status_code == 200, response.text
    assert response.json()["safety_state"] == "watchdog_timeout"


def test_global_stop_and_lease_release_are_the_deliberate_clears(client: TestClient):
    manager = get_manager()
    lease_id = _owner_armed(client)
    _pause_background_watchdog(manager)
    with manager._lock:
        manager._last_command_monotonic = time.monotonic() - 10
    manager._watchdog_tick()

    assert client.delete(f"/api/control/lease/{lease_id}").status_code == 200
    assert client.get("/api/control/state").json()["safety_state"] == "disarmed"

    with manager._lock:
        manager._safety_state = "operator_timeout"
    assert client.post("/api/robot/stop").status_code == 200
    assert client.get("/api/control/state").json()["safety_state"] == "disarmed"


def test_c5_a_release_that_loses_the_race_with_expiry_is_409_and_reports_only_the_expiry(
    manager_clock, client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    manager = get_manager()
    lease_id = _owner_armed(client)
    _pause_background_watchdog(manager)
    dispatch = manager._dispatch_stop
    raced: list[bool] = []

    def lease_lapses_in_between(kind, **kwargs):
        if not raced:  # between the release's lease check and its stop
            raced.append(True)
            manager_clock.advance(10)
            manager._expire_lease_if_needed()
        return dispatch(kind, **kwargs)

    monkeypatch.setattr(manager, "_dispatch_stop", lease_lapses_in_between)
    response = client.delete(f"/api/control/lease/{lease_id}")

    assert response.status_code == 409, response.text
    assert "lease" in response.json()["detail"]
    state = client.get("/api/control/state").json()
    assert state["armed"] is False and state["lease"] is None
    assert state["safety_state"] == "lease_expired"
    kinds = [event["type"] for event in client.get("/api/events").json()["events"]]
    assert "control.lease_expired" in kinds
    assert "control.lease_released" not in kinds


def test_c5_an_owner_halt_waits_for_an_expiry_already_in_flight(
    manager_clock, client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    # The agent executor's owner-scoped Halt can arrive while the watchdog is
    # still halting the lapsed lease; it must return only once the arm holds.
    manager = get_manager()
    lease_id = _owner_armed(client)
    _pause_background_watchdog(manager)
    manager_clock.advance(10)
    entered, release = threading.Event(), threading.Event()
    dispatch = manager._dispatch_stop

    def slow_expiry(kind, **kwargs):
        if kwargs.get("reason") == "lease expired":
            entered.set()
            assert release.wait(5)
        return dispatch(kind, **kwargs)

    monkeypatch.setattr(manager, "_dispatch_stop", slow_expiry)
    watchdog = threading.Thread(target=manager._expire_lease_if_needed)
    watchdog.start()
    assert entered.wait(5)
    # Any other caller (status, the next watchdog pass) skips the expiry in
    # flight instead of queueing behind it.
    manager._expire_lease_if_needed()
    assert manager._lease is not None and manager._lease.id == lease_id
    halted = threading.Event()

    def owner_halt():
        manager.request_stop(StopKind.HALT, reason="agent skill failed", lease_id=lease_id)
        halted.set()

    owner = threading.Thread(target=owner_halt)
    owner.start()
    assert not halted.wait(0.2)
    release.set()
    watchdog.join(5)
    owner.join(5)
    assert halted.is_set()
    assert manager.status()["stop_state"] == "halted"


def test_safety_b6_failed_arm_command_with_a_foreign_lease_keeps_the_owner_armed(
    client: TestClient,
):
    _owner_armed(client)

    response = client.post(
        "/api/control/arm-command", json=_first_command("not-my-lease", "other-tab")
    )

    assert response.status_code == 409
    assert "lease" in response.json()["detail"]
    _still_armed(client)


def test_safety_b6_failed_tracked_first_frame_keeps_the_owner_armed(client: TestClient):
    _owner_armed(client)

    response = client.post(
        "/api/tracking/arm-execute",
        json={
            "source": "browser-mediapipe",
            "confidence": 0.95,
            "intent": {"normalized_x": 0.5},
            "execute": True,
            "lease_id": "not-my-lease",
            "sequence": 0,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "ttl_ms": 300,
            "deadman": True,
        },
    )

    assert response.status_code == 409
    _still_armed(client)


def test_safety_b6_arm_command_during_teleop_is_409_not_500(client: TestClient):
    started = client.post("/api/teleop/start", json={"fps": 30, "operator_id": "test-operator"})
    assert started.status_code == 200, started.text

    response = client.post("/api/control/arm-command", json=_first_command("x", "ui"))

    assert response.status_code == 409, response.text
    assert client.get("/api/robot/status").json()["mode"] == "teleop"
    assert client.post("/api/teleop/stop").status_code == 200


def test_safety_b6_owner_cleanup_never_masks_the_original_error(client: TestClient, monkeypatch):
    manager = get_manager()
    lease = client.post("/api/control/lease", json={"source": "owner", "ttl_seconds": 5}).json()
    original_send = manager.send_external_command

    def failing_send(**kwargs):
        # Arming succeeded; now the first command is rejected, and the bus
        # stops answering, so the cleanup Halt cannot hold and its torque-off
        # fallback cannot confirm either.
        manager.follower().bus.fail_io = True
        return original_send(**{**kwargs, "deadman": False})

    monkeypatch.setattr(manager, "send_external_command", failing_send)

    response = client.post("/api/control/arm-command", json=_first_command(lease["id"], "owner"))

    # v0.1 raised the strict cleanup error inside the except block: HTTP 500.
    assert response.status_code == 409, response.text
    assert "deadman" in response.json()["detail"]
    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "fault"


def test_c5_a_fault_raised_while_estopped_keeps_the_estop_visible(client: TestClient):
    assert client.post("/api/robot/estop").status_code == 200

    get_manager()._set_fault("late loop fault", source="test")

    assert client.get("/api/control/state").json()["safety_state"] == "estopped"
    # The persisted snapshot (safety-B8) agrees: the fault did not overwrite it.
    saved = get_store().load_latches()
    assert saved is not None
    assert saved["safety_state"] == "estopped" and saved["fault"] == "late loop fault"
    # Once the E-stop is reset, the fault that is still latched shows.
    reset = client.post(
        "/api/robot/estop/reset", json={"operator_id": "test-operator", "confirm": True}
    )
    assert reset.status_code == 200
    assert client.get("/api/control/state").json()["safety_state"] == "fault"


def test_c5_an_operator_timeout_never_overwrites_a_fault(client: TestClient):
    manager = get_manager()
    _pause_background_watchdog(manager)
    manager._set_fault("loop fault", source="test")
    with manager._lock:
        # A loop that has not seen the fault yet, and whose operator is gone.
        manager.mode = Mode.TELEOP
        manager._stop.clear()
        manager._operation_owner = None

    manager._watchdog_tick()

    assert manager.mode is Mode.IDLE
    assert client.get("/api/control/state").json()["safety_state"] == "fault"


def test_c5_an_arm_revoked_mid_flight_leaves_the_trip_visible(client: TestClient, monkeypatch):
    manager = get_manager()
    lease_id = _owner_armed(client)
    _pause_background_watchdog(manager)
    with manager._lock:
        manager._last_command_monotonic = time.monotonic() - 10
    manager._watchdog_tick()
    assert client.get("/api/control/state").json()["safety_state"] == "watchdog_timeout"
    follower = manager.follower()
    read_positions = follower.read_positions

    def read_then_lose_authority():
        measured = read_positions()
        with manager._lock:
            manager._authority_epoch += 1  # another stop landed during the read
        return measured

    monkeypatch.setattr(follower, "read_positions", read_then_lose_authority)

    response = client.post("/api/control/arm", json={"lease_id": lease_id})

    assert response.status_code == 409, response.text
    state = client.get("/api/control/state").json()
    assert state["armed"] is False
    assert state["safety_state"] == "watchdog_timeout"
