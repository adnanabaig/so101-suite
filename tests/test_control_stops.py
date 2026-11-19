"""C11: E-stop is category 0 on every bus; one stop dispatcher for all three kinds.

Contract: docs/plans/2026-10-01-suite-v02-design.md §5 "Stop categories". The
Halt and Collapse behaviour itself is in test_control_halt.py and
test_control_collapse.py.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    ControlRejected,
    RobotManager,
    StopKind,
    StopResult,
    get_manager,
)


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat("test-operator")
        yield test_client


def _armed_manager() -> tuple[RobotManager, str]:
    manager = get_manager()
    lease = manager.acquire_lease("stop-test", 5)
    manager.arm_control(lease["id"])
    assert manager.follower().bus.torque_enabled is True
    return manager, lease["id"]


def _spy_on_request_stop(manager: RobotManager, monkeypatch) -> list[tuple[StopKind, dict]]:
    calls: list[tuple[StopKind, dict]] = []
    original = manager.request_stop

    def spy(kind, **kwargs):
        calls.append((StopKind(kind), kwargs))
        return original(kind, **kwargs)

    monkeypatch.setattr(manager, "request_stop", spy)
    return calls


def test_status_advertises_all_three_stop_kinds_and_their_fields(client: TestClient):
    body = client.get("/api/robot/status").json()
    assert body["stop_kinds"] == ["estop", "halt", "collapse"]
    assert body["stop_state"] == "none"
    assert body["torque_state"] == "off"
    assert body["halt"] is None and body["collapse"] is None


def test_request_stop_with_a_foreign_lease_is_a_no_op(client: TestClient):
    manager, lease_id = _armed_manager()

    result = manager.request_stop(StopKind.HALT, reason="stale client", lease_id="not-the-lease")

    assert isinstance(result, StopResult)
    assert result.applied is False
    state = manager.control_state()
    assert state["armed"] is True
    assert state["safety_state"] == "armed"
    assert manager.follower().bus.torque_enabled is True


def test_owner_scoped_halt_disarms_and_holds_but_keeps_the_lease(client: TestClient):
    manager, lease_id = _armed_manager()

    result = manager.request_stop(StopKind.HALT, reason="agent skill done", lease_id=lease_id)

    assert result.applied and result.ok and result.holding
    state = manager.control_state()
    assert state["armed"] is False
    assert state["lease"] is not None
    # Category 2: the motors stay on and hold (design doc §5).
    assert manager.follower().connected is True
    assert manager.follower().bus.torque_enabled is True
    assert manager.status()["stop_state"] == "halted"


def test_request_stop_accepts_kind_strings(client: TestClient):
    manager, lease_id = _armed_manager()
    result = manager.request_stop("collapse", reason="string kind", lease_id=lease_id)
    assert result.kind is StopKind.COLLAPSE
    assert result.applied and result.ok and result.collapsing


def test_estop_ignores_lease_scope(client: TestClient):
    manager, _lease_id = _armed_manager()

    result = manager.request_stop(StopKind.ESTOP, reason="test", lease_id="not-the-lease")

    assert result.applied and result.ok
    assert manager.guard.estopped is True
    assert manager.control_state()["lease"] is None
    assert manager.follower().bus.torque_enabled is False


def test_strict_request_stop_raises_when_torque_off_is_unconfirmed(client: TestClient):
    manager, lease_id = _armed_manager()
    # The bus stops answering: the halt cannot hold, and its category 0
    # fallback cannot confirm torque-off either.
    manager.follower().bus.fail_io = True

    with pytest.raises(ControlRejected, match="torque-off is unconfirmed"):
        manager.request_stop(StopKind.HALT, reason="strict", lease_id=lease_id, strict=True)
    # The non-strict default never raises; error paths can always call it.
    result = manager.request_stop(StopKind.HALT, reason="lenient")
    assert result.applied and not result.torque_off and not result.ok


def test_strict_request_stop_raises_when_a_confirmed_fallback_latched_a_fault(
    client: TestClient, monkeypatch
):
    manager, lease_id = _armed_manager()

    def dead_read():
        raise ConnectionError("no status packet")

    monkeypatch.setattr(manager.follower(), "read_positions", dead_read)

    with pytest.raises(ControlRejected, match="fell back to torque-off"):
        manager.request_stop(StopKind.HALT, reason="strict", lease_id=lease_id, strict=True)
    assert manager.fault is not None
    assert manager.follower().bus.torque_enabled is False


def test_soft_stop_is_a_global_halt(client: TestClient, monkeypatch):
    manager, _lease_id = _armed_manager()
    calls = _spy_on_request_stop(manager, monkeypatch)

    assert manager.soft_stop() is True

    assert [kind for kind, _ in calls] == [StopKind.HALT]
    assert calls[0][1].get("lease_id") is None
    assert manager.control_state()["armed"] is False


@pytest.mark.parametrize(
    "path",
    ["/api/robot/stop", "/api/teleop/stop"],
)
def test_legacy_stop_endpoints_route_through_halt(client: TestClient, monkeypatch, path: str):
    manager = get_manager()
    calls = _spy_on_request_stop(manager, monkeypatch)

    response = client.post(path)

    assert response.status_code == 200, response.text
    assert [kind for kind, _ in calls] == [StopKind.HALT]


def test_owner_disarm_endpoint_is_a_collapse(client: TestClient, monkeypatch):
    manager, lease_id = _armed_manager()
    calls = _spy_on_request_stop(manager, monkeypatch)

    response = client.post("/api/control/disarm", json={"lease_id": lease_id})

    assert response.status_code == 200, response.text
    assert [kind for kind, _ in calls] == [StopKind.COLLAPSE]
    assert calls[0][1]["lease_id"] == lease_id
    assert response.json()["armed"] is False


def _pause_background_watchdog(manager: RobotManager) -> None:
    """Drive supervision by hand so the test owns every tick."""

    manager._watchdog_shutdown.set()
    thread = manager._watchdog_thread
    if thread is not None:
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_watchdog_and_lease_expiry_are_halts(client: TestClient, monkeypatch):
    manager, _lease_id = _armed_manager()
    _pause_background_watchdog(manager)
    kinds: list[StopKind] = []
    original = manager._dispatch_stop

    def spy(kind, **kwargs):
        kinds.append(kind)
        return original(kind, **kwargs)

    monkeypatch.setattr(manager, "_dispatch_stop", spy)
    with manager._lock:
        manager._last_command_monotonic = time.monotonic() - 10
    manager._watchdog_tick()
    assert kinds == [StopKind.HALT]
    assert manager.control_state()["safety_state"] == "watchdog_timeout"

    with manager._lock:
        assert manager._lease is not None
        manager._lease.deadline = time.monotonic() - 1
    manager._expire_lease_if_needed()
    assert kinds == [StopKind.HALT, StopKind.HALT]
    assert manager.control_state()["safety_state"] == "lease_expired"
