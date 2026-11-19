"""Control-plane timing on an injected clock (devex-B4, docs/audit/v0.1/devex.md).

The manager reads one clock for leases, operator heartbeats, the command
deadman and the watchdog. These tests freeze it and advance it instead of
sleeping, so they pass on a loaded CI runner.

Arming starts the deadman only once torque is on, so a slow bus write cannot
eat the operator's 500 ms first-command window.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import Mode, get_manager  # noqa: E402

OPERATOR = "test-operator"


@pytest.fixture
def client(manager_clock):
    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        get_manager().operator_heartbeat(OPERATOR)
        yield test_client


def _wait_for(predicate, timeout_s: float = 1.0) -> bool:
    """Poll against a deadline. Shorter than any timer under test, so a
    real-time expiry cannot pass for an injected-clock one."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _state(client: TestClient) -> dict:
    return client.get("/api/control/state").json()


def _lease(client: TestClient, ttl_seconds: float = 30.0) -> str:
    response = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": ttl_seconds})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_devex_b4_leases_expire_on_the_manager_clock(client: TestClient, manager_clock):
    lease_id = _lease(client, ttl_seconds=30.0)

    manager_clock.advance(29.0)
    assert client.post(f"/api/control/lease/{lease_id}/heartbeat").status_code == 200
    manager_clock.advance(29.0)
    assert _state(client)["lease"] is not None, "the heartbeat renewed the lease"

    manager_clock.advance(2.0)

    assert _wait_for(lambda: _state(client)["lease"] is None)
    assert _state(client)["safety_state"] == "lease_expired"
    assert client.post(f"/api/control/lease/{lease_id}/heartbeat").status_code == 409


def test_devex_b4_the_deadman_starts_once_torque_is_on(
    client: TestClient, manager_clock, supervise, monkeypatch
):
    manager = get_manager()
    follower = manager.follower()
    enable_torque = follower.enable_torque

    def slow_enable_torque():
        manager_clock.advance(0.6)  # a bus write slower than the 500 ms window
        return enable_torque()

    monkeypatch.setattr(follower, "enable_torque", slow_enable_torque)
    lease_id = _lease(client)

    armed = client.post("/api/control/arm", json={"lease_id": lease_id})

    assert armed.status_code == 200, armed.text
    supervise()
    assert _state(client)["armed"] is True, "the enable time was charged to the operator"
    manager_clock.advance(0.45)
    supervise()
    assert _state(client)["armed"] is True

    manager_clock.advance(0.1)  # 550 ms after torque came on, with no command
    supervise()

    state = _state(client)
    assert state["armed"] is False
    assert state["safety_state"] == "watchdog_timeout"
    # The watchdog trip is a Halt: disarmed, holding with the motors on.
    assert follower.connected is True
    assert manager.status()["stop_state"] == "halted"


def test_devex_b4_the_operator_timeout_runs_on_the_manager_clock(
    client: TestClient, manager_clock, supervise, monkeypatch
):
    manager = get_manager()
    # Frozen time even while the loop runs: its pacing sleep only naps.
    monkeypatch.setattr(manager, "_sleep", lambda seconds: time.sleep(0.001))
    started = client.post("/api/teleop/start", json={"fps": 30, "operator_id": OPERATOR})
    assert started.status_code == 200, started.text

    supervise()
    assert manager.mode is Mode.TELEOP, "the heartbeat is fresh on the injected clock"

    manager_clock.advance(manager.operator_timeout_ms / 1000 + 0.1)
    supervise()

    assert _wait_for(lambda: manager.mode is Mode.IDLE, timeout_s=5.0)
    events = client.get("/api/events").json()["events"]
    assert any(event["type"] == "runtime.operator_timeout" for event in events)
    # The halted arm collapses at shutdown; its descent paces on the injected
    # clock, so let that clock move again (a frozen one never reaches rest).
    monkeypatch.setattr(manager, "_sleep", manager_clock.sleep)
