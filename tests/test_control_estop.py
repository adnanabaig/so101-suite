"""C11 item 1: E-stop is category 0 on every bus the manager owns.

Contract: docs/plans/2026-10-01-suite-v02-design.md §5 "Stop categories".
E-stop cuts torque on the follower, the leader and mock/sim actuators before
it joins threads or waits for calibration, and it never claims a torque-off it
could not confirm.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.hardware.motors import MockBus, TorqueOffUnconfirmed, torque_off_all  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import RobotManager, get_manager  # noqa: E402


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        get_manager().operator_heartbeat("test-operator")
        yield test_client


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _armed_manager() -> RobotManager:
    manager = get_manager()
    lease = manager.acquire_lease("estop-test", 5)
    manager.arm_control(lease["id"])
    assert manager.follower().bus.torque_enabled is True
    return manager


def test_mock_bus_keeps_torque_truth_when_torque_off_fails():
    bus = MockBus("follower")
    bus.connect()
    bus.set_torque(True)
    bus.fail_torque_off = True

    assert torque_off_all(bus) is False
    with pytest.raises(TorqueOffUnconfirmed):
        bus.disconnect()
    # The port always closes (no leaked handle), but a failed torque-off must
    # not pretend torque dropped: the servos hold and the state says so.
    assert bus.connected is False
    assert bus.torque_enabled is True
    assert bus.torque_state == "unconfirmed"

    bus.fail_torque_off = False
    bus.connect()
    assert torque_off_all(bus) is True
    assert bus.torque_enabled is False
    assert bus.torque_state == "off"


def test_estop_cuts_torque_on_follower_and_leader_before_joining_the_loop(
    client: TestClient, monkeypatch
):
    manager = get_manager()
    manager.start_teleop(fps=60, operator_id="test-operator")
    follower_bus = manager.follower().bus
    leader_bus = manager.leader().bus
    assert _wait_for(lambda: follower_bus.torque_enabled and leader_bus.connected)
    # Make the leader energized too so the test proves E-stop reaches it.
    leader_bus.set_torque(True)

    torque_when_blocking: list[tuple[bool, bool]] = []
    original_stop_loop = manager._stop_loop
    original_cancel = manager.cancel_calibration

    def recording_stop_loop(*args, **kwargs):
        torque_when_blocking.append((follower_bus.torque_enabled, leader_bus.torque_enabled))
        return original_stop_loop(*args, **kwargs)

    def recording_cancel(*args, **kwargs):
        torque_when_blocking.append((follower_bus.torque_enabled, leader_bus.torque_enabled))
        return original_cancel(*args, **kwargs)

    monkeypatch.setattr(manager, "_stop_loop", recording_stop_loop)
    monkeypatch.setattr(manager, "cancel_calibration", recording_cancel)

    response = client.post("/api/robot/estop")

    assert response.status_code == 200, response.text
    assert len(torque_when_blocking) == 2, "E-stop must join the loop and cancel calibration"
    assert all(state == (False, False) for state in torque_when_blocking)
    assert follower_bus.torque_enabled is False
    assert leader_bus.torque_enabled is False
    status = client.get("/api/robot/status").json()
    assert status["estopped"] is True
    assert status["mode"] == "idle"


def test_estop_still_reaches_the_leader_when_follower_torque_off_fails(client: TestClient):
    manager = _armed_manager()
    leader = manager.leader()
    leader.connect()
    leader.bus.set_torque(True)
    manager.follower().bus.fail_torque_off = True

    response = client.post("/api/robot/estop")

    assert response.status_code == 503
    assert "cut physical power" in response.json()["detail"]
    assert leader.bus.torque_enabled is False
    # The follower really is still energized; nothing may claim otherwise.
    assert manager.follower().bus.torque_enabled is True
    status = client.get("/api/robot/status").json()
    assert status["estopped"] is True
    assert status["torque_unconfirmed"] is True
    # The port closed anyway; a closed port is not torque off.
    assert status["follower_connected"] is False


def test_estop_uses_one_broadcast_torque_off_per_bus(client: TestClient, monkeypatch):
    manager = _armed_manager()
    calls: list[str] = []
    bus = manager.follower().bus
    original = bus.torque_off_all

    def counting() -> bool:
        calls.append(bus.name)
        return original()

    monkeypatch.setattr(bus, "torque_off_all", counting)
    assert client.post("/api/robot/estop").status_code == 200
    assert calls == ["follower"]
    assert client.get("/api/robot/status").json()["follower_connected"] is False


def test_loop_exits_cleanly_on_estop_without_a_secondary_fault(client: TestClient):
    manager = get_manager()
    manager.start_teleop(fps=60, operator_id="test-operator")
    assert _wait_for(lambda: manager.follower().bus.torque_enabled)

    assert client.post("/api/robot/estop").status_code == 200

    assert manager.fault is None
    assert manager.mode.value == "idle"
    events = client.get("/api/events").json()["events"]
    assert not any(event["type"] == "fault" for event in events)
