"""SO101_BACKEND=sim never reaches a real Feetech bus.

Settings accepts ``backend=sim`` and leaves ``mock`` False. ``make_bus`` used
to key off ``settings.mock`` only, so under sim it built a ``FeetechBus`` on
the real follower and leader ports, and every arm path (arming, teleop,
rollout, calibration, ``so101 hil run`` labelled REHEARSAL) moved the real arm.

``make_bus`` now builds the simulated bus under sim (stops wave 2). The guard
stays as a permanent invariant for any other route to a FeetechBus: it fails
closed on connect, before LeRobot is imported or a port is opened.
Construction stays non-raising, and without the ``[sim]`` extra the manager
still starts and arming fails closed with the install hint.
"""

from __future__ import annotations

import importlib.abc
import sys
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101.config import get_settings  # noqa: E402
from so101.hardware.follower import SO101Follower  # noqa: E402
from so101.hardware.motors import (  # noqa: E402
    FeetechBus,
    HardwareUnavailableError,
    SimBusUnavailable,
    make_bus,
)
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import get_manager, reset_manager  # noqa: E402


class _LeRobotImported(AssertionError):
    pass


class _NoLeRobot(importlib.abc.MetaPathFinder):
    """Fails any ``lerobot`` import loudly and records it. Not an ImportError:
    the install-hint path catches those."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        if fullname == "lerobot" or fullname.startswith("lerobot."):
            self.attempts.append(fullname)
            raise _LeRobotImported(f"{fullname} was imported before the guard ran")
        return None


def _backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.delenv("SO101_MOCK", raising=False)
    monkeypatch.setenv("SO101_BACKEND", backend)
    monkeypatch.setenv("SO101_SIM_REALTIME", "0")  # lockstep: no physics thread
    get_settings.cache_clear()
    reset_manager()


@pytest.fixture
def sim_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[_NoLeRobot]:
    _backend(monkeypatch, "sim")
    for name in [name for name in sys.modules if name.split(".")[0] == "lerobot"]:
        monkeypatch.delitem(sys.modules, name)
    finder = _NoLeRobot()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    yield finder
    get_settings.cache_clear()
    reset_manager()


def test_a_feetech_bus_refuses_to_connect_under_the_sim_backend(sim_backend: _NoLeRobot):
    bus = FeetechBus("COM7", 1_000_000, "follower")  # construction stays non-raising

    with pytest.raises(HardwareUnavailableError) as caught:
        bus.connect()

    message = str(caught.value)
    assert "SO101_BACKEND=sim" in message
    assert "real Feetech bus on COM7 is refused (it would move the real arm)" in message
    assert "Use SO101_BACKEND=mock, or `so101 sim` for physics." in message
    assert bus.connected is False
    assert sim_backend.attempts == []


def test_a_feetech_bus_in_the_follower_or_leader_refuses_too(sim_backend: _NoLeRobot):
    from so101.hardware.leader import SO101Leader

    follower = SO101Follower()
    leader = SO101Leader()
    # However a FeetechBus got there, it never opens a port under sim.
    follower.bus = FeetechBus("COM7", 1_000_000, "follower")
    leader.bus = FeetechBus("COM8", 1_000_000, "leader")

    with pytest.raises(HardwareUnavailableError, match="SO101_BACKEND=sim"):
        follower.connect(enable_torque=False)
    with pytest.raises(HardwareUnavailableError, match="SO101_BACKEND=sim"):
        leader.connect()
    assert follower.connected is False
    assert leader.connected is False
    assert sim_backend.attempts == []


def test_make_bus_builds_the_simulated_arm_under_sim(sim_backend: _NoLeRobot):
    pytest.importorskip("mujoco")
    from so101.simulation.devices import MujocoBus, SimExpertLeader

    assert isinstance(make_bus("COM7", name="follower"), MujocoBus)
    assert isinstance(make_bus("COM8", name="leader"), SimExpertLeader)
    assert sim_backend.attempts == []


def test_without_the_sim_extra_the_bus_fails_closed_with_the_install_hint(
    sim_backend: _NoLeRobot, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "so101.simulation.devices", None)  # import fails

    bus = make_bus("COM7", name="follower")

    assert isinstance(bus, SimBusUnavailable), "construction stays non-raising"
    assert bus.connected is False and bus.torque_state == "off"
    with pytest.raises(HardwareUnavailableError, match=r"\[sim\] extra"):
        bus.connect()
    with pytest.raises(HardwareUnavailableError):
        bus.read_positions()
    with pytest.raises(HardwareUnavailableError):
        bus.write_positions({"gripper": 1.0})
    with pytest.raises(HardwareUnavailableError):
        bus.set_torque(True)
    bus.disconnect()  # never raises
    assert sim_backend.attempts == []


def test_arming_under_the_sim_backend_drives_the_simulated_arm(sim_backend: _NoLeRobot):
    pytest.importorskip("mujoco")
    with TestClient(create_app()) as client:
        manager = get_manager()
        manager.operator_heartbeat("test-operator")
        assert client.get("/api/robot/status").status_code == 200

        lease = client.post("/api/control/lease", json={"source": "ui", "ttl_seconds": 30})
        assert lease.status_code == 201, lease.text
        response = client.post("/api/control/arm", json={"lease_id": lease.json()["id"]})

        assert response.status_code == 200, response.text
        assert sim_backend.attempts == [], "LeRobot never loaded"
        follower = manager.follower()
        assert not isinstance(follower.bus, FeetechBus)
        assert client.get("/api/robot/status").json()["follower_connected"] is True
        assert follower.torque_state == "on", "the simulated arm reports its torque truth"
        assert client.post("/api/robot/estop").status_code == 200
        assert follower.torque_state == "off"


def test_calibration_is_refused_under_the_sim_backend(sim_backend: _NoLeRobot):
    with TestClient(create_app()) as client:
        response = client.post("/api/calibration/start", json={"arm": "follower"})
    assert response.status_code == 409, response.text
    assert "sim backend" in response.json()["detail"]


@pytest.mark.parametrize("backend", ["real", "mock"])
def test_the_guard_leaves_real_and_mock_connects_alone(
    monkeypatch: pytest.MonkeyPatch, backend: str, tmp_path
):
    from so101.bringup import ownership
    from so101.config import get_settings

    _backend(monkeypatch, backend)
    follower_port = tmp_path / "follower"
    leader_port = tmp_path / "leader"
    follower_port.touch()
    leader_port.touch()
    monkeypatch.setenv("SO101_FOLLOWER_PORT", str(follower_port))
    monkeypatch.setenv("SO101_LEADER_PORT", str(leader_port))
    get_settings.cache_clear()
    bus = FeetechBus(str(follower_port), 1_000_000, "follower")
    device = type("Device", (), {"bus": object(), "is_connected": True})()
    monkeypatch.setattr(bus, "_build", lambda: ("modern", device))
    monkeypatch.setattr(bus, "_connect_modern_safely", lambda implementation: True)

    with ownership.DeviceBusLock(get_settings(), purpose="fake-driver backend guard"):
        bus.connect()

    assert bus.connected is True
