"""Idle-bus lending: ``RobotManager.run_idle_bus_diagnostic`` and preflight through it.

The manager stays the only owner of the motor buses (docs/control-plane-safety.md
"Invariants"). A diagnostic runs only while the robot is idle, under the
manager's bus lock, on a read-only view of the bus; an E-stop ends it at its
next read. ``POST /api/preflight``, ``POST /api/preflight/bus`` and
``so101 preflight`` with a server running all go through it
(docs/bring-up.md "Who may open the bus").
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from so101 import JOINT_NAMES  # noqa: E402
from so101.bringup import preflight as pf  # noqa: E402
from so101.bringup.buses import (  # noqa: E402
    BusDiagnosticOpenError,
    FakeFeetechBus,
    collect_bus_facts,
)
from so101.bringup.lerobot_surface import LeRobotSurface  # noqa: E402
from so101.bringup.ownership import ServerProbe  # noqa: E402
from so101.config import get_settings  # noqa: E402
from so101.services import robot_manager as rm  # noqa: E402
from so101.services.app import create_app  # noqa: E402
from so101.services.robot_manager import (  # noqa: E402
    BusDiagnosticRefused,
    ControlBusy,
    ControlConflict,
    RobotManager,
    get_manager,
)

OPERATOR = "test-operator"
STATUS_URL = "http://127.0.0.1:8816/api/robot/status"


class CountingBus(FakeFeetechBus):
    """The rehearsal bus, recording which thread read it and whether the lock was held."""

    def __init__(self, manager: RobotManager, arm: str = "follower") -> None:
        super().__init__(arm=arm)
        self.manager = manager
        self.reads = 0
        self.unlocked = 0

    def _require_connected(self) -> None:
        super()._require_connected()
        self.reads += 1
        if not self.manager._io_lock.locked():
            self.unlocked += 1


def _opener(manager: RobotManager, opened: list[CountingBus]) -> Any:
    def open_port(port: str) -> CountingBus:
        opened.append(CountingBus(manager, "leader" if "leader" in port else "follower"))
        return opened[-1]

    return open_port


def _facts(bus: Any) -> Any:
    return collect_bus_facts(bus, arm="follower", port="p", source="mock", sleep=lambda _: None)


def _armed(manager: RobotManager) -> str:
    lease = manager.acquire_lease("diagnostic-test", 600)
    manager.arm_control(lease["id"])
    return str(lease["id"])


def _wait_for(predicate: Any, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def manager(manager_clock: Any, supervise: Any) -> Iterator[RobotManager]:
    manager = get_manager()
    manager.operator_heartbeat(OPERATOR)
    yield manager


# --- the accessor ------------------------------------------------------------------------


def test_an_idle_bus_is_lent_read_only_under_the_bus_lock_and_closed(manager):
    opened: list[CountingBus] = []
    facts = manager.run_idle_bus_diagnostic("follower", _facts, opener=_opener(manager, opened))
    (bus,) = opened
    assert facts.opened and facts.responding == list(JOINT_NAMES)
    assert bus.reads > 0 and bus.unlocked == 0
    assert bus.calls[-1] == ("disconnect", False)  # closed with no torque write
    assert not manager._io_lock.locked()


def test_the_leader_port_comes_from_the_settings(manager):
    ports: list[str] = []

    def open_port(port: str) -> FakeFeetechBus:
        ports.append(port)
        return FakeFeetechBus(arm="leader")

    manager.run_idle_bus_diagnostic("leader", _facts, opener=open_port)
    assert ports == [get_settings().leader_port]


def test_a_lent_bus_cannot_write(manager):
    attempts: list[str] = []

    def write(bus: Any) -> None:
        for name in ("sync_write", "write", "disable_torque", "enable_torque", "disconnect"):
            try:
                getattr(bus, name)
            except AttributeError as exc:
                assert "read-only" in str(exc)
                attempts.append(name)

    manager.run_idle_bus_diagnostic("follower", write, opener=lambda port: FakeFeetechBus())
    assert attempts == ["sync_write", "write", "disable_torque", "enable_torque", "disconnect"]


@pytest.mark.parametrize(
    ("setup", "fragment"),
    [
        (lambda m: _armed(m), "armed"),
        (lambda m: m.estop(), "E-stop is latched"),
        (lambda m: setattr(m, "cal_in_progress", True), "calibration"),
        (lambda m: setattr(m, "_safety_state", "arming"), "armed"),
        (lambda m: setattr(m, "mode", rm.Mode.TELEOP), r"busy \(teleop\)"),
    ],
)
def test_a_busy_robot_refuses(manager, setup, fragment):
    setup(manager)
    with pytest.raises(BusDiagnosticRefused, match=fragment) as raised:
        manager.run_idle_bus_diagnostic("follower", _facts, opener=lambda p: FakeFeetechBus())
    assert isinstance(raised.value, ControlConflict)  # HTTP 409


def test_a_halt_and_a_collapse_refuse(manager, manager_clock, monkeypatch):
    lease = _armed(manager)
    manager.release_lease(lease)  # a Halt: the arm holds, motors on
    assert manager.status()["stop_state"] == "halted"
    with pytest.raises(BusDiagnosticRefused, match="Halt holds the arm"):
        manager.run_idle_bus_diagnostic("leader", _facts, opener=lambda p: FakeFeetechBus())

    parked, go = threading.Event(), threading.Event()

    def park(seconds: float) -> None:
        manager_clock.now += seconds
        parked.set()
        go.wait(10)

    monkeypatch.setattr(manager, "_sleep", park)
    try:
        assert manager.request_stop("collapse", reason="test").collapsing
        assert parked.wait(5)
        with pytest.raises(BusDiagnosticRefused, match="Collapse is lowering"):
            manager.run_idle_bus_diagnostic("leader", _facts, opener=lambda p: FakeFeetechBus())
    finally:
        go.set()
    assert _wait_for(lambda: manager.status()["stop_state"] == "none")
    manager.run_idle_bus_diagnostic("leader", _facts, opener=lambda p: FakeFeetechBus())


def test_an_energized_target_refuses(manager, monkeypatch):
    monkeypatch.setattr(manager.follower().bus, "_torque_state", "on")
    with pytest.raises(BusDiagnosticRefused, match="follower is energized"):
        manager.run_idle_bus_diagnostic("follower", _facts, opener=lambda p: FakeFeetechBus())


def test_backends_without_a_serial_bus_refuse(manager, monkeypatch):
    with pytest.raises(BusDiagnosticRefused, match="mock backend has no serial bus"):
        manager.run_idle_bus_diagnostic("follower", _facts)
    monkeypatch.setattr(rm, "current_backend", lambda settings: "sim")
    with pytest.raises(BusDiagnosticRefused, match="sim"):
        manager.run_idle_bus_diagnostic("follower", _facts, opener=lambda p: FakeFeetechBus())


def test_a_bad_target_or_timeout_is_a_value_error(manager):
    with pytest.raises(ValueError, match="target"):
        manager.run_idle_bus_diagnostic("gripper", _facts)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout_s"):
        manager.run_idle_bus_diagnostic("follower", _facts, timeout_s=float("nan"))


def test_a_busy_bus_answers_busy_and_reads_nothing(manager):
    opened: list[CountingBus] = []
    assert manager._io_lock.acquire(timeout=1)
    try:
        with pytest.raises(ControlBusy, match="waited 10 ms"):
            manager.run_idle_bus_diagnostic(
                "follower", _facts, timeout_s=0.01, opener=_opener(manager, opened)
            )
    finally:
        manager._io_lock.release()
    assert opened == []


def test_state_is_checked_again_after_the_wait_for_the_bus(manager):
    errors: list[BaseException] = []
    manager._io_lock.acquire()

    def diagnose() -> None:
        try:
            manager.run_idle_bus_diagnostic(
                "follower", _facts, timeout_s=5.0, opener=lambda p: FakeFeetechBus()
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=diagnose, daemon=True)
    thread.start()
    time.sleep(0.05)  # the diagnostic now waits for the bus
    manager.cal_in_progress = True  # calibration claimed the robot meanwhile
    manager._io_lock.release()
    thread.join(5)
    (error,) = errors
    assert isinstance(error, BusDiagnosticRefused) and "calibration" in str(error)


def test_an_e_stop_ends_the_diagnostic_at_its_next_read(manager):
    opened: list[CountingBus] = []
    reads_at_estop: list[int] = []

    def diagnose(bus: Any) -> Any:
        bus.ping("shoulder_pan")
        manager.latch_estop("test: E-stop during a diagnostic")
        reads_at_estop.append(opened[0].reads)
        return _facts(bus)  # every read now fails at once, without bus I/O

    with pytest.raises(BusDiagnosticRefused, match="E-stop preempted"):
        manager.run_idle_bus_diagnostic("follower", diagnose, opener=_opener(manager, opened))
    assert opened[0].reads == reads_at_estop[0]
    assert opened[0].calls[-1] == ("disconnect", False)


def test_a_port_that_does_not_open_is_reported(manager):
    def broken(port: str) -> Any:
        raise ConnectionError("could not open the port")

    with pytest.raises(BusDiagnosticOpenError, match="ConnectionError: could not open"):
        manager.run_idle_bus_diagnostic("follower", _facts, opener=broken)
    assert not manager._io_lock.locked()


def test_an_open_bus_is_lent_as_it_is_and_stays_open(manager, monkeypatch):
    raw = FakeFeetechBus()

    class OpenBus:
        connected = True
        torque_state = "off"
        register_bus = raw

        def disconnect(self) -> None:
            self.connected = False

    follower = manager.follower()
    monkeypatch.setattr(follower, "bus", OpenBus())
    facts = manager.run_idle_bus_diagnostic(
        "follower", _facts, opener=lambda port: pytest.fail("must not open the port twice")
    )
    assert facts.opened
    assert ("disconnect", False) not in raw.calls and follower.bus.connected


def test_an_open_bus_without_register_reads_refuses(manager, monkeypatch):
    follower = manager.follower()
    monkeypatch.setattr(follower.bus, "_connected", True)  # the mock arm's port is open
    with pytest.raises(BusDiagnosticRefused, match="no register reads"):
        manager.run_idle_bus_diagnostic("follower", _facts, opener=lambda p: FakeFeetechBus())


def test_the_real_backend_opens_a_bare_read_only_bus(manager, monkeypatch):
    monkeypatch.setattr(rm, "current_backend", lambda settings: "real")
    ports: list[str] = []

    def bare(port: str) -> FakeFeetechBus:
        ports.append(port)
        return FakeFeetechBus()

    monkeypatch.setattr(rm, "_open_diagnostic_bus", bare)
    manager.run_idle_bus_diagnostic("follower", _facts)
    assert ports == [manager.follower().port]


def test_the_default_opener_uses_the_bring_up_kit(monkeypatch):
    from so101.bringup import buses

    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        buses, "open_feetech_bus", lambda port, *, baudrate: seen.append((port, baudrate))
    )
    rm._open_diagnostic_bus("/dev/so101-follower")
    assert seen == [("/dev/so101-follower", 1_000_000)]


# --- preflight through the manager ----------------------------------------------------------


def _surface() -> LeRobotSurface:
    return LeRobotSurface(
        installed=True, version="0.6.1", importable=True, sdk_importable=True, tables_checked=True
    )


def test_api_preflight_on_the_mock_backend_borrows_through_the_manager(manager):
    report = pf.run_api_preflight(manager, lerobot_prober=_surface, sleep=lambda _: None)
    assert report.bus_access == "synthetic" and "RobotManager lent" in report.bus_access_detail
    by_key = {check.key: check.status for check in report.checks}
    assert by_key["motors.follower"] == "pass" and by_key["motors.leader"] == "pass"


def test_api_preflight_while_armed_is_refused_not_synthetic(manager):
    _armed(manager)
    report = pf.run_api_preflight(manager, lerobot_prober=_surface, sleep=lambda _: None)
    assert report.bus_access == "refused" and "armed" in report.bus_access_detail
    motors = {check.key: check for check in report.checks}["motors.follower"]
    assert motors.status == "skip" and motors.blocking
    assert report.overall == "incomplete"


def test_the_bus_route_returns_the_facts_or_the_refusal(manager):
    with TestClient(create_app()) as client:
        body = client.post("/api/preflight/bus", params={"arm": "leader"}).json()
        assert body["kind"] == "so101.preflight_bus" and body["backend"] == "mock"
        assert body["bus_access"] == "synthetic" and body["arms"] == ["leader"]
        assert set(body["buses"]) == {"leader"} and body["buses"]["leader"]["opened"] is True
        _armed(manager)
        refused = client.post("/api/preflight/bus").json()
    assert refused["bus_access"] == "refused" and refused["buses"] == {}
    assert "armed" in refused["bus_access_detail"] and refused["remediation"]


def test_the_api_preflight_route_runs_the_bus_checks(manager):
    with TestClient(create_app()) as client:
        body = client.post("/api/preflight", params={"save": "false"}).json()
    assert body["bus_access"] == "synthetic"
    motors = [check for check in body["checks"] if check["id"] == "motors"]
    assert motors and all(check["status"] == "pass" for check in motors)


# --- so101 preflight with a server running ------------------------------------------------------


def _probe() -> ServerProbe:
    return ServerProbe(running=True, url=STATUS_URL, detail="HTTP 200", status={})


def _through(client: TestClient) -> httpx.MockTransport:
    """Forward the CLI's request to the in-process server."""

    def handler(request: httpx.Request) -> httpx.Response:
        # The CLI's method and headers go through the server's guard as sent.
        headers = {key: value for key, value in request.headers.items() if key != "host"}
        answer = client.request(
            request.method, request.url.path, params=dict(request.url.params), headers=headers
        )
        return httpx.Response(answer.status_code, content=answer.content)

    return httpx.MockTransport(handler)


def test_the_cli_grades_the_bus_facts_a_running_server_lends(manager):
    settings = get_settings()

    def probe(_: Any) -> ServerProbe:
        return ServerProbe(
            running=True,
            url=STATUS_URL,
            detail="HTTP 200",
            status={"mode": "idle", "estopped": False, "live_cameras": ["0"]},
        )

    with TestClient(create_app()) as client:
        transport = _through(client)
        report = pf.run_cli_preflight(
            settings,
            server_probe=probe,
            lerobot_prober=_surface,
            sleep=lambda _: None,
            write_report=False,
            server_bus_facts=lambda p, arms, backend: pf.fetch_server_bus_facts(
                p, arms, backend, transport=transport
            ),
        )
    assert report.bus_access == "synthetic" and "the server at" in report.bus_access_detail
    by_key = {check.key: check.status for check in report.checks}
    assert by_key["motors.follower"] == "pass" and by_key["firmware.leader"] == "pass"


def test_fetching_from_a_server_on_another_backend_is_refused(manager):
    with TestClient(create_app()) as client:
        access, buses = pf.fetch_server_bus_facts(
            _probe(), ("follower",), "real", transport=_through(client)
        )
    assert access.mode == "refused" and buses == {}
    assert "mock backend" in access.detail and "SO101_MOCK" in access.remediation


@pytest.mark.parametrize(
    ("handler", "fragment"),
    [
        (lambda request: httpx.Response(404, json={"detail": "Not Found"}), "another build"),
        (lambda request: httpx.Response(405, json={"detail": "GET only"}), "another build"),
        (lambda request: httpx.Response(200, content=b"<html>"), "not a bus-facts reply"),
        (lambda request: httpx.Response(500, json={"detail": "boom"}), "not a bus-facts reply"),
    ],
)
def test_fetch_failures_skip_the_bus_checks_as_refused(handler, fragment):
    access, buses = pf.fetch_server_bus_facts(
        _probe(), pf.ARMS, "mock", transport=httpx.MockTransport(handler)
    )
    assert access.mode == "refused" and fragment in access.detail and buses == {}
    assert "so101 preflight" in access.remediation


def test_an_unreachable_server_is_refused():
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    access, buses = pf.fetch_server_bus_facts(
        _probe(), pf.ARMS, "mock", transport=httpx.MockTransport(unreachable)
    )
    assert access.mode == "refused" and "ConnectError" in access.detail and buses == {}


def test_a_server_refusal_passes_through(manager):
    _armed(manager)
    with TestClient(create_app()) as client:
        access, buses = pf.fetch_server_bus_facts(
            _probe(), pf.ARMS, "mock", transport=_through(client)
        )
    assert access.mode == "refused" and "armed" in access.detail and buses == {}
